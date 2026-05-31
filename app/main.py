from datetime import datetime
import threading
import time
from pathlib import Path

import secrets
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.v1 import router as api_router
from app.core.config import settings
from app.core.logger import logger
from app.ml.inference import get_model_status
from app.services.s3_service import verify_bucket
from app.workers.celery_app import celery
from app.schemas import HealthResponse, ReadyResponse

app = FastAPI(
    title=settings.app_name,
    version="1.0.0",
    description="Asynchronous audio inference API for VibeAI with GradCAM explainability.",
)

# Mount local artifacts directory to serve GradCAM images
app.mount(
    "/artifacts",
    StaticFiles(directory=str(settings.artifact_dir)),
    name="artifacts"
)

try:
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator().instrument(app).expose(app)
except ImportError:
    logger.warning(
        "Prometheus instrumentation is not available; skipping metrics instrumentation")


def start_cleanup_scheduler():
    def cleanup_loop():
        while True:
            try:
                logger.info("[Cleanup] Initiating scheduled directory retention sweep...")
                now = time.time()
                
                # 1. Clean temp_uploads older than 24 hours (86400 seconds)
                temp_dir = Path(settings.temp_upload_dir)
                if temp_dir.exists():
                    for item in temp_dir.iterdir():
                        if item.is_file() and item.name != ".gitignore":
                            mtime = item.stat().st_mtime
                            if now - mtime > 86400:
                                try:
                                    item.unlink()
                                    logger.info(f"[Cleanup] Auto-deleted transient upload file: {item.name}")
                                except Exception as e:
                                    logger.error(f"[Cleanup] Failed to delete transient file {item.name}: {e}")
                                    
                # 2. Clean artifacts/gradcam older than gradcam_retention_days (default 7 days)
                gradcam_dir = Path(settings.gradcam_dir)
                retention_seconds = settings.gradcam_retention_days * 86400
                if gradcam_dir.exists():
                    for item in gradcam_dir.iterdir():
                        if item.is_file() and item.name != ".gitignore":
                            mtime = item.stat().st_mtime
                            if now - mtime > retention_seconds:
                                try:
                                    item.unlink()
                                    logger.info(f"[Cleanup] Auto-deleted expired GradCAM artifact: {item.name}")
                                except Exception as e:
                                    logger.error(f"[Cleanup] Failed to delete expired artifact {item.name}: {e}")
            except Exception as exc:
                logger.error(f"[Cleanup] Error in cleanup scheduler loop: {exc}")
                
            # Sleep for 1 hour (3600 seconds)
            time.sleep(3600)

    thread = threading.Thread(target=cleanup_loop, daemon=True, name="vibeai-cleanup-scheduler")
    thread.start()
    logger.info("[Cleanup] Background retention cleanup scheduler started successfully.")


@app.middleware("http")
async def verify_internal_trust_token(request: Request, call_next):
    # Exempt non-API roots and metrics dashboard path
    if not request.url.path.startswith("/api/v1"):
        return await call_next(request)
        
    secret = settings.vibeai_internal_secret
    if secret:
        provided = request.headers.get("X-VibeAI-Internal-Secret")
        if not provided or not secrets.compare_digest(provided, secret):
            logger.warning(
                f"[SecurityBoundary] Blocked unauthorized internal access attempt on: {request.url.path}"
            )
            return JSONResponse(
                status_code=401,
                content={"detail": "Untrusted service communication"}
            )
            
    return await call_next(request)


@app.on_event("startup")
async def startup_event():
    settings.temp_upload_dir.mkdir(parents=True, exist_ok=True)
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    settings.gradcam_dir.mkdir(parents=True, exist_ok=True)
    settings.database_path.parent.mkdir(parents=True, exist_ok=True)
    app.state.started_at = datetime.utcnow().isoformat(timespec="seconds")
    logger.info(f"Upload directory ready: {settings.temp_upload_dir}")
    logger.info(f"Artifact directory ready: {settings.artifact_dir}")
    
    # Start the local storage cleanup loop
    start_cleanup_scheduler()


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.exception(f"Unhandled exception on {request.url.path}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error",
                 "error": "An unexpected error occurred"},
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request, exc):
    logger.warning(f"Validation error on {request.url.path}: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "error": "Validation error"},
    )


@app.get("/", tags=["Health"], summary="API root")
def root():
    return {
        "service": settings.app_name,
        "status": "running",
        "version": "1.0.0",
        "started_at": getattr(app.state, "started_at", None),
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"], summary="Service health")
def health():
    return HealthResponse(status="healthy", ready=False)


@app.get("/ready", response_model=ReadyResponse, tags=["Health"], summary="Readiness check")
def ready():
    redis_ok = False
    s3_ok = False
    model_ok = False

    try:
        redis_ok = bool(celery.control.inspect().ping())
    except Exception as exc:
        logger.warning(f"Redis readiness failed: {exc}")

    try:
        bucket_status = verify_bucket()
        s3_ok = bucket_status.get("status") == "ok"
    except Exception as exc:
        logger.warning(f"S3 readiness failed: {exc}")

    try:
        model_status = get_model_status()
        model_ok = all(value == "loaded" for value in model_status.values())
    except Exception as exc:
        logger.warning(f"Model readiness failed: {exc}")

    ready_status = redis_ok and s3_ok and model_ok
    return ReadyResponse(
        status="ready" if ready_status else "not_ready",
        redis=redis_ok,
        s3=s3_ok,
        model_loaded=model_ok,
        started_at=getattr(app.state, "started_at", None),
    )


app.include_router(api_router)
