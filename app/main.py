from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import uuid
import time

from celery.result import AsyncResult

from prometheus_fastapi_instrumentator import (
    Instrumentator
)

from prometheus_client import (
    Counter,
    Histogram
)

from app.workers.tasks import (
    process_audio_task
)

from app.core.storage import (
    upload_file_to_s3
)

from app.schemas import (
    TaskSubmitResponse,
    TaskStatusResponse,
    AnalysisResult
)


# ---------------------------------
# FastAPI App
# ---------------------------------

app = FastAPI(

    title="VibeAI Audio Intelligence API",

    version="1.0.0"
)

Instrumentator().instrument(app).expose(app)


# ---------------------------------
# Custom Metrics
# ---------------------------------

INFERENCE_REQUESTS = Counter(

    "inference_requests_total",

    "Total number of inference requests"
)

INFERENCE_FAILURES = Counter(

    "inference_failures_total",

    "Total number of failed inference requests"
)

INFERENCE_DURATION = Histogram(

    "inference_duration_seconds",

    "Inference latency in seconds"
)


# ---------------------------------
# Upload Directory
# ---------------------------------

UPLOAD_DIR = Path(
    "temp_uploads"
)

UPLOAD_DIR.mkdir(
    exist_ok=True
)


# ---------------------------------
# Root Endpoint
# ---------------------------------

@app.get("/")
def root():

    return {

        "service": "VibeAI Audio Intelligence API",

        "status": "running",

        "version": "1.0.0"
    }


# ---------------------------------
# Analyze Audio
# ---------------------------------

@app.post(

    "/api/v1/analyze",

    response_model=TaskSubmitResponse
)
async def analyze_audio(

    file: UploadFile = File(...)
):

    start_time = time.time()

    INFERENCE_REQUESTS.inc()

    try:

        # ---------------------------------
        # Save Upload Locally
        # ---------------------------------

        unique_name = f"{uuid.uuid4()}.mp3"

        temp_path = UPLOAD_DIR / unique_name

        with open(temp_path, "wb") as buffer:

            shutil.copyfileobj(

                file.file,

                buffer
            )

        # ---------------------------------
        # Upload To S3
        # ---------------------------------

        s3_key = f"uploads/{unique_name}"

        upload_file_to_s3(

            temp_path,

            s3_key
        )

        # ---------------------------------
        # Remove Temporary File
        # ---------------------------------

        temp_path.unlink(
            missing_ok=True
        )

        # ---------------------------------
        # Submit Celery Task
        # ---------------------------------

        task = process_audio_task.delay(
            s3_key
        )

        # ---------------------------------
        # Metrics
        # ---------------------------------

        duration = time.time() - start_time

        INFERENCE_DURATION.observe(
            duration
        )

        # ---------------------------------
        # Response
        # ---------------------------------

        return TaskSubmitResponse(

            task_id=task.id,

            status="submitted"
        )

    except Exception as e:

        INFERENCE_FAILURES.inc()

        return {

            "error": str(e)
        }


# ---------------------------------
# Get Task Result
# ---------------------------------

@app.get(

    "/api/v1/result/{task_id}",

    response_model=TaskStatusResponse
)
async def get_result(task_id: str):

    task = AsyncResult(task_id)

    # ---------------------------------
    # Pending
    # ---------------------------------

    if task.state == "PENDING":

        return TaskStatusResponse(

            task_id=task_id,

            status="pending"
        )

    # ---------------------------------
    # Success
    # ---------------------------------

    elif task.state == "SUCCESS":

        result_data = AnalysisResult(
            **task.result
        )

        return TaskStatusResponse(

            task_id=task_id,

            status="completed",

            result=result_data
        )

    # ---------------------------------
    # Failure
    # ---------------------------------

    elif task.state == "FAILURE":

        return TaskStatusResponse(

            task_id=task_id,

            status="failed"
        )

    # ---------------------------------
    # Other States
    # ---------------------------------

    return TaskStatusResponse(

        task_id=task_id,

        status=task.state.lower()
    )
