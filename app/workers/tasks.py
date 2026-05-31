import json
import time
from pathlib import Path

from app.core.config import settings
from app.core.logger import logger
from app.observability.metrics import TOTAL_TASK_DURATION_SECONDS
from app.persistence.db import save_inference_task
from app.services.inference_service import run_inference
from app.workers.celery_app import celery


@celery.task(
    bind=True,
    name="app.workers.tasks.process_audio_task",
    autoretry_for=(Exception,),
    retry_kwargs={"max_retries": 2},
    default_retry_delay=60,
)
def process_audio_task(self, task_id: str):
    task_start = time.time()
    temp_dir = Path(settings.temp_upload_dir)

    logger.info(f"Task {task_id} STARTED | Local processing")

    try:
        local_filename = f"{task_id}.mp3"
        local_path = temp_dir / local_filename

        if not local_path.exists():
            raise FileNotFoundError(f"Uploaded file not found locally: {local_path}")

        logger.info(f"Processing audio from local path: {local_path}")
        prediction = run_inference(local_path, task_id=task_id)

        gradcam_local_path = prediction.pop("gradcam_path", None)
        gradcam_url = ""
        if gradcam_local_path and Path(gradcam_local_path).exists():
            gradcam_filename = Path(gradcam_local_path).name
            # Construct the static artifact URL served directly by FastAPI
            gradcam_url = f"{settings.api_base_url}/artifacts/gradcam/{gradcam_filename}"
            prediction["gradcam_url"] = gradcam_url
            logger.info(f"GradCAM local artifact verified at: {gradcam_local_path} | URL: {gradcam_url}")
        else:
            logger.warning(f"GradCAM local path is empty or does not exist for task {task_id}")

        total_duration = time.time() - task_start
        TOTAL_TASK_DURATION_SECONDS.observe(total_duration)

        # Retrieve and preserve original filename from database
        original_filename = local_filename
        try:
            from app.persistence.db import get_inference_task
            existing_task = get_inference_task(task_id)
            if existing_task and existing_task.get("filename"):
                original_filename = existing_task["filename"]
        except Exception as db_err:
            logger.error(f"Failed to fetch original filename for task {task_id}: {db_err}")

        save_inference_task(
            task_id=task_id,
            filename=original_filename,
            s3_key="",  # No S3 usage
            predictions=prediction,
            gradcam_url=gradcam_url,
            duration=total_duration,
            status="success",
            benchmark=prediction.get("benchmark", {}),
            model_version=prediction.get(
                "model_info", {}).get("genre_model_version"),
            model_architecture=prediction.get(
                "model_info", {}).get("genre_model_architecture"),
            model_checkpoint_hash=prediction.get(
                "model_info", {}).get("genre_model_checkpoint_hash"),
            model_training_timestamp=prediction.get(
                "model_info", {}).get("genre_model_training_timestamp"),
            gpu_info=prediction.get("gpu_info", {}),
        )

        logger.info(
            f"Task {task_id} SUCCESS | duration={total_duration:.2f}s | gradcam_url={gradcam_url}"
        )
        return prediction
    except Exception as exc:
        logger.error(f"Task {task_id} FAILED: {exc}", exc_info=True)
        original_filename = f"{task_id}.mp3"
        try:
            from app.persistence.db import get_inference_task
            existing_task = get_inference_task(task_id)
            if existing_task and existing_task.get("filename"):
                original_filename = existing_task["filename"]
        except Exception:
            pass

        save_inference_task(
            task_id=task_id,
            filename=original_filename,
            s3_key="",
            predictions={},
            gradcam_url="",
            duration=time.time() - task_start,
            status="failed",
        )
        raise
