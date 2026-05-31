import time
import uuid
from typing import Optional

from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from celery.result import AsyncResult

from app.core.logger import logger
from app.core.config import settings
from app.observability.metrics import INFERENCE_DURATION, INFERENCE_FAILURES, INFERENCE_REQUESTS
from app.persistence.db import get_inference_task, list_inference_tasks, save_inference_task, delete_inference_task
from app.workers.celery_app import celery
from app.workers.tasks import process_audio_task
from app.schemas import (
    ErrorResponse,
    HistoryListResponse,
    HistoryItemResponse,
    InferenceResult,
    TaskStatusResponse,
    TaskSubmitResponse,
    ModelExportResponse,
)
from app.ml.inference import export_all_models

router = APIRouter(prefix="/api/v1", tags=["Inference"])


@router.post(
    "/analyze",
    response_model=TaskSubmitResponse,
    responses={200: {"description": "Task submitted successfully."},
               500: {"model": ErrorResponse}},
)
async def analyze_audio(file: UploadFile = File(...)):
    request_start = time.time()
    INFERENCE_REQUESTS.inc()

    try:
        task_id = str(uuid.uuid4())
        filename = f"{task_id}.mp3"
        upload_path = settings.temp_upload_dir / filename
        upload_path.parent.mkdir(parents=True, exist_ok=True)

        with upload_path.open("wb") as buffer:
            buffer.write(await file.read())

        logger.info(f"Saved uploaded audio locally for task {task_id}: {upload_path}")

        # Trigger Celery task using custom task_id
        task = process_audio_task.apply_async(args=[task_id], task_id=task_id)

        original_filename = file.filename or filename
        save_inference_task(
            task_id=task_id,
            filename=original_filename,
            s3_key="",  # No S3 usage
            predictions={},
            gradcam_url="",
            duration=0.0,
            status="pending",
        )

        INFERENCE_DURATION.observe(time.time() - request_start)
        return TaskSubmitResponse(task_id=task_id, status="submitted")
    except Exception as exc:
        INFERENCE_FAILURES.inc()
        logger.error(f"Unable to submit inference task: {exc}")
        raise HTTPException(
            status_code=500, detail="Unable to submit inference task")


@router.get(
    "/result/{task_id}",
    response_model=TaskStatusResponse,
    responses={200: {"description": "Task status returned."}, 404: {
        "model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def get_result(task_id: str):
    try:
        task = AsyncResult(task_id, app=celery)
        logger.info(f"Task {task_id} state: {task.state}")

        if task.state == "PENDING":
            return TaskStatusResponse(task_id=task_id, status="pending")

        if task.state == "STARTED":
            return TaskStatusResponse(task_id=task_id, status="running")

        if task.state == "SUCCESS":
            raw_result = task.result
            if not isinstance(raw_result, dict):
                raise HTTPException(
                    status_code=500, detail="Invalid task result format")
            result_data = InferenceResult(**raw_result)
            return TaskStatusResponse(task_id=task_id, status="completed", result=result_data)

        if task.state == "FAILURE":
            return TaskStatusResponse(task_id=task_id, status="failed", error=str(task.result))

        if task.state == "RETRY":
            return TaskStatusResponse(task_id=task_id, status="retrying")

        return TaskStatusResponse(task_id=task_id, status=task.state.lower())
    except Exception as exc:
        logger.error(f"Error fetching task result {task_id}: {exc}")
        raise HTTPException(
            status_code=500, detail="Unable to retrieve task result")


@router.get(
    "/history",
    response_model=HistoryListResponse,
    responses={200: {"description": "Historical inference tasks returned."}},
)
def get_history(
    page: int = Query(1, ge=1),
    limit: int = Query(20, ge=1, le=100),
    sort: str = Query("desc", regex="^(asc|desc)$"),
):
    history = list_inference_tasks(
        page=page, limit=limit, sort_desc=(sort == "desc"))
    items = [HistoryItemResponse(**item) for item in history["items"]]
    return HistoryListResponse(
        items=items,
        page=page,
        limit=limit,
        total=history["total"],
    )


@router.get(
    "/history/{task_id}",
    response_model=HistoryItemResponse,
    responses={200: {"description": "Historical inference task returned."}, 404: {
        "model": ErrorResponse}},
)
def get_history_item(task_id: str):
    item = get_inference_task(task_id)
    if not item:
        raise HTTPException(status_code=404, detail="Task not found")
    return HistoryItemResponse(**item)


@router.post(
    "/export-models",
    response_model=ModelExportResponse,
    responses={500: {"model": ErrorResponse}},
)
def export_models():
    try:
        exported_models = export_all_models()
        if not exported_models:
            raise HTTPException(
                status_code=500, detail="No models were available to export")
        return ModelExportResponse(exported_models=exported_models)
    except Exception as exc:
        logger.error(f"ONNX model export failed: {exc}")
        raise HTTPException(status_code=500, detail="Model export failed")


@router.delete(
    "/result/{task_id}",
    responses={200: {"description": "Task deleted successfully."}, 404: {
        "model": ErrorResponse}, 500: {"model": ErrorResponse}},
)
def delete_result(task_id: str):
    try:
        success = delete_inference_task(task_id)
        if not success:
            raise HTTPException(status_code=404, detail="Task not found or unable to delete")
        return {"status": "deleted", "task_id": task_id}
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Error deleting task {task_id}: {exc}")
        raise HTTPException(
            status_code=500, detail="Unable to delete task result")
