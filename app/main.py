from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import uuid
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram
import time
from app.workers.tasks import process_audio_task
from app.core.storage import upload_file_to_s3

app = FastAPI(
    title="Audio Emotion Recognition API"
)
Instrumentator().instrument(app).expose(app)
# ---------------------------------
# Custom ML Metrics
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

UPLOAD_DIR = Path("temp_uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


@app.get("/")
def root():

    return {
        "message": "Audio Emotion API Running"
    }


@app.post("/predict")
async def predict_audio(
    file: UploadFile = File(...)
):

    start_time = time.time()

    INFERENCE_REQUESTS.inc()

    try:

        # ---------------------------------
        # Save Uploaded File
        # ---------------------------------

        unique_name = f"{uuid.uuid4()}.mp3"

        temp_path = UPLOAD_DIR / unique_name

        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(
                file.file,
                buffer
            )

        # ---------------------------------
        # Submit Async Task
        # ---------------------------------

        # ---------------------------------
        # Upload To S3
        # ---------------------------------
        
        s3_key = f"uploads/{unique_name}"
        
        upload_file_to_s3(
            temp_path,
            s3_key
        )
        
        # ---------------------------------
        # Remove Local Upload
        # ---------------------------------
        
        temp_path.unlink(
            missing_ok=True
        )
        
        # ---------------------------------
        # Submit Async Task
        # ---------------------------------
        
        task = process_audio_task.delay(
            s3_key
        )
        
        duration = time.time() - start_time

        INFERENCE_DURATION.observe(duration)

        return {
            "message": "Inference task submitted",
            "task_id": task.id,
            "filename": file.filename,
            "request_time_seconds": round(duration, 4)
        }

    except Exception as e:
    
        INFERENCE_FAILURES.inc()

        return {
            "error": str(e)
        }
        
        
@app.get("/result/{task_id}")
async def get_result(task_id: str):

    task = process_audio_task.AsyncResult(task_id)

    if task.state == "PENDING":

        return {
            "status": "PENDING"
        }

    elif task.state == "SUCCESS":

        return {
            "status": "SUCCESS",
            "result": task.result
        }

    elif task.state == "FAILURE":

        return {
            "status": "FAILURE",
            "error": str(task.result)
        }

    else:

        return {
            "status": task.state
        }
