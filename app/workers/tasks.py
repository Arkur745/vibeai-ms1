from pathlib import Path
import uuid
import traceback

from app.workers.celery_app import celery

from app.ml.inference import predict_emotion

from app.core.storage import (
    download_file_from_s3
)

from app.core.config import (
    TEMP_UPLOAD_DIR
)


@celery.task
def process_audio_task(s3_key):

    try:

        print("\n====================")
        print("TASK STARTED")
        print("====================")

        print(f"S3 Key: {s3_key}")

        # ---------------------------------
        # Download Audio Locally
        # ---------------------------------

        local_filename = f"{uuid.uuid4()}.mp3"

        local_path = (
            TEMP_UPLOAD_DIR / local_filename
        )

        download_file_from_s3(
            s3_key,
            local_path
        )

        print(f"Downloaded: {local_path}")

        # ---------------------------------
        # Run Inference
        # ---------------------------------

        prediction = predict_emotion(
            local_path
        )

        print("\nPrediction Success")
        print(prediction)

        # ---------------------------------
        # Cleanup Local Temp File
        # ---------------------------------

        local_path.unlink(
            missing_ok=True
        )

        return prediction

    except Exception as e:

        print("\nTASK FAILED")
        print(str(e))

        traceback.print_exc()

        raise e
