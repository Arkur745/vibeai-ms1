from fastapi import FastAPI, UploadFile, File
from pathlib import Path
import shutil
import uuid

from app.inference import predict_emotion

app = FastAPI(
    title="Audio Emotion Recognition API"
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
    # Run Inference
    # ---------------------------------

    prediction = predict_emotion(
        temp_path
    )

    # ---------------------------------
    # Cleanup
    # ---------------------------------

    temp_path.unlink(missing_ok=True)

    return {
        "filename": file.filename,
        "prediction": prediction
    }