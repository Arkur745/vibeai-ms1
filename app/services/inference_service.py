from app.observability.metrics import (
    GRADCAM_GENERATION_SECONDS,
    MODEL_INFERENCE_SECONDS,
    SPECTROGRAM_GENERATION_SECONDS,
)
from app.observability.gpu import collect_gpu_status, update_gpu_metrics
from app.ml.mood_mapping import classify_mood
from app.ml.model_info import get_combined_model_metadata, get_model_readiness
from app.ml.inference import (
    DEVICE,
    genre_encoder,
    genre_model,
    image_transform,
    predict_emotion_dl,
    predict_genre,
)
from app.ml.gradcam import generate_gradcam
from app.core.logger import logger
from io import BytesIO
from pathlib import Path
import time
from typing import Dict, Optional

import librosa
import librosa.display
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")


def _create_spectrogram_image(y: np.ndarray, sr: int) -> Image.Image:
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    fig, ax = plt.subplots(figsize=(4, 4), dpi=100)
    ax.axis("off")
    librosa.display.specshow(mel_db, sr=sr, ax=ax)

    buffer = BytesIO()
    fig.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close(fig)
    buffer.seek(0)

    return Image.open(buffer).convert("RGB")


def _extract_tempo(y: np.ndarray, sr: int) -> float:
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    tempo_val = float(tempo.item()) if hasattr(tempo, "item") else float(tempo)
    if tempo_val < 90:
        tempo_val *= 2
    elif tempo_val > 180:
        tempo_val /= 2
    return round(tempo_val, 2)


def _extract_key(y: np.ndarray, sr: int) -> str:
    try:
        # 1. Compute constant-Q chromagram (superior pitch representation)
        chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
        chroma_profile = np.sum(chroma, axis=1)
        
        # Normalize pitch profile
        profile_sum = np.sum(chroma_profile)
        if profile_sum > 0:
            chroma_profile = chroma_profile / profile_sum
            
        # 2. Define Krumhansl-Schmuckler (K-S) key templates
        major_template = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88])
        minor_template = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17])
        
        pitches = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        
        best_key = "Unknown"
        best_corr = -1.0
        
        # Correlate profile with shifted major/minor transpositions
        for shift in range(12):
            shifted_major = np.roll(major_template, shift)
            shifted_minor = np.roll(minor_template, shift)
            
            corr_major = np.corrcoef(chroma_profile, shifted_major)[0, 1]
            corr_minor = np.corrcoef(chroma_profile, shifted_minor)[0, 1]
            
            if corr_major > best_corr:
                best_corr = corr_major
                best_key = f"{pitches[shift]} Major"
                
            if corr_minor > best_corr:
                best_corr = corr_minor
                best_key = f"{pitches[shift]} Minor"
                
        return best_key
    except Exception as exc:
        logger.warning(f"Key extraction failed: {exc}")
        return "Unknown"


def run_inference(audio_path: Path, task_id: str) -> Dict[str, Optional[object]]:
    result: Dict[str, Optional[object]] = {
        "genre": None,
        "tempo": None,
        "key": None,
        "valence": None,
        "arousal": None,
        "mood": None,
        "vibe": None,
        "gradcam_path": "",
        "benchmark": {},
        "model_info": {},
        "gpu_info": {},
    }

    update_gpu_metrics()
    task_start = time.time()

    y, sr = librosa.load(str(audio_path), sr=22050)

    spectrogram_start = time.time()
    with SPECTROGRAM_GENERATION_SECONDS.time():
        spectrogram_image = _create_spectrogram_image(y, sr)
    spectrogram_seconds = round(time.time() - spectrogram_start, 3)

    image_tensor = image_transform(spectrogram_image).unsqueeze(0).to(DEVICE)

    inference_start = time.time()
    with MODEL_INFERENCE_SECONDS.time():
        genre_idx = predict_genre(image_tensor)
        result["genre"] = (
            genre_encoder.inverse_transform([genre_idx])[0]
            if genre_encoder is not None
            else "unknown"
        )
        emotion_output = predict_emotion_dl(image_tensor)
        result["valence"] = emotion_output["valence"]
        result["arousal"] = emotion_output["arousal"]
    model_inference_seconds = round(time.time() - inference_start, 3)

    gradcam_start = time.time()
    with GRADCAM_GENERATION_SECONDS.time():
        try:
            result["gradcam_path"] = generate_gradcam(
                image_tensor,
                genre_idx,
                task_id,
                spectrogram_image,
                genre_model,
            )
        except Exception as exc:
            logger.exception(
                f"GradCAM generation failed for task {task_id} due to exception")
            result["gradcam_path"] = ""
    gradcam_seconds = round(time.time() - gradcam_start, 3)

    result["tempo"] = _extract_tempo(y, sr)
    result["key"] = _extract_key(y, sr)
    try:
        mood_data = classify_mood(
            float(result["valence"]), float(result["arousal"]))
        result["mood"] = mood_data.get("mood")
        result["vibe"] = mood_data.get("vibe")
    except Exception as exc:
        logger.warning(f"Mood mapping failed for task {task_id}: {exc}")
        result["mood"] = "unknown"
        result["vibe"] = "unknown"

    model_metadata = get_combined_model_metadata()
    gpu_status = collect_gpu_status()
    result["benchmark"] = {
        "spectrogram_seconds": spectrogram_seconds,
        "model_inference_seconds": model_inference_seconds,
        "gradcam_seconds": gradcam_seconds,
        "total_seconds": round(time.time() - task_start, 3),
    }
    result["model_info"] = model_metadata
    result["gpu_info"] = gpu_status

    return result


def check_dependencies() -> Dict[str, object]:
    readiness = get_model_readiness()
    gpu_status = collect_gpu_status()
    return {
        "models": readiness,
        "gpu": gpu_status,
    }
