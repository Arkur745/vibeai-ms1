import hashlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import torch

from app.core.config import settings
from app.core.logger import logger


@dataclass
class ModelMetadata:
    name: str
    architecture: str
    version: str
    checkpoint_hash: str
    training_timestamp: str


def _compute_file_hash(model_path: Path) -> str:
    hash_obj = hashlib.sha256()
    with model_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()


def _load_training_timestamp(model_path: Path) -> str:
    try:
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        metadata = checkpoint.get("metadata") if isinstance(
            checkpoint, dict) else None
        if metadata and metadata.get("training_timestamp"):
            return str(metadata["training_timestamp"])
    except Exception:
        logger.warning(f"Unable to read checkpoint metadata from {model_path}")
    return datetime.utcfromtimestamp(model_path.stat().st_mtime).isoformat(timespec="seconds")


def get_genre_model_metadata() -> ModelMetadata:
    model_path = Path(settings.genre_model_path)
    return ModelMetadata(
        name="genre_model",
        architecture="resnet18",
        version=model_path.stem,
        checkpoint_hash=_compute_file_hash(model_path),
        training_timestamp=_load_training_timestamp(model_path),
    )


def get_emotion_model_metadata() -> ModelMetadata:
    model_path = Path(settings.emotion_model_path)
    return ModelMetadata(
        name="emotion_model",
        architecture="resnet18",
        version=model_path.stem,
        checkpoint_hash=_compute_file_hash(model_path),
        training_timestamp=_load_training_timestamp(model_path),
    )


def get_model_readiness() -> Dict[str, str]:
    return {
        "genre_model": "loaded" if Path(settings.genre_model_path).exists() else "missing",
        "emotion_model": "loaded" if Path(settings.emotion_model_path).exists() else "missing",
        "label_encoder": "loaded" if Path(settings.genre_label_encoder_path).exists() else "missing",
    }


def get_combined_model_metadata() -> Dict[str, Optional[str]]:
    genre_metadata = get_genre_model_metadata()
    emotion_metadata = get_emotion_model_metadata()
    return {
        "genre_model_version": genre_metadata.version,
        "genre_model_architecture": genre_metadata.architecture,
        "genre_model_checkpoint_hash": genre_metadata.checkpoint_hash,
        "genre_model_training_timestamp": genre_metadata.training_timestamp,
        "emotion_model_version": emotion_metadata.version,
        "emotion_model_architecture": emotion_metadata.architecture,
        "emotion_model_checkpoint_hash": emotion_metadata.checkpoint_hash,
        "emotion_model_training_timestamp": emotion_metadata.training_timestamp,
    }
