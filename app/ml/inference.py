from pathlib import Path

import joblib
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from app.core.config import settings
from app.core.logger import logger


def export_model_to_onnx(
    model: nn.Module,
    export_path: Path,
    input_shape: tuple = (1, 3, 224, 224),
    output_names: tuple = ("output",),
) -> Path:
    export_path.parent.mkdir(parents=True, exist_ok=True)
    dummy_input = torch.randn(input_shape, device=DEVICE)
    torch.onnx.export(
        model,
        dummy_input,
        str(export_path),
        input_names=["input"],
        output_names=list(output_names),
        opset_version=13,
        do_constant_folding=True,
    )
    return export_path


def export_all_models() -> dict:
    results = {}
    if genre_model is not None:
        results["genre_model"] = str(
            export_model_to_onnx(
                genre_model,
                Path(settings.onnx_export_dir) / "genre_model.onnx",
            )
        )
    if emotion_model is not None:
        results["emotion_model"] = str(
            export_model_to_onnx(
                emotion_model,
                Path(settings.onnx_export_dir) / "emotion_model.onnx",
                output_names=("valence", "arousal"),
            )
        )
    return results


DEVICE = torch.device(
    "cuda" if settings.use_gpu and torch.cuda.is_available() else "cpu")
logger.info(f"Using device for inference: {DEVICE}")

image_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])


class GenreResNet(nn.Module):
    def __init__(self, num_classes: int):
        super().__init__()
        self.model = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(nn.Dropout(
            0.3), nn.Linear(in_features, num_classes))

    def forward(self, x):
        return self.model(x)


class EmotionResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()
        self.shared = nn.Sequential(
            nn.Linear(in_features, 256), nn.ReLU(), nn.Dropout(0.3))
        self.valence_head = nn.Linear(256, 1)
        self.arousal_head = nn.Linear(256, 1)

    def forward(self, x):
        features = self.backbone(x)
        shared = self.shared(features)
        valence = self.valence_head(shared)
        arousal = self.arousal_head(shared)
        return valence, arousal


def _load_checkpoint(path: Path):
    return torch.load(path, map_location=DEVICE, weights_only=False)


def _load_model(model_class, checkpoint_path: Path, model_kwargs=None):
    model_kwargs = model_kwargs or {}
    checkpoint = _load_checkpoint(checkpoint_path)
    model = model_class(**model_kwargs)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()
    return model


logger.info("Loading genre model...")
genre_model = None
try:
    genre_model = _load_model(GenreResNet, Path(
        settings.genre_model_path), {"num_classes": 10})
except Exception as exc:
    logger.error(f"Unable to load genre model: {exc}")

logger.info("Loading genre label encoder...")
genre_encoder = None
try:
    genre_encoder = joblib.load(Path(settings.genre_label_encoder_path))
except Exception as exc:
    logger.error(f"Unable to load genre label encoder: {exc}")

logger.info("Loading emotion model...")
emotion_model = None
try:
    emotion_model = _load_model(
        EmotionResNet, Path(settings.emotion_model_path))
except Exception as exc:
    logger.error(f"Unable to load emotion model: {exc}")


def predict_genre(image_tensor: torch.Tensor) -> int:
    if genre_model is None:
        raise RuntimeError("Genre model is not loaded")
    with torch.inference_mode():
        logits = genre_model(image_tensor)
        return int(torch.argmax(logits, dim=1).item())


def predict_emotion_dl(image_tensor: torch.Tensor) -> dict:
    if emotion_model is None:
        raise RuntimeError("Emotion model is not loaded")
    with torch.inference_mode():
        valence, arousal = emotion_model(image_tensor)
    return {
        "valence": round(float(valence.view(-1)[0].item()), 3),
        "arousal": round(float(arousal.view(-1)[0].item()), 3),
    }


def get_model_status() -> dict:
    return {
        "genre_model": "loaded" if genre_model is not None else "unavailable",
        "emotion_model": "loaded" if emotion_model is not None else "unavailable",
        "label_encoder": "loaded" if genre_encoder is not None else "unavailable",
    }


if __name__ == "__main__":
    print(get_model_status())
