"""
This module is deprecated. Please import metrics from app.observability.metrics instead.
"""
from app.observability.metrics import (
    INFERENCE_REQUESTS,
    INFERENCE_FAILURES,
    SPECTROGRAM_GENERATION_SECONDS,
    MODEL_INFERENCE_SECONDS,
    GRADCAM_GENERATION_SECONDS,
    TOTAL_TASK_DURATION_SECONDS as TOTAL_INFERENCE_SECONDS,
)
