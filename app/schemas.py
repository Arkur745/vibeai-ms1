from datetime import datetime
from typing import List, Optional
from typing import Any
from typing import Dict

from pydantic import BaseModel, HttpUrl


class TaskSubmitResponse(BaseModel):
    task_id: str
    status: str


class InferenceResult(BaseModel):
    genre: str
    tempo: float
    key: Optional[str] = None
    valence: float
    arousal: float
    mood: str
    vibe: str
    gradcam_url: Optional[str] = None
    explanations: Optional[List[str]] = None


class TaskStatusResponse(BaseModel):
    task_id: str
    status: str
    result: Optional[InferenceResult] = None
    error: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str
    error: str


class HistoryItemResponse(BaseModel):
    task_id: str
    created_at: datetime
    filename: Optional[str] = None
    s3_key: Optional[str] = None
    predictions: Optional[Any] = None
    gradcam_url: Optional[str] = None
    duration: float
    status: str
    benchmark: Optional[Any] = None
    model_version: Optional[str] = None
    model_architecture: Optional[str] = None
    model_checkpoint_hash: Optional[str] = None
    model_training_timestamp: Optional[str] = None
    gpu_info: Optional[Any] = None


class HistoryListResponse(BaseModel):
    items: List[HistoryItemResponse]
    page: int
    limit: int
    total: int


class ModelExportResponse(BaseModel):
    exported_models: Dict[str, str]


class HealthResponse(BaseModel):
    status: str
    ready: bool


class ReadyResponse(BaseModel):
    status: str
    redis: bool
    s3: bool
    model_loaded: bool
    started_at: Optional[datetime] = None
