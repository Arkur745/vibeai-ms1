import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    app_name: str = "VibeAI Audio Intelligence API"
    env: str = Field("development", env="ENV")
    debug: bool = Field(False, env="DEBUG")

    redis_url: str = Field("redis://localhost:6379/0", env="REDIS_URL")
    celery_result_backend: Optional[str] = Field(
        None, env="CELERY_RESULT_BACKEND")

    aws_access_key_id: Optional[str] = Field(None, env="AWS_ACCESS_KEY_ID")
    aws_secret_access_key: Optional[str] = Field(
        None, env="AWS_SECRET_ACCESS_KEY")
    aws_region: Optional[str] = Field(None, env="AWS_REGION")
    s3_bucket_name: Optional[str] = Field(None, env="S3_BUCKET_NAME")
    s3_endpoint_url: Optional[str] = Field(None, env="S3_ENDPOINT_URL")
    s3_public_url: Optional[str] = Field(None, env="S3_PUBLIC_URL")

    use_gpu: bool = Field(False, env="USE_GPU")

    model_dir: Path = BASE_DIR / "models"
    genre_model_path: Path = BASE_DIR / "models" / "best_resnet_genre_classifier.pth"
    emotion_model_path: Path = BASE_DIR / "models" / "best_resnet_emotion_model.pth"
    genre_label_encoder_path: Path = BASE_DIR / \
        "models" / "resnet_genre_label_encoder.pkl"
    temp_upload_dir: Path = BASE_DIR / "temp_uploads"
    artifact_dir: Path = BASE_DIR / "artifacts"
    gradcam_dir: Path = BASE_DIR / "artifacts" / "gradcam"
    onnx_export_dir: Path = BASE_DIR / "artifacts" / "onnx"
    database_path: Path = BASE_DIR / "data" / "vibeai.db"
    worker_metrics_port: int = Field(8001, env="WORKER_METRICS_PORT")
    api_base_url: str = Field("http://localhost:8000", env="API_BASE_URL")
    gradcam_retention_days: int = Field(7, env="GRADCAM_RETENTION_DAYS")
    vibeai_internal_secret: Optional[str] = Field(None, env="VIBEAI_INTERNAL_SECRET")

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "case_sensitive": False,
        "protected_namespaces": ("settings_",),
    }

    @field_validator("celery_result_backend", mode="after")
    def set_result_backend(cls, value, info):
        if value:
            return value
        return os.getenv("CELERY_RESULT_BACKEND") or info.data.get("redis_url")

    @model_validator(mode="after")
    def validate_production_vars(self) -> 'Settings':
        env_mode = self.env.lower()
        if env_mode == "production" or not self.debug:
            missing = []
            if not self.vibeai_internal_secret:
                missing.append("VIBEAI_INTERNAL_SECRET")
            if not self.aws_access_key_id:
                missing.append("AWS_ACCESS_KEY_ID")
            if not self.aws_secret_access_key:
                missing.append("AWS_SECRET_ACCESS_KEY")
            if not self.s3_bucket_name:
                missing.append("S3_BUCKET_NAME")
            if missing:
                raise ValueError(
                    f"CRITICAL: Missing required production environment variables: {', '.join(missing)}"
                )
        return self

    @property
    def s3_base_url(self) -> Optional[str]:
        if self.s3_public_url:
            return self.s3_public_url
        if self.s3_bucket_name:
            if self.aws_region:
                return f"https://{self.s3_bucket_name}.s3.{self.aws_region}.amazonaws.com"
            return f"https://{self.s3_bucket_name}.s3.amazonaws.com"
        return None


settings = Settings()
