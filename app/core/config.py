from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------
# Base Paths
# ---------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent.parent

MODELS_DIR = BASE_DIR / "models"

TEMP_UPLOAD_DIR = BASE_DIR / "temp_uploads"


# ---------------------------------
# Model Paths
# ---------------------------------

VALENCE_MODEL_PATH = "models/xgboost_valence_v2.pkl"

AROUSAL_MODEL_PATH = "models/xgboost_arousal_v2.pkl"


# ---------------------------------
# Redis / Celery
# ---------------------------------

REDIS_URL = os.getenv(
    "REDIS_URL",
    "redis://localhost:6379/0"
)

# ---------------------------------
# AWS / S3
# ---------------------------------

AWS_ACCESS_KEY_ID = os.getenv(
    "AWS_ACCESS_KEY_ID"
)

AWS_SECRET_ACCESS_KEY = os.getenv(
    "AWS_SECRET_ACCESS_KEY"
)

AWS_REGION = os.getenv(
    "AWS_REGION"
)

S3_BUCKET_NAME = os.getenv(
    "S3_BUCKET_NAME"
)
GENRE_MODEL_PATH = "models/xgboost_genre_classifier_v1.pkl"
