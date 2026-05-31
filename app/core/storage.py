from pathlib import Path
import logging

from app.core.config import settings
from app.core.s3_client import s3_client

logger = logging.getLogger(__name__)


def upload_file_to_s3(local_path: Path, s3_key: str) -> str:
    local_path = Path(local_path)
    s3_client.upload_file(
        Filename=str(local_path),
        Bucket=settings.s3_bucket_name,
        Key=s3_key,
    )
    return s3_key


def download_file_from_s3(s3_key: str, download_path: Path) -> Path:
    download_path = Path(download_path)
    download_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Downloading from S3: s3://{settings.s3_bucket_name}/{s3_key} -> {download_path}"
    )

    s3_client.download_file(
        Bucket=settings.s3_bucket_name,
        Key=s3_key,
        Filename=str(download_path),
    )
    logger.info(f"Successfully downloaded: {download_path}")
    return download_path


def get_s3_url(s3_key: str) -> str:
    base_url = settings.s3_base_url
    if not base_url:
        raise ValueError("S3 base URL is not configured")
    return f"{base_url.rstrip('/')}/{s3_key.lstrip('/')}"


def delete_file_from_s3(s3_key: str) -> None:
    s3_client.delete_object(
        Bucket=settings.s3_bucket_name,
        Key=s3_key,
    )
