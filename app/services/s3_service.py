import logging
import time
from pathlib import Path
from typing import Dict

import boto3
from botocore.exceptions import ClientError

from app.core.config import settings
from app.observability.metrics import AUDIO_DOWNLOAD_SECONDS, S3_UPLOAD_SECONDS

logger = logging.getLogger(__name__)


def get_s3_client():
    return boto3.client(
        "s3",
        aws_access_key_id=settings.aws_access_key_id,
        aws_secret_access_key=settings.aws_secret_access_key,
        region_name=settings.aws_region,
        endpoint_url=settings.s3_endpoint_url,
    )


def verify_bucket() -> Dict[str, object]:
    client = get_s3_client()
    try:
        client.head_bucket(Bucket=settings.s3_bucket_name)
        return {"status": "ok"}
    except ClientError as exc:
        logger.error(f"S3 bucket readiness check failed: {exc}")
        return {"status": "error", "message": str(exc)}


def get_s3_url(s3_key: str) -> str:
    if not settings.s3_bucket_name:
        raise ValueError("S3 bucket name is not configured")
    base_url = settings.s3_base_url
    if not base_url:
        raise ValueError("S3 base URL is not configured")
    return f"{base_url.rstrip('/')}/{s3_key.lstrip('/')}"


def download_file_from_s3(s3_key: str, download_path: Path) -> Path:
    download_path = Path(download_path)
    download_path.parent.mkdir(parents=True, exist_ok=True)
    client = get_s3_client()

    start = time.time()
    try:
        client.download_file(
            Bucket=settings.s3_bucket_name,
            Key=s3_key,
            Filename=str(download_path),
        )
    except ClientError as exc:
        logger.error(f"Failed to download {s3_key} from S3: {exc}")
        raise
    finally:
        AUDIO_DOWNLOAD_SECONDS.observe(time.time() - start)

    logger.info(f"Downloaded {s3_key} to {download_path}")
    return download_path


def upload_file_to_s3(local_path: Path, s3_key: str) -> str:
    local_path = Path(local_path)
    client = get_s3_client()

    start = time.time()
    try:
        client.upload_file(
            Filename=str(local_path),
            Bucket=settings.s3_bucket_name,
            Key=s3_key,
        )
    except ClientError as exc:
        logger.error(f"Failed to upload {local_path} to S3: {exc}")
        raise
    finally:
        S3_UPLOAD_SECONDS.observe(time.time() - start)

    logger.info(
        f"Uploaded {local_path} to s3://{settings.s3_bucket_name}/{s3_key}")
    return s3_key
