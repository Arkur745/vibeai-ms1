from pathlib import Path

from app.core.s3_client import s3_client
from app.core.config import S3_BUCKET_NAME


def upload_file_to_s3(
    local_path,
    s3_key
):

    s3_client.upload_file(
        Filename=str(local_path),

        Bucket=S3_BUCKET_NAME,

        Key=s3_key
    )

    return s3_key


def download_file_from_s3(
    s3_key,
    download_path
):

    s3_client.download_file(
        Bucket=S3_BUCKET_NAME,

        Key=s3_key,

        Filename=str(download_path)
    )

    return download_path


def delete_file_from_s3(
    s3_key
):

    s3_client.delete_object(
        Bucket=S3_BUCKET_NAME,
        Key=s3_key
    )
