import json
import sqlite3
from datetime import datetime
from pathlib import Path

from app.core.config import settings
from app.core.logger import logger

DB_PATH = Path(settings.database_path)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

connection = sqlite3.connect(DB_PATH, check_same_thread=False)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()

cursor.execute(
    """
    CREATE TABLE IF NOT EXISTS inference_tasks (
        task_id TEXT PRIMARY KEY,
        created_at TEXT NOT NULL,
        filename TEXT,
        s3_key TEXT,
        predictions TEXT,
        gradcam_url TEXT,
        duration REAL,
        status TEXT NOT NULL
    )
    """
)
connection.commit()


def save_inference_task(
    task_id: str,
    filename: str,
    s3_key: str,
    predictions: dict,
    gradcam_url: str,
    duration: float,
    status: str,
):
    try:
        cursor.execute(
            """
            INSERT OR REPLACE INTO inference_tasks
            (task_id, created_at, filename, s3_key, predictions, gradcam_url, duration, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                datetime.utcnow().isoformat(timespec='seconds'),
                filename,
                s3_key,
                json.dumps(predictions),
                gradcam_url,
                duration,
                status,
            ),
        )
        connection.commit()
    except Exception as exc:
        logger.error(
            f"Failed to save inference metadata for task {task_id}: {exc}")


def get_inference_task(task_id: str):
    row = cursor.execute(
        "SELECT * FROM inference_tasks WHERE task_id = ?",
        (task_id,),
    ).fetchone()
    if not row:
        return None

    return {
        "task_id": row["task_id"],
        "created_at": row["created_at"],
        "filename": row["filename"],
        "s3_key": row["s3_key"],
        "predictions": json.loads(row["predictions"]) if row["predictions"] else None,
        "gradcam_url": row["gradcam_url"],
        "duration": row["duration"],
        "status": row["status"],
    }
