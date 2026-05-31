import hashlib
import json
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.core.logger import logger

DB_PATH = Path(settings.database_path)
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

connection = sqlite3.connect(DB_PATH, check_same_thread=False)
connection.row_factory = sqlite3.Row
cursor = connection.cursor()


def _ensure_table_exists() -> None:
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
            status TEXT NOT NULL,
            benchmark TEXT,
            model_version TEXT,
            model_architecture TEXT,
            model_checkpoint_hash TEXT,
            model_training_timestamp TEXT,
            gpu_info TEXT
        )
        """
    )
    connection.commit()

    existing = {row[1] for row in cursor.execute(
        "PRAGMA table_info(inference_tasks)")}
    required_columns = {
        "benchmark",
        "model_version",
        "model_architecture",
        "model_checkpoint_hash",
        "model_training_timestamp",
        "gpu_info",
    }

    for column in required_columns - existing:
        cursor.execute(f"ALTER TABLE inference_tasks ADD COLUMN {column} TEXT")
    connection.commit()


_ensure_table_exists()


def save_inference_task(
    task_id: str,
    filename: str,
    s3_key: str,
    predictions: Dict[str, Any],
    gradcam_url: str,
    duration: float,
    status: str,
    benchmark: Optional[Dict[str, Any]] = None,
    model_version: Optional[str] = None,
    model_architecture: Optional[str] = None,
    model_checkpoint_hash: Optional[str] = None,
    model_training_timestamp: Optional[str] = None,
    gpu_info: Optional[Dict[str, Any]] = None,
) -> None:
    try:
        cursor.execute(
            """
            INSERT OR REPLACE INTO inference_tasks
            (task_id, created_at, filename, s3_key, predictions, gradcam_url, duration, status,
             benchmark, model_version, model_architecture, model_checkpoint_hash, model_training_timestamp, gpu_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                task_id,
                datetime.utcnow().isoformat(timespec="seconds"),
                filename,
                s3_key,
                json.dumps(predictions),
                gradcam_url,
                duration,
                status,
                json.dumps(benchmark or {}),
                model_version,
                model_architecture,
                model_checkpoint_hash,
                model_training_timestamp,
                json.dumps(gpu_info or {}),
            ),
        )
        connection.commit()
    except Exception as exc:
        logger.error(
            f"Failed to save inference metadata for task {task_id}: {exc}")


def get_inference_task(task_id: str) -> Optional[Dict[str, Any]]:
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
        "benchmark": json.loads(row["benchmark"]) if row["benchmark"] else None,
        "model_version": row["model_version"],
        "model_architecture": row["model_architecture"],
        "model_checkpoint_hash": row["model_checkpoint_hash"],
        "model_training_timestamp": row["model_training_timestamp"],
        "gpu_info": json.loads(row["gpu_info"]) if row["gpu_info"] else None,
    }


def list_inference_tasks(
    page: int = 1,
    limit: int = 20,
    sort_desc: bool = True,
) -> Dict[str, Any]:
    offset = max(page - 1, 0) * limit
    order = "DESC" if sort_desc else "ASC"
    total = cursor.execute(
        "SELECT COUNT(*) AS count FROM inference_tasks").fetchone()["count"]
    rows = cursor.execute(
        f"SELECT * FROM inference_tasks ORDER BY created_at {order} LIMIT ? OFFSET ?",
        (limit, offset),
    ).fetchall()
    items = []
    for row in rows:
        items.append(
            {
                "task_id": row["task_id"],
                "created_at": row["created_at"],
                "filename": row["filename"],
                "s3_key": row["s3_key"],
                "predictions": json.loads(row["predictions"]) if row["predictions"] else None,
                "gradcam_url": row["gradcam_url"],
                "duration": row["duration"],
                "status": row["status"],
                "benchmark": json.loads(row["benchmark"]) if row["benchmark"] else None,
                "model_version": row["model_version"],
                "model_architecture": row["model_architecture"],
                "model_checkpoint_hash": row["model_checkpoint_hash"],
                "model_training_timestamp": row["model_training_timestamp"],
                "gpu_info": json.loads(row["gpu_info"]) if row["gpu_info"] else None,
            }
        )
    return {"items": items, "page": page, "limit": limit, "total": total}


def delete_inference_task(task_id: str) -> bool:
    try:
        task = get_inference_task(task_id)
        if not task:
            logger.warning(f"Deletion failed: task {task_id} not found in database.")
            return False

        # Build local GradCAM file path
        gradcam_filename = f"gradcam_{task_id}.png"
        gradcam_path = Path(settings.gradcam_dir) / gradcam_filename
        if gradcam_path.exists():
            try:
                gradcam_path.unlink()
                logger.info(f"Deleted local GradCAM file for task {task_id}: {gradcam_path}")
            except Exception as file_exc:
                logger.error(f"Failed to delete local GradCAM file for task {task_id}: {file_exc}")
        else:
            logger.info(f"No local GradCAM file found for task {task_id} during deletion.")

        cursor.execute("DELETE FROM inference_tasks WHERE task_id = ?", (task_id,))
        connection.commit()
        logger.info(f"Successfully deleted inference task metadata for task {task_id}")
        return True
    except Exception as exc:
        logger.error(f"Failed to execute database deletion for task {task_id}: {exc}")
        return False


def compute_file_hash(path: Path) -> str:
    hash_obj = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8192), b""):
            hash_obj.update(chunk)
    return hash_obj.hexdigest()
