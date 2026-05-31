from celery import Celery
from kombu import Exchange, Queue
from prometheus_client import start_http_server

from app.core.config import settings
from app.core.logger import logger


celery = Celery(
    "vibeai_worker",
    broker=settings.redis_url,
    backend=settings.celery_result_backend,
)

celery.conf.update(
    task_track_started=True,
    task_time_limit=3600,
    task_soft_time_limit=3300,
    result_expires=3600,
    result_extended=True,
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_default_queue="default",
    task_default_exchange="tasks",
    task_default_routing_key="task.default",
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    worker_cancel_long_running_tasks_on_connection_loss=True,
)

celery.conf.include = [
    "app.workers.tasks",
]

celery.conf.task_queues = (
    Queue(
        "default",
        Exchange("tasks", type="direct"),
        routing_key="task.default",
    ),
)


if settings.worker_metrics_port:
    try:
        start_http_server(settings.worker_metrics_port)
        logger.info(
            f"Started Celery worker metrics server on port {settings.worker_metrics_port}")
    except Exception as exc:
        logger.warning(f"Unable to start worker metrics server: {exc}")


@celery.task(bind=True)
def debug_task(self):
    """Debug task for testing Celery connectivity"""
    logger.info(f"Debug task request: {self.request!r}")
