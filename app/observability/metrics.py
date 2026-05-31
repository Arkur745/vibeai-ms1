from prometheus_client import Counter, Gauge, Histogram

# Inference counters
INFERENCE_REQUESTS = Counter(
    "vibeai_inference_requests_total",
    "Total number of inference requests submitted to VibeAI",
)

INFERENCE_FAILURES = Counter(
    "vibeai_inference_failures_total",
    "Total number of failed inference tasks",
)

INFERENCE_DURATION = Histogram(
    "vibeai_inference_duration_seconds",
    "Time spent submitting inference requests and recording metadata",
)

# Timing histograms
AUDIO_DOWNLOAD_SECONDS = Histogram(
    "vibeai_audio_download_seconds",
    "Time spent downloading audio from S3",
)

S3_UPLOAD_SECONDS = Histogram(
    "vibeai_s3_upload_seconds",
    "Time spent uploading artifacts to S3",
)

SPECTROGRAM_GENERATION_SECONDS = Histogram(
    "vibeai_spectrogram_generation_seconds",
    "Time spent generating spectrogram images",
)

MODEL_INFERENCE_SECONDS = Histogram(
    "vibeai_model_inference_seconds",
    "Time spent running model inference",
)

GRADCAM_GENERATION_SECONDS = Histogram(
    "vibeai_gradcam_generation_seconds",
    "Time spent generating GradCAM visualizations",
)

TOTAL_TASK_DURATION_SECONDS = Histogram(
    "vibeai_total_task_duration_seconds",
    "Total time spent processing a task from download to upload",
)

# GPU gauges
GPU_CUDA_AVAILABLE = Gauge(
    "vibeai_gpu_cuda_available",
    "CUDA availability for VibeAI inference",
)

GPU_MEMORY_ALLOCATED_MB = Gauge(
    "vibeai_gpu_memory_allocated_mb",
    "GPU memory allocated by the process in megabytes",
)

GPU_MEMORY_RESERVED_MB = Gauge(
    "vibeai_gpu_memory_reserved_mb",
    "GPU memory reserved by the process in megabytes",
)

GPU_MEMORY_TOTAL_MB = Gauge(
    "vibeai_gpu_memory_total_mb",
    "Total GPU memory available on the current device in megabytes",
)
