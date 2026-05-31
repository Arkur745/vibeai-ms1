from typing import Dict

import torch

from app.observability.metrics import (
    GPU_CUDA_AVAILABLE,
    GPU_MEMORY_ALLOCATED_MB,
    GPU_MEMORY_RESERVED_MB,
    GPU_MEMORY_TOTAL_MB,
)


def collect_gpu_status() -> Dict[str, object]:
    available = torch.cuda.is_available()
    gpu_status = {
        "available": available,
        "name": None,
        "memory_allocated_mb": 0.0,
        "memory_reserved_mb": 0.0,
        "memory_total_mb": 0.0,
    }

    if not available:
        return gpu_status

    try:
        device_index = torch.cuda.current_device()
        gpu_status["name"] = torch.cuda.get_device_name(device_index)
        gpu_status["memory_allocated_mb"] = torch.cuda.memory_allocated(
            device_index) / 1024 ** 2
        gpu_status["memory_reserved_mb"] = torch.cuda.memory_reserved(
            device_index) / 1024 ** 2
        gpu_status["memory_total_mb"] = torch.cuda.get_device_properties(
            device_index).total_memory / 1024 ** 2
    except Exception:
        gpu_status["available"] = False

    return gpu_status


def update_gpu_metrics() -> Dict[str, object]:
    status = collect_gpu_status()
    GPU_CUDA_AVAILABLE.set(1.0 if status["available"] else 0.0)
    GPU_MEMORY_ALLOCATED_MB.set(status["memory_allocated_mb"])
    GPU_MEMORY_RESERVED_MB.set(status["memory_reserved_mb"])
    GPU_MEMORY_TOTAL_MB.set(status["memory_total_mb"])
    return status
