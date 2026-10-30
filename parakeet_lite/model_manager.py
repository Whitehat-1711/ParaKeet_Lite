from __future__ import annotations

import gc
import shutil
from typing import Any

from .models import GPUInfo


class ModelManager:
    def __init__(self) -> None:
        self.gpu = self.detect_gpu()
        self._whisper_model: Any = None
        self._whisper_name: str | None = None

    @staticmethod
    def detect_gpu() -> GPUInfo:
        try:
            import torch
        except ImportError:
            return GPUInfo()
        if not torch.cuda.is_available():
            return GPUInfo()
        properties = torch.cuda.get_device_properties(0)
        free, total = torch.cuda.mem_get_info(0)
        return GPUInfo(
            available=True,
            name=torch.cuda.get_device_name(0),
            memory_total=int(properties.total_memory / 1024 / 1024),
            memory_free=int(free / 1024 / 1024),
            cuda_version=str(torch.version.cuda or ""),
        )

    def load_whisper_model(self, model_name: str) -> Any:
        if self._whisper_model is not None and self._whisper_name == model_name:
            return self._whisper_model
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "faster-whisper is not installed. Install the transcription extra."
            ) from exc
        device = "cuda" if self.gpu.available else "cpu"
        compute_type = "float16" if self.gpu.available else "int8"
        self.unload_whisper_model()
        try:
            self._whisper_model = WhisperModel(model_name, device=device, compute_type=compute_type)
        except Exception as exc:
            raise RuntimeError(f"Unable to load Whisper model '{model_name}': {exc}") from exc
        self._whisper_name = model_name
        return self._whisper_model

    def unload_whisper_model(self) -> None:
        self._whisper_model = None
        self._whisper_name = None
        gc.collect()
        if self.gpu.available:
            try:
                import torch

                torch.cuda.empty_cache()
            except ImportError:
                pass

    def get_gpu_status(self) -> dict[str, Any]:
        return {
            "available": self.gpu.available,
            "name": self.gpu.name,
            "memory_total": self.gpu.memory_total,
            "memory_free": self.gpu.memory_free,
            "memory_used": max(self.gpu.memory_total - self.gpu.memory_free, 0),
            "cuda_version": self.gpu.cuda_version,
        }

    @staticmethod
    def get_memory_usage() -> dict[str, int]:
        usage = shutil.disk_usage(".")
        return {"disk_free": usage.free, "disk_total": usage.total}
