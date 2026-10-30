from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class AppState(str, Enum):
    IDLE = "idle"
    RECORDING = "recording"
    TRANSCRIBING = "transcribing"
    GENERATING = "generating"
    ERROR = "error"


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    channels: int = 1
    chunk_size: int = 1024
    max_duration: int = 60
    noise_threshold: float = 20.0


@dataclass
class TranscriptionConfig:
    model_name: str = "base.en"
    language: str = "en"
    compute_type: str = "float16"
    beam_size: int = 5
    best_of: int = 5
    temperature: float = 0.0


@dataclass
class LLMConfig:
    base_url: str = "http://localhost:11434"
    default_model: str = "qwen2.5:3b"
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    stream: bool = True
    timeout: float = 180.0


@dataclass
class UIConfig:
    window_title: str = "Parakeet Lite Desktop Assistant"
    window_size: tuple[int, int] = (800, 600)
    update_frequency: int = 100
    show_gpu_badge: bool = True


@dataclass
class HotkeyConfig:
    record: str = "<Control-r>"
    stop: str = "<Control-s>"
    answer: str = "<Control-a>"
    enabled: bool = True


@dataclass
class AppConfig:
    audio: AudioConfig = field(default_factory=AudioConfig)
    transcription: TranscriptionConfig = field(default_factory=TranscriptionConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    ui: UIConfig = field(default_factory=UIConfig)
    hotkeys: HotkeyConfig = field(default_factory=HotkeyConfig)
    version: str = "1.0.0"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class GPUInfo:
    available: bool = False
    name: str = "CPU"
    memory_total: int = 0
    memory_free: int = 0
    cuda_version: str = ""


@dataclass
class TranscriptionResult:
    text: str
    confidence_scores: list[float] = field(default_factory=list)
    duration: float = 0.0
    language: str = "en"
    error: str | None = None


@dataclass
class SystemStatus:
    state: AppState = AppState.IDLE
    gpu: GPUInfo = field(default_factory=GPUInfo)
    ollama_connected: bool = False
    message: str = "Ready"
    active_operations: list[str] = field(default_factory=list)
