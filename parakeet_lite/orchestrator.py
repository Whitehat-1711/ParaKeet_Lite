from __future__ import annotations

import threading
from collections.abc import Callable, Generator
from pathlib import Path

from .audio import AudioRecorder
from .config import ConfigManager
from .llm import LLMClient
from .model_manager import ModelManager
from .models import AppConfig, AppState, SystemStatus, TranscriptionResult
from .transcriber import Transcriber


class Orchestrator:
    def __init__(
        self,
        config: AppConfig,
        level_callback: Callable[[float], None] | None = None,
    ) -> None:
        self.config = config
        self.model_manager = ModelManager()
        self.llm = LLMClient(
            config.llm.base_url, config.llm.default_model, config.llm.timeout
        )
        self.transcriber = Transcriber(self.model_manager, config.transcription.model_name)
        self.recorder = AudioRecorder(
            config.audio.sample_rate,
            config.audio.channels,
            config.audio.chunk_size,
            config.audio.max_duration,
            level_callback,
        )
        self.status = SystemStatus(gpu=self.model_manager.gpu)
        self._lock = threading.Lock()

    def initialize(self) -> bool:
        connected = self.llm.check_connectivity()
        with self._lock:
            self.status.ollama_connected = connected
            self.status.message = "Ollama connected" if connected else "Ollama is not running"
        return connected

    def start_recording(self) -> bool:
        try:
            result = self.recorder.start_recording()
        except RuntimeError as exc:
            self._set_error(str(exc))
            raise
        with self._lock:
            self.status.state = AppState.RECORDING
            self.status.message = "Recording..."
        return result

    def stop_recording(self) -> bytes | None:
        data = self.recorder.stop_recording()
        with self._lock:
            self.status.state = AppState.IDLE
            self.status.message = "Recording stopped"
        return data

    def transcribe_audio(self, audio_data: bytes) -> TranscriptionResult:
        with self._lock:
            self.status.state = AppState.TRANSCRIBING
            self.status.message = "Transcribing..."
        result = self.transcriber.transcribe_audio(audio_data, self.config.audio.sample_rate)
        with self._lock:
            self.status.state = AppState.ERROR if result.error else AppState.IDLE
            self.status.message = result.error or "Transcription complete"
        return result

    def generate_response(self, transcription: str) -> Generator[str, None, None]:
        with self._lock:
            self.status.state = AppState.GENERATING
            self.status.message = "Generating answer..."
        try:
            self.llm.set_model_parameters(
                temperature=self.config.llm.temperature,
                top_p=self.config.llm.top_p,
                num_predict=self.config.llm.max_tokens,
            )
            yield from self.llm.generate_response(
                transcription, self.config.llm.default_model, self.config.llm.stream
            )
        except Exception as exc:
            self._set_error(str(exc))
            raise
        else:
            with self._lock:
                self.status.state = AppState.IDLE
                self.status.message = "Answer complete"

    def get_status(self) -> SystemStatus:
        with self._lock:
            return SystemStatus(
                self.status.state,
                self.status.gpu,
                self.status.ollama_connected,
                self.status.message,
                list(self.status.active_operations),
            )

    def shutdown(self, force: bool = False) -> bool:
        if self.recorder.is_recording():
            if not force:
                return False
            self.recorder.stop_recording()
        self.transcriber.unload_model()
        with self._lock:
            self.status.state = AppState.IDLE
            self.status.message = "Shutdown complete"
        return True

    def _set_error(self, message: str) -> None:
        with self._lock:
            self.status.state = AppState.ERROR
            self.status.message = message


def load_or_create_config(path: str | Path = "config.json") -> AppConfig:
    return ConfigManager(path).load_config()
