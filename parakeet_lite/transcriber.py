from __future__ import annotations

import time

from .model_manager import ModelManager
from .models import TranscriptionResult


class Transcriber:
    def __init__(self, model_manager: ModelManager, model_name: str = "base.en") -> None:
        self.model_manager = model_manager
        self.model_name = model_name

    def transcribe_audio(self, audio_data: bytes, sample_rate: int = 16000) -> TranscriptionResult:
        started = time.monotonic()
        if not audio_data:
            return TranscriptionResult("", duration=0.0, error="No audio was recorded")
        try:
            model = self.model_manager.load_whisper_model(self.model_name)
            audio = self._pcm_to_float32(audio_data)
            segments, _ = model.transcribe(
                audio, language="en", beam_size=5, vad_filter=True, temperature=0.0
            )
            parts: list[str] = []
            scores: list[float] = []
            for segment in segments:
                text = segment.text.strip()
                if text:
                    parts.append(text)
                    scores.extend([float(getattr(segment, "avg_logprob", 0.0) + 1.0)] * len(text.split()))
            return TranscriptionResult(
                " ".join(parts).strip(),
                [max(0.0, min(1.0, score)) for score in scores],
                time.monotonic() - started,
                "en",
            )
        except Exception as exc:
            return TranscriptionResult(
                "", duration=time.monotonic() - started, language="en", error=str(exc)
            )

    @staticmethod
    def _pcm_to_float32(audio_data: bytes):
        try:
            import numpy as np
        except ImportError as exc:
            raise RuntimeError(
                "NumPy is required for transcription. Install the transcription extra."
            ) from exc
        sample_count = len(audio_data) // 2
        samples = np.frombuffer(audio_data[: sample_count * 2], dtype="<i2")
        return samples.astype(np.float32) / 32768.0

    def unload_model(self) -> None:
        self.model_manager.unload_whisper_model()

    @staticmethod
    def get_available_models() -> list[str]:
        return ["tiny.en", "base.en", "small.en", "medium.en"]
