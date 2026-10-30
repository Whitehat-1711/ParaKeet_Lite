from __future__ import annotations

import math
import struct
import threading
import time
from collections.abc import Callable


def calculate_audio_level(data: bytes) -> float:
    if not data:
        return 0.0
    sample_count = len(data) // 2
    if not sample_count:
        return 0.0
    samples = struct.unpack(f"<{sample_count}h", data[: sample_count * 2])
    rms = math.sqrt(sum(sample * sample for sample in samples) / sample_count)
    return min(rms / 32768.0, 1.0)


class AudioRecorder:
    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        max_duration: int = 60,
        level_callback: Callable[[float], None] | None = None,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.max_duration = max_duration
        self.level_callback = level_callback
        self._recording = False
        self._started_at = 0.0
        self._frames: list[bytes] = []
        self._stream = None
        self._audio = None
        self._backend = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start_recording(self) -> bool:
        with self._lock:
            if self._recording:
                return False
            try:
                import pyaudio

                self._audio = pyaudio.PyAudio()
                self._stream = self._audio.open(
                    format=pyaudio.paInt16,
                    channels=self.channels,
                    rate=self.sample_rate,
                    input=True,
                    frames_per_buffer=self.chunk_size,
                )
                self._backend = "pyaudio"
            except ImportError:
                try:
                    import sounddevice as sd
                except ImportError as exc:
                    raise RuntimeError(
                        "No audio backend is installed. Install the audio extra."
                    ) from exc
                self._frames = []

                def callback(indata, frame_count, timing, status):
                    if status:
                        return
                    data = bytes(indata.astype("<i2").tobytes())
                    with self._lock:
                        if self._recording:
                            self._frames.append(data)
                    if self.level_callback:
                        self.level_callback(calculate_audio_level(data))

                self._stream = sd.InputStream(
                    samplerate=self.sample_rate,
                    channels=self.channels,
                    dtype="int16",
                    blocksize=self.chunk_size,
                    callback=callback,
                )
                self._stream.start()
                self._backend = "sounddevice"
            except Exception:
                if self._audio is not None:
                    self._audio.terminate()
                    self._audio = None
                raise
            self._frames = []
            self._started_at = time.monotonic()
            self._recording = True
            if self._backend == "pyaudio":
                self._thread = threading.Thread(target=self._capture, daemon=True)
                self._thread.start()
            return True

    def _capture(self) -> None:
        while self.is_recording():
            if self.get_elapsed_time() >= self.max_duration:
                self.stop_recording()
                return
            try:
                data = self._stream.read(self.chunk_size, exception_on_overflow=False)
            except Exception:
                self.stop_recording()
                return
            with self._lock:
                self._frames.append(data)
            if self.level_callback:
                self.level_callback(calculate_audio_level(data))

    def stop_recording(self) -> bytes | None:
        with self._lock:
            if not self._recording:
                return None
            self._recording = False
            stream, audio = self._stream, self._audio
            self._stream = self._audio = None
            data = b"".join(self._frames)
        if stream is not None and self._backend == "pyaudio":
            stream.stop_stream()
            stream.close()
        elif stream is not None:
            stream.stop()
            stream.close()
        if audio is not None:
            audio.terminate()
        self._backend = None
        return data

    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def get_elapsed_time(self) -> float:
        return time.monotonic() - self._started_at if self.is_recording() else 0.0
