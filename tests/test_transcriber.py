import numpy as np

from parakeet_lite.transcriber import Transcriber


class FakeModel:
    def __init__(self):
        self.received = None

    def transcribe(self, audio, **kwargs):
        self.received = audio
        return [], None


class FakeManager:
    def __init__(self):
        self.model = FakeModel()

    def load_whisper_model(self, name):
        return self.model

    def unload_whisper_model(self):
        pass


def test_transcriber_passes_pcm_array_to_whisper():
    manager = FakeManager()
    result = Transcriber(manager).transcribe_audio(b"\x00\x40\x00\xc0")
    assert result.error is None
    assert np.allclose(manager.model.received, [0.5, -0.5])
