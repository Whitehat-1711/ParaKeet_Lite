from parakeet_lite.models import AppConfig, AppState, GPUInfo, TranscriptionResult
from parakeet_lite.orchestrator import Orchestrator


def test_shutdown_requires_confirmation_when_recording(monkeypatch):
    orchestrator = Orchestrator(AppConfig())
    monkeypatch.setattr(orchestrator.recorder, "is_recording", lambda: True)
    assert not orchestrator.shutdown()
    assert orchestrator.shutdown(force=True)


def test_status_starts_idle(monkeypatch):
    monkeypatch.setattr("parakeet_lite.orchestrator.ModelManager", lambda: type(
        "Manager", (), {"gpu": GPUInfo(), "unload_whisper_model": lambda self: None}
    )())
    orchestrator = Orchestrator(AppConfig())
    assert orchestrator.get_status().state == AppState.IDLE
