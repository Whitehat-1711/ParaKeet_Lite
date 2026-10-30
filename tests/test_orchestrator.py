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


def test_improve_transcription_uses_interview_context(monkeypatch):
    class FakeLLM:
        def __init__(self, *args):
            self.prompt = ""

        def set_model_parameters(self, **kwargs):
            pass

        def generate_response(self, prompt, model, stream):
            self.prompt = prompt
            yield "corrected sentence"

    monkeypatch.setattr("parakeet_lite.orchestrator.LLMClient", FakeLLM)
    monkeypatch.setattr("parakeet_lite.orchestrator.ModelManager", lambda: type(
        "Manager", (), {"gpu": GPUInfo(), "unload_whisper_model": lambda self: None}
    )())
    orchestrator = Orchestrator(AppConfig())
    result = orchestrator.improve_transcription("aws quesion", "Backend Developer", "Technology")
    assert result == "corrected sentence"
    assert "Backend Developer" in orchestrator.llm.prompt
    assert "Interview domain: Technology" in orchestrator.llm.prompt
    assert "LangChain" in orchestrator.llm.prompt
    assert "LangGraph" in orchestrator.llm.prompt
    assert "LangSmith" in orchestrator.llm.prompt
    assert "Hugging Face" in orchestrator.llm.prompt


def test_improve_transcription_uses_non_technical_domain_terminology(monkeypatch):
    class FakeLLM:
        def __init__(self, *args):
            self.prompt = ""

        def set_model_parameters(self, **kwargs):
            pass

        def generate_response(self, prompt, model, stream):
            self.prompt = prompt
            yield "corrected sentence"

    monkeypatch.setattr("parakeet_lite.orchestrator.LLMClient", FakeLLM)
    monkeypatch.setattr("parakeet_lite.orchestrator.ModelManager", lambda: type(
        "Manager", (), {"gpu": GPUInfo(), "unload_whisper_model": lambda self: None}
    )())
    orchestrator = Orchestrator(AppConfig())
    orchestrator.improve_transcription("we reviewed the kpi", "Project Manager", "Management")
    assert "stakeholder" in orchestrator.llm.prompt
    assert "roadmap" in orchestrator.llm.prompt
    assert "Interview domain: Management" in orchestrator.llm.prompt
