from types import SimpleNamespace

from parakeet_lite.llm import LLMClient


class FakeResponse:
    def __init__(self, lines):
        self.lines = lines

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        return self.lines

    def close(self):
        return None


def test_streaming_response_parsing(monkeypatch):
    response = FakeResponse([
        '{"response":"Hello"}',
        '{"response":" world","done":true}',
    ])
    monkeypatch.setattr(
        "parakeet_lite.llm.requests.post",
        lambda *args, **kwargs: response,
    )
    assert "".join(LLMClient().generate_response("question")) == "Hello world"


def test_connectivity(monkeypatch):
    fake = SimpleNamespace(raise_for_status=lambda: None)
    monkeypatch.setattr("parakeet_lite.llm.requests.get", lambda *args, **kwargs: fake)
    assert LLMClient().check_connectivity()
