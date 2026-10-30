from types import SimpleNamespace

from parakeet_lite.llm import LLMClient
from parakeet_lite.web_search import (
    DuckDuckGoSearch,
    SearchDiscriminator,
    SearchResult,
    format_search_context,
    needs_web_search,
)


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


def test_web_search_decision_and_context_formatting():
    discriminator = SearchDiscriminator()
    assert discriminator.decide("What is the latest LangChain release?").should_search
    assert not discriminator.decide("Explain dependency injection").should_search
    assert discriminator.decide("Search the web for current LangChain docs").confidence > 0.95
    assert "local model" in discriminator.decide("How does LangChain work?").reason
    context = format_search_context([SearchResult("Docs", "https://example.com", "A result")])
    assert "[1] Docs" in context
    assert "https://example.com" in context


def test_duckduckgo_result_parsing(monkeypatch):
    class Response:
        text = """
        <a class="result__a" href="https://example.com">LangChain docs</a>
        <a class="result__snippet">Official documentation</a>
        """

        def raise_for_status(self):
            pass

    monkeypatch.setattr("parakeet_lite.web_search.requests.get", lambda *args, **kwargs: Response())
    results = DuckDuckGoSearch().search("LangChain docs")
    assert results == [SearchResult("LangChain docs", "https://example.com", "Official documentation")]


def test_search_falls_back_to_bing(monkeypatch):
    class Response:
        def __init__(self, text):
            self.text = text

        def raise_for_status(self):
            pass

    calls = []

    def fake_get(url, **kwargs):
        calls.append(url)
        if "duckduckgo" in url:
            return Response("<html>challenge</html>")
        return Response(
            '<li class="b_algo"><h2><a href="https://example.com">Result</a></h2>'
            '<div class="b_caption"><p>Snippet</p></div></li>'
        )

    monkeypatch.setattr("parakeet_lite.web_search.requests.get", fake_get)
    results = DuckDuckGoSearch().search("latest AI news")
    assert results == [SearchResult("Result", "https://example.com", "Snippet")]
    assert len(calls) == 2
