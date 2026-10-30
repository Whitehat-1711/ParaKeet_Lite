from __future__ import annotations

from dataclasses import dataclass
from html.parser import HTMLParser
import base64
from urllib.parse import parse_qs, unquote, urlparse

import requests


class WebSearchError(RuntimeError):
    """Raised when the web search provider cannot return results."""


@dataclass(frozen=True)
class SearchResult:
    title: str
    url: str
    snippet: str


@dataclass(frozen=True)
class SearchDecision:
    should_search: bool
    confidence: float
    reason: str


class SearchDiscriminator:
    """Classifies whether a prompt needs current external information."""

    _explicit = (
        "search the web", "search online", "web search", "look it up",
        "look up", "browse the web", "find online", "check online",
    )
    _freshness = (
        "latest", "current", "today", "tonight", "now", "recent", "recently",
        "this week", "this month", "this year", "upcoming", "breaking news",
        "as of", "updated",
    )
    _lookup = (
        "news", "weather", "stock price", "share price", "exchange rate",
        "price", "release date", "release notes", "current version",
        "documentation", "official docs", "who is the current",
        "where can i", "when is",
    )
    _conceptual = (
        "explain", "teach me", "how do i", "how does", "what is",
        "what are", "difference between", "give me an example",
        "write code", "debug", "summarize this",
    )

    def decide(self, query: str) -> SearchDecision:
        text = " ".join(query.lower().split())
        if not text:
            return SearchDecision(False, 1.0, "empty prompt")
        if any(marker in text for marker in self._explicit):
            return SearchDecision(True, 0.99, "explicit web-search request")
        freshness = any(marker in text for marker in self._freshness)
        lookup = any(marker in text for marker in self._lookup)
        conceptual = any(marker in text for marker in self._conceptual)
        if freshness and (lookup or len(text.split()) >= 4):
            return SearchDecision(True, 0.95, "prompt asks for current or recent information")
        if lookup and not conceptual:
            return SearchDecision(True, 0.85, "prompt asks for an external factual lookup")
        return SearchDecision(False, 0.9, "prompt can be answered from local model knowledge")


class _DuckDuckGoParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._url = ""
        self._title = ""
        self._snippet = ""
        self._in_title = False
        self._in_snippet = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = attributes.get("class", "") or ""
        if tag == "a" and "result__a" in classes:
            self._url = self._clean_url(attributes.get("href", "") or "")
            self._title = ""
            self._in_title = True
        elif "result__snippet" in classes:
            self._snippet = ""
            self._in_snippet = True

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title += data
        if self._in_snippet:
            self._snippet += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_title:
            self._in_title = False
        if self._in_snippet and tag in {"a", "div"}:
            self._in_snippet = False
            if self._url and self._title:
                self.results.append(
                    SearchResult(
                        " ".join(self._title.split()),
                        self._url,
                        " ".join(self._snippet.split()),
                    )
                )
                self._url = self._title = self._snippet = ""

    @staticmethod
    def _clean_url(url: str) -> str:
        parsed = urlparse(url)
        if "duckduckgo.com" in parsed.netloc and parsed.path == "/l/":
            return parse_qs(parsed.query).get("uddg", [url])[0]
        return url


class _BingParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.results: list[SearchResult] = []
        self._url = ""
        self._title = ""
        self._snippet = ""
        self._in_title = False
        self._in_snippet = False
        self._in_heading = False

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        classes = attributes.get("class", "") or ""
        if tag == "li" and "b_algo" in classes:
            self._url = self._title = self._snippet = ""
        elif tag == "h2":
            self._in_heading = True
        elif tag == "a" and self._in_heading and not self._url:
            href = attributes.get("href", "") or ""
            if href.startswith(("http://", "https://")):
                self._url = self._clean_url(href)
                self._in_title = True
        elif self._url and "b_caption" in classes:
            self._in_snippet = True

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._title += data
        elif self._in_snippet:
            self._snippet += data

    def handle_endtag(self, tag: str) -> None:
        if tag == "h2":
            self._in_heading = False
        elif tag == "a" and self._in_title:
            self._in_title = False
        elif self._in_snippet and tag in {"p", "div"}:
            self._in_snippet = False
            if self._url and self._title:
                self.results.append(
                    SearchResult(
                        " ".join(self._title.split()),
                        self._url,
                        " ".join(self._snippet.split()),
                    )
                )
                self._url = self._title = self._snippet = ""

    @staticmethod
    def _clean_url(url: str) -> str:
        parsed = urlparse(unquote(url))
        encoded = parse_qs(parsed.query).get("u", [""])[0]
        if encoded.startswith("a1"):
            try:
                return base64.urlsafe_b64decode(encoded[2:] + "==").decode("utf-8")
            except (ValueError, UnicodeDecodeError):
                pass
        return url


class DuckDuckGoSearch:
    def __init__(self, max_results: int = 5, timeout: float = 10.0) -> None:
        self.max_results = max(1, max_results)
        self.timeout = timeout

    def search(self, query: str) -> list[SearchResult]:
        if not query.strip():
            return []
        headers = {"User-Agent": "Mozilla/5.0 (compatible; Parakeet-Lite/1.0)"}
        try:
            response = requests.get(
                "https://html.duckduckgo.com/html/",
                params={"q": query},
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
            parser = _DuckDuckGoParser()
            parser.feed(response.text)
            if parser.results:
                return parser.results[: self.max_results]
        except requests.RequestException:
            pass

        try:
            response = requests.get(
                "https://www.bing.com/search",
                params={"q": query, "count": self.max_results},
                headers=headers,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise WebSearchError(f"Web search failed for DuckDuckGo and Bing: {exc}") from exc
        parser = _BingParser()
        parser.feed(response.text)
        if not parser.results:
            raise WebSearchError("Web search returned no usable results")
        return parser.results[: self.max_results]


def needs_web_search(query: str) -> bool:
    return SearchDiscriminator().decide(query).should_search


def format_search_context(results: list[SearchResult]) -> str:
    if not results:
        return ""
    lines = ["Web search results (use these as sources; mention uncertainty):"]
    for index, result in enumerate(results, start=1):
        lines.append(f"[{index}] {result.title}\nURL: {result.url}\n{result.snippet}")
    return "\n\n".join(lines)
