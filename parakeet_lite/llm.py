from __future__ import annotations

import json
import time
from collections.abc import Generator
from typing import Any

import requests


class LLMError(RuntimeError):
    """Raised when Ollama cannot satisfy a request."""


class LLMClient:
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:3b",
        timeout: float = 45.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self.parameters: dict[str, Any] = {}

    def check_connectivity(self) -> bool:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            response.raise_for_status()
            return True
        except requests.RequestException:
            return False

    def get_available_models(self) -> list[str]:
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=3)
            response.raise_for_status()
            data = response.json()
        except (requests.RequestException, ValueError) as exc:
            raise LLMError(f"Unable to list Ollama models: {exc}") from exc
        return [str(item["name"]) for item in data.get("models", []) if "name" in item]

    def set_model_parameters(self, **kwargs: Any) -> None:
        self.parameters.update(kwargs)

    def generate_response(
        self, prompt: str, model: str | None = None, stream: bool = True
    ) -> Generator[str, None, None]:
        if not prompt.strip():
            raise LLMError("Cannot generate an answer for an empty prompt")
        payload = {
            "model": model or self.model,
            "prompt": prompt,
            "stream": stream,
            "options": self.parameters,
        }
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                stream=stream,
                timeout=self.timeout,
            )
            response.raise_for_status()
        except requests.RequestException as exc:
            raise LLMError(f"Ollama request failed: {exc}") from exc
        if not stream:
            try:
                yield str(response.json().get("response", ""))
            except ValueError as exc:
                raise LLMError(f"Invalid Ollama response: {exc}") from exc
            return
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise LLMError(f"Invalid Ollama streaming response: {exc}") from exc
                token = data.get("response", "")
                if token:
                    yield str(token)
                if data.get("done"):
                    return
        finally:
            response.close()

    def timed_response(self, prompt: str) -> tuple[str, float]:
        started = time.monotonic()
        response = "".join(self.generate_response(prompt))
        return response, time.monotonic() - started
