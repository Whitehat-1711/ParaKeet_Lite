from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path
from typing import Any, TypeVar

from .models import (
    AppConfig,
    AudioConfig,
    HotkeyConfig,
    LLMConfig,
    TranscriptionConfig,
    UIConfig,
)

T = TypeVar("T")


class ConfigError(ValueError):
    """Raised when a configuration file is invalid."""


def _section(cls: type[T], data: dict[str, Any], name: str) -> T:
    allowed = {item.name for item in fields(cls)}
    unknown = set(data) - allowed
    if unknown:
        raise ConfigError(f"{name} contains unknown fields: {', '.join(sorted(unknown))}")
    if cls is UIConfig and "window_size" in data:
        data = dict(data)
        value = data["window_size"]
        if not isinstance(value, (list, tuple)) or len(value) != 2:
            raise ConfigError("ui.window_size must contain exactly two values")
        data["window_size"] = tuple(int(item) for item in value)
    try:
        return cls(**data)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"Invalid {name}: {exc}") from exc


class ConfigManager:
    def __init__(self, config_path: str | Path):
        self.path = Path(config_path)

    @staticmethod
    def get_default_config() -> AppConfig:
        return AppConfig()

    def load_config(self) -> AppConfig:
        if not self.path.exists():
            config = self.get_default_config()
            self.save_config(config)
            return config
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigError(f"Unable to read configuration {self.path}: {exc}") from exc
        if not isinstance(data, dict):
            raise ConfigError("Configuration root must be a JSON object")
        try:
            config = AppConfig(
                audio=_section(AudioConfig, data.get("audio", {}), "audio"),
                transcription=_section(
                    TranscriptionConfig, data.get("transcription", {}), "transcription"
                ),
                llm=_section(LLMConfig, data.get("llm", {}), "llm"),
                ui=_section(UIConfig, data.get("ui", {}), "ui"),
                hotkeys=_section(HotkeyConfig, data.get("hotkeys", {}), "hotkeys"),
                version=data.get("version", "1.0.0"),
            )
        except (TypeError, ConfigError) as exc:
            raise ConfigError(str(exc)) from exc
        errors = self.validate_config(config)
        if errors:
            raise ConfigError("; ".join(errors))
        return config

    def save_config(self, config: AppConfig) -> None:
        errors = self.validate_config(config)
        if errors:
            raise ConfigError("; ".join(errors))
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(config.to_dict(), indent=2) + "\n", encoding="utf-8")

    @staticmethod
    def validate_config(config: AppConfig) -> list[str]:
        errors: list[str] = []
        if not 8000 <= config.audio.sample_rate <= 48000:
            errors.append("audio.sample_rate must be between 8000 and 48000")
        if config.audio.channels not in (1, 2):
            errors.append("audio.channels must be 1 or 2")
        if not 10 <= config.audio.max_duration <= 300:
            errors.append("audio.max_duration must be between 10 and 300 seconds")
        if config.transcription.language != "en":
            errors.append("transcription.language must be 'en'")
        if not config.llm.base_url.startswith(("http://", "https://")):
            errors.append("llm.base_url must be an HTTP(S) URL")
        if config.ui.update_frequency <= 0:
            errors.append("ui.update_frequency must be positive")
        return errors
