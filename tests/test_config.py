import json

import pytest

from parakeet_lite.config import ConfigError, ConfigManager
from parakeet_lite.models import AppConfig


def test_config_round_trip(tmp_path):
    path = tmp_path / "config.json"
    manager = ConfigManager(path)
    original = AppConfig()
    manager.save_config(original)
    assert manager.load_config() == original
    assert json.loads(path.read_text())["version"] == "1.0.0"


def test_invalid_config_has_descriptive_error(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"audio": {"sample_rate": 100}}')
    with pytest.raises(ConfigError, match="sample_rate"):
        ConfigManager(path).load_config()


def test_unknown_section_field_is_rejected(tmp_path):
    path = tmp_path / "config.json"
    path.write_text('{"llm": {"unexpected": true}}')
    with pytest.raises(ConfigError, match="unknown fields"):
        ConfigManager(path).load_config()
