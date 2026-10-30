from __future__ import annotations

import argparse
from pathlib import Path

from .config import ConfigError, ConfigManager
from .orchestrator import Orchestrator
from .ui import UIController


def main() -> None:
    parser = argparse.ArgumentParser(description="Parakeet Lite local desktop assistant")
    parser.add_argument("--config", type=Path, default=Path("config.json"))
    args = parser.parse_args()
    try:
        config = ConfigManager(args.config).load_config()
    except ConfigError as exc:
        raise SystemExit(f"Configuration error: {exc}") from exc
    orchestrator = Orchestrator(config)
    ui = UIController(orchestrator)
    connected = orchestrator.initialize()
    if not connected:
        ui.show_error(
            "Ollama service unavailable",
            "Start Ollama at http://localhost:11434, then use Suggest Answer when it is ready.",
        )
    ui.run()


if __name__ == "__main__":
    main()
