from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """Load project configuration and resolve all project-relative paths."""
    config_path = Path(path) if path else PROJECT_ROOT / "config.yaml"
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    for key, value in config["paths"].items():
        path_value = Path(value)
        config["paths"][key] = path_value if path_value.is_absolute() else PROJECT_ROOT / path_value
    return config


def ensure_output_directories(config: dict[str, Any]) -> None:
    """Create all parent/output directories used by the pipeline."""
    for key, path in config["paths"].items():
        if key.endswith("_dir"):
            path.mkdir(parents=True, exist_ok=True)
        else:
            path.parent.mkdir(parents=True, exist_ok=True)
