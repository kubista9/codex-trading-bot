from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file into a dictionary."""
    with Path(path).open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        msg = f"Expected YAML mapping at {path}"
        raise ValueError(msg)
    return data


def require_env_name(config: dict[str, Any], key: str) -> str:
    """Return the environment variable name configured under a source block."""
    value = config.get(key)
    if not isinstance(value, str) or not value:
        msg = f"Missing required environment variable name for {key}"
        raise ValueError(msg)
    return value
