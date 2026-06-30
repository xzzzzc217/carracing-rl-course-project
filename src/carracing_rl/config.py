from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


def deep_update(base: dict, update: dict) -> dict:
    result = deepcopy(base)
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_update(result[key], value)
        else:
            result[key] = deepcopy(value)
    return result


def load_config(path: str | Path) -> dict:
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    if "inherits" in config:
        parent_path = (config_path.parent / config.pop("inherits")).resolve()
        parent = load_config(parent_path)
        config = deep_update(parent, config)

    config["_config_path"] = str(config_path.resolve())
    return config


def save_config(config: dict, path: str | Path) -> None:
    clean = {key: value for key, value in config.items() if not key.startswith("_")}
    with Path(path).open("w", encoding="utf-8") as handle:
        yaml.safe_dump(clean, handle, allow_unicode=True, sort_keys=False)
