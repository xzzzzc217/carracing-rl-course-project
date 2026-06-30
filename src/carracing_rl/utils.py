from __future__ import annotations

import csv
import json
import os
import random
from pathlib import Path
from typing import Iterable

import numpy as np


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def resolve_device(name: str):
    import torch

    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def configure_torch_runtime(config: dict) -> None:
    runtime_config = config.get("runtime", {})
    try:
        import torch
    except ImportError:
        return

    num_threads = runtime_config.get("num_threads")
    if num_threads is not None:
        torch.set_num_threads(int(num_threads))

    num_interop_threads = runtime_config.get("num_interop_threads")
    if num_interop_threads is not None:
        try:
            torch.set_num_interop_threads(int(num_interop_threads))
        except RuntimeError:
            # PyTorch allows setting inter-op threads only before parallel work starts.
            pass


def ensure_dir(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def save_json(data: dict, path: str | Path) -> None:
    with Path(path).open("w", encoding="utf-8") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


class CsvLogger:
    def __init__(self, path: str | Path, fieldnames: Iterable[str]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.fieldnames = list(fieldnames)
        self._initialized = self.path.exists() and self.path.stat().st_size > 0

    def write(self, row: dict) -> None:
        filtered = {field: row.get(field, "") for field in self.fieldnames}
        with self.path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=self.fieldnames)
            if not self._initialized:
                writer.writeheader()
                self._initialized = True
            writer.writerow(filtered)
            handle.flush()
            os.fsync(handle.fileno())


def moving_average(values: list[float], window: int) -> list[float]:
    if not values:
        return []
    result = []
    for index in range(len(values)):
        start = max(0, index - window + 1)
        result.append(float(np.mean(values[start : index + 1])))
    return result
