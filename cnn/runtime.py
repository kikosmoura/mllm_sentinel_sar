"""Reprodutibilidade e escrita atomica de artefatos do baseline CNN."""

from __future__ import annotations

import importlib.metadata
import json
import math
import os
import platform
import random
from pathlib import Path

import numpy as np
import torch

from cnn.data import CNNError
from mllm.data_utils import atomic_write_csv, atomic_write_json


def json_safe(value):
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else str(value)
    if isinstance(value, dict):
        return {key: json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    return value


def write_json(path: Path, payload):
    atomic_write_json(path, json_safe(payload))


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        raise CNNError(f"nao gravar CSV vazio: {path}")
    atomic_write_csv(path, list(rows[0]), [json_safe(row) for row in rows])


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise CNNError(f"esperado objeto JSON: {path}")
    return value


def save_checkpoint(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)


def seed_worker(worker_id: int):
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


def select_device(requested: str) -> torch.device:
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        raise CNNError("CUDA solicitada, mas indisponivel; use --device cpu ou auto")
    return torch.device(requested)


def environment(device: torch.device) -> dict:
    versions = {}
    for package in ("torch", "torchvision", "Pillow", "numpy", "scikit-learn", "matplotlib"):
        versions[package] = importlib.metadata.version(package)
    return {
        "library_versions": versions,
        "hardware": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
            "torch_threads": torch.get_num_threads(),
            "ram_bytes": os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"),
            "device": str(device),
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count(),
            "cuda_devices": [torch.cuda.get_device_name(i)
                             for i in range(torch.cuda.device_count())],
        },
    }
