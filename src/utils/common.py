"""Small dependency-light utilities shared by the pipeline scripts."""

from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path
from typing import Any


def configure_logging(name: str = "llm_assignment") -> logging.Logger:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)


def set_seed(seed: int) -> None:
    """Seed Python, NumPy, and Torch when those optional runtimes are installed."""
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def model_dtype() -> Any:
    """Use bfloat16 on CUDA as required by the T4 assignment, float32 on CPU."""
    try:
        import torch
        return torch.bfloat16 if torch.cuda.is_available() else torch.float32
    except ImportError:
        return None

