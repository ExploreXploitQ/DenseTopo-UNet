"""Raw three-dimensional scalar-volume I/O."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal

import numpy as np

from densetopo_unet.config import VolumeConfig


def raw_dtype(config: VolumeConfig) -> np.dtype[np.float32]:
    """Return the configured IEEE-754 float32 dtype."""

    return np.dtype("<f4" if config.byte_order == "little" else ">f4")


def expected_raw_bytes(config: VolumeConfig) -> int:
    """Return the only valid byte count for a configured raw volume."""

    return math.prod(config.shape) * np.dtype(np.float32).itemsize


def validate_raw_volume(path: Path, config: VolumeConfig) -> None:
    """Validate existence and exact byte count before memory mapping a file."""

    if not path.is_file():
        raise FileNotFoundError(f"raw volume does not exist: {path}")
    expected = expected_raw_bytes(config)
    observed = path.stat().st_size
    if observed != expected:
        raise ValueError(f"{path}: expected {expected} bytes, observed {observed}")


def open_raw_volume(
    path: Path,
    config: VolumeConfig,
    mode: Literal["r", "r+", "w+"] = "r",
) -> np.memmap:
    """Open an exact-size flat raw file as a `[D, H, W]` memory map."""

    if mode != "w+":
        validate_raw_volume(path, config)
    return np.memmap(path, dtype=raw_dtype(config), mode=mode, shape=config.shape, order="C")


def validate_finite_values(path: Path, config: VolumeConfig, chunk_elements: int = 1 << 20) -> None:
    """Reject NaN or infinite values without loading the full volume into RAM."""

    if chunk_elements <= 0:
        raise ValueError("chunk_elements must be positive")
    volume = open_raw_volume(path, config).reshape(-1)
    for start in range(0, volume.size, chunk_elements):
        chunk = np.asarray(volume[start : start + chunk_elements])
        if not np.isfinite(chunk).all():
            raise ValueError(f"{path}: contains non-finite floating-point values")
