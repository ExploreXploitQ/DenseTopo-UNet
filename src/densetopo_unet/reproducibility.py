"""Deterministic execution and environment provenance."""

from __future__ import annotations

import os
import platform
import random
import sys
import warnings
from typing import Any, Mapping, cast

import numpy as np
import torch


def cuda_is_usable() -> bool:
    """Probe CUDA without leaking optional NVML warnings into CPU workflows."""

    try:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*initialize NVML.*", category=UserWarning)
            return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except (OSError, RuntimeError):
        return False


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, CPU PyTorch, and every visible CUDA device."""

    if seed < 0:
        raise ValueError("seed must be nonnegative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if cuda_is_usable():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*initialize NVML.*", category=UserWarning)
            torch.cuda.manual_seed_all(seed)


def capture_rng_state() -> dict[str, Any]:
    """Capture all random-number generators used by the package."""

    cuda_states: list[torch.Tensor] = []
    if cuda_is_usable():
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*initialize NVML.*", category=UserWarning)
            cuda_states = torch.cuda.get_rng_state_all()
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": cuda_states,
    }


def restore_rng_state(state: Mapping[str, Any]) -> None:
    """Restore a state produced by :func:`capture_rng_state`."""

    required = {"python", "numpy", "torch_cpu", "torch_cuda"}
    missing = sorted(required - set(state))
    if missing:
        raise ValueError(f"RNG state is missing keys: {', '.join(missing)}")
    random.setstate(cast(tuple[Any, ...], state["python"]))
    np.random.set_state(cast(tuple[Any, ...], state["numpy"]))
    torch.set_rng_state(cast(torch.Tensor, state["torch_cpu"]))
    cuda_states = cast(list[torch.Tensor], state["torch_cuda"])
    if cuda_states and cuda_is_usable():
        torch.cuda.set_rng_state_all(cuda_states)


def environment_report() -> dict[str, Any]:
    """Return machine-readable runtime provenance without external commands."""

    cuda_available = cuda_is_usable()
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "torch": torch.__version__,
        "cuda_available": cuda_available,
        "cuda_runtime": torch.version.cuda,
        "visible_cuda_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "device_names": [
            torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())
        ]
        if cuda_available
        else [],
    }
