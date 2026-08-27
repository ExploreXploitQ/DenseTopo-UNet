"""Single-volume inference with no reference or topology-label input."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, cast

import numpy as np
import torch

from densetopo_unet import __version__
from densetopo_unet.checkpoint import CheckpointError, load_checkpoint
from densetopo_unet.config import VolumeConfig
from densetopo_unet.data import normalization_scale
from densetopo_unet.engine import select_device
from densetopo_unet.io import open_raw_volume, raw_dtype
from densetopo_unet.model import DenseTopoUNet3D
from densetopo_unet.tiling import restore_volume_tiled


@dataclass(frozen=True)
class InferenceRequest:
    """All information permitted at deployment time."""

    checkpoint: Path
    input_path: Path
    output_path: Path
    shape: tuple[int, int, int]
    byte_order: Literal["little", "big"] = "little"
    batch_size: int = 2
    device: Literal["auto", "cpu", "cuda"] = "auto"


@dataclass(frozen=True)
class InferenceRecord:
    """Machine-readable provenance for one restored volume."""

    input_path: str
    output_path: str
    checkpoint_path: str
    shape: tuple[int, int, int]
    dtype: str
    byte_order: str
    input_scale: float
    error_bound: float
    correction_scale: float
    device: str
    seconds: float
    input_sha256: str
    output_sha256: str
    checkpoint_sha256: str
    package_version: str

    def to_dict(self) -> dict[str, Any]:
        return cast(dict[str, Any], asdict(self))


def file_sha256(path: Path) -> str:
    """Return the SHA-256 digest of a local file."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointError(f"checkpoint {location} must be a mapping")
    return cast(Mapping[str, Any], value)


def _triple(value: object, location: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise CheckpointError(f"checkpoint {location} must contain three dimensions")
    try:
        result = tuple(int(item) for item in value)
    except (TypeError, ValueError) as exc:
        raise CheckpointError(f"checkpoint {location} dimensions must be integers") from exc
    if any(item <= 0 for item in result):
        raise CheckpointError(f"checkpoint {location} dimensions must be positive")
    return cast(tuple[int, int, int], result)


def _write_raw_atomic(path: Path, values: np.ndarray, dtype: np.dtype[np.float32]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            np.asarray(values).astype(dtype, copy=False).tofile(handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def restore_file(request: InferenceRequest) -> InferenceRecord:
    """Restore one already decompressed raw volume from a user-trained checkpoint."""

    started = time.perf_counter()
    checkpoint_path = request.checkpoint.resolve()
    input_path = request.input_path.resolve()
    output_path = request.output_path.resolve()
    provenance_path = output_path.with_suffix(output_path.suffix + ".json")
    if output_path.exists() or provenance_path.exists():
        raise ValueError(f"inference output already exists: {output_path}")
    if request.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if len(request.shape) != 3 or any(value <= 0 for value in request.shape):
        raise ValueError("shape must contain positive [D, H, W] dimensions")

    state = load_checkpoint(checkpoint_path)
    volume_section = _mapping(state.config.get("volume"), "config.volume")
    model_section = _mapping(state.config.get("model"), "config.model")
    compression_section = _mapping(state.config.get("compression"), "config.compression")
    normalization_section = _mapping(state.config.get("normalization"), "config.normalization")
    value_domain = str(volume_section.get("value_domain"))
    if value_domain not in {"signed", "nonnegative"}:
        raise CheckpointError("checkpoint value_domain must be signed or nonnegative")
    mode = str(normalization_section.get("mode"))
    if mode not in {"max_abs", "positive_max"}:
        raise CheckpointError("checkpoint normalization mode is unsupported")
    epsilon = float(normalization_section.get("epsilon"))
    xi = float(compression_section.get("absolute_error_bound"))
    correction_scale = float(model_section.get("correction_scale"))
    base_channels = int(model_section.get("base_channels"))
    patch_size = _triple(model_section.get("patch_size"), "config.model.patch_size")

    volume_config = VolumeConfig(
        shape=request.shape,
        dtype="float32",
        byte_order=request.byte_order,
        axis_order="zyx",
        value_domain=cast(Literal["signed", "nonnegative"], value_domain),
    )
    decompressed = open_raw_volume(input_path, volume_config)
    input_scale = normalization_scale(
        decompressed,
        cast(Literal["max_abs", "positive_max"], mode),
        epsilon,
    )
    device = select_device(request.device)
    model = DenseTopoUNet3D(
        base_channels=base_channels,
        correction_scale=correction_scale,
        nonnegative=value_domain == "nonnegative",
    ).to(device)
    try:
        model.load_state_dict(state.model_state, strict=True)
    except RuntimeError as exc:
        raise CheckpointError(f"checkpoint model state is incompatible: {exc}") from exc
    restored = restore_volume_tiled(
        model,
        decompressed,
        input_scale,
        patch_size,
        device,
        request.batch_size,
        xi,
    )
    _write_raw_atomic(output_path, restored, raw_dtype(volume_config))
    record = InferenceRecord(
        input_path=str(input_path),
        output_path=str(output_path),
        checkpoint_path=str(checkpoint_path),
        shape=request.shape,
        dtype="float32",
        byte_order=request.byte_order,
        input_scale=input_scale,
        error_bound=xi,
        correction_scale=correction_scale,
        device=str(device),
        seconds=time.perf_counter() - started,
        input_sha256=file_sha256(input_path),
        output_sha256=file_sha256(output_path),
        checkpoint_sha256=file_sha256(checkpoint_path),
        package_version=__version__,
    )
    provenance_path.write_text(json.dumps(record.to_dict(), indent=2), encoding="utf-8")
    return record
