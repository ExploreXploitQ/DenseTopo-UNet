"""Versioned and atomic DenseTopo-UNet checkpoints."""

from __future__ import annotations

import hashlib
import os
import tempfile
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import torch

from densetopo_unet.config import ExperimentConfig

CHECKPOINT_SCHEMA_VERSION = 1


class CheckpointError(ValueError):
    """Raised when a checkpoint is corrupt or incompatible."""


@dataclass(frozen=True)
class CheckpointState:
    """Complete state required for inference or exact training resume."""

    schema_version: int
    package_version: str
    config: dict[str, Any]
    manifest_sha256: str
    model_state: dict[str, torch.Tensor]
    optimizer_state: dict[str, Any]
    scheduler_state: dict[str, Any]
    scaler_state: dict[str, Any] | None
    epoch: int
    best_score: float
    bad_epochs: int
    history: list[dict[str, float]]
    rng_state: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def manifest_fingerprint(path: Path) -> str:
    """Return a SHA-256 digest over the exact manifest bytes."""

    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                digest.update(chunk)
    except OSError as exc:
        raise CheckpointError(f"cannot fingerprint manifest {path}: {exc}") from exc
    return digest.hexdigest()


def save_checkpoint_atomic(path: Path, state: CheckpointState) -> None:
    """Write a checkpoint completely before atomically replacing its destination."""

    if state.schema_version != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            f"cannot save checkpoint schema {state.schema_version}; "
            f"supported schema is {CHECKPOINT_SCHEMA_VERSION}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(state.to_dict(), handle)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _require_mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise CheckpointError(f"{location} must be a mapping")
    return cast(Mapping[str, Any], value)


def _state_from_mapping(value: object) -> CheckpointState:
    raw = _require_mapping(value, "checkpoint")
    names = {
        "schema_version",
        "package_version",
        "config",
        "manifest_sha256",
        "model_state",
        "optimizer_state",
        "scheduler_state",
        "scaler_state",
        "epoch",
        "best_score",
        "bad_epochs",
        "history",
        "rng_state",
    }
    missing = sorted(names - set(raw))
    unknown = sorted(set(raw) - names)
    if missing or unknown:
        raise CheckpointError(
            f"checkpoint fields are invalid; missing={missing}, unknown={unknown}"
        )
    if raw["schema_version"] != CHECKPOINT_SCHEMA_VERSION:
        raise CheckpointError(
            f"unsupported checkpoint schema {raw['schema_version']}; "
            f"expected {CHECKPOINT_SCHEMA_VERSION}"
        )
    config = dict(_require_mapping(raw["config"], "checkpoint.config"))
    model_state_raw = _require_mapping(raw["model_state"], "checkpoint.model_state")
    return CheckpointState(
        schema_version=int(raw["schema_version"]),
        package_version=str(raw["package_version"]),
        config=config,
        manifest_sha256=str(raw["manifest_sha256"]),
        model_state={str(key): cast(torch.Tensor, item) for key, item in model_state_raw.items()},
        optimizer_state=dict(_require_mapping(raw["optimizer_state"], "optimizer_state")),
        scheduler_state=dict(_require_mapping(raw["scheduler_state"], "scheduler_state")),
        scaler_state=None
        if raw["scaler_state"] is None
        else dict(_require_mapping(raw["scaler_state"], "scaler_state")),
        epoch=int(raw["epoch"]),
        best_score=float(raw["best_score"]),
        bad_epochs=int(raw["bad_epochs"]),
        history=[dict(item) for item in cast(list[Mapping[str, float]], raw["history"])],
        rng_state=dict(_require_mapping(raw["rng_state"], "rng_state")),
    )


def _config_mapping(value: ExperimentConfig | Mapping[str, Any]) -> Mapping[str, Any]:
    return value.to_dict() if isinstance(value, ExperimentConfig) else value


def _check_compatibility(
    checkpoint_config: Mapping[str, Any],
    expected_config: ExperimentConfig | Mapping[str, Any],
) -> None:
    expected = _config_mapping(expected_config)
    for section in ("model", "compression", "normalization", "loss"):
        if checkpoint_config.get(section) != expected.get(section):
            label = "model configuration" if section == "model" else f"{section} configuration"
            raise CheckpointError(f"checkpoint {label} does not match the expected configuration")
    checkpoint_volume = _require_mapping(checkpoint_config.get("volume"), "config.volume")
    expected_volume = _require_mapping(expected.get("volume"), "expected.volume")
    if checkpoint_volume.get("value_domain") != expected_volume.get("value_domain"):
        raise CheckpointError("checkpoint value domain does not match the expected configuration")


def load_checkpoint(
    path: Path,
    expected_config: ExperimentConfig | Mapping[str, Any] | None = None,
) -> CheckpointState:
    """Load a checkpoint on CPU and optionally enforce experiment compatibility."""

    try:
        raw = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError, EOFError) as exc:
        raise CheckpointError(f"cannot load checkpoint {path}: {exc}") from exc
    state = _state_from_mapping(raw)
    if expected_config is not None:
        _check_compatibility(state.config, expected_config)
    return state
