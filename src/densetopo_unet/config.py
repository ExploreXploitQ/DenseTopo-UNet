"""Typed and strictly validated experiment configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal, Mapping, TypeVar, cast

import yaml


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the public schema."""


@dataclass(frozen=True)
class VolumeConfig:
    shape: tuple[int, int, int]
    dtype: Literal["float32"] = "float32"
    byte_order: Literal["little", "big"] = "little"
    axis_order: Literal["zyx"] = "zyx"
    value_domain: Literal["signed", "nonnegative"] = "signed"


@dataclass(frozen=True)
class CompressionConfig:
    absolute_error_bound: float


@dataclass(frozen=True)
class TopologyConfig:
    persistence_threshold: float
    match_radius: int = 0
    neighborhood: Literal["cube26"] = "cube26"


@dataclass(frozen=True)
class NormalizationConfig:
    mode: Literal["max_abs", "positive_max"] = "max_abs"
    epsilon: float = 1.0e-12


@dataclass(frozen=True)
class ModelConfig:
    patch_size: tuple[int, int, int] = (32, 64, 64)
    base_channels: int = 12
    correction_scale: float = 0.75


@dataclass(frozen=True)
class LossConfig:
    mse_mix: float = 0.7
    charbonnier_mix: float = 0.3
    gradient: float = 0.10
    critical: float = 10.0
    topology: float = 5.0
    gate: float = 0.20
    error_bound: float = 25.0
    correction: float = 0.005
    gate_negative: float = 0.02
    error_bound_tail: float = 5.0


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int = 1000
    batch_size: int = 4
    validation_batch_size: int = 4
    samples_per_epoch: int = 256
    validation_samples: int = 128
    num_workers: int = 0
    learning_rate: float = 2.0e-4
    weight_decay: float = 1.0e-5
    minimum_epochs: int = 300
    early_stopping_patience: int = 120
    topology_warmup_start: int = 20
    topology_warmup_end: int = 120
    mixed_precision: bool = True
    seed: int = 2026
    device: Literal["auto", "cpu", "cuda"] = "auto"


@dataclass(frozen=True)
class ExperimentConfig:
    volume: VolumeConfig
    compression: CompressionConfig
    topology: TopologyConfig
    normalization: NormalizationConfig
    model: ModelConfig
    training: TrainingConfig
    loss: LossConfig = field(default_factory=LossConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a checkpoint-safe representation containing only primitives."""

        return cast(dict[str, Any], asdict(self))


T = TypeVar("T")


def _require_mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"{location} must be a mapping")
    if not all(isinstance(key, str) for key in value):
        raise ConfigError(f"{location} keys must be strings")
    return cast(Mapping[str, Any], value)


def _check_keys(
    mapping: Mapping[str, Any], location: str, allowed: set[str], required: set[str]
) -> None:
    unknown = sorted(set(mapping) - allowed)
    missing = sorted(required - set(mapping))
    if unknown:
        raise ConfigError(f"{location} has unknown keys: {', '.join(unknown)}")
    if missing:
        raise ConfigError(f"{location} is missing required keys: {', '.join(missing)}")


def _triple(value: object, location: str) -> tuple[int, int, int]:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ConfigError(f"{location} must contain exactly three dimensions [D, H, W]")
    if not all(isinstance(item, int) and not isinstance(item, bool) for item in value):
        raise ConfigError(f"{location} dimensions must be integers")
    result = cast(tuple[int, int, int], tuple(value))
    if any(item <= 0 for item in result):
        raise ConfigError(f"{location} dimensions must be positive")
    return result


def _positive_float(value: object, location: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) <= 0:
        raise ConfigError(f"{location} must be a positive number")
    return float(value)


def _nonnegative_float(value: object, location: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool) or float(value) < 0:
        raise ConfigError(f"{location} must be a nonnegative number")
    return float(value)


def _positive_int(value: object, location: str, *, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if not isinstance(value, int) or isinstance(value, bool) or value < minimum:
        qualifier = "nonnegative" if allow_zero else "positive"
        raise ConfigError(f"{location} must be a {qualifier} integer")
    return value


def _choice(value: object, location: str, choices: set[str]) -> str:
    if not isinstance(value, str) or value not in choices:
        raise ConfigError(f"{location} must be one of: {', '.join(sorted(choices))}")
    return value


def _parse_experiment_mapping(raw: object) -> ExperimentConfig:
    root = _require_mapping(raw, "configuration")
    sections = {
        "volume",
        "compression",
        "topology",
        "normalization",
        "model",
        "loss",
        "training",
    }
    _check_keys(root, "configuration", sections, sections)

    volume_raw = _require_mapping(root["volume"], "volume")
    volume_keys = {"shape", "dtype", "byte_order", "axis_order", "value_domain"}
    _check_keys(volume_raw, "volume", volume_keys, volume_keys)
    volume = VolumeConfig(
        shape=_triple(volume_raw["shape"], "volume.shape"),
        dtype=cast(Literal["float32"], _choice(volume_raw["dtype"], "volume.dtype", {"float32"})),
        byte_order=cast(
            Literal["little", "big"],
            _choice(volume_raw["byte_order"], "volume.byte_order", {"little", "big"}),
        ),
        axis_order=cast(Literal["zyx"], _choice(volume_raw["axis_order"], "volume.axis_order", {"zyx"})),
        value_domain=cast(
            Literal["signed", "nonnegative"],
            _choice(volume_raw["value_domain"], "volume.value_domain", {"signed", "nonnegative"}),
        ),
    )

    compression_raw = _require_mapping(root["compression"], "compression")
    _check_keys(
        compression_raw,
        "compression",
        {"absolute_error_bound"},
        {"absolute_error_bound"},
    )
    compression = CompressionConfig(
        absolute_error_bound=_positive_float(
            compression_raw["absolute_error_bound"], "compression.absolute_error_bound"
        )
    )

    topology_raw = _require_mapping(root["topology"], "topology")
    topology_keys = {"persistence_threshold", "match_radius", "neighborhood"}
    _check_keys(topology_raw, "topology", topology_keys, topology_keys)
    topology = TopologyConfig(
        persistence_threshold=_positive_float(
            topology_raw["persistence_threshold"], "topology.persistence_threshold"
        ),
        match_radius=_positive_int(
            topology_raw["match_radius"], "topology.match_radius", allow_zero=True
        ),
        neighborhood=cast(
            Literal["cube26"],
            _choice(topology_raw["neighborhood"], "topology.neighborhood", {"cube26"}),
        ),
    )

    normalization_raw = _require_mapping(root["normalization"], "normalization")
    normalization_keys = {"mode", "epsilon"}
    _check_keys(normalization_raw, "normalization", normalization_keys, normalization_keys)
    normalization = NormalizationConfig(
        mode=cast(
            Literal["max_abs", "positive_max"],
            _choice(normalization_raw["mode"], "normalization.mode", {"max_abs", "positive_max"}),
        ),
        epsilon=_positive_float(normalization_raw["epsilon"], "normalization.epsilon"),
    )

    model_raw = _require_mapping(root["model"], "model")
    model_keys = {"patch_size", "base_channels", "correction_scale"}
    _check_keys(model_raw, "model", model_keys, model_keys)
    patch_size = _triple(model_raw["patch_size"], "model.patch_size")
    if any(item % 8 != 0 for item in patch_size):
        raise ConfigError("model.patch_size dimensions must be divisible by 8")
    if any(patch > full for patch, full in zip(patch_size, volume.shape)):
        raise ConfigError("model.patch_size must not exceed volume.shape")
    model = ModelConfig(
        patch_size=patch_size,
        base_channels=_positive_int(model_raw["base_channels"], "model.base_channels"),
        correction_scale=_positive_float(model_raw["correction_scale"], "model.correction_scale"),
    )

    loss_raw = _require_mapping(root["loss"], "loss")
    loss_keys = {
        "mse_mix",
        "charbonnier_mix",
        "gradient",
        "critical",
        "topology",
        "gate",
        "error_bound",
        "correction",
        "gate_negative",
        "error_bound_tail",
    }
    _check_keys(loss_raw, "loss", loss_keys, loss_keys)
    loss = LossConfig(
        **{
            key: _nonnegative_float(loss_raw[key], f"loss.{key}")
            for key in loss_keys
        }
    )
    if abs(loss.mse_mix + loss.charbonnier_mix - 1.0) > 1.0e-6:
        raise ConfigError("loss.mse_mix and loss.charbonnier_mix must sum to 1")

    training_raw = _require_mapping(root["training"], "training")
    training_keys = {
        "epochs",
        "batch_size",
        "validation_batch_size",
        "samples_per_epoch",
        "validation_samples",
        "num_workers",
        "learning_rate",
        "weight_decay",
        "minimum_epochs",
        "early_stopping_patience",
        "topology_warmup_start",
        "topology_warmup_end",
        "mixed_precision",
        "seed",
        "device",
    }
    _check_keys(training_raw, "training", training_keys, training_keys)
    mixed_precision = training_raw["mixed_precision"]
    if not isinstance(mixed_precision, bool):
        raise ConfigError("training.mixed_precision must be a boolean")
    training = TrainingConfig(
        epochs=_positive_int(training_raw["epochs"], "training.epochs"),
        batch_size=_positive_int(training_raw["batch_size"], "training.batch_size"),
        validation_batch_size=_positive_int(
            training_raw["validation_batch_size"], "training.validation_batch_size"
        ),
        samples_per_epoch=_positive_int(
            training_raw["samples_per_epoch"], "training.samples_per_epoch"
        ),
        validation_samples=_positive_int(
            training_raw["validation_samples"], "training.validation_samples"
        ),
        num_workers=_positive_int(
            training_raw["num_workers"], "training.num_workers", allow_zero=True
        ),
        learning_rate=_positive_float(
            training_raw["learning_rate"], "training.learning_rate"
        ),
        weight_decay=_positive_float(training_raw["weight_decay"], "training.weight_decay"),
        minimum_epochs=_positive_int(
            training_raw["minimum_epochs"], "training.minimum_epochs"
        ),
        early_stopping_patience=_positive_int(
            training_raw["early_stopping_patience"], "training.early_stopping_patience"
        ),
        topology_warmup_start=_positive_int(
            training_raw["topology_warmup_start"],
            "training.topology_warmup_start",
            allow_zero=True,
        ),
        topology_warmup_end=_positive_int(
            training_raw["topology_warmup_end"],
            "training.topology_warmup_end",
            allow_zero=True,
        ),
        mixed_precision=mixed_precision,
        seed=_positive_int(training_raw["seed"], "training.seed", allow_zero=True),
        device=cast(
            Literal["auto", "cpu", "cuda"],
            _choice(training_raw["device"], "training.device", {"auto", "cpu", "cuda"}),
        ),
    )
    if training.minimum_epochs > training.epochs:
        raise ConfigError("training.minimum_epochs must not exceed training.epochs")
    if training.topology_warmup_end < training.topology_warmup_start:
        raise ConfigError(
            "training.topology_warmup_end must be greater than or equal to the start"
        )

    return ExperimentConfig(volume, compression, topology, normalization, model, training, loss)


def load_experiment_config(path: Path) -> ExperimentConfig:
    """Load an experiment configuration from a UTF-8 YAML file."""

    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ConfigError(f"cannot read configuration {path}: {exc}") from exc
    return _parse_experiment_mapping(raw)
