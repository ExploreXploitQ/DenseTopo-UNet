"""Generic experiment manifest and topology-label validation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Mapping, Sequence, cast

import numpy as np
import yaml

from densetopo_unet.config import VolumeConfig
from densetopo_unet.io import validate_raw_volume


class ManifestError(ValueError):
    """Raised when a data manifest or topology label violates its schema."""


Split = Literal["train", "validation", "test"]


@dataclass(frozen=True)
class SampleRecord:
    sample_id: str
    split: Split
    decompressed: Path
    reference: Path | None = None
    false_cases: Path | None = None
    critical_points: Path | None = None


@dataclass(frozen=True)
class DataManifest:
    schema_version: int
    experiment: str
    path: Path
    samples: tuple[SampleRecord, ...]

    def by_split(self, split: Split) -> tuple[SampleRecord, ...]:
        """Return records in manifest order for one split."""

        return tuple(sample for sample in self.samples if sample.split == split)


def _mapping(value: object, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ManifestError(f"{location} must be a mapping with string keys")
    return cast(Mapping[str, Any], value)


def _exact_keys(
    mapping: Mapping[str, Any],
    location: str,
    allowed: set[str],
    required: set[str],
) -> None:
    unknown = sorted(set(mapping) - allowed)
    missing = sorted(required - set(mapping))
    if unknown:
        raise ManifestError(f"{location} has unknown keys: {', '.join(unknown)}")
    if missing:
        raise ManifestError(f"{location} is missing required keys: {', '.join(missing)}")


def _manifest_volume(value: object, expected: VolumeConfig) -> None:
    raw = _mapping(value, "volume")
    keys = {"shape", "dtype", "byte_order", "axis_order"}
    _exact_keys(raw, "volume", keys, keys)
    shape_raw = raw["shape"]
    if not isinstance(shape_raw, Sequence) or isinstance(shape_raw, (str, bytes)):
        raise ManifestError("volume.shape must be [D, H, W]")
    shape = tuple(shape_raw)
    observed = {
        "shape": shape,
        "dtype": raw["dtype"],
        "byte_order": raw["byte_order"],
        "axis_order": raw["axis_order"],
    }
    expected_values = {
        "shape": expected.shape,
        "dtype": expected.dtype,
        "byte_order": expected.byte_order,
        "axis_order": expected.axis_order,
    }
    if observed != expected_values:
        raise ManifestError(
            f"manifest volume metadata {observed} does not match configuration {expected_values}"
        )


def _resolve_optional(root: Path, value: object, location: str) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ManifestError(f"{location} must be a nonempty path string")
    return (root / value).resolve()


def _sample(value: object, index: int, root: Path, volume: VolumeConfig) -> SampleRecord:
    raw = _mapping(value, f"samples[{index}]")
    allowed = {"id", "split", "decompressed", "reference", "false_cases", "critical_points"}
    _exact_keys(raw, f"samples[{index}]", allowed, {"id", "split", "decompressed"})
    sample_id = raw["id"]
    if not isinstance(sample_id, str) or not sample_id.strip():
        raise ManifestError(f"samples[{index}].id must be a nonempty string")
    split = raw["split"]
    if split not in {"train", "validation", "test"}:
        raise ManifestError(f"samples[{index}].split must be train, validation, or test")

    decompressed = _resolve_optional(root, raw["decompressed"], f"samples[{index}].decompressed")
    assert decompressed is not None
    reference = _resolve_optional(root, raw.get("reference"), f"samples[{index}].reference")
    false_cases = _resolve_optional(
        root, raw.get("false_cases"), f"samples[{index}].false_cases"
    )
    critical_points = _resolve_optional(
        root, raw.get("critical_points"), f"samples[{index}].critical_points"
    )
    if split in {"train", "validation"}:
        for name, path in (
            ("reference", reference),
            ("false_cases", false_cases),
            ("critical_points", critical_points),
        ):
            if path is None:
                raise ManifestError(f"samples[{index}].{name} is required for split {split}")

    try:
        validate_raw_volume(decompressed, volume)
        if reference is not None:
            validate_raw_volume(reference, volume)
    except (FileNotFoundError, ValueError) as exc:
        raise ManifestError(f"sample {sample_id}: {exc}") from exc
    for label in (false_cases, critical_points):
        if label is not None and not label.is_file():
            raise ManifestError(f"sample {sample_id}: label file does not exist: {label}")

    return SampleRecord(
        sample_id=sample_id,
        split=cast(Split, split),
        decompressed=decompressed,
        reference=reference,
        false_cases=false_cases,
        critical_points=critical_points,
    )


def load_manifest(path: Path, volume: VolumeConfig) -> DataManifest:
    """Load and validate a generic data manifest."""

    resolved = path.resolve()
    try:
        raw_value = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise ManifestError(f"cannot read manifest {resolved}: {exc}") from exc
    raw = _mapping(raw_value, "manifest")
    keys = {"schema_version", "experiment", "volume", "samples"}
    _exact_keys(raw, "manifest", keys, keys)
    if raw["schema_version"] != 1:
        raise ManifestError("schema_version must be 1")
    experiment = raw["experiment"]
    if not isinstance(experiment, str) or not experiment.strip():
        raise ManifestError("experiment must be a nonempty string")
    _manifest_volume(raw["volume"], volume)
    samples_raw = raw["samples"]
    if not isinstance(samples_raw, list) or not samples_raw:
        raise ManifestError("samples must be a nonempty list")
    samples = tuple(_sample(item, index, resolved.parent, volume) for index, item in enumerate(samples_raw))
    ids = [sample.sample_id for sample in samples]
    duplicates = sorted({sample_id for sample_id in ids if ids.count(sample_id) > 1})
    if duplicates:
        raise ManifestError(f"sample IDs must be unique; duplicates: {', '.join(duplicates)}")
    return DataManifest(1, experiment, resolved, samples)


def _coordinate(row: Mapping[str, str], line: int, shape: tuple[int, int, int]) -> tuple[int, int, int]:
    try:
        point = (int(row["z"]), int(row["y"]), int(row["x"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ManifestError(f"line {line}: z, y, and x must be integers") from exc
    if any(value < 0 or value >= limit for value, limit in zip(point, shape)):
        raise ManifestError(f"line {line}: coordinate {point} is outside shape {shape}")
    return point


def _csv_rows(path: Path, expected_header: list[str]) -> list[dict[str, str]]:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != expected_header:
                raise ManifestError(
                    f"{path}: expected CSV header {','.join(expected_header)}, "
                    f"observed {','.join(reader.fieldnames or [])}"
                )
            return list(reader)
    except OSError as exc:
        raise ManifestError(f"cannot read label CSV {path}: {exc}") from exc


def load_false_cases(path: Path, shape: tuple[int, int, int]) -> dict[str, np.ndarray]:
    """Load strict FP/FN/FT coordinates grouped by case."""

    grouped: dict[str, list[tuple[int, int, int]]] = {"fp": [], "fn": [], "ft": []}
    for line, row in enumerate(_csv_rows(path, ["case", "z", "y", "x"]), start=2):
        case = row["case"].lower()
        if case not in grouped:
            raise ManifestError(f"line {line}: case must be fp, fn, or ft")
        grouped[case].append(_coordinate(row, line, shape))
    return {
        case: np.asarray(points, dtype=np.int32).reshape(-1, 3)
        for case, points in grouped.items()
    }


def load_critical_points(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    """Load strict reference local-extremum coordinates."""

    points: list[tuple[int, int, int]] = []
    allowed = {"local_maximum", "local_minimum"}
    for line, row in enumerate(
        _csv_rows(path, ["critical_type", "z", "y", "x"]), start=2
    ):
        critical_type = row["critical_type"]
        if critical_type not in allowed:
            raise ManifestError(
                f"line {line}: critical_type must be local_maximum or local_minimum"
            )
        points.append(_coordinate(row, line, shape))
    return np.asarray(points, dtype=np.int32).reshape(-1, 3)

