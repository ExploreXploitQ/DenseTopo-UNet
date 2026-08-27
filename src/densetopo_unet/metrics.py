"""Numerical metrics and aggregation of external topology summaries."""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np

from densetopo_unet.config import ExperimentConfig
from densetopo_unet.io import open_raw_volume
from densetopo_unet.manifest import DataManifest, Split


@dataclass(frozen=True)
class ErrorMetrics:
    max_abs_error: float
    rmse: float
    psnr: float
    eb_violations: int
    eb_violation_fraction: float

    def to_dict(self) -> dict[str, float | int]:
        return cast(dict[str, float | int], asdict(self))


@dataclass(frozen=True)
class FCSummary:
    fp: int
    fn: int
    ft: int

    def __post_init__(self) -> None:
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in (self.fp, self.fn, self.ft)
        ):
            raise ValueError("FP, FN, and FT counts must be nonnegative integers")

    @property
    def total(self) -> int:
        return self.fp + self.fn + self.ft

    def to_dict(self) -> dict[str, int]:
        return {"FP": self.fp, "FN": self.fn, "FT": self.ft, "FC_total": self.total}

    def __add__(self, other: FCSummary) -> FCSummary:
        return FCSummary(self.fp + other.fp, self.fn + other.fn, self.ft + other.ft)


@dataclass(frozen=True)
class EvaluationRow:
    sample_id: str
    baseline_error: ErrorMetrics
    restored_error: ErrorMetrics
    baseline_fc: FCSummary | None = None
    restored_fc: FCSummary | None = None

    def to_dict(self) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sample_id": self.sample_id,
            "baseline_error": self.baseline_error.to_dict(),
            "restored_error": self.restored_error.to_dict(),
        }
        if self.baseline_fc is not None and self.restored_fc is not None:
            result["baseline_fc"] = self.baseline_fc.to_dict()
            result["restored_fc"] = self.restored_fc.to_dict()
            result["fc_removed_fraction"] = 1.0 - (
                self.restored_fc.total / max(self.baseline_fc.total, 1)
            )
        return result


@dataclass(frozen=True)
class EvaluationSummary:
    rows: tuple[EvaluationRow, ...]
    baseline_fc: FCSummary | None
    restored_fc: FCSummary | None
    fc_removed_fraction: float | None
    restored_eb_violations: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "samples": [row.to_dict() for row in self.rows],
            "aggregate": {
                "baseline_fc": self.baseline_fc.to_dict() if self.baseline_fc else None,
                "restored_fc": self.restored_fc.to_dict() if self.restored_fc else None,
                "fc_removed_fraction": self.fc_removed_fraction,
                "restored_eb_violations": self.restored_eb_violations,
            },
        }


def compute_error_metrics(
    reference: np.ndarray,
    candidate: np.ndarray,
    error_bound: float,
    data_range: float,
    chunk_elements: int = 1 << 20,
) -> ErrorMetrics:
    """Compute error metrics in bounded memory with float64 accumulation."""

    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate must have identical shapes")
    if reference.size == 0:
        raise ValueError("metric arrays must not be empty")
    if error_bound <= 0 or data_range <= 0 or chunk_elements <= 0:
        raise ValueError("error_bound, data_range, and chunk_elements must be positive")
    reference_flat = np.asarray(reference).reshape(-1)
    candidate_flat = np.asarray(candidate).reshape(-1)
    squared_sum = 0.0
    maximum = 0.0
    violations = 0
    for start in range(0, reference_flat.size, chunk_elements):
        stop = min(reference_flat.size, start + chunk_elements)
        error = np.abs(
            candidate_flat[start:stop].astype(np.float64)
            - reference_flat[start:stop].astype(np.float64)
        )
        if not np.isfinite(error).all():
            raise ValueError("metric arrays must contain only finite values")
        squared_sum += float(np.dot(error, error))
        maximum = max(maximum, float(np.max(error)))
        violations += int(np.count_nonzero(error > error_bound))
    mse = squared_sum / reference_flat.size
    rmse = math.sqrt(mse)
    psnr = math.inf if mse == 0.0 else 20.0 * math.log10(data_range / rmse)
    return ErrorMetrics(
        max_abs_error=maximum,
        rmse=rmse,
        psnr=psnr,
        eb_violations=violations,
        eb_violation_fraction=violations / reference_flat.size,
    )


def load_fc_summary(path: Path) -> FCSummary:
    """Load exact FP/FN/FT counts produced by an external topology evaluator."""

    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read FC summary {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"FC summary {path} must contain a JSON object")
    missing = [key for key in ("FP", "FN", "FT") if key not in value]
    if missing:
        raise ValueError(f"FC summary {path} is missing keys: {', '.join(missing)}")
    counts = [value[key] for key in ("FP", "FN", "FT")]
    if any(not isinstance(item, int) or isinstance(item, bool) or item < 0 for item in counts):
        raise ValueError(f"FC summary {path} counts must be nonnegative integers")
    return FCSummary(*counts)


def aggregate_evaluation(rows: Sequence[EvaluationRow]) -> EvaluationSummary:
    """Aggregate error-bound violations and optional topology counts."""

    if not rows:
        raise ValueError("evaluation rows must not be empty")
    topology_presence = [
        row.baseline_fc is not None and row.restored_fc is not None for row in rows
    ]
    topology_absence = [row.baseline_fc is None and row.restored_fc is None for row in rows]
    if not (all(topology_presence) or all(topology_absence)):
        raise ValueError("every evaluation row must consistently include or omit both FC summaries")
    baseline_fc: FCSummary | None = None
    restored_fc: FCSummary | None = None
    removed: float | None = None
    if all(topology_presence):
        baseline_fc = FCSummary(0, 0, 0)
        restored_fc = FCSummary(0, 0, 0)
        for row in rows:
            assert row.baseline_fc is not None and row.restored_fc is not None
            baseline_fc = baseline_fc + row.baseline_fc
            restored_fc = restored_fc + row.restored_fc
        removed = 1.0 - restored_fc.total / max(baseline_fc.total, 1)
    return EvaluationSummary(
        rows=tuple(rows),
        baseline_fc=baseline_fc,
        restored_fc=restored_fc,
        fc_removed_fraction=removed,
        restored_eb_violations=sum(row.restored_error.eb_violations for row in rows),
    )


def _data_range(reference: np.ndarray) -> float:
    minimum = float(np.min(reference))
    maximum = float(np.max(reference))
    span = maximum - minimum
    if not math.isfinite(span):
        raise ValueError("reference volume contains non-finite values")
    return span if span > 0 else max(abs(minimum), abs(maximum), 1.0)


def evaluate_restored(
    config: ExperimentConfig,
    manifest: DataManifest,
    restored_root: Path,
    split: Split,
    output_dir: Path,
    baseline_topology_dir: Path | None = None,
    restored_topology_dir: Path | None = None,
) -> EvaluationSummary:
    """Evaluate existing restored files against manifest references."""

    if (baseline_topology_dir is None) != (restored_topology_dir is None):
        raise ValueError("baseline and restored topology directories must be supplied together")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError(f"evaluation output directory must be new or empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[EvaluationRow] = []
    for sample in manifest.by_split(split):
        if sample.reference is None:
            raise ValueError(f"sample {sample.sample_id} has no reference volume for evaluation")
        restored_path = restored_root / f"{sample.sample_id}.restored.f32"
        reference = open_raw_volume(sample.reference, config.volume)
        decompressed = open_raw_volume(sample.decompressed, config.volume)
        restored = open_raw_volume(restored_path, config.volume)
        span = _data_range(reference)
        baseline_fc = None
        restored_fc = None
        if baseline_topology_dir is not None and restored_topology_dir is not None:
            baseline_fc = load_fc_summary(baseline_topology_dir / f"{sample.sample_id}.json")
            restored_fc = load_fc_summary(restored_topology_dir / f"{sample.sample_id}.json")
        rows.append(
            EvaluationRow(
                sample_id=sample.sample_id,
                baseline_error=compute_error_metrics(
                    reference,
                    decompressed,
                    config.compression.absolute_error_bound,
                    span,
                ),
                restored_error=compute_error_metrics(
                    reference,
                    restored,
                    config.compression.absolute_error_bound,
                    span,
                ),
                baseline_fc=baseline_fc,
                restored_fc=restored_fc,
            )
        )
    summary = aggregate_evaluation(rows)
    (output_dir / "evaluation_summary.json").write_text(
        json.dumps(summary.to_dict(), indent=2), encoding="utf-8"
    )
    flat_rows: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {"sample_id": row.sample_id}
        flat.update(
            {f"baseline_{key}": value for key, value in row.baseline_error.to_dict().items()}
        )
        flat.update(
            {f"restored_{key}": value for key, value in row.restored_error.to_dict().items()}
        )
        if row.baseline_fc is not None and row.restored_fc is not None:
            flat.update(
                {f"baseline_{key}": value for key, value in row.baseline_fc.to_dict().items()}
            )
            flat.update(
                {f"restored_{key}": value for key, value in row.restored_fc.to_dict().items()}
            )
        flat_rows.append(flat)
    with (output_dir / "metrics_by_sample.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat_rows[0]))
        writer.writeheader()
        writer.writerows(flat_rows)
    return summary
