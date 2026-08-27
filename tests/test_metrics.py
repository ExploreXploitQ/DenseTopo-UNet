import json
import math
from pathlib import Path

import numpy as np
import pytest

from densetopo_unet.metrics import (
    EvaluationRow,
    FCSummary,
    aggregate_evaluation,
    compute_error_metrics,
    load_fc_summary,
)


def test_error_metrics_counts_strict_error_bound_violations() -> None:
    reference = np.zeros(3, dtype=np.float32)
    candidate = np.array([0.0, 1.0e-4, 1.1e-4], dtype=np.float32)

    metrics = compute_error_metrics(reference, candidate, 1.0e-4, 1.0)

    assert metrics.eb_violations == 1
    assert metrics.max_abs_error == pytest.approx(1.1e-4)
    assert metrics.eb_violation_fraction == pytest.approx(1.0 / 3.0)


def test_identical_arrays_have_infinite_psnr() -> None:
    values = np.arange(8, dtype=np.float32)

    metrics = compute_error_metrics(values, values, 1.0e-4, 7.0)

    assert metrics.rmse == 0.0
    assert math.isinf(metrics.psnr)


def test_fc_summary_loader_requires_exact_integer_counts(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"FP": 4, "FN": 5, "FT": 1}), encoding="utf-8")

    summary = load_fc_summary(path)

    assert summary == FCSummary(fp=4, fn=5, ft=1)
    assert summary.total == 10


def test_fc_summary_loader_rejects_missing_key(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text(json.dumps({"FP": 4, "FN": 5}), encoding="utf-8")

    with pytest.raises(ValueError, match="FT"):
        load_fc_summary(path)


def test_aggregate_evaluation_sums_fc_and_restored_violations() -> None:
    zero = compute_error_metrics(np.zeros(2), np.zeros(2), 1.0e-4, 1.0)
    one_violation = compute_error_metrics(
        np.zeros(2), np.array([0.0, 2.0e-4]), 1.0e-4, 1.0
    )
    rows = [
        EvaluationRow("a", zero, one_violation, FCSummary(3, 2, 0), FCSummary(2, 1, 0)),
        EvaluationRow("b", zero, zero, FCSummary(2, 1, 0), FCSummary(1, 1, 0)),
    ]

    aggregate = aggregate_evaluation(rows)

    assert aggregate.baseline_fc == FCSummary(5, 3, 0)
    assert aggregate.restored_fc == FCSummary(3, 2, 0)
    assert aggregate.fc_removed_fraction == pytest.approx(3.0 / 8.0)
    assert aggregate.restored_eb_violations == 1
