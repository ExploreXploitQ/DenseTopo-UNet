#!/usr/bin/env python3
"""Generate a deterministic, analytic 3D topology-restoration example."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import yaml


ERROR_BOUND = 0.02


def _validate_shape(shape: tuple[int, int, int]) -> None:
    if len(shape) != 3 or any(item < 8 or item % 8 != 0 for item in shape):
        raise ValueError("synthetic shape must contain three multiples of 8 in [D, H, W] order")


def _coordinate(value: float, size: int) -> int:
    return int(round((value + 1.0) * 0.5 * (size - 1)))


def _analytic_field(shape: tuple[int, int, int], phase: float) -> np.ndarray:
    z_axis = np.linspace(-1.0, 1.0, shape[0], dtype=np.float64)
    y_axis = np.linspace(-1.0, 1.0, shape[1], dtype=np.float64)
    x_axis = np.linspace(-1.0, 1.0, shape[2], dtype=np.float64)
    z, y, x = np.meshgrid(z_axis, y_axis, x_axis, indexing="ij")
    positive_peak = 0.95 * np.exp(
        -((z + 0.35) ** 2 / 0.13 + (y + 0.15) ** 2 / 0.09 + (x - 0.25) ** 2 / 0.08)
    )
    negative_peak = -0.72 * np.exp(
        -((z - 0.30) ** 2 / 0.10 + (y - 0.25) ** 2 / 0.12 + (x + 0.30) ** 2 / 0.10)
    )
    background = 0.08 * np.sin(2.5 * x + phase) * np.cos(2.0 * y - phase)
    trend = 0.04 * z
    return np.asarray(positive_peak + negative_peak + background + trend, dtype=np.float32)


def _lossy_decompression_proxy(reference: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    perturbation = rng.uniform(-0.95 * ERROR_BOUND, 0.95 * ERROR_BOUND, reference.shape)
    return np.asarray(reference.astype(np.float64) + perturbation, dtype=np.float32)


def _write_labels(
    root: Path,
    sample_id: str,
    shape: tuple[int, int, int],
) -> tuple[Path, Path]:
    positive = tuple(
        _coordinate(value, size) for value, size in zip((-0.35, -0.15, 0.25), shape)
    )
    negative = tuple(
        _coordinate(value, size) for value, size in zip((0.30, 0.25, -0.30), shape)
    )
    center = tuple(size // 2 for size in shape)
    false_cases = root / f"{sample_id}.false-cases.csv"
    false_cases.write_text(
        "case,z,y,x\n"
        f"fn,{positive[0]},{positive[1]},{positive[2]}\n"
        f"fp,{center[0]},{center[1]},{center[2]}\n"
        f"ft,{negative[0]},{negative[1]},{negative[2]}\n",
        encoding="utf-8",
    )
    critical_points = root / f"{sample_id}.critical-points.csv"
    critical_points.write_text(
        "critical_type,z,y,x\n"
        f"local_maximum,{positive[0]},{positive[1]},{positive[2]}\n"
        f"local_minimum,{negative[0]},{negative[1]},{negative[2]}\n",
        encoding="utf-8",
    )
    return false_cases, critical_points


def _configuration(shape: tuple[int, int, int], seed: int) -> dict[str, object]:
    return {
        "volume": {
            "shape": list(shape),
            "dtype": "float32",
            "byte_order": "little",
            "axis_order": "zyx",
            "value_domain": "signed",
        },
        "compression": {"absolute_error_bound": ERROR_BOUND},
        "topology": {
            "persistence_threshold": 0.06,
            "match_radius": 0,
            "neighborhood": "cube26",
        },
        "normalization": {"mode": "max_abs", "epsilon": 1.0e-12},
        "model": {
            "patch_size": list(shape),
            "base_channels": 2,
            "correction_scale": 0.75,
        },
        "training": {
            "epochs": 2,
            "batch_size": 1,
            "validation_batch_size": 1,
            "samples_per_epoch": 2,
            "validation_samples": 2,
            "num_workers": 0,
            "learning_rate": 0.0005,
            "weight_decay": 1.0e-5,
            "minimum_epochs": 1,
            "early_stopping_patience": 2,
            "topology_warmup_start": 0,
            "topology_warmup_end": 1,
            "mixed_precision": False,
            "seed": seed,
            "device": "cpu",
        },
    }


def generate_dataset(
    output: Path,
    shape: tuple[int, int, int] = (8, 16, 16),
    seed: int = 2026,
) -> Path:
    """Write a complete deterministic manifest and return its path."""

    _validate_shape(shape)
    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f"output directory must be new or empty: {output}")
    volume_root = output / "volumes"
    label_root = output / "labels"
    volume_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    samples: list[dict[str, object]] = []
    for index, split in enumerate(("train", "validation", "test"), start=1):
        sample_id = f"sample-{index:03d}"
        reference = _analytic_field(shape, phase=0.35 * index)
        decompressed = _lossy_decompression_proxy(reference, rng)
        reference_path = volume_root / f"{sample_id}.reference.f32"
        decompressed_path = volume_root / f"{sample_id}.lossy.f32"
        reference.astype("<f4", copy=False).tofile(reference_path)
        decompressed.astype("<f4", copy=False).tofile(decompressed_path)
        false_cases, critical_points = _write_labels(label_root, sample_id, shape)
        samples.append(
            {
                "id": sample_id,
                "split": split,
                "decompressed": str(decompressed_path.relative_to(output)),
                "reference": str(reference_path.relative_to(output)),
                "false_cases": str(false_cases.relative_to(output)),
                "critical_points": str(critical_points.relative_to(output)),
            }
        )

    manifest = {
        "schema_version": 1,
        "experiment": "deterministic-analytic-example",
        "volume": {
            "shape": list(shape),
            "dtype": "float32",
            "byte_order": "little",
            "axis_order": "zyx",
        },
        "samples": samples,
    }
    manifest_path = output / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest, sort_keys=False), encoding="utf-8")
    (output / "experiment.yaml").write_text(
        yaml.safe_dump(_configuration(shape, seed), sort_keys=False),
        encoding="utf-8",
    )
    return manifest_path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shape", nargs=3, type=int, default=(8, 16, 16), metavar=("D", "H", "W"))
    parser.add_argument("--seed", type=int, default=2026)
    arguments = parser.parse_args()
    manifest = generate_dataset(arguments.output, tuple(arguments.shape), arguments.seed)
    print(manifest)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
