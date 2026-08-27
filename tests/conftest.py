from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from densetopo_unet.config import VolumeConfig


@pytest.fixture
def volume_config() -> VolumeConfig:
    return VolumeConfig(
        shape=(4, 8, 8),
        dtype="float32",
        byte_order="little",
        axis_order="zyx",
        value_domain="signed",
    )


def write_raw(path: Path, shape: tuple[int, int, int], offset: float = 0.0) -> None:
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 100.0
    (values + offset).astype("<f4").tofile(path)


def write_label_files(root: Path) -> tuple[Path, Path]:
    false_cases = root / "false-cases.csv"
    false_cases.write_text("case,z,y,x\nfn,1,2,3\nfp,2,3,4\n", encoding="utf-8")
    critical_points = root / "critical-points.csv"
    critical_points.write_text(
        "critical_type,z,y,x\nlocal_maximum,1,2,3\nlocal_minimum,2,3,4\n",
        encoding="utf-8",
    )
    return false_cases, critical_points


def write_manifest(root: Path, shape: tuple[int, int, int] = (4, 8, 8)) -> Path:
    data_dir = root / "data"
    labels_dir = root / "labels"
    data_dir.mkdir()
    labels_dir.mkdir()
    for name, offset in (("input.f32", 0.01), ("reference.f32", 0.0)):
        write_raw(data_dir / name, shape, offset)
    false_cases, critical_points = write_label_files(labels_dir)
    manifest = root / "manifest.yaml"
    manifest.write_text(
        f"""
schema_version: 1
experiment: fixture
volume:
  shape: [{shape[0]}, {shape[1]}, {shape[2]}]
  dtype: float32
  byte_order: little
  axis_order: zyx
samples:
  - id: sample-001
    split: train
    decompressed: data/input.f32
    reference: data/reference.f32
    false_cases: labels/{false_cases.name}
    critical_points: labels/{critical_points.name}
""",
        encoding="utf-8",
    )
    return manifest
