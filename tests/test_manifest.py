from pathlib import Path

import numpy as np
import pytest
from conftest import write_manifest

from densetopo_unet.config import VolumeConfig
from densetopo_unet.io import open_raw_volume, validate_raw_volume
from densetopo_unet.manifest import (
    ManifestError,
    load_false_cases,
    load_manifest,
)


def test_manifest_resolves_paths_relative_to_manifest(
    tmp_path: Path, volume_config: VolumeConfig
) -> None:
    manifest = load_manifest(write_manifest(tmp_path), volume_config)

    assert manifest.experiment == "fixture"
    assert manifest.samples[0].decompressed == (tmp_path / "data/input.f32").resolve()
    assert manifest.samples[0].reference == (tmp_path / "data/reference.f32").resolve()
    assert manifest.by_split("train")[0].sample_id == "sample-001"


def test_validate_raw_volume_rejects_wrong_byte_count(
    tmp_path: Path, volume_config: VolumeConfig
) -> None:
    path = tmp_path / "short.f32"
    np.zeros(31, dtype="<f4").tofile(path)

    with pytest.raises(ValueError, match=r"expected 1024 bytes, observed 124"):
        validate_raw_volume(path, volume_config)


def test_open_raw_volume_preserves_zyx_shape(tmp_path: Path, volume_config: VolumeConfig) -> None:
    path = tmp_path / "volume.f32"
    np.arange(256, dtype="<f4").tofile(path)

    volume = open_raw_volume(path, volume_config)

    assert volume.shape == (4, 8, 8)
    assert float(volume[3, 7, 7]) == 255.0


def test_manifest_rejects_missing_training_labels(
    tmp_path: Path, volume_config: VolumeConfig
) -> None:
    path = write_manifest(tmp_path)
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("    critical_points: labels/critical-points.csv\n", ""),
        encoding="utf-8",
    )

    with pytest.raises(ManifestError, match=r"critical_points.*required"):
        load_manifest(path, volume_config)


def test_false_case_loader_rejects_out_of_bounds_coordinate(tmp_path: Path) -> None:
    path = tmp_path / "false.csv"
    path.write_text("case,z,y,x\nfn,4,0,0\n", encoding="utf-8")

    with pytest.raises(ManifestError, match="outside shape"):
        load_false_cases(path, (4, 8, 8))
