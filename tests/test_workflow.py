from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

from densetopo_unet.cli import main
from densetopo_unet.config import load_experiment_config
from densetopo_unet.manifest import load_manifest
from scripts.generate_synthetic_data import generate_dataset


def test_synthetic_generator_writes_complete_training_contract(tmp_path: Path) -> None:
    manifest_path = generate_dataset(tmp_path / "first", shape=(8, 16, 16), seed=2026)
    repeated_path = generate_dataset(tmp_path / "second", shape=(8, 16, 16), seed=2026)

    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    assert {sample["split"] for sample in raw["samples"]} == {
        "train",
        "validation",
        "test",
    }
    for sample in raw["samples"]:
        for key in ("decompressed", "reference", "false_cases", "critical_points"):
            assert (manifest_path.parent / sample[key]).is_file()

    config = load_experiment_config(manifest_path.parent / "experiment.yaml")
    manifest = load_manifest(manifest_path, config.volume)
    assert len(manifest.samples) == 3
    assert config.compression.absolute_error_bound == pytest.approx(0.02)

    first_files = {
        path.relative_to(manifest_path.parent): path.read_bytes()
        for path in manifest_path.parent.rglob("*")
        if path.is_file()
    }
    second_files = {
        path.relative_to(repeated_path.parent): path.read_bytes()
        for path in repeated_path.parent.rglob("*")
        if path.is_file()
    }
    assert first_files == second_files


@pytest.mark.integration
def test_cpu_training_and_single_file_inference_workflow(tmp_path: Path) -> None:
    root = tmp_path / "synthetic"
    manifest_path = generate_dataset(root, shape=(8, 16, 16), seed=7)
    config_path = root / "experiment.yaml"
    run_path = tmp_path / "run"

    assert main(
        [
            "train",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
            "--output",
            str(run_path),
        ]
    ) == 0

    raw = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    test_sample = next(sample for sample in raw["samples"] if sample["split"] == "test")
    input_path = root / test_sample["decompressed"]
    restored_path = tmp_path / "restored.f32"
    assert main(
        [
            "infer",
            "--checkpoint",
            str(run_path / "best.pt"),
            "--input",
            str(input_path),
            "--shape",
            "8",
            "16",
            "16",
            "--output",
            str(restored_path),
            "--device",
            "cpu",
        ]
    ) == 0

    restored = np.fromfile(restored_path, dtype="<f4")
    assert restored.size == 8 * 16 * 16
    assert np.isfinite(restored).all()
    assert restored_path.with_suffix(".f32.json").is_file()
