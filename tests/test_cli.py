from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import write_manifest
from densetopo_unet.cli import build_parser, main


def write_config(path: Path) -> Path:
    path.write_text(
        """
volume:
  shape: [8, 8, 8]
  dtype: float32
  byte_order: little
  axis_order: zyx
  value_domain: signed
compression:
  absolute_error_bound: 0.0001
topology:
  persistence_threshold: 0.06
  match_radius: 0
  neighborhood: cube26
normalization:
  mode: max_abs
  epsilon: 1.0e-12
model:
  patch_size: [8, 8, 8]
  base_channels: 2
  correction_scale: 0.75
training:
  epochs: 1
  batch_size: 1
  validation_batch_size: 1
  samples_per_epoch: 1
  validation_samples: 1
  num_workers: 0
  learning_rate: 0.0002
  weight_decay: 0.00001
  minimum_epochs: 1
  early_stopping_patience: 1
  topology_warmup_start: 0
  topology_warmup_end: 1
  mixed_precision: false
  seed: 2026
  device: cpu
""".lstrip(),
        encoding="utf-8",
    )
    return path


def test_infer_help_enforces_deployment_information_boundary(capsys: pytest.CaptureFixture[str]) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit, match="0"):
        parser.parse_args(["infer", "--help"])

    help_text = capsys.readouterr().out
    assert "--input" in help_text
    assert "--shape" in help_text
    assert "--reference" not in help_text
    assert "--false-cases" not in help_text
    assert "--critical-points" not in help_text


def test_validate_manifest_prints_machine_readable_summary(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = write_config(tmp_path / "config.yaml")
    manifest_path = write_manifest(tmp_path, shape=(8, 8, 8))

    exit_code = main(
        [
            "validate-manifest",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
        ]
    )

    summary = json.loads(capsys.readouterr().out)
    assert exit_code == 0
    assert summary["valid"] is True
    assert summary["sample_count"] == 1
    assert summary["splits"] == {"test": 0, "train": 1, "validation": 0}


def test_cli_returns_nonzero_for_malformed_configuration(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    config_path = tmp_path / "broken.yaml"
    config_path.write_text("volume: []\n", encoding="utf-8")
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text("schema_version: 1\n", encoding="utf-8")

    exit_code = main(
        [
            "validate-manifest",
            "--config",
            str(config_path),
            "--manifest",
            str(manifest_path),
        ]
    )

    assert exit_code == 2
    assert "error:" in capsys.readouterr().err.lower()


def test_module_entry_point_displays_help() -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "densetopo_unet", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0
    assert "validate-manifest" in completed.stdout
    assert "inspect-checkpoint" in completed.stdout
