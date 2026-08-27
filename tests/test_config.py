from pathlib import Path

import pytest

from densetopo_unet.config import ConfigError, load_experiment_config


VALID_CONFIG = """
volume:
  shape: [16, 24, 32]
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
  patch_size: [8, 16, 16]
  base_channels: 4
  correction_scale: 0.75
training:
  epochs: 2
  batch_size: 2
  validation_batch_size: 2
  samples_per_epoch: 8
  validation_samples: 4
  num_workers: 0
  learning_rate: 0.0002
  weight_decay: 0.00001
  minimum_epochs: 1
  early_stopping_patience: 2
  topology_warmup_start: 0
  topology_warmup_end: 1
  mixed_precision: false
  seed: 7
  device: cpu
"""


def write_config(tmp_path: Path, text: str = VALID_CONFIG) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def test_load_experiment_config_resolves_valid_3d_settings(tmp_path: Path) -> None:
    config = load_experiment_config(write_config(tmp_path))

    assert config.volume.shape == (16, 24, 32)
    assert config.model.patch_size == (8, 16, 16)
    assert config.compression.absolute_error_bound == pytest.approx(1.0e-4)
    assert config.training.device == "cpu"
    assert config.to_dict()["normalization"]["mode"] == "max_abs"


def test_config_rejects_patch_incompatible_with_three_downsamplings(
    tmp_path: Path,
) -> None:
    invalid = VALID_CONFIG.replace("patch_size: [8, 16, 16]", "patch_size: [7, 16, 16]")

    with pytest.raises(ConfigError, match="divisible by 8"):
        load_experiment_config(write_config(tmp_path, invalid))


def test_config_rejects_unknown_keys(tmp_path: Path) -> None:
    invalid = VALID_CONFIG.replace("  seed: 7", "  seed: 7\n  hidden_option: true")

    with pytest.raises(ConfigError, match="unknown keys.*hidden_option"):
        load_experiment_config(write_config(tmp_path, invalid))
