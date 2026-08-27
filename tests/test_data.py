from pathlib import Path

import numpy as np
import pytest
import torch
from conftest import write_manifest

from densetopo_unet.config import (
    CompressionConfig,
    ExperimentConfig,
    ModelConfig,
    NormalizationConfig,
    TopologyConfig,
    TrainingConfig,
    VolumeConfig,
)
from densetopo_unet.data import TopologyPatchDataset, normalization_scale
from densetopo_unet.manifest import load_manifest


def experiment_config(shape: tuple[int, int, int] = (8, 16, 16)) -> ExperimentConfig:
    return ExperimentConfig(
        volume=VolumeConfig(shape=shape),
        compression=CompressionConfig(absolute_error_bound=1.0e-4),
        topology=TopologyConfig(persistence_threshold=0.06),
        normalization=NormalizationConfig(mode="max_abs"),
        model=ModelConfig(patch_size=shape, base_channels=4, correction_scale=0.75),
        training=TrainingConfig(
            epochs=2,
            batch_size=1,
            validation_batch_size=1,
            samples_per_epoch=4,
            validation_samples=2,
            num_workers=0,
            minimum_epochs=1,
            early_stopping_patience=2,
            topology_warmup_start=0,
            topology_warmup_end=1,
            mixed_precision=False,
            device="cpu",
        ),
    )


def test_max_abs_normalization_supports_signed_values() -> None:
    volume = np.array([-4.0, 0.0, 2.0], dtype=np.float32)

    assert normalization_scale(volume, "max_abs", 1.0e-12) == pytest.approx(4.0)


def test_positive_max_normalization_preserves_zero_anchor() -> None:
    volume = np.array([-1.0, 0.0, 2.0], dtype=np.float32)

    assert normalization_scale(volume, "positive_max", 1.0e-12) == pytest.approx(2.0)


def test_normalization_rejects_zero_scale() -> None:
    with pytest.raises(ValueError, match="normalization scale"):
        normalization_scale(np.zeros(8, dtype=np.float32), "max_abs", 1.0e-12)


def test_same_seed_and_epoch_select_same_patch(tmp_path: Path) -> None:
    config = experiment_config()
    manifest = load_manifest(write_manifest(tmp_path, config.volume.shape), config.volume)
    left = TopologyPatchDataset(manifest, config, "train", 4, seed=17, augment=False)
    right = TopologyPatchDataset(manifest, config, "train", 4, seed=17, augment=False)

    left_sample = left[0]
    right_sample = right[0]

    torch.testing.assert_close(left_sample["input"], right_sample["input"])
    torch.testing.assert_close(left_sample["topo_weight"], right_sample["topo_weight"])


def test_patch_sample_has_one_channel_and_supervised_topology(tmp_path: Path) -> None:
    config = experiment_config()
    manifest = load_manifest(write_manifest(tmp_path, config.volume.shape), config.volume)
    dataset = TopologyPatchDataset(manifest, config, "train", 4, seed=17, augment=False)

    sample = dataset[0]

    assert sample["input"].shape == (1, 8, 16, 16)
    assert sample["decompressed"].shape == (1, 8, 16, 16)
    assert sample["target"].shape == (1, 8, 16, 16)
    assert sample["topo_weight"].shape == (1, 8, 16, 16)
    assert sample["sample_id"] == "sample-001"
    assert torch.max(sample["topo_weight"]).item() == 5.0
