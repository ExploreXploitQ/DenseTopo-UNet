from pathlib import Path

import numpy as np
import pytest

from densetopo_unet.checkpoint import load_checkpoint
from densetopo_unet.config import (
    CompressionConfig,
    ExperimentConfig,
    ModelConfig,
    NormalizationConfig,
    TopologyConfig,
    TrainingConfig,
    VolumeConfig,
)
from densetopo_unet.engine import topology_warmup, train
from densetopo_unet.manifest import load_manifest


def write_training_fixture(root: Path) -> tuple[ExperimentConfig, Path]:
    shape = (8, 8, 8)
    samples = []
    for index, split in enumerate(("train", "validation"), start=1):
        reference = np.linspace(0.01, 1.0, np.prod(shape), dtype=np.float32).reshape(shape)
        decompressed = reference + np.float32(index * 1.0e-5)
        decompressed.astype("<f4").tofile(root / f"input-{index}.f32")
        reference.astype("<f4").tofile(root / f"reference-{index}.f32")
        (root / f"false-{index}.csv").write_text(
            "case,z,y,x\nfn,3,3,3\n",
            encoding="utf-8",
        )
        (root / f"critical-{index}.csv").write_text(
            "critical_type,z,y,x\nlocal_maximum,3,3,3\n",
            encoding="utf-8",
        )
        samples.append(
            f"""  - id: sample-{index}
    split: {split}
    decompressed: input-{index}.f32
    reference: reference-{index}.f32
    false_cases: false-{index}.csv
    critical_points: critical-{index}.csv
"""
        )
    manifest_path = root / "manifest.yaml"
    manifest_path.write_text(
        """schema_version: 1
experiment: engine-fixture
volume: {shape: [8, 8, 8], dtype: float32, byte_order: little, axis_order: zyx}
samples:
"""
        + "".join(samples),
        encoding="utf-8",
    )
    config = ExperimentConfig(
        volume=VolumeConfig(shape=shape, value_domain="nonnegative"),
        compression=CompressionConfig(absolute_error_bound=1.0e-4),
        topology=TopologyConfig(persistence_threshold=0.06),
        normalization=NormalizationConfig(mode="positive_max"),
        model=ModelConfig(patch_size=shape, base_channels=2, correction_scale=0.75),
        training=TrainingConfig(
            epochs=1,
            batch_size=1,
            validation_batch_size=1,
            samples_per_epoch=1,
            validation_samples=1,
            num_workers=0,
            learning_rate=1.0e-3,
            weight_decay=1.0e-5,
            minimum_epochs=1,
            early_stopping_patience=1,
            topology_warmup_start=0,
            topology_warmup_end=1,
            mixed_precision=False,
            seed=11,
            device="cpu",
        ),
    )
    return config, manifest_path


def test_topology_warmup_is_piecewise_linear() -> None:
    assert topology_warmup(10, 20, 120) == 0.0
    assert topology_warmup(70, 20, 120) == pytest.approx(0.5)
    assert topology_warmup(120, 20, 120) == 1.0


@pytest.mark.integration
def test_one_epoch_cpu_training_writes_reproducible_artifacts(tmp_path: Path) -> None:
    config, manifest_path = write_training_fixture(tmp_path)
    manifest = load_manifest(manifest_path, config.volume)
    output = tmp_path / "run"

    summary = train(config, manifest, output)

    assert summary.stopped_epoch == 1
    assert summary.best_epoch == 1
    assert (output / "resolved_config.json").is_file()
    assert (output / "environment.json").is_file()
    assert (output / "history.csv").is_file()
    assert (output / "best.pt").is_file()
    assert (output / "latest.pt").is_file()
    assert (output / "training_summary.json").is_file()
    checkpoint = load_checkpoint(output / "best.pt")
    assert checkpoint.epoch == 1
    assert checkpoint.model_state["head.weight"].count_nonzero().item() > 0
