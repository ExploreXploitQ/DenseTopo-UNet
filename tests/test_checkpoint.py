from pathlib import Path

import numpy as np
import pytest
import torch

from densetopo_unet.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointError,
    CheckpointState,
    load_checkpoint,
    manifest_fingerprint,
    save_checkpoint_atomic,
)
from densetopo_unet.reproducibility import (
    capture_rng_state,
    restore_rng_state,
    seed_everything,
)


def checkpoint_state() -> CheckpointState:
    return CheckpointState(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        package_version="0.1.0",
        config={
            "model": {"base_channels": 4, "patch_size": [8, 8, 8], "correction_scale": 0.75},
            "compression": {"absolute_error_bound": 0.0001},
            "normalization": {"mode": "max_abs", "epsilon": 1.0e-12},
            "volume": {"value_domain": "signed", "shape": [8, 8, 8]},
        },
        manifest_sha256="a" * 64,
        model_state={"weight": torch.tensor([1.0, 2.0])},
        optimizer_state={"step": 3},
        scheduler_state={"best": 0.25},
        scaler_state=None,
        epoch=3,
        best_score=0.25,
        bad_epochs=1,
        history=[{"epoch": 3.0, "validation_total": 0.25}],
        rng_state=capture_rng_state(),
    )


def test_checkpoint_round_trip_preserves_model_and_metadata(tmp_path: Path) -> None:
    state = checkpoint_state()
    path = tmp_path / "model.pt"

    save_checkpoint_atomic(path, state)
    loaded = load_checkpoint(path)

    assert loaded.schema_version == 1
    assert loaded.epoch == 3
    assert loaded.best_score == pytest.approx(0.25)
    assert loaded.manifest_sha256 == state.manifest_sha256
    torch.testing.assert_close(loaded.model_state["weight"], torch.tensor([1.0, 2.0]))
    assert list(tmp_path.glob("*.tmp")) == []


def test_checkpoint_rejects_incompatible_expected_architecture(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    save_checkpoint_atomic(path, checkpoint_state())
    incompatible = checkpoint_state().config.copy()
    incompatible["model"] = {
        "base_channels": 8,
        "patch_size": [8, 8, 8],
        "correction_scale": 0.75,
    }

    with pytest.raises(CheckpointError, match="model configuration"):
        load_checkpoint(path, expected_config=incompatible)


def test_rng_state_restore_reproduces_numpy_and_torch_draws() -> None:
    seed_everything(42)
    state = capture_rng_state()
    expected_numpy = np.random.random(3)
    expected_torch = torch.rand(3)

    restore_rng_state(state)

    np.testing.assert_array_equal(np.random.random(3), expected_numpy)
    torch.testing.assert_close(torch.rand(3), expected_torch)


def test_manifest_fingerprint_changes_with_content(tmp_path: Path) -> None:
    path = tmp_path / "manifest.yaml"
    path.write_text("schema_version: 1\n", encoding="utf-8")
    first = manifest_fingerprint(path)
    path.write_text("schema_version: 2\n", encoding="utf-8")

    assert manifest_fingerprint(path) != first
