import dataclasses
from pathlib import Path

import numpy as np
import pytest

from densetopo_unet.checkpoint import (
    CHECKPOINT_SCHEMA_VERSION,
    CheckpointState,
    save_checkpoint_atomic,
)
from densetopo_unet.inference import InferenceRequest, restore_file
from densetopo_unet.model import DenseTopoUNet3D
from densetopo_unet.reproducibility import capture_rng_state


def write_identity_checkpoint(path: Path) -> None:
    model = DenseTopoUNet3D(base_channels=2, correction_scale=0.75, nonnegative=False)
    state = CheckpointState(
        schema_version=CHECKPOINT_SCHEMA_VERSION,
        package_version="0.1.0",
        config={
            "volume": {
                "shape": [8, 8, 8],
                "dtype": "float32",
                "byte_order": "little",
                "axis_order": "zyx",
                "value_domain": "signed",
            },
            "compression": {"absolute_error_bound": 0.0001},
            "topology": {
                "persistence_threshold": 0.06,
                "match_radius": 0,
                "neighborhood": "cube26",
            },
            "normalization": {"mode": "max_abs", "epsilon": 1.0e-12},
            "model": {
                "patch_size": [8, 8, 8],
                "base_channels": 2,
                "correction_scale": 0.75,
            },
            "training": {},
        },
        manifest_sha256="b" * 64,
        model_state=model.state_dict(),
        optimizer_state={},
        scheduler_state={},
        scaler_state=None,
        epoch=1,
        best_score=0.1,
        bad_epochs=0,
        history=[],
        rng_state=capture_rng_state(),
    )
    save_checkpoint_atomic(path, state)


def test_inference_request_exposes_no_reference_or_topology_fields() -> None:
    names = {field.name for field in dataclasses.fields(InferenceRequest)}

    assert "reference" not in names
    assert "false_cases" not in names
    assert "critical_points" not in names
    assert names == {
        "checkpoint",
        "input_path",
        "output_path",
        "shape",
        "byte_order",
        "batch_size",
        "device",
    }


def test_identity_checkpoint_restores_one_raw_input(tmp_path: Path) -> None:
    checkpoint = tmp_path / "identity.pt"
    write_identity_checkpoint(checkpoint)
    input_path = tmp_path / "input.f32"
    values = np.linspace(-1.0, 1.0, 512, dtype=np.float32).reshape(8, 8, 8)
    values.astype("<f4").tofile(input_path)
    output_path = tmp_path / "restored.f32"

    record = restore_file(
        InferenceRequest(
            checkpoint=checkpoint,
            input_path=input_path,
            output_path=output_path,
            shape=(8, 8, 8),
            byte_order="little",
            batch_size=2,
            device="cpu",
        )
    )

    restored = np.fromfile(output_path, dtype="<f4").reshape(8, 8, 8)
    np.testing.assert_array_equal(restored, values)
    assert record.shape == (8, 8, 8)
    assert record.input_sha256 != ""
    assert record.output_sha256 != ""
    assert (tmp_path / "restored.f32.json").is_file()


def test_inference_rejects_incorrect_input_byte_count(tmp_path: Path) -> None:
    checkpoint = tmp_path / "identity.pt"
    write_identity_checkpoint(checkpoint)
    input_path = tmp_path / "short.f32"
    np.zeros(10, dtype="<f4").tofile(input_path)

    with pytest.raises(ValueError, match="expected 2048 bytes"):
        restore_file(
            InferenceRequest(
                checkpoint,
                input_path,
                tmp_path / "output.f32",
                (8, 8, 8),
                "little",
                1,
                "cpu",
            )
        )
