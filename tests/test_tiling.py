import numpy as np
import pytest
import torch
from torch import nn

from densetopo_unet.model import ModelOutput
from densetopo_unet.tiling import make_axis_plan, restore_volume_tiled


class IdentityEditor(nn.Module):
    def forward(
        self, normalized_input: torch.Tensor, decompressed: torch.Tensor, xi: float
    ) -> ModelOutput:
        del normalized_input, xi
        return ModelOutput(
            restored=decompressed,
            correction_ratio=torch.zeros_like(decompressed),
            gate=torch.zeros_like(decompressed),
        )


@pytest.mark.parametrize("length,patch", [(8, 8), (9, 8), (17, 8), (31, 16)])
def test_axis_plan_covers_every_output_index_once(length: int, patch: int) -> None:
    plan = make_axis_plan(length, patch)
    coverage = np.zeros(length, dtype=np.int32)

    for index, _ in enumerate(plan.starts):
        output_start = index * plan.core
        output_stop = min(length, output_start + plan.core)
        coverage[output_start:output_stop] += 1

    assert np.all(coverage == 1)
    assert plan.left_pad == patch // 4
    assert plan.left_pad + length + plan.right_pad >= plan.starts[-1] + patch


@pytest.mark.parametrize("shape", [(8, 16, 16), (9, 17, 19)])
def test_identity_tiling_round_trip_for_divisible_and_nondivisible_shapes(
    shape: tuple[int, int, int],
) -> None:
    values = np.arange(np.prod(shape), dtype=np.float32).reshape(shape) / 1000.0

    restored = restore_volume_tiled(
        model=IdentityEditor(),
        decompressed=values,
        input_scale=float(np.max(values)),
        patch_size=(8, 8, 8),
        device=torch.device("cpu"),
        batch_size=3,
        xi=1.0e-4,
    )

    np.testing.assert_array_equal(restored, values)


def test_tiling_rejects_nonpositive_input_scale() -> None:
    with pytest.raises(ValueError, match="input_scale"):
        restore_volume_tiled(
            IdentityEditor(),
            np.ones((8, 8, 8), dtype=np.float32),
            0.0,
            (8, 8, 8),
            torch.device("cpu"),
            1,
            1.0e-4,
        )
