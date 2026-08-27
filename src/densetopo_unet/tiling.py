"""Center-core context tiling for complete three-dimensional inference."""

from __future__ import annotations

import itertools
import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class AxisPlan:
    """Padding and patch starts for one spatial axis."""

    left_pad: int
    right_pad: int
    core: int
    starts: tuple[int, ...]


def make_axis_plan(length: int, patch: int) -> AxisPlan:
    """Plan 50-percent center cores that cover an axis exactly once."""

    if length <= 0 or patch <= 0:
        raise ValueError("length and patch must be positive")
    if patch > length:
        raise ValueError("patch must not exceed the volume dimension")
    if patch % 4 != 0:
        raise ValueError("patch must be divisible by four for center-core tiling")
    halo = patch // 4
    core = patch - 2 * halo
    count = math.ceil(length / core)
    covered = count * core
    right_pad = halo + covered - length
    starts = tuple(index * core for index in range(count))
    return AxisPlan(halo, right_pad, core, starts)


def _validate_volume_and_patch(
    decompressed: np.ndarray,
    patch_size: Sequence[int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    if decompressed.ndim != 3:
        raise ValueError("decompressed must have shape [D, H, W]")
    shape = (
        int(decompressed.shape[0]),
        int(decompressed.shape[1]),
        int(decompressed.shape[2]),
    )
    if len(patch_size) != 3:
        raise ValueError("patch_size must contain [Dp, Hp, Wp]")
    patch = (int(patch_size[0]), int(patch_size[1]), int(patch_size[2]))
    for full, part in zip(shape, patch, strict=False):
        make_axis_plan(full, part)
    return shape, patch


@torch.inference_mode()
def restore_volume_tiled(
    model: nn.Module,
    decompressed: np.ndarray,
    input_scale: float,
    patch_size: tuple[int, int, int],
    device: torch.device,
    batch_size: int,
    xi: float,
) -> np.ndarray:
    """Restore a complete volume by copying each predicted center core once."""

    shape, patch = _validate_volume_and_patch(decompressed, patch_size)
    if not np.isfinite(input_scale) or input_scale <= 0:
        raise ValueError("input_scale must be finite and positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if xi <= 0:
        raise ValueError("xi must be positive")

    plans = tuple(make_axis_plan(full, part) for full, part in zip(shape, patch, strict=False))
    pad_width = tuple((plan.left_pad, plan.right_pad) for plan in plans)
    padded = np.pad(np.asarray(decompressed, dtype=np.float32), pad_width, mode="reflect")
    restored = np.empty(shape, dtype=np.float32)
    write_count = np.zeros(shape, dtype=np.uint8)

    jobs = list(
        itertools.product(
            enumerate(plans[0].starts),
            enumerate(plans[1].starts),
            enumerate(plans[2].starts),
        )
    )
    model.eval()
    for first in range(0, len(jobs), batch_size):
        batch_jobs = jobs[first : first + batch_size]
        patches = []
        for (_, patch_z), (_, patch_y), (_, patch_x) in batch_jobs:
            patches.append(
                padded[
                    patch_z : patch_z + patch[0],
                    patch_y : patch_y + patch[1],
                    patch_x : patch_x + patch[2],
                ]
            )
        decompressed_tensor = torch.from_numpy(np.stack(patches)).unsqueeze(1).to(device)
        normalized_tensor = decompressed_tensor / float(input_scale)
        prediction = model(normalized_tensor, decompressed_tensor, xi=xi).restored
        prediction_array = prediction[:, 0].float().cpu().numpy()

        for local_index, job in enumerate(batch_jobs):
            axis_indices = tuple(item[0] for item in job)
            output_starts = tuple(
                index * plan.core for index, plan in zip(axis_indices, plans, strict=False)
            )
            output_stops = tuple(
                min(full, start + plan.core)
                for full, start, plan in zip(shape, output_starts, plans, strict=False)
            )
            lengths = tuple(
                stop - start for start, stop in zip(output_starts, output_stops, strict=False)
            )
            output_slices = tuple(
                slice(start, stop) for start, stop in zip(output_starts, output_stops, strict=False)
            )
            source_slices = tuple(
                slice(plan.left_pad, plan.left_pad + length)
                for plan, length in zip(plans, lengths, strict=False)
            )
            restored[output_slices] = prediction_array[local_index][source_slices]
            write_count[output_slices] += 1

    if not np.all(write_count == 1):
        minimum = int(write_count.min())
        maximum = int(write_count.max())
        raise RuntimeError(
            "internal tiling error: output write count must be one, "
            f"observed [{minimum}, {maximum}]"
        )
    return restored
