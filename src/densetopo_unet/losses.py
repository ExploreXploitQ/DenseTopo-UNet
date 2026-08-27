"""Topology-aware objective for DenseTopo-UNet."""

from __future__ import annotations

import math
from dataclasses import dataclass, fields

import torch
import torch.nn.functional as functional

from densetopo_unet.model import ModelOutput


OFFSETS_26 = torch.tensor(
    [
        (z, y, x)
        for z in (-1, 0, 1)
        for y in (-1, 0, 1)
        for x in (-1, 0, 1)
        if (z, y, x) != (0, 0, 0)
    ],
    dtype=torch.long,
)


@dataclass(frozen=True)
class LossWeights:
    """Public weights for every term in the historical dense objective."""

    mse_mix: float = 0.7
    charbonnier_mix: float = 0.3
    gradient: float = 0.10
    critical: float = 10.0
    topology: float = 5.0
    gate: float = 0.20
    error_bound: float = 25.0
    correction: float = 0.005
    gate_negative: float = 0.02
    error_bound_tail: float = 5.0


@dataclass(frozen=True)
class LossBreakdown:
    """Differentiable total objective and named component tensors."""

    total: torch.Tensor
    reconstruction: torch.Tensor
    mse: torch.Tensor
    charbonnier: torch.Tensor
    gradient: torch.Tensor
    topology_order: torch.Tensor
    critical: torch.Tensor
    gate_supervision: torch.Tensor
    error_bound: torch.Tensor
    correction_regularization: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


def topology_order_loss(
    restored: torch.Tensor,
    target: torch.Tensor,
    topo_weight: torch.Tensor,
    xi: float,
) -> torch.Tensor:
    """Penalize wrong 26-neighbor ordering at supervised interior extrema."""

    if xi <= 0:
        raise ValueError("xi must be positive")
    if restored.shape != target.shape or restored.shape != topo_weight.shape:
        raise ValueError("restored, target, and topo_weight must have identical shapes")
    indices = torch.nonzero(topo_weight[:, 0] > 0, as_tuple=False)
    if indices.numel() == 0:
        return restored.sum() * 0.0

    _, _, depth, height, width = restored.shape
    interior = (
        (indices[:, 1] > 0)
        & (indices[:, 1] < depth - 1)
        & (indices[:, 2] > 0)
        & (indices[:, 2] < height - 1)
        & (indices[:, 3] > 0)
        & (indices[:, 3] < width - 1)
    )
    indices = indices[interior]
    if indices.numel() == 0:
        return restored.sum() * 0.0

    offsets = OFFSETS_26.to(device=restored.device)
    batch, z_coord, y_coord, x_coord = (indices[:, index] for index in range(4))
    neighbor_z = z_coord[:, None] + offsets[None, :, 0]
    neighbor_y = y_coord[:, None] + offsets[None, :, 1]
    neighbor_x = x_coord[:, None] + offsets[None, :, 2]
    neighbor_batch = batch[:, None].expand_as(neighbor_z)

    predicted_center = restored[batch, 0, z_coord, y_coord, x_coord][:, None]
    target_center = target[batch, 0, z_coord, y_coord, x_coord][:, None]
    predicted_difference = (
        predicted_center - restored[neighbor_batch, 0, neighbor_z, neighbor_y, neighbor_x]
    ) / float(xi)
    target_difference = (
        target_center - target[neighbor_batch, 0, neighbor_z, neighbor_y, neighbor_x]
    ) / float(xi)

    valid = target_difference.abs() > 1.0e-5
    margin = torch.clamp(0.5 * target_difference.abs(), min=1.0e-4, max=0.05)
    violation = functional.relu(
        margin - target_difference.sign() * predicted_difference
    )
    violation = torch.where(valid, violation, torch.zeros_like(violation))
    point_violation = violation.max(dim=1).values
    point_weight = topo_weight[batch, 0, z_coord, y_coord, x_coord]
    point_weight = point_weight * valid.any(dim=1)
    return (point_violation * point_weight).sum() / point_weight.sum().clamp_min(1.0)


def compute_losses(
    output: ModelOutput,
    target: torch.Tensor,
    decompressed: torch.Tensor,
    topo_weight: torch.Tensor,
    topology_lambda: float,
    weights: LossWeights,
    xi: float,
) -> LossBreakdown:
    """Compute the complete dense topology-restoration objective."""

    restored = output.restored
    if xi <= 0:
        raise ValueError("xi must be positive")
    if not 0.0 <= topology_lambda <= 1.0:
        raise ValueError("topology_lambda must be between zero and one")
    if restored.shape != target.shape or restored.shape != decompressed.shape:
        raise ValueError("restored, target, and decompressed must have identical shapes")
    if topo_weight.shape != restored.shape:
        raise ValueError("topo_weight must have the restored-field shape")

    normalized_error = (restored - target) / float(xi)
    signal = ((target != 0) | (decompressed != 0)).to(dtype=restored.dtype)
    topology_neighborhood = functional.max_pool3d(
        topo_weight, kernel_size=3, stride=1, padding=1
    )
    spatial_weight = 1.0 + signal + topology_neighborhood
    weight_sum = spatial_weight.sum().clamp_min(1.0)

    mse = (spatial_weight * normalized_error.square()).sum() / weight_sum
    charbonnier = (
        spatial_weight * torch.sqrt(normalized_error.square() + 1.0e-6)
    ).sum() / weight_sum
    reconstruction = weights.mse_mix * mse + weights.charbonnier_mix * charbonnier

    gradient_terms = []
    for axis in (2, 3, 4):
        predicted_gradient = torch.diff(restored, dim=axis) / float(xi)
        target_gradient = torch.diff(target, dim=axis) / float(xi)
        gradient_terms.append((predicted_gradient - target_gradient).abs().mean())
    gradient = torch.stack(gradient_terms).mean()

    topology = topology_order_loss(restored, target, topo_weight, xi)

    false_weight = torch.where(
        topo_weight > 1.0,
        topo_weight,
        torch.zeros_like(topo_weight),
    )
    false_weight_sum = false_weight.sum()
    if false_weight_sum.detach().item() > 0:
        critical_penalty = (
            weights.mse_mix * normalized_error.square()
            + weights.charbonnier_mix
            * torch.sqrt(normalized_error.square() + 1.0e-6)
        )
        critical = (critical_penalty * false_weight).sum() / false_weight_sum
    else:
        critical = restored.sum() * 0.0

    gate_target = functional.max_pool3d(
        (topo_weight > 1.0).to(dtype=restored.dtype),
        kernel_size=3,
        stride=1,
        padding=1,
    )
    positive = gate_target > 0
    negative = ~positive
    gate_safe = output.gate.float().clamp(1.0e-4, 1.0 - 1.0e-4)
    gate_positive = (
        -torch.log(gate_safe[positive]).mean()
        if positive.any().item()
        else restored.sum() * 0.0
    )
    gate_negative = (
        -torch.log1p(-gate_safe[negative]).mean()
        if negative.any().item()
        else restored.sum() * 0.0
    )
    gate_supervision = gate_positive + weights.gate_negative * gate_negative

    excess = functional.relu(normalized_error.abs() - 1.0)
    error_bound_mean = excess.square().mean()
    flattened = excess.flatten(1)
    tail_count = max(1, int(math.ceil(flattened.shape[1] * 0.001)))
    error_bound_tail = torch.topk(flattened, tail_count, dim=1).values.square().mean()
    error_bound = error_bound_mean + weights.error_bound_tail * error_bound_tail

    correction_regularization = (
        output.correction_ratio.abs().mean() + 0.1 * output.gate.mean()
    )
    total = (
        reconstruction
        + weights.gradient * gradient
        + weights.critical * critical
        + weights.topology * float(topology_lambda) * topology
        + weights.gate * gate_supervision
        + weights.error_bound * error_bound
        + weights.correction * correction_regularization
    )
    return LossBreakdown(
        total=total,
        reconstruction=reconstruction,
        mse=mse,
        charbonnier=charbonnier,
        gradient=gradient,
        topology_order=topology,
        critical=critical,
        gate_supervision=gate_supervision,
        error_bound=error_bound,
        correction_regularization=correction_regularization,
    )
