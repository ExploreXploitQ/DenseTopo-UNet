import pytest
import torch

from densetopo_unet.losses import LossWeights, compute_losses, topology_order_loss
from densetopo_unet.model import ModelOutput


def test_perfect_reconstruction_has_zero_error_and_gradient_losses() -> None:
    target = torch.ones(1, 1, 4, 4, 4)
    output = ModelOutput(target.clone(), torch.zeros_like(target), torch.zeros_like(target))

    losses = compute_losses(
        output=output,
        target=target,
        decompressed=target,
        topo_weight=torch.zeros_like(target),
        topology_lambda=1.0,
        weights=LossWeights(),
        xi=1.0e-4,
    )

    assert losses.mse.item() == pytest.approx(0.0)
    assert losses.gradient.item() == pytest.approx(0.0)
    assert losses.error_bound.item() == pytest.approx(0.0)


def test_error_bound_loss_penalizes_only_excess_error() -> None:
    target = torch.zeros(1, 1, 4, 4, 4)
    restored = target.clone()
    restored[..., 0, 0, 0] = 2.0e-4
    output = ModelOutput(restored, torch.zeros_like(target), torch.zeros_like(target))

    losses = compute_losses(
        output,
        target,
        target,
        torch.zeros_like(target),
        0.0,
        LossWeights(),
        1.0e-4,
    )

    assert losses.error_bound.item() > 0.0


def test_topology_order_loss_detects_reversed_center_neighbor_order() -> None:
    target = torch.zeros(1, 1, 5, 5, 5)
    target[0, 0, 2, 2, 2] = 1.0e-4
    restored = torch.zeros_like(target)
    restored[0, 0, 2, 2, 2] = -1.0e-4
    topo_weight = torch.zeros_like(target)
    topo_weight[0, 0, 2, 2, 2] = 5.0

    loss = topology_order_loss(restored, target, topo_weight, xi=1.0e-4)

    assert loss.item() > 0.0


def test_topology_order_loss_is_zero_for_empty_supervision() -> None:
    values = torch.randn(1, 1, 4, 4, 4)

    loss = topology_order_loss(values, values, torch.zeros_like(values), xi=1.0e-4)

    assert loss.item() == pytest.approx(0.0)


def test_total_loss_backward_produces_finite_gradients() -> None:
    target = torch.rand(1, 1, 4, 4, 4)
    restored = (target + 2.0e-5).detach().requires_grad_(True)
    correction = torch.full_like(target, 0.2, requires_grad=True)
    gate = torch.full_like(target, 0.6, requires_grad=True)
    output = ModelOutput(restored, correction, gate)

    losses = compute_losses(
        output,
        target,
        target + 1.0e-5,
        torch.zeros_like(target),
        1.0,
        LossWeights(),
        1.0e-4,
    )
    losses.total.backward()

    assert restored.grad is not None and torch.isfinite(restored.grad).all()
    assert correction.grad is not None and torch.isfinite(correction.grad).all()
    assert gate.grad is not None and torch.isfinite(gate.grad).all()
