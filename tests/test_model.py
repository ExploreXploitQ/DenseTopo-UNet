import pytest
import torch

from densetopo_unet.model import DenseTopoUNet3D


def test_model_returns_three_same_shape_fields_and_starts_as_identity() -> None:
    model = DenseTopoUNet3D(base_channels=4, correction_scale=0.75, nonnegative=False)
    normalized = torch.randn(2, 1, 8, 16, 16)
    decompressed = torch.randn_like(normalized)

    output = model(normalized, decompressed, xi=1.0e-4)

    assert output.restored.shape == normalized.shape
    assert output.correction_ratio.shape == normalized.shape
    assert output.gate.shape == normalized.shape
    torch.testing.assert_close(output.restored, decompressed)


def test_correction_is_bounded_by_scale_times_error_bound() -> None:
    model = DenseTopoUNet3D(base_channels=4, correction_scale=0.25, nonnegative=False)
    with torch.no_grad():
        model.head.bias[:] = torch.tensor([20.0, 20.0])
    values = torch.ones(1, 1, 8, 8, 8)

    output = model(values, values, xi=2.0e-4)

    assert torch.max(torch.abs(output.correction_ratio)).item() <= 0.25
    assert torch.max(torch.abs(output.restored - values)).item() <= 5.01e-5


def test_nonnegative_domain_clamps_negative_restoration() -> None:
    model = DenseTopoUNet3D(base_channels=4, correction_scale=0.75, nonnegative=True)
    decompressed = torch.full((1, 1, 8, 8, 8), -1.0e-5)

    output = model(torch.zeros_like(decompressed), decompressed, xi=1.0e-4)

    assert torch.count_nonzero(output.restored).item() == 0


def test_model_rejects_non_scalar_channel_input() -> None:
    model = DenseTopoUNet3D(base_channels=4, correction_scale=0.75, nonnegative=False)
    values = torch.zeros(1, 2, 8, 8, 8)

    with pytest.raises(ValueError, match="one input channel"):
        model(values, values, xi=1.0e-4)


def test_model_backward_produces_finite_gradients() -> None:
    model = DenseTopoUNet3D(base_channels=4, correction_scale=0.75, nonnegative=False)
    normalized = torch.randn(1, 1, 8, 8, 8)
    decompressed = torch.randn_like(normalized)

    model(normalized, decompressed, xi=1.0e-4).restored.sum().backward()

    gradients = [parameter.grad for parameter in model.parameters() if parameter.grad is not None]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
