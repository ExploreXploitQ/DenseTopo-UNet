"""Gated residual 3D U-Net for dense topology restoration."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as functional
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DoubleConv(nn.Module):
    """Two shape-preserving convolutions with group normalization and SiLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class Down(nn.Module):
    """Downsample by two and refine features."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(nn.MaxPool3d(2), DoubleConv(in_channels, out_channels))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.block(inputs)


class Up(nn.Module):
    """Upsample, concatenate the encoder skip, and refine features."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(out_channels + skip_channels, out_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        upsampled = self.up(inputs)
        if upsampled.shape[2:] != skip.shape[2:]:
            upsampled = functional.interpolate(
                upsampled,
                size=skip.shape[2:],
                mode="trilinear",
                align_corners=False,
            )
        return self.conv(torch.cat((skip, upsampled), dim=1))


@dataclass(frozen=True)
class ModelOutput:
    """Restored scalar field and the two interpretable internal fields."""

    restored: torch.Tensor
    correction_ratio: torch.Tensor
    gate: torch.Tensor


class DenseTopoUNet3D(nn.Module):
    """One-channel, three-level gated residual 3D U-Net."""

    def __init__(
        self,
        base_channels: int = 12,
        correction_scale: float = 0.75,
        nonnegative: bool = False,
    ) -> None:
        super().__init__()
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        if correction_scale <= 0:
            raise ValueError("correction_scale must be positive")
        channels = int(base_channels)
        self.correction_scale = float(correction_scale)
        self.nonnegative = bool(nonnegative)

        self.in_conv = DoubleConv(1, channels)
        self.down1 = Down(channels, 2 * channels)
        self.down2 = Down(2 * channels, 4 * channels)
        self.down3 = Down(4 * channels, 8 * channels)
        self.up2 = Up(8 * channels, 4 * channels, 4 * channels)
        self.up1 = Up(4 * channels, 2 * channels, 2 * channels)
        self.up0 = Up(2 * channels, channels, channels)
        self.head = nn.Conv3d(channels, 2, kernel_size=1)
        nn.init.zeros_(self.head.weight)
        assert self.head.bias is not None
        nn.init.zeros_(self.head.bias)

    @staticmethod
    def _validate_inputs(normalized_input: torch.Tensor, decompressed: torch.Tensor) -> None:
        if normalized_input.ndim != 5 or decompressed.ndim != 5:
            raise ValueError("model inputs must have shape [B, 1, D, H, W]")
        if normalized_input.shape[1] != 1 or decompressed.shape[1] != 1:
            raise ValueError("DenseTopo-UNet requires exactly one input channel")
        if normalized_input.shape != decompressed.shape:
            raise ValueError("normalized_input and decompressed must have identical shapes")
        if any(dimension < 8 for dimension in normalized_input.shape[2:]):
            raise ValueError("each spatial input dimension must be at least 8")

    def forward(
        self,
        normalized_input: torch.Tensor,
        decompressed: torch.Tensor,
        xi: float,
    ) -> ModelOutput:
        """Restore one batch while bounding the learned residual by `xi`."""

        self._validate_inputs(normalized_input, decompressed)
        if xi <= 0:
            raise ValueError("xi must be positive")

        encoder0 = self.in_conv(normalized_input)
        encoder1 = self.down1(encoder0)
        encoder2 = self.down2(encoder1)
        bottleneck = self.down3(encoder2)
        decoded = self.up2(bottleneck, encoder2)
        decoded = self.up1(decoded, encoder1)
        decoded = self.up0(decoded, encoder0)

        correction_logit, gate_logit = self.head(decoded).chunk(2, dim=1)
        gate = torch.sigmoid(gate_logit)
        correction_ratio = self.correction_scale * torch.tanh(correction_logit) * gate
        restored = decompressed + float(xi) * correction_ratio
        if self.nonnegative:
            restored = torch.clamp_min(restored, 0.0)
        return ModelOutput(restored, correction_ratio, gate)
