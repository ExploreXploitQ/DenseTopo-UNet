# Architecture

## Problem setting

DenseTopo-UNet learns a correction from a lossy-decompressed scalar field to a reference field. The model is trained with numerical and topology-derived supervision, but deployment remains a one-file problem: only one decompressed volume and a user-trained checkpoint are available.

![DenseTopo-UNet deployment and training information flow](../assets/architecture.svg)

## Tensors and channels

For a patch size `[Dp, Hp, Wp]`, the primary tensors are:

| Tensor | Shape | Meaning |
| --- | --- | --- |
| `normalized_input` | `[B, 1, Dp, Hp, Wp]` | Decompressed scalar values divided by one per-volume scale. |
| `decompressed` | `[B, 1, Dp, Hp, Wp]` | Unnormalized decompressed values used as the residual anchor. |
| `target` | `[B, 1, Dp, Hp, Wp]` | Reference values; training and evaluation only. |
| `topo_weight` | `[B, 1, Dp, Hp, Wp]` | Sparse FC and extrema weights; training only. |
| `restored` | `[B, 1, Dp, Hp, Wp]` | Corrected scalar values. |
| `correction_ratio` | `[B, 1, Dp, Hp, Wp]` | Final signed, gated correction expressed in error-bound units. |
| `gate` | `[B, 1, Dp, Hp, Wp]` | A value in `[0,1]` that controls correction density. |

The input has exactly one channel. This channel is the scalar value at each voxel. Reference values, FC masks, coordinates, error maps, gradients, and persistence values are not input channels.

Feature channels are internal learned arrays. With `base_channels: 12`, widths are `12 -> 24 -> 48 -> 96 -> 48 -> 24 -> 12`. Individual feature channels do not correspond to named physical variables; training determines which local shapes, gradients, or contextual patterns they respond to.

## 3D U-Net

The encoder has a shape-preserving double-convolution block followed by three levels of `2x2x2` max pooling. Every double-convolution block applies `3x3x3 Conv3d -> GroupNorm -> SiLU` twice. Group normalization avoids dependence on large batch statistics, which is useful when 3D memory limits require small batches.

The decoder uses three stride-two transposed convolutions. Each upsampled feature tensor is concatenated with the matching encoder skip tensor and refined by another double-convolution block. If odd intermediate dimensions disagree, trilinear interpolation aligns the decoder tensor to the skip shape. Public patch dimensions must be divisible by eight, so this alignment is normally exact.

## Bounded gated residual

The final `1x1x1` convolution produces two logits:

- `c`, transformed by `tanh`, proposes a signed correction;
- `a`, transformed by `sigmoid`, opens or closes the spatial gate.

For decompressed value `d`, absolute error bound `xi`, and `correction_scale` `s`, the model calculates:

```text
gate             = sigmoid(a)
correction_ratio = s * tanh(c) * gate
restored         = d + xi * correction_ratio
```

Thus the learned edit relative to `d` is bounded in magnitude by `xi * s`. This bound limits the neural edit; it does not prove that `restored` lies within `xi` of the unknown reference. If `value_domain` is `nonnegative`, the final value is additionally clamped at zero.

The output head starts at zero weights and biases. The initial correction is therefore zero, making the untrained model an identity mapping around the decompressed field.

## Training objective

The total objective is a weighted sum:

```text
total = reconstruction
      + w_gradient   * gradient
      + w_critical   * critical
      + w_topology   * schedule * topology_order
      + w_gate       * gate_supervision
      + w_error_bound * error_bound
      + w_correction * correction_regularization
```

`reconstruction` mixes error-bound-normalized MSE and Charbonnier loss. `gradient` compares finite differences along `z`, `y`, and `x`. `critical` applies stronger reconstruction loss at labeled FC voxels. `topology_order` checks whether a supervised center retains the reference ordering relative to its 26 immediate neighbors. `gate_supervision` encourages gates near false cases and discourages widespread background activation. `error_bound` penalizes errors beyond the configured bound and upweights the largest 0.1% errors. `correction_regularization` discourages large correction ratios and dense gates.

All scalar weights are explicit in the `loss` configuration section. The topology schedule increases linearly between `topology_warmup_start` and `topology_warmup_end`.

The 26-neighbor ordering loss is a differentiable local surrogate. It is not a persistence algorithm, does not construct a contour tree, and cannot guarantee exact FC removal.

## Patch selection and inference tiling

Each training draw first selects a manifest record. Patch centers are then sampled from false negatives with probability 0.35, false positives with probability 0.25, reference extrema with probability 0.20, or a random voxel with probability 0.20. Small random center jitter and independent spatial flips improve local variation. FT points receive loss weight 4 but are not a separate center-sampling branch; they remain supervised whenever a selected patch contains them.

Full-volume inference uses reflection-padded context patches. Each patch contributes its center core, whose dimensions are half the patch dimensions. Consequently, adjacent core starts use a 50% patch stride. Cores do not overlap in the output, including at non-divisible boundaries, and a runtime invariant checks that every voxel is written exactly once.
