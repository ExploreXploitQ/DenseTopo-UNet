# Experiment configuration

DenseTopo-UNet uses one strict YAML configuration per training run. Unknown keys, missing keys, invalid choices, and incompatible dimensions are rejected. The resolved configuration is embedded in every checkpoint and copied to the run directory.

See [`configs/model.yaml`](../configs/model.yaml) for a complete generic example.

## `volume`

| Key | Type | Meaning |
| --- | --- | --- |
| `shape` | three positive integers | Full logical `[D, H, W]` dimensions shared by the manifest. |
| `dtype` | `float32` | Only supported scalar storage type. |
| `byte_order` | `little` or `big` | Raw input and output byte order. |
| `axis_order` | `zyx` | Only supported logical axis order. |
| `value_domain` | `signed` or `nonnegative` | Whether the final network output may be negative. |

## `compression`

| Key | Type | Meaning |
| --- | --- | --- |
| `absolute_error_bound` | positive float | `xi`, in original scalar units; scales model corrections and normalized losses. |

One run assumes one bound. The field is a scientific condition supplied by the user; the package does not infer or verify the settings used by the upstream compressor.

## `topology`

| Key | Type | Meaning |
| --- | --- | --- |
| `persistence_threshold` | positive float | Persistence used to generate external topology labels and summaries. |
| `match_radius` | nonnegative integer | External FC matching radius recorded for provenance. |
| `neighborhood` | `cube26` | Supported local neighborhood for the differentiable ordering surrogate. |

`persistence_threshold` and `match_radius` are not recomputed inside the neural loss. Label generation and final evaluation must apply them independently.

## `normalization`

| Key | Type | Meaning |
| --- | --- | --- |
| `mode` | `max_abs` or `positive_max` | Per-volume scale rule. |
| `epsilon` | positive float | Minimum accepted scale. |

`max_abs` divides by the largest absolute value and therefore includes both the negative minimum and positive maximum. No minimum is subtracted. `positive_max` divides by the largest positive value and is intended for nonnegative fields.

## `model`

| Key | Type | Meaning |
| --- | --- | --- |
| `patch_size` | three positive integers | `[Dp, Hp, Wp]`; each value must be divisible by eight and no larger than the corresponding volume dimension. |
| `base_channels` | positive integer | First learned feature width `C`; later widths are `2C`, `4C`, and `8C`. |
| `correction_scale` | positive float | Maximum model edit as a fraction of `absolute_error_bound`. |

A larger patch supplies more spatial context but increases activation memory. A larger feature width increases representation capacity and parameter count. Neither is automatically selected.

## `loss`

| Key | Type | Meaning |
| --- | --- | --- |
| `mse_mix` | nonnegative float | MSE fraction inside reconstruction and critical-point losses. |
| `charbonnier_mix` | nonnegative float | Robust Charbonnier fraction; the two mix values must sum to one. |
| `gradient` | nonnegative float | Weight on three-axis finite-difference consistency. |
| `critical` | nonnegative float | Weight on reconstruction error at FC voxels. |
| `topology` | nonnegative float | Maximum weight on 26-neighbor ordering after warm-up. |
| `gate` | nonnegative float | Weight on spatial gate supervision. |
| `error_bound` | nonnegative float | Weight on reference error exceeding `xi`. |
| `correction` | nonnegative float | Weight discouraging large or dense neural edits. |
| `gate_negative` | nonnegative float | Background term inside gate supervision. |
| `error_bound_tail` | nonnegative float | Multiplier for the worst 0.1% excess errors. |

Zero disables the corresponding weighted component where applicable. Changing loss weights changes the optimization problem and should be treated as a separate experiment.

## `training`

| Key | Type | Meaning |
| --- | --- | --- |
| `epochs` | positive integer | Maximum number of epochs. |
| `batch_size` | positive integer | Training patches per optimizer step. |
| `validation_batch_size` | positive integer | Validation patches evaluated together. |
| `samples_per_epoch` | positive integer | Deterministic training patch draws per epoch. |
| `validation_samples` | positive integer | Deterministic validation draws per epoch. |
| `num_workers` | nonnegative integer | Data-loader worker processes. |
| `learning_rate` | positive float | Initial AdamW learning rate. |
| `weight_decay` | positive float | AdamW weight decay. |
| `minimum_epochs` | positive integer | Earliest epoch at which early stopping is allowed; no greater than `epochs`. |
| `early_stopping_patience` | positive integer | Non-improving validation epochs tolerated after the minimum. |
| `topology_warmup_start` | nonnegative integer | Epoch through which the topology multiplier remains zero. |
| `topology_warmup_end` | nonnegative integer | Epoch at which the topology multiplier reaches one; no earlier than the start. |
| `mixed_precision` | Boolean | Enable CUDA automatic mixed precision. CPU execution remains full precision. |
| `seed` | nonnegative integer | Python, NumPy, and PyTorch seed. |
| `device` | `auto`, `cpu`, or `cuda` | Requested execution device. Explicit `cuda` fails if unavailable. |

AdamW is paired with `ReduceLROnPlateau` using factor `0.5` and scheduler patience `5`. The validation total loss selects `best.pt`. Improvement must exceed `1e-6`; otherwise the early-stopping counter increases.
