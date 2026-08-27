# DenseTopo-UNet Repository Design

**Date:** 2026-08-26  
**Status:** Approved design, pending implementation plan  
**Repository:** `ExploreXploitQ/DenseTopo-UNet`

## 1. Purpose

DenseTopo-UNet is an alpha research package for restoring the topology of
three-dimensional scalar fields after error-bounded lossy decompression. The
package trains a gated residual 3D U-Net to edit a decompressed volume while
using reference values and precomputed false-critical-point labels only as
training supervision.

The public repository is compressor-agnostic. It can be used with reconstructed
floating-point output produced by compressors such as SPERR, SZ3, ZFP, MGARD,
and HPEZ, provided that the output is converted to the documented raw-volume
format. DenseTopo-UNet does not compress data, invoke a decompressor, or read an
encoded compressor bitstream.

The repository will not contain a research dataset, a pretrained checkpoint,
or dataset-specific benchmark claims. Users train their own checkpoints from a
manifest. Every public file, code comment, command message, configuration key,
figure label, and document will be written in English.

## 2. Scientific Information Boundary

### Training

Each training or validation record may provide:

1. one lossy-decompressed scalar volume;
2. the corresponding reference volume;
3. a CSV file of false critical points;
4. a CSV file of reference extrema; and
5. explicit storage metadata and a split label.

The network input is only the decompressed scalar volume. Reference values and
topology labels select patches and define losses; they are never concatenated
to the model input.

### Inference

Inference accepts:

1. one lossy-decompressed scalar volume;
2. its shape, dtype, byte order, and axis layout;
3. a checkpoint produced by DenseTopo-UNet; and
4. an output path.

The inference command must not accept a reference-volume path, false-critical-
point path, critical-point path, or topology summary. This restriction is part
of the tested public API.

### Evaluation

Evaluation is a separate offline operation. It may read reference data and
topology summaries because its purpose is to measure, not produce, a restored
volume. The neural package will validate and aggregate topology results but
will not embed the specialized topology extraction engine.

## 3. Data Contract

### Raw volume representation

The initial release supports flat, headerless IEEE-754 floating-point files.
The required representation is:

- dtype: 32-bit floating point;
- default byte order: little-endian, represented as NumPy `<f4`;
- axis order: `zyx`;
- memory order: C-contiguous, with `x` changing fastest;
- logical shape: `[depth, height, width]`, abbreviated `[D, H, W]`;
- required element count: `D * H * W`;
- required byte count: `D * H * W * 4`.

For example, a volume configured as `[32, 64, 96]` must contain exactly
`32 * 64 * 96 = 196,608` values and `786,432` bytes. The example dimensions are
synthetic and are not a benchmark declaration.

The restored output uses the same shape, dtype, byte order, axis layout, and
flat storage convention as the decompressed input.

### Manifest schema

One YAML manifest describes a scientifically coherent experiment. All records
in the initial release share the configured error bound, topology threshold,
storage convention, and shape.

```yaml
schema_version: 1
experiment: example-topology-restoration

volume:
  shape: [32, 64, 96]
  dtype: float32
  byte_order: little
  axis_order: zyx
  value_domain: signed

compression:
  absolute_error_bound: 1.0e-4

topology:
  persistence_threshold: 0.06
  match_radius: 0
  neighborhood: cube26

normalization:
  mode: max_abs
  epsilon: 1.0e-12

samples:
  - id: volume-001
    split: train
    decompressed: data/volume-001.lossy.f32
    reference: data/volume-001.reference.f32
    false_cases: labels/volume-001.false-cases.csv
    critical_points: labels/volume-001.critical-points.csv
```

Allowed split values are `train`, `validation`, and `test`. Training and
validation records require all four paths. Test records may omit label paths
when they are used only for inference. Relative paths resolve from the
manifest's directory, not the current working directory.

### False-critical-point schema

The false-case CSV header is:

```csv
case,z,y,x
```

`case` must be `fp`, `fn`, or `ft`. Coordinates are zero-based integer indices
and must satisfy `0 <= z < D`, `0 <= y < H`, and `0 <= x < W`.

### Critical-point schema

The reference-extrema CSV header is:

```csv
critical_type,z,y,x
```

The topology-focused sampler recognizes `local_maximum` and `local_minimum`.
Unknown values are rejected rather than silently ignored.

### Normalization and value domain

`max_abs` divides each decompressed volume by the maximum absolute finite value
and supports signed fields. `positive_max` reproduces the historical behavior
for nonnegative fields and divides by the largest positive value without
shifting zero. Both modes use the configured epsilon to reject a zero scale.

`value_domain: nonnegative` clamps restored values below zero. `signed` applies
no value clamp. The resolved normalization mode and value-domain rule are saved
in every checkpoint and cannot be changed silently at inference.

## 4. Model Contract

The model class is `DenseTopoUNet3D`. A training patch has shape
`[B, 1, Dp, Hp, Wp]`, where `B` is the batch size and the single channel is the
normalized lossy-decompressed scalar value.

The default feature widths are:

```text
1 -> 12 -> 24 -> 48 -> 96 -> 48 -> 24 -> 12 -> 2
```

Each encoder or decoder stage uses two 3D convolutions, group normalization,
and SiLU activation. Three max-pooling stages form the encoder. Transposed
convolutions, skip concatenation, and double-convolution blocks form the
decoder.

The two internal output channels are:

1. a correction logit transformed by `tanh`; and
2. a gate logit transformed by `sigmoid`.

For decompressed value `d`, configured absolute error bound `xi`, correction
ratio `r`, and gate `g`, the restored value is:

```text
restored = d + xi * correction_scale * r * g
```

The model returns the restored patch, correction ratio, and gate. The restored
patch has shape `[B, 1, Dp, Hp, Wp]`. The public configuration validates that
each patch dimension is positive, does not exceed the volume dimension, and is
compatible with three downsampling stages.

## 5. Objective

The training objective preserves the historical dense model and exposes every
weight in YAML:

```text
total = reconstruction
      + 0.10 * gradient
      + 10.0 * critical
      + 5.0 * topology_schedule * topology_order
      + 0.20 * gate_supervision
      + 25.0 * error_bound
      + 0.005 * correction_regularization
```

- `reconstruction` combines weighted MSE and Charbonnier penalties.
- `gradient` compares finite differences along the z, y, and x axes.
- `critical` gives non-vanishing reconstruction supervision to false cases.
- `topology_order` penalizes incorrect center-neighbor ordering at supervised
  extrema.
- `gate_supervision` opens the gate near supervised false cases and lightly
  penalizes background activation.
- `error_bound` penalizes reference error beyond `xi` and upweights the worst
  0.1 percent of patch errors.
- `correction_regularization` discourages large or spatially dense edits.

The initial release implements the historical 26-neighbor cube surrogate. The
documentation must state that this differentiable local ordering objective is
not an exact persistent-topology calculation and does not guarantee removal of
false critical points.

## 6. Patch Sampling and Full-Volume Inference

Training samples topology-focused patches using configurable category weights
for false negatives, false positives, preserved extrema, and random locations.
Random geometric augmentation may flip each spatial axis independently.

Full-volume inference uses context tiling. For default patch shape
`[32, 64, 64]`, each patch contributes only its center core of
`[16, 32, 32]`; the resulting core stride is 50 percent of the patch size.
Reflection padding supplies context at volume boundaries. The tiler must prove
through tests that every output voxel is written exactly once for divisible and
non-divisible volume dimensions.

## 7. Public Commands

The installed executable is `densetopo`, with `python -m densetopo_unet` as an
equivalent entry point.

```text
densetopo validate-manifest --manifest PATH
densetopo train --config PATH --manifest PATH --output NEW_DIRECTORY
densetopo infer --checkpoint PATH --input PATH --shape D H W --output PATH
densetopo evaluate --manifest PATH --checkpoint PATH --split test --output NEW_DIRECTORY
densetopo inspect-checkpoint --checkpoint PATH
```

`train` writes an immutable resolved configuration, manifest fingerprint,
environment report, CSV history, best checkpoint, latest checkpoint, and run
summary. Checkpoints preserve model configuration, optimizer, scheduler,
precision scaler, early-stopping state, best score, epoch, random-number states,
and package/checkpoint schema versions.

Output directories for train and evaluate must be new or empty. Checkpoint
writes use a temporary sibling followed by an atomic rename. Resume validates
the manifest fingerprint and architecture before restoring training state.

The inference CLI obtains error-bound, normalization, patch, correction-scale,
and value-domain settings from the checkpoint. It requires explicit input shape
because a headerless file cannot encode dimensions. It writes an output file
and a JSON provenance record without requesting reference information.

## 8. Repository Structure

The repository follows the engineering conventions established by the related
PTU-Net project while keeping DenseTopo-UNet scientifically independent.

```text
.github/                   CI, issue forms, pull-request template, CODEOWNERS
assets/                    English SVG wordmark and architecture diagram
configs/                   Model, synthetic smoke, and manifest examples
docs/                      Architecture, data contract, usage, reproducibility
scripts/                   Deterministic synthetic workflow generator
src/densetopo_unet/        Installable Python package
tests/                     Unit, integration, CLI, and optional GPU tests
.editorconfig
.gitattributes
.gitignore
CHANGELOG.md
CITATION.cff
CODE_OF_CONDUCT.md
CONTRIBUTING.md
Makefile
MODEL_CARD.md
README.md
SECURITY.md
pyproject.toml
```

The Python package is decomposed by responsibility:

```text
config.py        typed YAML configuration
manifest.py      data-contract parsing and validation
io.py            exact-size memory-mapped raw-volume access
data.py          topology-focused patch dataset
model.py         DenseTopoUNet3D
losses.py        seven-part objective
tiling.py        full-volume context tiling
checkpoint.py    versioned atomic checkpoint format
engine.py        training and validation loop
inference.py     one-volume inference service
metrics.py       numerical and topology aggregation
reproducibility.py
cli.py
__main__.py
```

No model weight file, raw research volume, generated restoration, or local run
directory is tracked by Git.

## 9. Documentation and Presentation

The README uses an original SVG wordmark, CI and Python badges, an evidence
status callout, an architecture figure, a five-command quick start, the exact
input boundary, the raw-volume dimensions, compressor examples, the repository
layout, and links to focused documents.

Documentation includes:

- `docs/architecture.md`: tensor shapes, network stages, residual/gate formula;
- `docs/data-contract.md`: bytes, dtype, shape, layouts, CSV schemas;
- `docs/configuration.md`: every YAML key and validation rule;
- `docs/usage.md`: validation, training, resume, inference, evaluation;
- `docs/reproducibility.md`: seeds, fingerprints, checkpoints, environment;
- `docs/topology-labels.md`: external-label boundary and accepted schemas;
- `docs/limitations.md`: surrogate topology loss and error-bound limitations;
- `MODEL_CARD.md`: intended use, out-of-scope use, risks, and evidence status.

The repository will describe SPERR, SZ3, ZFP, MGARD, and HPEZ only as examples
of upstream compressors whose reconstructed floating-point output can be
prepared according to the data contract. It will not claim tested compatibility
or performance without machine-readable evidence.

## 10. Synthetic Workflow

A deterministic generator creates tiny signed and nonnegative reference
volumes, lossy-decompressed proxies, and valid topology-label CSV files. These
files test installation, manifests, data loading, training, checkpointing, and
single-volume inference on CPU.

The generator output is explicitly labeled a software fixture. Its perturbation
is not attributed to any compressor, and smoke-test success is not presented as
scientific evidence.

## 11. Verification and CI

The development stack uses Python 3.10 or newer, PyTorch 2.1 or newer, NumPy,
PyYAML, pytest, coverage, Ruff, and mypy. GitHub Actions runs:

1. Ruff linting and formatting checks;
2. mypy over `src/densetopo_unet`;
3. CPU tests on Python 3.10 and 3.12;
4. branch coverage with a minimum threshold of 75 percent; and
5. package build and wheel-install smoke checks.

Required tests cover:

- exact file-size, dtype, coordinate, split, and manifest validation;
- normalized input and tensor dimensions;
- encoder/decoder output shapes;
- bounded correction and optional nonnegative clamp;
- each loss component and finite gradients;
- topology warm-up behavior;
- deterministic patch sampling;
- complete tiled coverage and boundary behavior;
- checkpoint round trip and incompatible-checkpoint rejection;
- inference parser rejection of reference or topology arguments;
- numerical metrics; and
- a tiny end-to-end CPU train/infer workflow.

GPU tests are marked separately and are not required by public CPU CI.

## 12. Error Handling

Public commands fail before allocating large tensors when a manifest is invalid,
a required file is absent, byte count is incorrect, coordinates are out of
bounds, values are non-finite, splits overlap by sample ID, a patch is invalid,
or checkpoint metadata conflicts with the request.

Errors name the sample, field, expected value, observed value, and corrective
action. The package does not silently infer shape, byte order, value domain,
topology semantics, or missing labels from filenames.

## 13. Publication Boundary

The package is versioned as `0.1.0` and labeled alpha research software.
`CITATION.cff` cites the software version or commit without claiming an
associated paper or institution. As in the reference PTU-Net repository, the
initial release does not invent a license grant; the README explicitly states
the resulting copyright restriction.

The current scope is local repository creation only. Implementation may be
committed to the local Git history, but no `git push`, GitHub Release, pull
request, or remote API mutation is permitted. The local repository contains
source, tests, synthetic fixtures generated on demand, configuration examples,
and documentation, but no pretrained checkpoint or research data.
