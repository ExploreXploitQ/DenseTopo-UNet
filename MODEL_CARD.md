# Model card: DenseTopo-UNet 0.1

## Model summary

DenseTopo-UNet is a gated residual 3D U-Net for restoring topology-related local structure in lossy-decompressed scalar fields. The model receives one normalized decompressed channel and predicts a bounded edit around the unnormalized decompressed value.

This repository defines an architecture and training procedure, not a released trained model. No pretrained checkpoint is included.

## Intended use

The package is intended for research on structured restoration of regular three-dimensional scalar grids after error-bounded lossy decompression. Appropriate use requires users to train a checkpoint with reference fields and FC labels created under a documented topology protocol.

Possible upstream compressors include SPERR, SZ3, ZFP, MGARD, and HPEZ after their output has been converted to the documented raw format. Mention of these tools describes interface compatibility, not measured validation.

## Out-of-scope use

The software is not designed for encoded bitstream decoding, compression, contour-tree construction, medical diagnosis, safety-critical control, irregular meshes, vector fields, or inference without a checkpoint trained for a compatible data distribution.

It should not be presented as guaranteeing persistent-topology correctness or original-reference error-bound preservation.

## Inputs and outputs

Inference input is one headerless `float32` scalar volume with shape `[D, H, W]`, declared byte order, and `zyx` layout. The model tensor has shape `[B, 1, Dp, Hp, Wp]`; the single channel is the normalized decompressed scalar.

Output is one restored volume with the same shape and storage convention. The model also computes a correction-ratio field and a spatial gate internally. Reference values and topology labels are used only for training and evaluation.

## Architecture

The default feature widths are 12, 24, 48, and 96. Three pooling stages and three skip-connected decoder stages produce two logits. `tanh` bounds the signed proposal, `sigmoid` gates its spatial support, and the configured absolute error bound scales the residual.

See [the architecture guide](docs/architecture.md) for exact tensors and losses.

## Training data

Training and validation records require paired decompressed and reference volumes, false-case coordinates (`FP`, `FN`, `FT`), and reference local extrema. This repository supplies only deterministic analytic fixtures for software tests. It supplies no scientific training data, data license, or provenance claim.

## Evaluation status

No scientific performance result is bundled. A valid evaluation should report per-sample and aggregate FC counts from an independent topology tool, strict error-bound violations, maximum error, RMSE, PSNR, runtime, memory, multiple seeds, and failure cases.

The project target of removing 90% of FCs remains unevaluated at repository level.

## Risks and limitations

- The local ordering surrogate can disagree with global persistent topology.
- A residual bounded around decompressed values can still exceed the reference-relative error bound.
- Results may not transfer across variables, spatial resolutions, compressors, error bounds, or persistence thresholds.
- Label-generation errors directly alter patch sampling and loss weighting.
- Small 3D patches may omit structures needed to make a globally correct edit.
- GPU training can be nondeterministic across hardware and library versions.

See [limitations](docs/limitations.md) for the complete evidence boundary.

## Reproducibility and maintenance

Every training run records resolved configuration, environment information, a manifest fingerprint, loss history, and versioned atomic checkpoints. Users remain responsible for recording upstream decompressor and topology-extractor versions.

The software is alpha quality. Report vulnerabilities through [the security policy](SECURITY.md) and scientific or implementation issues through the repository issue tracker.
