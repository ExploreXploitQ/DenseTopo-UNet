# Changelog

All notable project changes are documented here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and version numbers follow [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Planned

- Empirical evaluation on independently governed scientific data.
- Ablations for the gate, topology surrogate, correction scale, and patch context.
- Additional reviewed data adapters without weakening the raw-volume contract.

## [0.1.0] - 2026-08-26

### Added

- Strict experiment and manifest schemas for 3D `float32` scalar fields.
- Gated residual 3D U-Net with bounded corrections and identity initialization.
- Topology-focused patch sampling and configurable compound loss.
- Reproducible CPU/CUDA training, atomic checkpoints, and exact resume state.
- Single-decompressed-volume context-tiled inference.
- Numerical metrics and aggregation of external FP/FN/FT summaries.
- English CLI, deterministic synthetic workflow, technical documentation, and CI configuration.

[Unreleased]: https://github.com/ExploreXploitQ/DenseTopo-UNet/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ExploreXploitQ/DenseTopo-UNet/releases/tag/v0.1.0
