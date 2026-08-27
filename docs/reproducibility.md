# Reproducibility

## Run identity

A reproducible run is defined by more than a checkpoint filename. Preserve the experiment configuration, exact manifest bytes, raw input hashes or a governed data snapshot, topology extractor version and settings, source commit, package environment, and output directory.

Training stores the resolved configuration and a SHA-256 manifest fingerprint. The environment report records Python, platform, NumPy, PyTorch, CUDA runtime, visible CUDA device string, and discoverable device names. It does not capture the full operating-system image, compiler toolchain, storage hardware, upstream decompressor version, or external topology software; record those separately.

## Randomness

`training.seed` initializes Python, NumPy, CPU PyTorch, and all visible CUDA generators. Patch draws derive deterministically from the seed, epoch, and item index. Validation always uses epoch zero so the validation patch set remains fixed across epochs.

Checkpoints capture and restore Python, NumPy, CPU PyTorch, and visible CUDA random-number states. Data-loader order is fixed because the dataset performs its own deterministic sampling and the loader does not shuffle indices.

GPU kernels and hardware libraries may still introduce nondeterministic floating-point behavior. The package does not force PyTorch deterministic algorithms, and CPU and GPU trajectories are not expected to be bitwise identical. Use repeated seeds and report variability for scientific claims.

## Atomic and compatible checkpoints

Checkpoint writes use a temporary sibling file, flush and synchronize its bytes, then atomically replace the target. Schema version 1 stores:

- package and checkpoint schema versions;
- full resolved experiment configuration;
- manifest SHA-256 fingerprint;
- model, optimizer, scheduler, and optional AMP scaler states;
- current epoch, best score, non-improvement count, and history;
- all captured random-number-generator states.

Inference loads checkpoints on CPU first and constructs the architecture from checkpoint metadata. Resume rejects incompatible architecture, compression, normalization, loss, value-domain, or manifest settings.

## Split discipline

Train records update parameters. Validation records select checkpoints and stopping time. Test records are reserved for final deployment-like inference and evaluation. Do not tune patch size, loss weights, threshold, error bound, or epoch selection from test FC counts.

When comparing compressors, error bounds, or persistence values, create separate named configurations unless the scientific protocol explicitly defines a joint-training study. Report whether the model saw multiple conditions during training.

## Suggested experiment record

For every reported result, retain:

1. repository commit and package version;
2. full configuration and manifest fingerprint;
3. upstream compressor name, version, command, and effective error bound;
4. topology extractor name, version, persistence threshold, neighborhood, and matching rule;
5. hardware and environment JSON;
6. all per-sample numerical metrics and FP/FN/FT counts;
7. random seeds, convergence history, and checkpoint-selection rule;
8. failed runs and deviations from the preregistered protocol.

The deterministic synthetic workflow checks that these software paths execute. It does not establish cross-machine numerical identity or scientific effectiveness.
