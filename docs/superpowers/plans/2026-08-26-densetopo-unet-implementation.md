# DenseTopo-UNet Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a professional, English-only, locally committed research repository for training and applying DenseTopo-UNet to topology restoration of three-dimensional scalar fields after lossy decompression.

**Architecture:** The repository exposes a typed manifest and configuration layer, an installable PyTorch package, a gated residual 3D U-Net, a seven-part training objective, topology-focused patch sampling, context-tiled inference, versioned checkpoints, separate evaluation utilities, and a deterministic CPU smoke workflow. Inference accepts exactly one decompressed volume plus storage metadata and a user-trained checkpoint; reference values and topology labels remain training/evaluation-only information.

**Tech Stack:** Python 3.10+, PyTorch 2.1+, NumPy 1.24+, PyYAML 6+, Hatchling, pytest, coverage, Ruff, mypy, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-08-26-densetopo-unet-repository-design.md`

## Global Constraints

- All repository content, code comments, CLI messages, configuration keys, figures, and documentation are English-only.
- The package reads already decompressed floating-point fields, not encoded compressor bitstreams.
- SPERR, SZ3, ZFP, MGARD, and HPEZ appear only as upstream-compressor examples, not as verified compatibility claims.
- No research dataset, pretrained checkpoint, generated restoration, or dataset-specific result is committed.
- Raw volumes are flat IEEE-754 float32 arrays with explicit shape `[D, H, W]`, byte order, and `zyx` axis order.
- Training input contains one decompressed scalar channel; reference data and topology labels are supervision only.
- Inference must not accept reference data or topology-label arguments.
- The initial topology surrogate uses the historical 26-neighbor cube ordering.
- The initial project version is `0.1.0` and is labeled alpha research software.
- The current delivery scope is local repository creation and local commits only. Never run `git push`, create a release, open a pull request, or mutate a remote API.
- Every production behavior follows red-green-refactor: write a focused test, observe the expected failure, implement the minimum behavior, and rerun the relevant test before refactoring.

## File Map

```text
src/densetopo_unet/
  __init__.py             package version and public exports
  __main__.py             python -m entry point
  config.py               typed experiment configuration
  manifest.py             typed data manifest and label validation
  io.py                   exact-size raw float32 memory maps
  model.py                DenseTopoUNet3D and ModelOutput
  losses.py               loss components and weighted objective
  data.py                 topology-focused patch dataset
  tiling.py               center-core full-volume inference
  checkpoint.py           atomic versioned checkpoint state
  reproducibility.py      seeds, RNG state, environment metadata
  engine.py               training and validation loop
  inference.py            one-decompressed-volume inference service
  metrics.py              numerical and topology aggregation
  cli.py                  public command parser and dispatch
tests/
  conftest.py             reusable raw-volume and label fixtures
  test_config.py          configuration validation
  test_manifest.py        manifest, storage, and CSV validation
  test_model.py           model tensor and correction contracts
  test_losses.py          seven loss components and gradients
  test_data.py            deterministic topology-focused sampling
  test_tiling.py          full-volume coverage and boundary behavior
  test_checkpoint.py      checkpoint round trip and compatibility
  test_engine.py          one-epoch CPU training behavior
  test_inference.py       one-input inference boundary
  test_metrics.py         numerical and FC aggregation
  test_cli.py             command parsing and error messages
  test_workflow.py        synthetic end-to-end workflow
scripts/generate_synthetic_data.py
configs/model.yaml
configs/synthetic.yaml
configs/manifest.example.yaml
docs/*.md
assets/densetopo-unet-wordmark.svg
assets/architecture.svg
.github/*
pyproject.toml
Makefile
README.md
MODEL_CARD.md
CITATION.cff
```

---

### Task 1: Package Metadata and Typed Experiment Configuration

**Files:**
- Create: `pyproject.toml`
- Create: `src/densetopo_unet/__init__.py`
- Create: `src/densetopo_unet/config.py`
- Create: `tests/test_config.py`
- Create: `configs/model.yaml`

**Interfaces:**
- Produces: `VolumeConfig`, `CompressionConfig`, `TopologyConfig`, `NormalizationConfig`, `ModelConfig`, `TrainingConfig`, `ExperimentConfig`.
- Produces: `load_experiment_config(path: Path) -> ExperimentConfig`.
- Produces: `ExperimentConfig.to_dict() -> dict[str, object]` for checkpoint serialization.

- [ ] **Step 1: Add a failing configuration test**

```python
def test_load_experiment_config_resolves_valid_3d_settings(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text(
        """
volume: {shape: [16, 24, 32], dtype: float32, byte_order: little, axis_order: zyx, value_domain: signed}
compression: {absolute_error_bound: 0.0001}
topology: {persistence_threshold: 0.06, match_radius: 0, neighborhood: cube26}
normalization: {mode: max_abs, epsilon: 1.0e-12}
model: {patch_size: [8, 16, 16], base_channels: 4, correction_scale: 0.75}
training: {epochs: 2, batch_size: 2, samples_per_epoch: 8, seed: 7}
"""
    )
    config = load_experiment_config(path)
    assert config.volume.shape == (16, 24, 32)
    assert config.model.patch_size == (8, 16, 16)
    assert config.compression.absolute_error_bound == pytest.approx(1.0e-4)
```

- [ ] **Step 2: Run the test and observe the missing-module failure**

Run: `python -m pytest tests/test_config.py -v`  
Expected: FAIL because `densetopo_unet.config` is not implemented.

- [ ] **Step 3: Implement frozen dataclasses and strict YAML validation**

```python
@dataclass(frozen=True)
class VolumeConfig:
    shape: tuple[int, int, int]
    dtype: Literal["float32"] = "float32"
    byte_order: Literal["little", "big"] = "little"
    axis_order: Literal["zyx"] = "zyx"
    value_domain: Literal["signed", "nonnegative"] = "signed"

@dataclass(frozen=True)
class ExperimentConfig:
    volume: VolumeConfig
    compression: CompressionConfig
    topology: TopologyConfig
    normalization: NormalizationConfig
    model: ModelConfig
    training: TrainingConfig

def load_experiment_config(path: Path) -> ExperimentConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _parse_experiment_mapping(raw)
```

Validation rejects unknown top-level keys, non-3D shapes, nonpositive dimensions,
nonpositive error bounds, unsupported topology neighborhoods, invalid patch sizes,
and patch dimensions not divisible by eight.

- [ ] **Step 4: Add focused invalid-shape, unknown-key, and patch-divisibility cases**

Run: `python -m pytest tests/test_config.py -v`  
Expected: PASS for valid config and PASS for all rejection assertions.

- [ ] **Step 5: Add package metadata and verify editable installation metadata**

`pyproject.toml` defines package `densetopo-unet`, version `0.1.0`, Hatchling
build configuration, dependencies, `densetopo = "densetopo_unet.cli:main"`,
pytest markers, 75 percent coverage floor, Ruff rules, and strict mypy settings.
The development extra contains pytest, pytest-cov, coverage, Ruff, mypy,
types-PyYAML, and `build>=1.2`.

Run: `python -m pip install -e . --no-deps`  
Expected: installation completes and `python -c "import densetopo_unet"` exits 0.

- [ ] **Step 6: Commit locally**

```bash
git add pyproject.toml src/densetopo_unet tests/test_config.py configs/model.yaml
git commit -m "feat: add typed experiment configuration"
```

### Task 2: Raw Volume I/O and Generic Manifest Validation

**Files:**
- Create: `src/densetopo_unet/io.py`
- Create: `src/densetopo_unet/manifest.py`
- Create: `tests/conftest.py`
- Create: `tests/test_manifest.py`
- Create: `configs/manifest.example.yaml`

**Interfaces:**
- Consumes: `VolumeConfig` from Task 1.
- Produces: `SampleRecord`, `DataManifest`, `load_manifest(path: Path, volume: VolumeConfig) -> DataManifest`.
- Produces: `validate_raw_volume(path: Path, volume: VolumeConfig) -> None`.
- Produces: `open_raw_volume(path: Path, volume: VolumeConfig, mode: str = "r") -> np.memmap`.
- Produces: `load_false_cases(path: Path, shape: tuple[int, int, int]) -> dict[str, np.ndarray]`.
- Produces: `load_critical_points(path: Path, shape: tuple[int, int, int]) -> np.ndarray`.

- [ ] **Step 1: Write failing exact-size and path-resolution tests**

```python
def test_manifest_resolves_paths_relative_to_manifest(tmp_path: Path) -> None:
    manifest_path = write_valid_manifest_fixture(tmp_path)
    manifest = load_manifest(manifest_path, volume_fixture())
    assert manifest.samples[0].decompressed == (tmp_path / "data/input.f32").resolve()

def test_validate_raw_volume_rejects_wrong_byte_count(tmp_path: Path) -> None:
    path = tmp_path / "short.f32"
    np.zeros(31, dtype="<f4").tofile(path)
    with pytest.raises(ValueError, match="expected .* bytes, observed"):
        validate_raw_volume(path, VolumeConfig(shape=(2, 4, 4)))
```

- [ ] **Step 2: Run the tests and verify expected missing-function failures**

Run: `python -m pytest tests/test_manifest.py -v`  
Expected: FAIL because manifest and raw-I/O functions do not exist.

- [ ] **Step 3: Implement exact storage validation and memory mapping**

```python
def raw_dtype(config: VolumeConfig) -> np.dtype[np.float32]:
    return np.dtype("<f4" if config.byte_order == "little" else ">f4")

def validate_raw_volume(path: Path, config: VolumeConfig) -> None:
    expected = math.prod(config.shape) * np.dtype(np.float32).itemsize
    observed = path.stat().st_size
    if observed != expected:
        raise ValueError(f"{path}: expected {expected} bytes, observed {observed}")

def open_raw_volume(path: Path, config: VolumeConfig, mode: str = "r") -> np.memmap:
    validate_raw_volume(path, config)
    return np.memmap(path, dtype=raw_dtype(config), mode=mode, shape=config.shape)
```

- [ ] **Step 4: Implement manifest and CSV schemas**

Reject duplicate sample IDs, invalid split names, missing training/validation
labels, extra CSV columns, unsupported case/type values, noninteger coordinates,
out-of-bounds coordinates, and non-finite raw values sampled during validation.

- [ ] **Step 5: Run manifest tests**

Run: `python -m pytest tests/test_manifest.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit locally**

```bash
git add src/densetopo_unet/io.py src/densetopo_unet/manifest.py tests configs/manifest.example.yaml
git commit -m "feat: validate generic 3D volume manifests"
```

### Task 3: Gated Residual 3D U-Net

**Files:**
- Create: `src/densetopo_unet/model.py`
- Create: `tests/test_model.py`

**Interfaces:**
- Consumes: `ModelConfig` and `value_domain` from Task 1.
- Produces: `ModelOutput(restored, correction_ratio, gate)`.
- Produces: `DenseTopoUNet3D(base_channels: int, correction_scale: float, nonnegative: bool)`.
- `forward(normalized_input: Tensor, decompressed: Tensor, xi: float) -> ModelOutput`.

- [ ] **Step 1: Write failing shape and identity-initialization tests**

```python
def test_model_returns_three_same_shape_fields() -> None:
    model = DenseTopoUNet3D(base_channels=4, correction_scale=0.75, nonnegative=False)
    normalized = torch.randn(2, 1, 8, 16, 16)
    decompressed = torch.randn_like(normalized)
    output = model(normalized, decompressed, xi=1.0e-4)
    assert output.restored.shape == normalized.shape
    assert output.correction_ratio.shape == normalized.shape
    assert output.gate.shape == normalized.shape
    torch.testing.assert_close(output.restored, decompressed)
```

- [ ] **Step 2: Run tests and observe the missing-model failure**

Run: `python -m pytest tests/test_model.py -v`  
Expected: FAIL because `DenseTopoUNet3D` is not implemented.

- [ ] **Step 3: Implement the encoder, decoder, skip paths, and two-channel head**

```python
@dataclass(frozen=True)
class ModelOutput:
    restored: torch.Tensor
    correction_ratio: torch.Tensor
    gate: torch.Tensor

def forward(self, normalized_input: Tensor, decompressed: Tensor, xi: float) -> ModelOutput:
    features = self.decode(self.encode(normalized_input))
    correction_logit, gate_logit = self.head(features).chunk(2, dim=1)
    gate = torch.sigmoid(gate_logit)
    ratio = self.correction_scale * torch.tanh(correction_logit) * gate
    restored = decompressed + float(xi) * ratio
    if self.nonnegative:
        restored = torch.clamp_min(restored, 0.0)
    return ModelOutput(restored, ratio, gate)
```

Initialize the final head weights and bias to zero so an untrained model is the
identity editor for signed data.

- [ ] **Step 4: Add bounded-correction, nonnegative-clamp, invalid-shape, and backward tests**

Run: `python -m pytest tests/test_model.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit locally**

```bash
git add src/densetopo_unet/model.py tests/test_model.py
git commit -m "feat: implement gated residual 3D U-Net"
```

### Task 4: Seven-Part Topology-Aware Objective

**Files:**
- Create: `src/densetopo_unet/losses.py`
- Create: `tests/test_losses.py`

**Interfaces:**
- Consumes: `ModelOutput` from Task 3.
- Produces: `LossWeights`, `LossBreakdown`.
- Produces: `topology_order_loss(restored, target, topo_weight, xi) -> Tensor`.
- Produces: `compute_losses(output, target, decompressed, topo_weight, topology_lambda, weights, xi) -> LossBreakdown`.

- [ ] **Step 1: Write a failing perfect-reconstruction test**

```python
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
```

- [ ] **Step 2: Run tests and verify the objective is missing**

Run: `python -m pytest tests/test_losses.py -v`  
Expected: FAIL because `LossWeights` and `compute_losses` do not exist.

- [ ] **Step 3: Implement reconstruction, gradient, critical, gate, error-bound, and correction losses**

Normalize reconstruction error by `xi`, use the configured weighted formula,
compute finite differences on tensor axes 2, 3, and 4, and calculate the error-
bound tail from the worst 0.1 percent of elements per sample.

- [ ] **Step 4: Implement the 26-neighbor topology-order surrogate**

```python
OFFSETS_26 = torch.tensor(
    [(z, y, x) for z in (-1, 0, 1) for y in (-1, 0, 1) for x in (-1, 0, 1)
     if (z, y, x) != (0, 0, 0)],
    dtype=torch.long,
)
```

Use the maximum valid neighbor violation per supervised point so a single wrong
ordering is not averaged away.

- [ ] **Step 5: Add tests for wrong ordering, exceeded error bound, positive gate target, empty topology mask, and finite backward gradients**

Run: `python -m pytest tests/test_losses.py -v`  
Expected: PASS.

- [ ] **Step 6: Commit locally**

```bash
git add src/densetopo_unet/losses.py tests/test_losses.py
git commit -m "feat: add topology-aware training objective"
```

### Task 5: Deterministic Topology-Focused Patch Dataset

**Files:**
- Create: `src/densetopo_unet/data.py`
- Create: `tests/test_data.py`

**Interfaces:**
- Consumes: `DataManifest`, `ExperimentConfig`, raw-I/O functions, and topology CSV loaders.
- Produces: `TopologyPatchDataset(manifest, config, split, samples_per_epoch, seed, augment)`.
- Produces batches with keys `input`, `decompressed`, `target`, `topo_weight`, `input_scale`, and `sample_id`.
- Produces: `normalization_scale(volume: np.ndarray, mode: str, epsilon: float) -> float`.

- [ ] **Step 1: Write failing normalization and deterministic-sampling tests**

```python
def test_max_abs_normalization_supports_signed_values() -> None:
    volume = np.array([-4.0, 0.0, 2.0], dtype=np.float32)
    assert normalization_scale(volume, "max_abs", 1.0e-12) == pytest.approx(4.0)

def test_same_seed_and_epoch_select_same_patch(synthetic_manifest, experiment_config) -> None:
    left = TopologyPatchDataset(synthetic_manifest, experiment_config, "train", 4, 17, False)
    right = TopologyPatchDataset(synthetic_manifest, experiment_config, "train", 4, 17, False)
    torch.testing.assert_close(left[0]["input"], right[0]["input"])
```

- [ ] **Step 2: Run tests and verify the dataset is missing**

Run: `python -m pytest tests/test_data.py -v`  
Expected: FAIL because dataset functions do not exist.

- [ ] **Step 3: Implement bounded memmap caching and category-weighted patch centers**

The default sampling probabilities are FN 0.35, FP 0.25, preserved extrema
0.20, and random 0.20. Assign topology weights FN 5, FP 3, FT 4, preserved
extremum 1. Apply optional independent flips along z, y, and x.

- [ ] **Step 4: Add shape, coordinate, augmentation, and zero-scale tests**

Run: `python -m pytest tests/test_data.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit locally**

```bash
git add src/densetopo_unet/data.py tests/test_data.py
git commit -m "feat: add topology-focused 3D patch sampling"
```

### Task 6: Center-Core Tiled Inference

**Files:**
- Create: `src/densetopo_unet/tiling.py`
- Create: `tests/test_tiling.py`

**Interfaces:**
- Produces: `AxisPlan(left_pad, right_pad, core, starts)`.
- Produces: `make_axis_plan(length: int, patch: int) -> AxisPlan`.
- Produces: `restore_volume_tiled(model, decompressed, input_scale, patch_size, device, batch_size, xi) -> np.ndarray`.

- [ ] **Step 1: Write failing exact-coverage tests**

```python
@pytest.mark.parametrize("length,patch", [(32, 8), (31, 8), (7, 8)])
def test_axis_plan_covers_every_output_index_once(length: int, patch: int) -> None:
    plan = make_axis_plan(length, patch)
    coverage = np.zeros(length, dtype=np.int32)
    for start in plan.starts:
        output_start = max(0, start - plan.left_pad)
        output_stop = min(length, output_start + plan.core)
        coverage[output_start:output_stop] += 1
    assert np.all(coverage == 1)
```

- [ ] **Step 2: Run tests and verify tiling functions are missing**

Run: `python -m pytest tests/test_tiling.py -v`  
Expected: FAIL because `make_axis_plan` is not implemented.

- [ ] **Step 3: Implement reflection-padding jobs and batched center-core copies**

The halo is one quarter of each patch dimension and the copied core is one half.
Record an internal integer write-count array in test mode and reject incomplete
or overlapping coverage.

- [ ] **Step 4: Add identity-model round-trip tests for divisible and non-divisible 3D shapes**

Run: `python -m pytest tests/test_tiling.py -v`  
Expected: PASS with bitwise-identical identity reconstruction.

- [ ] **Step 5: Commit locally**

```bash
git add src/densetopo_unet/tiling.py tests/test_tiling.py
git commit -m "feat: add complete center-core volume tiling"
```

### Task 7: Reproducibility and Versioned Atomic Checkpoints

**Files:**
- Create: `src/densetopo_unet/reproducibility.py`
- Create: `src/densetopo_unet/checkpoint.py`
- Create: `tests/test_checkpoint.py`

**Interfaces:**
- Produces: `seed_everything(seed: int) -> None`.
- Produces: `capture_rng_state() -> dict[str, object]` and `restore_rng_state(state) -> None`.
- Produces: `manifest_fingerprint(path: Path) -> str` using SHA-256.
- Produces: `save_checkpoint_atomic(path: Path, state: CheckpointState) -> None`.
- Produces: `load_checkpoint(path: Path, expected_config: ExperimentConfig | None = None) -> CheckpointState`.

- [ ] **Step 1: Write a failing checkpoint round-trip test**

```python
def test_checkpoint_round_trip_preserves_model_and_metadata(tmp_path: Path) -> None:
    state = checkpoint_fixture(epoch=3, best_score=0.25)
    path = tmp_path / "model.pt"
    save_checkpoint_atomic(path, state)
    loaded = load_checkpoint(path)
    assert loaded.schema_version == 1
    assert loaded.epoch == 3
    assert loaded.best_score == pytest.approx(0.25)
    assert loaded.manifest_sha256 == state.manifest_sha256
```

- [ ] **Step 2: Run tests and verify checkpoint functions are missing**

Run: `python -m pytest tests/test_checkpoint.py -v`  
Expected: FAIL because checkpoint functions do not exist.

- [ ] **Step 3: Implement serializable state, temporary-sibling write, fsync, and atomic replace**

Checkpoint state includes schema version, package version, resolved config,
manifest SHA-256, model, optimizer, scheduler, scaler, epoch, best score, bad
epoch count, history, and CPU/CUDA/NumPy/Python RNG states.

- [ ] **Step 4: Add incompatible-architecture, corrupted-file, and RNG-restoration tests**

Run: `python -m pytest tests/test_checkpoint.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit locally**

```bash
git add src/densetopo_unet/reproducibility.py src/densetopo_unet/checkpoint.py tests/test_checkpoint.py
git commit -m "feat: add reproducible atomic checkpoints"
```

### Task 8: Training and Validation Engine

**Files:**
- Create: `src/densetopo_unet/engine.py`
- Create: `tests/test_engine.py`

**Interfaces:**
- Consumes: config, manifest, dataset, model, objective, and checkpoint modules.
- Produces: `topology_warmup(epoch: int, start: int, end: int) -> float`.
- Produces: `run_epoch(model, loader, device, loss_context, optimizer=None, scaler=None) -> dict[str, float]`.
- Produces: `train(config, manifest, output_dir, resume=None) -> TrainingSummary`.

- [ ] **Step 1: Write failing warm-up and one-epoch CPU tests**

```python
def test_topology_warmup_is_piecewise_linear() -> None:
    assert topology_warmup(10, 20, 120) == 0.0
    assert topology_warmup(70, 20, 120) == pytest.approx(0.5)
    assert topology_warmup(120, 20, 120) == 1.0
```

- [ ] **Step 2: Run tests and verify engine functions are missing**

Run: `python -m pytest tests/test_engine.py -v`  
Expected: FAIL because the training engine does not exist.

- [ ] **Step 3: Implement device selection, loaders, AMP, optimizer, scheduler, history, early stopping, best/latest checkpoints, and resume**

Use AdamW and ReduceLROnPlateau. CUDA mixed precision is optional; CPU runs in
float32. Validation always evaluates the full topology-loss weight. Output
directories are rejected unless absent or empty.

- [ ] **Step 4: Add tests that parameters change, metrics are finite, output artifacts exist, and resume advances the epoch**

Run: `python -m pytest tests/test_engine.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit locally**

```bash
git add src/densetopo_unet/engine.py tests/test_engine.py
git commit -m "feat: add reproducible training engine"
```

### Task 9: Single-Volume Inference Service

**Files:**
- Create: `src/densetopo_unet/inference.py`
- Create: `tests/test_inference.py`

**Interfaces:**
- Consumes: raw I/O, checkpoint loading, model, and tiling.
- Produces: `InferenceRequest(checkpoint, input_path, output_path, shape, byte_order, batch_size, device)`.
- Produces: `restore_file(request: InferenceRequest) -> InferenceRecord`.

- [ ] **Step 1: Write a failing one-input API test**

```python
def test_inference_request_exposes_no_reference_or_topology_fields() -> None:
    names = {field.name for field in dataclasses.fields(InferenceRequest)}
    assert "reference" not in names
    assert "false_cases" not in names
    assert "critical_points" not in names
```

- [ ] **Step 2: Run tests and verify the inference service is missing**

Run: `python -m pytest tests/test_inference.py -v`  
Expected: FAIL because `InferenceRequest` is not implemented.

- [ ] **Step 3: Implement checkpoint-derived preprocessing and exact-format output**

Load normalization mode, epsilon, value domain, error bound, patch size, and
correction scale from the checkpoint. Validate requested shape and byte count,
run tiled inference, write a temporary output, atomically rename it, and emit a
JSON record containing hashes, shape, dtype, device, runtime, and package version.

- [ ] **Step 4: Add identity-checkpoint round trip and incorrect-byte-count tests**

Run: `python -m pytest tests/test_inference.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit locally**

```bash
git add src/densetopo_unet/inference.py tests/test_inference.py
git commit -m "feat: add one-volume topology restoration inference"
```

### Task 10: Numerical and Topology Evaluation

**Files:**
- Create: `src/densetopo_unet/metrics.py`
- Create: `tests/test_metrics.py`

**Interfaces:**
- Produces: `ErrorMetrics(max_abs_error, rmse, psnr, eb_violations, eb_violation_fraction)`.
- Produces: `compute_error_metrics(reference, candidate, error_bound, data_range) -> ErrorMetrics`.
- Produces: `FCSummary(fp: int, fn: int, ft: int)` with property `total`.
- Produces: `aggregate_evaluation(rows: Sequence[EvaluationRow]) -> EvaluationSummary`.
- Produces: `evaluate_restored(manifest, restored_root, split, output_dir, baseline_topology_dir=None, restored_topology_dir=None) -> EvaluationSummary`.

- [ ] **Step 1: Write failing exact metric tests**

```python
def test_error_metrics_counts_strict_error_bound_violations() -> None:
    reference = np.zeros(3, dtype=np.float32)
    candidate = np.array([0.0, 1.0e-4, 1.1e-4], dtype=np.float32)
    metrics = compute_error_metrics(reference, candidate, 1.0e-4, 1.0)
    assert metrics.eb_violations == 1
    assert metrics.max_abs_error == pytest.approx(1.1e-4)
```

- [ ] **Step 2: Run tests and verify metrics are missing**

Run: `python -m pytest tests/test_metrics.py -v`  
Expected: FAIL because evaluation metrics do not exist.

- [ ] **Step 3: Implement chunk-safe RMSE, PSNR, EB counts, FC summaries, and aggregate JSON/CSV records**

The package reads topology summary JSON produced externally and requires keys
`FP`, `FN`, and `FT`. It never runs or emulates a persistent-topology extractor.
Restored files resolve as `<restored-root>/<sample-id>.restored.f32`. Topology
aggregation is enabled only when both baseline and restored summary directories
are supplied; providing one without the other raises a configuration error.

- [ ] **Step 4: Add zero-MSE, aggregate-FC, malformed-summary, and empty-input tests**

Run: `python -m pytest tests/test_metrics.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit locally**

```bash
git add src/densetopo_unet/metrics.py tests/test_metrics.py
git commit -m "feat: add numerical and topology evaluation metrics"
```

### Task 11: English Command-Line Interface

**Files:**
- Create: `src/densetopo_unet/cli.py`
- Create: `src/densetopo_unet/__main__.py`
- Create: `tests/test_cli.py`

**Interfaces:**
- Consumes all service interfaces from Tasks 1-10.
- Produces: `build_parser() -> argparse.ArgumentParser`.
- Produces: `main(argv: Sequence[str] | None = None) -> int`.
- Commands: `validate-manifest`, `train`, `infer`, `evaluate`, `inspect-checkpoint`.

- [ ] **Step 1: Write failing command and information-boundary tests**

```python
def test_infer_help_contains_only_decompressed_input_metadata(capsys) -> None:
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["infer", "--help"])
    help_text = capsys.readouterr().out
    assert "--input" in help_text
    assert "--shape" in help_text
    assert "--reference" not in help_text
    assert "--false-cases" not in help_text
```

- [ ] **Step 2: Run tests and verify CLI is missing**

Run: `python -m pytest tests/test_cli.py -v`  
Expected: FAIL because `build_parser` does not exist.

- [ ] **Step 3: Implement command parsing and service dispatch**

All help text and errors are English. `validate-manifest` prints a machine-
readable JSON summary; `train`, `evaluate`, and `infer` refuse nonempty output
targets; `inspect-checkpoint` reads metadata without constructing a GPU model.

- [ ] **Step 4: Add exit-code, malformed-manifest, missing-file, and module-entry tests**

Run: `python -m pytest tests/test_cli.py -v`  
Expected: PASS and `python -m densetopo_unet --help` exits 0.

- [ ] **Step 5: Commit locally**

```bash
git add src/densetopo_unet/cli.py src/densetopo_unet/__main__.py tests/test_cli.py
git commit -m "feat: expose DenseTopo-UNet command line tools"
```

### Task 12: Deterministic Synthetic Workflow

**Files:**
- Create: `scripts/generate_synthetic_data.py`
- Create: `configs/synthetic.yaml`
- Create: `tests/test_workflow.py`
- Create: `scripts/smoke_test.sh`

**Interfaces:**
- Produces: `generate_dataset(output: Path, shape: tuple[int, int, int], seed: int) -> Path` returning the manifest path.
- Exercises manifest validation, CPU training, checkpoint inspection, and single-file inference.

- [ ] **Step 1: Write a failing synthetic-artifact test**

```python
def test_synthetic_generator_writes_complete_training_contract(tmp_path: Path) -> None:
    manifest_path = generate_dataset(tmp_path, shape=(8, 16, 16), seed=2026)
    manifest = yaml.safe_load(manifest_path.read_text())
    assert {sample["split"] for sample in manifest["samples"]} == {
        "train", "validation", "test"
    }
    assert all((manifest_path.parent / sample["decompressed"]).exists()
               for sample in manifest["samples"])
```

- [ ] **Step 2: Run tests and verify the generator is missing**

Run: `python -m pytest tests/test_workflow.py -v`  
Expected: FAIL because `generate_dataset` does not exist.

- [ ] **Step 3: Implement deterministic signed and nonnegative scalar fixtures and valid label CSV files**

The perturbation is called a `lossy-decompression proxy` and is not named after
any compressor. Use small analytic Gaussian peaks, seeded bounded perturbations,
and deterministic label coordinates.

- [ ] **Step 4: Implement and run the end-to-end CPU smoke test**

Run:

```bash
bash scripts/smoke_test.sh .artifacts/smoke
```

Expected: configuration validation, two training epochs, checkpoint inspection,
and one-volume inference all exit 0; the output raw file has the configured byte
count.

- [ ] **Step 5: Commit locally**

```bash
git add scripts configs/synthetic.yaml tests/test_workflow.py
git commit -m "test: add deterministic end-to-end workflow"
```

### Task 13: Academic Documentation, Native SVG Assets, and Repository Governance

**Files:**
- Replace: `README.md`
- Create: `assets/densetopo-unet-wordmark.svg`
- Create: `assets/architecture.svg`
- Create: `docs/index.md`
- Create: `docs/architecture.md`
- Create: `docs/data-contract.md`
- Create: `docs/configuration.md`
- Create: `docs/usage.md`
- Create: `docs/reproducibility.md`
- Create: `docs/topology-labels.md`
- Create: `docs/limitations.md`
- Create: `MODEL_CARD.md`
- Create: `CITATION.cff`
- Create: `CHANGELOG.md`
- Create: `CONTRIBUTING.md`
- Create: `CODE_OF_CONDUCT.md`
- Create: `SECURITY.md`
- Create: `.editorconfig`
- Create: `.gitattributes`
- Create: `.gitignore`
- Create: `Makefile`
- Create: `.github/CODEOWNERS`
- Create: `.github/PULL_REQUEST_TEMPLATE.md`
- Create: `.github/ISSUE_TEMPLATE/*.yml`
- Create: `.github/dependabot.yml`
- Create: `.github/workflows/ci.yml`
- Create: `tests/test_repository.py`

**Interfaces:**
- Documents every public config key, file schema, tensor dimension, CLI command,
  information boundary, limitation, and evidence-status statement.
- Uses the same engineering presentation standard as PTU-Net without copying its
  branding or making shared empirical claims.

- [ ] **Step 1: Write failing repository-integrity tests**

```python
def test_readme_states_decompressed_input_and_no_pretrained_weights() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "already decompressed" in readme
    assert "No pretrained checkpoint" in readme
    assert "[D, H, W]" in readme

def test_public_text_contains_no_cjk_characters() -> None:
    roots = [Path("README.md"), Path("MODEL_CARD.md"), Path("docs"), Path("src")]
    text = "\n".join(path.read_text(encoding="utf-8") for root in roots
                     for path in ([root] if root.is_file() else root.rglob("*"))
                     if path.is_file() and path.suffix in {".md", ".py"})
    assert not re.search(r"[\u3400-\u9fff]", text)
```

- [ ] **Step 2: Run tests and observe the README/documentation failure**

Run: `python -m pytest tests/test_repository.py -v`  
Expected: FAIL because the initial README and required documents do not satisfy
the repository contract.

- [ ] **Step 3: Create the English academic README and focused documents**

README order: wordmark, badges, title, evidence-status callout, method summary,
architecture figure, compressor examples, installation, synthetic workflow,
training, inference, data boundary, repository layout, documentation links,
contributing/security, citation, and no-license status.

State that SPERR, SZ3, ZFP, MGARD, and HPEZ are examples of upstream tools and
that the repository does not claim validation on their outputs.

- [ ] **Step 4: Draw repository-native accessible SVG assets**

The architecture figure shows:

```text
one decompressed [D,H,W] field
  -> normalized [1,D,H,W]
  -> encoder 12/24/48/96
  -> decoder 48/24/12
  -> correction + gate
  -> bounded residual addition
  -> restored [D,H,W] field
```

Both SVG files include `<title>` and `<desc>`, use readable colors, and contain
only English labels.

- [ ] **Step 5: Add governance, ignore rules, Make targets, and CPU CI**

CI runs Ruff, formatting, mypy, pytest with at least 75 percent coverage on
Python 3.10 and 3.12, then builds and installs the wheel. `.gitignore` excludes
raw fields, checkpoints, runs, outputs, caches, and environments while allowing
small reviewed test fixtures.

- [ ] **Step 6: Run repository tests and link checks**

Run:

```bash
python -m pytest tests/test_repository.py -v
python -m ruff check .
python -m ruff format --check .
python -m mypy src/densetopo_unet
```

Expected: all commands exit 0.

- [ ] **Step 7: Commit locally**

```bash
git add README.md MODEL_CARD.md CITATION.cff CHANGELOG.md CONTRIBUTING.md CODE_OF_CONDUCT.md SECURITY.md Makefile assets docs .github .editorconfig .gitattributes .gitignore tests/test_repository.py
git commit -m "docs: publish academic project documentation"
```

### Task 14: Final Local Verification and Delivery Audit

**Files:**
- Modify only files required by failures discovered during the audit.

**Interfaces:**
- Verifies the complete specification without changing the remote repository.

- [ ] **Step 1: Verify Git contains no large or prohibited artifacts**

Run:

```bash
if git ls-files | grep -Eq '\.(pt|pth|ckpt|f32|dat|npy|npz)$'; then
  echo "Tracked model weight or raw-volume artifact detected"
  exit 1
fi
git status --short
```

Expected: no weight/raw-data matches and no unexplained worktree changes.

- [ ] **Step 2: Run the complete quality suite in the `compressor` environment**

Run:

```bash
source /home/c_jxiao@lmumain.edu/anaconda3/etc/profile.d/conda.sh
conda activate compressor
python -m pip install -e '.[dev]' --no-deps
make check
```

Expected: Ruff, formatting, mypy, and all CPU tests exit 0 with coverage at or
above 75 percent.

- [ ] **Step 3: Run a fresh synthetic workflow**

Run: `bash scripts/smoke_test.sh .artifacts/final-smoke`  
Expected: manifest validation, training, checkpointing, and inference exit 0.

- [ ] **Step 4: Build and install the wheel in an isolated temporary virtual environment**

Run:

```bash
python -m build
DENSETOPO_WHEEL_ENV=$(mktemp -d /tmp/densetopo-wheel-check.XXXXXX)
python -m venv --system-site-packages "$DENSETOPO_WHEEL_ENV"
"$DENSETOPO_WHEEL_ENV/bin/python" -m pip install dist/densetopo_unet-0.1.0-py3-none-any.whl --no-deps
PYTHONPATH= "$DENSETOPO_WHEEL_ENV/bin/python" -m densetopo_unet --help
```

Expected: build, installation, and installed-module help all exit 0.

- [ ] **Step 5: Audit requirements against the design spec**

Check every section of
`docs/superpowers/specs/2026-08-26-densetopo-unet-repository-design.md` against
the repository tree, tests, CLI help, and documentation. Correct any uncovered
gap, rerun the affected test first in red-green order, then rerun `make check`.

- [ ] **Step 6: Create a final local-only commit if the audit changed files**

```bash
git add -A
git commit -m "chore: finalize DenseTopo-UNet research package"
git status --short
git log --oneline --decorate -15
```

Expected: clean local worktree with all implementation commits ahead of
`origin/main`. Do not push.
