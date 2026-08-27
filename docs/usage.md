# Usage

## Prepare an experiment

1. Decompress each upstream bitstream outside this package.
2. Export every field as exact headerless `float32` in `[D, H, W]` order.
3. Generate training FC coordinates, reference extrema, and evaluation summaries with an external topology tool.
4. Copy and edit [`configs/model.yaml`](../configs/model.yaml).
5. Copy and edit [`configs/manifest.example.yaml`](../configs/manifest.example.yaml).
6. Keep train, validation, and test records scientifically disjoint.

## Validate before training

```bash
densetopo validate-manifest \
  --config path/to/experiment.yaml \
  --manifest path/to/manifest.yaml
```

Successful validation prints JSON containing the resolved manifest path, experiment name, sample count, and split counts. Validation reads each raw field in bounded chunks to check exact size and finite values; it can therefore take measurable time on large data.

## Train

```bash
densetopo train \
  --config path/to/experiment.yaml \
  --manifest path/to/manifest.yaml \
  --output runs/run-001
```

The output directory must be new or empty. A successful run writes:

| File | Purpose |
| --- | --- |
| `resolved_config.json` | Exact settings used by the process. |
| `environment.json` | Python, platform, NumPy, PyTorch, CUDA, and visible-device metadata. |
| `manifest.sha256` | Fingerprint of the exact manifest bytes. |
| `history.csv` | Per-epoch training and validation loss components. |
| `best.pt` | Checkpoint with the lowest validation total loss. |
| `latest.pt` | Most recent epoch, including optimizer and resume state. |
| `training_summary.json` | Stopping epoch, best epoch, and best score. |

Training length is controlled by `epochs`, `minimum_epochs`, and `early_stopping_patience`. Large defaults are intended to permit convergence. They are not a promise that a model converges, achieves the FC target, or needs the same wall time on different hardware.

## Resume

```bash
densetopo train \
  --config path/to/experiment.yaml \
  --manifest path/to/manifest.yaml \
  --output runs/run-001 \
  --resume runs/run-001/latest.pt
```

Resume validates the model, compression, normalization, loss, and value-domain configuration plus the exact manifest fingerprint. It restores model, optimizer, scheduler, precision scaler, epoch, best score, early-stopping counter, history, and random-number-generator states.

## Inspect a checkpoint

```bash
densetopo inspect-checkpoint --checkpoint runs/run-001/best.pt
```

Inspection loads the file on CPU and prints JSON metadata. It does not construct a model or reserve GPU memory.

## Infer from one file

```bash
densetopo infer \
  --checkpoint runs/run-001/best.pt \
  --input data/one.lossy.f32 \
  --shape 128 256 256 \
  --byte-order little \
  --batch-size 2 \
  --device auto \
  --output restored/one.restored.f32
```

Only one decompressed file is required. The command has no options for a reference, FC labels, critical points, or a topology summary. The output and sibling provenance JSON must not already exist. `batch-size` controls the number of 3D context patches processed together and may be reduced to lower inference memory.

## Evaluate

Place restored volumes under one directory using the exact name `<sample-id>.restored.f32`, then run:

```bash
densetopo evaluate \
  --config path/to/experiment.yaml \
  --manifest path/to/manifest.yaml \
  --restored-root restored \
  --split test \
  --output evaluations/numerical-001
```

This produces `evaluation_summary.json` and `metrics_by_sample.csv`. Each selected manifest record must provide a reference volume.

To add independent FC measurements, provide both directories:

```bash
densetopo evaluate \
  --config path/to/experiment.yaml \
  --manifest path/to/manifest.yaml \
  --restored-root restored \
  --split test \
  --baseline-topology-dir topology/decompressed \
  --restored-topology-dir topology/restored \
  --output evaluations/topology-001
```

The evaluation directory must be new or empty. See [topology labels](topology-labels.md) for the required JSON schema.

## Synthetic integration check

```bash
bash scripts/smoke_test.sh artifacts/smoke-001
```

Choose a new path for each invocation. The script does not delete or overwrite an earlier run. The generated data are analytic software fixtures, not scientific benchmark evidence.

Both command forms are supported:

```bash
densetopo --help
python -m densetopo_unet --help
```
