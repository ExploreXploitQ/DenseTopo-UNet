# DenseTopo-UNet documentation

DenseTopo-UNet is research software for training and applying bounded neural corrections to lossy-decompressed three-dimensional scalar fields. Start with the data contract, then select the guide that matches the task.

## Core guides

- [Architecture](architecture.md): model inputs, feature channels, output heads, loss terms, and tiling.
- [Data contract](data-contract.md): raw volume and manifest schemas.
- [Configuration](configuration.md): every supported YAML key and validation rule.
- [Usage](usage.md): manifest validation, training, resume, inference, evaluation, and checkpoint inspection.
- [Topology labels](topology-labels.md): FC coordinate labels, reference extrema, and evaluation summaries.
- [Reproducibility](reproducibility.md): seeds, checkpoints, environment records, and experimental protocol.
- [Limitations](limitations.md): scientific assumptions, unsupported claims, and failure modes.
- [Model card](../MODEL_CARD.md): intended use and risk summary.

## Recommended reading order

1. Confirm that each volume satisfies the exact `float32` `[D, H, W]` storage contract.
2. Generate FC and reference-extrema labels with an independent topology tool.
3. Create one configuration for a fixed shape, error bound, persistence threshold, value domain, and normalization rule.
4. Create a manifest with disjoint train, validation, and test records.
5. Validate the manifest before starting a long run.
6. Train a checkpoint and inspect its recorded metadata.
7. Run inference using only a decompressed field.
8. Evaluate numerical error and independently generated FC summaries.

The repository includes no pretrained weights and no compressor-produced benchmark data.
