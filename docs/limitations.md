# Limitations and evidence status

## Current evidence

The repository verifies software behavior with unit tests and a deterministic analytic workflow. It does not include a trained checkpoint, research dataset, output from a named compressor, comparison baseline, ablation, timing table, memory table, or measured FC-reduction result.

Accordingly, the following statements are implementation facts:

- the model accepts one decompressed scalar channel;
- inference does not accept original data or topology labels;
- corrections are bounded relative to the decompressed value by the configured scale;
- full-volume tiling writes every output voxel exactly once;
- the training objective contains numerical and local topology-derived terms;
- FC summaries can be aggregated when supplied by an external evaluator.

The following statements require new experiments and are not claimed:

- 90% FC removal on scientific data;
- preservation of an absolute error bound relative to the original field;
- generalization across datasets, variables, compressors, error bounds, or persistence values;
- improvement over compressor-integrated topology correction or other restoration networks;
- production-scale runtime, memory, stability, or fault tolerance.

## Scientific limitations

The 26-neighbor ordering term is local and differentiable. Persistent topology is global, threshold-dependent, and affected by matching conventions. Minimizing the surrogate can improve local ordering without eliminating an externally measured FP, FN, or FT, and can create new false cases elsewhere.

The neural residual is bounded around the decompressed value, not around the unavailable reference. If the upstream decompressed error already approaches the compressor bound, an additional correction in the wrong direction can exceed that reference-relative bound. The error-bound loss discourages this during supervised training but provides no inference-time proof.

One checkpoint assumes a coherent shape family, scalar value domain, normalization mode, absolute error bound, persistence threshold, and label protocol. Distribution shifts in spatial resolution, physical range, variable semantics, compressor artifacts, or topology density may reduce effectiveness.

Per-volume maximum normalization preserves zero but removes absolute amplitude from the neural input. The raw residual anchor and error-bound scaling retain original units; nevertheless, fields whose important structures depend on global amplitude may require a richer normalization study.

## Engineering limitations

Version 0.1 supports only headerless `float32` scalar grids in `zyx` order. It does not read metadata containers, irregular meshes, vector fields, adaptive grids, encoded compressor streams, or contour-tree files.

All volumes within one manifest must share dimensions and storage convention. Patch dimensions must be divisible by eight and cannot exceed the volume. Reflection padding and a four-level receptive field may behave poorly when important structures are larger than the patch context.

Training requires reference volumes and precomputed coordinate labels. Label quality, matching radius, persistence threshold, and extractor bugs directly affect supervision. The package records these settings but cannot verify their scientific correctness.

## Evaluation needed before publication

A defensible empirical claim should include multiple scientific fields, multiple timesteps or independent samples, named compressor outputs, fixed train-validation-test splits, several seeds, numerical error-bound violation counts, per-type FP/FN/FT results, runtime and memory, strong baselines, and ablations of gate, topology loss, critical weighting, tiling, and correction scale.

Any reported 90% target should state whether it refers to aggregate FCs, per-sample median, each FC type, or a success-rate threshold. It should also report samples where restoration increases FCs.
