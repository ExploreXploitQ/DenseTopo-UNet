# Topology labels and FC evaluation

## External topology boundary

DenseTopo-UNet does not extract critical points, compute persistence, or build contour trees. Users generate labels and evaluation summaries with an independent topology implementation under the same neighborhood, persistence threshold, matching rule, and coordinate convention used by the experiment.

This separation is intentional: a neural surrogate supplies gradients during optimization, whereas a domain topology tool supplies the final scientific measurement.

## False-case coordinate CSV

Each training and validation record points to a UTF-8 CSV with the exact header:

```csv
case,z,y,x
fn,10,20,30
fp,11,22,31
ft,12,24,32
```

Coordinates are zero-based integers satisfying `0 <= z < D`, `0 <= y < H`, and `0 <= x < W`. Labels are case-insensitive when loaded and are converted to lowercase.

- `fn`: a reference critical point not correctly recovered in the decompressed field.
- `fp`: a critical point present in the decompressed field without a matching reference point.
- `ft`: a matched location with an incorrect critical-point type according to the external evaluator.

The exact matching semantics remain the responsibility of the label generator. Document its version and settings with the experiment.

Sparse training weights are 5 for FN, 3 for FP, 4 for FT, and 1 for reference extrema. If multiple labels share a voxel, the maximum weight is retained.

## Reference-extrema CSV

The second file uses the exact header:

```csv
critical_type,z,y,x
local_maximum,10,20,30
local_minimum,12,24,32
```

Only `local_maximum` and `local_minimum` are accepted. The current differentiable loss uses the reference center-neighbor ordering rather than the text label to determine orientation, but strict vocabulary prevents ambiguous data from silently entering training.

## FC summary JSON for evaluation

Offline evaluation optionally reads one JSON file per sample from both a baseline directory and a restored directory. A summary must contain exact nonnegative integer counts:

```json
{
  "FP": 12,
  "FN": 8,
  "FT": 3
}
```

For sample ID `volume-001`, both directories must contain `volume-001.json`. Providing only one topology directory is an error. The aggregate count and reported removal fraction are:

```text
FC_total = FP + FN + FT
FC_removed_fraction = 1 - restored_FC_total / max(baseline_FC_total, 1)
```

A value of `0.90` corresponds to 90% fewer aggregate FCs. A negative value means the restored output contains more FCs. When the baseline count is zero, the denominator is clamped to one to keep the result defined; the individual counts should then be inspected instead of treating the fraction as a conventional reduction rate.

## Error-bound reporting

Evaluation also reports maximum absolute error, RMSE, PSNR, and the number of strict error-bound violations. A voxel is a violation only when `abs(restored - reference) > absolute_error_bound`. Equality is not counted.

FC summaries measure topology; error metrics measure numerical fidelity. Neither substitutes for the other. A credible experiment reports both and retains the raw per-sample results.
