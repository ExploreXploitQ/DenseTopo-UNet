# Data contract

## Raw scalar volumes

Version 0.1 accepts flat, headerless IEEE-754 `float32` volumes. Each file must use:

- logical axis order `zyx`;
- logical shape `[D, H, W]`;
- C-contiguous storage with `x` changing fastest;
- declared byte order `little` (`<f4`) or `big` (`>f4`);
- exactly `D * H * W` values and `D * H * W * 4` bytes;
- finite values only.

No dimensions, dtype, units, coordinates, compressor metadata, or error bound are inferred from the filename. Store scientific provenance separately and declare the machine-readable fields in the experiment configuration and manifest.

The package expects already decompressed values. It cannot read an encoded bitstream from SPERR, SZ3, ZFP, MGARD, HPEZ, or another compressor.

## Experiment and manifest separation

The experiment configuration records model and scientific settings. The manifest records sample paths, splits, and storage metadata. Keeping these files separate permits the same manifest structure to be used with different model-capacity studies while still storing the fully resolved configuration in each checkpoint.

All samples in one manifest share shape, dtype, byte order, and axis order. The configuration fixes one absolute error bound and one persistence threshold for the run. Mixing incompatible settings in one manifest is unsupported.

## Manifest schema

```yaml
schema_version: 1
experiment: example-topology-restoration
volume:
  shape: [32, 64, 96]
  dtype: float32
  byte_order: little
  axis_order: zyx
samples:
  - id: volume-001
    split: train
    decompressed: data/volume-001.lossy.f32
    reference: data/volume-001.reference.f32
    false_cases: labels/volume-001.false-cases.csv
    critical_points: labels/volume-001.critical-points.csv
```

`schema_version` must be integer `1`. `experiment` is a nonempty descriptive identifier. Manifest volume metadata must exactly match the experiment configuration. Sample IDs must be unique.

Paths may be absolute or relative. Relative paths resolve from the directory containing the manifest, not from the process working directory.

## Split requirements

| Split | Decompressed | Reference | FC labels | Reference extrema |
| --- | --- | --- | --- | --- |
| `train` | required | required | required | required |
| `validation` | required | required | required | required |
| `test` | required | optional | optional | optional |

The parser validates file existence, exact raw byte counts, finite raw values, exact CSV headers, coordinate ranges, and allowed label vocabulary before returning a manifest. It rejects unknown YAML keys to catch spelling errors.

Inference itself does not consume a manifest. The `infer` command receives one decompressed path, shape, and byte order; all learned and scientific settings required by the model come from the checkpoint.

## Value domains and normalization

`max_abs` calculates `max(abs(decompressed))` and divides every voxel by that positive scale. It is the normal choice for signed fields because both the negative minimum and positive maximum contribute through absolute value. It does not subtract the minimum or mean, so zero remains zero and the model keeps the physical zero anchor.

`positive_max` divides by `max(decompressed)` and is intended only for fields whose configured domain is nonnegative. Both methods reject a scale that is non-finite or no greater than `epsilon`.

Normalization affects the neural input only. The residual anchor, target, losses, output, and error-bound calculations remain in the original scalar units.

## Restored output

The restored volume uses the input shape and byte order and is written atomically as headerless `float32`. A sibling `<output>.json` records paths, hashes, scale, error bound, correction scale, device, runtime, dimensions, and package version. Output and provenance targets must not already exist.
