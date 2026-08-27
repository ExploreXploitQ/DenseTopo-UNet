#!/usr/bin/env bash
set -euo pipefail

smoke_root="${1:-artifacts/smoke}"

python -m scripts.generate_synthetic_data --output "${smoke_root}/data"
densetopo validate-manifest \
  --config "${smoke_root}/data/experiment.yaml" \
  --manifest "${smoke_root}/data/manifest.yaml"
densetopo train \
  --config "${smoke_root}/data/experiment.yaml" \
  --manifest "${smoke_root}/data/manifest.yaml" \
  --output "${smoke_root}/training"
densetopo inspect-checkpoint \
  --checkpoint "${smoke_root}/training/best.pt"
densetopo infer \
  --checkpoint "${smoke_root}/training/best.pt" \
  --input "${smoke_root}/data/volumes/sample-003.lossy.f32" \
  --shape 8 16 16 \
  --output "${smoke_root}/sample-003.restored.f32" \
  --device cpu

echo "Smoke workflow completed in ${smoke_root}"
