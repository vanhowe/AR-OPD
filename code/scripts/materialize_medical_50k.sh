#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
INPUT_DATASET=${INPUT_DATASET:-$ROOT_DIR/data/packages/medical/train/medmcqa_100k_sdft.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT_DIR/data/derived/medical_clean_50k}
TARGET_SIZE=${TARGET_SIZE:-50000}
SEED=${SEED:-42}
PARTIAL_RATIO=${PARTIAL_RATIO:-0.5}

if [[ ! -f "$INPUT_DATASET" && "$INPUT_DATASET" == "$ROOT_DIR/"* ]]; then
  python "$ROOT_DIR/scripts/restore_github_chunks.py" \
    --root "$ROOT_DIR" \
    --manifest "$ROOT_DIR/data/manifests/github_chunks.json" \
    --only "${INPUT_DATASET#$ROOT_DIR/}" \
    --quiet
fi

python "$ROOT_DIR/scripts/build_medmcqa_clean_dataset.py"   --input_dataset "$INPUT_DATASET"   --output_root "$OUTPUT_ROOT"   --target_size "$TARGET_SIZE"   --seed "$SEED"   --partial_ratio "$PARTIAL_RATIO"
