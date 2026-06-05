#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
INPUT_JSONL=${INPUT_JSONL:-$ROOT_DIR/data/packages/math/train/numinamath_cot_100k_sdft.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT_DIR/data/derived/math_dual_100k}
PARTIAL_RATIO=${PARTIAL_RATIO:-0.5}

if [[ ! -f "$INPUT_JSONL" && "$INPUT_JSONL" == "$ROOT_DIR/"* ]]; then
  python "$ROOT_DIR/scripts/restore_github_chunks.py" \
    --root "$ROOT_DIR" \
    --manifest "$ROOT_DIR/data/manifests/github_chunks.json" \
    --only "${INPUT_JSONL#$ROOT_DIR/}" \
    --quiet
fi

python "$ROOT_DIR/scripts/build_dual_numina_asft_dataset.py" \
  --input_jsonl "$INPUT_JSONL" \
  --output_root "$OUTPUT_ROOT" \
  --partial_ratio "$PARTIAL_RATIO"
