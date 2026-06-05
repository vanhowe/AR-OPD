#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
INPUT_JSONL=${INPUT_JSONL:-$ROOT_DIR/data/packages/code/train/code_100k_source.jsonl}
OUTPUT_ROOT=${OUTPUT_ROOT:-$ROOT_DIR/data/derived/code_10k}
MODEL_NAME=${MODEL_NAME:-Qwen/Qwen3-4B}
TOKENIZER_MODEL_NAME=${TOKENIZER_MODEL_NAME:-$MODEL_NAME}
PARTIAL_RATIO=${PARTIAL_RATIO:-0.5}
MAX_RECORDS=${MAX_RECORDS:-10000}
MAX_PROMPT_TOKENS=${MAX_PROMPT_TOKENS:-1024}
MAX_COMPLETION_TOKENS=${MAX_COMPLETION_TOKENS:-1024}
MAX_TOTAL_TOKENS=${MAX_TOTAL_TOKENS:-2048}
LOCAL_CACHE_FALLBACK=${LOCAL_CACHE_FALLBACK:-}
LOCAL_TOKENIZER_FALLBACK=${LOCAL_TOKENIZER_FALLBACK:-}

if [[ ! -f "$INPUT_JSONL" && "$INPUT_JSONL" == "$ROOT_DIR/"* ]]; then
  python "$ROOT_DIR/scripts/restore_github_chunks.py" \
    --root "$ROOT_DIR" \
    --manifest "$ROOT_DIR/data/manifests/github_chunks.json" \
    --only "${INPUT_JSONL#$ROOT_DIR/}" \
    --quiet
fi

if [[ ! -e "$TOKENIZER_MODEL_NAME" && "$TOKENIZER_MODEL_NAME" == "Qwen/Qwen3-4B" && -n "$LOCAL_TOKENIZER_FALLBACK" && -d "$LOCAL_TOKENIZER_FALLBACK" ]]; then
  echo "[materialize_code_10k] tokenizer fallback: using local tokenizer at $LOCAL_TOKENIZER_FALLBACK for length filtering"
  TOKENIZER_MODEL_NAME="$LOCAL_TOKENIZER_FALLBACK"
fi

ARGS=(
  --model_name "$TOKENIZER_MODEL_NAME"
  --output_root "$OUTPUT_ROOT"
  --partial_ratio "$PARTIAL_RATIO"
  --max_records "$MAX_RECORDS"
  --max_prompt_tokens "$MAX_PROMPT_TOKENS"
  --max_completion_tokens "$MAX_COMPLETION_TOKENS"
  --max_total_tokens "$MAX_TOTAL_TOKENS"
)

if [[ -f "$INPUT_JSONL" ]]; then
  ARGS+=(--input_jsonl "$INPUT_JSONL")
elif [[ -n "$LOCAL_CACHE_FALLBACK" && -f "$LOCAL_CACHE_FALLBACK" ]]; then
  echo "[materialize_code_10k] packaged source missing; using local cache fallback at $LOCAL_CACHE_FALLBACK"
  ARGS+=(--input_jsonl "$LOCAL_CACHE_FALLBACK")
else
  echo "[materialize_code_10k] local source not found at $INPUT_JSONL; falling back to source_url download path in builder"
fi

python "$ROOT_DIR/scripts/build_dual_code_dataset.py" "${ARGS[@]}"
