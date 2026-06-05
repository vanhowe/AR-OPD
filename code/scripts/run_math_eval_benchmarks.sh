#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT_DIR/scripts/common.sh"

MODEL_PATH=${MODEL_PATH:-}
RUN_DIR=${RUN_DIR:-}
EVAL_ROOT=${EVAL_ROOT:-$ROOT_DIR/eval_outputs}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-1024}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.85}
TENSOR_PARALLEL_SIZE=${TENSOR_PARALLEL_SIZE:-1}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
PYTHON_BIN=${PYTHON_BIN:-python}
BENCHMARKS=${BENCHMARKS:-math500,minerva_math,olympiadbench,aime_2024,amc_2023}

EVAL_DATA_ROOT=${EVAL_DATA_ROOT:-$ROOT_DIR/data/asft_math_data/eval_data}
if [[ ! -d "$EVAL_DATA_ROOT" ]]; then
  echo "Missing eval data root: $EVAL_DATA_ROOT" >&2
  echo "Run scripts/prepare_data_layout.sh or set EVAL_DATA_ROOT explicitly." >&2
  exit 1
fi

if [[ -z "$MODEL_PATH" ]]; then
  if [[ -z "$RUN_DIR" ]]; then
    echo "Either MODEL_PATH or RUN_DIR must be set." >&2
    exit 1
  fi
  MODEL_PATH="$(resolve_eval_model_path "$RUN_DIR")"
fi

if [[ ! -d "$MODEL_PATH" ]]; then
  echo "Resolved MODEL_PATH does not exist: $MODEL_PATH" >&2
  exit 1
fi

RUN_NAME=${RUN_NAME:-$(basename "${RUN_DIR:-$MODEL_PATH}")}
OUT_BASE="$EVAL_ROOT/$RUN_NAME"
mkdir -p "$OUT_BASE"

echo "[math-eval] model_path=$MODEL_PATH"
echo "[math-eval] run_name=$RUN_NAME"
echo "[math-eval] out_base=$OUT_BASE"
echo "[math-eval] eval_data_root=$EVAL_DATA_ROOT"
echo "[math-eval] benchmarks=$BENCHMARKS"

run_one_benchmark() {
  local benchmark_name="$1"
  local dataset_jsonl="$2"
  if [[ ! -f "$dataset_jsonl" ]]; then
    echo "Missing dataset for $benchmark_name: $dataset_jsonl" >&2
    exit 1
  fi

  "$PYTHON_BIN" "$ROOT_DIR/scripts/eval_math_benchmark.py" \
    --model_path "$MODEL_PATH" \
    --dataset_jsonl "$dataset_jsonl" \
    --benchmark_name "$benchmark_name" \
    --output_dir "$OUT_BASE/$benchmark_name" \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --gpu_memory_utilization "$GPU_MEMORY_UTILIZATION" \
    --tensor_parallel_size "$TENSOR_PARALLEL_SIZE" \
    --max_model_len "$MAX_MODEL_LEN"
}

IFS=',' read -r -a benchmark_list <<<"$BENCHMARKS"
for benchmark in "${benchmark_list[@]}"; do
  benchmark="$(echo "$benchmark" | xargs)"
  case "$benchmark" in
    math500|math_oai)
      run_one_benchmark "math500" "$EVAL_DATA_ROOT/math500_eval.jsonl"
      ;;
    minerva|minerva_math)
      run_one_benchmark "minerva_math" "$EVAL_DATA_ROOT/minerva_math_eval.jsonl"
      ;;
    olympiad|olympiadbench)
      run_one_benchmark "olympiadbench" "$EVAL_DATA_ROOT/olympiadbench_oe_to_maths_en_comp_eval.jsonl"
      ;;
    aime24|aime_2024)
      run_one_benchmark "aime_2024" "$EVAL_DATA_ROOT/aime_2024_eval.jsonl"
      ;;
    amc23|amc_2023)
      run_one_benchmark "amc_2023" "$EVAL_DATA_ROOT/amc_2023_eval.jsonl"
      ;;
    "")
      ;;
    *)
      echo "Unsupported benchmark name: $benchmark" >&2
      exit 1
      ;;
  esac
done
