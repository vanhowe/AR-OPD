#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
source "$ROOT_DIR/scripts/common.sh"

TRAIN_PID=${TRAIN_PID:?TRAIN_PID required}
RUN_DIR=${RUN_DIR:?RUN_DIR required}
RUN_NAME=${RUN_NAME:?RUN_NAME required}

NEXT_RUN_NAME=${NEXT_RUN_NAME:-math10k_sdft_latter_partial_h20x2_server_$(date -u +%Y%m%d_%H%M%S)}
POLL_SECONDS=${POLL_SECONDS:-60}
POST_EXIT_GRACE_SECONDS=${POST_EXIT_GRACE_SECONDS:-20}
EVAL_GPU_IDS=${EVAL_GPU_IDS:-0}
EVAL_ROOT=${EVAL_ROOT:-$ROOT_DIR/eval_outputs}
EVAL_BENCHMARKS=${EVAL_BENCHMARKS:-math500,minerva_math,olympiadbench,aime_2024,amc_2023}
GLOBAL_BATCH=${GLOBAL_BATCH:-64}
MAX_COMPLETION_LENGTH=${MAX_COMPLETION_LENGTH:-1024}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-5}
SAVE_STEPS=${SAVE_STEPS:-25}

echo "[chain] wait_start pid=$TRAIN_PID run_name=$RUN_NAME run_dir=$RUN_DIR"
while kill -0 "$TRAIN_PID" >/dev/null 2>&1; do
  sleep "$POLL_SECONDS"
done

echo "[chain] train_pid_exited pid=$TRAIN_PID"
sleep "$POST_EXIT_GRACE_SECONDS"

EVAL_MODEL_PATH="$(resolve_eval_model_path "$RUN_DIR" || true)"
if [[ -z "$EVAL_MODEL_PATH" ]]; then
  echo "[chain] no final export or checkpoint found under $RUN_DIR" >&2
  exit 1
fi

echo "[chain] eval_model=$EVAL_MODEL_PATH"
CUDA_VISIBLE_DEVICES="$EVAL_GPU_IDS" \
MODEL_PATH="$EVAL_MODEL_PATH" \
RUN_DIR="$RUN_DIR" \
RUN_NAME="$RUN_NAME" \
EVAL_ROOT="$EVAL_ROOT" \
MAX_NEW_TOKENS="$MAX_COMPLETION_LENGTH" \
BENCHMARKS="$EVAL_BENCHMARKS" \
  bash "$ROOT_DIR/scripts/run_math_eval_benchmarks.sh"

echo "[chain] eval_done run_name=$RUN_NAME"
echo "[chain] launching_next_run=$NEXT_RUN_NAME"
