#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

DOMAIN=${1:?domain required}
METHOD=${2:?method required}

HARDWARE_ENV=${HARDWARE_ENV:-$ROOT_DIR/configs/hardware/a800x16.env}
BASE_ENV=${BASE_ENV:-$ROOT_DIR/configs/base.env}
DOMAIN_ENV=${DOMAIN_ENV:-$ROOT_DIR/configs/domains/${DOMAIN}.env}
METHOD_ENV=${METHOD_ENV:-$ROOT_DIR/configs/methods/${METHOD}.env}
load_env_stack "$HARDWARE_ENV" "$BASE_ENV" "$DOMAIN_ENV" "$METHOD_ENV"

RUN_NAME=${RUN_NAME:-${DOMAIN}_${METHOD}_$(timestamp_utc)}
OUTPUT_DIR=${OUTPUT_DIR:-$CHECKPOINT_ROOT/$RUN_NAME}
TRAIN_LOG=${TRAIN_LOG:-$ARTIFACT_ROOT/${RUN_NAME}.log}
MODEL_PATH=${MODEL_PATH:-$MODEL_NAME}
FSDP_LAYER_CLS=${FSDP_LAYER_CLS:-$(python "$ROOT_DIR/scripts/detect_fsdp_layer.py" --model-name "$MODEL_PATH")}

mkdir -p "$OUTPUT_DIR" "$ARTIFACT_ROOT"

if [[ ! -f "$RAW_SFT_DATASET" ]]; then
  echo "Missing raw SFT dataset: $RAW_SFT_DATASET" >&2
  exit 1
fi

RESUME_ARGS=()
LATEST_CKPT="$(latest_checkpoint "$OUTPUT_DIR" || true)"
if [[ -n "$LATEST_CKPT" ]]; then
  RESUME_ARGS=(--resume_from_checkpoint "$LATEST_CKPT")
fi

export CUDA_VISIBLE_DEVICES
export OMP_NUM_THREADS
export WANDB_MODE=${WANDB_MODE:-offline}
export FSDP_STATE_DICT_TYPE=${FSDP_STATE_DICT_TYPE:-FULL_STATE_DICT}
ensure_gpu_count_or_exit "$NUM_GPUS"
LATEST_CHECKPOINT_KEEP_OPTIMIZER=${LATEST_CHECKPOINT_KEEP_OPTIMIZER:-0}
apply_fsdp_checkpoint_guard "$OUTPUT_DIR"
CHECKPOINT_MODE_ARG=()
if [[ "$LATEST_CHECKPOINT_KEEP_OPTIMIZER" == "1" ]]; then
  CHECKPOINT_MODE_ARG=(--latest_checkpoint_keep_optimizer)
else
  CHECKPOINT_MODE_ARG=(--no_latest_checkpoint_keep_optimizer)
fi

python -m torch.distributed.run \
  --nproc_per_node="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  "$ROOT_DIR/scripts/train_sft_reasoning.py" \
  --model_name "$MODEL_PATH" \
  --dataset_path "$RAW_SFT_DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --run_name "$RUN_NAME" \
  --learning_rate "$LEARNING_RATE" \
  --num_train_epochs "$NUM_TRAIN_EPOCHS" \
  --max_steps "$MAX_STEPS" \
  --per_device_train_batch_size "$PER_DEVICE_BATCH" \
  --gradient_accumulation_steps "$GRAD_ACC" \
  --save_steps "$SAVE_STEPS" \
  --save_total_limit "$SAVE_TOTAL_LIMIT" \
  --warmup_steps 10 \
  --max_length "$MAX_SFT_LENGTH" \
  --report_to "$REPORT_TO" \
  --seed "$SEED" \
  --fsdp "$FSDP_MODE" \
  --fsdp_transformer_layer_cls_to_wrap "$FSDP_LAYER_CLS" \
  "${CHECKPOINT_MODE_ARG[@]}" \
  "${RESUME_ARGS[@]}" 2>&1 | tee "$TRAIN_LOG"
