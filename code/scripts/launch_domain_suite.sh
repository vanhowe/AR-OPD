#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

DOMAIN=${1:?domain required}
METHODS=${METHODS:-"sft asft sdft gdsdft_l06 gdsdft_l08 gdsdft_l12"}
CHECKPOINT_ROOT=${CHECKPOINT_ROOT:-$ROOT_DIR/checkpoints}
ARTIFACT_ROOT=${ARTIFACT_ROOT:-$ROOT_DIR/artifacts}

mkdir -p "$ARTIFACT_ROOT" "$CHECKPOINT_ROOT"

for METHOD in $METHODS; do
  RUN_NAME="${DOMAIN}_${METHOD}_$(timestamp_utc)"
  OUTPUT_DIR="$CHECKPOINT_ROOT/$RUN_NAME"
  ARTIFACT_DIR="$ARTIFACT_ROOT/$RUN_NAME"
  mkdir -p "$ARTIFACT_DIR"

  echo "[suite] domain=$DOMAIN method=$METHOD run_name=$RUN_NAME"

  case "$METHOD" in
    sft|asft)
      RUN_NAME="$RUN_NAME" OUTPUT_DIR="$OUTPUT_DIR" \
        bash "$ROOT_DIR/scripts/train_reasoning_sft_fsdp.sh" "$DOMAIN" "$METHOD"
      ;;
    sdft|gdsdft_l06|gdsdft_l08|gdsdft_l12)
      RUN_NAME="$RUN_NAME" OUTPUT_DIR="$OUTPUT_DIR" \
        bash "$ROOT_DIR/scripts/train_distill_fsdp.sh" "$DOMAIN" "$METHOD"
      ;;
    *)
      echo "Unknown method: $METHOD" >&2
      exit 1
      ;;
  esac

  MODEL_FOR_EVAL="$(latest_checkpoint "$OUTPUT_DIR" || true)"
  if [[ -z "$MODEL_FOR_EVAL" ]]; then
    MODEL_FOR_EVAL="$OUTPUT_DIR"
  fi
  MODEL_FOR_EVAL="$(prepare_eval_model "$MODEL_FOR_EVAL" "$ARTIFACT_DIR/exported_model")"

  bash "$ROOT_DIR/scripts/eval_domain_suite.sh" "$DOMAIN" "$MODEL_FOR_EVAL" "$ARTIFACT_DIR/evals"
done
