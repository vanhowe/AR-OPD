#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/common.sh"

DOMAIN=${1:?domain required}
MODEL_DIR=${2:?model dir required}
OUT_ROOT=${3:?out root required}
HARDWARE_ENV=${HARDWARE_ENV:-$ROOT_DIR/configs/hardware/a800x16.env}
BASE_ENV=${BASE_ENV:-$ROOT_DIR/configs/base.env}
DOMAIN_ENV=${DOMAIN_ENV:-$ROOT_DIR/configs/domains/${DOMAIN}.env}
METHOD_ENV=${METHOD_ENV:-$ROOT_DIR/configs/methods/sdft.env}
load_env_stack "$HARDWARE_ENV" "$BASE_ENV" "$DOMAIN_ENV" "$METHOD_ENV"
mkdir -p "$OUT_ROOT"
IFS=',' read -r -a requested_targets <<<"${EVAL_TARGETS:-}"

has_target() {
  local needle="$1"
  if [[ ${#requested_targets[@]} -eq 0 ]]; then
    return 0
  fi
  for target in "${requested_targets[@]}"; do
    if [[ "$target" == "$needle" ]]; then
      return 0
    fi
  done
  return 1
}

case "$DOMAIN" in
  math)
    for pair in "math500:$EVAL_MATH500" "aime_2024:$EVAL_AIME_2024" "numina_5k_test:$EVAL_NUMINA_5K_TEST"; do
      IFS=':' read -r name path <<<"$pair"
      has_target "$name" || continue
      [[ -d "$path" ]] || continue
      max_samples_var=""
      case "$name" in
        math500) max_samples_var="${EVAL_MATH500_MAX_SAMPLES:-}" ;;
        aime_2024) max_samples_var="${EVAL_AIME_2024_MAX_SAMPLES:-}" ;;
        numina_5k_test) max_samples_var="${EVAL_NUMINA_5K_TEST_MAX_SAMPLES:-}" ;;
      esac
      if [[ -n "$max_samples_var" ]]; then
        python "$ROOT_DIR/eval_math.py" --model_path "$MODEL_DIR" --dataset_path "$path" --output_dir "$OUT_ROOT/$name" --max_samples "$max_samples_var"
      else
        python "$ROOT_DIR/eval_math.py" --model_path "$MODEL_DIR" --dataset_path "$path" --output_dir "$OUT_ROOT/$name"
      fi
    done
    ;;
  code)
    if has_target "humaneval"; then
      python "$ROOT_DIR/eval_code.py" --model_path "$MODEL_DIR" --dataset humaneval --output_dir "$OUT_ROOT/humaneval"
    fi
    if has_target "mbpp"; then
      python "$ROOT_DIR/eval_code.py" --model_path "$MODEL_DIR" --dataset mbpp --output_dir "$OUT_ROOT/mbpp"
    fi
    ;;
  medical)
    python "$ROOT_DIR/eval_medical.py" --model_path "$MODEL_DIR" --output_dir "$OUT_ROOT/medical"
    ;;
  *)
    echo "Unsupported domain: $DOMAIN" >&2
    exit 1
    ;;
esac
