#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
DERIVED_ROOT=${DERIVED_ROOT:-$ROOT_DIR/data/derived}

DOMAINS=${DOMAINS:-"math code"}
PARTIAL_RATES=${PARTIAL_RATES:-"0.25 0.50 0.75"}
BASELINE_PARTIAL_RATIO=${BASELINE_PARTIAL_RATIO:-0.50}
BASELINE_METHODS=${BASELINE_METHODS:-"sft asft sdft"}
SWEEP_METHOD=${SWEEP_METHOD:-gdsdft_l08}
LIMIT=${LIMIT:-10000}
EVAL_TARGETS_MATH_OVERRIDE=${EVAL_TARGETS_MATH_OVERRIDE:-math500,aime_2024,numina_5k_test}
EVAL_TARGETS_CODE_OVERRIDE=${EVAL_TARGETS_CODE_OVERRIDE:-humaneval,mbpp}
MAX_PROMPT_LENGTH_OVERRIDE=${MAX_PROMPT_LENGTH_OVERRIDE:-1024}
MAX_COMPLETION_LENGTH_OVERRIDE=${MAX_COMPLETION_LENGTH_OVERRIDE:-1024}
MAX_SFT_LENGTH_OVERRIDE=${MAX_SFT_LENGTH_OVERRIDE:-2048}
SKIP_MATERIALIZE=${SKIP_MATERIALIZE:-0}
SKIP_BASELINES=${SKIP_BASELINES:-0}
SKIP_SWEEP=${SKIP_SWEEP:-0}
MATERIALIZE_ONLY=${MATERIALIZE_ONLY:-0}

ratio_tag() {
  python - "$1" <<'PY'
import sys
ratio = float(sys.argv[1])
print(f"pr{int(round(ratio * 100)):03d}")
PY
}

materialize_domain_ratio() {
  local domain="$1"
  local ratio="$2"
  local tag
  tag=$(ratio_tag "$ratio")

  if [[ "$domain" == "math" ]]; then
    local output_root="$DERIVED_ROOT/math_dual_10k_${tag}"
    if [[ "$SKIP_MATERIALIZE" != "1" ]]; then
      PARTIAL_RATIO="$ratio" LIMIT="$LIMIT" OUTPUT_ROOT="$output_root" \
        bash "$ROOT_DIR/scripts/materialize_math_10k.sh" >&2
    fi
    local raw_sft="$output_root/raw/numinamath_cot_100k_sdft_${LIMIT}.jsonl"
    local dual_dataset="$output_root/datasets/train_dual_numinamath_cot_100k_sdft_${LIMIT}_prefixhalf"
    local env_path="$output_root/domain.env"
    cat >"$env_path" <<EOF
DOMAIN=math
RAW_SFT_DATASET=$raw_sft
DUAL_DATASET=$dual_dataset
EVAL_TARGETS=$EVAL_TARGETS_MATH_OVERRIDE
EVAL_MATH500=$ROOT_DIR/data/packages/math/eval/eval_math500
EVAL_AIME_2024=$ROOT_DIR/data/packages/math/eval/eval_aime_2024
EVAL_NUMINA_5K_TEST=$ROOT_DIR/data/packages/math/eval/eval_numina_5k_test
EOF
    printf '%s\n' "$env_path"
    return
  fi

  if [[ "$domain" == "code" ]]; then
    local output_root="$DERIVED_ROOT/code_10k_${tag}"
    if [[ "$SKIP_MATERIALIZE" != "1" ]]; then
      PARTIAL_RATIO="$ratio" MAX_RECORDS="$LIMIT" OUTPUT_ROOT="$output_root" \
        MAX_PROMPT_TOKENS="$MAX_PROMPT_LENGTH_OVERRIDE" \
        MAX_COMPLETION_TOKENS="$MAX_COMPLETION_LENGTH_OVERRIDE" \
        MAX_TOTAL_TOKENS="$((MAX_PROMPT_LENGTH_OVERRIDE + MAX_COMPLETION_LENGTH_OVERRIDE))" \
        bash "$ROOT_DIR/scripts/materialize_code_10k.sh" >&2
    fi
    local env_path="$output_root/domain.env"
    python - "$output_root/manifest.json" "$ROOT_DIR" "$env_path" <<'PY'
import json
import os
import sys
from pathlib import Path

manifest = json.loads(Path(sys.argv[1]).read_text())
root = sys.argv[2]
env_path = Path(sys.argv[3])
env_path.parent.mkdir(parents=True, exist_ok=True)
env_path.write_text(
    "\n".join(
        [
            "DOMAIN=code",
            f"RAW_SFT_DATASET={manifest['baseline_jsonl']}",
            f"DUAL_DATASET={manifest['dual_dataset_path']}",
            f"EVAL_TARGETS={os.environ.get('EVAL_TARGETS_CODE_OVERRIDE', 'humaneval,mbpp')}",
        ]
    )
    + "\n",
    encoding="utf-8",
)
print(env_path)
PY
    return
  fi

  echo "Unsupported domain: $domain" >&2
  exit 1
}

run_method() {
  local base_domain="$1"
  local domain_label="$2"
  local method="$3"
  local tag="$4"
  local domain_env="$5"
  local run_name="${domain_label}_${method}_${tag}_$(date -u +%Y%m%d_%H%M%S)"
  local method_env="$ROOT_DIR/configs/methods/${method}.env"
  local checkpoint_dir="$CHECKPOINT_ROOT/$run_name"
  local eval_root="$ARTIFACT_ROOT/$run_name/evals"

  echo "[10k-sweep] domain=$domain_label base_domain=$base_domain method=$method tag=$tag"

  if [[ "$method" == "sft" || "$method" == "asft" ]]; then
    DOMAIN_ENV="$domain_env" METHOD_ENV="$method_env" \
      MAX_SFT_LENGTH="$MAX_SFT_LENGTH_OVERRIDE" \
      RUN_NAME="$run_name" OUTPUT_DIR="$checkpoint_dir" \
      bash "$ROOT_DIR/scripts/train_reasoning_sft_fsdp.sh" "$base_domain" "$method"
  else
    DOMAIN_ENV="$domain_env" METHOD_ENV="$method_env" \
      MAX_PROMPT_LENGTH="$MAX_PROMPT_LENGTH_OVERRIDE" \
      MAX_COMPLETION_LENGTH="$MAX_COMPLETION_LENGTH_OVERRIDE" \
      RUN_NAME="$run_name" OUTPUT_DIR="$checkpoint_dir" \
      bash "$ROOT_DIR/scripts/train_distill_fsdp.sh" "$base_domain" "$method"
  fi

  local model_for_eval
  model_for_eval=$(find "$checkpoint_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1 || true)
  if [[ -z "$model_for_eval" ]]; then
    model_for_eval="$checkpoint_dir"
  fi
  model_for_eval=$(prepare_eval_model "$model_for_eval" "$ARTIFACT_ROOT/$run_name/exported_model")

  DOMAIN_ENV="$domain_env" METHOD_ENV="$method_env" \
    bash "$ROOT_DIR/scripts/eval_domain_suite.sh" \
    "$base_domain" \
    "$model_for_eval" \
    "$eval_root"
}

for domain in $DOMAINS; do
  case "$domain" in
    math)
      domain_label="math10k"
      ;;
    code)
      domain_label="code10k"
      ;;
    *)
      echo "Unsupported domain: $domain" >&2
      exit 1
      ;;
  esac

  baseline_tag=$(ratio_tag "$BASELINE_PARTIAL_RATIO")
  baseline_env=$(materialize_domain_ratio "$domain" "$BASELINE_PARTIAL_RATIO")

  for ratio in $PARTIAL_RATES; do
    materialize_domain_ratio "$domain" "$ratio" >/dev/null
  done

  if [[ "$MATERIALIZE_ONLY" == "1" ]]; then
    continue
  fi

  if [[ "$SKIP_BASELINES" != "1" ]]; then
    for method in $BASELINE_METHODS; do
      run_method "$domain" "$domain_label" "$method" "$baseline_tag" "$baseline_env"
    done
  fi

  if [[ "$SKIP_SWEEP" != "1" ]]; then
    for ratio in $PARTIAL_RATES; do
      tag=$(ratio_tag "$ratio")
      env_path="$DERIVED_ROOT/${domain}_10k_${tag}/domain.env"
      if [[ "$domain" == "math" ]]; then
        env_path="$DERIVED_ROOT/math_dual_10k_${tag}/domain.env"
      fi
      run_method "$domain" "$domain_label" "$SWEEP_METHOD" "$tag" "$env_path"
    done
  fi
done
