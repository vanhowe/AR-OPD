#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
export ROOT_DIR

load_env_stack() {
  local hardware_env="$1"
  local base_env="$2"
  local domain_env="$3"
  local method_env="$4"
  # shellcheck disable=SC1090
  source "$hardware_env"
  # shellcheck disable=SC1090
  source "$base_env"
  # shellcheck disable=SC1090
  source "$domain_env"
  # shellcheck disable=SC1090
  source "$method_env"
}

latest_checkpoint() {
  local run_dir="$1"
  find "$run_dir" -maxdepth 1 -type d -name 'checkpoint-*' | sort -V | tail -n 1
}

resolve_eval_model_path() {
  local run_dir="$1"
  local final_dir="$run_dir/final"
  local latest_ckpt=""

  if [[ -d "$final_dir" ]]; then
    echo "$final_dir"
    return 0
  fi

  latest_ckpt="$(latest_checkpoint "$run_dir" || true)"
  if [[ -n "$latest_ckpt" ]]; then
    echo "$latest_ckpt"
    return 0
  fi

  echo "" >&2
  return 1
}

prepare_eval_model() {
  local checkpoint_dir="$1"
  local export_dir="$2"

  if [[ ! -d "$checkpoint_dir" ]]; then
    echo "$checkpoint_dir"
    return 0
  fi

  local sharded_dir="$checkpoint_dir/pytorch_model_fsdp_0"
  if [[ ! -d "$sharded_dir" ]]; then
    echo "$checkpoint_dir"
    return 0
  fi

  mkdir -p "$export_dir"
  if [[ ! -f "$export_dir/model.safetensors" && ! -f "$export_dir/pytorch_model.bin" ]]; then
    echo "[fsdp-export] merging sharded checkpoint $sharded_dir -> $export_dir" >&2
    python -m accelerate.commands.merge "$sharded_dir" "$export_dir" >&2
  else
    echo "[fsdp-export] reusing existing merged weights in $export_dir" >&2
  fi

  local pattern
  for pattern in \
    config.json \
    generation_config.json \
    tokenizer.json \
    tokenizer_config.json \
    special_tokens_map.json \
    added_tokens.json \
    merges.txt \
    vocab.json \
    chat_template.jinja \
    preprocessor_config.json \
    README.md; do
    if [[ -f "$checkpoint_dir/$pattern" && ! -f "$export_dir/$pattern" ]]; then
      cp "$checkpoint_dir/$pattern" "$export_dir/$pattern"
    fi
  done

  echo "$export_dir"
}

timestamp_utc() {
  date -u +%Y%m%d_%H%M%S
}

disk_free_gib() {
  local target_path="$1"
  python - "$target_path" <<'PY'
import shutil
import sys
from pathlib import Path

target = Path(sys.argv[1]).resolve()
probe = target
while not probe.exists():
    if probe.parent == probe:
        raise FileNotFoundError(target)
    probe = probe.parent
usage = shutil.disk_usage(probe)
print(usage.free / (1 << 30))
PY
}

apply_fsdp_checkpoint_guard() {
  local target_path="$1"
  local min_free_gib="${OVSDFT_FSDP_MIN_FREE_GB_FOR_OPTIMIZER_SAVE:-40}"
  local free_gib
  free_gib="$(disk_free_gib "$target_path")"

  echo "[checkpoint-guard] free_gib=${free_gib} target=${target_path}" >&2
  if [[ "${LATEST_CHECKPOINT_KEEP_OPTIMIZER:-0}" != "1" ]]; then
    return 0
  fi

  if python - "$free_gib" "$min_free_gib" <<'PY'
import sys
free_gib = float(sys.argv[1])
min_free_gib = float(sys.argv[2])
raise SystemExit(0 if free_gib < min_free_gib else 1)
PY
  then
    echo "[checkpoint-guard] disabling optimizer-state checkpointing because free disk ${free_gib} GiB < required ${min_free_gib} GiB" >&2
    LATEST_CHECKPOINT_KEEP_OPTIMIZER=0
  fi
}

ensure_gpu_count_or_exit() {
  local expected_gpus="$1"
  local visible_devices="${CUDA_VISIBLE_DEVICES:-}"
  local detected_gpus
  detected_gpus="$(
    python - <<'PY'
import torch
print(torch.cuda.device_count())
PY
  )"

  if [[ "$detected_gpus" -lt "$expected_gpus" ]]; then
    echo "[gpu-check] expected at least ${expected_gpus} visible GPUs, found ${detected_gpus}. CUDA_VISIBLE_DEVICES=${visible_devices:-<unset>}" >&2
    return 1
  fi

  echo "[gpu-check] detected ${detected_gpus} visible GPUs (need ${expected_gpus})" >&2
}
