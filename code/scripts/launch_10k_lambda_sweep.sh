#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

DOMAINS=${DOMAINS:-"math code"}
PARTIAL_RATIO=${PARTIAL_RATIO:-0.50}
BASELINE_METHODS=${BASELINE_METHODS:-"sft asft sdft"}
SWEEP_METHODS=${SWEEP_METHODS:-"gdsdft_l04 gdsdft_l06 gdsdft_l08 gdsdft_l10 gdsdft_l12 gdsdft_l14"}
INCLUDE_BASELINES=${INCLUDE_BASELINES:-1}
SKIP_MATERIALIZE=${SKIP_MATERIALIZE:-0}

first_method=1
for method in $SWEEP_METHODS; do
  extra_baseline_flag=1
  extra_materialize_flag=1

  if [[ "$first_method" == "1" ]]; then
    extra_materialize_flag=$SKIP_MATERIALIZE
    if [[ "$INCLUDE_BASELINES" == "1" ]]; then
      extra_baseline_flag=0
    fi
  fi

  echo "[10k-lambda-sweep] method=$method partial_ratio=$PARTIAL_RATIO domains=$DOMAINS"
  DOMAINS="$DOMAINS" \
  PARTIAL_RATES="$PARTIAL_RATIO" \
  BASELINE_PARTIAL_RATIO="$PARTIAL_RATIO" \
  BASELINE_METHODS="$BASELINE_METHODS" \
  SWEEP_METHOD="$method" \
  SKIP_MATERIALIZE="$extra_materialize_flag" \
  SKIP_BASELINES="$extra_baseline_flag" \
  bash "$ROOT_DIR/scripts/launch_10k_partial_rate_sweep.sh"

  first_method=0
done
