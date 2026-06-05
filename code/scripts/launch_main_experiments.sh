#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

MAIN_METHODS=${MAIN_METHODS:-"sft asft sdft gdsdft_l06 gdsdft_l08 gdsdft_l12"}
STAGE1_DOMAINS=${STAGE1_DOMAINS:-"math code"}
STAGE2_DOMAINS=${STAGE2_DOMAINS:-"medical"}

run_stage() {
  local stage_name="$1"
  local domains="$2"
  for domain in $domains; do
    echo "[main-experiments] stage=$stage_name domain=$domain methods=$MAIN_METHODS"
    METHODS="$MAIN_METHODS" \
      bash "$ROOT_DIR/scripts/launch_domain_suite.sh" "$domain"
  done
}

run_stage "stage1" "$STAGE1_DOMAINS"
run_stage "stage2" "$STAGE2_DOMAINS"
