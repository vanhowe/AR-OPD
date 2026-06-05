#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
for domain in math code medical; do
  bash "$ROOT_DIR/scripts/launch_domain_suite.sh" "$domain"
done
