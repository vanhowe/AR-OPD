#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)

bash "$ROOT_DIR/scripts/materialize_math_100k.sh"
bash "$ROOT_DIR/scripts/materialize_code_100k.sh"
bash "$ROOT_DIR/scripts/materialize_medical_50k.sh"
