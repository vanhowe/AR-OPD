#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
VENV_PATH=${VENV_PATH:-$ROOT_DIR/.venv}
PYTHON_BIN=${PYTHON_BIN:-python3}

$PYTHON_BIN -m venv "$VENV_PATH"
# shellcheck disable=SC1091
source "$VENV_PATH/bin/activate"
pip install --upgrade pip wheel setuptools
pip install -r "$ROOT_DIR/requirements.txt"
