#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "$0")/.." && pwd)
MANIFEST="$ROOT_DIR/data/manifests/assets.json"
CHUNK_MANIFEST="$ROOT_DIR/data/manifests/github_chunks.json"
CHUNK_ROOT="$ROOT_DIR/data/github_chunks"

if [[ -f "$CHUNK_MANIFEST" && -d "$CHUNK_ROOT" ]]; then
  python "$ROOT_DIR/scripts/restore_github_chunks.py" \
    --root "$ROOT_DIR" \
    --manifest "$CHUNK_MANIFEST"
elif [[ -f "$CHUNK_MANIFEST" ]]; then
  echo "Skipping raw-train chunk restore because $CHUNK_ROOT is not bundled in this release."
fi

python "$ROOT_DIR/scripts/validate_assets.py" --manifest "$MANIFEST" --root "$ROOT_DIR"
