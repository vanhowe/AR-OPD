#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--chunk-manifest')
    args = parser.parse_args()
    root = Path(args.root)
    manifest = json.loads(Path(args.manifest).read_text())
    recoverable = set()
    chunk_manifest_arg = args.chunk_manifest or manifest.get('github_chunk_manifest')
    if chunk_manifest_arg:
        chunk_manifest_path = Path(chunk_manifest_arg)
        if not chunk_manifest_path.is_absolute():
            chunk_manifest_path = root / chunk_manifest_path
        if chunk_manifest_path.exists():
            chunk_manifest = json.loads(chunk_manifest_path.read_text())
            recoverable = {artifact["restore_path"] for artifact in chunk_manifest.get("artifacts", [])}
    missing = []
    chunkable_missing = []
    for domain, entries in manifest.get('expected_packages', {}).items():
        for key, rel in entries.items():
            if rel == 'evalplus':
                continue
            path = root / rel
            if not path.exists():
                if rel in recoverable:
                    chunkable_missing.append((domain, key, str(path)))
                else:
                    missing.append((domain, key, str(path)))
    if missing:
        print('Missing assets:')
        for item in missing:
            print('	'.join(item))
        raise SystemExit(1)
    if chunkable_missing:
        print('Recoverable from GitHub chunks:')
        for item in chunkable_missing:
            print('	'.join(item))
        print('All required non-chunk assets are present.')
    else:
        print('All expected assets are present.')

if __name__ == '__main__':
    main()
