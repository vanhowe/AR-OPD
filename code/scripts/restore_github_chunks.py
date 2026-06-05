#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def selected(artifact: dict, only_paths: set[str]) -> bool:
    return not only_paths or artifact["restore_path"] in only_paths


def restore_artifact(root: Path, artifact: dict, force: bool = False, quiet: bool = False) -> str:
    dest = root / artifact["restore_path"]
    expected_sha = artifact["sha256"]
    expected_size = artifact["size_bytes"]

    if dest.exists() and not force:
        if dest.is_file() and dest.stat().st_size == expected_size and sha256_file(dest) == expected_sha:
            if not quiet:
                print(f"[restore_github_chunks] already valid: {artifact['restore_path']}")
            return "already_valid"
        if not quiet:
            print(f"[restore_github_chunks] refreshing invalid existing file: {artifact['restore_path']}")

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.with_name(dest.name + ".tmp")
    hasher = hashlib.sha256()
    written = 0

    with tmp_path.open("wb") as out_handle:
        for part in artifact["parts"]:
            part_path = root / part["path"]
            if not part_path.is_file():
                raise FileNotFoundError(f"Missing chunk file: {part_path}")
            payload = part_path.read_bytes()
            part_sha = hashlib.sha256(payload).hexdigest()
            if len(payload) != part["size_bytes"] or part_sha != part["sha256"]:
                raise ValueError(f"Chunk verification failed: {part_path}")
            out_handle.write(payload)
            hasher.update(payload)
            written += len(payload)

    final_sha = hasher.hexdigest()
    if written != expected_size or final_sha != expected_sha:
        tmp_path.unlink(missing_ok=True)
        raise ValueError(f"Restore verification failed for {dest}")

    os.replace(tmp_path, dest)
    if not quiet:
        print(f"[restore_github_chunks] restored: {artifact['restore_path']}")
    return "restored"


def main() -> None:
    parser = argparse.ArgumentParser(description="Restore full raw dataset files from GitHub-safe chunk files.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--manifest", default="data/manifests/github_chunks.json")
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    manifest_path = Path(args.manifest)
    if not manifest_path.is_absolute():
        manifest_path = root / manifest_path
    if not manifest_path.is_file():
        if not args.quiet:
            print(f"[restore_github_chunks] manifest not found, skipping: {manifest_path}")
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    only_paths = set(args.only)
    restored = 0
    already_valid = 0
    for artifact in manifest.get("artifacts", []):
        if not selected(artifact, only_paths):
            continue
        status = restore_artifact(root, artifact, force=args.force, quiet=args.quiet)
        if status == "restored":
            restored += 1
        else:
            already_valid += 1

    if not args.quiet:
        print(
            json.dumps(
                {
                    "manifest": str(manifest_path.relative_to(root)),
                    "restored": restored,
                    "already_valid": already_valid,
                    "selected": sorted(only_paths),
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
