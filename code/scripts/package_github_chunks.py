#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from package_local_assets import PACKAGE_MAP

CHUNK_TARGETS = (
    ("math", "train_raw"),
    ("code", "train_source"),
    ("medical", "train_raw"),
)


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_chunks(src: Path, chunk_dir: Path, chunk_size_bytes: int) -> list[dict]:
    chunk_dir.mkdir(parents=True, exist_ok=True)
    for stale in chunk_dir.glob("*.part-*"):
        stale.unlink()

    parts: list[dict] = []
    index = 0
    with src.open("rb") as handle:
        while True:
            payload = handle.read(chunk_size_bytes)
            if not payload:
                break
            chunk_name = f"{src.name}.part-{index:04d}"
            chunk_path = chunk_dir / chunk_name
            chunk_path.write_bytes(payload)
            parts.append(
                {
                    "path": str(chunk_path),
                    "size_bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
            index += 1

    if not parts:
        raise ValueError(f"No chunks were written for {src}.")
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Split repo-critical raw datasets into GitHub-safe chunk files.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument("--manifest-out", default="data/manifests/github_chunks.json")
    parser.add_argument("--chunk-size-mib", type=int, default=90)
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    manifest_path = repo_root / args.manifest_out
    chunk_size_bytes = args.chunk_size_mib * 1024 * 1024

    artifacts = []
    for domain, key in CHUNK_TARGETS:
        spec = PACKAGE_MAP[domain][key]
        src = Path(spec["src"])
        if not src.is_file():
            raise FileNotFoundError(f"Missing source file for {domain}:{key}: {src}")

        relative_chunk_dir = Path("data/github_chunks") / domain / key
        chunk_dir = repo_root / relative_chunk_dir
        parts = write_chunks(src, chunk_dir, chunk_size_bytes)
        artifacts.append(
            {
                "domain": domain,
                "key": key,
                "restore_path": spec["dst"],
                "source_path": str(src),
                "size_bytes": src.stat().st_size,
                "sha256": sha256_file(src),
                "parts": [
                    {
                        **part,
                        "path": str(Path(part["path"]).resolve().relative_to(repo_root)),
                    }
                    for part in parts
                ],
            }
        )

    manifest = {
        "version": 1,
        "chunk_size_bytes": chunk_size_bytes,
        "artifacts": artifacts,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
