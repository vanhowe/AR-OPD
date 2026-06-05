#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path


PACKAGE_MAP = {
    "math": {
        "train_raw": {
            "src": "local://asft_math_data/train_data/numinamath_cot_100k_sdft.jsonl",
            "dst": "data/packages/math/train/numinamath_cot_100k_sdft.jsonl",
        },
        "eval_math500": {
            "src": "local://asft_math_data/eval_disk/eval_math500",
            "dst": "data/packages/math/eval/eval_math500",
        },
        "eval_aime_2024": {
            "src": "local://asft_math_data/eval_disk/eval_aime_2024",
            "dst": "data/packages/math/eval/eval_aime_2024",
        },
        "eval_numina_5k_test": {
            "src": "local://asft_math_data/eval_disk/eval_numina_5k_test",
            "dst": "data/packages/math/eval/eval_numina_5k_test",
        },
    },
    "code": {
        "train_source": {
            "src": "local://code/train/data-evol_instruct-decontaminated.jsonl",
            "dst": "data/packages/code/train/code_100k_source.jsonl",
        },
    },
    "medical": {
        "train_raw": {
            "src": "local://asft_medical_data/train_data/medmcqa_100k_sdft.jsonl",
            "dst": "data/packages/medical/train/medmcqa_100k_sdft.jsonl",
        },
        "eval_mmlu_medical": {
            "src": "local://asft_medical_data/eval_data/mmlu_medical_eval.jsonl",
            "dst": "data/packages/medical/eval/mmlu_medical_eval.jsonl",
        },
        "eval_medqa": {
            "src": "local://asft_medical_data/eval_data/medqa_usmle_4_options_eval.jsonl",
            "dst": "data/packages/medical/eval/medqa_usmle_4_options_eval.jsonl",
        },
        "eval_medmcqa": {
            "src": "local://asft_medical_data/eval_data/medmcqa_test_eval.jsonl",
            "dst": "data/packages/medical/eval/medmcqa_test_eval.jsonl",
        },
    },
}


def sha256_of_path(path: Path) -> str:
    hasher = hashlib.sha256()
    if path.is_file():
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    for child in sorted(p for p in path.rglob("*") if p.is_file()):
        hasher.update(str(child.relative_to(path)).encode("utf-8"))
        with child.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1 << 20), b""):
                hasher.update(chunk)
    return hasher.hexdigest()


def copy_entry(src: Path, dst: Path):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        if dst.exists():
            shutil.rmtree(dst)
        shutil.copytree(src, dst)
        return
    shutil.copy2(src, dst)


def resolve_source(src: str, asset_root: Path) -> Path:
    if src.startswith("local://"):
        return asset_root / src.removeprefix("local://")
    return Path(src)


def main():
    parser = argparse.ArgumentParser(description="Copy locally available training/eval assets into the collaborator repo.")
    parser.add_argument("--repo-root", required=True)
    parser.add_argument(
        "--asset-root",
        default=os.environ.get("AROPD_LOCAL_ASSET_ROOT", "."),
        help="Root used to resolve local:// sources. Can also be set with AROPD_LOCAL_ASSET_ROOT.",
    )
    args = parser.parse_args()

    repo_root = Path(args.repo_root).resolve()
    asset_root = Path(args.asset_root).resolve()
    records = []
    missing = []

    for domain, entries in PACKAGE_MAP.items():
        for key, spec in entries.items():
            src = resolve_source(spec["src"], asset_root)
            dst = repo_root / spec["dst"]
            if not src.exists():
                missing.append({"domain": domain, "key": key, "src": spec["src"], "resolved": str(src)})
                continue

            copy_entry(src, dst)
            records.append(
                {
                    "domain": domain,
                    "key": key,
                    "src": spec["src"],
                    "dst": str(dst.relative_to(repo_root)),
                    "sha256": sha256_of_path(dst),
                }
            )

    checksums_path = repo_root / "data/manifests/local_asset_checksums.json"
    checksums_path.write_text(json.dumps(records, indent=2) + "\n", encoding="utf-8")

    summary = {"copied": records, "missing": missing}
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
