#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset, load_from_disk


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a single-teacher-view medical dataset by replacing the full teacher view with the partial 0.5 view."
    )
    parser.add_argument(
        "--input_dataset",
        type=Path,
        default=Path("data/derived/medmcqa_clean_5k/datasets/train_dual_medmcqa_clean_5000_prefixhalf"),
        help="Prepared dual-view medical dataset.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("data/derived/medmcqa_clean_5k_partial_teacher"),
        help="Output root for the single-teacher-view dataset.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_from_disk(str(args.input_dataset))
    args.output_root.mkdir(parents=True, exist_ok=True)
    datasets_dir = args.output_root / "datasets"
    datasets_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for row in dataset:
        rows.append(
            {
                "prompt": row["prompt"],
                "counterfactual_prompt": row["counterfactual_prompt"],
                "teacher_prompt": row["partial_teacher_prompt"],
                "gold_answer": row["gold_answer"],
                "gold_trace": row["partial_trace"],
                "output_text": row["partial_trace"],
                "source_split": row.get("source_split", "train"),
                "source_dataset": "medmcqa_partial_teacher",
                "source_index": row.get("source_index"),
                "partial_view_mode": row.get("partial_view_mode", "prefix_half_with_answer"),
                "partial_ratio": row.get("partial_ratio", 0.5),
            }
        )

    output_path = datasets_dir / "train_sdft_medmcqa_partial_teacher_5000"
    Dataset.from_list(rows).save_to_disk(str(output_path))

    manifest = {
        "source_dataset_path": str(args.input_dataset),
        "output_dataset_path": str(output_path),
        "rows": len(rows),
        "notes": [
            "teacher_prompt is replaced with partial_teacher_prompt from the dual medical dataset.",
            "This is the single-teacher-view baseline requested by the user.",
        ],
    }
    with (args.output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
