#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


METHOD_ORDER = ["sft", "asft", "sdft", "gdsdft_l06", "gdsdft_l08", "gdsdft_l12"]


def parse_float(value: str | None) -> float | None:
    if value in {None, ""}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def method_rank(method: str) -> int:
    return METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)


def main():
    parser = argparse.ArgumentParser(description="Render markdown paper tables from collected run metrics.")
    parser.add_argument("--results-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    with open(args.results_csv, encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    grouped = defaultdict(list)
    for row in rows:
        benchmark = row.get("benchmark") or "no_eval"
        subset = row.get("subset") or "aggregate"
        grouped[f"{benchmark}::{subset}"].append(row)

    out = Path(args.output_md)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        handle.write("# Paper Tables\n\n")
        for group_name in sorted(grouped):
            benchmark, subset = group_name.split("::", 1)
            title = benchmark if subset == "aggregate" else f"{benchmark} / {subset}"
            handle.write(f"## {title}\n\n")
            handle.write("| domain | method | run_name | step | accuracy | correct | total | min_loss | eval_path |\n")
            handle.write("| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n")
            for row in sorted(
                grouped[group_name],
                key=lambda item: (
                    item.get("domain") or "",
                    method_rank(item.get("method") or ""),
                    -(parse_float(item.get("accuracy")) or -1.0),
                    item.get("run_name") or "",
                ),
            ):
                accuracy = parse_float(row.get("accuracy"))
                accuracy_text = f"{accuracy:.4f}" if accuracy is not None else ""
                min_loss = parse_float(row.get("min_logged_loss"))
                min_loss_text = f"{min_loss:.4f}" if min_loss is not None else ""
                handle.write(
                    f"| {row.get('domain','')} | {row.get('method','')} | {row.get('run_name','')} | "
                    f"{row.get('global_step','')} | {accuracy_text} | {row.get('num_correct','')} | "
                    f"{row.get('num_total','')} | {min_loss_text} | {row.get('eval_path','')} |\n"
                )
            handle.write("\n")
    print(out)


if __name__ == "__main__":
    main()
