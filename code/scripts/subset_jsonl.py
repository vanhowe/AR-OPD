#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Copy the first N non-empty lines from a JSONL file.")
    parser.add_argument("--input", required=True, help="Source JSONL path")
    parser.add_argument("--output", required=True, help="Destination JSONL path")
    parser.add_argument("--limit", required=True, type=int, help="Maximum number of non-empty lines to copy")
    args = parser.parse_args()

    src = Path(args.input)
    dst = Path(args.output)
    dst.parent.mkdir(parents=True, exist_ok=True)

    kept = 0
    with src.open("r", encoding="utf-8") as fin, dst.open("w", encoding="utf-8") as fout:
        for line in fin:
            if not line.strip():
                continue
            fout.write(line)
            kept += 1
            if kept >= args.limit:
                break

    print(dst)
    print(f"rows={kept}")


if __name__ == "__main__":
    main()
