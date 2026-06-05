#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
from pathlib import Path
from string import Template

from datasets import Dataset
from huggingface_hub import hf_hub_download


SYSTEM_PROMPT = (
    "You are solving a math problem. Respond in the following format:\n"
    "<reasoning>\n...\n</reasoning>\n"
    "<answer>\n...\n</answer>\n\n"
    "For the answer, output only the final answer expression or value, without extra explanation."
)

TEACHER_TEMPLATE = Template(
    """
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""".strip()
)

WORD_CHAR_PATTERN = re.compile(r"[\w\\]", re.UNICODE)
REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)


def normalize_text(text: str) -> str:
    text = str(text).replace("\r\n", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def build_xml_trace(reasoning: str, answer: str) -> str:
    return f"<reasoning>\n{reasoning.strip()}\n</reasoning>\n<answer>\n{answer.strip()}\n</answer>"


def truncate_reasoning_prefix_half(solution: str, ratio: float) -> str:
    text = (solution or "").strip()
    if not text:
        return text
    if ratio >= 1.0:
        return text
    if len(text.split()) <= 2:
        return text

    cutoff = max(1, min(len(text), math.ceil(len(text) * ratio)))
    if cutoff >= len(text):
        return text

    def is_word_char(index: int) -> bool:
        return 0 <= index < len(text) and WORD_CHAR_PATTERN.match(text[index]) is not None

    if is_word_char(cutoff - 1) and is_word_char(cutoff):
        while cutoff < len(text) and is_word_char(cutoff):
            cutoff += 1

    truncated = text[:cutoff].rstrip()
    if truncated:
        return truncated

    first_token = text.split(maxsplit=1)[0]
    return first_token if first_token else text


def extract_last_boxed(text: str) -> str | None:
    best_start = max(text.rfind("\\boxed{"), text.rfind("\\fbox{"))
    if best_start == -1:
        return None
    brace_start = text.find("{", best_start)
    if brace_start == -1:
        return None
    depth = 0
    for idx in range(brace_start, len(text)):
        char = text[idx]
        if char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return normalize_text(text[brace_start + 1 : idx])
    return None


def parse_numina_response(response: str) -> tuple[str, str]:
    text = normalize_text(response)
    reasoning_match = REASONING_RE.search(text)
    answer_match = ANSWER_RE.search(text)
    if reasoning_match and answer_match:
        return normalize_text(reasoning_match.group(1)), normalize_text(answer_match.group(1))

    if "####" in text:
        reasoning, answer = text.rsplit("####", 1)
        return normalize_text(reasoning), normalize_text(answer)

    boxed = extract_last_boxed(text)
    if boxed:
        return text, boxed

    non_empty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    if non_empty_lines:
        return text, normalize_text(non_empty_lines[-1].rstrip("."))

    return text, text


def iter_jsonl(path: Path, limit: int | None = None):
    with path.open(encoding="utf-8") as f:
        seen = 0
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            seen += 1
            if limit is not None and seen >= limit:
                break


def count_jsonl_rows(path: Path, limit: int | None = None) -> int:
    count = 0
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                count += 1
                if limit is not None and count >= limit:
                    break
    return count


def save_jsonl(rows: list[dict], path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def fetch_source_file(args) -> Path:
    if args.input_jsonl is not None:
        return args.input_jsonl

    local_dir = args.output_root / "raw"
    local_dir.mkdir(parents=True, exist_ok=True)
    return Path(
        hf_hub_download(
            repo_id=args.repo_id,
            repo_type="dataset",
            filename=args.filename,
            local_dir=str(local_dir),
            endpoint=args.hf_endpoint,
        )
    )


def make_train_row(row: dict, partial_ratio: float, row_idx: int, source_name: str) -> dict:
    if "instruction" in row and "response" in row:
        question = normalize_text(row["instruction"])
        full_reasoning, answer = parse_numina_response(row["response"])
        system_prompt = SYSTEM_PROMPT
    elif "messages" in row and "output_text" in row:
        question = normalize_text(row["messages"][1]["content"])
        full_reasoning, answer = parse_numina_response(row["output_text"])
        system_prompt = normalize_text(row["messages"][0]["content"])
    else:
        raise KeyError("Expected either {instruction,response} or {messages,output_text} row format.")
    partial_reasoning = truncate_reasoning_prefix_half(full_reasoning, partial_ratio)
    full_trace = build_xml_trace(full_reasoning, answer)
    partial_trace = build_xml_trace(partial_reasoning, answer)

    prompt = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question},
    ]
    partial_teacher_prompt = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": TEACHER_TEMPLATE.substitute(orig_content=question, output_text=partial_trace),
        },
    ]
    full_teacher_prompt = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": TEACHER_TEMPLATE.substitute(orig_content=question, output_text=full_trace),
        },
    ]

    return {
        "prompt": prompt,
        "partial_teacher_prompt": partial_teacher_prompt,
        "full_teacher_prompt": full_teacher_prompt,
        "gold_answer": answer,
        "gold_trace": full_trace,
        "partial_trace": partial_trace,
        "problem": question,
        "source_split": "train",
        "source_repo": "released_asft_numina",
        "source_file": source_name,
        "source_index": row_idx,
        "partial_view_mode": "prefix_half_with_answer",
        "partial_ratio": partial_ratio,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build an AR-OPD-ready dataset from the released ASFT Numina CoT JSONL files."
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("data/derived/asft_numina_dual_10k"),
        help="Root directory for raw files, saved datasets, and manifest.",
    )
    parser.add_argument("--input_jsonl", type=Path, default=None, help="Optional local source JSONL path.")
    parser.add_argument("--repo_id", type=str, default="chichi56/ASFT", help="HF dataset repo id.")
    parser.add_argument("--filename", type=str, default="numina_cot_10k.jsonl", help="HF filename to download.")
    parser.add_argument(
        "--hf_endpoint",
        type=str,
        default="https://hf-mirror.com",
        help="HF endpoint used for download. Default uses the mirror for faster academic access.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional number of rows to keep from the source file.")
    parser.add_argument("--partial_ratio", type=float, default=0.5, help="Character ratio used for the partial view.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.partial_ratio <= 1.0:
        raise ValueError("--partial_ratio must be in (0, 1].")

    output_root = args.output_root
    raw_dir = output_root / "raw"
    ds_dir = output_root / "datasets"
    raw_dir.mkdir(parents=True, exist_ok=True)
    ds_dir.mkdir(parents=True, exist_ok=True)

    source_path = fetch_source_file(args)
    source_name = args.input_jsonl.name if args.input_jsonl is not None else args.filename

    if args.input_jsonl is not None:
        raw_copy = raw_dir / source_name
        if raw_copy.resolve() != args.input_jsonl.resolve():
            shutil.copy2(args.input_jsonl, raw_copy)

    def train_row_generator():
        for idx, row in enumerate(iter_jsonl(source_path, limit=args.limit)):
            yield make_train_row(row, partial_ratio=args.partial_ratio, row_idx=idx, source_name=source_name)

    dataset_suffix = source_name.replace(".jsonl", "")
    if args.limit is not None:
        dataset_suffix = f"{dataset_suffix}_{args.limit}"
    train_dataset_path = ds_dir / f"train_dual_{dataset_suffix}_prefixhalf"
    Dataset.from_generator(train_row_generator).save_to_disk(str(train_dataset_path))
    source_rows = count_jsonl_rows(source_path, limit=args.limit)

    manifest = {
        "source_repo_id": args.repo_id,
        "source_filename": args.filename,
        "source_local_path": str(source_path),
        "source_rows": source_rows,
        "partial_ratio": args.partial_ratio,
        "partial_view_mode": "prefix_half_with_answer",
        "train_dataset_path": str(train_dataset_path),
        "format_notes": {
            "source_fields": ["messages", "output_text"] if args.input_jsonl is not None else ["instruction", "response"],
            "train": "Dual-view dataset with prompt/partial_teacher_prompt/full_teacher_prompt/gold_answer.",
        },
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
