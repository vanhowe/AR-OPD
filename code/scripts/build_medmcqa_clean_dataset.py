#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import re
from pathlib import Path
from string import Template

from datasets import Dataset, load_from_disk


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
VALID_ANSWER_RE = re.compile(r"^[ABCD]$", re.IGNORECASE)
CHOICE_LINE_RE = re.compile(r"^([ABCD]):\s*(.*)$")
NOT_RELATED_RE = re.compile(r"\bNOT\s+RELATED\b", re.IGNORECASE)
PUNCT_ONLY_RE = re.compile(r"^[^\w]+$", re.UNICODE)

# Conservative filters to drop traces that explicitly reveal the option letter in the rationale.
ANSWER_CUE_RE = re.compile(
    r"\b(?:ans(?:wer)?|correct\s+answer|correct\s+option|option\s*[ABCD]|[ABCD]\s+is\s+correct)\b",
    re.IGNORECASE,
)
LEADING_LETTER_RE = re.compile(
    r"^\s*\(?\s*[ABCD]\s*\)?\s*(?:[.:,\-]|\)|\]|\s|i\.?e\.?)",
    re.IGNORECASE,
)
LEADING_ANSWER_CUE_RE = re.compile(
    r"^\s*(?:the\s+)?(?:ans(?:wer)?|correct\s+answer|correct\s+option)\s*(?:is|:|=)?\s*",
    re.IGNORECASE,
)


def normalize_text(text: str) -> str:
    text = str(text).replace("\r\n", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def iter_json_rows(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def save_jsonl_rows(path: Path, rows):
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_local_rows(path: Path):
    if path.is_dir():
        if (path / "state.json").exists() and any(path.glob("*.arrow")):
            dataset = load_from_disk(str(path))
            for row in dataset:
                yield row
            return

        jsonl_files = sorted(path.glob("*.jsonl"))
        json_files = sorted(
            p for p in path.glob("*.json") if p.name not in {"manifest.json", "dataset_info.json", "state.json"}
        )
        for file_path in [*jsonl_files, *json_files]:
            yield from iter_json_rows(file_path)
        return

    if path.suffix.lower() in {".json", ".jsonl"}:
        yield from iter_json_rows(path)
        return

    dataset = load_from_disk(str(path))
    for row in dataset:
        yield row


def build_xml_trace(reasoning: str, answer: str) -> str:
    return f"<reasoning>\n{reasoning.strip()}\n</reasoning>\n<answer>\n{answer.strip()}\n</answer>"


def normalize_compare_text(text: str) -> str:
    text = normalize_text(text).rstrip(".")
    return re.sub(r"\s+", " ", text).lower()


def extract_choice_map(user_question: str) -> dict[str, str]:
    choices: dict[str, str] = {}
    for raw_line in str(user_question).splitlines():
        match = CHOICE_LINE_RE.match(raw_line.strip())
        if match:
            choices[match.group(1).upper()] = normalize_text(match.group(2))
    return choices


def question_has_artifact(user_question: str) -> bool:
    question_stem = str(user_question).split("\n\n", 1)[0]
    return bool(NOT_RELATED_RE.search(question_stem))


def is_punctuation_only(reasoning: str) -> bool:
    stripped = reasoning.strip()
    return bool(stripped) and PUNCT_ONLY_RE.fullmatch(stripped) is not None


def is_trivial_reasoning(reasoning: str) -> bool:
    return len(reasoning.split()) <= 5


def is_exact_answer_copy(reasoning: str, answer: str, user_question: str) -> bool:
    choice_map = extract_choice_map(user_question)
    answer_text = choice_map.get(answer.upper(), "")
    if not answer_text:
        return False
    return normalize_compare_text(reasoning) == normalize_compare_text(answer_text)


def mask_leading_answer_prefix(reasoning: str, answer: str, user_question: str) -> tuple[str, bool]:
    choice_map = extract_choice_map(user_question)
    answer_text = choice_map.get(answer.upper(), "").strip()
    if not answer_text:
        return reasoning, False

    pattern = re.compile(rf"^\s*{re.escape(answer_text)}(?=\b|[^\w]|$)", re.IGNORECASE)
    if pattern.search(reasoning):
        masked = pattern.sub("This option", reasoning, count=1).strip()
        return normalize_text(masked), masked != reasoning

    return reasoning, False


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


def parse_medical_response(response: str) -> tuple[str, str]:
    text = normalize_text(response)
    reasoning_match = REASONING_RE.search(text)
    answer_match = ANSWER_RE.search(text)
    reasoning = normalize_text(reasoning_match.group(1)) if reasoning_match else ""
    answer = normalize_text(answer_match.group(1)).upper() if answer_match else ""
    return reasoning, answer


def has_answer_leakage(reasoning: str) -> bool:
    return bool(ANSWER_CUE_RE.search(reasoning) or LEADING_LETTER_RE.search(reasoning))


def strip_leading_answer_prefix(reasoning: str, answer: str, user_question: str) -> tuple[str, bool]:
    text = normalize_text(reasoning)
    original = text

    text = LEADING_ANSWER_CUE_RE.sub("", text, count=1).lstrip()

    if LEADING_LETTER_RE.search(text):
        letter_match = re.match(r"^\s*\(?\s*([ABCD])\s*\)?\s*(?:[.:,\-]|\)|\]|\s|i\.?e\.?)+", text, re.IGNORECASE)
        if letter_match and letter_match.group(1).upper() == answer.upper():
            text = text[letter_match.end() :].lstrip()

    choice_map = extract_choice_map(user_question)
    answer_text = choice_map.get(answer.upper(), "").strip()
    if answer_text:
        answer_text_pattern = re.compile(rf"^\s*{re.escape(answer_text)}", re.IGNORECASE)
        text = answer_text_pattern.sub("", text, count=1).lstrip(" \t\n\r\f\v:;,-.)]")

    changed = normalize_text(text) != original
    if changed:
        text = normalize_text(text)
        if text and not text.lower().startswith("this option"):
            text = f"This option {text}"
        text = normalize_text(text)
    return text, changed


def build_teacher_prompt(system_message: dict, user_question: str, output_text: str) -> list[dict]:
    return [
        system_message,
        {
            "role": "user",
            "content": TEACHER_TEMPLATE.substitute(orig_content=user_question, output_text=output_text),
        },
    ]


def build_sdft_row(messages: list[dict], output_text: str, answer: str, source_index: int) -> dict:
    system_message = messages[0]
    user_question = normalize_text(messages[1]["content"])
    teacher_prompt = build_teacher_prompt(system_message, user_question, output_text)
    return {
        "prompt": messages,
        "counterfactual_prompt": messages,
        "teacher_prompt": teacher_prompt,
        "gold_answer": answer,
        "gold_trace": output_text,
        "output_text": output_text,
        "source_split": "train",
        "source_dataset": "medmcqa_clean",
        "source_index": source_index,
    }


def build_dual_row(messages: list[dict], output_text: str, answer: str, partial_ratio: float, source_index: int) -> dict:
    system_message = messages[0]
    user_question = normalize_text(messages[1]["content"])
    reasoning, _ = parse_medical_response(output_text)
    partial_reasoning = truncate_reasoning_prefix_half(reasoning, partial_ratio)
    partial_trace = build_xml_trace(partial_reasoning, answer)
    full_teacher_prompt = build_teacher_prompt(system_message, user_question, output_text)
    partial_teacher_prompt = build_teacher_prompt(system_message, user_question, partial_trace)
    return {
        "prompt": messages,
        "counterfactual_prompt": messages,
        "teacher_prompt": full_teacher_prompt,
        "partial_teacher_prompt": partial_teacher_prompt,
        "full_teacher_prompt": full_teacher_prompt,
        "gold_answer": answer,
        "gold_trace": output_text,
        "partial_trace": partial_trace,
        "output_text": output_text,
        "source_split": "train",
        "source_dataset": "medmcqa_clean",
        "source_index": source_index,
        "partial_view_mode": "prefix_half_with_answer",
        "partial_ratio": partial_ratio,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build a cleaned 5k MedMCQA dataset for SDfT and GD-SDFT training."
    )
    parser.add_argument(
        "--input_dataset",
        type=Path,
        default=Path("data/packages/medical/train/medmcqa_100k_sdft.jsonl"),
        help="Source MedMCQA JSONL or HF dataset path.",
    )
    parser.add_argument(
        "--output_root",
        type=Path,
        default=Path("data/derived/medmcqa_clean_5k"),
        help="Directory where cleaned raw and prepared datasets will be written.",
    )
    parser.add_argument("--target_size", type=int, default=5000, help="Number of cleaned rows to sample.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling.")
    parser.add_argument(
        "--partial_ratio",
        type=float,
        default=0.5,
        help="Character ratio used for the partial-view GD-SDFT teacher trace.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.target_size <= 0:
        raise ValueError("--target_size must be positive.")
    if not 0.0 < args.partial_ratio <= 1.0:
        raise ValueError("--partial_ratio must be in (0, 1].")

    output_root = args.output_root
    raw_dir = output_root / "raw"
    datasets_dir = output_root / "datasets"
    raw_dir.mkdir(parents=True, exist_ok=True)
    datasets_dir.mkdir(parents=True, exist_ok=True)

    chosen = []
    clean_candidate_rows = 0
    rng = random.Random(args.seed)
    stats = {
        "source_rows": 0,
        "skipped_invalid_messages": 0,
        "skipped_missing_reasoning": 0,
        "skipped_none_reasoning": 0,
        "skipped_invalid_answer": 0,
        "skipped_answer_leakage": 0,
        "skipped_question_artifact": 0,
        "skipped_punctuation_only_reasoning": 0,
        "skipped_trivial_reasoning": 0,
        "masked_leading_answer_cue": 0,
        "masked_leading_answer_prefix": 0,
        "skipped_exact_answer_copy_after_mask": 0,
    }

    for idx, row in enumerate(iter_local_rows(args.input_dataset)):
        stats["source_rows"] += 1
        messages = row.get("messages") or []
        if len(messages) < 2:
            stats["skipped_invalid_messages"] += 1
            continue

        user_question = normalize_text(messages[1]["content"])
        reasoning, answer = parse_medical_response(row.get("output_text", ""))
        if not reasoning:
            stats["skipped_missing_reasoning"] += 1
            continue
        if reasoning.lower() == "none":
            stats["skipped_none_reasoning"] += 1
            continue
        if not VALID_ANSWER_RE.fullmatch(answer):
            stats["skipped_invalid_answer"] += 1
            continue
        if question_has_artifact(user_question):
            stats["skipped_question_artifact"] += 1
            continue
        reasoning, masked_leading_cue = strip_leading_answer_prefix(reasoning, answer, user_question)
        if masked_leading_cue:
            stats["masked_leading_answer_cue"] += 1
        reasoning, masked_prefix = mask_leading_answer_prefix(reasoning, answer, user_question)
        if masked_prefix:
            stats["masked_leading_answer_prefix"] += 1
        if has_answer_leakage(reasoning):
            stats["skipped_answer_leakage"] += 1
            continue
        if is_punctuation_only(reasoning):
            stats["skipped_punctuation_only_reasoning"] += 1
            continue
        if is_trivial_reasoning(reasoning):
            stats["skipped_trivial_reasoning"] += 1
            continue
        if is_exact_answer_copy(reasoning, answer, user_question):
            stats["skipped_exact_answer_copy_after_mask"] += 1
            continue

        clean_trace = build_xml_trace(reasoning, answer)
        candidate = {
            "messages": messages,
            "output_text": clean_trace,
            "gold_answer": answer,
            "source_index": idx,
        }
        clean_candidate_rows += 1
        if len(chosen) < args.target_size:
            chosen.append(candidate)
        else:
            replacement_index = rng.randrange(clean_candidate_rows)
            if replacement_index < args.target_size:
                chosen[replacement_index] = candidate

    if clean_candidate_rows < args.target_size:
        raise ValueError(
            f"Only found {clean_candidate_rows} clean candidates, fewer than requested target_size={args.target_size}."
        )

    chosen.sort(key=lambda row: row["source_index"])

    raw_jsonl_path = raw_dir / f"medmcqa_{args.target_size}_sdft_clean.jsonl"
    save_jsonl_rows(
        raw_jsonl_path,
        ({"messages": row["messages"], "output_text": row["output_text"]} for row in chosen),
    )

    sdft_records_jsonl = raw_dir / f"medmcqa_{args.target_size}_sdft_records.jsonl"
    dual_records_jsonl = raw_dir / f"medmcqa_{args.target_size}_dual_records.jsonl"
    save_jsonl_rows(
        sdft_records_jsonl,
        (
            build_sdft_row(row["messages"], row["output_text"], row["gold_answer"], row["source_index"])
            for row in chosen
        ),
    )
    save_jsonl_rows(
        dual_records_jsonl,
        (
            build_dual_row(
                row["messages"],
                row["output_text"],
                row["gold_answer"],
                partial_ratio=args.partial_ratio,
                source_index=row["source_index"],
            )
            for row in chosen
        ),
    )

    sdft_dataset_path = datasets_dir / f"train_sdft_medmcqa_clean_{args.target_size}"
    dual_dataset_path = datasets_dir / f"train_dual_medmcqa_clean_{args.target_size}_prefixhalf"
    Dataset.from_generator(lambda: iter_json_rows(sdft_records_jsonl)).save_to_disk(str(sdft_dataset_path))
    Dataset.from_generator(lambda: iter_json_rows(dual_records_jsonl)).save_to_disk(str(dual_dataset_path))

    manifest = {
        "source_dataset_path": str(args.input_dataset),
        "source_rows": stats["source_rows"],
        "clean_candidate_rows": clean_candidate_rows,
        "selected_rows": len(chosen),
        "seed": args.seed,
        "partial_ratio": args.partial_ratio,
        "filtering": {
            "skip_invalid_messages": stats["skipped_invalid_messages"],
            "skip_missing_reasoning": stats["skipped_missing_reasoning"],
            "skip_none_reasoning": stats["skipped_none_reasoning"],
            "skip_invalid_answer": stats["skipped_invalid_answer"],
            "skip_answer_leakage": stats["skipped_answer_leakage"],
            "skip_question_artifact": stats["skipped_question_artifact"],
            "skip_punctuation_only_reasoning": stats["skipped_punctuation_only_reasoning"],
            "skip_trivial_reasoning": stats["skipped_trivial_reasoning"],
            "masked_leading_answer_cue": stats["masked_leading_answer_cue"],
            "masked_leading_answer_prefix": stats["masked_leading_answer_prefix"],
            "skip_exact_answer_copy_after_mask": stats["skipped_exact_answer_copy_after_mask"],
            "heuristics": [
                "drop rows with empty or literal 'None' reasoning",
                "mask leading answer cue prefixes such as 'Ans. is C' when they match the gold label",
                "drop rows whose remaining reasoning explicitly exposes the answer letter",
                "drop rows whose question stem contains dataset artifact markers such as 'NOT RELATED'",
                "replace leading answer-text prefixes in reasoning with 'This option'",
                "drop rows with punctuation-only reasoning",
                "drop rows with reasoning of 5 words or fewer",
                "drop rows whose reasoning is still exactly the gold answer text after masking",
                "keep the final answer only in the <answer> tag as a single A/B/C/D letter",
            ],
        },
        "artifacts": {
            "raw_jsonl": str(raw_jsonl_path),
            "sdft_records_jsonl": str(sdft_records_jsonl),
            "dual_records_jsonl": str(dual_records_jsonl),
            "sdft_dataset_path": str(sdft_dataset_path),
            "dual_dataset_path": str(dual_dataset_path),
        },
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
