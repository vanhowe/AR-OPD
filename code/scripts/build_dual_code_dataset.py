#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from string import Template

import requests
from datasets import load_dataset
from transformers import AutoTokenizer


CODE_SYSTEM_PROMPT = """You are a careful software engineer. Solve the coding task and respond in the following format:
<reasoning>
Briefly explain the implementation strategy and any important edge cases.
</reasoning>
<answer>
```<language>
# solution
```
</answer>

Do not include any text outside these tags."""

TEACHER_TEMPLATE = Template(
    """
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""".strip()
)

MAGICODER_URL = "https://hf-mirror.com/datasets/ise-uiuc/Magicoder-Evol-Instruct-110K/resolve/main/data-evol_instruct-decontaminated.jsonl"
WORD_CHAR_PATTERN = re.compile(r"[\w\\]", re.UNICODE)
CODE_BLOCK_RE = re.compile(r"```([a-zA-Z0-9_+-]*)\s*(.*?)```", re.DOTALL)

def normalize_text(text: str) -> str:
    text = str(text).replace("\r\n", "\n").strip()
    return re.sub(r"\n{3,}", "\n\n", text)


def build_code_output_text(reasoning: str, code: str, language: str = "python") -> str:
    language = normalize_text(language or "python")
    return (
        "<reasoning>\n"
        f"{normalize_text(reasoning)}\n"
        "</reasoning>\n"
        "<answer>\n"
        f"```{language}\n"
        f"{normalize_text(code)}\n"
        "```\n"
        "</answer>"
    )


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


def extract_code_and_reasoning(response_text: str) -> tuple[str, str, dict]:
    response_text = normalize_text(response_text)
    code_matches = CODE_BLOCK_RE.findall(response_text)
    if code_matches:
        language, raw_code = code_matches[0]
        code = normalize_text(raw_code)
        reasoning = normalize_text(CODE_BLOCK_RE.sub("", response_text))
    else:
        lines = response_text.splitlines()
        code_lines = []
        reasoning_lines = []
        in_code = False
        for line in lines:
            stripped = line.strip()
            looks_like_code = (
                stripped.startswith(
                    (
                        "def ",
                        "class ",
                        "import ",
                        "from ",
                        "if ",
                        "for ",
                        "while ",
                        "try:",
                        "with ",
                        "function ",
                        "public ",
                        "private ",
                        "protected ",
                        "const ",
                        "let ",
                        "var ",
                        "#include",
                        "using ",
                    )
                )
                or stripped.endswith(("{", "}", ":", ";"))
                or stripped.startswith(("return ", "print(", "@"))
            )
            if looks_like_code or in_code:
                in_code = True
                code_lines.append(line)
            else:
                reasoning_lines.append(line)
        code = normalize_text("\n".join(code_lines))
        reasoning = normalize_text("\n".join(reasoning_lines))
        language = ""
    return reasoning, code, {"language": normalize_text(language or "python"), "had_code_block": bool(code_matches)}


def looks_like_code_solution(code: str, had_code_block: bool) -> bool:
    code = normalize_text(code)
    if not code:
        return False
    if had_code_block:
        return True
    return any(
        token in code
        for token in [
            "def ",
            "return ",
            "import ",
            "from ",
            "class ",
            "function ",
            "public ",
            "private ",
            "const ",
            "let ",
            "var ",
            "#include",
            "System.out",
        ]
    )


def build_user_prompt(instruction: str) -> str:
    return normalize_text(
        "\n".join(
            [
                "Write a correct solution for the following coding instruction.",
                "",
                normalize_text(instruction),
                "",
                "Return a complete solution that is robust on edge cases.",
            ]
        )
    )


def count_tokens(tokenizer, messages: list[dict], output_text: str) -> tuple[int, int]:
    prompt_tokens = len(tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False))
    completion_tokens = len(tokenizer(output_text, add_special_tokens=False)["input_ids"])
    return prompt_tokens, completion_tokens


def iter_magicoder_rows(url: str):
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        for raw_line in response.iter_lines():
            if not raw_line:
                continue
            yield json.loads(raw_line.decode("utf-8"))


def iter_local_rows(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def make_dual_row(
    messages: list[dict],
    output_text: str,
    reasoning: str,
    code: str,
    row_idx: int,
    partial_ratio: float,
    language: str,
) -> dict:
    partial_reasoning = truncate_reasoning_prefix_half(reasoning, partial_ratio)
    full_trace = build_code_output_text(reasoning, code, language=language)
    partial_trace = build_code_output_text(partial_reasoning, code, language=language)
    question_text = normalize_text(messages[1]["content"])
    partial_teacher_prompt = [
        messages[0],
        {
            "role": "user",
            "content": TEACHER_TEMPLATE.substitute(orig_content=question_text, output_text=partial_trace),
        },
    ]
    full_teacher_prompt = [
        messages[0],
        {
            "role": "user",
            "content": TEACHER_TEMPLATE.substitute(orig_content=question_text, output_text=full_trace),
        },
    ]
    return {
        "prompt": messages,
        "counterfactual_prompt": messages,
        "teacher_prompt": full_teacher_prompt,
        "partial_teacher_prompt": partial_teacher_prompt,
        "full_teacher_prompt": full_teacher_prompt,
        "gold_answer": normalize_text(code),
        "gold_trace": full_trace,
        "partial_trace": partial_trace,
        "output_text": output_text,
        "source_split": "train",
        "source_dataset": "magicoder_code_filtered",
        "source_index": row_idx,
        "partial_view_mode": "prefix_half_with_answer",
        "partial_ratio": partial_ratio,
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Build baseline and dual-view code datasets for collaborator-scale training runs.")
    parser.add_argument("--input_jsonl", type=Path, default=None, help="Optional local source JSONL path.")
    parser.add_argument("--source_url", type=str, default=MAGICODER_URL, help="JSONL source URL.")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B", help="Tokenizer path used for token-length filtering.")
    parser.add_argument("--output_root", type=Path, default=Path("data/derived/code_100k"), help="Output root.")
    parser.add_argument("--max_records", type=int, default=100000, help="Number of accepted examples to keep.")
    parser.add_argument("--partial_ratio", type=float, default=0.5, help="Partial-trace character ratio.")
    parser.add_argument("--max_prompt_tokens", type=int, default=1024, help="Prompt token cap.")
    parser.add_argument("--max_completion_tokens", type=int, default=1024, help="Completion token cap.")
    parser.add_argument("--max_total_tokens", type=int, default=1200, help="Prompt+completion token cap.")
    return parser.parse_args()


def main():
    args = parse_args()
    if not 0.0 < args.partial_ratio <= 1.0:
        raise ValueError("--partial_ratio must be in (0, 1].")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    output_root = args.output_root
    ds_dir = output_root / "datasets"
    train_dir = output_root / "train_data"
    ds_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    baseline_jsonl = train_dir / "magicoder_code_sdft_train.jsonl"
    baseline_records_jsonl = train_dir / "magicoder_code_sdft_records.jsonl"
    dual_records_jsonl = train_dir / "magicoder_code_dual_records.jsonl"
    for path in [baseline_jsonl, baseline_records_jsonl, dual_records_jsonl]:
        if path.exists():
            path.unlink()

    accepted = 0
    seen = 0
    skipped = {
            "parse_failure": 0,
            "missing_reasoning": 0,
            "synthesized_reasoning": 0,
            "bad_code": 0,
            "too_long": 0,
        }

    row_iter = iter_local_rows(args.input_jsonl) if args.input_jsonl is not None else iter_magicoder_rows(args.source_url)

    with (
        baseline_jsonl.open("a", encoding="utf-8") as baseline_handle,
        baseline_records_jsonl.open("a", encoding="utf-8") as baseline_records_handle,
        dual_records_jsonl.open("a", encoding="utf-8") as dual_records_handle,
    ):
        for row_idx, row in enumerate(row_iter):
            seen += 1
            instruction = normalize_text(row.get("instruction", ""))
            response = normalize_text(row.get("response", ""))
            if not instruction or not response:
                skipped["parse_failure"] += 1
                continue

            reasoning, code, meta = extract_code_and_reasoning(response)
            if not reasoning:
                reasoning = "The solution follows the requested implementation and is provided below."
                skipped["synthesized_reasoning"] += 1
            if not looks_like_code_solution(code, had_code_block=meta["had_code_block"]):
                skipped["bad_code"] += 1
                continue

            messages = [
                {"role": "system", "content": CODE_SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(instruction)},
            ]
            output_text = build_code_output_text(reasoning, code, language=meta["language"])
            prompt_tokens, completion_tokens = count_tokens(tokenizer, messages, output_text)
            if (
                prompt_tokens > args.max_prompt_tokens
                or completion_tokens > args.max_completion_tokens
                or prompt_tokens + completion_tokens > args.max_total_tokens
            ):
                skipped["too_long"] += 1
                continue

            baseline_record = {
                "prompt": messages,
                "teacher_prompt": [
                    messages[0],
                    {
                        "role": "user",
                        "content": TEACHER_TEMPLATE.substitute(
                            orig_content=normalize_text(messages[1]["content"]),
                            output_text=output_text,
                        ),
                    },
                ],
                "gold_trace": output_text,
                "output_text": output_text,
                "source_dataset": "magicoder_code_filtered",
                "source_index": row_idx,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
            }
            dual_record = make_dual_row(
                messages,
                output_text,
                reasoning,
                code,
                row_idx=row_idx,
                partial_ratio=args.partial_ratio,
                language=meta["language"],
            )

            baseline_handle.write(json.dumps({"messages": messages, "output_text": output_text}, ensure_ascii=False) + "\n")
            baseline_records_handle.write(json.dumps(baseline_record, ensure_ascii=False) + "\n")
            dual_records_handle.write(json.dumps(dual_record, ensure_ascii=False) + "\n")
            accepted += 1
            if accepted >= args.max_records:
                break

    baseline_dataset_path = ds_dir / "train_sdft_code_magicoder_filtered"
    dual_dataset_path = ds_dir / "train_dual_code_magicoder_prefixhalf"
    load_dataset("json", data_files=str(baseline_records_jsonl), split="train").save_to_disk(str(baseline_dataset_path))
    load_dataset("json", data_files=str(dual_records_jsonl), split="train").save_to_disk(str(dual_dataset_path))

    source_desc = str(args.input_jsonl) if args.input_jsonl is not None else args.source_url
    manifest = {
        "source_url": source_desc,
        "seen_rows": seen,
        "accepted_rows": accepted,
        "skipped": skipped,
        "max_records": args.max_records,
        "partial_ratio": args.partial_ratio,
        "max_prompt_tokens": args.max_prompt_tokens,
        "max_completion_tokens": args.max_completion_tokens,
        "max_total_tokens": args.max_total_tokens,
        "baseline_jsonl": str(baseline_jsonl),
        "baseline_records_jsonl": str(baseline_records_jsonl),
        "dual_records_jsonl": str(dual_records_jsonl),
        "baseline_dataset_path": str(baseline_dataset_path),
        "dual_dataset_path": str(dual_dataset_path),
    }
    with (output_root / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
