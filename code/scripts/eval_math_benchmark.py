#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "You are solving a math problem. Respond in the following format:\n"
    "<reasoning>\n...\n</reasoning>\n"
    "<answer>\n...\n</answer>\n\n"
    "For the answer, output only the final answer expression or value, without extra explanation."
)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a math model on JSONL benchmarks such as Math500 or AIME.")
    parser.add_argument("--model_path", required=True, help="Model or checkpoint directory.")
    parser.add_argument("--dataset_jsonl", required=True, help="JSONL benchmark file.")
    parser.add_argument("--benchmark_name", default=None, help="Optional label stored in outputs.")
    parser.add_argument("--output_dir", required=True, help="Directory for summary and per-sample outputs.")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Generation cap.")
    parser.add_argument("--temperature", type=float, default=0.0, help="0 for greedy.")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.85, help="vLLM GPU memory fraction.")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="vLLM tensor parallel size.")
    parser.add_argument("--max_model_len", type=int, default=4096, help="vLLM max context length.")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for smoke tests.")
    return parser.parse_args()


def load_rows(path: Path, max_samples: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
            if max_samples is not None and len(rows) >= max_samples:
                break
    return rows


def build_prompts(rows: list[dict]) -> list[list[dict[str, str]]]:
    prompts = []
    for row in rows:
        question = str(row["question"]).strip()
        prompts.append(
            [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ]
        )
    return prompts


def load_model_and_tokenizer(model_path: str, gpu_memory_utilization: float, tensor_parallel_size: int, max_model_len: int):
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        tensor_parallel_size=tensor_parallel_size,
        max_model_len=max_model_len,
        dtype=torch.bfloat16,
        trust_remote_code=True,
    )
    return llm, tokenizer


def generate_responses(llm, tokenizer, prompts, max_new_tokens: int, temperature: float):
    formatted_prompts = [
        tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts
    ]
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )
    outputs = llm.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def extract_boxed(text: str) -> Optional[str]:
    for marker in ("\\boxed{", "\\fbox{", "\\framebox{"):
        start = text.rfind(marker)
        if start == -1:
            continue
        index = start + len(marker)
        depth = 1
        chars: list[str] = []
        while index < len(text) and depth > 0:
            ch = text[index]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    break
            chars.append(ch)
            index += 1
        value = "".join(chars).strip()
        if value:
            return value
    return None


def extract_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    if "####" in text:
        return text.rsplit("####", 1)[-1].strip()
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return lines[-1] if lines else text.strip()


def normalize_text(text: str) -> str:
    text = str(text).strip()
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\dfrac", "\\frac").replace("\\tfrac", "\\frac")
    text = text.replace("\\%", "%")
    text = re.sub(r"\\text\s*{([^{}]*)}", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = text.rstrip(".")
    return text


def latexish_to_sympy(text: str) -> Optional[str]:
    text = normalize_text(text)
    text = text.replace("^", "**")
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("\\pi", "pi")
    text = text.replace("\\cdot", "*")
    text = text.replace("\\times", "*")
    text = text.replace("\\sqrt", "sqrt")
    text = text.replace("\\frac", "frac")

    frac_pattern = re.compile(r"frac\(([^()]+)\)\(([^()]+)\)")
    prev = None
    while prev != text:
        prev = text
        text = frac_pattern.sub(r"((\1)/(\2))", text)

    text = text.replace(" ", "")
    if "\\" in text:
        return None
    return text


def equivalent_math(pred: str, gold: str) -> bool:
    pred_norm = normalize_text(pred)
    gold_norm = normalize_text(gold)
    if pred_norm == gold_norm:
        return True

    pred_expr = latexish_to_sympy(pred_norm)
    gold_expr = latexish_to_sympy(gold_norm)
    if pred_expr is None or gold_expr is None:
        return False

    try:
        pred_value = parse_expr(pred_expr, evaluate=True)
        gold_value = parse_expr(gold_expr, evaluate=True)
        return simplify(pred_value - gold_value) == 0
    except Exception:
        return False


def get_row_id(row: dict, index: int) -> str:
    for key in ("unique_id", "id", "url"):
        value = row.get(key)
        if value is not None:
            return str(value)
    return str(index)


def main():
    args = parse_args()
    dataset_path = Path(args.dataset_jsonl)
    rows = load_rows(dataset_path, args.max_samples)
    if not rows:
        raise ValueError(f"No rows loaded from {dataset_path}")

    benchmark_name = args.benchmark_name or dataset_path.stem
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    llm, tokenizer = load_model_and_tokenizer(
        model_path=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=args.max_model_len,
    )

    prompts = build_prompts(rows)
    responses = generate_responses(llm, tokenizer, prompts, args.max_new_tokens, args.temperature)

    per_sample = []
    scores = []
    for idx, (row, response) in enumerate(zip(rows, responses)):
        gold_answer = str(row["answer"]).strip()
        predicted_answer = extract_answer(response)
        correct = equivalent_math(predicted_answer, gold_answer)
        scores.append(int(correct))
        per_sample.append(
            {
                "row_id": get_row_id(row, idx),
                "benchmark_name": benchmark_name,
                "question": row["question"],
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "correct": bool(correct),
                "raw_response": response,
                "meta": {key: row[key] for key in row.keys() if key not in {"question", "answer"}},
            }
        )

    accuracy = float(np.mean(scores)) if scores else 0.0
    summary = {
        "benchmark_name": benchmark_name,
        "model_path": args.model_path,
        "dataset_jsonl": str(dataset_path),
        "num_samples": len(per_sample),
        "num_correct": int(sum(scores)),
        "accuracy": accuracy,
        "accuracy_percent": accuracy * 100.0,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
    }

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    (output_dir / "predictions.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in per_sample),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
