#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


MEDICAL_SYSTEM_PROMPT = """Answer the medical multiple-choice question. Respond in the following format:
<reasoning>
...
</reasoning>
<answer>
...
</answer>

For the answer, output only the option letter (A, B, C, or D)."""

ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
LETTER_RE = re.compile(r"\b([ABCD])\b", re.IGNORECASE)
REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_EVAL_FILES = {
    "mmlu_medical": REPO_ROOT / "data/packages/medical/eval/mmlu_medical_eval.jsonl",
    "medqa_usmle_4_options": REPO_ROOT / "data/packages/medical/eval/medqa_usmle_4_options_eval.jsonl",
    "medmcqa_test": REPO_ROOT / "data/packages/medical/eval/medmcqa_test_eval.jsonl",
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on the standardized medical QA multiple-choice sets.")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save evaluation results")
    parser.add_argument("--max_new_tokens", type=int, default=512, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument(
        "--sources",
        nargs="+",
        default=["mmlu_medical", "medqa_usmle_4_options", "medmcqa_test"],
        choices=sorted(DEFAULT_EVAL_FILES.keys()),
        help="One or more medical eval sources to include.",
    )
    parser.add_argument(
        "--max_samples_per_source",
        type=int,
        default=None,
        help="Optional cap for each source; useful for preliminary runs.",
    )
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="vLLM GPU memory fraction")
    parser.add_argument("--seed", type=int, default=42, help="Sampling seed")
    return parser.parse_args()


def normalize_text(text: str) -> str:
    return str(text).replace("\r\n", "\n").strip()


def load_jsonl(path: str | Path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def format_question(question: str, choices: dict[str, str]) -> str:
    parts = [normalize_text(question), ""]
    for key in ["A", "B", "C", "D"]:
        if key in choices:
            parts.append(f"{key}: {normalize_text(choices[key])}")
    return "\n".join(parts).strip()


def load_eval_rows(sources: list[str], max_samples_per_source: int | None):
    rows = []
    for source in sources:
        eval_path = Path(DEFAULT_EVAL_FILES[source])
        if not eval_path.exists():
            raise FileNotFoundError(f"Missing eval file for {source}: {eval_path}")
        source_rows = load_jsonl(eval_path)
        if max_samples_per_source is not None:
            source_rows = source_rows[:max_samples_per_source]
        for idx, row in enumerate(source_rows):
            prompt_messages = [
                {"role": "system", "content": MEDICAL_SYSTEM_PROMPT},
                {"role": "user", "content": format_question(row["question"], row["choices"])},
            ]
            rows.append(
                {
                    "source": source,
                    "source_index": idx,
                    "question": row["question"],
                    "choices": row["choices"],
                    "answer": normalize_text(row["answer"]).upper(),
                    "prompt_messages": prompt_messages,
                    "raw_row": row,
                }
            )
    return rows


def load_model_and_tokenizer(model_path: str, gpu_memory_utilization: float):
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left", trust_remote_code=True)
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=torch.bfloat16,
        max_model_len=4096,
        trust_remote_code=True,
    )
    return llm, tokenizer


def generate_responses(llm, tokenizer, rows, max_new_tokens: int, temperature: float, seed: int):
    formatted_prompts = [
        tokenizer.apply_chat_template(row["prompt_messages"], tokenize=False, add_generation_prompt=True)
        for row in rows
    ]
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        seed=seed,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )
    print(f"Generating responses for {len(formatted_prompts)} prompts...")
    outputs = llm.generate(formatted_prompts, sampling_params)
    return formatted_prompts, [output.outputs[0].text for output in outputs]


def extract_answer_letter(text: str) -> tuple[str, bool]:
    normalized = normalize_text(text)
    answer_match = ANSWER_RE.search(normalized)
    if answer_match:
        answer_text = normalize_text(answer_match.group(1)).upper()
        letter_match = LETTER_RE.search(answer_text)
        if letter_match:
            return letter_match.group(1).upper(), True
    letter_matches = LETTER_RE.findall(normalized.upper())
    if letter_matches:
        return letter_matches[-1].upper(), False
    return "", False


def build_summary(records: list[dict], config: dict) -> dict:
    num_correct = sum(1 for record in records if record["correct"])
    accuracy = (num_correct / len(records)) if records else 0.0
    per_source = {}
    grouped = defaultdict(list)
    for record in records:
        grouped[record["source"]].append(record)
    for source, source_records in grouped.items():
        source_correct = sum(1 for record in source_records if record["correct"])
        per_source[source] = {
            "num_total": len(source_records),
            "num_correct": source_correct,
            "accuracy": source_correct / len(source_records) if source_records else 0.0,
            "valid_xml_rate": sum(1 for record in source_records if record["used_xml_answer"]) / len(source_records),
            "predicted_answer_distribution": dict(Counter(record["predicted_answer"] or "<empty>" for record in source_records)),
        }

    return {
        "num_total": len(records),
        "num_correct": num_correct,
        "accuracy": accuracy,
        "valid_xml_rate": sum(1 for record in records if record["used_xml_answer"]) / len(records) if records else 0.0,
        "per_source": per_source,
        "config": config,
    }


def main():
    args = parse_args()
    llm, tokenizer = load_model_and_tokenizer(args.model_path, args.gpu_memory_utilization)
    rows = load_eval_rows(args.sources, args.max_samples_per_source)
    formatted_prompts, responses = generate_responses(
        llm=llm,
        tokenizer=tokenizer,
        rows=rows,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    records = []
    for row, formatted_prompt, response in zip(rows, formatted_prompts, responses):
        predicted_answer, used_xml_answer = extract_answer_letter(response)
        gold_answer = row["answer"]
        records.append(
            {
                "source": row["source"],
                "source_index": row["source_index"],
                "question": row["question"],
                "choices": row["choices"],
                "gold_answer": gold_answer,
                "predicted_answer": predicted_answer,
                "used_xml_answer": used_xml_answer,
                "correct": predicted_answer == gold_answer,
                "prompt_messages": row["prompt_messages"],
                "formatted_prompt": formatted_prompt,
                "response": response,
            }
        )

    output_dir = args.output_dir if args.output_dir else args.model_path
    os.makedirs(output_dir, exist_ok=True)
    config = {
        "model_path": args.model_path,
        "sources": args.sources,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "max_samples_per_source": args.max_samples_per_source,
        "seed": args.seed,
    }
    summary = build_summary(records, config=config)

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"  Total samples: {summary['num_total']}")
    print(f"  Correct: {summary['num_correct']}")
    print(f"  Accuracy: {summary['accuracy']:.4f} ({summary['accuracy'] * 100:.2f}%)")
    print(f"  Valid XML answer rate: {summary['valid_xml_rate']:.4f} ({summary['valid_xml_rate'] * 100:.2f}%)")
    for source, stats in summary["per_source"].items():
        print(
            f"  {source}: {stats['num_correct']}/{stats['num_total']} = "
            f"{stats['accuracy'] * 100:.2f}% | xml={stats['valid_xml_rate'] * 100:.2f}%"
        )
    print("=" * 60)

    with open(Path(output_dir) / "eval_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(Path(output_dir) / "eval_responses.json", "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nSaved results to {Path(output_dir) / 'eval_results.json'}")
    print(f"Saved responses to {Path(output_dir) / 'eval_responses.json'}")


if __name__ == "__main__":
    main()
