import argparse
import json
import os
import re
from typing import Optional

import numpy as np
import torch
from datasets import Dataset
from sympy import simplify
from sympy.parsing.sympy_parser import parse_expr
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate a model on the ASFT math500 fixed eval set")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the trained model or checkpoint")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default="data/packages/math/eval/eval_math500",
        help="Path to the eval dataset saved with datasets.save_to_disk",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory to save evaluation results (defaults to model_path/eval_math_500)",
    )
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature (0 for greedy)")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.35, help="vLLM GPU memory fraction")
    parser.add_argument("--max_samples", type=int, default=None, help="Optional cap for smoke testing")
    return parser.parse_args()


def load_model_and_tokenizer(model_path: str, gpu_memory_utilization: float):
    print(f"Loading model from {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="left")
    llm = LLM(
        model=model_path,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype=torch.bfloat16,
        max_model_len=4096,
        trust_remote_code=True,
    )
    return llm, tokenizer


def load_eval_data(dataset_path: str, max_samples: Optional[int]) -> Dataset:
    print(f"Loading eval dataset from {dataset_path}")
    dataset = Dataset.load_from_disk(dataset_path)
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    return dataset


def generate_responses(llm, tokenizer, prompts, max_new_tokens: int, temperature: float):
    formatted_prompts = [
        tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True) for prompt in prompts
    ]
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )
    print(f"Generating responses for {len(formatted_prompts)} prompts...")
    outputs = llm.generate(formatted_prompts, sampling_params)
    return [output.outputs[0].text for output in outputs]


def extract_boxed(text: str) -> Optional[str]:
    marker = "\\boxed{"
    start = text.rfind(marker)
    if start == -1:
        return None
    i = start + len(marker)
    depth = 1
    chars = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
        chars.append(ch)
        i += 1
    return "".join(chars).strip() or None


def extract_answer(text: str) -> str:
    if "<answer>" in text and "</answer>" in text:
        return text.split("<answer>")[-1].split("</answer>")[0].strip()
    boxed = extract_boxed(text)
    if boxed:
        return boxed
    return text.strip()


def normalize_text(text: str) -> str:
    text = text.strip()
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


def evaluate_correctness(responses, answers):
    scores = []
    extracted_answers = []
    for response, answer in zip(responses, answers):
        pred = extract_answer(response)
        extracted_answers.append(pred)
        scores.append(1 if equivalent_math(pred, answer) else 0)
    return scores, extracted_answers


def main():
    args = parse_args()

    llm, tokenizer = load_model_and_tokenizer(args.model_path, args.gpu_memory_utilization)
    eval_data = load_eval_data(args.dataset_path, args.max_samples)

    prompts = [example["prompt"] for example in eval_data]
    answers = [example["answer"] for example in eval_data]
    problems = [example["problem"] for example in eval_data]
    subjects = [example["subject"] for example in eval_data]
    levels = [example["level"] for example in eval_data]
    unique_ids = [example["unique_id"] for example in eval_data]

    responses = generate_responses(llm, tokenizer, prompts, args.max_new_tokens, args.temperature)

    print("\nEvaluating responses...")
    scores, predictions = evaluate_correctness(responses, answers)
    accuracy = np.mean(scores) if scores else 0.0

    print("\n" + "=" * 60)
    print("Evaluation Results:")
    print(f"  Total samples: {len(scores)}")
    print(f"  Correct: {sum(scores)}")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print("=" * 60)

    output_dir = args.output_dir if args.output_dir else os.path.join(args.model_path, "eval_math_500")
    os.makedirs(output_dir, exist_ok=True)

    summary = {
        "accuracy": float(accuracy),
        "num_correct": int(sum(scores)),
        "num_total": len(scores),
        "config": {
            "model_path": args.model_path,
            "dataset_path": args.dataset_path,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "max_samples": args.max_samples,
        },
    }

    summary_path = os.path.join(output_dir, "eval_results.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    rows = []
    for i in range(len(scores)):
        rows.append(
            {
                "unique_id": unique_ids[i],
                "subject": subjects[i],
                "level": levels[i],
                "problem": problems[i],
                "gold_answer": answers[i],
                "predicted_answer": predictions[i],
                "response": responses[i],
                "correct": bool(scores[i]),
            }
        )

    rows_path = os.path.join(output_dir, "eval_responses.json")
    with open(rows_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"Saved per-sample responses to {rows_path}")


if __name__ == "__main__":
    main()
