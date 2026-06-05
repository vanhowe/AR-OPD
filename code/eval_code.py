#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import torch
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

CODE_SYSTEM_PROMPT = """You are a careful Python programmer. Solve the task and respond in the following format:
<reasoning>
Briefly explain the implementation strategy and any important edge cases.
</reasoning>
<answer>
```python
# solution
```
</answer>

Do not include any text outside these tags."""

CODE_BLOCK_RE = re.compile(r"```(?:python|py)?\s*(.*?)```", re.DOTALL | re.IGNORECASE)
ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
LOCAL_CODE_EVAL_ROOT = Path(__file__).resolve().parent / "data" / "packages" / "code" / "eval"
LOCAL_DATASET_PATHS = {
    "humaneval": LOCAL_CODE_EVAL_ROOT / "humanevalplus_eval.jsonl",
    "mbpp": LOCAL_CODE_EVAL_ROOT / "mbppplus_eval.jsonl",
}


def normalize_text(text: str) -> str:
    return str(text).replace("\r\n", "\n").strip()


def build_humaneval_prompt(example: dict) -> str:
    public_checks = []
    for line in str(example.get("test", "")).splitlines():
        stripped = line.strip()
        if stripped.startswith("assert "):
            public_checks.append(stripped)
        if len(public_checks) >= 5:
            break
    sections = [
        "Write a correct Python solution for the following EvalPlus HumanEval+ task.",
        f"Required entry point: {example['entry_point']}",
        "",
        normalize_text(example["prompt"]),
    ]
    if public_checks:
        sections.extend(["", "Reference checks (subset):", *public_checks])
    sections.extend(["", "Your code should be robust to hidden edge cases checked by EvalPlus."])
    return "\n".join(sections).strip()


def build_mbpp_prompt(example: dict) -> str:
    prompt = [
        "Write a correct Python solution for the following EvalPlus MBPP+ task.",
        normalize_text(example["prompt"]),
    ]
    test_list = example.get("assertion") or []
    if isinstance(test_list, str):
        test_list = [line.strip() for line in test_list.splitlines() if line.strip()]
    if test_list:
        prompt.extend(["", "Reference checks (subset):", *test_list[:5]])
    prompt.extend(["", "Your code should satisfy the examples and remain robust on additional EvalPlus checks."])
    return "\n".join(prompt).strip()


def load_local_jsonl(dataset_name: str, dataset_path: Path):
    items = []
    prompt_builder = build_humaneval_prompt if dataset_name == "humaneval" else build_mbpp_prompt
    with dataset_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            problem = json.loads(line)
            task_id = str(problem.get("task_id", "")).strip()
            if not task_id:
                raise ValueError(f"Missing task_id in {dataset_path}")
            if dataset_name == "mbpp" and "/" not in task_id:
                task_id = f"Mbpp/{task_id}"
            problem["task_id"] = task_id
            if dataset_name == "mbpp" and "assertion" not in problem and "test_list" in problem:
                problem["assertion"] = problem["test_list"]
            messages = [
                {"role": "system", "content": CODE_SYSTEM_PROMPT},
                {"role": "user", "content": prompt_builder(problem)},
            ]
            items.append({"task_id": task_id, "problem": problem, "messages": messages})
    return items


def load_evalplus_problems(dataset_name: str):
    try:
        from evalplus.data import get_human_eval_plus, get_mbpp_plus
    except ImportError as exc:
        raise ImportError(
            f"Local dataset file for '{dataset_name}' was not found and evalplus is not installed. "
            f"Either place the JSONL under {LOCAL_CODE_EVAL_ROOT} or install evalplus."
        ) from exc
    if dataset_name == "humaneval":
        problems = get_human_eval_plus()
        prompt_builder = build_humaneval_prompt
    elif dataset_name == "mbpp":
        problems = get_mbpp_plus()
        prompt_builder = build_mbpp_prompt
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    items = []
    for task_id, problem in problems.items():
        messages = [
            {"role": "system", "content": CODE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt_builder(problem)},
        ]
        items.append({"task_id": task_id, "problem": problem, "messages": messages})
    return items


def load_problems(dataset_name: str, dataset_path: str | None = None):
    if dataset_name not in {"humaneval", "mbpp"}:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    candidate_path = Path(dataset_path) if dataset_path else LOCAL_DATASET_PATHS[dataset_name]
    if candidate_path.exists():
        return load_local_jsonl(dataset_name, candidate_path)
    return load_evalplus_problems(dataset_name)


def extract_completion(response_text: str) -> tuple[str, dict]:
    response_text = normalize_text(response_text)
    meta = {"used_answer_tag": False, "used_code_block": False}
    answer_match = ANSWER_RE.search(response_text)
    if answer_match:
        response_text = normalize_text(answer_match.group(1))
        meta["used_answer_tag"] = True
    code_match = CODE_BLOCK_RE.search(response_text)
    if code_match:
        meta["used_code_block"] = True
        return normalize_text(code_match.group(1)), meta
    return response_text, meta


def parse_args():
    parser = argparse.ArgumentParser(description="Generate code completions for EvalPlus HumanEval/MBPP.")
    parser.add_argument("--model_path", type=str, required=True, help="Trained model path")
    parser.add_argument("--dataset", type=str, required=True, choices=["humaneval", "mbpp"], help="Evaluation benchmark")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Optional local JSONL path. If unset, the script first checks repo-local benchmark files before falling back to evalplus.",
    )
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    parser.add_argument("--prompt_style", type=str, default="plain", choices=["plain", "xml"], help="Prompt format used at eval time.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--seed", type=int, default=0, help="Sampling seed")
    parser.add_argument("--max_new_tokens", type=int, default=1024, help="Max generation tokens")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.8, help="vLLM memory fraction")
    return parser.parse_args()


def main():
    args = parse_args()
    items = load_problems(args.dataset, args.dataset_path)
    if args.prompt_style == "plain":
        for item in items:
            item["messages"] = [
                {"role": "system", "content": "You are a careful Python programmer. Return a correct Python solution as a code block."},
                {"role": "user", "content": item["messages"][1]["content"]},
            ]
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, padding_side="left", trust_remote_code=True)
    llm = LLM(
        model=args.model_path,
        gpu_memory_utilization=args.gpu_memory_utilization,
        dtype=torch.bfloat16,
        max_model_len=4096,
        trust_remote_code=True,
    )
    prompts = [tokenizer.apply_chat_template(item["messages"], tokenize=False, add_generation_prompt=True) for item in items]
    sampling_params = SamplingParams(
        temperature=args.temperature,
        seed=args.seed,
        max_tokens=args.max_new_tokens,
        stop_token_ids=[tokenizer.eos_token_id] if tokenizer.eos_token_id else None,
    )
    outputs = llm.generate(prompts, sampling_params)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_path = out_dir / f"{args.dataset}_samples.jsonl"
    responses_path = out_dir / f"{args.dataset}_responses.json"

    responses_payload = []
    with samples_path.open("w", encoding="utf-8") as f:
        for item, prompt, output in zip(items, prompts, outputs):
            response_text = output.outputs[0].text
            completion, meta = extract_completion(response_text)
            f.write(json.dumps({"task_id": item["task_id"], "completion": completion}, ensure_ascii=False) + "\n")
            responses_payload.append(
                {
                    "task_id": item["task_id"],
                    "response": response_text,
                    "completion": completion,
                    "used_answer_tag": meta["used_answer_tag"],
                    "used_code_block": meta["used_code_block"],
                    "prompt": prompt,
                }
            )

    with responses_path.open("w", encoding="utf-8") as f:
        json.dump(responses_payload, f, indent=2, ensure_ascii=False)

    print(samples_path)
    print(responses_path)


if __name__ == "__main__":
    main()
