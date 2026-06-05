#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from datasets import load_dataset, load_from_disk
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Generic SFT trainer for messages+output_text datasets.")
    parser.add_argument("--model_name", type=str, required=True, help="Base model or checkpoint path")
    parser.add_argument("--dataset_path", type=str, required=True, help="HF dataset dir or JSON/JSONL file")
    parser.add_argument("--output_dir", type=str, required=True, help="Output checkpoint directory")
    parser.add_argument("--run_name", type=str, default=None, help="Optional run name")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Epoch count")
    parser.add_argument("--max_steps", type=int, default=-1, help="Maximum train steps override")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Per-device batch size")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=32, help="Gradient accumulation")
    parser.add_argument("--save_steps", type=int, default=50, help="Checkpoint save interval")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Max checkpoints to keep")
    parser.add_argument("--warmup_steps", type=int, default=10, help="Warmup steps")
    parser.add_argument("--max_length", type=int, default=2048, help="Maximum sequence length")
    parser.add_argument("--report_to", type=str, default="none", help="Trainer reporting target")
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Optional checkpoint to resume from")
    parser.add_argument("--fsdp", type=str, default=None, help='FSDP mode string, e.g. "full_shard auto_wrap"')
    parser.add_argument(
        "--latest_checkpoint_keep_optimizer",
        action="store_true",
        help="Accepted for launcher compatibility; SFT smoke runs currently save model-only checkpoints.",
    )
    parser.add_argument(
        "--no_latest_checkpoint_keep_optimizer",
        dest="latest_checkpoint_keep_optimizer",
        action="store_false",
        help="Accepted for launcher compatibility; no-op because SFT runs save model-only checkpoints.",
    )
    parser.add_argument(
        "--fsdp_transformer_layer_cls_to_wrap",
        type=str,
        default=None,
        help="Transformer block class name for FSDP auto wrapping, e.g. Qwen3DecoderLayer",
    )
    parser.set_defaults(latest_checkpoint_keep_optimizer=False)
    return parser.parse_args()


def load_reasoning_dataset(path: str, tokenizer, seed: int):
    dataset_path = Path(path)
    if dataset_path.is_dir():
        dataset = load_from_disk(str(dataset_path))
    elif dataset_path.suffix.lower() in {".json", ".jsonl"}:
        dataset = load_dataset("json", data_files=str(dataset_path), split="train")
    else:
        dataset = load_from_disk(str(dataset_path))

    def format_example(example):
        messages = list(example["messages"]) + [{"role": "assistant", "content": example["output_text"]}]
        return {"text": tokenizer.apply_chat_template(messages, tokenize=False)}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    return dataset.shuffle(seed=seed)


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = load_reasoning_dataset(args.dataset_path, tokenizer, args.seed)
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.bfloat16)

    fsdp_config = None
    if args.fsdp_transformer_layer_cls_to_wrap:
        fsdp_config = {"transformer_layer_cls_to_wrap": [args.fsdp_transformer_layer_cls_to_wrap]}
        fsdp_config["activation_checkpointing"] = True

    config = SFTConfig(
        output_dir=args.output_dir,
        run_name=args.run_name,
        seed=args.seed,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        warmup_steps=args.warmup_steps,
        lr_scheduler_type="cosine",
        bf16=True,
        fp16=False,
        weight_decay=0.0,
        max_grad_norm=1.0,
        max_length=args.max_length,
        dataset_text_field="text",
        packing=False,
        # Transformers now rejects enabling both trainer gradient checkpointing
        # and FSDP activation checkpointing at the same time.
        gradient_checkpointing=not bool(args.fsdp),
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to=args.report_to,
        save_only_model=True,
        completion_only_loss=False,
        fsdp=args.fsdp,
        fsdp_config=fsdp_config,
    )

    trainer = SFTTrainer(
        model=model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)


if __name__ == "__main__":
    main()
