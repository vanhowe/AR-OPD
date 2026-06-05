from distil_trainer import DistilTrainer
from distil_config import DistilConfig
from cad_utils import extract_science_answer
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from datasets import Dataset, load_dataset, load_from_disk
from string import Template
import argparse
import torch.distributed as dist
from pathlib import Path
import os
import math
import re

WORD_CHAR_PATTERN = re.compile(r"[\w\\]", re.UNICODE)
SCIENCE_REASONING_RE = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL | re.IGNORECASE)
SCIENCE_ANSWER_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)

def parse_args():
    parser = argparse.ArgumentParser(description="Distil Trainer")
    parser.add_argument("--learning_rate", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--max_steps", type=int, default=-1, help="Override total optimizer steps; -1 keeps epoch-based training")
    parser.add_argument("--num_prompts_per_batch", type=int, default=32, help="Number of prompts per batch")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1, help="Per-device train batch size")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=None,
        help="Gradient accumulation steps; if unset, falls back to --num_prompts_per_batch for backward compatibility",
    )
    parser.add_argument("--alpha", type=float, default=0.0, help="Distillation alpha: 0=forward KL, 1=reverse KL")
    parser.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature for rollout generation")
    parser.add_argument("--ref_model_mixup_alpha", type=float, default=0.01, help="Reference model mixup alpha")
    parser.add_argument("--base_kl_weight", type=float, default=0.0, help="KL weight for the frozen base-model anchor")
    parser.add_argument("--base_model_name", type=str, default=None, help="Frozen base model used for RASD-base")
    parser.add_argument("--save_steps", type=int, default=100, help="Save checkpoint every N optimizer steps")
    parser.add_argument("--save_total_limit", type=int, default=3, help="Maximum number of checkpoints to keep")
    parser.add_argument("--max_prompt_length", type=int, default=1024, help="Maximum prompt length")
    parser.add_argument("--max_completion_length", type=int, default=1024, help="Maximum completion length")
    parser.add_argument(
        "--latest_checkpoint_keep_optimizer",
        dest="latest_checkpoint_keep_optimizer",
        action="store_true",
        help="Keep optimizer/scheduler/rng state only in the latest checkpoint.",
    )
    parser.add_argument(
        "--no_latest_checkpoint_keep_optimizer",
        dest="latest_checkpoint_keep_optimizer",
        action="store_false",
        help="Save model-only checkpoints.",
    )
    parser.add_argument(
        "--keep_optimizer_for_all_checkpoints",
        action="store_true",
        help="Retain optimizer/scheduler/rng state in every saved checkpoint instead of only the latest one.",
    )
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.2, help="GPU memory fraction reserved for colocated vLLM")
    parser.add_argument("--use_vllm", dest="use_vllm", action="store_true", help="Use vLLM for generation.")
    parser.add_argument("--no_use_vllm", dest="use_vllm", action="store_false", help="Disable vLLM and use transformers generation.")
    parser.add_argument("--vllm_mode", type=str, default="colocate", choices=["colocate", "server"], help="Whether to run vLLM in-process or connect to an external vLLM server")
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1, help="Tensor parallel size for colocated vLLM")
    parser.add_argument("--vllm_server_base_url", type=str, default=None, help="Base URL for an external vLLM server")
    parser.add_argument("--vllm_server_host", type=str, default="127.0.0.1", help="Host for the external vLLM server when base URL is not provided")
    parser.add_argument("--vllm_server_port", type=int, default=8000, help="Port for the external vLLM server when base URL is not provided")
    parser.add_argument("--vllm_server_timeout", type=float, default=240.0, help="Timeout in seconds for the external vLLM server")
    parser.add_argument("--vllm_enable_sleep_mode", action="store_true", help="Enable vLLM sleep mode when using colocated mode")
    parser.add_argument("--fsdp", type=str, default=None, help='FSDP mode string, e.g. "full_shard auto_wrap"')
    parser.add_argument(
        "--fsdp_transformer_layer_cls_to_wrap",
        type=str,
        default=None,
        help="Transformer layer class name for FSDP auto wrapping, e.g. Qwen2DecoderLayer",
    )
    parser.add_argument("--output_dir", type=str, help="Output directory")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Model name")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Checkpoint path to resume training from")
    parser.add_argument("--report_to", type=str, default="none", help="Trainer reporting target, e.g. none or wandb")
    parser.add_argument(
        "--skip_final_export",
        action="store_true",
        help="Do not write an extra final model snapshot after training; rely on the latest checkpoint directly.",
    )
    parser.add_argument(
        "--skip_save_state",
        action="store_true",
        help="Skip the trailing trainer.save_state() call; useful for short pilot runs that only need the latest checkpoint.",
    )
    parser.add_argument("--dataset_name", type=str, default="tooluse", help="Dataset name", choices=["tooluse", "science"])
    parser.add_argument(
        "--train_dataset_path",
        type=str,
        default=None,
        help="Optional prepared dataset saved with datasets.save_to_disk(); overrides --dataset_name loading.",
    )
    parser.add_argument("--cdsdft_enable", action="store_true", help="Enable CD-SDFT")
    parser.add_argument("--cdsdft_delta_metric", type=str, default="kl", choices=["kl"], help="Counterfactual delta metric")
    parser.add_argument("--cdsdft_delta_threshold", type=float, default=0.02, help="Token delta gate threshold")
    parser.add_argument("--cdsdft_retention_weight", type=float, default=0.02, help="Counterfactual retention weight")
    parser.add_argument("--cdsdft_gate_temperature", type=float, default=10.0, help="Token delta gate temperature")
    parser.add_argument("--cad_enable", action="store_true", help="Enable CAD")
    parser.add_argument("--cad_delta_metric", type=str, default="kl", choices=["kl"], help="Counterfactual delta metric for CAD")
    parser.add_argument("--cad_delta_threshold", type=float, default=0.02, help="Token delta gate threshold for CAD")
    parser.add_argument("--cad_gate_temperature", type=float, default=10.0, help="Token delta gate temperature for CAD")
    parser.add_argument("--cad_local_retention_weight", type=float, default=0.02, help="Local counterfactual retention weight for CAD")
    parser.add_argument("--cad_advantage_min", type=float, default=0.0, help="Lower clip for CAD advantage weights")
    parser.add_argument("--cad_advantage_max", type=float, default=2.0, help="Upper clip for CAD advantage weights")
    parser.add_argument("--cad_value_max", type=float, default=2.0, help="Normalization scale for CAD verifier values")
    parser.add_argument("--osdft_enable", action="store_true", help="Enable Orthogonal Gradient SDFT")
    parser.add_argument("--osdft_acquisition_weight", type=float, default=1.0, help="Weight on the acquisition gradient in OVSDFT")
    parser.add_argument("--osdft_preservation_weight", type=float, default=1.0, help="Weight on the preservation gradient in OVSDFT")
    parser.add_argument(
        "--osdft_preservation_source",
        type=str,
        default="counterfactual",
        choices=["counterfactual", "base"],
        help="Preservation target used by OVSDFT",
    )
    parser.add_argument(
        "--osdft_project_all",
        action="store_true",
        help="Project the acquisition gradient against the preservation gradient even when they are not in conflict",
    )
    parser.add_argument("--dualsdft_enable", action="store_true", help="Enable DualSDFT over partial/full teacher views")
    parser.add_argument("--dual_delta_threshold", type=float, default=0.02, help="KL(full||partial) threshold")
    parser.add_argument("--dual_gate_temperature", type=float, default=10.0, help="Gate sharpness for partial->full interpolation")
    parser.add_argument("--dual_alpha_floor", type=float, default=0.10, help="Minimum full-view weight")
    parser.add_argument("--dual_alpha_cap", type=float, default=0.90, help="Maximum full-view weight")
    parser.add_argument("--dual_confidence_power", type=float, default=1.0, help="Exponent on the full-view margin confidence")
    parser.add_argument("--gdsdft_enable", action="store_true", help="Enable simplified GD-SDFT over partial/full teacher views")
    parser.add_argument("--gdsdft_lambda", type=float, default=1.0, help="GD-SDFT residual scaling: <1 conservative, 1 full, >1 extrapolative")
    parser.add_argument("--gdsdft_residual_clip", type=float, default=5.0, help="Absolute clip on log q_full - log q_partial in GD-SDFT")
    parser.add_argument(
        "--dual_teacher_cpu_offload",
        action="store_true",
        help="Serialize full/partial teacher log-probs through CPU to reduce peak GPU memory in DualSDFT.",
    )
    parser.add_argument(
        "--dual_selected_logps_only",
        action="store_true",
        help="Single-GPU survival mode for GD-SDFT: distill only on selected completion-token log-probs.",
    )
    parser.add_argument(
        "--dual_exact_chunked_loss",
        action="store_true",
        help="Single-GPU exact-ish GD-SDFT path: compute vocab-level KD losses chunk-by-chunk.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Seed")
    parser.set_defaults(use_vllm=True)
    parser.set_defaults(latest_checkpoint_keep_optimizer=True)
    return parser.parse_args()

def load_tooluse_dataset(seed=42) -> Dataset:
    """Load and prepare tooluse dataset with formatted prompts."""
    train_dir = 'data/tooluse_data/train_data'
    train_dataset = load_from_disk(train_dir)

    def format_example(example):

        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        return {
            "prompt": [{"role": "user", "content": example['prompt']}],
            "counterfactual_prompt": [{"role": "user", "content": example['prompt']}],
            "teacher_prompt": [{"role": "user", "content": teacher_prompt.substitute(orig_content=example['prompt'], output_text='\n'.join(example['golden_response']))}],
            "golden_answer": example["golden_answer"],
        }

    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    train_dataset = train_dataset.shuffle(seed=seed)
    return train_dataset, None


def load_science_dataset(seed=42) -> Dataset:
    """Load and prepare science dataset with formatted prompts."""
    path = 'data/science_data/train_data'
    print(f"Loading science dataset from {path}")
    dataset = load_from_disk(path)

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

    def parse_science_response(response: str) -> tuple[str, str]:
        text = normalize_text(response)
        reasoning_match = SCIENCE_REASONING_RE.search(text)
        answer_match = SCIENCE_ANSWER_RE.search(text)
        if reasoning_match and answer_match:
            return normalize_text(reasoning_match.group(1)), normalize_text(answer_match.group(1))

        if answer_match:
            return text, normalize_text(answer_match.group(1))

        return text, ""

    def format_example(example):
        teacher_prompt = Template("""
$orig_content

This is an example for a response to the question:
$output_text

Now answer with a response of your own, including the thinking process.
""")

        question_text = normalize_text(example["messages"][1]["content"])
        full_reasoning, answer = parse_science_response(example["output_text"])
        partial_reasoning = truncate_reasoning_prefix_half(full_reasoning, 0.5)
        full_trace = build_xml_trace(full_reasoning, answer)
        partial_trace = build_xml_trace(partial_reasoning, answer)
        partial_teacher_prompt = [
            example["messages"][0],
            {'role': 'user', 'content': teacher_prompt.substitute(
                orig_content=question_text,
                output_text=partial_trace
            )},
        ]
        full_teacher_prompt = [
            example["messages"][0],
            {'role': 'user', 'content': teacher_prompt.substitute(
                orig_content=question_text,
                output_text=full_trace
            )},
        ]

        return {
            "prompt": example["messages"],
            "counterfactual_prompt": example["messages"],
            "teacher_prompt": full_teacher_prompt,
            "partial_teacher_prompt": partial_teacher_prompt,
            "full_teacher_prompt": full_teacher_prompt,
            "cad_answer": extract_science_answer(example["output_text"]),
            "gold_answer": answer,
            "gold_trace": full_trace,
            "partial_trace": partial_trace,
            "partial_view_mode": "prefix_half_with_answer",
            "partial_ratio": 0.5,
        }

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded {len(dataset)} training examples")
    return dataset, None


def load_prepared_dataset(dataset_path: str, seed=42) -> Dataset:
    """Load a pre-built prepared dataset for plain SDFT training."""
    dataset = load_from_disk(dataset_path)
    dataset = dataset.shuffle(seed=seed)
    print(f"Loaded prepared dataset from {dataset_path} with {len(dataset)} rows")
    return dataset, None


if __name__ == "__main__":
    args = parse_args()
    if args.fsdp and "auto_wrap" in args.fsdp and not args.fsdp_transformer_layer_cls_to_wrap:
        raise ValueError("--fsdp_transformer_layer_cls_to_wrap is required when using FSDP auto_wrap")

    use_cpu = not torch.cuda.is_available()
    model_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model_init_path = args.resume_from_checkpoint or args.model_name
    model = AutoModelForCausalLM.from_pretrained(
        model_init_path,
        torch_dtype=model_dtype,
    )
    teacher_model = AutoModelForCausalLM.from_pretrained(
        model_init_path,
        torch_dtype=model_dtype,
    )
    base_model = None
    need_base_model = args.base_kl_weight > 0 or (args.osdft_enable and args.osdft_preservation_source == "base")
    if need_base_model:
        base_model_name = args.base_model_name or args.model_name
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            torch_dtype=model_dtype,
        )
        base_model.eval()
        for parameter in base_model.parameters():
            parameter.requires_grad = False
    tokenizer = AutoTokenizer.from_pretrained(model_init_path)
    if args.train_dataset_path:
        dataset, _ = load_prepared_dataset(args.train_dataset_path, args.seed)
    elif args.dataset_name == "tooluse":
        dataset, _ = load_tooluse_dataset(args.seed)
    elif args.dataset_name == "science":
        dataset, _ = load_science_dataset(args.seed)
    else:
        raise ValueError(f"Invalid dataset name: {args.dataset_name}")

    gradient_accumulation_steps = (
        args.gradient_accumulation_steps
        if args.gradient_accumulation_steps is not None
        else args.num_prompts_per_batch
    )
    fsdp_config = None
    if args.fsdp_transformer_layer_cls_to_wrap:
        fsdp_config = {"transformer_layer_cls_to_wrap": [args.fsdp_transformer_layer_cls_to_wrap]}
    if args.fsdp:
        fsdp_config = fsdp_config or {}
        fsdp_config["activation_checkpointing"] = True

    config = DistilConfig(
        seed=args.seed,
        use_vllm = args.use_vllm,
        vllm_mode=args.vllm_mode,
        vllm_tensor_parallel_size=args.vllm_tensor_parallel_size,
        vllm_server_base_url=args.vllm_server_base_url,
        vllm_server_host=args.vllm_server_host,
        vllm_server_port=args.vllm_server_port,
        vllm_server_timeout=args.vllm_server_timeout,
        vllm_gpu_memory_utilization=args.vllm_gpu_memory_utilization,
        vllm_enable_sleep_mode=args.vllm_enable_sleep_mode,
        learning_rate = args.learning_rate,
        warmup_ratio = 0.1,
        lr_scheduler_type = "cosine",
        logging_steps = 1,
        bf16 = not use_cpu,
        fp16 = False,
        use_cpu = use_cpu,
        gradient_checkpointing = not bool(args.fsdp),
        per_device_train_batch_size = args.per_device_train_batch_size,
        gradient_accumulation_steps = gradient_accumulation_steps,
        max_prompt_length = args.max_prompt_length,
        max_completion_length = args.max_completion_length,
        num_train_epochs = args.num_train_epochs,
        max_steps = args.max_steps,
        num_iterations = 1,
        num_generations = 1,
        save_steps = args.save_steps,
        save_total_limit = args.save_total_limit,
        save_only_model = not args.latest_checkpoint_keep_optimizer,
        max_grad_norm = 1,
        report_to = args.report_to,
        output_dir = args.output_dir,
        log_completions = False, # True for debugging
        sync_ref_model = True,
        ref_model_sync_steps = 1,
        ref_model_mixup_alpha = args.ref_model_mixup_alpha,
        vllm_importance_sampling_correction = True,
        num_loss_tokens_to_skip = 3,
        fsdp=args.fsdp,
        fsdp_config=fsdp_config,
    )
    config.alpha = args.alpha
    config.temperature = args.temperature
    config.latest_checkpoint_keep_optimizer = args.latest_checkpoint_keep_optimizer
    config.keep_optimizer_for_all_checkpoints = args.keep_optimizer_for_all_checkpoints
    config.cdsdft_enable = args.cdsdft_enable
    config.cdsdft_delta_metric = args.cdsdft_delta_metric
    config.cdsdft_delta_threshold = args.cdsdft_delta_threshold
    config.cdsdft_retention_weight = args.cdsdft_retention_weight
    config.cdsdft_gate_temperature = args.cdsdft_gate_temperature
    config.cad_enable = args.cad_enable
    config.cad_delta_metric = args.cad_delta_metric
    config.cad_delta_threshold = args.cad_delta_threshold
    config.cad_gate_temperature = args.cad_gate_temperature
    config.cad_local_retention_weight = args.cad_local_retention_weight
    config.cad_advantage_min = args.cad_advantage_min
    config.cad_advantage_max = args.cad_advantage_max
    config.cad_value_max = args.cad_value_max
    config.osdft_enable = args.osdft_enable
    config.osdft_acquisition_weight = args.osdft_acquisition_weight
    config.osdft_preservation_weight = args.osdft_preservation_weight
    config.osdft_preservation_source = args.osdft_preservation_source
    config.osdft_project_all = args.osdft_project_all
    config.dualsdft_enable = args.dualsdft_enable or args.gdsdft_enable
    config.dual_delta_threshold = args.dual_delta_threshold
    config.dual_gate_temperature = args.dual_gate_temperature
    config.dual_alpha_floor = args.dual_alpha_floor
    config.dual_alpha_cap = args.dual_alpha_cap
    config.dual_confidence_power = args.dual_confidence_power
    config.gdsdft_enable = args.gdsdft_enable
    config.gdsdft_lambda = args.gdsdft_lambda
    config.gdsdft_residual_clip = args.gdsdft_residual_clip
    config.dual_teacher_cpu_offload = args.dual_teacher_cpu_offload
    config.dual_selected_logps_only = args.dual_selected_logps_only
    config.dual_exact_chunked_loss = args.dual_exact_chunked_loss
    if args.osdft_enable:
        config.gradient_checkpointing_kwargs = {"use_reentrant": False}
    trainer = DistilTrainer(
        model=model,
        ref_model=teacher_model,
        base_model=base_model,
        args=config,
        train_dataset=dataset,
        processing_class=tokenizer,
        base_kl_weight=args.base_kl_weight,
    )
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
        if args.fsdp:
            print(
                "FSDP run complete; skipping inline final export to avoid duplicate full-state saves during teardown. "
                "Use a dedicated export step from the saved checkpoint instead."
            )
        elif args.skip_final_export:
            print(
                "Non-FSDP run complete; skipping inline final export as requested. "
                "Use the latest checkpoint directly to avoid duplicate end-of-run saves."
            )
        else:
            final_output_dir = Path(args.output_dir) / "final"
            final_output_dir.mkdir(parents=True, exist_ok=True)
            trainer.save_model(str(final_output_dir))
        if not args.skip_save_state:
            trainer.save_state()
    finally:
        if dist.is_available() and dist.is_initialized():
            # `trainer.train()` already performs the required distributed synchronization for checkpointing.
            # An extra teardown barrier here can deadlock if ranks leave the trainer at slightly different times
            # after FSDP save, which is exactly what short smoke runs were hitting.
            dist.destroy_process_group()
