#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


KNOWN_DOMAINS = {"math", "code", "medical", "math10k", "code10k"}
KNOWN_METHODS = {
    "sft",
    "asft",
    "sdft",
    "gdsdft_l04",
    "gdsdft_l06",
    "gdsdft_l08",
    "gdsdft_l10",
    "gdsdft_l12",
    "gdsdft_l14",
}


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def read_latest_metrics_history(metrics_history_path: Path) -> dict:
    if not metrics_history_path.exists():
        return {}
    latest_train_row = None
    try:
        with metrics_history_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if isinstance(payload, dict) and payload.get("mode") == "train":
                    latest_train_row = payload
    except Exception:
        return {}
    return latest_train_row or {}


def infer_domain_and_method(run_name: str) -> tuple[str, str]:
    parts = run_name.split("_")
    domain = parts[0] if parts and parts[0] in KNOWN_DOMAINS else "unknown"
    method = "unknown"
    for size in range(min(3, len(parts)), 0, -1):
        candidate = "_".join(parts[1 : 1 + size])
        if candidate in KNOWN_METHODS:
            method = candidate
            break
    return domain, method


def find_trainer_state(run_dir: Path) -> tuple[dict | None, Path | None]:
    candidates = [run_dir / "trainer_state.json"]
    candidates.extend(sorted(run_dir.glob("checkpoint-*/trainer_state.json"), key=lambda p: p.parent.name))
    existing = [path for path in candidates if path.exists()]
    if not existing:
        return None, None
    best_path = existing[-1]
    return read_json(best_path), best_path


def extract_train_stats(trainer_state: dict | None, metrics_history_row: dict | None = None) -> dict:
    metrics_history_row = metrics_history_row or {}
    log_history = trainer_state.get("log_history", [])
    train_losses = [entry["loss"] for entry in log_history if isinstance(entry, dict) and "loss" in entry]
    learning_rates = [
        entry["learning_rate"] for entry in log_history if isinstance(entry, dict) and "learning_rate" in entry
    ]
    tracked_metrics = {
        "final_grad_norm": "grad_norm",
        "final_entropy": "entropy",
        "final_teacher_entropy": "teacher_entropy",
        "final_entropy_gap": "entropy_gap",
        "final_overlap_ratio": "overlap_ratio",
        "final_overlap_token_advantage": "overlap_token_advantage",
        "final_prompt_mean_length": "prompt/mean_length",
        "final_prompt_p90_length": "prompt/p90_length",
        "final_completion_mean_length": "completion/mean_length",
        "final_completion_p90_length": "completion/p90_length",
        "final_completion_clipped_ratio": "completions/clipped_ratio",
        "final_privileged_belief_shift": "privileged_belief_shift",
        "final_residual_uptake": "residual_uptake",
        "final_residual_capture_ratio": "residual_capture_ratio",
        "final_residual_transfer_efficiency": "residual_transfer_efficiency",
        "final_anchor_drift_to_partial": "anchor_drift_to_partial",
        "final_dual_delta": "dual_delta",
        "final_dual_gate": "dual_gate",
        "final_dual_alpha": "dual_alpha",
        "final_tokens_per_sec": "tokens_per_sec",
        "final_wall_time_sec": "wall_time_sec",
    }
    tracked_values = {}
    for output_key, history_key in tracked_metrics.items():
        values = [entry[history_key] for entry in log_history if isinstance(entry, dict) and history_key in entry]
        tracked_values[output_key] = values[-1] if values else metrics_history_row.get(history_key)
    return {
        "global_step": trainer_state.get("global_step") or metrics_history_row.get("step"),
        "best_metric": trainer_state.get("best_metric"),
        "num_log_entries": len(log_history),
        "final_logged_loss": train_losses[-1] if train_losses else metrics_history_row.get("loss"),
        "min_logged_loss": min(train_losses) if train_losses else None,
        "final_learning_rate": learning_rates[-1] if learning_rates else metrics_history_row.get("learning_rate"),
        **tracked_values,
    }


def iter_eval_rows(
    artifact_base: Path,
    run_name: str,
    run_dir: Path,
    train_stats: dict,
    trainer_state_path: Path | None,
):
    artifact_root = artifact_base / run_name
    metrics_history_path = run_dir / "metrics_history.jsonl"
    eval_files = sorted(artifact_root.glob("**/eval_results.json"))
    if not eval_files:
        yield {
            "run_name": run_name,
            "domain": infer_domain_and_method(run_name)[0],
            "method": infer_domain_and_method(run_name)[1],
            "checkpoint_dir": str(run_dir),
            "trainer_state_path": str(trainer_state_path) if trainer_state_path else None,
            "metrics_history_path": str(metrics_history_path) if metrics_history_path.exists() else None,
            "global_step": train_stats.get("global_step"),
            "best_metric": train_stats.get("best_metric"),
            "final_logged_loss": train_stats.get("final_logged_loss"),
            "min_logged_loss": train_stats.get("min_logged_loss"),
            "final_learning_rate": train_stats.get("final_learning_rate"),
            "final_grad_norm": train_stats.get("final_grad_norm"),
            "final_entropy": train_stats.get("final_entropy"),
            "final_teacher_entropy": train_stats.get("final_teacher_entropy"),
            "final_entropy_gap": train_stats.get("final_entropy_gap"),
            "final_overlap_ratio": train_stats.get("final_overlap_ratio"),
            "final_overlap_token_advantage": train_stats.get("final_overlap_token_advantage"),
            "final_prompt_mean_length": train_stats.get("final_prompt_mean_length"),
            "final_prompt_p90_length": train_stats.get("final_prompt_p90_length"),
            "final_completion_mean_length": train_stats.get("final_completion_mean_length"),
            "final_completion_p90_length": train_stats.get("final_completion_p90_length"),
            "final_completion_clipped_ratio": train_stats.get("final_completion_clipped_ratio"),
            "final_privileged_belief_shift": train_stats.get("final_privileged_belief_shift"),
            "final_residual_uptake": train_stats.get("final_residual_uptake"),
            "final_residual_capture_ratio": train_stats.get("final_residual_capture_ratio"),
            "final_residual_transfer_efficiency": train_stats.get("final_residual_transfer_efficiency"),
            "final_anchor_drift_to_partial": train_stats.get("final_anchor_drift_to_partial"),
            "final_dual_delta": train_stats.get("final_dual_delta"),
            "final_dual_gate": train_stats.get("final_dual_gate"),
            "final_dual_alpha": train_stats.get("final_dual_alpha"),
            "final_tokens_per_sec": train_stats.get("final_tokens_per_sec"),
            "final_wall_time_sec": train_stats.get("final_wall_time_sec"),
            "benchmark": None,
            "subset": None,
            "accuracy": None,
            "num_correct": None,
            "num_total": None,
            "valid_xml_rate": None,
            "eval_path": None,
        }
        return

    domain, method = infer_domain_and_method(run_name)
    for eval_json in eval_files:
        payload = read_json(eval_json) or {}
        relative_parts = eval_json.relative_to(artifact_root).parts
        benchmark = relative_parts[0] if relative_parts else eval_json.parent.name
        subset = "/".join(relative_parts[:-1]) if len(relative_parts) > 1 else benchmark
        if isinstance(payload.get("per_source"), dict):
            for source_name, source_payload in sorted(payload["per_source"].items()):
                yield {
                    "run_name": run_name,
                    "domain": domain,
                    "method": method,
                    "checkpoint_dir": str(run_dir),
                    "trainer_state_path": str(trainer_state_path) if trainer_state_path else None,
                    "metrics_history_path": str(metrics_history_path) if metrics_history_path.exists() else None,
                    "global_step": train_stats.get("global_step"),
                    "best_metric": train_stats.get("best_metric"),
                    "final_logged_loss": train_stats.get("final_logged_loss"),
                    "min_logged_loss": train_stats.get("min_logged_loss"),
                    "final_learning_rate": train_stats.get("final_learning_rate"),
                    "final_grad_norm": train_stats.get("final_grad_norm"),
                    "final_entropy": train_stats.get("final_entropy"),
                    "final_teacher_entropy": train_stats.get("final_teacher_entropy"),
                    "final_entropy_gap": train_stats.get("final_entropy_gap"),
                    "final_overlap_ratio": train_stats.get("final_overlap_ratio"),
                    "final_overlap_token_advantage": train_stats.get("final_overlap_token_advantage"),
                    "final_prompt_mean_length": train_stats.get("final_prompt_mean_length"),
                    "final_prompt_p90_length": train_stats.get("final_prompt_p90_length"),
                    "final_completion_mean_length": train_stats.get("final_completion_mean_length"),
                    "final_completion_p90_length": train_stats.get("final_completion_p90_length"),
                    "final_completion_clipped_ratio": train_stats.get("final_completion_clipped_ratio"),
                    "final_privileged_belief_shift": train_stats.get("final_privileged_belief_shift"),
                    "final_residual_uptake": train_stats.get("final_residual_uptake"),
                    "final_residual_capture_ratio": train_stats.get("final_residual_capture_ratio"),
                    "final_residual_transfer_efficiency": train_stats.get("final_residual_transfer_efficiency"),
                    "final_anchor_drift_to_partial": train_stats.get("final_anchor_drift_to_partial"),
                    "final_dual_delta": train_stats.get("final_dual_delta"),
                    "final_dual_gate": train_stats.get("final_dual_gate"),
                    "final_dual_alpha": train_stats.get("final_dual_alpha"),
                    "final_tokens_per_sec": train_stats.get("final_tokens_per_sec"),
                    "final_wall_time_sec": train_stats.get("final_wall_time_sec"),
                    "benchmark": benchmark,
                    "subset": source_name,
                    "accuracy": source_payload.get("accuracy"),
                    "num_correct": source_payload.get("num_correct"),
                    "num_total": source_payload.get("num_total"),
                    "valid_xml_rate": source_payload.get("valid_xml_rate"),
                    "eval_path": str(eval_json),
                }
        else:
            yield {
                "run_name": run_name,
                "domain": domain,
                "method": method,
                "checkpoint_dir": str(run_dir),
                "trainer_state_path": str(trainer_state_path) if trainer_state_path else None,
                "metrics_history_path": str(metrics_history_path) if metrics_history_path.exists() else None,
                "global_step": train_stats.get("global_step"),
                "best_metric": train_stats.get("best_metric"),
                "final_logged_loss": train_stats.get("final_logged_loss"),
                "min_logged_loss": train_stats.get("min_logged_loss"),
                "final_learning_rate": train_stats.get("final_learning_rate"),
                "final_grad_norm": train_stats.get("final_grad_norm"),
                "final_entropy": train_stats.get("final_entropy"),
                "final_teacher_entropy": train_stats.get("final_teacher_entropy"),
                "final_entropy_gap": train_stats.get("final_entropy_gap"),
                "final_overlap_ratio": train_stats.get("final_overlap_ratio"),
                "final_overlap_token_advantage": train_stats.get("final_overlap_token_advantage"),
                "final_prompt_mean_length": train_stats.get("final_prompt_mean_length"),
                "final_prompt_p90_length": train_stats.get("final_prompt_p90_length"),
                "final_completion_mean_length": train_stats.get("final_completion_mean_length"),
                "final_completion_p90_length": train_stats.get("final_completion_p90_length"),
                "final_completion_clipped_ratio": train_stats.get("final_completion_clipped_ratio"),
                "final_privileged_belief_shift": train_stats.get("final_privileged_belief_shift"),
                "final_residual_uptake": train_stats.get("final_residual_uptake"),
                "final_residual_capture_ratio": train_stats.get("final_residual_capture_ratio"),
                "final_residual_transfer_efficiency": train_stats.get("final_residual_transfer_efficiency"),
                "final_anchor_drift_to_partial": train_stats.get("final_anchor_drift_to_partial"),
                "final_dual_delta": train_stats.get("final_dual_delta"),
                "final_dual_gate": train_stats.get("final_dual_gate"),
                "final_dual_alpha": train_stats.get("final_dual_alpha"),
                "final_tokens_per_sec": train_stats.get("final_tokens_per_sec"),
                "final_wall_time_sec": train_stats.get("final_wall_time_sec"),
                "benchmark": benchmark,
                "subset": subset,
                "accuracy": payload.get("accuracy"),
                "num_correct": payload.get("num_correct"),
                "num_total": payload.get("num_total"),
                "valid_xml_rate": payload.get("valid_xml_rate"),
                "eval_path": str(eval_json),
            }


def collect_rows(checkpoint_base: Path, artifact_base: Path):
    rows = []
    for run_dir in sorted(checkpoint_base.glob("*")):
        if not run_dir.is_dir():
            continue
        trainer_state, trainer_state_path = find_trainer_state(run_dir)
        metrics_history_path = run_dir / "metrics_history.jsonl"
        metrics_history_row = read_latest_metrics_history(metrics_history_path)
        train_stats = extract_train_stats(trainer_state or {}, metrics_history_row)
        rows.extend(iter_eval_rows(artifact_base, run_dir.name, run_dir, train_stats, trainer_state_path))
    return rows


def main():
    parser = argparse.ArgumentParser(description="Collect training/eval metrics across collaborator runs.")
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--checkpoint-root", default=None)
    parser.add_argument("--artifact-root", default=None)
    args = parser.parse_args()

    root = Path(args.root)
    checkpoint_base = Path(args.checkpoint_root) if args.checkpoint_root else root / "checkpoints"
    artifact_base = Path(args.artifact_root) if args.artifact_root else root / "artifacts"

    rows = collect_rows(checkpoint_base, artifact_base)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "run_name",
                "domain",
                "method",
                "checkpoint_dir",
                "trainer_state_path",
                "metrics_history_path",
                "global_step",
                "best_metric",
                "final_logged_loss",
                "min_logged_loss",
                "final_learning_rate",
                "final_grad_norm",
                "final_entropy",
                "final_teacher_entropy",
                "final_entropy_gap",
                "final_overlap_ratio",
                "final_overlap_token_advantage",
                "final_prompt_mean_length",
                "final_prompt_p90_length",
                "final_completion_mean_length",
                "final_completion_p90_length",
                "final_completion_clipped_ratio",
                "final_privileged_belief_shift",
                "final_residual_uptake",
                "final_residual_capture_ratio",
                "final_residual_transfer_efficiency",
                "final_anchor_drift_to_partial",
                "final_dual_delta",
                "final_dual_gate",
                "final_dual_alpha",
                "final_tokens_per_sec",
                "final_wall_time_sec",
                "benchmark",
                "subset",
                "accuracy",
                "num_correct",
                "num_total",
                "valid_xml_rate",
                "eval_path",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(out)


if __name__ == "__main__":
    main()
