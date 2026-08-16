#!/usr/bin/env python3
"""Train one gated EI difficulty-aware ablation arm from fresh H0 LoRA."""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lean_prover.lean_training.sft_pipeline.config import SFTTrainConfig
from lean_prover.lean_training.sft_pipeline.trainer import FixedManifestSFTTrainer, build_trainer
from scripts.audit_stage2_ratio_phase_c1_feasibility import record_id, sha256, theorem_group
from scripts.train_anchor_ratio_eos_fixed_arm import directory_tensor_hash, validate_adapter

H0_HASH = "6b0ea36dcc8dfc71f9b85220797c29703660679dac214af368de084e7a176431"
ARMS = ("A_medium_hard", "B_medium_only", "C_add_repair", "D_add_replay")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument("--arm", required=True, choices=ARMS)
    args = parser.parse_args()
    project = args.project.resolve()
    arm_root = project / "outputs/expert_iteration/difficulty_ablation" / args.arm
    contract_path = arm_root / "training_contract.json"
    audit_path = arm_root / "data_audit.json"
    manifest_path = arm_root / "manifest.jsonl"
    contract = read_json(contract_path)
    audit = read_json(audit_path)
    config = SFTTrainConfig(**contract["resolved_config"])
    trainer_dir = Path(config.output_dir)
    adapter_dir = arm_root / "checkpoint/adapter"
    identity_path = arm_root / "checkpoint/training_identity.json"
    summary_path = arm_root / "checkpoint/training_summary.json"
    existing = [path for path in (trainer_dir, adapter_dir, identity_path, summary_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing continuation or overwrite: {existing}")
    if not audit.get("passed") or sha256(manifest_path) != contract["manifest_sha256"]:
        raise RuntimeError("manifest audit/hash gate failed")
    if sha256(Path(config.model_name_or_path) / "model.safetensors") != H0_HASH:
        raise RuntimeError("frozen H0 checkpoint drifted")
    if config.adapter_path is not None:
        raise RuntimeError("fresh LoRA initialization required")

    fixed = {
        "learning_rate": 1e-5, "num_train_epochs": 1.0, "lora_r": 32, "lora_alpha": 64,
        "lora_dropout": 0.05, "gradient_accumulation_steps": 16, "max_seq_length": 1024,
        "requested_packing": False, "require_supervised_eos": True,
        "sampling_strategy": "fixed_manifest_without_replacement", "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "linear", "warmup_ratio": 0.03, "weight_decay": 0.01,
    }
    drift = {key: {"expected": value, "actual": getattr(config, key)} for key, value in fixed.items() if getattr(config, key) != value}
    if drift:
        raise RuntimeError(f"training contract drifted: {drift}")

    rows = read_jsonl(manifest_path)
    expected = int(contract["expected_rows"])
    ids = [record_id(row) for row in rows]
    groups = [theorem_group(row) for row in rows]
    roles = Counter(row["difficulty_ablation"]["role"] for row in rows)
    if len(rows) != expected or len(set(ids)) != expected or len(set(groups)) != expected:
        raise RuntimeError("manifest uniqueness gate failed")
    if dict(roles) != contract["expected_role_counts"] or any(row.get("pantograph_verified") is not True for row in rows):
        raise RuntimeError("manifest composition/verification gate failed")

    identity = {
        "experiment": args.arm, "starting_checkpoint": config.model_name_or_path,
        "starting_checkpoint_model_sha256": H0_HASH, "manifest_sha256": sha256(manifest_path),
        "data_audit_sha256": sha256(audit_path), "training_contract_sha256": sha256(contract_path),
        "rows": expected, "role_counts": dict(roles), "initialization": "fresh_lora_on_frozen_merged_H0",
        "resume_from_checkpoint": False, "optimizer_state_shared": False, "scheduler_state_shared": False,
        "adapter_state_shared": False, "resolved_config": config.__dict__,
    }
    write_json(identity_path, identity)
    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer, tokenizer = build_trainer(config)
    if not isinstance(trainer, FixedManifestSFTTrainer):
        raise TypeError("fixed-manifest trainer not selected")
    eos = read_json(trainer_dir / "sft_supervised_eos_contract.json")["source_preflight"]
    if not eos.get("passed") or eos.get("total_records") != expected or eos.get("records_with_supervised_eos") != expected:
        raise RuntimeError("trainer source EOS gate failed")
    if eos.get("records_whose_last_valid_label_is_eos") != expected or eos.get("zero_label_records") != 0 or eos.get("records_with_semantic_truncation") != 0:
        raise RuntimeError("trainer effective source labels invalid")

    result = trainer.train()
    expected_steps = math.ceil(expected / 16)
    if trainer.state.global_step != expected_steps:
        raise RuntimeError(f"optimizer step drift: {trainer.state.global_step}/{expected_steps}")
    sampler = trainer.fixed_manifest_sampler
    if sampler is None or sampler.iteration_count != 1:
        raise RuntimeError("fixed sampler trace missing")
    indices = list(sampler.drawn_indices)
    counts = Counter(ids[index] for index in indices)
    if len(indices) != expected or len(counts) != expected or max(counts.values()) != 1:
        raise RuntimeError("training was not exactly once without replacement")
    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.save_state()
    validate_adapter(adapter_dir)
    effective_eos = read_json(trainer_dir / "sft_supervised_eos_contract.json")["effective_trainer_labels"]
    if not effective_eos.get("passed"):
        raise RuntimeError("effective trainer EOS gate failed")
    final_hash = sha256(Path(config.model_name_or_path) / "model.safetensors")
    if final_hash != H0_HASH:
        raise RuntimeError("training modified H0")
    eval_rows = [row for row in trainer.state.log_history if "eval_loss" in row]
    summary = {
        **identity, "optimizer_steps": trainer.state.global_step, "draws": len(indices),
        "unique_draws": len(counts), "duplicate_draws": len(indices) - len(counts),
        "max_repeat": max(counts.values()), "adapter_tensor_hash": directory_tensor_hash(adapter_dir),
        "train_loss": getattr(result, "training_loss", None),
        "eval_loss": eval_rows[-1].get("eval_loss") if eval_rows else None,
        "eval_token_accuracy": eval_rows[-1].get("eval_mean_token_accuracy", eval_rows[-1].get("eval_token_accuracy")) if eval_rows else None,
        "runtime_seconds": time.monotonic() - started,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "frozen_h0_hash_after_training": final_hash,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
