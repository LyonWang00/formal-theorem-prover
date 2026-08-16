#!/usr/bin/env python3
"""Train one gated EI data-composition arm from a fresh H0 LoRA."""

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


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument("--arm", choices=("EI-A_success_only", "EI-B_success_frontier", "EI-C_success_frontier_replay"), required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    arm_root = project / "outputs/expert_iteration/ei_ablation" / args.arm
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
        raise RuntimeError("every arm must initialize a fresh LoRA")

    fixed = {
        "learning_rate": 1e-5,
        "num_train_epochs": 1.0,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "gradient_accumulation_steps": 16,
        "max_seq_length": 1024,
        "requested_packing": False,
        "require_supervised_eos": True,
        "sampling_strategy": "fixed_manifest_without_replacement",
        "optim": "paged_adamw_8bit",
        "lr_scheduler_type": "linear",
        "warmup_ratio": 0.03,
        "weight_decay": 0.01,
    }
    drift = {key: {"expected": expected, "actual": getattr(config, key)} for key, expected in fixed.items() if getattr(config, key) != expected}
    if drift:
        raise RuntimeError(f"training contract drifted: {drift}")

    rows = read_jsonl(manifest_path)
    expected_rows = int(contract["expected_rows"])
    ids = [record_id(row) for row in rows]
    groups = [theorem_group(row) for row in rows]
    roles = Counter(row["ei_ablation"]["role"] for row in rows)
    if (
        len(rows) != expected_rows
        or len(set(ids)) != expected_rows
        or len(set(groups)) != expected_rows
        or dict(roles) != contract["expected_role_counts"]
        or any(row.get("pantograph_verified") is not True for row in rows)
    ):
        raise RuntimeError("training manifest identity/composition gate failed")

    identity = {
        "experiment": args.arm,
        "starting_checkpoint": config.model_name_or_path,
        "starting_checkpoint_model_sha256": H0_HASH,
        "manifest_sha256": sha256(manifest_path),
        "data_audit_sha256": sha256(audit_path),
        "training_contract_sha256": sha256(contract_path),
        "rows": expected_rows,
        "role_counts": dict(roles),
        "initialization": "fresh_lora_on_frozen_merged_H0",
        "resume_from_checkpoint": False,
        "optimizer_state_shared": False,
        "scheduler_state_shared": False,
        "adapter_state_shared": False,
        "resolved_config": config.__dict__,
    }
    write_json(identity_path, identity)

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer, tokenizer = build_trainer(config)
    if not isinstance(trainer, FixedManifestSFTTrainer):
        raise TypeError("fixed-manifest without-replacement trainer was not selected")
    eos_contract = read_json(trainer_dir / "sft_supervised_eos_contract.json")
    source_eos = eos_contract["source_preflight"]
    if (
        not source_eos.get("passed")
        or source_eos.get("total_records") != expected_rows
        or source_eos.get("records_with_supervised_eos") != expected_rows
        or source_eos.get("records_whose_last_valid_label_is_eos") != expected_rows
        or source_eos.get("zero_label_records") != 0
        or source_eos.get("records_with_semantic_truncation") != 0
    ):
        raise RuntimeError("trainer source EOS gate failed")

    result = trainer.train()
    expected_steps = math.ceil(expected_rows / 16)
    if trainer.state.global_step != expected_steps:
        raise RuntimeError(f"optimizer step drift: {trainer.state.global_step}/{expected_steps}")
    sampler = trainer.fixed_manifest_sampler
    if sampler is None or sampler.iteration_count != 1:
        raise RuntimeError("fixed sampler trace missing")
    indices = list(sampler.drawn_indices)
    counts = Counter(ids[index] for index in indices)
    if len(indices) != expected_rows or len(counts) != expected_rows or max(counts.values()) != 1:
        raise RuntimeError("training was not exactly once without replacement")

    trainer.save_model(str(adapter_dir))
    tokenizer.save_pretrained(str(adapter_dir))
    trainer.save_state()
    validate_adapter(adapter_dir)
    effective_eos = read_json(trainer_dir / "sft_supervised_eos_contract.json")["effective_trainer_labels"]
    if not effective_eos.get("passed"):
        raise RuntimeError("effective trainer EOS gate failed")
    final_h0_hash = sha256(Path(config.model_name_or_path) / "model.safetensors")
    if final_h0_hash != H0_HASH:
        raise RuntimeError("training modified H0")
    eval_rows = [row for row in trainer.state.log_history if "eval_loss" in row]
    summary = {
        **identity,
        "optimizer_steps": trainer.state.global_step,
        "draws": len(indices),
        "unique_draws": len(counts),
        "duplicate_draws": len(indices) - len(counts),
        "max_repeat": max(counts.values()),
        "adapter_tensor_hash": directory_tensor_hash(adapter_dir),
        "train_loss": getattr(result, "training_loss", None),
        "eval_loss": eval_rows[-1].get("eval_loss") if eval_rows else None,
        "eval_token_accuracy": eval_rows[-1].get("eval_mean_token_accuracy", eval_rows[-1].get("eval_token_accuracy")) if eval_rows else None,
        "runtime_seconds": time.monotonic() - started,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "frozen_h0_hash_after_training": final_h0_hash,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
