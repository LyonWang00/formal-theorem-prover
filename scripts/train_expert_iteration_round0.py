"""Train the gated 250-row Expert Iteration Round-0 LoRA adapter."""

from __future__ import annotations

import argparse
import json
import math
import resource
import shutil
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch

from lean_prover.lean_training.sft_pipeline.config import SFTTrainConfig
from lean_prover.lean_training.sft_pipeline.trainer import (
    FixedManifestSFTTrainer,
    build_trainer,
)
from scripts.audit_stage2_ratio_phase_c1_feasibility import record_id, sha256, theorem_group
from scripts.train_anchor_ratio_eos_fixed_arm import directory_tensor_hash, validate_adapter


EXPECTED_ROWS = 250
EXPECTED_ROLE_COUNTS = {"Success Bank": 200, "Frontier replay": 50}
EXPECTED_H0_MODEL_HASH = (
    "6b0ea36dcc8dfc71f9b85220797c29703660679dac214af368de084e7a176431"
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / "outputs/expert_iteration/round0"
    contract_path = root / "training_manifest/training_contract.json"
    gate_path = root / "training_manifest/manifest_gate.json"
    contract = read_json(contract_path)
    gate = read_json(gate_path)
    config = SFTTrainConfig(**contract["resolved_config"])
    manifest = Path(config.train_file)
    base_model = Path(config.model_name_or_path)
    trainer_dir = Path(config.output_dir)
    checkpoint_root = root / "checkpoint/EI-Round0"
    identity_path = root / "checkpoint/training_identity.json"
    summary_path = root / "checkpoint/training_summary.json"
    status_path = root / "status.json"

    existing = [path for path in (trainer_dir, checkpoint_root, identity_path, summary_path) if path.exists()]
    if existing:
        raise FileExistsError(f"refusing continuation/overwrite: {existing}")
    if not gate.get("passed") or gate.get("rows") != EXPECTED_ROWS:
        raise RuntimeError("EI manifest gate did not pass")
    if sha256(manifest) != gate["manifest_sha256"]:
        raise RuntimeError("EI manifest hash drifted")
    if sha256(base_model / "model.safetensors") != EXPECTED_H0_MODEL_HASH:
        raise RuntimeError("frozen H0 checkpoint drifted")
    if config.adapter_path is not None:
        raise RuntimeError("EI Round 0 must initialize a fresh LoRA adapter")

    fixed = {
        "learning_rate": 1e-5,
        "num_train_epochs": 1.0,
        "lora_r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.05,
        "require_supervised_eos": True,
        "require_pantograph_verified": True,
        "requested_packing": False,
        "sampling_strategy": "fixed_manifest_without_replacement",
        "gradient_accumulation_steps": 16,
        "max_seq_length": 1024,
    }
    drift = {
        key: {"expected": expected, "actual": getattr(config, key)}
        for key, expected in fixed.items()
        if getattr(config, key) != expected
    }
    if drift:
        raise RuntimeError(f"EI fixed training contract drifted: {drift}")

    rows = read_jsonl(manifest)
    ids = [record_id(row) for row in rows]
    groups = [theorem_group(row) for row in rows]
    roles = Counter(row["ei_round0"]["top_level_role"] for row in rows)
    if (
        len(rows) != EXPECTED_ROWS
        or len(set(ids)) != EXPECTED_ROWS
        or len(set(groups)) != EXPECTED_ROWS
        or any(not value for value in ids + groups)
        or dict(roles) != EXPECTED_ROLE_COUNTS
        or any(row.get("pantograph_verified") is not True for row in rows)
        or any(row["ei_round0"].get("target_origin") != "pantograph_verified_h0_generation" for row in rows)
    ):
        raise RuntimeError("EI manifest identity/composition/verification gate failed")

    identity = {
        "experiment": "Expert Iteration Round 0",
        "model": "EI-Round0",
        "starting_checkpoint": str(base_model),
        "starting_checkpoint_model_sha256": EXPECTED_H0_MODEL_HASH,
        "manifest_sha256": sha256(manifest),
        "manifest_gate_sha256": sha256(gate_path),
        "training_contract_sha256": sha256(contract_path),
        "rows": len(rows),
        "role_counts": dict(roles),
        "initialization": "fresh_lora_on_frozen_merged_H0",
        "adapter_path": None,
        "resume_from_checkpoint": False,
        "optimizer_state_shared": False,
        "scheduler_state_shared": False,
        "adapter_state_shared": False,
        "resolved_config": config.__dict__,
    }
    write_json(identity_path, identity)
    status = read_json(status_path)
    status.update({"status": "EI_SFT_PRETRAIN_CHECKS_RUNNING", "trainer_started": False, "grpo_started": False})
    write_json(status_path, status)

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer, tokenizer = build_trainer(config)
    status.update({"status": "EI_SFT_RUNNING", "trainer_started": True, "grpo_started": False})
    write_json(status_path, status)
    if not isinstance(trainer, FixedManifestSFTTrainer):
        raise TypeError("fixed-manifest without-replacement trainer was not selected")
    eos_contract = read_json(trainer_dir / "sft_supervised_eos_contract.json")
    diagnostics = read_json(trainer_dir / "sft_tokenization_diagnostics.json")
    actual_eos = eos_contract["effective_trainer_labels"]
    if (
        not actual_eos.get("passed")
        or actual_eos.get("total_records") != EXPECTED_ROWS
        or actual_eos.get("records_with_supervised_eos") != EXPECTED_ROWS
        or actual_eos.get("records_whose_last_valid_label_is_eos") != EXPECTED_ROWS
        or actual_eos.get("zero_label_records") != 0
        or actual_eos.get("records_with_semantic_truncation") != 0
        or diagnostics.get("num_examples") != EXPECTED_ROWS
        or diagnostics.get("truncated_examples") != 0
    ):
        raise RuntimeError("trainer-effective EI EOS/token gate failed")

    step_zero = checkpoint_root / "step_0"
    trainer.save_model(str(step_zero))
    tokenizer.save_pretrained(str(step_zero))
    validate_adapter(step_zero)
    step_zero_hash = directory_tensor_hash(step_zero)

    result = trainer.train()
    expected_steps = math.ceil(EXPECTED_ROWS / 16)
    if contract["expected_optimizer_steps"] != expected_steps or trainer.state.global_step != expected_steps:
        raise RuntimeError(f"unexpected EI optimizer steps: {trainer.state.global_step}/{expected_steps}")
    sampler = trainer.fixed_manifest_sampler
    if sampler is None or sampler.iteration_count != 1:
        raise RuntimeError("EI sampler trace missing or iterated more than once")
    indices = list(sampler.drawn_indices)
    draw_counts = Counter(ids[index] for index in indices)
    if len(indices) != EXPECTED_ROWS or len(draw_counts) != EXPECTED_ROWS or max(draw_counts.values()) != 1:
        raise RuntimeError("EI training was not exactly once without replacement")

    best_source = Path(str(trainer.state.best_model_checkpoint or ""))
    if not best_source.is_dir():
        raise FileNotFoundError("EI best eval-loss checkpoint is missing")
    shutil.copytree(best_source, checkpoint_root / "best")
    trainer.save_model(str(checkpoint_root / "final"))
    tokenizer.save_pretrained(str(checkpoint_root / "final"))
    trainer.save_state()
    validate_adapter(checkpoint_root / "best")
    validate_adapter(checkpoint_root / "final")

    trace_path = root / "checkpoint/sampling_trace.jsonl"
    with trace_path.open("w", encoding="utf-8", newline="\n") as handle:
        for draw_index, dataset_index in enumerate(indices):
            row = rows[dataset_index]
            handle.write(
                json.dumps(
                    {
                        "draw_index": draw_index,
                        "dataset_index": dataset_index,
                        "record_id": ids[dataset_index],
                        "theorem_group_id": groups[dataset_index],
                        "top_level_role": row["ei_round0"]["top_level_role"],
                        "candidate_id": row["ei_round0"]["candidate_id"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    final_base_hash = sha256(base_model / "model.safetensors")
    if final_base_hash != EXPECTED_H0_MODEL_HASH:
        raise RuntimeError("training modified the frozen H0 checkpoint")
    eval_rows = [row for row in trainer.state.log_history if "eval_loss" in row]
    final_eval = eval_rows[-1] if eval_rows else {}
    summary = {
        **identity,
        "optimizer_steps": trainer.state.global_step,
        "draws": len(indices),
        "unique_draws": len(draw_counts),
        "duplicate_draws": len(indices) - len(draw_counts),
        "max_repeat": max(draw_counts.values()),
        "supervised_eos_records": actual_eos["records_with_supervised_eos"],
        "supervised_label_tokens": diagnostics["non_ignored_label_tokens_total"],
        "step_0_adapter_hash": step_zero_hash,
        "best_adapter_hash": directory_tensor_hash(checkpoint_root / "best"),
        "final_adapter_hash": directory_tensor_hash(checkpoint_root / "final"),
        "eval_loss": final_eval.get("eval_loss"),
        "eval_token_accuracy": final_eval.get("eval_mean_token_accuracy", final_eval.get("eval_token_accuracy")),
        "metrics": dict(getattr(result, "metrics", {}) or {}),
        "best_checkpoint": str(checkpoint_root / "best"),
        "final_checkpoint": str(checkpoint_root / "final"),
        "runtime_seconds": time.monotonic() - started,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "frozen_h0_hash_after_training": final_base_hash,
    }
    write_json(summary_path, summary)
    status.update(
        {
            "status": "EI_SFT_COMPLETED_AWAITING_MERGE",
            "trainer_started": True,
            "trainer_completed": True,
            "grpo_started": False,
            "optimizer_steps": trainer.state.global_step,
            "training_summary": str(summary_path),
        }
    )
    write_json(status_path, status)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
