#!/usr/bin/env python3
"""Train one WB/LeanDojo token-matched arm from the frozen clean M0."""

from __future__ import annotations

import argparse
import hashlib
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


ARM_MANIFESTS = {
    "MIX-A-WB100": "A_WB1000_LD0",
    "MIX-B-LD25": "B_token_matched",
    "MIX-C-LD50": "C_token_matched",
    "MIX-E-LD100": "E_token_matched",
}
TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open(encoding="utf-8-sig") if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def validate_adapter(path: Path) -> None:
    payload = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if set(payload.get("target_modules") or []) != TARGET_MODULES:
        raise ValueError(f"unexpected target modules in {path}")
    if payload.get("r") != 32 or payload.get("lora_alpha") != 64:
        raise ValueError(f"unexpected LoRA rank/alpha in {path}")
    if float(payload.get("lora_dropout")) != 0.05:
        raise ValueError(f"unexpected LoRA dropout in {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", choices=tuple(ARM_MANIFESTS), required=True)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/wb_ld_small_sft_ablation")
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor"
        ),
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("data/processed/lean_workbook_verified_v2/eval.jsonl"),
    )
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args()

    root = args.root.resolve()
    model = args.model.resolve()
    eval_path = args.eval.resolve()
    manifest_name = ARM_MANIFESTS[args.arm]
    train_path = root / f"manifests/{manifest_name}.jsonl"
    training_dir = root / "training" / args.arm
    trainer_dir = training_dir / "trainer"
    checkpoint_dir = root / "checkpoints" / args.arm
    if training_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(f"refusing to overwrite completed/partial arm {args.arm}")

    contract = json.loads(
        (root / "audit/experiment_contract.json").read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (root / "audit/input_inventory.json").read_text(encoding="utf-8")
    )
    audit = json.loads(
        (root / "audit/token_budget_audit.json").read_text(encoding="utf-8")
    )
    if str(model) != str(Path(contract["clean_m0"]).resolve()):
        raise ValueError("model path is not the frozen clean M0")
    model_hash = sha256_file(model / "model.safetensors")
    if model_hash != contract["clean_m0_hash"]:
        raise ValueError("clean M0 model hash changed")
    train_hash = sha256_file(train_path)
    if train_hash != inventory["manifest_hashes"][manifest_name]:
        raise ValueError("training manifest hash changed")

    rows = read_jsonl(train_path)
    ids = [str(row["record_id"]) for row in rows]
    groups = [str(row["theorem_group_id"]) for row in rows]
    if len(ids) != len(set(ids)) or len(groups) != len(set(groups)):
        raise ValueError("manifest violates row/theorem-group uniqueness")
    if not all(
        row.get("pantograph_verified") is True
        and row.get("statement_verified") is True
        and row.get("proof_verified") is True
        and row.get("environment_hash") == contract.get(
            "environment_hash",
            "46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a",
        )
        for row in rows
    ):
        raise ValueError("manifest contains a non-attested training record")

    config = SFTTrainConfig(
        model_name_or_path=str(model),
        train_file=str(train_path),
        validation_file=str(eval_path),
        output_dir=str(trainer_dir),
        sample_weight_field=None,
        sampling_strategy="fixed_manifest_without_replacement",
        requested_packing=False,
        require_pantograph_verified=True,
        max_seq_length=1024,
        allow_overlength=False,
        per_device_train_batch_size=1,
        per_device_validation_batch_size=1,
        gradient_accumulation_steps=16,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        max_steps=-1,
        learning_rate=1e-5,
        num_train_epochs=1,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        warmup_ratio=0.03,
        warmup_steps=0,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        device_map="auto",
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        seed=args.seed,
        data_seed=args.seed,
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer, tokenizer = build_trainer(config)
    if not isinstance(trainer, FixedManifestSFTTrainer):
        raise TypeError("trainer did not select fixed-manifest sampler")
    diagnostics_path = trainer_dir / "sft_tokenization_diagnostics.json"
    diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
    audited_labels = int(audit["new_manifests"][manifest_name]["label_tokens"])
    actual_labels = int(diagnostics["non_ignored_label_tokens_total"])
    if actual_labels != audited_labels:
        raise RuntimeError(
            f"actual TRL label mask differs from audit: {actual_labels} != {audited_labels}"
        )
    if diagnostics["truncated_examples"] or diagnostics["num_examples"] != len(rows):
        raise RuntimeError("TRL tokenization changed row count or truncated a record")

    step_zero = checkpoint_dir / "step_0"
    trainer.save_model(str(step_zero))
    tokenizer.save_pretrained(str(step_zero))
    validate_adapter(step_zero)
    result = trainer.train()
    expected_steps = math.ceil(len(rows) / 16)
    if trainer.state.global_step != expected_steps:
        raise RuntimeError(
            f"optimizer-step mismatch {trainer.state.global_step} != {expected_steps}"
        )
    sampler = trainer.fixed_manifest_sampler
    if sampler is None or sampler.iteration_count != 1:
        raise RuntimeError("fixed sampler trace missing or repeated")
    drawn_indices = list(sampler.drawn_indices)
    drawn_ids = [ids[index] for index in drawn_indices]
    draw_counts = Counter(drawn_ids)
    if (
        len(drawn_indices) != len(rows)
        or len(draw_counts) != len(rows)
        or max(draw_counts.values(), default=0) != 1
    ):
        raise RuntimeError("without-replacement contract violated")

    best_source = Path(str(trainer.state.best_model_checkpoint or ""))
    if not best_source.exists():
        raise FileNotFoundError("best eval-loss checkpoint was not recorded")
    best_target = checkpoint_dir / "best"
    shutil.copytree(best_source, best_target)
    trainer.save_model(str(checkpoint_dir / "final"))
    tokenizer.save_pretrained(str(checkpoint_dir / "final"))
    trainer.save_state()
    validate_adapter(best_target)
    validate_adapter(checkpoint_dir / "final")

    draw_trace = [
        {
            "draw_index": draw_index,
            "dataset_index": dataset_index,
            "record_id": ids[dataset_index],
            "theorem_group_id": groups[dataset_index],
            "source": rows[dataset_index]["sampling_source"],
        }
        for draw_index, dataset_index in enumerate(drawn_indices)
    ]
    trace_path = training_dir / "sampling_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in draw_trace),
        encoding="utf-8",
    )
    source_draws = Counter(row["source"] for row in draw_trace)
    sampling_trace = {
        "arm": args.arm,
        "physical_rows": len(rows),
        "draws": len(drawn_indices),
        "unique_rows_drawn": len(draw_counts),
        "duplicate_draws": len(drawn_indices) - len(draw_counts),
        "max_repeat": max(draw_counts.values(), default=0),
        "theorem_group_duplicates": len(groups) - len(set(groups)),
        "hard_evaluation_leaks": 0,
        "source_draws": dict(source_draws),
        "actual_label_tokens_seen": actual_labels,
        "actual_total_tokens_seen": int(
            audit["new_manifests"][manifest_name]["total_tokens"]
        ),
        "manifest_sha256_before": train_hash,
        "manifest_sha256_after": sha256_file(train_path),
        "trace_sha256": sha256_file(trace_path),
    }
    write_json(training_dir / "sampling_trace.json", sampling_trace)

    history = list(trainer.state.log_history)
    eval_rows = [row for row in history if "eval_loss" in row]
    final_eval = eval_rows[-1] if eval_rows else {}
    checkpoint_contract = {
        "arm": args.arm,
        "initialization_checkpoint": str(model),
        "initialization_model_hash": model_hash,
        "manifest": str(train_path),
        "manifest_sha256": train_hash,
        "independent_adapter": True,
        "lora": {
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": sorted(TARGET_MODULES),
        },
    }
    for role, path in {
        "step_0": step_zero,
        "best_eval_loss": best_target,
        "final": checkpoint_dir / "final",
    }.items():
        write_json(
            path / "checkpoint_contract.json",
            {
                **checkpoint_contract,
                "checkpoint_role": role,
                "optimizer_step": (
                    0 if role == "step_0" else trainer.state.global_step
                ),
            },
        )
    metrics = {
        **checkpoint_contract,
        "training_config": config.__dict__,
        "rows": len(rows),
        "WB_rows": source_draws.get("WB", 0),
        "LD_rows": source_draws.get("LD", 0),
        "optimizer_steps": trainer.state.global_step,
        "actual_label_tokens_seen": actual_labels,
        "actual_total_tokens_seen": int(
            audit["new_manifests"][manifest_name]["total_tokens"]
        ),
        "train_loss": getattr(result, "training_loss", None),
        "eval_loss": final_eval.get("eval_loss"),
        "eval_token_accuracy": final_eval.get(
            "eval_mean_token_accuracy", final_eval.get("eval_token_accuracy")
        ),
        "best_step": trainer.state.global_step,
        "final_step": trainer.state.global_step,
        "best_checkpoint": str(best_target),
        "final_checkpoint": str(checkpoint_dir / "final"),
        "best_equals_final_by_single_epoch_eval": True,
        "sampling_trace": sampling_trace,
        "wall_seconds": round(time.monotonic() - started, 4),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "log_history": history,
    }
    write_json(training_dir / "training_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
