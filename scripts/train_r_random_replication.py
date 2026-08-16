#!/usr/bin/env python3
"""Train one independently initialized R-Random replication adapter."""

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
from scripts.prepare_r_random_replication import (
    BASE,
    EXPECTED_BASE_HASH,
    NEW_MODELS,
    OUTPUT,
)
from scripts.prepare_wb_ld_budget_support_ablation import sha256
from scripts.train_anchor_ratio_eos_fixed_arm import (
    directory_tensor_hash,
    validate_adapter,
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", choices=tuple(NEW_MODELS), required=True)
    parser.add_argument("--project", type=Path, default=Path("."))
    parser.add_argument("--root", type=Path, default=OUTPUT)
    parser.add_argument("--model", type=Path, default=BASE)
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("data/processed/lean_workbook_verified_v2/eval.jsonl"),
    )
    args = parser.parse_args()

    project = args.project.resolve()
    root = (project / args.root).resolve()
    model = (project / args.model).resolve()
    eval_path = (project / args.eval).resolve()
    name = args.model_name
    relative = NEW_MODELS[name][1]
    source_manifest = root / relative
    arm_root = root / "training" / name
    manifest = arm_root / "input/effective_train.jsonl"
    trainer_dir = arm_root / "trainer"
    checkpoint_dir = root / "checkpoints" / name
    if trainer_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(f"refusing to overwrite partial/completed {name}")

    gate_name = "s2" if "S2" in name else "s3"
    gate_path = root / "audit" / f"eos_gate_{gate_name}.json"
    contract_path = root / "audit/frozen_training_contract.json"
    gate = read_json(gate_path)
    contract = read_json(contract_path)
    rows = read_jsonl(manifest)
    row_count = len(rows)
    if sha256(model / "model.safetensors") != EXPECTED_BASE_HASH:
        raise ValueError("base model identity changed")
    if sha256(source_manifest) != gate["source_manifest_sha256"]:
        raise ValueError("source manifest identity changed")
    if sha256(manifest) != gate["effective_manifest_sha256"]:
        raise ValueError("effective EOS manifest identity changed")
    if gate.get("leakage_gate_status") != "LEAKAGE_GATE_PASSED":
        raise RuntimeError(f"{name} did not pass the protected leakage gate")
    if (
        gate.get("status") != "EOS_GATE_PASSED"
        or gate.get("total_records") != 3000
        or gate.get("records_with_supervised_eos") != 3000
        or gate.get("records_whose_last_valid_label_is_eos") != 3000
        or gate.get("records_without_supervised_eos") != 0
        or gate.get("zero_label_records") != 0
        or gate.get("records_with_semantic_truncation") != 0
    ):
        raise RuntimeError(f"{name} did not pass the formal EOS launch gate")

    ids = [str(row["record_id"]) for row in rows]
    groups = [str(row["theorem_group_id"]) for row in rows]
    if row_count != 3000 or len(set(ids)) != 3000 or len(set(groups)) != 3000:
        raise ValueError("effective manifest violates no-replacement identity")
    if not all(
        row.get("pantograph_verified") is True
        and row.get("statement_verified") is True
        and row.get("proof_verified") is True
        for row in rows
    ):
        raise ValueError("effective manifest contains unattested rows")

    config = SFTTrainConfig(
        model_name_or_path=str(model),
        train_file=str(manifest),
        validation_file=str(eval_path),
        output_dir=str(trainer_dir),
        sample_weight_field=None,
        sampling_strategy="fixed_manifest_without_replacement",
        requested_packing=True,
        require_pantograph_verified=True,
        require_supervised_eos=True,
        max_seq_length=1024,
        allow_overlength=False,
        per_device_train_batch_size=1,
        per_device_validation_batch_size=1,
        gradient_accumulation_steps=16,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        max_steps=-1,
        learning_rate=5e-5,
        num_train_epochs=1,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_steps=200,
        save_total_limit=2,
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
        seed=42,
        data_seed=42,
    )
    for key, value in contract["static_sft_config"].items():
        if key in {"model_name_or_path", "train_file", "validation_file", "output_dir"}:
            continue
        if getattr(config, key) != value:
            raise ValueError(f"training contract drift at {key}")

    identity = {
        "model": name,
        "base_model_sha256": EXPECTED_BASE_HASH,
        "source_manifest_sha256": sha256(source_manifest),
        "effective_manifest_sha256": sha256(manifest),
        "training_contract_sha256": sha256(contract_path),
        "eos_gate_sha256": sha256(gate_path),
        "physical_rows": row_count,
        "initialization": "independent_base_new_adapter",
        "optimizer_state_shared": False,
        "scheduler_state_shared": False,
        "gradient_state_shared": False,
        "adapter_state_shared": False,
        "resume_from_checkpoint": False,
        "resolved_config": config.__dict__,
    }
    write_json(arm_root / "training_identity.json", identity)

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer, tokenizer = build_trainer(config)
    if not isinstance(trainer, FixedManifestSFTTrainer):
        raise TypeError("fixed-manifest sampler was not selected")
    actual_gate = read_json(
        trainer_dir / "sft_supervised_eos_contract.json"
    )["effective_trainer_labels"]
    diagnostics = read_json(trainer_dir / "sft_tokenization_diagnostics.json")
    if (
        not actual_gate.get("passed")
        or actual_gate.get("records_with_supervised_eos") != row_count
        or actual_gate.get("records_whose_last_valid_label_is_eos") != row_count
        or actual_gate.get("zero_label_records") != 0
        or actual_gate.get("records_with_semantic_truncation") != 0
        or diagnostics["num_examples"] != row_count
        or diagnostics["truncated_examples"] != 0
        or diagnostics["non_ignored_label_tokens_total"]
        != gate["valid_label_tokens_total"]
    ):
        raise RuntimeError("trainer-effective EOS/token exposure gate failed")

    step_zero = checkpoint_dir / "step_0"
    trainer.save_model(str(step_zero))
    tokenizer.save_pretrained(str(step_zero))
    validate_adapter(step_zero)
    step_zero_hash = directory_tensor_hash(step_zero)
    result = trainer.train()
    expected_steps = math.ceil(row_count / 16)
    if trainer.state.global_step != 188 or expected_steps != 188:
        raise RuntimeError(
            f"unexpected optimizer steps: {trainer.state.global_step} != 188"
        )
    sampler = trainer.fixed_manifest_sampler
    if sampler is None or sampler.iteration_count != 1:
        raise RuntimeError("fixed-manifest sampler trace missing")
    indices = list(sampler.drawn_indices)
    counts = Counter(ids[index] for index in indices)
    if (
        len(indices) != row_count
        or len(counts) != row_count
        or max(counts.values()) != 1
    ):
        raise RuntimeError("without-replacement training contract violated")

    best_source = Path(str(trainer.state.best_model_checkpoint or ""))
    if not best_source.is_dir():
        raise FileNotFoundError("best eval-loss checkpoint is missing")
    shutil.copytree(best_source, checkpoint_dir / "best")
    trainer.save_model(str(checkpoint_dir / "final"))
    tokenizer.save_pretrained(str(checkpoint_dir / "final"))
    trainer.save_state()
    validate_adapter(checkpoint_dir / "best")
    validate_adapter(checkpoint_dir / "final")

    trace = arm_root / "sampling_trace.jsonl"
    with trace.open("w", encoding="utf-8", newline="\n") as handle:
        for draw_index, dataset_index in enumerate(indices):
            row = rows[dataset_index]
            handle.write(
                json.dumps(
                    {
                        "draw_index": draw_index,
                        "dataset_index": dataset_index,
                        "record_id": ids[dataset_index],
                        "theorem_group_id": groups[dataset_index],
                        "source": row["sampling_source"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    eval_rows = [row for row in trainer.state.log_history if "eval_loss" in row]
    final_eval = eval_rows[-1] if eval_rows else {}
    trajectory = [
        {
            key: row[key]
            for key in (
                "step",
                "epoch",
                "loss",
                "learning_rate",
                "grad_norm",
                "eval_loss",
                "eval_mean_token_accuracy",
                "eval_token_accuracy",
            )
            if key in row
        }
        for row in trainer.state.log_history
    ]
    input_tokens = sum(int(row["input_tokens"]) for row in rows)
    source_label_tokens = sum(
        int(row["label_tokens"]) for row in rows
    )
    total_tokens = sum(int(row["total_tokens"]) for row in rows)
    summary = {
        **identity,
        "physical_rows": row_count,
        "wb_rows": sum(row.get("sampling_source") == "WB" for row in rows),
        "ld_rows": sum(row.get("sampling_source") == "LD" for row in rows),
        "micro_batches": row_count,
        "optimizer_steps": trainer.state.global_step,
        "examples_seen": len(indices),
        "draws": len(indices),
        "unique_draws": len(counts),
        "duplicate_draws": len(indices) - len(counts),
        "max_repeat": max(counts.values()),
        "input_tokens_seen": input_tokens,
        "source_label_tokens_seen_excluding_added_eos": source_label_tokens,
        "supervised_label_tokens_seen": diagnostics[
            "non_ignored_label_tokens_total"
        ],
        "total_source_tokens_seen_excluding_added_eos": total_tokens,
        "supervised_eos_records": actual_gate["records_with_supervised_eos"],
        "step_0_adapter_hash": step_zero_hash,
        "best_adapter_hash": directory_tensor_hash(checkpoint_dir / "best"),
        "final_adapter_hash": directory_tensor_hash(checkpoint_dir / "final"),
        "eval_loss": final_eval.get("eval_loss"),
        "eval_token_accuracy": final_eval.get(
            "eval_mean_token_accuracy", final_eval.get("eval_token_accuracy")
        ),
        "training_trajectory": trajectory,
        "metrics": dict(getattr(result, "metrics", {}) or {}),
        "best_checkpoint": str(checkpoint_dir / "best"),
        "runtime_seconds": time.monotonic() - started,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "process_max_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
    }
    write_json(arm_root / "training_summary.json", summary)
    resource_path = root / "runtime/training_resources.jsonl"
    resource_path.parent.mkdir(parents=True, exist_ok=True)
    with resource_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(
            json.dumps(
                {
                    "model": name,
                    "runtime_seconds": summary["runtime_seconds"],
                    "gpu_peak_allocated_bytes": summary[
                        "gpu_peak_allocated_bytes"
                    ],
                    "gpu_peak_reserved_bytes": summary[
                        "gpu_peak_reserved_bytes"
                    ],
                    "process_max_rss_kib": summary["process_max_rss_kib"],
                },
                ensure_ascii=False,
            )
            + "\n"
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
