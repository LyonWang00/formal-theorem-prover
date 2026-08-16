#!/usr/bin/env python3
"""Train one B/C arm with one auditable, without-replacement epoch."""

from __future__ import annotations

import argparse
import json
import math
import resource
import time
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from transformers import TrainerCallback

from lean_prover.lean_training.data.random_manifest import (
    file_sha256,
    read_jsonl,
    record_id,
    statement_id,
    validate_manifest,
    write_json,
)
from lean_prover.lean_training.sft_pipeline.config import SFTTrainConfig
from lean_prover.lean_training.sft_pipeline.trainer import (
    FixedManifestSFTTrainer,
    build_trainer,
)


EXPECTED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


class ExactCheckpointCallback(TrainerCallback):
    def __init__(self, steps: set[int]) -> None:
        self.steps = steps

    def on_step_end(self, args, state, control, **kwargs):
        del args, kwargs
        if state.global_step in self.steps:
            control.should_save = True
        return control


def validate_adapter(path: Path) -> None:
    payload = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if set(payload.get("target_modules") or []) != EXPECTED_TARGET_MODULES:
        raise ValueError(f"unexpected target modules: {path}")
    if payload.get("r") != 32 or payload.get("lora_alpha") != 64:
        raise ValueError(f"unexpected LoRA rank/alpha: {path}")
    if float(payload.get("lora_dropout")) != 0.05:
        raise ValueError(f"unexpected LoRA dropout: {path}")


def source_token_stats(rows: list[dict[str, Any]], tokenizer) -> dict[str, Any]:
    source_rows = Counter(str(row["sampling_source"]) for row in rows)
    source_tokens: Counter[str] = Counter()
    proof_lengths: dict[str, list[int]] = {}
    for row in rows:
        source = str(row["sampling_source"])
        proof = str(row.get("completion") or row.get("proof") or "")
        count = len(tokenizer(proof, add_special_tokens=False)["input_ids"])
        source_tokens[source] += count
        proof_lengths.setdefault(source, []).append(count)
    total = sum(source_tokens.values())
    return {
        "source_physical_rows": dict(source_rows),
        "source_label_tokens": dict(source_tokens),
        "source_label_token_share": {
            source: count / max(1, total) for source, count in source_tokens.items()
        },
        "source_mean_proof_tokens": {
            source: sum(values) / len(values)
            for source, values in proof_lengths.items()
        },
        "label_token_count": total,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True, choices=("B1", "B2", "C1", "C2", "C3"))
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--train", required=True, type=Path)
    parser.add_argument("--eval", required=True, type=Path)
    parser.add_argument("--experiment-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--trace-dir", required=True, type=Path)
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--scheduler", required=True)
    parser.add_argument("--warmup-ratio", type=float, default=0.0)
    parser.add_argument("--warmup-steps", type=int, default=0)
    parser.add_argument("--checkpoint-steps", required=True)
    parser.add_argument("--seed", type=int, default=20260721)
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {output}")
    trace_dir = args.trace_dir.resolve()
    trace_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.train)
    allow_discovery = args.arm in {"B2", "C2", "C3"}
    validation = validate_manifest(
        rows, allow_origin_discovery=allow_discovery
    )
    physical_rows = len(rows)
    effective_batch_size = 16
    expected_steps = math.ceil(physical_rows / effective_batch_size)
    requested_steps = {
        int(value.strip())
        for value in args.checkpoint_steps.split(",")
        if value.strip()
    }
    if 0 not in requested_steps:
        raise ValueError("checkpoint schedule must contain step 0")
    requested_steps.add(expected_steps)
    if any(step < 0 or step > expected_steps for step in requested_steps):
        raise ValueError(
            f"checkpoint schedule {sorted(requested_steps)} exceeds final step {expected_steps}"
        )
    experiment_manifest = json.loads(
        args.experiment_manifest.read_text(encoding="utf-8")
    )
    if str(args.model.resolve()) != experiment_manifest["m0"]["absolute_path"]:
        raise ValueError("training model does not match recorded clean M0 path")
    train_hash = file_sha256(args.train)
    expected_hash = experiment_manifest["sha256"][args.arm]
    if train_hash != expected_hash:
        raise ValueError(
            f"training manifest hash mismatch: {train_hash} != {expected_hash}"
        )

    config = SFTTrainConfig(
        model_name_or_path=str(args.model.resolve()),
        train_file=str(args.train.resolve()),
        validation_file=str(args.eval.resolve()),
        output_dir=str(output),
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
        learning_rate=args.learning_rate,
        num_train_epochs=1,
        logging_steps=1,
        eval_strategy="epoch",
        save_strategy="no",
        save_total_limit=None,
        load_best_model_at_end=False,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=0.01,
        lr_scheduler_type=args.scheduler,
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
        raise TypeError("effective trainer is not FixedManifestSFTTrainer")
    trainer.add_callback(ExactCheckpointCallback(requested_steps - {0}))
    step_zero = output / "step_0"
    trainer.save_model(str(step_zero))
    tokenizer.save_pretrained(str(step_zero))
    validate_adapter(step_zero)

    trainable = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    result = trainer.train()
    if trainer.state.global_step != expected_steps:
        raise RuntimeError(
            f"optimizer step contract violated: {trainer.state.global_step} != {expected_steps}"
        )
    sampler = trainer.fixed_manifest_sampler
    if sampler is None:
        raise RuntimeError("fixed manifest sampler trace was not captured")
    drawn_indices = list(sampler.drawn_indices)
    drawn_record_ids = [record_id(rows[index]) for index in drawn_indices]
    counts = Counter(drawn_record_ids)
    duplicate_draws = len(drawn_record_ids) - len(counts)
    if not (
        len(drawn_indices) == physical_rows
        and len(counts) == physical_rows
        and duplicate_draws == 0
        and max(counts.values(), default=0) == 1
        and sampler.iteration_count == 1
    ):
        raise RuntimeError("without-replacement sampling contract violated")

    source_stats = source_token_stats(rows, tokenizer)
    draw_rows = []
    for draw_index, dataset_index in enumerate(drawn_indices):
        row = rows[dataset_index]
        draw_rows.append(
            {
                "draw_index": draw_index,
                "dataset_index": dataset_index,
                "record_id": record_id(row),
                "statement_id": statement_id(row),
                "source": row["sampling_source"],
            }
        )
    trace_jsonl = trace_dir / f"{args.arm}_sampling_trace.jsonl"
    trace_jsonl.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in draw_rows),
        encoding="utf-8",
    )
    drawn_sources = Counter(row["source"] for row in draw_rows)
    trace = {
        "arm": args.arm,
        "requested_sampling_strategy": "fixed_manifest_without_replacement",
        "effective_sampling_strategy": "fixed_manifest_without_replacement",
        "replacement": False,
        "shuffle": True,
        "drop_last": False,
        "physical_rows": physical_rows,
        "total_draws": len(drawn_indices),
        "unique_records_seen": len(counts),
        "duplicate_draw_count": duplicate_draws,
        "max_draws_per_record": max(counts.values(), default=0),
        "mean_draws_per_record": len(drawn_indices) / len(counts),
        "optimizer_steps": trainer.state.global_step,
        "micro_batches": physical_rows,
        "effective_batch_size": effective_batch_size,
        "last_partial_batch_size": physical_rows % effective_batch_size or effective_batch_size,
        "sampler_iteration_count": sampler.iteration_count,
        "anchor_physical_rows": source_stats["source_physical_rows"].get("anchor", 0),
        "expert_physical_rows": source_stats["source_physical_rows"].get("expert", 0),
        "anchor_draws": drawn_sources.get("anchor", 0),
        "expert_draws": drawn_sources.get("expert", 0),
        "anchor_unique_seen": len(
            {row["record_id"] for row in draw_rows if row["source"] == "anchor"}
        ),
        "expert_unique_seen": len(
            {row["record_id"] for row in draw_rows if row["source"] == "expert"}
        ),
        "anchor_duplicate_draws": 0,
        "expert_duplicate_draws": 0,
        "anchor_label_tokens": source_stats["source_label_tokens"].get("anchor", 0),
        "expert_label_tokens": source_stats["source_label_tokens"].get("expert", 0),
        "anchor_label_token_share": source_stats["source_label_token_share"].get("anchor", 0.0),
        "expert_label_token_share": source_stats["source_label_token_share"].get("expert", 0.0),
        "anchor_mean_proof_tokens": source_stats["source_mean_proof_tokens"].get("anchor"),
        "expert_mean_proof_tokens": source_stats["source_mean_proof_tokens"].get("expert"),
        "label_token_count": source_stats["label_token_count"],
        "train_manifest": str(args.train.resolve()),
        "train_manifest_sha256": train_hash,
        "trace_jsonl": str(trace_jsonl),
        "trace_jsonl_sha256": file_sha256(trace_jsonl),
    }
    write_json(trace_dir / f"{args.arm}_sampling_trace.json", trace)

    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    trainer.save_state()
    saved = [step_zero]
    for step in sorted(requested_steps - {0}):
        source = output / f"checkpoint-{step}"
        target = output / f"step_{step}"
        if not source.exists():
            raise FileNotFoundError(f"required checkpoint was not saved: {source}")
        source.rename(target)
        validate_adapter(target)
        saved.append(target)
    validate_adapter(output)

    checkpoint_contract = {
        "checkpoint_role": "expert_sft_ablation_candidate",
        "not_m1": True,
        "arm": args.arm,
        "initialization_checkpoint": str(args.model.resolve()),
        "m0_model_hash": experiment_manifest["m0"]["model_hash"],
        "m0_config_hash": experiment_manifest["m0"]["config_hash"],
        "tokenizer_hash": experiment_manifest["m0"]["tokenizer_hash"],
        "generation_config_hash": experiment_manifest["m0"]["generation_config_hash"],
        "training_manifest_sha256": train_hash,
        "lora": {
            "rank": 32,
            "alpha": 64,
            "dropout": 0.05,
            "target_modules": sorted(EXPECTED_TARGET_MODULES),
        },
    }
    for path in saved:
        step = int(path.name.split("_")[-1])
        write_json(path / "checkpoint_contract.json", {**checkpoint_contract, "step": step})

    history = list(trainer.state.log_history)
    eval_rows = [row for row in history if "eval_loss" in row]
    metrics = {
        **checkpoint_contract,
        "validation": validation,
        "training_config": config.__dict__,
        "physical_rows": physical_rows,
        "optimizer_steps": trainer.state.global_step,
        "expected_optimizer_steps": expected_steps,
        "effective_batch_size": effective_batch_size,
        "trainable_parameter_count": trainable,
        "train_loss": getattr(result, "training_loss", None),
        "train_loss_by_step": [row for row in history if "loss" in row],
        "eval_loss": eval_rows[-1].get("eval_loss") if eval_rows else None,
        "eval_token_accuracy": (
            eval_rows[-1].get(
                "eval_mean_token_accuracy", eval_rows[-1].get("eval_token_accuracy")
            )
            if eval_rows
            else None
        ),
        "checkpoint_save_steps": sorted(requested_steps),
        "saved_checkpoints": [str(path) for path in saved],
        "sampling_trace": trace,
        "wall_seconds": round(time.monotonic() - started, 4),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }
    write_json(output / "training_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
