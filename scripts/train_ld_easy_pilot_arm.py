#!/usr/bin/env python3
"""Train one no-replacement LD-easy pilot arm independently from clean M0."""

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


ARMS = {
    "LDE-A2-WB100": "A2_WB1000",
    "LDE-B2-LD10": "B2_WB900_LDE100",
    "LDE-C2-LD20": "C2_WB800_LDE200",
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


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.open(encoding="utf-8-sig")
        if line.strip()
    ]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
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
    parser.add_argument("--arm", choices=tuple(ARMS), required=True)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/pilot_sft"),
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
    parser.add_argument("--seed", type=int, default=20260803)
    args = parser.parse_args()
    root = args.root.resolve()
    model = args.model.resolve()
    eval_path = args.eval.resolve()
    manifest_name = ARMS[args.arm]
    manifest = root / f"manifests/{manifest_name}.jsonl"
    training_dir = root / "training" / args.arm
    trainer_dir = training_dir / "trainer"
    checkpoint_dir = root / "checkpoints" / args.arm
    if training_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(f"refusing to overwrite partial/completed {args.arm}")

    audit = json.loads(
        (root / "audit/pilot_manifest_audit.json").read_text(encoding="utf-8")
    )
    inventory = json.loads(
        (root / "audit/input_inventory.json").read_text(encoding="utf-8")
    )
    if not audit["manual_gate"]["pilot_gate_passed"]:
        raise RuntimeError("manual difficulty gate is not frozen as passed")
    if str(model) != str(Path(audit["clean_m0_path"]).resolve()):
        raise ValueError("model path is not frozen clean M0")
    if sha256(model / "model.safetensors") != audit["clean_m0_model_sha256"]:
        raise ValueError("clean M0 identity changed")
    if sha256(manifest) != inventory[manifest_name]:
        raise ValueError("training manifest identity changed")
    rows = read_jsonl(manifest)
    ids = [str(row["record_id"]) for row in rows]
    groups = [str(row["theorem_group_id"]) for row in rows]
    if len(rows) != 1000 or len(ids) != len(set(ids)) or len(groups) != len(set(groups)):
        raise ValueError("manifest violates 1000-row/no-replacement contract")
    if not all(
        row.get("pantograph_verified") is True
        and row.get("statement_verified") is True
        and row.get("proof_verified") is True
        for row in rows
    ):
        raise ValueError("manifest contains an unattested training record")

    config = SFTTrainConfig(
        model_name_or_path=str(model),
        train_file=str(manifest),
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
        learning_rate=2e-6,
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
        raise TypeError("trainer did not select the fixed-manifest sampler")
    diagnostics = json.loads(
        (trainer_dir / "sft_tokenization_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    expected_labels = int(
        audit["manifests"][manifest_name]["label_tokens"]
    )
    if diagnostics["non_ignored_label_tokens_total"] != expected_labels:
        raise RuntimeError("actual supervised label mask differs from manifest audit")
    if diagnostics["truncated_examples"] or diagnostics["num_examples"] != 1000:
        raise RuntimeError("tokenization truncated or changed row count")

    step_zero = checkpoint_dir / "step_0"
    trainer.save_model(str(step_zero))
    tokenizer.save_pretrained(str(step_zero))
    validate_adapter(step_zero)
    result = trainer.train()
    expected_steps = math.ceil(1000 / 16)
    if trainer.state.global_step != expected_steps:
        raise RuntimeError(
            f"optimizer steps {trainer.state.global_step} != {expected_steps}"
        )
    sampler = trainer.fixed_manifest_sampler
    if sampler is None or sampler.iteration_count != 1:
        raise RuntimeError("fixed-manifest sampler trace missing")
    indices = list(sampler.drawn_indices)
    drawn_ids = [ids[index] for index in indices]
    counts = Counter(drawn_ids)
    if len(indices) != 1000 or len(counts) != 1000 or max(counts.values()) != 1:
        raise RuntimeError("without-replacement training contract violated")

    best_source = Path(str(trainer.state.best_model_checkpoint or ""))
    if not best_source.exists():
        raise FileNotFoundError("best eval-loss checkpoint is missing")
    shutil.copytree(best_source, checkpoint_dir / "best")
    trainer.save_model(str(checkpoint_dir / "final"))
    tokenizer.save_pretrained(str(checkpoint_dir / "final"))
    trainer.save_state()
    validate_adapter(checkpoint_dir / "best")
    validate_adapter(checkpoint_dir / "final")

    trace_path = training_dir / "sampling_trace.jsonl"
    trace_path.parent.mkdir(parents=True, exist_ok=True)
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
                        "source": row["sampling_source"],
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    metrics = dict(getattr(result, "metrics", {}) or {})
    summary = {
        "arm": args.arm,
        "manifest": str(manifest),
        "manifest_sha256": sha256(manifest),
        "clean_m0_sha256": audit["clean_m0_model_sha256"],
        "initialization": "independent_clean_m0",
        "learning_rate": 2e-6,
        "epochs": 1,
        "physical_rows": 1000,
        "optimizer_steps": trainer.state.global_step,
        "effective_batch_size": 16,
        "sampling": {
            "strategy": "fixed_manifest_without_replacement",
            "draws": len(indices),
            "unique_rows": len(counts),
            "duplicate_draws": len(indices) - len(counts),
            "max_repeat": max(counts.values(), default=0),
        },
        "train_metrics": metrics,
        "best_model_checkpoint": str(checkpoint_dir / "best"),
        "final_checkpoint": str(checkpoint_dir / "final"),
        "eval_loss": trainer.state.best_metric,
        "gpu_peak_bytes": int(torch.cuda.max_memory_allocated()),
        "ram_peak_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        "wall_seconds": round(time.monotonic() - started, 4),
    }
    write_json(training_dir / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
