#!/usr/bin/env python3
"""Train one independently initialized EOS-fixed anchor-ratio arm."""

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
    "A0": "ANCHOR-A0-EOS-FIXED",
    "A5": "ANCHOR-A5-EOS-FIXED",
    "A10": "ANCHOR-A10-EOS-FIXED",
    "A20": "ANCHOR-A20-EOS-FIXED",
}
EXPECTED_BASE_HASH = (
    "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
)
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


def directory_tensor_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(
        item
        for item in path.rglob("*")
        if item.is_file()
        and (
            item.suffix in {".safetensors", ".bin"}
            or item.name == "adapter_config.json"
        )
    )
    if not files:
        raise FileNotFoundError(f"no adapter tensors found under {path}")
    for item in files:
        digest.update(item.relative_to(path).as_posix().encode())
        digest.update(bytes.fromhex(sha256(item)))
    return digest.hexdigest()


def validate_adapter(path: Path) -> None:
    payload = read_json(path / "adapter_config.json")
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
        default=Path("outputs/anchor_ratio_eos_fixed"),
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/Qwen2.5-1.5B-Instruct"),
    )
    parser.add_argument(
        "--eval",
        type=Path,
        default=Path("data/processed/lean_workbook_verified_v2/eval.jsonl"),
    )
    args = parser.parse_args()

    root = args.root.resolve()
    name = ARMS[args.arm]
    arm_root = root / "training" / name
    manifest = arm_root / "input/effective_train.jsonl"
    trainer_dir = arm_root / "trainer"
    checkpoint_dir = root / "checkpoints" / name
    if trainer_dir.exists() or checkpoint_dir.exists():
        raise FileExistsError(f"refusing to overwrite partial/completed {name}")

    model = args.model.resolve()
    eval_path = args.eval.resolve()
    identity = read_json(root / "audit/frozen_manifest_identity.json")
    gate_path = root / f"audit/eos_gate_{args.arm}.json"
    contract_path = root / "audit/frozen_training_contract.json"
    gate = read_json(gate_path)
    contract = read_json(contract_path)
    training_contract_hash = sha256(contract_path)
    eos_gate_hash = sha256(gate_path)
    frozen_source = root / f"manifests/{args.arm}_source_manifest.jsonl"
    expected_source_hash = identity["arms"][args.arm]["source_sha256"]
    if sha256(model / "model.safetensors") != EXPECTED_BASE_HASH:
        raise ValueError("base model identity changed")
    if sha256(frozen_source) != expected_source_hash:
        raise ValueError("frozen source manifest identity changed")
    if sha256(manifest) != gate["effective_manifest_sha256"]:
        raise ValueError("effective EOS manifest identity changed")
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

    rows = read_jsonl(manifest)
    ids = [str(row["record_id"]) for row in rows]
    groups = [str(row["theorem_group_id"]) for row in rows]
    if (
        len(rows) != 3000
        or len(set(ids)) != 3000
        or len(set(groups)) != 3000
    ):
        raise ValueError("effective manifest violates no-replacement identity")
    if not all(
        row.get("pantograph_verified") is True
        and row.get("statement_verified") is True
        and row.get("proof_verified") is True
        for row in rows
    ):
        raise ValueError("effective manifest contains unattested rows")

    static = contract["static_sft_config"]
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
    for key, value in static.items():
        if key in {"model_name_or_path", "train_file", "validation_file", "output_dir"}:
            continue
        if getattr(config, key) != value:
            raise ValueError(f"training contract drift at {key}")
    write_json(
        arm_root / "training_identity.json",
        {
            "arm": args.arm,
            "model_name": name,
            "base_model_sha256": EXPECTED_BASE_HASH,
            "source_manifest_sha256": expected_source_hash,
            "effective_manifest_sha256": sha256(manifest),
            "training_contract_sha256": training_contract_hash,
            "eos_gate_sha256": eos_gate_hash,
            "initialization": "independent_base_new_adapter",
            "optimizer_state_shared": False,
            "adapter_state_shared": False,
            "resolved_config": config.__dict__,
        },
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer, tokenizer = build_trainer(config)
    if not isinstance(trainer, FixedManifestSFTTrainer):
        raise TypeError("fixed-manifest sampler was not selected")
    actual_gate_document = read_json(
        trainer_dir / "sft_supervised_eos_contract.json"
    )
    actual_gate = actual_gate_document.get("effective_trainer_labels") or {}
    if (
        not actual_gate.get("passed")
        or actual_gate.get("records_with_supervised_eos") != 3000
        or actual_gate.get("records_whose_last_valid_label_is_eos") != 3000
        or actual_gate.get("zero_label_records") != 0
        or actual_gate.get("records_with_semantic_truncation") != 0
    ):
        raise RuntimeError("trainer-effective labels failed the EOS contract")
    diagnostics = read_json(trainer_dir / "sft_tokenization_diagnostics.json")
    if (
        diagnostics["num_examples"] != 3000
        or diagnostics["truncated_examples"] != 0
        or diagnostics["non_ignored_label_tokens_total"]
        != gate["valid_label_tokens_total"]
    ):
        raise RuntimeError("trainer token exposure differs from frozen audit")

    step_zero = checkpoint_dir / "step_0"
    trainer.save_model(str(step_zero))
    tokenizer.save_pretrained(str(step_zero))
    validate_adapter(step_zero)
    step_zero_hash = directory_tensor_hash(step_zero)

    result = trainer.train()
    if trainer.state.global_step != math.ceil(3000 / 16):
        raise RuntimeError("unexpected optimizer step count")
    sampler = trainer.fixed_manifest_sampler
    if sampler is None or sampler.iteration_count != 1:
        raise RuntimeError("fixed-manifest sampler trace missing")
    indices = list(sampler.drawn_indices)
    counts = Counter(ids[index] for index in indices)
    if len(indices) != 3000 or len(counts) != 3000 or max(counts.values()) != 1:
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

    trace_path = arm_root / "sampling_trace.jsonl"
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
    eval_rows = [row for row in trainer.state.log_history if "eval_loss" in row]
    final_eval = eval_rows[-1] if eval_rows else {}
    best_hash = directory_tensor_hash(checkpoint_dir / "best")
    final_hash = directory_tensor_hash(checkpoint_dir / "final")
    summary = {
        "arm": args.arm,
        "model_name": name,
        "base_model_sha256": EXPECTED_BASE_HASH,
        "source_manifest_sha256": expected_source_hash,
        "effective_manifest_sha256": sha256(manifest),
        "training_contract_sha256": training_contract_hash,
        "eos_gate_sha256": eos_gate_hash,
        "initialization": "independent_base_new_adapter",
        "optimizer_steps": trainer.state.global_step,
        "physical_rows": 3000,
        "draws": len(indices),
        "unique_draws": len(counts),
        "duplicate_draws": len(indices) - len(counts),
        "max_repeat": max(counts.values()),
        "supervised_label_tokens": diagnostics["non_ignored_label_tokens_total"],
        "supervised_eos_records": actual_gate["records_with_supervised_eos"],
        "step_0_adapter_hash": step_zero_hash,
        "best_adapter_hash": best_hash,
        "final_adapter_hash": final_hash,
        "step_0_differs_from_best": step_zero_hash != best_hash,
        "step_0_differs_from_final": step_zero_hash != final_hash,
        "eval_loss": final_eval.get("eval_loss"),
        "eval_token_accuracy": final_eval.get(
            "eval_mean_token_accuracy", final_eval.get("eval_token_accuracy")
        ),
        "metrics": dict(getattr(result, "metrics", {}) or {}),
        "best_checkpoint": str(checkpoint_dir / "best"),
        "runtime_seconds": time.monotonic() - started,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "process_max_rss_kib": int(
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        ),
        "resolved_config": config.__dict__,
    }
    write_json(arm_root / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
