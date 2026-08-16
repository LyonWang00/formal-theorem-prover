#!/usr/bin/env python3
"""Train one isolated CLEAN-M0/A0 causal reproduction arm."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import resource
import shutil
import time
from pathlib import Path
from typing import Any

import torch
from transformers import AutoTokenizer

from lean_prover.lean_training.sft_pipeline.config import SFTTrainConfig
from lean_prover.lean_training.sft_pipeline.trainer import (
    FixedManifestSFTTrainer,
    build_trainer,
)


EXPERIMENTS = {
    "R0_HISTORICAL": {
        "source": "data/processed/lean_workbook_verified_v2/train.jsonl",
        "pipeline": "historical_supervised_eos",
        "sampling_strategy": "auto",
        "purpose": "historical data, historical EOS supervision and default sampler",
    },
    "R1_HIST_DATA_CURRENT_PIPELINE": {
        "source": "data/processed/lean_workbook_verified_v2/train.jsonl",
        "pipeline": "current_missing_eos",
        "sampling_strategy": "fixed_manifest_without_replacement",
        "purpose": "historical data with current no-EOS pipeline",
    },
    "R2_CURRENT_DATA_HIST_PIPELINE": {
        "source": (
            "outputs/initial_anchor_ratio_ablation/manifests/"
            "A0_WB3000_LD0.jsonl"
        ),
        "pipeline": "historical_supervised_eos",
        "sampling_strategy": "auto",
        "purpose": "current reordered A0 data with historical EOS pipeline",
    },
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
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def validate_adapter(path: Path) -> None:
    payload = json.loads(
        (path / "adapter_config.json").read_text(encoding="utf-8")
    )
    if set(payload.get("target_modules") or []) != TARGET_MODULES:
        raise ValueError(f"unexpected target modules in {path}")
    if payload.get("r") != 32 or payload.get("lora_alpha") != 64:
        raise ValueError(f"unexpected LoRA rank/alpha in {path}")
    if float(payload.get("lora_dropout")) != 0.05:
        raise ValueError(f"unexpected LoRA dropout in {path}")


def prepare_effective_manifest(
    source: Path,
    destination: Path,
    *,
    pipeline: str,
    tokenizer: Any,
    expected_rows: int,
) -> dict[str, Any]:
    rows = read_jsonl(source)
    ids = [str(row.get("record_id") or row.get("id") or "") for row in rows]
    if len(rows) != expected_rows or len(ids) != len(set(ids)):
        raise ValueError(
            f"reproduction source must contain {expected_rows} unique rows"
        )
    effective: list[dict[str, Any]] = []
    appended = 0
    pretokenized = 0
    for original in rows:
        row = dict(original)
        completion = str(row.get("completion") or row.get("proof") or "")
        if pipeline == "historical_supervised_eos":
            if not completion.endswith(tokenizer.eos_token):
                completion += tokenizer.eos_token
                appended += 1
        elif pipeline == "current_missing_eos":
            # The frozen A0 diagnostics prove that its prompt-completion rows
            # were tokenized without a supervised EOS.  Current TRL now adds
            # EOS automatically, so feed deterministic trainer-ready columns
            # to reconstruct the recorded A0 behavior without changing the
            # production trainer or installed package versions.
            prompt = str(row.get("prompt") or "")
            prompt_ids = tokenizer(text=prompt)["input_ids"]
            full_ids = tokenizer(text=prompt + completion)["input_ids"]
            if full_ids[: len(prompt_ids)] != prompt_ids:
                raise ValueError("prompt is not a token prefix of prompt+completion")
            if full_ids and full_ids[-1] == tokenizer.eos_token_id:
                raise ValueError("current-pipeline reconstruction gained EOS")
            if len(full_ids) > 1024:
                raise ValueError("unexpected truncation in current pipeline")
            row["input_ids"] = full_ids
            row["attention_mask"] = [1] * len(full_ids)
            row["labels"] = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
            pretokenized += 1
        else:
            raise ValueError(f"unknown pipeline: {pipeline}")
        row["completion"] = completion
        effective.append(row)
    write_jsonl(destination, effective)
    return {
        "source_path": str(source),
        "source_sha256": sha256(source),
        "effective_path": str(destination),
        "effective_sha256": sha256(destination),
        "rows": len(effective),
        "unique_ids": len(set(ids)),
        "eos_appended_rows": appended,
        "pretokenized_rows": pretokenized,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", choices=tuple(EXPERIMENTS), required=True)
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("outputs/clean_m0_a0_reproduction_audit"),
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
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = args.project_root.resolve()
    output_root = args.output_root.resolve()
    experiment = EXPERIMENTS[args.experiment]
    experiment_root = output_root / "reproductions" / args.experiment
    if experiment_root.exists():
        raise FileExistsError(f"refusing to overwrite {experiment_root}")
    trainer_dir = experiment_root / "trainer"
    checkpoint_dir = experiment_root / "checkpoint"
    input_dir = experiment_root / "input"
    input_dir.mkdir(parents=True)

    model = (root / args.model).resolve() if not args.model.is_absolute() else args.model
    eval_path = (root / args.eval).resolve() if not args.eval.is_absolute() else args.eval
    source = (root / experiment["source"]).resolve()
    expected_base_hash = (
        "dd924a11b4c220f385b51ffa522daea7c9f3d850e31b162bb5661df483c6d3ee"
    )
    if sha256(model / "model.safetensors") != expected_base_hash:
        raise ValueError("Base model hash changed")
    tokenizer_probe = AutoTokenizer.from_pretrained(
        model,
        trust_remote_code=True,
    )
    if tokenizer_probe.eos_token != "<|im_end|>" or tokenizer_probe.eos_token_id != 151645:
        raise ValueError("unexpected Qwen EOS contract")
    effective_manifest = input_dir / "effective_train.jsonl"
    manifest = prepare_effective_manifest(
        source,
        effective_manifest,
        pipeline=experiment["pipeline"],
        tokenizer=tokenizer_probe,
        expected_rows=3000,
    )
    if experiment["pipeline"] == "current_missing_eos":
        effective_eval = input_dir / "effective_eval.jsonl"
        eval_manifest = prepare_effective_manifest(
            eval_path,
            effective_eval,
            pipeline=experiment["pipeline"],
            tokenizer=tokenizer_probe,
            expected_rows=160,
        )
    else:
        effective_eval = eval_path
        eval_manifest = {
            "source_path": str(eval_path),
            "source_sha256": sha256(eval_path),
            "effective_path": str(eval_path),
            "effective_sha256": sha256(eval_path),
            "rows": 160,
            "eos_appended_by_trl": True,
        }
    del tokenizer_probe

    identity = {
        "experiment": args.experiment,
        "purpose": experiment["purpose"],
        "pipeline": experiment["pipeline"],
        "sampling_strategy": experiment["sampling_strategy"],
        "seed": args.seed,
        "data_seed": args.seed,
        "base_path": str(model),
        "base_sha256": expected_base_hash,
        "eval_path": str(eval_path),
        "eval_sha256": sha256(eval_path),
        "effective_eval": eval_manifest,
        "manifest": manifest,
        "initialization": "independent_base_new_adapter",
        "optimizer_state_shared": False,
        "adapter_state_shared": False,
    }
    write_json(experiment_root / "experiment_identity.json", identity)

    config = SFTTrainConfig(
        model_name_or_path=str(model),
        train_file=str(effective_manifest),
        validation_file=str(effective_eval),
        output_dir=str(trainer_dir),
        sample_weight_field=None,
        sampling_strategy=experiment["sampling_strategy"],
        requested_packing=True,
        require_pantograph_verified=True,
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
        seed=args.seed,
        data_seed=args.seed,
    )
    write_json(experiment_root / "resolved_request.json", config.__dict__)

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer, tokenizer = build_trainer(config)
    if experiment["sampling_strategy"] == "fixed_manifest_without_replacement":
        if not isinstance(trainer, FixedManifestSFTTrainer):
            raise TypeError("current pipeline did not use the fixed sampler")
    else:
        if isinstance(trainer, FixedManifestSFTTrainer):
            raise TypeError("historical pipeline unexpectedly used the fixed sampler")
    diagnostics = json.loads(
        (trainer_dir / "sft_tokenization_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    expect_eos = experiment["pipeline"] == "historical_supervised_eos"
    sample_eos = [
        bool(row.get("ends_with_eos")) for row in diagnostics.get("samples") or []
    ]
    if not sample_eos or any(value != expect_eos for value in sample_eos):
        raise RuntimeError("effective EOS label contract differs from experiment")
    if diagnostics["truncated_examples"] or diagnostics["num_examples"] != 3000:
        raise RuntimeError("tokenization truncated or changed the row count")

    trainer.save_model(str(checkpoint_dir / "step_0"))
    tokenizer.save_pretrained(str(checkpoint_dir / "step_0"))
    validate_adapter(checkpoint_dir / "step_0")
    result = trainer.train()
    if trainer.state.global_step != math.ceil(3000 / 16):
        raise RuntimeError("unexpected optimizer step count")
    best_source = Path(str(trainer.state.best_model_checkpoint or ""))
    if not best_source.is_dir():
        raise FileNotFoundError("best checkpoint is missing")
    shutil.copytree(best_source, checkpoint_dir / "best")
    trainer.save_model(str(checkpoint_dir / "final"))
    tokenizer.save_pretrained(str(checkpoint_dir / "final"))
    trainer.save_state()
    validate_adapter(checkpoint_dir / "best")
    validate_adapter(checkpoint_dir / "final")

    eval_rows = [
        row for row in trainer.state.log_history if "eval_loss" in row
    ]
    final_eval = eval_rows[-1] if eval_rows else {}
    summary = {
        **identity,
        "optimizer_steps": trainer.state.global_step,
        "physical_rows": 3000,
        "draws": 3000,
        "unique_draws": 3000,
        "duplicate_draws": 0,
        "max_repeat": 1,
        "effective_batch_size": 16,
        "supervised_label_tokens": diagnostics[
            "non_ignored_label_tokens_total"
        ],
        "sample_ends_with_eos": sample_eos,
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
    write_json(experiment_root / "training_summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
