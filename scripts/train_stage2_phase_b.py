"""Train the single approved S2-Hard-Replay Phase B adapter."""

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
from scripts.prepare_stage2_sft_phase0 import record_id, sha256, theorem_group
from scripts.train_anchor_ratio_eos_fixed_arm import (
    directory_tensor_hash,
    validate_adapter,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    parser.add_argument(
        "--phase-root",
        type=Path,
        default=Path("outputs/stage2_data_ablation/phaseB_hard_replay"),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    root = args.phase_root if args.phase_root.is_absolute() else project / args.phase_root
    contract_path = root / "training_contract.json"
    gate_path = root / "audit/phaseB_manifest_gate.json"
    contract = read_json(contract_path)
    gate = read_json(gate_path)
    config = SFTTrainConfig(**contract["resolved_config"])
    manifest = Path(config.train_file)
    model = Path(config.model_name_or_path)
    trainer_dir = Path(config.output_dir)
    checkpoint_root = root / "checkpoints/S2-Hard-Replay"
    identity_path = root / "training/training_identity.json"
    summary_path = root / "training/training_summary.json"

    forbidden_existing = [
        path
        for path in (trainer_dir, checkpoint_root, identity_path, summary_path)
        if path.exists()
    ]
    if forbidden_existing:
        raise FileExistsError(
            f"refusing continuation/overwrite of Phase B training: {forbidden_existing}"
        )
    if not gate.get("passed") or gate.get("rows") != 800:
        raise RuntimeError("Phase B manifest gate did not pass")
    if sha256(manifest) != gate["effective_manifest_sha256"]:
        raise RuntimeError("effective training manifest hash drifted")
    if sha256(model / "model.safetensors") != gate["m0_model_sha256"]:
        raise RuntimeError("frozen M0 model hash drifted")
    if config.adapter_path is not None:
        raise RuntimeError("Phase B must initialize a fresh adapter")
    fixed = {
        "learning_rate": 1e-5,
        "num_train_epochs": 1.0,
        "lora_r": 32,
        "lora_alpha": 64,
        "require_supervised_eos": True,
        "requested_packing": False,
        "sampling_strategy": "fixed_manifest_without_replacement",
        "gradient_accumulation_steps": 16,
    }
    drift = {
        key: {"expected": value, "actual": getattr(config, key)}
        for key, value in fixed.items()
        if getattr(config, key) != value
    }
    if drift:
        raise RuntimeError(f"Phase B fixed training contract drift: {drift}")

    rows = read_jsonl(manifest)
    ids = [record_id(row) for row in rows]
    groups = [theorem_group(row) for row in rows]
    if (
        len(rows) != 800
        or len(set(ids)) != 800
        or len(set(groups)) != 800
        or any(not value for value in ids + groups)
    ):
        raise RuntimeError("Phase B effective manifest violates unique identity")
    bucket_counts = Counter(row["stage2_phase_b"]["bucket"] for row in rows)
    if dict(bucket_counts) != {
        "Hard-A": 399,
        "Hard-B": 349,
        "Frontier": 40,
        "Stable/Core": 12,
    }:
        raise RuntimeError(f"Phase B composition drifted: {bucket_counts}")

    identity = {
        "phase": "B",
        "model": "S2-Hard-Replay",
        "starting_checkpoint": str(model),
        "starting_checkpoint_model_sha256": gate["m0_model_sha256"],
        "manifest_sha256": sha256(manifest),
        "manifest_gate_sha256": sha256(gate_path),
        "training_contract_sha256": sha256(contract_path),
        "rows": len(rows),
        "bucket_counts": dict(bucket_counts),
        "initialization": "fresh_lora_on_frozen_merged_m0",
        "adapter_path": None,
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
    diagnostics = read_json(trainer_dir / "sft_tokenization_diagnostics.json")
    actual_eos = eos_contract["effective_trainer_labels"]
    if (
        not actual_eos.get("passed")
        or actual_eos.get("total_records") != 800
        or actual_eos.get("records_with_supervised_eos") != 800
        or actual_eos.get("records_whose_last_valid_label_is_eos") != 800
        or actual_eos.get("zero_label_records") != 0
        or actual_eos.get("records_with_semantic_truncation") != 0
        or diagnostics.get("num_examples") != 800
        or diagnostics.get("truncated_examples") != 0
    ):
        raise RuntimeError("trainer-effective Phase B EOS/token gate failed")

    step_zero = checkpoint_root / "step_0"
    trainer.save_model(str(step_zero))
    tokenizer.save_pretrained(str(step_zero))
    validate_adapter(step_zero)
    step_zero_hash = directory_tensor_hash(step_zero)

    result = trainer.train()
    expected_steps = int(contract["expected_optimizer_steps"])
    if expected_steps != math.ceil(800 / 16) or trainer.state.global_step != expected_steps:
        raise RuntimeError(
            f"unexpected Phase B optimizer steps: {trainer.state.global_step}/{expected_steps}"
        )
    sampler = trainer.fixed_manifest_sampler
    if sampler is None or sampler.iteration_count != 1:
        raise RuntimeError("Phase B sampler trace missing or iterated more than once")
    indices = list(sampler.drawn_indices)
    draw_counts = Counter(ids[index] for index in indices)
    if (
        len(indices) != 800
        or len(draw_counts) != 800
        or max(draw_counts.values(), default=0) != 1
    ):
        raise RuntimeError("Phase B training was not exactly once without replacement")

    best_source = Path(str(trainer.state.best_model_checkpoint or ""))
    if not best_source.is_dir():
        raise FileNotFoundError("Phase B best eval-loss checkpoint is missing")
    shutil.copytree(best_source, checkpoint_root / "best")
    trainer.save_model(str(checkpoint_root / "final"))
    tokenizer.save_pretrained(str(checkpoint_root / "final"))
    trainer.save_state()
    validate_adapter(checkpoint_root / "best")
    validate_adapter(checkpoint_root / "final")

    trace_path = root / "training/sampling_trace.jsonl"
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
                        "bucket": row["stage2_phase_b"]["bucket"],
                        "source": row.get("sampling_source"),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
    final_base_hash = sha256(model / "model.safetensors")
    if final_base_hash != gate["m0_model_sha256"]:
        raise RuntimeError("training modified the frozen M0 checkpoint")
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
        "eval_token_accuracy": final_eval.get(
            "eval_mean_token_accuracy", final_eval.get("eval_token_accuracy")
        ),
        "metrics": dict(getattr(result, "metrics", {}) or {}),
        "best_checkpoint": str(checkpoint_root / "best"),
        "final_checkpoint": str(checkpoint_root / "final"),
        "runtime_seconds": time.monotonic() - started,
        "gpu_peak_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "gpu_peak_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "process_max_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        "frozen_m0_hash_after_training": final_base_hash,
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
