"""Train one B1/B2 QLoRA arm from the fixed clean M0 checkpoint."""

from __future__ import annotations

import argparse
import json
import resource
import time
from pathlib import Path

import torch

from lean_prover.lean_training.expert_iteration.utils import (
    file_sha256,
    write_json_atomic,
)
from lean_prover.lean_training.sft_pipeline.config import SFTTrainConfig
from lean_prover.lean_training.sft_pipeline.trainer import build_trainer


EXPECTED_TARGET_MODULES = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}
CHECKPOINT_STEPS = (25, 50, 75, 100)


def validate_adapter(path: Path) -> None:
    config_path = path / "adapter_config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"missing adapter config: {config_path}")
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    actual = set(payload.get("target_modules") or [])
    if actual != EXPECTED_TARGET_MODULES:
        raise ValueError(f"unexpected target modules in {config_path}: {sorted(actual)}")
    if payload.get("r") != 32 or payload.get("lora_alpha") != 64:
        raise ValueError(f"unexpected LoRA rank/alpha in {config_path}")
    if float(payload.get("lora_dropout")) != 0.05:
        raise ValueError(f"unexpected LoRA dropout in {config_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--arm", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--eval", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train_path = Path(args.train)
    row_count = sum(
        1 for line in train_path.read_text(encoding="utf-8").splitlines() if line.strip()
    )
    if row_count != 1600:
        raise ValueError(f"expected exactly 1600 training draws, got {row_count}")

    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    config = SFTTrainConfig(
        model_name_or_path=args.model,
        adapter_path=None,
        train_file=args.train,
        validation_file=args.eval,
        output_dir=str(output),
        sample_weight_field=None,
        requested_packing=False,
        require_pantograph_verified=True,
        max_seq_length=1024,
        per_device_train_batch_size=1,
        per_device_validation_batch_size=1,
        gradient_accumulation_steps=16,
        learning_rate=1e-5,
        num_train_epochs=1,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="steps",
        save_steps=25,
        save_total_limit=None,
        load_best_model_at_end=False,
        warmup_ratio=0.03,
        weight_decay=0.01,
        lr_scheduler_type="linear",
        max_grad_norm=1.0,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        seed=args.seed,
        data_seed=args.seed,
    )

    torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    trainer, tokenizer = build_trainer(config)
    trainable = sum(
        parameter.numel()
        for parameter in trainer.model.parameters()
        if parameter.requires_grad
    )
    result = trainer.train()
    if trainer.state.global_step != 100:
        raise RuntimeError(
            f"optimizer step contract violated: {trainer.state.global_step} != 100"
        )
    trainer.save_model(str(output))
    tokenizer.save_pretrained(str(output))
    trainer.save_state()

    saved_steps: list[str] = []
    for step in CHECKPOINT_STEPS:
        source = output / f"checkpoint-{step}"
        target = output / f"step_{step}"
        if not source.exists():
            raise FileNotFoundError(f"required checkpoint was not saved: {source}")
        if target.exists():
            raise FileExistsError(f"refusing to overwrite existing checkpoint: {target}")
        source.rename(target)
        validate_adapter(target)
        saved_steps.append(str(target))
    validate_adapter(output)

    history = list(trainer.state.log_history)
    eval_rows = [row for row in history if "eval_loss" in row]
    metrics = {
        "arm": args.arm,
        "initialization_checkpoint": args.model,
        "adapter_initialization": None,
        "train_dataset": args.train,
        "train_dataset_sha256": file_sha256(args.train),
        "eval_dataset": args.eval,
        "eval_dataset_sha256": file_sha256(args.eval),
        "optimizer_steps": trainer.state.global_step,
        "effective_batch_size": 16,
        "trainable_parameter_count": trainable,
        "train_loss": getattr(result, "training_loss", None),
        "eval_loss": eval_rows[-1].get("eval_loss") if eval_rows else None,
        "eval_token_accuracy": (
            eval_rows[-1].get(
                "eval_mean_token_accuracy", eval_rows[-1].get("eval_token_accuracy")
            )
            if eval_rows
            else None
        ),
        "saved_checkpoints": saved_steps,
        "wall_seconds": round(time.monotonic() - started, 4),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "process_peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "training_config": config.__dict__,
        "target_modules": sorted(EXPECTED_TARGET_MODULES),
    }
    write_json_atomic(output / "b_ablation_training_metrics.json", metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
