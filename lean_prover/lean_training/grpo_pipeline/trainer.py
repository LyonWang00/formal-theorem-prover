"""GRPO training entry point for prepared Lean proof prompts.

This script is training-only. Dataset downloading, sampling, schema adaptation,
and proof filtering live in ``prepare_datasets.py``. The expected input here is
a JSON/JSONL file whose rows contain proof-free ``prompt`` records plus the
``lean_statement`` and preamble fields needed to score generated proofs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset

from lean_prover.lean_training.data.training import load_prepared_dataset
from lean_prover.lean_training.grpo_pipeline.config import GRPOTrainConfig
from lean_prover.lean_training.grpo_pipeline.rewards import PantographRewardFunction
from lean_prover.lean_training.modeling.lora import build_grpo_lora_config
from lean_prover.lean_training.modeling.quantization import build_qlora_model, qlora_compute_dtype
from lean_prover.lean_training.modeling.runtime import package_version, set_reproducible_seeds
from lean_prover.lean_training.modeling.tokenizer import load_tokenizer

def parse_args() -> GRPOTrainConfig:
    """Parse GRPO command-line options into an immutable configuration.

    Returns:
        The complete ``GRPOTrainConfig`` used to construct training components.
    """
    parser = argparse.ArgumentParser(
        description="Run GRPO on prepared proof-free Lean prompt records."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--statement_field", default="lean_statement")
    parser.add_argument("--id_field", default="id")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_validation_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--eval_strategy", default="steps")
    parser.add_argument("--save_strategy", default="steps")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument(
        "--gradient_checkpointing",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--device_map", default="auto")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lean_project_path",
        default=str(Path(__file__).resolve().parents[3] / "lean_project"),
    )
    parser.add_argument("--pantograph_imports", default="Mathlib")
    parser.add_argument("--lean_timeout", type=int, default=120)
    parser.add_argument("--compile_success_reward", type=float, default=1.0)
    parser.add_argument("--format_reward", type=float, default=0.1)
    parser.add_argument("--brevity_reward", type=float, default=0.1)
    parser.add_argument("--failure_reward", type=float, default=0.0)
    parser.add_argument("--reward_log_file", default=None)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--data_seed", type=int, default=20260715)
    args = parser.parse_args()
    return GRPOTrainConfig(**vars(args))


def ensure_grpo_fields(dataset: Dataset, config: GRPOTrainConfig, *, name: str) -> None:
    """Check that prompts, target statements, and IDs are available for rewards.

    Args:
        dataset: Prepared GRPO dataset.
        config: Configuration naming the required columns.
        name: Dataset label used in errors.

    Raises:
        ValueError: If any required column is missing.
    """
    required = [config.prompt_field, config.statement_field, config.id_field]
    if config.brevity_reward:
        required.append("reference_proof_length_tokens")
    missing = [field for field in required if field not in dataset.column_names]
    if missing:
        raise ValueError(
            f"{name} dataset is missing GRPO fields {missing}; "
            f"columns={dataset.column_names}"
        )
    leaked_targets = [
        field
        for field in ("proof", "completion", "text", "reference_proof")
        if field in dataset.column_names
    ]
    if leaked_targets:
        raise ValueError(
            f"{name} dataset exposes target proof fields {leaked_targets}; "
            "regenerate it with build_grpo_training_record"
        )


def validate_dataset_prompts(
    dataset: Dataset,
    tokenizer,
    *,
    config: GRPOTrainConfig,
    name: str,
) -> None:
    """Ensure every generation prompt fits within the configured token limit.

    Args:
        dataset: Prepared GRPO dataset.
        tokenizer: Tokenizer used for length measurement.
        config: Configuration containing prompt field and maximum length.
        name: Dataset label used in diagnostics.

    Raises:
        ValueError: If any prompt exceeds ``max_prompt_length``.
    """
    overlength = 0
    for index, row in enumerate(dataset):
        prompt = str(row[config.prompt_field])
        token_count = len(tokenizer(prompt, add_special_tokens=True)["input_ids"])
        if token_count > config.max_prompt_length:
            overlength += 1
            raise ValueError(
                f"{name} dataset row {index} prompt has {token_count} tokens, "
                f"exceeding max_prompt_length={config.max_prompt_length}. "
                "Run prepare_datasets.py with --filter_overlength or raise "
                "--max_prompt_length."
            )
    if overlength:
        print(f"WARNING: {name} dataset contains {overlength} overlength prompts")


def validate_trl_grpo_support():
    try:
        from trl import GRPOConfig, GRPOTrainer
    except ImportError as error:
        raise RuntimeError(
            "Installed TRL does not expose GRPOConfig/GRPOTrainer. "
            f"Detected trl={package_version('trl')}."
        ) from error
    return GRPOConfig, GRPOTrainer


def supported_config_kwargs(config_cls: Any, kwargs: dict[str, Any]) -> dict[str, Any]:
    fields = getattr(config_cls, "__dataclass_fields__", {})
    if fields:
        return {key: value for key, value in kwargs.items() if key in fields}
    return kwargs


def print_startup_report(
    config: GRPOTrainConfig,
    *,
    train_dataset: Dataset,
    validation_dataset: Dataset | None,
) -> None:
    report = {
        "trl_version": package_version("trl"),
        "transformers_version": package_version("transformers"),
        "peft_version": package_version("peft"),
        "torch_version": torch.__version__,
        "train_columns": list(train_dataset.column_names),
        "validation_columns": (
            list(validation_dataset.column_names)
            if validation_dataset is not None
            else None
        ),
        "train_samples": len(train_dataset),
        "validation_samples": len(validation_dataset) if validation_dataset else 0,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": config.max_completion_length,
        "num_generations": config.num_generations,
        "reward_backend": "pantograph",
    }
    print("GRPO_STARTUP " + json.dumps(report, ensure_ascii=False))


def save_training_config(config: GRPOTrainConfig) -> None:
    """Persist GRPO settings and dependency versions in the output directory.

    Args:
        config: Configuration to serialize.

    Output:
        Writes ``grpo_training_config.json`` under ``config.output_dir``.
    """
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": asdict(config),
        "versions": {
            "trl": package_version("trl"),
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "torch": torch.__version__,
        },
    }
    (output_dir / "grpo_training_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_trainer(config: GRPOTrainConfig):
    """Assemble datasets, QLoRA model, Pantograph reward, and GRPO trainer.

    Args:
        config: Complete training, generation, adapter, and reward settings.

    Returns:
        The configured trainer, tokenizer, and reward-function instance.

    Raises:
        ValueError: If dataset fields or prompt lengths are invalid.
        RuntimeError: If the installed TRL lacks GRPO support or CUDA is unavailable.
    """
    GRPOConfig, GRPOTrainer = validate_trl_grpo_support()
    set_reproducible_seeds(config)
    train_dataset = load_prepared_dataset(config.train_file)
    ensure_grpo_fields(train_dataset, config, name="train")
    validation_dataset = None
    if config.validation_file:
        validation_dataset = load_prepared_dataset(config.validation_file)
        ensure_grpo_fields(validation_dataset, config, name="validation")

    tokenizer = load_tokenizer(config.model_name_or_path, padding_side="left")
    validate_dataset_prompts(train_dataset, tokenizer, config=config, name="train")
    if validation_dataset is not None:
        validate_dataset_prompts(
            validation_dataset,
            tokenizer,
            config=config,
            name="validation",
        )
    print_startup_report(
        config,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
    )
    compute_dtype = qlora_compute_dtype("GRPO QLoRA")
    model = build_qlora_model(
        config.model_name_or_path,
        compute_dtype=compute_dtype,
        device_map=config.device_map,
    )
    has_validation_dataset = validation_dataset is not None
    eval_strategy = config.eval_strategy if has_validation_dataset else "no"
    training_args = GRPOConfig(
        **supported_config_kwargs(
            GRPOConfig,
            {
                "output_dir": config.output_dir,
                "max_prompt_length": config.max_prompt_length,
                "max_completion_length": config.max_completion_length,
                "num_generations": config.num_generations,
                "per_device_train_batch_size": config.per_device_train_batch_size,
                "per_device_eval_batch_size": config.per_device_validation_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "learning_rate": config.learning_rate,
                "num_train_epochs": config.num_train_epochs,
                "logging_steps": config.logging_steps,
                "eval_strategy": eval_strategy,
                "eval_steps": config.eval_steps,
                "save_strategy": config.save_strategy,
                "save_steps": config.save_steps,
                "save_total_limit": config.save_total_limit,
                "warmup_ratio": config.warmup_ratio,
                "weight_decay": config.weight_decay,
                "lr_scheduler_type": config.lr_scheduler_type,
                "max_grad_norm": config.max_grad_norm,
                "gradient_checkpointing": config.gradient_checkpointing,
                "bf16": compute_dtype is torch.bfloat16,
                "fp16": compute_dtype is torch.float16,
                "report_to": "none",
                "seed": config.seed,
                "data_seed": config.data_seed,
            },
        )
    )
    reward_func = PantographRewardFunction(config)
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=build_grpo_lora_config(
            r=config.lora_r,
            alpha=config.lora_alpha,
            dropout=config.lora_dropout,
        ),
    )
    return trainer, tokenizer, reward_func


def main() -> None:
    """Run GRPO training and save adapter, tokenizer, state, and configuration."""
    config = parse_args()
    save_training_config(config)
    trainer, tokenizer, reward_func = build_trainer(config)
    try:
        trainer.train()
        trainer.save_model(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        trainer.save_state()
        save_training_config(config)
    finally:
        reward_func.close()


if __name__ == "__main__":
    main()
