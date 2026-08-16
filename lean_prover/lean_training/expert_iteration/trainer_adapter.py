"""Adapter that reuses the existing SFT/QLoRA trainer for each iteration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

from lean_prover.lean_training.sft_pipeline.config import SFTTrainConfig
from lean_prover.lean_training.sft_pipeline.trainer import build_trainer

from .config import ExpertIterationConfig
from .utils import file_sha256, write_json_atomic, write_jsonl_atomic


class IterationTrainer(Protocol):
    def train_iteration(
        self,
        *,
        iteration: int,
        train_path: str,
        eval_path: str,
        output_dir: str,
        initialization_checkpoint: str,
        previous_adapter: str | None,
        expected_eval_hash: str,
    ) -> dict[str, Any]: ...


class SFTTrainerAdapter:
    def __init__(self, config: ExpertIterationConfig) -> None:
        self.config = config

    def train_iteration(
        self,
        *,
        iteration: int,
        train_path: str,
        eval_path: str,
        output_dir: str,
        initialization_checkpoint: str,
        previous_adapter: str | None,
        expected_eval_hash: str,
    ) -> dict[str, Any]:
        """Train one weighted SFT round while enforcing the fixed eval hash."""

        eval_hash = file_sha256(eval_path)
        if eval_hash != expected_eval_hash:
            raise ValueError(
                f"eval dataset changed at iteration={iteration}: "
                f"expected={expected_eval_hash}, actual={eval_hash}"
            )
        training = self.config.training
        adapter_path = (
            previous_adapter
            if self.config.initial_model.checkpoint_strategy == "continue_adapter"
            else None
        )
        train_config = SFTTrainConfig(
            model_name_or_path=initialization_checkpoint,
            adapter_path=adapter_path,
            train_file=train_path,
            validation_file=eval_path,
            output_dir=output_dir,
            sample_weight_field="sample_weight",
            requested_packing=training.packing,
            max_seq_length=training.max_seq_length,
            per_device_train_batch_size=training.per_device_train_batch_size,
            per_device_validation_batch_size=training.per_device_eval_batch_size,
            gradient_accumulation_steps=training.gradient_accumulation_steps,
            learning_rate=(
                training.initial_learning_rate
                if iteration < 0 and training.initial_learning_rate is not None
                else training.learning_rate
            ),
            num_train_epochs=training.num_train_epochs,
            logging_steps=training.logging_steps,
            eval_strategy=training.eval_strategy,
            eval_steps=training.eval_steps,
            save_strategy=training.save_strategy,
            save_steps=training.save_steps,
            load_best_model_at_end=training.load_best_model_at_end,
            metric_for_best_model=training.metric_for_best_model,
            greater_is_better=training.greater_is_better,
            warmup_ratio=training.warmup_ratio,
            weight_decay=training.weight_decay,
            max_grad_norm=training.max_grad_norm,
            gradient_checkpointing=training.gradient_checkpointing,
            optim=training.optimizer,
            lora_r=training.lora_rank,
            lora_alpha=training.lora_alpha,
            lora_dropout=training.lora_dropout,
            # Keep preprocessing, sampler seeding, and eval ordering identical
            # across rounds so eval_loss remains directly comparable.
            seed=self.config.seed,
            data_seed=self.config.seed,
        )
        trainer, tokenizer = build_trainer(train_config)
        trainable_parameter_count = sum(
            parameter.numel()
            for parameter in trainer.model.parameters()
            if parameter.requires_grad
        )
        checkpoint_root = Path(output_dir)
        checkpoint_candidates = sorted(
            (
                path
                for path in checkpoint_root.glob("checkpoint-*")
                if path.name.rsplit("-", 1)[-1].isdigit()
            ),
            key=lambda path: int(path.name.rsplit("-", 1)[-1]),
        )
        resume_checkpoint = str(checkpoint_candidates[-1]) if checkpoint_candidates else None
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        trainer.save_state()
        history = list(trainer.state.log_history)
        eval_rows = [row for row in history if "eval_loss" in row]
        train_rows = [
            row for row in history if "loss" in row and "eval_loss" not in row
        ]
        best_row = min(eval_rows, key=lambda row: row["eval_loss"]) if eval_rows else None
        final_row = eval_rows[-1] if eval_rows else None
        metrics = {
            "iteration": iteration,
            "train_initialization_checkpoint": initialization_checkpoint,
            "adapter_initialization": adapter_path,
            "output_checkpoint": output_dir,
            "eval_dataset_hash": eval_hash,
            "train_loss": getattr(train_result, "training_loss", None),
            "initial_train_loss": train_rows[0].get("loss") if train_rows else None,
            "final_train_loss": train_rows[-1].get("loss") if train_rows else None,
            "best_eval_loss": best_row.get("eval_loss") if best_row else None,
            "best_eval_step": best_row.get("step") if best_row else None,
            "final_eval_loss": final_row.get("eval_loss") if final_row else None,
            "best_eval_token_accuracy": (
                (
                    best_row.get("eval_mean_token_accuracy")
                    if best_row.get("eval_mean_token_accuracy") is not None
                    else best_row.get("eval_token_accuracy")
                )
                if best_row
                else None
            ),
            "final_eval_token_accuracy": (
                (
                    final_row.get("eval_mean_token_accuracy")
                    if final_row.get("eval_mean_token_accuracy") is not None
                    else final_row.get("eval_token_accuracy")
                )
                if final_row
                else None
            ),
            "training_skipped": False,
            "resumed_from_checkpoint": resume_checkpoint,
            "optimizer_steps": trainer.state.global_step,
            "effective_batch_size": (
                training.per_device_train_batch_size
                * training.gradient_accumulation_steps
            ),
            "trainable_parameter_count": trainable_parameter_count,
            "requested_packing": training.packing,
            "effective_packing": False,
            "packing_disabled_reason": (
                "source-weighted sampler requires unpacked row boundaries"
                if training.packing
                else None
            ),
        }
        eval_dir = Path(output_dir).parent / "eval"
        write_json_atomic(eval_dir / "eval_metrics.json", metrics)
        write_jsonl_atomic(eval_dir / "eval_history.jsonl", eval_rows)
        return metrics


def prepare_fixed_anchor(
    config: ExpertIterationConfig,
    *,
    sft_adapter: str | None = None,
) -> str:
    """Merge the initial SFT adapter in BF16/FP16 and cache the anchor model."""

    model_config = config.initial_model
    if not model_config.merged_sft0_path:
        raise ValueError("checkpoint_strategy=fixed_anchor requires merged_sft0_path")
    destination = Path(model_config.merged_sft0_path).expanduser()
    if (destination / "config.json").exists():
        return str(destination)
    adapter = sft_adapter or model_config.sft_adapter
    if not adapter:
        raise ValueError("creating fixed anchor requires initial_model.sft_adapter")
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = AutoModelForCausalLM.from_pretrained(
        model_config.base_model,
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
    )
    merged = PeftModel.from_pretrained(base, adapter).merge_and_unload()
    destination.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(destination, safe_serialization=True)
    tokenizer_name = model_config.tokenizer or model_config.base_model
    AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True).save_pretrained(destination)
    return str(destination)
