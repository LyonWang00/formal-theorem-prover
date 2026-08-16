"""GRPO training entry point for prepared Lean proof prompts.

This script is training-only. Dataset downloading, sampling, schema adaptation,
and proof filtering live in ``prepare_datasets.py``. The expected input here is
a JSON/JSONL file whose rows contain proof-free ``prompt`` records plus the
``lean_statement`` and preamble fields needed to score generated proofs.
"""

from __future__ import annotations

import argparse
import atexit
import json
import random
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import Dataset, load_dataset
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lean_prover.lean_training.prepare_datasets import (
    build_lean_source_with_preamble,
    compose_lean_theorem,
    contains_forbidden_proof_token,
)
from lean_prover.lean_training.pantograph_verifier import PantographTheoremVerifier


@dataclass(frozen=True)
class GRPOTrainConfig:
    """Complete GRPO, QLoRA, Pantograph reward, and output configuration."""

    model_name_or_path: str
    train_file: str
    output_dir: str
    validation_file: str | None = None
    prompt_field: str = "prompt"
    statement_field: str = "lean_statement"
    id_field: str = "id"
    max_prompt_length: int = 1024
    max_completion_length: int = 256
    num_generations: int = 4
    per_device_train_batch_size: int = 1
    per_device_validation_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-5
    num_train_epochs: float = 1.0
    logging_steps: int = 10
    eval_steps: int | None = None
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    save_steps: int = 200
    save_total_limit: int | None = None
    warmup_ratio: float = 0.03
    weight_decay: float = 0.0
    lr_scheduler_type: str = "linear"
    max_grad_norm: float = 1.0
    gradient_checkpointing: bool = True
    device_map: str = "auto"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lean_project_path: str = "lean_project"
    pantograph_imports: str = "Mathlib"
    lean_timeout: int = 120
    compile_success_reward: float = 1.0
    format_reward: float = 0.1
    failure_reward: float = 0.0
    reward_log_file: str | None = None
    seed: int = 20260715
    data_seed: int = 20260715


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
        default=str(Path(__file__).resolve().parents[2] / "lean_project"),
    )
    parser.add_argument("--pantograph_imports", default="Mathlib")
    parser.add_argument("--lean_timeout", type=int, default=120)
    parser.add_argument("--compile_success_reward", type=float, default=1.0)
    parser.add_argument("--format_reward", type=float, default=0.1)
    parser.add_argument("--failure_reward", type=float, default=0.0)
    parser.add_argument("--reward_log_file", default=None)
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--data_seed", type=int, default=20260715)
    args = parser.parse_args()
    return GRPOTrainConfig(**vars(args))


def load_prepared_dataset(path: str) -> Dataset:
    """Load a prepared proof-prompt JSON/JSONL file.

    Args:
        path: Local prepared dataset path.

    Returns:
        The file contents as a Hugging Face ``Dataset``.

    Raises:
        FileNotFoundError: If the path does not exist.
        ValueError: If the file extension is unsupported.
    """
    dataset_path = Path(path).expanduser()
    if not dataset_path.exists():
        raise FileNotFoundError(f"dataset file not found: {dataset_path}")
    if dataset_path.suffix.lower() not in {".json", ".jsonl"}:
        raise ValueError(f"expected a .json or .jsonl file, got {dataset_path}")
    return load_dataset("json", data_files=str(dataset_path), split="train")


def ensure_grpo_fields(dataset: Dataset, config: GRPOTrainConfig, *, name: str) -> None:
    """Check that prompts, target statements, and IDs are available for rewards.

    Args:
        dataset: Prepared GRPO dataset.
        config: Configuration naming the required columns.
        name: Dataset label used in errors.

    Raises:
        ValueError: If any required column is missing.
    """
    required = (config.prompt_field, config.statement_field, config.id_field)
    missing = [field for field in required if field not in dataset.column_names]
    if missing:
        raise ValueError(
            f"{name} dataset is missing GRPO fields {missing}; "
            f"columns={dataset.column_names}"
        )


def build_tokenizer(model_name_or_path: str):
    """Load the tokenizer and configure left padding for batched generation.

    Args:
        model_name_or_path: Hugging Face model ID or local model path.

    Returns:
        The configured tokenizer, with EOS reused as padding when necessary.
    """
    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    return tokenizer


def qlora_compute_dtype() -> torch.dtype:
    """Select the CUDA compute dtype for 4-bit GRPO QLoRA.

    Returns:
        BF16 when supported, otherwise FP16.

    Raises:
        RuntimeError: If CUDA is unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("GRPO QLoRA requires CUDA; no CUDA device is available.")
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def build_model(model_name_or_path: str, *, compute_dtype: torch.dtype, device_map: str):
    """Load and prepare a 4-bit NF4 causal language model for adapter training.

    Args:
        model_name_or_path: Hugging Face model ID or local model path.
        compute_dtype: Arithmetic dtype used by quantized layers.
        device_map: Transformers device-placement strategy.

    Returns:
        A quantized model prepared for k-bit training.
    """
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=compute_dtype,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        quantization_config=quantization_config,
        device_map=device_map,
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return prepare_model_for_kbit_training(model)


def build_lora_config(config: GRPOTrainConfig) -> LoraConfig:
    """Build the PEFT LoRA configuration used by the GRPO trainer.

    Args:
        config: Training configuration containing LoRA hyperparameters.

    Returns:
        A causal-language-model LoRA setup targeting all linear layers.
    """
    return LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )


def package_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "not-installed"


def set_reproducible_seeds(config: GRPOTrainConfig) -> None:
    random.seed(config.seed)
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)


class PantographRewardFunction:
    """Compile generated Lean proofs and convert verification outcomes to rewards.

    A verifier is created lazily and reused across reward calls. Successful Lean
    compilation receives ``compile_success_reward``; valid-format compilation
    failures receive ``format_reward``; malformed or rejected outputs retain the
    configured failure reward.
    """

    def __init__(self, config: GRPOTrainConfig) -> None:
        self.config = config
        self.imports = tuple(
            item.strip() for item in config.pantograph_imports.split(",") if item.strip()
        )
        self.verifier: PantographTheoremVerifier | None = None
        self.reward_log_path = (
            Path(config.reward_log_file).expanduser()
            if config.reward_log_file
            else None
        )
        if self.reward_log_path:
            self.reward_log_path.parent.mkdir(parents=True, exist_ok=True)
        atexit.register(self.close)

    def __call__(self, completions: list[Any], **kwargs: Any) -> list[float]:
        """Score a trainer batch of completions in input order.

        Args:
            completions: Generated proof outputs supplied by TRL.
            **kwargs: Batched dataset columns associated with those outputs.

        Returns:
            One numeric reward per completion.
        """
        rewards: list[float] = []
        for index, completion in enumerate(completions):
            row = self._row_kwargs(kwargs, index)
            rewards.append(self.score_completion(completion, row, index))
        return rewards

    def score_completion(
        self,
        completion: Any,
        row: dict[str, Any],
        index: int,
    ) -> float:
        """Validate, compile, log, and score one generated proof.

        Args:
            completion: Raw completion in a TRL-supported representation.
            row: Dataset fields associated with this generated completion.
            index: Position used for fallback IDs and logging.

        Returns:
            The configured reward corresponding to formatting and compilation status.
        """
        problem_id = str(row.get(self.config.id_field, f"row/{index}"))
        statement = str(row.get(self.config.statement_field, "") or "")
        proof = extract_completion_text(completion)
        log_row: dict[str, Any] = {
            "problem_id": problem_id,
            "reward_index": index,
            "statement_hash": row.get("statement_hash"),
            "generated_proof": proof,
            "success": False,
            "reward": self.config.failure_reward,
            "reason": "",
            "diagnostics": "",
        }
        start = time.monotonic()
        try:
            if not proof.strip():
                log_row["reason"] = "empty_completion"
                return self._finish(log_row, start)
            if contains_forbidden_proof_token(proof):
                log_row["reason"] = "forbidden_proof_token"
                return self._finish(log_row, start)
            declaration = compose_lean_theorem(statement, proof)
            source = build_lean_source_with_preamble(
                declaration,
                context_lines=coerce_string_tuple(row.get("context_lines")),
                label=problem_id,
            )
            result = self._verifier().check_source(
                source,
                timeout=self.config.lean_timeout,
                reject_forbidden=True,
            )
            log_row.update(
                {
                    "lean_code": source,
                    "success": result.success,
                    "diagnostics": result.diagnostics,
                    "compile_messages": list(result.messages),
                    "compile_errors": list(result.errors),
                    "compile_warnings": list(result.warnings),
                    "verification_seconds": result.check_seconds,
                    "timed_out": result.timed_out,
                }
            )
            if result.success:
                log_row["reward"] = self.config.compile_success_reward
                log_row["reason"] = "compiled"
            else:
                log_row["reward"] = self.config.format_reward
                log_row["reason"] = result.error_type or "compile_failed"
            return self._finish(log_row, start)
        except ValueError as error:
            log_row["reason"] = "invalid_proof_format"
            log_row["diagnostics"] = str(error)
            return self._finish(log_row, start)
        except Exception as error:
            log_row["reason"] = "reward_error"
            log_row["diagnostics"] = repr(error)
            return self._finish(log_row, start)

    def _row_kwargs(self, kwargs: dict[str, Any], index: int) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple)) and index < len(value):
                row[key] = value[index]
            else:
                row[key] = value
        return row

    def _verifier(self) -> PantographTheoremVerifier:
        """Return the warm, lazily initialized Pantograph verifier."""
        if self.verifier is None:
            self.verifier = PantographTheoremVerifier(
                self.config.lean_project_path,
                imports=self.imports,
                timeout=self.config.lean_timeout,
            )
            self.verifier.warmup(timeout=self.config.lean_timeout)
        return self.verifier

    def _finish(self, log_row: dict[str, Any], start: float) -> float:
        log_row["reward_seconds"] = round(time.monotonic() - start, 4)
        reward = float(log_row["reward"])
        self._write_log(log_row)
        return reward

    def _write_log(self, row: dict[str, Any]) -> None:
        if self.reward_log_path is None:
            return
        with self.reward_log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")

    def close(self) -> None:
        if self.verifier is not None:
            self.verifier.close()
            self.verifier = None


def extract_completion_text(completion: Any) -> str:
    """Normalize string, message, or message-list completions to plain text.

    Args:
        completion: Raw completion value emitted by TRL.

    Returns:
        Stripped generated proof text.
    """
    if isinstance(completion, str):
        return completion.strip()
    if isinstance(completion, dict):
        return str(completion.get("content") or completion.get("text") or "").strip()
    if isinstance(completion, list) and completion:
        last = completion[-1]
        if isinstance(last, dict):
            return str(last.get("content") or last.get("text") or "").strip()
        return str(last).strip()
    return str(completion or "").strip()


def coerce_string_tuple(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return tuple(line.strip() for line in value.splitlines() if line.strip())
    if isinstance(value, Iterable):
        return tuple(str(item).strip() for item in value if str(item).strip())
    return (str(value).strip(),) if str(value).strip() else ()


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

    tokenizer = build_tokenizer(config.model_name_or_path)
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
    compute_dtype = qlora_compute_dtype()
    model = build_model(
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
        peft_config=build_lora_config(config),
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
