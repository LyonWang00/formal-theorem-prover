"""QLoRA SFT entry point for prepared Lean proof data.

This script is training-only. Dataset downloading, sampling, schema adaptation,
and proof filtering live in ``lean_training.data.cli``. The expected input here is
a JSON/JSONL file whose rows contain only ``prompt`` and ``completion`` (plus an
optional sampling-weight column), paired with a rich verified manifest sidecar.

Example:

    python -m lean_prover.lean_training.sft \
        --model_name_or_path Qwen/Qwen2.5-0.5B-Instruct \
        --train_file outputs/lean_workbook/train20000.jsonl \
        --output_dir outputs/qwen2_5_0_5b_lean_sft/model
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset
from trl import SFTConfig, SFTTrainer
from torch.utils.data import Sampler, WeightedRandomSampler

from lean_prover.lean_training.data.contracts import make_attestation_id
from lean_prover.lean_training.data.preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
    proof_hash,
    statement_hash,
)
from lean_prover.lean_training.data.training import (
    build_generation_prompt,
    load_prepared_dataset,
)
from lean_prover.lean_training.modeling.lora import build_sft_lora_config
from lean_prover.lean_training.modeling.quantization import build_qlora_model, qlora_compute_dtype
from lean_prover.lean_training.modeling.runtime import package_version, set_reproducible_seeds
from lean_prover.lean_training.modeling.tokenizer import load_tokenizer
from lean_prover.lean_training.sft_pipeline.config import SFTTrainConfig
from lean_prover.lean_training.sft_pipeline.distributed import (
    DistributedContext,
    global_batch_size,
    initialize_distributed_context,
    resolve_qlora_device_map,
    validate_no_duplicate_train_sharding,
)


def completion_token_accuracy(eval_prediction) -> dict[str, float]:
    """Measure next-token accuracy only where completion labels are supervised."""

    predictions, labels = eval_prediction
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    if predictions.ndim == labels.ndim + 1:
        predictions = predictions.argmax(axis=-1)
    # Transformers may concatenate batch-size-one evaluation tensors into a
    # flat vector.  Shift on the sequence axis when it is retained, or on the
    # sole axis when the evaluator has flattened it.
    if predictions.ndim == 1:
        shifted_predictions = predictions[:-1]
        shifted_labels = labels[1:]
    else:
        shifted_predictions = predictions[..., :-1]
        shifted_labels = labels[..., 1:]
    supervised = shifted_labels != -100
    supervised_count = int(supervised.sum())
    if supervised_count == 0:
        return {"token_accuracy": 0.0}
    correct = (shifted_predictions == shifted_labels) & supervised
    return {"token_accuracy": float(correct.sum() / supervised_count)}


def completion_token_predictions(logits, labels):
    """Reduce evaluation logits to token IDs before Trainer stores them."""

    del labels
    if isinstance(logits, tuple):
        logits = logits[0]
    # TRL 1.8 may already return token IDs with shape (batch, sequence).
    # Only collapse the vocabulary axis when full causal-LM logits remain.
    return logits.argmax(dim=-1) if logits.ndim >= 3 else logits


class WeightedSFTTrainer(SFTTrainer):
    """SFTTrainer variant used only when expert-iteration sampling weights exist."""

    sample_weight_field: str | None = None
    sample_weights: list[float] | None = None

    def get_train_dataloader(self):
        """Capture weights before Trainer removes non-model dataset columns."""

        field = self.sample_weight_field
        dataset = self.train_dataset
        raw_weights = self.sample_weights
        if raw_weights is None and field and dataset is not None and field in dataset.column_names:
            raw_weights = dataset[field]
        if field and dataset is not None and raw_weights is not None:
            weights = torch.as_tensor(raw_weights, dtype=torch.double)
            if not torch.isfinite(weights).all() or torch.any(weights <= 0):
                raise ValueError("sample weights must be finite and strictly positive")

            def weighted_sampler(processed_dataset):
                if len(processed_dataset) != len(weights):
                    raise ValueError("dataset row count changed before weighted sampling")
                return WeightedRandomSampler(
                    weights,
                    num_samples=len(weights),
                    replacement=True,
                )

            return self._get_dataloader(
                dataset=dataset,
                description="Training",
                batch_size=self._train_batch_size,
                sampler_fn=weighted_sampler,
                is_training=True,
            )
        return super().get_train_dataloader()


class FixedManifestSampler(Sampler[int]):
    """One deterministic, without-replacement permutation for exactly one epoch."""

    def __init__(self, size: int, *, seed: int) -> None:
        if size <= 0:
            raise ValueError("fixed manifest sampler requires at least one row")
        self.size = size
        self.seed = seed
        self.drawn_indices: list[int] = []
        self.iteration_count = 0

    def __len__(self) -> int:
        return self.size

    def __iter__(self):
        if self.iteration_count:
            raise RuntimeError(
                "fixed_manifest_without_replacement refuses a second epoch"
            )
        self.iteration_count += 1
        generator = torch.Generator()
        generator.manual_seed(self.seed)
        self.drawn_indices = torch.randperm(
            self.size, generator=generator
        ).tolist()
        return iter(self.drawn_indices)


class FixedManifestSFTTrainer(SFTTrainer):
    """SFT trainer with an auditable one-epoch, without-replacement sampler."""

    fixed_manifest_sampler: FixedManifestSampler | None = None

    def get_train_dataloader(self):
        dataset = self.train_dataset
        if dataset is None:
            raise ValueError("fixed manifest training requires a train dataset")

        def sampler_fn(processed_dataset):
            sampler = FixedManifestSampler(
                len(processed_dataset), seed=int(self.args.data_seed)
            )
            self.fixed_manifest_sampler = sampler
            return sampler

        return self._get_dataloader(
            dataset=dataset,
            description="Training",
            batch_size=self._train_batch_size,
            sampler_fn=sampler_fn,
            is_training=True,
        )


def parse_args() -> SFTTrainConfig:
    """Parse QLoRA SFT options into an immutable training configuration.

    Returns:
        A validated field mapping represented as ``SFTTrainConfig``.
    """
    parser = argparse.ArgumentParser(
        description="Run QLoRA SFT on prepared Lean JSONL records."
    )
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--adapter_path", default=None)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", default=None)
    parser.add_argument(
        "--train_manifest_file",
        default=None,
        help=(
            "Rich verified manifest paired with the minimal train JSONL. "
            "Defaults to <train_file stem>.manifest.jsonl."
        ),
    )
    parser.add_argument(
        "--validation_manifest_file",
        default=None,
        help=(
            "Rich verified manifest paired with the minimal validation JSONL. "
            "Defaults to <validation_file stem>.manifest.jsonl."
        ),
    )
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--completion_field", default="completion")
    parser.add_argument("--sample_weight_field", default=None)
    parser.add_argument(
        "--sampling_strategy",
        choices=("auto", "source_weighted_replacement", "fixed_manifest_without_replacement"),
        default="auto",
    )
    parser.add_argument("--packing", action="store_true")
    parser.add_argument(
        "--require_pantograph_verified",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--require_supervised_eos",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--text_field", default=None, help="Deprecated; ignored.")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--allow_overlength", action="store_true")
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_validation_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=8)
    parser.add_argument("--dataloader_drop_last", action="store_true")
    parser.add_argument("--dataloader_num_workers", type=int, default=0)
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=None)
    parser.add_argument("--eval_strategy", default="steps")
    parser.add_argument("--save_strategy", default="steps")
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=None)
    parser.add_argument("--load_best_model_at_end", action="store_true")
    parser.add_argument("--metric_for_best_model", default="eval_loss")
    parser.add_argument("--greater_is_better", action="store_true")
    parser.add_argument("--warmup_ratio", type=float, default=0.03)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--lr_scheduler_type", default="linear")
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--device_map",
        default="local_rank",
        help=(
            "QLoRA placement. Use local_rank for Accelerate/DDP. "
            "Legacy single-process runs may explicitly request auto."
        ),
    )
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=20260711)
    parser.add_argument("--data_seed", type=int, default=20260711)
    args = parser.parse_args()
    if args.text_field:
        print("WARNING: --text_field is deprecated; SFT now uses prompt/completion.")

    return SFTTrainConfig(
        model_name_or_path=args.model_name_or_path,
        adapter_path=args.adapter_path,
        train_file=args.train_file,
        validation_file=args.validation_file,
        train_manifest_file=args.train_manifest_file,
        validation_manifest_file=args.validation_manifest_file,
        prompt_field=args.prompt_field,
        completion_field=args.completion_field,
        sample_weight_field=args.sample_weight_field,
        sampling_strategy=args.sampling_strategy,
        requested_packing=args.packing,
        require_pantograph_verified=args.require_pantograph_verified,
        require_supervised_eos=args.require_supervised_eos,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        allow_overlength=args.allow_overlength,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_validation_batch_size=args.per_device_validation_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        dataloader_drop_last=args.dataloader_drop_last,
        dataloader_num_workers=args.dataloader_num_workers,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        num_train_epochs=args.num_train_epochs,
        logging_steps=args.logging_steps,
        eval_steps=args.eval_steps,
        eval_strategy=args.eval_strategy,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=args.load_best_model_at_end,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=args.greater_is_better,
        warmup_ratio=args.warmup_ratio,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        lr_scheduler_type=args.lr_scheduler_type,
        max_grad_norm=args.max_grad_norm,
        gradient_checkpointing=args.gradient_checkpointing,
        device_map=args.device_map,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        seed=args.seed,
        data_seed=args.data_seed,
    )


def ensure_prompt_completion_fields(
    dataset: Dataset,
    *,
    prompt_field: str,
    completion_field: str,
    name: str,
) -> None:
    """Ensure a prepared dataset exposes the configured supervised fields.

    Args:
        dataset: Dataset whose schema is checked.
        prompt_field: Column containing model inputs.
        completion_field: Column containing target proofs.
        name: Human-readable dataset label used in errors.

    Raises:
        ValueError: If either required column is absent.
    """
    required = (prompt_field, completion_field)
    missing = [field for field in required if field not in dataset.column_names]
    if missing:
        raise ValueError(
            f"{name} dataset is missing required SFT supervision fields "
            f"{missing}; "
            f"columns={dataset.column_names}"
        )


def validate_trl_support() -> None:
    fields = getattr(SFTConfig, "__dataclass_fields__", {})
    if "completion_only_loss" not in fields:
        trl_version = package_version("trl")
        raise RuntimeError(
            "Installed TRL does not support SFTConfig(completion_only_loss=...). "
            f"Detected trl={trl_version}. Please install a TRL version with "
            "native prompt-completion completion-only SFT support."
        )


def validate_pantograph_attestation(dataset: Dataset, *, name: str) -> None:
    """Reject missing, stale, incomplete, or internally inconsistent attestations."""

    field = "pantograph_verified"
    if field not in dataset.column_names:
        raise ValueError(
            f"{name} dataset has no {field!r} column; regenerate it with "
            "lean_training.data.cli --verify_with_pantograph"
        )
    rejected = sum(value is not True for value in dataset[field])
    if rejected:
        raise ValueError(
            f"{name} dataset contains {rejected} rows without successful "
            "Pantograph proof validation"
        )
    required = (
        "record_id",
        "data_state",
        "statement_verified",
        "proof_verified",
        "environment_hash",
        "assembler_version",
        "normalization_version",
        "attestation_id",
        "assembled_source_hash",
    )
    missing = [field for field in required if field not in dataset.column_names]
    if missing:
        raise ValueError(
            f"{name} dataset is missing attestation contract fields {missing}"
        )
    environment_hashes: set[str] = set()
    for index, row in enumerate(dataset):
        environment_hash = str(row.get("environment_hash") or "")
        environment_hashes.add(environment_hash)
        expected = make_attestation_id(
            record_id=str(row.get("record_id") or ""),
            environment_hash=environment_hash,
            assembler_version=str(row.get("assembler_version") or ""),
            normalization_version=str(row.get("normalization_version") or ""),
            assembled_source_hash=str(row.get("assembled_source_hash") or ""),
        )
        invalid = []
        if row.get("data_state") != "verified":
            invalid.append("data_state")
        if row.get("statement_verified") is not True:
            invalid.append("statement_verified")
        if row.get("proof_verified") is not True:
            invalid.append("proof_verified")
        if not environment_hash:
            invalid.append("environment_hash")
        if row.get("assembler_version") != ASSEMBLER_VERSION:
            invalid.append("assembler_version")
        if row.get("normalization_version") != NORMALIZATION_VERSION:
            invalid.append("normalization_version")
        if not row.get("assembled_source_hash"):
            invalid.append("assembled_source_hash")
        if row.get("attestation_id") != expected:
            invalid.append("attestation_id")
        if invalid:
            raise ValueError(
                f"{name} dataset row {index} has invalid attestation fields {invalid}"
            )
    if len(environment_hashes) != 1:
        raise ValueError(
            f"{name} dataset mixes {len(environment_hashes)} environment hashes"
        )


def default_sft_manifest_path(training_file: str | Path) -> Path:
    """Return the conventional rich-manifest sidecar for a minimal SFT file."""

    path = Path(training_file).expanduser()
    return path.with_name(f"{path.stem}.manifest{path.suffix}")


def validate_sft_manifest_projection(
    dataset: Dataset,
    *,
    manifest_file: str | Path,
    prompt_field: str,
    completion_field: str,
    sample_weight_field: str | None = None,
    name: str,
) -> None:
    """Bind a two-column SFT dataset to its verified rich manifest."""

    allowed_columns = {prompt_field, completion_field}
    if sample_weight_field:
        allowed_columns.add(sample_weight_field)
    unexpected = sorted(set(dataset.column_names) - allowed_columns)
    if unexpected:
        raise ValueError(
            f"{name} minimal SFT projection has unexpected columns {unexpected}"
        )
    manifest = load_prepared_dataset(str(manifest_file))
    if len(dataset) != len(manifest):
        raise ValueError(
            f"{name} SFT projection has {len(dataset)} rows but its manifest has "
            f"{len(manifest)} rows"
        )
    for index, (training_row, manifest_row) in enumerate(zip(dataset, manifest)):
        statement = str(manifest_row.get("lean_statement") or "").strip()
        proof = str(manifest_row.get("proof") or "").strip()
        invalid: list[str] = []
        if manifest_row.get("schema_version") != "sft_manifest":
            invalid.append("schema_version")
        if manifest_row.get("pantograph_verified") is not True:
            invalid.append("pantograph_verified")
        if manifest_row.get("verification_scope") != "full_proof":
            invalid.append("verification_scope")
        if (
            not statement
            or manifest_row.get("statement_hash") != statement_hash(statement)
        ):
            invalid.append("statement_hash")
        if not proof or manifest_row.get("proof_hash") != proof_hash(proof):
            invalid.append("proof_hash")
        if invalid:
            raise ValueError(
                f"{name} manifest row {index} has invalid verification fields {invalid}"
            )
        expected_prompt = build_generation_prompt(manifest_row)
        if str(training_row[prompt_field]) != expected_prompt:
            raise ValueError(
                f"{name} SFT projection row {index} does not match its manifest prompt"
            )
        if str(training_row[completion_field]).strip() != proof:
            raise ValueError(
                f"{name} SFT projection row {index} does not match its manifest proof"
            )


def validate_sft_verification_contract(
    dataset: Dataset,
    *,
    training_file: str | Path,
    manifest_file: str | Path | None,
    prompt_field: str,
    completion_field: str,
    sample_weight_field: str | None,
    name: str,
) -> None:
    """Validate a minimal projection via sidecar, with legacy-row fallback."""

    resolved_manifest = (
        Path(manifest_file).expanduser()
        if manifest_file is not None
        else default_sft_manifest_path(training_file)
    )
    if resolved_manifest.exists():
        validate_sft_manifest_projection(
            dataset,
            manifest_file=resolved_manifest,
            prompt_field=prompt_field,
            completion_field=completion_field,
            sample_weight_field=sample_weight_field,
            name=name,
        )
        return
    if "pantograph_verified" in dataset.column_names:
        validate_pantograph_attestation(dataset, name=name)
        return
    raise ValueError(
        f"{name} uses the minimal SFT schema but verified manifest sidecar "
        f"{resolved_manifest} does not exist"
    )


def validate_dataset_lengths(
    dataset: Dataset,
    tokenizer,
    *,
    config: SFTTrainConfig,
    name: str,
) -> None:
    """Reject or report examples longer than the configured sequence limit.

    Args:
        dataset: Prepared prompt/completion dataset.
        tokenizer: Tokenizer used to measure the concatenated sequence.
        config: Length fields and overlength policy.
        name: Dataset label used in diagnostics.

    Raises:
        ValueError: On the first overlength row unless explicitly allowed.
    """
    overlength = 0
    for index, row in enumerate(dataset):
        prompt = str(row[config.prompt_field])
        completion = str(row[config.completion_field])
        token_count = len(
            tokenizer(prompt + completion, add_special_tokens=True)["input_ids"]
        )
        if token_count > config.max_seq_length:
            overlength += 1
            if not config.allow_overlength:
                raise ValueError(
                    f"{name} dataset row {index} has {token_count} tokens, "
                    f"exceeding max_seq_length={config.max_seq_length}. "
                    "Run lean_training.data.cli with --filter_overlength or pass "
                    "--allow_overlength explicitly."
                )
    if overlength:
        print(
            f"WARNING: {name} dataset contains {overlength} rows over "
            f"max_seq_length={config.max_seq_length}; continuing because "
            "--allow_overlength was set."
        )


def inspect_supervised_eos_contract(
    dataset: Dataset,
    tokenizer,
    *,
    prompt_field: str = "prompt",
    completion_field: str = "completion",
    max_seq_length: int = 1024,
    labels_provider=None,
) -> dict[str, Any]:
    """Inspect every row of a completion-only dataset for exact EOS supervision.

    ``labels_provider`` may return the effective ``(input_ids, labels)`` for a
    row after a trainer/collator has processed it.  Without a provider, this
    function deterministically constructs the expected completion-only labels
    from the frozen prompt and completion text.  In both modes the supervised
    token sequence must equal the complete, untruncated completion and end in
    exactly one tokenizer EOS token.
    """

    eos_token = tokenizer.eos_token
    eos_token_id = tokenizer.eos_token_id
    pad_token_id = tokenizer.pad_token_id
    if not eos_token or eos_token_id is None:
        raise RuntimeError("supervised EOS contract requires a tokenizer EOS")
    if pad_token_id is None:
        raise RuntimeError("supervised EOS contract requires a tokenizer PAD")
    if eos_token_id == pad_token_id:
        raise RuntimeError("supervised EOS contract requires EOS != PAD")

    counters: dict[str, Any] = {
        "total_records": len(dataset),
        "nonempty_completion_records": 0,
        "empty_completion_records": 0,
        "zero_label_records": 0,
        "records_with_supervised_eos": 0,
        "records_without_supervised_eos": 0,
        "records_whose_last_valid_label_is_eos": 0,
        "records_whose_last_valid_label_is_not_eos": 0,
        "records_with_eos_masked_as_minus_100": 0,
        "records_with_semantic_truncation": 0,
        "records_with_multiple_supervised_eos": 0,
        "valid_label_tokens_total": 0,
        "min_valid_label_tokens": None,
        "max_valid_label_tokens": 0,
        "error_examples": [],
        "eos_token": eos_token,
        "eos_token_id": eos_token_id,
        "pad_token_id": pad_token_id,
        "max_seq_length": max_seq_length,
    }
    valid_counts: list[int] = []

    for index in range(len(dataset)):
        row = dataset[index]
        prompt = str(row.get(prompt_field) or "")
        completion = str(row.get(completion_field) or "")
        issues: list[str] = []
        proof_body = (
            completion[: -len(eos_token)]
            if completion.endswith(eos_token)
            else completion
        )
        if proof_body.strip():
            counters["nonempty_completion_records"] += 1
        else:
            counters["empty_completion_records"] += 1
            issues.append("empty_completion")

        prompt_ids = tokenizer(prompt, add_special_tokens=True)["input_ids"]
        full_ids = tokenizer(
            prompt + completion, add_special_tokens=True
        )["input_ids"]
        if full_ids[: len(prompt_ids)] != prompt_ids:
            issues.append("prompt_not_token_prefix")
        expected_completion_ids = full_ids[len(prompt_ids) :]
        semantic_truncation = len(full_ids) > max_seq_length
        if not completion.endswith(eos_token):
            issues.append("completion_missing_eos_text")
        if not expected_completion_ids or expected_completion_ids[-1] != eos_token_id:
            issues.append("completion_tokens_do_not_end_in_eos")

        if labels_provider is None:
            input_ids = full_ids
            labels = [-100] * len(prompt_ids) + expected_completion_ids
        else:
            input_ids, labels = labels_provider(index, row)
            input_ids = list(input_ids)
            labels = list(labels)
            valid_from_effective = [
                value for value in labels if value != -100
            ]
            if valid_from_effective != expected_completion_ids:
                semantic_truncation = True
                issues.append("effective_labels_differ_from_full_completion")

        valid_positions = [
            position for position, label in enumerate(labels) if label != -100
        ]
        valid_count = len(valid_positions)
        valid_counts.append(valid_count)
        counters["valid_label_tokens_total"] += valid_count
        if not valid_positions:
            counters["zero_label_records"] += 1
            issues.append("zero_labels")
        else:
            last_valid_position = valid_positions[-1]
            last_valid_label = labels[last_valid_position]
            supervised_eos_count = sum(
                labels[position] == eos_token_id for position in valid_positions
            )
            if supervised_eos_count:
                counters["records_with_supervised_eos"] += 1
            else:
                counters["records_without_supervised_eos"] += 1
                issues.append("missing_supervised_eos")
            if supervised_eos_count > 1:
                counters["records_with_multiple_supervised_eos"] += 1
                issues.append("multiple_supervised_eos")
            if last_valid_label == eos_token_id:
                counters["records_whose_last_valid_label_is_eos"] += 1
            else:
                counters["records_whose_last_valid_label_is_not_eos"] += 1
                issues.append("last_valid_label_is_not_eos")
        if any(
            token == eos_token_id and labels[position] == -100
            for position, token in enumerate(input_ids[: len(labels)])
        ):
            counters["records_with_eos_masked_as_minus_100"] += 1
            issues.append("eos_masked_as_minus_100")

        if semantic_truncation:
            counters["records_with_semantic_truncation"] += 1
            issues.append("semantic_truncation")
        if issues and len(counters["error_examples"]) < 20:
            counters["error_examples"].append(
                {"index": index, "issues": sorted(set(issues))}
            )

    counters["min_valid_label_tokens"] = min(valid_counts) if valid_counts else 0
    counters["max_valid_label_tokens"] = max(valid_counts) if valid_counts else 0
    counters["records_without_supervised_eos"] = (
        counters["total_records"] - counters["records_with_supervised_eos"]
    )
    counters["mean_valid_label_tokens"] = (
        round(sum(valid_counts) / len(valid_counts), 6) if valid_counts else 0.0
    )
    counters["passed"] = not any(
        (
            counters["empty_completion_records"],
            counters["zero_label_records"],
            counters["records_without_supervised_eos"],
            counters["records_whose_last_valid_label_is_not_eos"],
            counters["records_with_eos_masked_as_minus_100"],
            counters["records_with_semantic_truncation"],
            counters["records_with_multiple_supervised_eos"],
        )
    )
    return counters


def assert_supervised_eos_contract(
    dataset: Dataset,
    tokenizer,
    *,
    prompt_field: str = "prompt",
    completion_field: str = "completion",
    max_seq_length: int = 1024,
    labels_provider=None,
) -> dict[str, Any]:
    """Fail fast unless every row has complete, unmasked final EOS supervision."""

    report = inspect_supervised_eos_contract(
        dataset,
        tokenizer,
        prompt_field=prompt_field,
        completion_field=completion_field,
        max_seq_length=max_seq_length,
        labels_provider=labels_provider,
    )
    if not report["passed"]:
        raise RuntimeError(
            "EOS_GATE_FAILED: "
            + json.dumps(report, ensure_ascii=False, sort_keys=True)
        )
    return report


def print_startup_report(
    config: SFTTrainConfig,
    *,
    train_dataset: Dataset,
    validation_dataset: Dataset | None,
    distributed_context: DistributedContext,
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
        "max_seq_length": config.max_seq_length,
        "completion_only_loss": True,
        "require_supervised_eos": config.require_supervised_eos,
        "requested_packing": config.requested_packing,
        "requested_sampling_strategy": config.sampling_strategy,
        "effective_sampling_strategy": (
            "fixed_manifest_without_replacement"
            if config.sampling_strategy == "fixed_manifest_without_replacement"
            else "source_weighted_replacement"
            if config.sample_weight_field
            else "trainer_default_without_replacement"
        ),
        "effective_packing": False,
        "distributed": distributed_context.as_dict(),
        "global_batch_size": global_batch_size(
            per_device_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            world_size=distributed_context.world_size,
        ),
        "accelerate_even_batches": False,
        "packing_disabled_reason": (
            "source-weighted sampler requires unpacked row boundaries"
            if config.requested_packing and config.sample_weight_field
            else "packing is disabled pending prompt/completion mask validation"
            if config.requested_packing
            else None
        ),
    }
    print("SFT_STARTUP " + json.dumps(report, ensure_ascii=False))


def save_training_config(
    config: SFTTrainConfig,
    distributed_context: DistributedContext,
) -> None:
    """Persist training parameters and dependency versions in the output directory.

    Args:
        config: Configuration to serialize alongside runtime package versions.

    Output:
        Writes ``training_config.json`` under ``config.output_dir``.
    """
    if not distributed_context.is_main_process:
        return
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
        "distributed_runtime": distributed_context.as_dict(),
        "global_batch_size": global_batch_size(
            per_device_batch_size=config.per_device_train_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            world_size=distributed_context.world_size,
        ),
    }
    (output_dir / "training_config.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def build_trainer(
    config: SFTTrainConfig,
    distributed_context: DistributedContext | None = None,
) -> tuple[SFTTrainer, Any]:
    """Validate inputs and assemble the tokenizer, QLoRA model, and SFT trainer.

    Args:
        config: Complete SFT dataset, optimization, evaluation, and LoRA settings.

    Returns:
        The configured ``SFTTrainer`` and tokenizer used for training/saving.

    Raises:
        ValueError: If dataset fields, lengths, or evaluation/save settings conflict.
        RuntimeError: If the installed TRL lacks required completion-only support.
    """
    validate_trl_support()
    distributed_context = distributed_context or initialize_distributed_context()
    if config.sampling_strategy == "fixed_manifest_without_replacement":
        if config.sample_weight_field:
            raise ValueError(
                "fixed_manifest_without_replacement forbids sample weights"
            )
        if config.num_train_epochs != 1:
            raise ValueError(
                "fixed_manifest_without_replacement requires num_train_epochs=1"
            )
        if config.max_steps != -1:
            raise ValueError(
                "fixed_manifest_without_replacement requires max_steps=-1"
            )
        if config.dataloader_drop_last:
            raise ValueError(
                "fixed_manifest_without_replacement requires dataloader_drop_last=False"
            )
    elif config.sampling_strategy == "source_weighted_replacement" and not config.sample_weight_field:
        raise ValueError(
            "source_weighted_replacement requires sample_weight_field"
        )
    set_reproducible_seeds(config)
    train_dataset = load_prepared_dataset(config.train_file)
    validate_no_duplicate_train_sharding(
        dataset_size=len(train_dataset),
        per_device_batch_size=config.per_device_train_batch_size,
        world_size=distributed_context.world_size,
        drop_last=config.dataloader_drop_last,
    )
    ensure_prompt_completion_fields(
        train_dataset,
        prompt_field=config.prompt_field,
        completion_field=config.completion_field,
        name="train",
    )
    validation_dataset = None
    if config.validation_file:
        validation_dataset = load_prepared_dataset(config.validation_file)
        ensure_prompt_completion_fields(
            validation_dataset,
            prompt_field=config.prompt_field,
            completion_field=config.completion_field,
            name="validation",
        )
    if config.require_pantograph_verified:
        validate_sft_verification_contract(
            train_dataset,
            training_file=config.train_file,
            manifest_file=config.train_manifest_file,
            prompt_field=config.prompt_field,
            completion_field=config.completion_field,
            sample_weight_field=config.sample_weight_field,
            name="train",
        )
        if validation_dataset is not None:
            validate_sft_verification_contract(
                validation_dataset,
                training_file=config.validation_file or "",
                manifest_file=config.validation_manifest_file,
                prompt_field=config.prompt_field,
                completion_field=config.completion_field,
                sample_weight_field=None,
                name="validation",
            )
    elif distributed_context.is_main_process:
        print(
            "DANGER_UNVERIFIED_SFT: Pantograph attestation enforcement is disabled; "
            "this option is for development only and must not be used for formal training."
        )

    tokenizer = load_tokenizer(config.model_name_or_path, padding_side="right")
    eos_contract_path = Path(config.output_dir) / "sft_supervised_eos_contract.json"
    if config.require_supervised_eos:
        source_eos_report = assert_supervised_eos_contract(
            train_dataset,
            tokenizer,
            prompt_field=config.prompt_field,
            completion_field=config.completion_field,
            max_seq_length=config.max_seq_length,
        )
        if distributed_context.is_main_process:
            eos_contract_path.parent.mkdir(parents=True, exist_ok=True)
            eos_contract_path.write_text(
                json.dumps(
                    {"source_preflight": source_eos_report},
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    validate_dataset_lengths(
        train_dataset,
        tokenizer,
        config=config,
        name="train",
    )
    if validation_dataset is not None:
        validate_dataset_lengths(
            validation_dataset,
            tokenizer,
            config=config,
            name="validation",
        )
    if distributed_context.is_main_process:
        print_startup_report(
            config,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            distributed_context=distributed_context,
        )
    compute_dtype = qlora_compute_dtype("QLoRA")
    model = build_qlora_model(
        config.model_name_or_path,
        compute_dtype=compute_dtype,
        device_map=resolve_qlora_device_map(config.device_map, distributed_context),
    )
    peft_config = build_sft_lora_config(
        r=config.lora_r,
        alpha=config.lora_alpha,
        dropout=config.lora_dropout,
    )
    if config.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(
            model,
            config.adapter_path,
            is_trainable=True,
        )
        peft_config = None
    bf16_supported = compute_dtype is torch.bfloat16
    has_validation_dataset = validation_dataset is not None
    eval_strategy = config.eval_strategy if has_validation_dataset else "no"
    if config.load_best_model_at_end and (
        eval_strategy == "no" or config.save_strategy != eval_strategy
    ):
        raise ValueError(
            "load_best_model_at_end requires matching non-'no' save_strategy "
            "and eval_strategy"
        )

    training_args = SFTConfig(
        output_dir=config.output_dir,
        max_length=config.max_seq_length,
        completion_only_loss=True,
        packing=False,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_validation_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        dataloader_drop_last=config.dataloader_drop_last,
        dataloader_num_workers=config.dataloader_num_workers,
        max_steps=config.max_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        eval_strategy=eval_strategy,
        eval_steps=config.eval_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        warmup_ratio=config.warmup_ratio,
        warmup_steps=config.warmup_steps,
        weight_decay=config.weight_decay,
        lr_scheduler_type=config.lr_scheduler_type,
        max_grad_norm=config.max_grad_norm,
        gradient_checkpointing=config.gradient_checkpointing,
        optim=config.optim,
        do_eval=has_validation_dataset,
        bf16=bf16_supported,
        fp16=not bf16_supported,
        report_to="none",
        seed=config.seed,
        data_seed=config.data_seed,
        accelerator_config={"even_batches": False},
        ddp_find_unused_parameters=(
            False if distributed_context.is_distributed else None
        ),
    )

    sample_weights = None
    if config.sample_weight_field and config.sample_weight_field in train_dataset.column_names:
        sample_weights = [float(value) for value in train_dataset[config.sample_weight_field]]
    trainer_class = (
        FixedManifestSFTTrainer
        if config.sampling_strategy == "fixed_manifest_without_replacement"
        else WeightedSFTTrainer
        if config.sample_weight_field
        else SFTTrainer
    )
    trainer = trainer_class(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        # TRL computes completion-label token accuracy inside ``compute_loss``
        # and reports it as ``eval_mean_token_accuracy``.  Avoid a second
        # prediction hook, which is both redundant and version-shape-sensitive.
        compute_metrics=None,
        preprocess_logits_for_metrics=None,
    )
    if config.require_supervised_eos:
        processed = trainer.train_dataset
        if processed is None or len(processed) != len(train_dataset):
            raise RuntimeError(
                "EOS_GATE_FAILED: trainer changed the training row count"
            )

        def effective_labels(index, _row):
            feature = dict(processed[index])
            batch = trainer.data_collator([feature])
            return (
                batch["input_ids"][0].tolist(),
                batch["labels"][0].tolist(),
            )

        effective_eos_report = assert_supervised_eos_contract(
            train_dataset,
            tokenizer,
            prompt_field=config.prompt_field,
            completion_field=config.completion_field,
            max_seq_length=config.max_seq_length,
            labels_provider=effective_labels,
        )
        if distributed_context.is_main_process:
            eos_contract_path.write_text(
                json.dumps(
                    {
                        "source_preflight": source_eos_report,
                        "effective_trainer_labels": effective_eos_report,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
                + "\n",
                encoding="utf-8",
            )
    if isinstance(trainer, WeightedSFTTrainer):
        trainer.sample_weight_field = config.sample_weight_field
        trainer.sample_weights = sample_weights
    if distributed_context.is_main_process:
        save_tokenization_diagnostics(
            trainer,
            tokenizer,
            raw_dataset=train_dataset,
            config=config,
        )
    return trainer, tokenizer


def save_tokenization_diagnostics(
    trainer: SFTTrainer,
    tokenizer,
    *,
    raw_dataset: Dataset,
    config: SFTTrainConfig,
) -> None:
    """Persist the exact TRL labels for five rows plus full length statistics."""

    processed = trainer.train_dataset
    if processed is None or len(processed) == 0:
        raise ValueError("TRL produced an empty training dataset")
    samples: list[dict[str, Any]] = []
    non_ignored_total = 0
    invalid_label_rows = 0
    completion_lengths: list[int] = []
    total_lengths: list[int] = []
    raw_total_lengths: list[int] = []
    for index in range(len(processed)):
        feature = dict(processed[index])
        batch = trainer.data_collator([feature])
        input_ids = batch["input_ids"][0].tolist()
        labels = batch["labels"][0].tolist()
        attention_tensor = batch.get("attention_mask")
        attention = (
            attention_tensor[0].tolist()
            if attention_tensor is not None
            else [1] * len(input_ids)
        )
        valid_indices = [position for position, value in enumerate(labels) if value != -100]
        valid_count = len(valid_indices)
        non_ignored_total += valid_count
        invalid_label_rows += int(valid_count == 0)
        completion_lengths.append(valid_count)
        total_lengths.append(sum(attention))
        raw_row = raw_dataset[index]
        raw_total_lengths.append(
            len(
                tokenizer(
                    str(raw_row[config.prompt_field])
                    + str(raw_row[config.completion_field]),
                    add_special_tokens=True,
                )["input_ids"]
            )
        )
        if index < 5:
            raw = raw_row
            samples.append(
                {
                    "index": index,
                    "prompt": raw.get(config.prompt_field),
                    "proof": raw.get(config.completion_field),
                    "formatted_text": str(raw[config.prompt_field])
                    + str(raw[config.completion_field]),
                    "decoded": tokenizer.decode(input_ids, skip_special_tokens=False),
                    "input_ids": input_ids,
                    "labels": labels,
                    "attention_mask": attention,
                    "completion_token_interval": (
                        [valid_indices[0], valid_indices[-1] + 1]
                        if valid_indices
                        else None
                    ),
                    "masked_ratio": round(1 - valid_count / max(1, len(labels)), 6),
                    "valid_label_tokens": valid_count,
                    "ends_with_eos": bool(
                        valid_indices
                        and tokenizer.eos_token_id is not None
                        and labels[valid_indices[-1]] == tokenizer.eos_token_id
                    ),
                }
            )
    if invalid_label_rows:
        raise ValueError(
            f"TRL label precheck found {invalid_label_rows} rows with all labels=-100"
        )

    def distribution(values: list[int]) -> dict[str, int | float]:
        ordered = sorted(values)

        def pick(ratio: float) -> int:
            return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * ratio))]

        return {
            "mean": round(sum(ordered) / len(ordered), 4),
            "p50": pick(0.50),
            "p90": pick(0.90),
            "p95": pick(0.95),
            "p99": pick(0.99),
            "max": ordered[-1],
        }

    payload = {
        "num_examples": len(processed),
        "max_seq_length": config.max_seq_length,
        "truncated_examples": sum(
            value > config.max_seq_length for value in raw_total_lengths
        ),
        "truncated_ratio": round(
            sum(value > config.max_seq_length for value in raw_total_lengths)
            / len(raw_total_lengths),
            8,
        ),
        "raw_total_tokens": distribution(raw_total_lengths),
        "effective_total_tokens": distribution(total_lengths),
        "completion_tokens": distribution(completion_lengths),
        "non_ignored_label_tokens_total": non_ignored_total,
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token": tokenizer.pad_token,
        "pad_token_id": tokenizer.pad_token_id,
        "chat_template": tokenizer.chat_template,
        "prompt_format": "plain_text_lean_sections_v1",
        "samples": samples,
    }
    destination = Path(config.output_dir) / "sft_tokenization_diagnostics.json"
    destination.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    """Run QLoRA SFT and save the adapter, tokenizer, state, and configuration."""
    config = parse_args()
    distributed_context = initialize_distributed_context()
    save_training_config(config, distributed_context)
    distributed_context.wait_for_everyone()
    trainer, tokenizer = build_trainer(config, distributed_context)
    trainer.train()
    trainer.accelerator.wait_for_everyone()
    if distributed_context.is_main_process:
        trainer.save_model(config.output_dir)
        tokenizer.save_pretrained(config.output_dir)
        trainer.save_state()
        save_training_config(config, distributed_context)
    trainer.accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()


