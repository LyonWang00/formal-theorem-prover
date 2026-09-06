"""GRPO training entry point for prepared Lean proof prompts.

This script is training-only. Dataset downloading, sampling, schema adaptation,
and filtering live in ``lean_training.data.cli``. The expected input here is
a JSON/JSONL file whose rows contain proof-free ``prompt`` records plus the
``lean_statement`` and preamble fields needed to score generated proofs.
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import random
import time
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Any, Iterable

import torch
from datasets import Dataset, load_dataset
from peft import PeftConfig, PeftModel, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from lean_prover.lean_training.data.preparation import (
    compose_lean_theorem,
    contains_forbidden_proof_token,
)
from lean_prover.lean_training.sft_pipeline.distributed import (
    DistributedContext,
    global_batch_size,
    initialize_distributed_context,
    resolve_qlora_device_map,
    validate_no_duplicate_train_sharding,
)
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)
from lean_prover.lean_training.verification.schema import VerificationTask


@dataclass(frozen=True)
class GRPOTrainConfig:
    """Complete GRPO, LoRA, Pantograph reward, and output configuration."""

    model_name_or_path: str
    adapter_path: str
    train_file: str
    output_dir: str
    validation_file: str | None = None
    prompt_field: str = "prompt"
    statement_field: str = "lean_statement"
    id_field: str = "id"
    max_prompt_length: int = 1024
    max_completion_length: int = 256
    num_generations: int = 8
    per_device_train_batch_size: int = 1
    per_device_validation_batch_size: int = 1
    gradient_accumulation_steps: int = 2
    learning_rate: float = 1e-5
    num_train_epochs: float = 1.0
    max_steps: int = -1
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
    device_map: str = "local_rank"
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    load_in_4bit: bool = False
    lean_project_path: str = "lean_project"
    pantograph_imports: str = "Mathlib"
    lean_timeout: int = 120
    pantograph_num_workers: int = 1
    pantograph_warmup_timeout: int = 180
    compile_success_reward: float = 1.0
    failure_reward: float = 0.0
    use_vllm: bool = True
    vllm_mode: str = "colocate"
    vllm_gpu_memory_utilization: float = 0.30
    vllm_tensor_parallel_size: int = 1
    vllm_enable_sleep_mode: bool = True
    vllm_enforce_eager: bool = True
    num_iterations: int = 4
    expected_sft_epoch: float = 2.0
    reward_log_file: str | None = None
    resume_from_checkpoint: str | None = None
    vllm_model_name_or_path: str | None = None
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
    parser.add_argument("--adapter_path", required=True)
    parser.add_argument("--train_file", required=True)
    parser.add_argument("--validation_file", default=None)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--prompt_field", default="prompt")
    parser.add_argument("--statement_field", default="lean_statement")
    parser.add_argument("--id_field", default="id")
    parser.add_argument("--max_prompt_length", type=int, default=1024)
    parser.add_argument("--max_completion_length", type=int, default=256)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_validation_batch_size", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--num_train_epochs", type=float, default=1.0)
    parser.add_argument("--max_steps", type=int, default=-1)
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
    parser.add_argument("--device_map", default="local_rank")
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--load_in_4bit",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Load the trainable model with bitsandbytes NF4. Disabled by default: "
            "merging a LoRA adapter into 4-bit weights for colocated vLLM can corrupt rollouts."
        ),
    )
    parser.add_argument(
        "--lean_project_path",
        default=str(Path(__file__).resolve().parents[2] / "lean_project"),
    )
    parser.add_argument("--pantograph_imports", default="Mathlib")
    parser.add_argument("--lean_timeout", type=int, default=120)
    parser.add_argument("--pantograph_num_workers", type=int, default=1)
    parser.add_argument("--pantograph_warmup_timeout", type=int, default=180)
    parser.add_argument("--compile_success_reward", type=float, default=1.0)
    parser.add_argument("--failure_reward", type=float, default=0.0)
    parser.add_argument("--use_vllm", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--vllm_mode", choices=("colocate", "server"), default="colocate")
    parser.add_argument("--vllm_gpu_memory_utilization", type=float, default=0.30)
    parser.add_argument("--vllm_tensor_parallel_size", type=int, default=1)
    parser.add_argument(
        "--vllm_enable_sleep_mode",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--vllm_enforce_eager",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Disable vLLM CUDA graphs; required for stable RTX-5090/SM12 rollout.",
    )
    parser.add_argument("--num_iterations", type=int, default=4)
    parser.add_argument("--expected_sft_epoch", type=float, default=2.0)
    parser.add_argument("--reward_log_file", default=None)
    parser.add_argument("--resume_from_checkpoint", default=None)
    parser.add_argument(
        "--vllm_model_name_or_path",
        default=None,
        help=(
            "Optional fully merged checkpoint used only to initialize colocated "
            "vLLM. The trainable policy still uses model_name_or_path plus the "
            "LoRA adapter and Trainer resume state."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260715)
    parser.add_argument("--data_seed", type=int, default=20260715)
    args = parser.parse_args()
    if args.num_generations != 8:
        parser.error("GRPO contract requires exactly --num_generations 8")
    if args.num_iterations not in {1, 2, 4}:
        parser.error("GRPO contract requires --num_iterations to be 1, 2, or 4")
    if (args.lora_r, args.lora_alpha) not in {(16, 32), (32, 64)}:
        parser.error(
            "GRPO must preserve one of the frozen SFT LoRA structures: "
            "(r=16, alpha=32) or (r=32, alpha=64)"
        )
    if args.compile_success_reward != 1.0 or args.failure_reward != 0.0:
        parser.error("GRPO reward contract is binary: success=1 and failure=0")
    if not args.use_vllm:
        parser.error("GRPO rollouts must use vLLM")
    return GRPOTrainConfig(**vars(args))


def patch_vllm_none_vocab_tokens() -> None:
    """Keep vLLM detokenization alive when a tokenizer vocab entry is null.

    vLLM 0.27.1 assumes every raw vocab piece is a string.  The DeepSeek
    tokenizer contains rare null pieces, so a sampled token can otherwise raise
    ``TypeError`` after generation has already produced valid token IDs and
    log-probabilities.  Falling back to the tokenizer-decoded string preserves
    those IDs and scores and changes no sampling or optimization setting.
    """
    from vllm.tokenizers import detokenizer_utils

    original = detokenizer_utils._restore_leading_spaces
    if getattr(original, "_grpo_accepts_none_vocab_token", False):
        return

    def safe_restore_leading_spaces(
        raw_token: str | None,
        token_str: str,
        marker: str,
    ) -> str:
        if raw_token is None:
            return token_str
        return original(raw_token, token_str, marker)

    safe_restore_leading_spaces._grpo_accepts_none_vocab_token = True  # type: ignore[attr-defined]
    detokenizer_utils._restore_leading_spaces = safe_restore_leading_spaces


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
    forbidden = {
        "proof",
        "completion",
        "generated_proof",
        "raw_completion",
        "normalized_proof",
        "reference_proof",
        "formal_proof",
        "lean_code",
        "answer",
    }.intersection(dataset.column_names)
    if forbidden:
        raise ValueError(
            f"{name} GRPO dataset contains proof-bearing fields: {sorted(forbidden)}"
        )


def materialize_repeat_tickets(dataset: Dataset) -> Dataset:
    """Expand positive repeat counts without mutating the physical manifest."""

    if "repeat" not in dataset.column_names:
        raise ValueError("GRPO training data must contain an explicit repeat field")
    rows: list[dict[str, Any]] = []
    for physical_ordinal, row in enumerate(dataset):
        repeat = int(row["repeat"])
        weight = float(row.get("sample_weight", 1.0))
        if repeat < 0:
            raise ValueError(f"negative repeat at physical row {physical_ordinal}")
        if repeat and weight <= 0:
            raise ValueError(
                f"positive repeat has non-positive sample_weight at row {physical_ordinal}"
            )
        for ticket_ordinal in range(repeat):
            ticket = dict(row)
            ticket["physical_ordinal"] = physical_ordinal
            ticket["repeat_ordinal"] = ticket_ordinal
            rows.append(ticket)
    if not rows:
        raise ValueError("repeat expansion produced an empty GRPO dataset")
    return Dataset.from_list(rows)


def validate_ddp_ticket_divisibility(
    *,
    ticket_count: int,
    config: GRPOTrainConfig,
    distributed: DistributedContext,
) -> int:
    """Reject unequal DDP shards or partial optimizer batches."""

    validate_no_duplicate_train_sharding(
        dataset_size=ticket_count,
        per_device_batch_size=config.per_device_train_batch_size,
        world_size=distributed.world_size,
        drop_last=False,
    )
    batch = global_batch_size(
        per_device_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        world_size=distributed.world_size,
    )
    if ticket_count % batch:
        raise ValueError(
            "repeat-expanded GRPO tickets must be divisible by global_batch_size: "
            f"tickets={ticket_count}, global_batch_size={batch}"
        )
    if batch % config.num_generations:
        raise ValueError(
            f"global_batch_size={batch} must be divisible by num_generations="
            f"{config.num_generations}"
        )
    return batch


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


def validate_epoch2_adapter(config: GRPOTrainConfig) -> PeftConfig:
    """Bind GRPO initialization to the completed epoch=2 SFT adapter."""

    adapter_path = Path(config.adapter_path).expanduser()
    trainer_state_path = adapter_path / "trainer_state.json"
    if not trainer_state_path.exists():
        raise FileNotFoundError(f"SFT trainer state not found: {trainer_state_path}")
    trainer_state = json.loads(trainer_state_path.read_text(encoding="utf-8"))
    epoch = float(trainer_state.get("epoch", -1))
    if epoch != config.expected_sft_epoch:
        raise ValueError(
            f"GRPO requires SFT epoch={config.expected_sft_epoch}, got {epoch} "
            f"from {trainer_state_path}"
        )
    peft_config = PeftConfig.from_pretrained(str(adapter_path))
    if int(getattr(peft_config, "r", -1)) != config.lora_r:
        raise ValueError(
            f"SFT adapter LoRA rank differs from GRPO: "
            f"adapter={getattr(peft_config, 'r', None)}, grpo={config.lora_r}"
        )
    if int(getattr(peft_config, "lora_alpha", -1)) != config.lora_alpha:
        raise ValueError("SFT adapter LoRA alpha differs from GRPO configuration")
    if float(getattr(peft_config, "lora_dropout", -1)) != config.lora_dropout:
        raise ValueError("SFT adapter LoRA dropout differs from GRPO configuration")
    base = str(getattr(peft_config, "base_model_name_or_path", "") or "")
    if Path(base).resolve() != Path(config.model_name_or_path).expanduser().resolve():
        raise ValueError(
            "--model_name_or_path must exactly match the epoch=2 adapter base: "
            f"{config.model_name_or_path!r} != {base!r}"
        )
    return peft_config


def build_model(
    config: GRPOTrainConfig,
    *,
    compute_dtype: torch.dtype,
    device_map: str | dict[str, int] | None,
):
    """Load the epoch=2 SFT LoRA adapter as the trainable GRPO model.

    Args:
        config: GRPO configuration with exact base and epoch=2 adapter paths.
        compute_dtype: Arithmetic dtype used by the model.
        device_map: Transformers device-placement strategy.

    Returns:
        The exact epoch=2 adapter over either BF16/FP16 or optional NF4 base weights.
    """
    model_kwargs: dict[str, Any] = {
        "device_map": device_map,
        "trust_remote_code": True,
        "local_files_only": True,
    }
    if config.load_in_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute_dtype,
        )
    else:
        model_kwargs["dtype"] = compute_dtype
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name_or_path,
        **model_kwargs,
    )
    model.config.use_cache = False
    if config.load_in_4bit:
        model = prepare_model_for_kbit_training(model)
    adapter_init_path = config.resume_from_checkpoint or config.adapter_path
    model = PeftModel.from_pretrained(
        model,
        adapter_init_path,
        is_trainable=True,
    )
    if config.vllm_model_name_or_path:
        vllm_path = Path(config.vllm_model_name_or_path).expanduser()
        if not (vllm_path / "config.json").exists():
            raise FileNotFoundError(
                f"merged vLLM initialization model not found: {vllm_path}"
            )
        # TRL's VLLMGeneration initializes LLM(model=model.name_or_path).
        # Point only that initial load at an exact merged checkpoint policy;
        # the PeftModel above remains the trainable base-plus-LoRA policy.
        model.name_or_path = str(vllm_path)
    return model


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

    A persistent multi-process pool is started and warmed before Trainer setup,
    then reused across every rollout batch.  The reward is strictly binary:
    compilation without ``sorry``/``admit`` is 1; every other outcome is 0.
    """

    def __init__(self, config: GRPOTrainConfig, *, rank: int = 0) -> None:
        # TRL normalizes reward callables through ``__name__`` during trainer
        # construction.  Callable instances do not expose it by default.
        self.__name__ = "pantograph_binary_compile_reward"
        self.config = config
        self.rank = rank
        self.imports = tuple(
            item.strip() for item in config.pantograph_imports.split(",") if item.strip()
        )
        spool_dir = Path(config.output_dir) / "pantograph_tasks" / f"rank_{rank:03d}"
        self.pool = VerificationPool(
            VerificationPoolConfig(
                lean_project_path=config.lean_project_path,
                imports=self.imports,
                timeout=config.lean_timeout,
                warmup_timeout=config.pantograph_warmup_timeout,
                num_workers=config.pantograph_num_workers,
                task_spool_dir=str(spool_dir),
            )
        )
        self._next_problem_index = 0
        self._next_reward_batch_index = 0
        self.reward_log_path = (
            Path(config.reward_log_file).expanduser().with_suffix(f".rank{rank}.jsonl")
            if config.reward_log_file
            else None
        )
        if self.reward_log_path:
            self.reward_log_path.parent.mkdir(parents=True, exist_ok=True)
        atexit.register(self.close)

    def start(self) -> None:
        """Start and warm all Pantograph workers before any rollout request."""

        self.pool.start()

    def __call__(self, completions: list[Any], **kwargs: Any) -> list[float]:
        """Score a trainer batch of completions in input order.

        Args:
            completions: Generated proof outputs supplied by TRL.
            **kwargs: Batched dataset columns associated with those outputs.

        Returns:
            One numeric reward per completion.
        """
        batch_start = time.monotonic()
        reward_batch_index = self._next_reward_batch_index
        self._next_reward_batch_index += 1
        rewards = [self.config.failure_reward] * len(completions)
        pending: list[tuple[int, VerificationTask, dict[str, Any]]] = []
        for index, completion in enumerate(completions):
            row = self._row_kwargs(kwargs, index)
            problem_id = str(row.get(self.config.id_field, f"row/{index}"))
            statement = str(row.get(self.config.statement_field, "") or "")
            proof = extract_completion_text(completion)
            log_row: dict[str, Any] = {
                "problem_id": problem_id,
                "reward_batch_index": reward_batch_index,
                "reward_index": index,
                "statement_hash": row.get("statement_hash"),
                "generated_proof": proof,
                "success": False,
                "reward": self.config.failure_reward,
                "reason": "",
                "diagnostics": "",
                "rank": self.rank,
            }
            try:
                if not proof.strip():
                    log_row["reason"] = "empty_completion"
                    self._finish(log_row, batch_start)
                    continue
                if contains_forbidden_proof_token(proof):
                    log_row["reason"] = "forbidden_proof_token"
                    self._finish(log_row, batch_start)
                    continue
                # Keep this payload unassembled. VerificationTask owns the
                # declaration plus proof, while PantographTaskVerifier is the
                # single assembly boundary for context and namespace. Building
                # a full source here as well would duplicate imports/context.
                declaration = compose_lean_theorem(statement, proof)
                problem_index = self._next_problem_index
                self._next_problem_index += 1
                task = VerificationTask(
                    priority=0,
                    problem_index=problem_index,
                    attempt_index=0,
                    problem_id=problem_id,
                    prompt=str(row.get(self.config.prompt_field, "")),
                    generated_proof=proof,
                    raw_completion=proof,
                    lean_code=declaration,
                    imports=coerce_string_tuple(row.get("imports")) or self.imports,
                    context_lines=coerce_string_tuple(row.get("context_lines")),
                    payload={
                        "reward_batch_index": reward_batch_index,
                        "reward_index": index,
                        "statement_hash": row.get("statement_hash"),
                    },
                    reject_forbidden=True,
                )
                pending.append((index, task, log_row))
            except ValueError as error:
                log_row["reason"] = "invalid_proof_format"
                log_row["diagnostics"] = str(error)
                self._finish(log_row, batch_start)
        if pending:
            run = self.pool.run_batch(task for _, task, _ in pending)
            if run.fatal_errors:
                raise RuntimeError(f"Pantograph reward pool failed: {run.fatal_errors}")
            by_attempt = {str(result["attempt_id"]): result for result in run.results}
            for index, task, log_row in pending:
                result = by_attempt.get(f"{task.problem_index},{task.attempt_index}")
                if result is None:
                    log_row["reason"] = "missing_verification_result"
                else:
                    success = bool(result.get("success"))
                    log_row.update(
                        {
                            "success": success,
                            "reward": (
                                self.config.compile_success_reward
                                if success
                                else self.config.failure_reward
                            ),
                            "reason": "compiled" if success else (result.get("error_type") or "compile_failed"),
                            "diagnostics": result.get("diagnostics", ""),
                            "compile_messages": result.get("compile_messages", []),
                            "compile_errors": result.get("compile_errors", []),
                            "compile_warnings": result.get("compile_warnings", []),
                            "verification_seconds": result.get("verification_seconds"),
                            "timed_out": bool(result.get("timed_out")),
                            "worker_id": result.get("worker_id"),
                            "worker_pid": result.get("worker_pid"),
                        }
                    )
                rewards[index] = self._finish(log_row, batch_start)
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
        kwargs = {key: [value] for key, value in row.items()}
        return self([completion], **kwargs)[0]

    def _row_kwargs(self, kwargs: dict[str, Any], index: int) -> dict[str, Any]:
        row: dict[str, Any] = {}
        for key, value in kwargs.items():
            if isinstance(value, (list, tuple)) and index < len(value):
                row[key] = value[index]
            else:
                row[key] = value
        return row

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
        self.pool.close()


def prewarm_pantograph_workers(
    reward_func: PantographRewardFunction,
    distributed: DistributedContext,
) -> None:
    """Warm one persistent pool per DDP rank without Mathlib contention.

    All ranks call this function. Startup is deliberately serialized by rank
    because concurrent Pantograph/Mathlib initialization against one workspace
    can exceed the startup deadline. Once warm, each worker remains alive while
    later ranks warm and throughout every rollout batch.
    """

    for rank_to_warm in range(distributed.world_size):
        if distributed.rank == rank_to_warm:
            start = time.monotonic()
            reward_func.start()
            print(
                "PANTOGRAPH_PREWARM "
                + json.dumps(
                    {
                        "rank": distributed.rank,
                        "world_size": distributed.world_size,
                        "seconds": round(time.monotonic() - start, 4),
                        "runtime": reward_func.pool.runtime_snapshot(),
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                flush=True,
            )
        distributed.wait_for_everyone()


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
                "Run lean_training.data.cli with --filter_overlength or raise "
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


def configure_vllm_runtime(*, enforce_eager: bool) -> None:
    """Inject eager execution into TRL's colocated vLLM constructor."""
    if not enforce_eager:
        return
    from trl.generation import vllm_generation

    current = vllm_generation.LLM
    if getattr(current, "_lean_prover_enforce_eager", False):
        return

    def eager_llm(*args: Any, **kwargs: Any):
        kwargs["enforce_eager"] = True
        # vLLM 0.27 enables FlashInfer autotuning by default on Blackwell.
        # Its optional tuner currently faults on the cluster's RTX-5090
        # runtime, while regular eager FlashAttention inference is stable.
        kwargs["enable_flashinfer_autotune"] = False
        return current(*args, **kwargs)

    eager_llm._lean_prover_enforce_eager = True  # type: ignore[attr-defined]
    vllm_generation.LLM = eager_llm


def print_startup_report(
    config: GRPOTrainConfig,
    *,
    physical_train_rows: int,
    train_dataset: Dataset,
    validation_dataset: Dataset | None,
    distributed: DistributedContext,
    effective_global_batch_size: int,
) -> None:
    unique_prompts_per_generation_batch = (
        effective_global_batch_size // config.num_generations
    )
    expected_generation_batches_per_epoch = (
        len(train_dataset) * config.num_generations // effective_global_batch_size
    )
    expected_optimizer_steps_per_epoch = (
        expected_generation_batches_per_epoch * config.num_iterations
    )
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
        "physical_train_rows": physical_train_rows,
        "repeat_expanded_train_tickets": len(train_dataset),
        "validation_samples": len(validation_dataset) if validation_dataset else 0,
        "max_prompt_length": config.max_prompt_length,
        "max_completion_length": config.max_completion_length,
        "num_generations": config.num_generations,
        "num_iterations": config.num_iterations,
        "effective_global_batch_size": effective_global_batch_size,
        "unique_prompts_per_generation_batch": unique_prompts_per_generation_batch,
        "expected_generation_batches_per_epoch": expected_generation_batches_per_epoch,
        "expected_optimizer_steps_per_epoch": expected_optimizer_steps_per_epoch,
        "model_precision": "nf4" if config.load_in_4bit else str(qlora_compute_dtype()),
        "distributed": distributed.as_dict(),
        "reward_backend": "persistent_pantograph_pool_binary",
        "rollout_backend": "vllm",
        "sft_adapter_path": config.adapter_path,
        "policy_adapter_init_path": (
            config.resume_from_checkpoint or config.adapter_path
        ),
        "vllm_initial_model_path": (
            config.vllm_model_name_or_path or config.model_name_or_path
        ),
        "resume_from_checkpoint": config.resume_from_checkpoint,
        "expected_sft_epoch": config.expected_sft_epoch,
    }
    print("GRPO_STARTUP " + json.dumps(report, ensure_ascii=False))


def save_training_config(config: GRPOTrainConfig) -> None:
    """Persist GRPO settings and dependency versions in the output directory.

    Args:
        config: Configuration to serialize.

    Output:
        Writes ``grpo_training_config.json`` under ``config.output_dir``.
    """
    if int(os.environ.get("RANK", "0")) != 0:
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
    if config.use_vllm and config.vllm_mode == "colocate":
        configure_vllm_runtime(enforce_eager=config.vllm_enforce_eager)
    set_reproducible_seeds(config)
    distributed = initialize_distributed_context()
    physical_train_dataset = load_prepared_dataset(config.train_file)
    ensure_grpo_fields(physical_train_dataset, config, name="train")
    train_dataset = materialize_repeat_tickets(physical_train_dataset)
    effective_batch = validate_ddp_ticket_divisibility(
        ticket_count=len(train_dataset),
        config=config,
        distributed=distributed,
    )
    validate_epoch2_adapter(config)
    validation_dataset = None
    if config.validation_file:
        validation_dataset = load_prepared_dataset(config.validation_file)
        ensure_grpo_fields(validation_dataset, config, name="validation")

    tokenizer = build_tokenizer(config.adapter_path)
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
        physical_train_rows=len(physical_train_dataset),
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        distributed=distributed,
        effective_global_batch_size=effective_batch,
    )
    compute_dtype = qlora_compute_dtype()
    device_map = resolve_qlora_device_map(config.device_map, distributed)
    reward_func = PantographRewardFunction(config, rank=distributed.rank)
    # Pantograph workers must be fully warm before model/vLLM initialization;
    # no rollout request is allowed to pay process startup or Mathlib warmup.
    prewarm_pantograph_workers(reward_func, distributed)
    try:
        model = build_model(
            config,
            compute_dtype=compute_dtype,
            device_map=device_map,
        )
    except Exception:
        reward_func.close()
        raise
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
                "num_iterations": config.num_iterations,
                "generation_batch_size": effective_batch,
                "per_device_train_batch_size": config.per_device_train_batch_size,
                "per_device_eval_batch_size": config.per_device_validation_batch_size,
                "gradient_accumulation_steps": config.gradient_accumulation_steps,
                "learning_rate": config.learning_rate,
                "num_train_epochs": config.num_train_epochs,
                "max_steps": config.max_steps,
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
                "gradient_checkpointing_kwargs": {"use_reentrant": False},
                "ddp_find_unused_parameters": False,
                "dataloader_drop_last": False,
                "bf16": compute_dtype is torch.bfloat16,
                "fp16": compute_dtype is torch.float16,
                "report_to": "none",
                "log_completions": True,
                "use_vllm": config.use_vllm,
                "vllm_mode": config.vllm_mode,
                "vllm_gpu_memory_utilization": config.vllm_gpu_memory_utilization,
                "vllm_tensor_parallel_size": config.vllm_tensor_parallel_size,
                "vllm_enable_sleep_mode": config.vllm_enable_sleep_mode,
                "vllm_max_model_length": config.max_prompt_length + config.max_completion_length,
                "beta": 0.0,
                "seed": config.seed,
                "data_seed": config.data_seed,
            },
        )
    )
    trainer = GRPOTrainer(
        model=model,
        reward_funcs=[reward_func],
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=validation_dataset,
        processing_class=tokenizer,
        # The trainable adapter is the exact SFT epoch=2 adapter loaded above;
        # passing a new PEFT config would incorrectly create a second adapter.
        peft_config=None,
    )
    return trainer, tokenizer, reward_func


def main() -> None:
    """Run GRPO training and save adapter, tokenizer, state, and configuration."""
    config = parse_args()
    patch_vllm_none_vocab_tokens()
    save_training_config(config)
    trainer, tokenizer, reward_func = build_trainer(config)
    try:
        trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
        trainer.save_model(config.output_dir)
        if trainer.is_world_process_zero():
            tokenizer.save_pretrained(config.output_dir)
        trainer.save_state()
        save_training_config(config)
    finally:
        reward_func.close()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
