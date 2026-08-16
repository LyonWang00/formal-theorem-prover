"""Validated configuration for multi-round expert iteration."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from lean_prover.lean_training.runtime.config import RuntimeConfig
from lean_prover.lean_training.runtime.profile import resolve_runtime_profile


class DataConfig(BaseModel):
    train_path: str
    eval_path: str
    discovery_path: str
    monitor_path: str | None = None
    benchmark_path: str | None = None
    benchmark_dev_path: str | None = None
    benchmark_test_path: str | None = None
    overlap_policy: Literal["error", "warning"] = "error"

    @model_validator(mode="before")
    @classmethod
    def map_legacy_names(cls, values: Any) -> Any:
        if not isinstance(values, dict):
            return values
        mapped = dict(values)
        for old, new in {
            "validation_path": "eval_path",
            "test_path": "benchmark_path",
            "train_seed_path": "train_path",
        }.items():
            if old in mapped and new not in mapped:
                mapped[new] = mapped[old]
        return mapped


class InitialModelConfig(BaseModel):
    base_model: str
    sft_adapter: str | None = None
    tokenizer: str | None = None
    skip_initial_sft: bool = True
    checkpoint_strategy: Literal["fixed_anchor", "continue_adapter"] = "fixed_anchor"
    merged_sft0_path: str | None = None


class IterationsConfig(BaseModel):
    max_iterations: int = Field(default=3, gt=0)
    minimum_new_proofs_to_train: int = Field(default=1, ge=0)
    stop_when_training_skipped: bool = False
    append_discovery_size_if_insufficient: int = Field(default=0, ge=0)
    terminate_if_new_proofs_below: int = Field(default=0, ge=0)


class GenerationConfig(BaseModel):
    backend: Literal["auto", "transformers", "vllm"] = "auto"
    samples_per_statement: int = Field(default=4, gt=0)
    temperature: float = Field(default=0.9, ge=0)
    top_p: float = Field(default=0.95, gt=0, le=1)
    max_new_tokens: int = Field(default=1024, gt=0)
    batch_size: int = Field(default=8, gt=0)
    load_in_4bit: bool = True
    max_model_len: int | None = Field(default=None, gt=0)
    gpu_memory_utilization: float = Field(default=0.9, gt=0, le=1)
    enforce_eager: bool = False
    kv_cache_memory_bytes: int | None = Field(default=None, gt=0)
    generation_contract_path: str | None = None
    generation_contract_sha256: str | None = None


class DiscoveryConfig(BaseModel):
    statements_per_iteration: int = Field(default=10000, gt=0)
    batch_statements: int | None = Field(default=None, gt=0)
    max_consecutive_zero_success_rounds: int = Field(default=2, gt=0)
    hard_archive_after_rounds: int = Field(default=3, gt=0)
    solved_easy_success_ratio: float = Field(default=1.0, gt=0, le=1)
    pool_mix: dict[str, float] = Field(
        default_factory=lambda: {
            "new": 0.30,
            "frontier": 0.35,
            "unsolved": 0.30,
            "audit": 0.05,
        }
    )
    generation: GenerationConfig = Field(default_factory=GenerationConfig)
    sampling_budget: dict[str, int] = Field(
        default_factory=lambda: {"new": 4, "frontier": 4, "unsolved": 6, "audit": 2}
    )
    iteration_bucket_counts: dict[int, dict[str, int]] = Field(default_factory=dict)
    iteration_sampling_budget: dict[int, dict[str, int]] = Field(default_factory=dict)
    iteration_generation_seeds: dict[int, int] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_discovery(self) -> "DiscoveryConfig":
        required = {"new", "frontier", "unsolved", "audit"}
        if set(self.pool_mix) != required:
            raise ValueError(f"discovery.pool_mix must contain exactly {sorted(required)}")
        if abs(sum(self.pool_mix.values()) - 1.0) > 1e-6:
            raise ValueError("discovery.pool_mix values must sum to 1")
        if any(value < 0 for value in self.pool_mix.values()):
            raise ValueError("discovery.pool_mix values must be non-negative")
        if set(self.sampling_budget) != required or any(
            value <= 0 for value in self.sampling_budget.values()
        ):
            raise ValueError("discovery.sampling_budget requires positive values for every bucket")
        for iteration, counts in self.iteration_bucket_counts.items():
            if iteration < 0 or set(counts) != required or any(value < 0 for value in counts.values()):
                raise ValueError(
                    "discovery.iteration_bucket_counts requires non-negative values "
                    "for every bucket at each non-negative iteration"
                )
        for iteration, budget in self.iteration_sampling_budget.items():
            if iteration < 0 or set(budget) != required or any(value <= 0 for value in budget.values()):
                raise ValueError(
                    "discovery.iteration_sampling_budget requires positive values "
                    "for every bucket at each non-negative iteration"
                )
        return self


class VerificationConfig(BaseModel):
    lean_project_path: str = "lean_project"
    imports: list[str] = Field(default_factory=lambda: ["Mathlib"])
    num_workers: int = Field(default=2, gt=0)
    queue_maxsize: int = Field(default=128, gt=0)
    timeout_seconds: int = Field(default=120, gt=0)
    warmup_timeout_seconds: int = Field(default=120, gt=0)
    heartbeat_interval_seconds: float = Field(default=5.0, gt=0)
    heartbeat_timeout_seconds: float = Field(default=30.0, gt=0)
    max_worker_restarts: int = Field(default=3, ge=0)
    max_task_retries: int = Field(default=2, ge=0)
    shutdown_timeout_seconds: float = Field(default=10.0, gt=0)
    reject_sorry: bool = True
    reject_admit: bool = True
    reject_axiom: bool = True
    use_cache: bool = True
    cache_backend: Literal["sqlite"] = "sqlite"
    save_full_source_on_failure_only: bool = True

    @model_validator(mode="after")
    def validate_heartbeat(self) -> "VerificationConfig":
        if self.heartbeat_timeout_seconds <= self.heartbeat_interval_seconds:
            raise ValueError(
                "heartbeat_timeout_seconds must exceed heartbeat_interval_seconds"
            )
        return self


class ProofBankConfig(BaseModel):
    max_verified_proofs_per_statement: int = Field(default=3, gt=0)
    max_training_proofs_per_statement: int = Field(default=1, gt=0)
    selection_strategy: Literal[
        "random_verified", "shortest", "fastest_compile"
    ] = "random_verified"


class TrainMixConfig(BaseModel):
    anchor: float = Field(ge=0)
    historical_expert: float = Field(default=0, ge=0)
    current_expert: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_sum(self) -> "TrainMixConfig":
        if abs(self.anchor + self.historical_expert + self.current_expert - 1.0) > 1e-6:
            raise ValueError("train mix weights must sum to 1")
        return self


class CategorySamplingConfig(BaseModel):
    enabled: bool = False
    strategy: Literal["sqrt_inverse_frequency"] = "sqrt_inverse_frequency"
    max_oversample_factor: float = Field(default=4.0, ge=1)


class TrainingConfig(BaseModel):
    use_qlora: bool = True
    lora_rank: int = Field(default=64, gt=0)
    lora_alpha: int = Field(default=128, gt=0)
    lora_dropout: float = Field(default=0.05, ge=0, lt=1)
    learning_rate: float = Field(default=2e-5, gt=0)
    initial_learning_rate: float | None = Field(default=None, gt=0)
    num_train_epochs: float = Field(default=1.0, gt=0)
    per_device_train_batch_size: int = Field(default=1, gt=0)
    per_device_eval_batch_size: int = Field(default=1, gt=0)
    gradient_accumulation_steps: int = Field(default=8, gt=0)
    gradient_checkpointing: bool = True
    max_seq_length: int = Field(default=2048, gt=0)
    packing: bool = False
    logging_steps: int = Field(default=10, gt=0)
    optimizer: str = "paged_adamw_8bit"
    weight_decay: float = Field(default=0.0, ge=0)
    warmup_ratio: float = Field(default=0.03, ge=0, lt=1)
    max_grad_norm: float = Field(default=1.0, gt=0)
    eval_strategy: str = "steps"
    eval_steps: int | None = Field(default=100, gt=0)
    save_strategy: str = "steps"
    save_steps: int = Field(default=100, gt=0)
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    greater_is_better: bool = False

    @model_validator(mode="after")
    def validate_eval_direction(self) -> "TrainingConfig":
        if self.metric_for_best_model == "eval_loss" and self.greater_is_better:
            raise ValueError("greater_is_better must be false for eval_loss")
        return self


class ExecutionConfig(BaseModel):
    isolate_gpu_stages: bool = True
    verify_after_each_generation_batch: bool = False
    generation_timeout_seconds: int = Field(default=7200, gt=0)
    training_timeout_seconds: int = Field(default=86400, gt=0)
    flashinfer_sampler: bool = True


class LeanSmokePrecheckConfig(BaseModel):
    enabled: bool = True
    required_pass_rate: float = Field(default=1.0, ge=0, le=1)


class ReferenceRoundtripPrecheckConfig(BaseModel):
    enabled: bool = True
    train_samples: int = Field(default=10, ge=0)
    eval_samples: int = Field(default=10, ge=0)
    benchmark_samples: int = Field(default=10, ge=0)
    required_pass_rate: float = Field(default=1.0, ge=0, le=1)


class GenerationSmokePrecheckConfig(BaseModel):
    enabled: bool = True
    statements: int = Field(default=10, gt=0, le=20)
    samples_per_statement: int = Field(default=2, gt=0, le=4)
    max_length_finish_ratio: float = Field(default=0.30, ge=0, le=1)
    minimum_extraction_success_rate: float = Field(default=0.90, ge=0, le=1)


class CheckpointPrecheckConfig(BaseModel):
    enabled: bool = True
    require_adapter_or_merged_checkpoint: bool = True


class PrecheckConfig(BaseModel):
    enabled: bool = False
    fail_fast: bool = True
    lean_smoke_test: LeanSmokePrecheckConfig = Field(
        default_factory=LeanSmokePrecheckConfig
    )
    pantograph_smoke_test: LeanSmokePrecheckConfig = Field(
        default_factory=LeanSmokePrecheckConfig
    )
    reference_roundtrip: ReferenceRoundtripPrecheckConfig = Field(
        default_factory=ReferenceRoundtripPrecheckConfig
    )
    generation_smoke_test: GenerationSmokePrecheckConfig = Field(
        default_factory=GenerationSmokePrecheckConfig
    )
    checkpoint_check: CheckpointPrecheckConfig = Field(
        default_factory=CheckpointPrecheckConfig
    )


class MonitorConfig(BaseModel):
    enabled: bool = False
    pass_k: list[int] = Field(default_factory=lambda: [1, 4, 8])
    samples_per_statement: int = Field(default=8, gt=0)
    fixed_generation_seed: bool = True
    generation_seed: int | None = None
    evaluate_initial_checkpoint: bool = False


class BenchmarkConfig(BaseModel):
    enabled: bool = True
    run_mode: Literal["final_only", "manual"] = "final_only"
    pass_k: list[int] = Field(default_factory=lambda: [1, 4, 8])
    samples_per_statement: int = Field(default=8, gt=0)
    fixed_generation_seed: bool = True
    generation_seed: int | None = None


class RuntimeBudgetConfig(BaseModel):
    target_hours: float | None = Field(default=None, gt=0)
    hard_limit_hours: float | None = Field(default=None, gt=0)
    reduce_iteration_1_if_elapsed_hours_above: float | None = Field(default=None, ge=0)
    reduced_iteration_1_statements: int | None = Field(default=None, gt=0)
    reduce_iteration_2_if_elapsed_hours_above: float | None = Field(default=None, ge=0)
    reduced_iteration_2_statements: int | None = Field(default=None, gt=0)
    reduce_benchmark_k_if_elapsed_hours_above: float | None = Field(default=None, ge=0)
    reduced_benchmark_samples_per_statement: int | None = Field(default=None, gt=0)


class CheckpointSelectionConfig(BaseModel):
    metric: Literal[
        "latest_completed",
        "best_eval_loss",
        "best_monitor_pass_at_1",
        "best_monitor_pass_at_k",
        "best_benchmark_dev_pass_at_k",
    ] = "latest_completed"


class StoppingConfig(BaseModel):
    min_discovery_new_solved_ratio: float = Field(default=0.02, ge=0, le=1)
    max_eval_loss_degradation: float = Field(default=0.10, ge=0)
    min_monitor_improvement: float = Field(default=0.002, ge=0)
    patience: int = Field(default=2, gt=0)


class ExpertIterationConfig(BaseModel):
    run_name: str
    output_dir: str
    seed: int = 42
    data: DataConfig
    initial_model: InitialModelConfig
    iterations: IterationsConfig = Field(default_factory=IterationsConfig)
    discovery: DiscoveryConfig = Field(default_factory=DiscoveryConfig)
    verification: VerificationConfig = Field(default_factory=VerificationConfig)
    proof_bank: ProofBankConfig = Field(default_factory=ProofBankConfig)
    train_mix_first_iteration: TrainMixConfig = Field(
        default_factory=lambda: TrainMixConfig(anchor=0.65, current_expert=0.35)
    )
    train_mix: TrainMixConfig = Field(
        default_factory=lambda: TrainMixConfig(
            anchor=0.50,
            historical_expert=0.25,
            current_expert=0.25,
        )
    )
    category_sampling: CategorySamplingConfig = Field(default_factory=CategorySamplingConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)
    precheck: PrecheckConfig = Field(default_factory=PrecheckConfig)
    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    benchmark: BenchmarkConfig = Field(default_factory=BenchmarkConfig)
    runtime_budget: RuntimeBudgetConfig = Field(default_factory=RuntimeBudgetConfig)
    checkpoint_selection: CheckpointSelectionConfig = Field(default_factory=CheckpointSelectionConfig)
    stopping: StoppingConfig = Field(default_factory=StoppingConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)

    @model_validator(mode="after")
    def validate_feature_dependencies(self) -> "ExpertIterationConfig":
        runtime_profile = resolve_runtime_profile(self.runtime)
        self.verification.num_workers = runtime_profile.pantograph_workers
        self.verification.queue_maxsize = runtime_profile.max_queue_size
        self.verification.cache_backend = runtime_profile.cache_backend
        self.verification.save_full_source_on_failure_only = (
            runtime_profile.save_full_source_on_failure_only
        )
        if self.discovery.batch_statements is None:
            self.discovery.batch_statements = runtime_profile.vllm_batch_size
        if self.monitor.enabled and not self.data.monitor_path:
            raise ValueError("monitor.enabled=true requires data.monitor_path")
        if self.benchmark.enabled and not self.data.benchmark_path:
            raise ValueError("benchmark.enabled=true requires data.benchmark_path")
        metric = self.checkpoint_selection.metric
        if metric.startswith("best_monitor") and not self.monitor.enabled:
            raise ValueError(f"checkpoint selection {metric} requires monitor.enabled=true")
        if metric.startswith("best_benchmark_dev") and not self.data.benchmark_dev_path:
            raise ValueError(f"checkpoint selection {metric} requires benchmark_dev_path")
        if (
            self.initial_model.checkpoint_strategy == "fixed_anchor"
            and not self.initial_model.merged_sft0_path
        ):
            raise ValueError(
                "checkpoint_strategy=fixed_anchor requires initial_model.merged_sft0_path"
            )
        if (
            self.proof_bank.max_training_proofs_per_statement
            > self.proof_bank.max_verified_proofs_per_statement
        ):
            raise ValueError(
                "max_training_proofs_per_statement cannot exceed "
                "max_verified_proofs_per_statement"
            )
        if (
            self.discovery.hard_archive_after_rounds
            < self.discovery.max_consecutive_zero_success_rounds
        ):
            raise ValueError(
                "hard_archive_after_rounds must be >= "
                "max_consecutive_zero_success_rounds"
            )
        if self.training.load_best_model_at_end and (
            self.training.eval_strategy == "no"
            or self.training.save_strategy != self.training.eval_strategy
        ):
            raise ValueError(
                "load_best_model_at_end requires matching non-'no' "
                "training eval/save strategies"
            )
        if not self.training.use_qlora:
            raise ValueError("expert iteration currently supports QLoRA training only")
        return self

    def config_hash(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def validate_paths(self) -> None:
        paths = {
            "train": self.data.train_path,
            "eval": self.data.eval_path,
            "discovery": self.data.discovery_path,
            "monitor": self.data.monitor_path if self.monitor.enabled else None,
            "benchmark": self.data.benchmark_path if self.benchmark.enabled else None,
            "benchmark_dev": self.data.benchmark_dev_path,
            "benchmark_test": self.data.benchmark_test_path,
        }
        missing = [name for name, value in paths.items() if value and not Path(value).expanduser().exists()]
        if missing:
            raise FileNotFoundError(f"expert iteration data paths do not exist: {missing}")


def load_expert_iteration_config(path: str | Path) -> ExpertIterationConfig:
    """Load a JSON or YAML expert-iteration configuration."""

    config_path = Path(path).expanduser()
    text = config_path.read_text(encoding="utf-8-sig")
    if config_path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        try:
            import yaml
        except ImportError as error:
            raise RuntimeError("YAML configuration requires PyYAML; JSON is also supported") from error
        payload = yaml.safe_load(text)
    if not isinstance(payload, dict):
        raise ValueError(f"configuration root must be an object: {config_path}")
    return ExpertIterationConfig.model_validate(payload)
