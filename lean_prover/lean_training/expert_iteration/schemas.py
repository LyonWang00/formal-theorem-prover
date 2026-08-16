"""Typed records and stable identifiers for expert iteration artifacts."""

from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

from lean_prover.Data.compile_errors import LeanVerificationStatus
from lean_prover.lean_training.data.preparation import statement_hash as lean_statement_hash
from lean_prover.lean_training.data.training import build_generation_prompt


class DataRole(StrEnum):
    TRAIN = "train"
    EVAL = "eval"
    DISCOVERY = "discovery"
    MONITOR = "monitor"
    BENCHMARK_DEV = "benchmark_dev"
    BENCHMARK_TEST = "benchmark_test"
    BENCHMARK = "benchmark"


VerificationStatus = LeanVerificationStatus


class DiscoveryBucket(StrEnum):
    NEW = "new"
    FRONTIER = "frontier"
    UNSOLVED = "unsolved"
    SOLVED_EASY = "solved_easy"
    HARD_ARCHIVE = "hard_archive"


class IterationStage(StrEnum):
    # Global gates that precede the resumable per-iteration loop.  They are
    # represented in the same state vocabulary so reports can describe the
    # complete RAW -> clean M0 -> expert-iteration lifecycle.
    DATA_AUDIT = "data_audit"
    DATA_BUILD = "data_build"
    PRECHECK = "precheck"
    TRAIN_INITIAL = "train_initial"
    CHECKPOINT_VERIFY = "checkpoint_verify"
    TRAIN_MEMORIZATION = "train_memorization"
    DISCOVERY_BOOTSTRAP = "discovery_bootstrap"

    # Canonical short names alias the original serialized values.  Existing
    # run state therefore remains readable and old single-stage CLI calls keep
    # working while new reports can use the requested lifecycle terminology.
    SELECT_POOL = "select_discovery_pool"
    SELECT_DISCOVERY_POOL = "select_discovery_pool"
    GENERATE = "generate_discovery"
    GENERATE_DISCOVERY = "generate_discovery"
    VERIFY = "verify_discovery"
    VERIFY_DISCOVERY = "verify_discovery"
    UPDATE_BANKS = "update_banks"
    BUILD_DATASET = "build_train_dataset"
    BUILD_TRAIN_DATASET = "build_train_dataset"
    TRAIN_EXPERT = "train"
    TRAIN = "train"
    EVAL = "eval"
    MONITOR = "monitor"
    FINALIZE = "finalize_iteration"
    FINALIZE_ITERATION = "finalize_iteration"
    BENCHMARK = "benchmark"


FULL_PIPELINE_STAGE_ORDER = (
    IterationStage.DATA_AUDIT,
    IterationStage.DATA_BUILD,
    IterationStage.PRECHECK,
    IterationStage.TRAIN_INITIAL,
    IterationStage.CHECKPOINT_VERIFY,
    IterationStage.TRAIN_MEMORIZATION,
    IterationStage.DISCOVERY_BOOTSTRAP,
    IterationStage.SELECT_POOL,
    IterationStage.GENERATE,
    IterationStage.VERIFY,
    IterationStage.UPDATE_BANKS,
    IterationStage.BUILD_DATASET,
    IterationStage.TRAIN_EXPERT,
    IterationStage.EVAL,
    IterationStage.MONITOR,
    IterationStage.FINALIZE,
    IterationStage.BENCHMARK,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def normalize_statement(statement: str) -> str:
    """Normalize whitespace and Lean comments for stable overlap hashes."""

    without_block = re.sub(r"/-.*?-/", " ", statement, flags=re.DOTALL)
    without_line = re.sub(r"--[^\n]*", " ", without_block)
    return re.sub(r"\s+", " ", without_line).strip()


def normalized_proof(proof: str) -> str:
    text = proof.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip() == "```":
            lines.pop()
        text = "\n".join(lines)
    return re.sub(r"\s+", " ", text).strip()


def stable_hash(*parts: object) -> str:
    payload = "\0".join(str(part) for part in parts)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def stable_statement_id(source: str, source_id: str | None, statement: str) -> str:
    identity = source_id.strip() if source_id and source_id.strip() else normalize_statement(statement)
    return f"stmt_{stable_hash(source, identity)[:24]}"


class StatementRecord(BaseModel):
    statement_id: str
    source_id: str | None = None
    source: str
    data_role: DataRole
    category: str | None = None
    static_difficulty: str | int | float | None = None
    imports: list[str] = Field(default_factory=list)
    namespace: str | None = None
    context: str | None = None
    context_lines: list[str] = Field(default_factory=list)
    statement: str
    reference_proof: str | None = None
    statement_hash: str
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_prepared(cls, row: dict[str, Any], role: DataRole) -> "StatementRecord":
        declared_role = row.get("data_role")
        if declared_role is not None and declared_role != role.value:
            raise ValueError(
                f"prepared row declares data_role={declared_role!r}, loaded as {role.value!r}"
            )
        statement = str(row.get("lean_statement") or row.get("statement") or "").strip()
        if not statement:
            raise ValueError(f"prepared {role} row has no lean_statement: id={row.get('id')}")
        source = str(row.get("source_name") or row.get("source") or "unknown")
        source_id = str(row.get("id") or "") or None
        proof = row.get("proof") or row.get("completion") or row.get("reference_proof")
        return cls(
            statement_id=stable_statement_id(source, source_id, statement),
            source_id=source_id,
            source=source,
            data_role=role,
            category=str(row.get("category") or "unknown"),
            static_difficulty=row.get("difficulty"),
            imports=list(row.get("imports") or (row.get("preamble") or {}).get("imports") or []),
            namespace=row.get("namespace"),
            context=row.get("context"),
            context_lines=list(row.get("context_lines") or (row.get("preamble") or {}).get("context_lines") or []),
            statement=statement,
            reference_proof=str(proof).strip() if proof else None,
            statement_hash=lean_statement_hash(statement),
            metadata={
                **{
                    key: value
                    for key, value in row.items()
                    if key
                    not in {
                        "proof",
                        "completion",
                        "reference_proof",
                        "text",
                        "prompt",
                    }
                },
                **(
                    {"frozen_generation_prompt": str(row["prompt"])}
                    if str(row.get("prompt") or "").strip()
                    else {}
                ),
            },
        )

    def generation_prompt(self) -> str:
        """Build a role-safe prompt that never reads ``reference_proof``."""

        frozen = str(self.metadata.get("frozen_generation_prompt") or "")
        if frozen.strip():
            return frozen
        informal = str(self.metadata.get("informal_statement") or "").strip()
        return build_generation_prompt(
            {
                "informal_statement": informal,
                "lean_statement": self.statement,
            }
        )


class GenerationRecord(BaseModel):
    generation_id: str
    statement_id: str
    data_role: DataRole
    iteration: int | None
    checkpoint: str
    sample_index: int
    prompt: str | None = None
    raw_output: str
    extracted_proof: str | None = None
    normalized_proof: str | None = None
    proof_format: str | None = None
    finish_reason: Literal["stop", "eos", "length", "abort", "error"] | None = None
    generation_seed: int | None = None
    temperature: float
    top_p: float
    max_new_tokens: int
    created_at: str = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class VerificationRecord(BaseModel):
    generation_id: str
    statement_id: str
    data_role: DataRole
    iteration: int | None
    verified: bool
    status: VerificationStatus
    error_type: str | None = None
    error_message: str | None = None
    first_error_position: int | None = None
    valid_prefix_length: int | None = None
    remaining_goals: list[str] | None = None
    compile_time_ms: int | None = None
    timed_out: bool = False
    contains_sorry: bool = False
    contains_admit: bool = False
    contains_axiom: bool = False
    lean_version: str = "unknown"
    mathlib_commit: str | None = None
    environment_hash: str
    assembler_version: str | None = None
    normalization_version: str | None = None
    statement: str | None = None
    normalized_proof: str | None = None
    proof_format: str | None = None
    assembled_source: str | None = None
    assembled_source_hash: str | None = None
    imports: list[str] = Field(default_factory=list)
    namespace: str | None = None
    context: str | None = None
    context_lines: list[str] = Field(default_factory=list)
    verified_at: str = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class ProofBankRecord(BaseModel):
    proof_id: str
    statement_id: str
    proof: str
    iteration_found: int
    generator_checkpoint: str
    generation_id: str
    proof_tokens: int
    proof_chars: int
    compile_time_ms: int | None = None
    tactic_signature: list[str] | None = None
    selected_as_primary: bool = False
    environment_hash: str
    lean_version: str = "unknown"
    mathlib_commit: str | None = None
    needs_reverification: bool = False
    created_at: str = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class FailureBankRecord(BaseModel):
    failure_id: str
    generation_id: str
    statement_id: str
    iteration: int
    generator_checkpoint: str
    raw_output: str
    extracted_proof: str | None = None
    status: VerificationStatus
    error_type: str | None = None
    error_message: str | None = None
    first_error_position: int | None = None
    valid_prefix_length: int | None = None
    remaining_goals: list[str] | None = None
    timed_out: bool = False
    environment_hash: str
    created_at: str = Field(default_factory=utc_now)
    metadata: dict[str, Any] = Field(default_factory=dict)


class DiscoveryStatementState(BaseModel):
    statement_id: str
    attempted_iterations: list[int] = Field(default_factory=list)
    total_attempts: int = 0
    total_successes: int = 0
    last_iteration_attempted: int | None = None
    last_success_count: int = 0
    last_sample_count: int = 0
    consecutive_zero_success_rounds: int = 0
    ever_solved: bool = False
    first_solved_iteration: int | None = None
    last_solved_iteration: int | None = None
    current_bucket: DiscoveryBucket = DiscoveryBucket.NEW
    audit_count: int = 0
    proof_bank_ids: list[str] = Field(default_factory=list)

    def update(
        self,
        *,
        iteration: int,
        sample_count: int,
        success_count: int,
        hard_archive_after_rounds: int,
        solved_easy_success_ratio: float,
    ) -> None:
        retrying_iteration = (
            iteration in self.attempted_iterations
            and self.last_iteration_attempted == iteration
        )
        if retrying_iteration:
            self.total_attempts -= self.last_sample_count
            self.total_successes -= self.last_success_count
            if self.last_success_count == 0 and self.consecutive_zero_success_rounds > 0:
                self.consecutive_zero_success_rounds -= 1
        else:
            self.attempted_iterations.append(iteration)
        self.total_attempts += sample_count
        self.total_successes += success_count
        self.last_iteration_attempted = iteration
        self.last_success_count = success_count
        self.last_sample_count = sample_count
        if success_count == 0:
            self.consecutive_zero_success_rounds += 1
        else:
            self.consecutive_zero_success_rounds = 0
            self.ever_solved = True
            if self.first_solved_iteration is None:
                self.first_solved_iteration = iteration
            self.last_solved_iteration = iteration
        ratio = success_count / sample_count if sample_count else 0
        if self.consecutive_zero_success_rounds >= hard_archive_after_rounds:
            self.current_bucket = DiscoveryBucket.HARD_ARCHIVE
        elif success_count == 0:
            self.current_bucket = DiscoveryBucket.UNSOLVED
        elif ratio >= solved_easy_success_ratio:
            self.current_bucket = DiscoveryBucket.SOLVED_EASY
        else:
            self.current_bucket = DiscoveryBucket.FRONTIER


class IterationState(BaseModel):
    iteration: int
    status: str = "running"
    current_stage: IterationStage = IterationStage.SELECT_DISCOVERY_POOL
    discovery_checkpoint: str
    train_initialization_checkpoint: str
    output_checkpoint: str | None = None
    config_hash: str
    eval_dataset_hash: str
    precheck_completed: bool = False
    pool_selected: bool = False
    generation_completed: bool = False
    verification_completed: bool = False
    banks_updated: bool = False
    train_dataset_built: bool = False
    training_completed: bool = False
    training_skipped: bool = False
    eval_completed: bool = False
    monitor_completed: bool = False
    benchmark_completed: bool = False
    started_at: str = Field(default_factory=utc_now)
    updated_at: str = Field(default_factory=utc_now)
    metrics: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_iteration(self) -> "IterationState":
        if self.iteration < 0:
            raise ValueError("iteration must be non-negative")
        return self
