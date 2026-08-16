"""Unified strict schemas for inference, Planner, Prover, and dataset roles."""

from __future__ import annotations

from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from lean_prover.Planner.schemas import (
    Blueprint,
    LeanFeedback,
    NodeStatus,
    PlannerResult,
    ProblemInputKind,
    RawTheoremInput,
    TheoremProblem,
    ProofAttempt,
)

from .compile_errors import LeanFailureDetail, LeanVerificationStatus


class StrictDataModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class PlannerDataStatus(StrEnum):
    SUCCESS = "success"
    FAILURE = "failure"


class ProverDataStatus(StrEnum):
    NOT_RUN = "not_run"
    SUCCESS = "success"
    FAILURE = "failure"


class DatasetStage(StrEnum):
    SFT = "sft"
    EI = "ei"
    GRPO = "grpo"
    EVALUATION = "evaluation"


class DatasetSplit(StrEnum):
    TRAIN = "train"
    EVAL = "eval"
    DISCOVERY = "discovery"
    MONITOR = "monitor"
    BENCHMARK = "benchmark"


class NormalizationFamily(StrEnum):
    LEAN_WORKBOOK = "lean_workbook"
    MINIF2F = "minif2f"
    GENERIC = "generic"


class RawInferenceData(StrictDataModel):
    """The only pre-Planner inference record: one natural or Lean problem."""

    schema_version: Literal["raw_inference_v1"] = "raw_inference_v1"
    input: RawTheoremInput
    detected_input_kind: ProblemInputKind | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class BlueprintData(StrictDataModel):
    """Planner artifact wrapper; the Blueprint payload itself is unchanged."""

    schema_version: Literal["planner_blueprint_v1"] = "planner_blueprint_v1"
    input_hash: str = Field(min_length=64, max_length=64)
    problem_hash: str = Field(pattern=r"^(?:|[0-9a-f]{64})$")
    planner_status: PlannerDataStatus
    planner_stage: str
    problem: TheoremProblem | None = None
    blueprint: Blueprint | None = None
    failed_node_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_planner_result(cls, result: PlannerResult) -> "BlueprintData":
        return cls(
            input_hash=result.input_hash,
            problem_hash=result.problem.problem_hash if result.problem else "",
            planner_status=(
                PlannerDataStatus.SUCCESS
                if result.success
                else PlannerDataStatus.FAILURE
            ),
            planner_stage=result.stage,
            problem=result.problem,
            blueprint=result.blueprint,
            failed_node_ids=result.failed_node_ids,
        )


class ProverNodeResult(StrictDataModel):
    node_id: str = Field(min_length=1)
    status: NodeStatus
    verification_status: LeanVerificationStatus
    failure_detail: LeanFailureDetail = LeanFailureDetail.NONE
    attempt_count: int = Field(ge=0)
    has_verified_proof: bool
    latest_diagnostics: str = ""
    skipped_reason: str = ""


class ProverRootResult(StrictDataModel):
    """Verified final target after all ordinary DAG nodes have been handled."""

    node_id: Literal["ROOT"] = "ROOT"
    dependencies: list[str] = Field(default_factory=list)
    target_lean_decl: str = Field(min_length=1)
    status: NodeStatus
    verification_status: LeanVerificationStatus
    failure_detail: LeanFailureDetail = LeanFailureDetail.NONE
    proof_attempts: list[ProofAttempt] = Field(default_factory=list)
    lean_feedback: list[LeanFeedback] = Field(default_factory=list)
    verified_proof: str | None = None
    latest_diagnostics: str = ""
    skipped_reason: str = ""


class ProverProblemResult(StrictDataModel):
    """One completed Prover run and its independent success/failure route."""

    schema_version: Literal["prover_problem_result_v1"] = "prover_problem_result_v1"
    input_hash: str = Field(min_length=64, max_length=64)
    problem_hash: str = Field(min_length=64, max_length=64)
    planner_status: PlannerDataStatus
    prover_status: ProverDataStatus
    all_subproblems_verified: bool
    problem: TheoremProblem
    blueprint: Blueprint
    node_results: list[ProverNodeResult]
    root_result: ProverRootResult
    failure_counts: dict[LeanVerificationStatus, int] = Field(default_factory=dict)
    attempt_failure_counts: dict[LeanVerificationStatus, int] = Field(
        default_factory=dict
    )
    failure_detail_counts: dict[LeanFailureDetail, int] = Field(
        default_factory=dict
    )
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_route(self) -> "ProverProblemResult":
        all_verified = all(
            node.status == NodeStatus.PROVED
            and node.verification_status == LeanVerificationStatus.SUCCESS
            and node.has_verified_proof
            for node in self.node_results
        )
        if self.all_subproblems_verified != all_verified:
            raise ValueError("all_subproblems_verified does not match node results")
        root_verified = (
            self.root_result.status == NodeStatus.PROVED
            and self.root_result.verification_status
            == LeanVerificationStatus.SUCCESS
            and bool(self.root_result.verified_proof)
        )
        expected = (
            ProverDataStatus.SUCCESS
            if all_verified and root_verified
            else ProverDataStatus.FAILURE
        )
        if self.prover_status != expected:
            raise ValueError(
                "prover_status does not match the subproblem-and-root gate"
            )
        if self.planner_status != PlannerDataStatus.SUCCESS:
            raise ValueError("a Prover result requires a successful Planner result")
        return self


class SFTData(StrictDataModel):
    schema_version: Literal["sft_data_v1"] = "sft_data_v1"
    data_stage: Literal[DatasetStage.SFT] = DatasetStage.SFT
    split: DatasetSplit = DatasetSplit.TRAIN
    record_id: str = Field(min_length=1)
    lean_statement: str = Field(min_length=1)
    verified_proof: str = Field(min_length=1)
    source: str = Field(min_length=1)
    normalization_family: NormalizationFamily = NormalizationFamily.GENERIC
    metadata: dict[str, Any] = Field(default_factory=dict)


class EIData(StrictDataModel):
    schema_version: Literal["ei_data_v1"] = "ei_data_v1"
    data_stage: Literal[DatasetStage.EI] = DatasetStage.EI
    split: DatasetSplit
    record_id: str = Field(min_length=1)
    lean_statement: str = Field(min_length=1)
    proof: str | None = None
    pantograph_verified: bool = False
    source: str = Field(min_length=1)
    iteration: int = Field(ge=0)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def verified_proof_contract(self) -> "EIData":
        if self.pantograph_verified and not (self.proof or "").strip():
            raise ValueError("verified EI data requires a non-empty proof")
        return self


class GRPOData(StrictDataModel):
    """GRPO input intentionally has a statement and no proof target."""

    schema_version: Literal["grpo_data_v1"] = "grpo_data_v1"
    data_stage: Literal[DatasetStage.GRPO] = DatasetStage.GRPO
    split: DatasetSplit = DatasetSplit.TRAIN
    record_id: str = Field(min_length=1)
    lean_statement: str = Field(min_length=1)
    source: str = Field(min_length=1)
    pantograph_verified: Literal["success"] = "success"
    verification_scope: Literal["statement_only"] = "statement_only"
    metadata: dict[str, Any] = Field(default_factory=dict)


class EvaluationData(StrictDataModel):
    schema_version: Literal["evaluation_data_v1"] = "evaluation_data_v1"
    data_stage: Literal[DatasetStage.EVALUATION] = DatasetStage.EVALUATION
    split: DatasetSplit = DatasetSplit.EVAL
    record_id: str = Field(min_length=1)
    lean_statement: str = Field(min_length=1)
    reference_proof: str | None = None
    source: str = Field(min_length=1)
    metadata: dict[str, Any] = Field(default_factory=dict)
