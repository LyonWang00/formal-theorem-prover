"""Strict schemas for the versioned EI discovery RepairData pipeline."""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


RepairAssessment = Literal[
    "fully_incorrect",
    "partially_incorrect",
    "incomplete_completable",
]
ResolvedRepairAssessment = Literal[
    "already_valid",
    "fully_incorrect",
    "partially_incorrect",
    "incomplete_completable",
    "indeterminate",
]


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class EnvironmentContract(StrictModel):
    """Exact Lean/Pantograph environment used in prompts and verification."""

    schema_version: Literal["lean_environment_contract_v1"] = "lean_environment_contract_v1"
    lean_version: str
    lean_commit: str
    mathlib_commit: str
    pantograph_version: str
    imports: list[str] = Field(default_factory=lambda: ["Mathlib"])
    lean_toolchain_sha256: str
    lake_manifest_sha256: str
    repository_commit: str = "unknown"
    environment_hash: str = ""

    @field_validator("lean_version", "lean_commit", "mathlib_commit", "pantograph_version")
    @classmethod
    def exact_versions_are_required(cls, value: str) -> str:
        if not value or value.lower() in {"unknown", "current", "latest"}:
            raise ValueError("exact Lean, Mathlib, and Pantograph versions are required")
        return value

    @model_validator(mode="after")
    def validate_environment_hash(self) -> "EnvironmentContract":
        expected = canonical_sha256(self.model_dump(exclude={"environment_hash"}, mode="json"))
        if self.environment_hash and self.environment_hash != expected:
            raise ValueError("environment_hash does not match the environment contract")
        self.environment_hash = expected
        return self


class AggregatedAttempt(StrictModel):
    """One failed candidate included in a statement-level API context."""

    attempt_id: str
    attempt_rank: int | None = None
    failure_proof: str
    error_message: str
    failure_class: str
    proof_state: str = ""
    original_verification_status: str = "failed"

    @field_validator("attempt_id", "failure_proof", "error_message", "failure_class")
    @classmethod
    def non_empty_fields(cls, value: str) -> str:
        if not value:
            raise ValueError("aggregated attempt fields must not be empty")
        return value


class RepairRequest(StrictModel):
    """One target attempt plus all failed attempts from the same statement."""

    schema_version: Literal["repair_request_v2"] = "repair_request_v2"
    source: str
    theorem_name: str = ""
    lean_statement: str
    target_attempt_id: str
    target_attempt_rank: int | None = None
    target_failure_proof: str
    target_error_message: str
    target_failure_class: str
    target_proof_state: str = ""
    statement_id: str
    aggregation_group_id: str
    aggregated_attempts: list[AggregatedAttempt]
    attempt_fingerprint: str
    context_lines: list[str] = Field(default_factory=list)
    environment: EnvironmentContract
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def require_consistent_group(self) -> "RepairRequest":
        for field in (
            "source", "lean_statement", "target_attempt_id", "target_failure_proof",
            "target_error_message", "target_failure_class", "statement_id",
            "aggregation_group_id", "attempt_fingerprint",
        ):
            if not str(getattr(self, field)).strip():
                raise ValueError(f"{field} must not be empty")
        ids = [row.attempt_id for row in self.aggregated_attempts]
        if len(ids) != len(set(ids)):
            raise ValueError("aggregated_attempts contains duplicate attempt_id values")
        if self.target_attempt_id not in ids:
            raise ValueError("target attempt must be present in aggregated_attempts")
        return self

    @property
    def imports(self) -> list[str]:
        return self.environment.imports


class RepairProposal(StrictModel):
    """Strict JSON object returned by the repair model."""

    schema_version: Literal["repair_api_response_v2"]
    target_attempt_id: str
    attempt_assessment: RepairAssessment
    assessment_evidence: str
    minimal_repair_proof: str
    minimal_repair_change_summary: str
    clean_proof: str

    @field_validator(
        "target_attempt_id", "assessment_evidence", "minimal_repair_proof",
        "minimal_repair_change_summary", "clean_proof",
    )
    @classmethod
    def response_fields_are_non_empty(cls, value: str) -> str:
        if not value:
            raise ValueError("repair response fields must not be empty")
        return value


class APIObservation(StrictModel):
    provider: str
    base_url: str
    model: str
    request_id: str = ""
    finish_reason: str = ""
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    reasoning_tokens: int | None = None
    total_tokens: int | None = None
    latency_ms: int
    response_content_empty: bool
    reasoning_content_present: bool = False
    response_schema: str = "repair_api_response_v2"


class APICallAttempt(StrictModel):
    call_index: int
    success: bool
    error_kind: str = ""
    error_message: str = ""
    observation: APIObservation | None = None


class RepairClientResponse(StrictModel):
    raw_content: str
    observation: APIObservation


class VerificationReceipt(StrictModel):
    status: str
    verified: bool
    check_seconds: float
    timed_out: bool = False
    diagnostics: str = ""
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    assembled_source_sha256: str
    environment_hash: str


class ProofCandidateResult(StrictModel):
    strategy: Literal["minimal_repair", "clean_from_scratch"]
    raw_proof: str
    normalized_proof: str
    normalization_actions: list[str] = Field(default_factory=list)
    proof_sha256: str
    proof_character_count: int
    proof_token_count: int
    parse_success: bool
    verification: VerificationReceipt


class RepairData(StrictModel):
    """One auditable attempt-level repair record. No ambiguous proof aliases."""

    schema_version: Literal["repair_data_v2"] = "repair_data_v2"
    source: str
    theorem_name: str = ""
    lean_statement: str
    statement_id: str
    aggregation_group_id: str
    target_attempt_id: str
    target_attempt_rank: int | None = None
    attempt_fingerprint: str
    peer_attempt_ids: list[str]
    target_failure_proof: str
    target_error_message: str
    target_failure_class: str
    target_proof_state: str = ""
    original_attempt_verification: VerificationReceipt | None = None
    model_attempt_assessment: RepairAssessment | None = None
    resolved_attempt_assessment: ResolvedRepairAssessment = "indeterminate"
    assessment_evidence: str = ""
    assessment_resolution_reason: str = ""
    minimal_repair: ProofCandidateResult | None = None
    clean_repair: ProofCandidateResult | None = None
    selected_verified_proof: str = ""
    selected_strategy: Literal["minimal_repair", "clean_from_scratch", "none"] = "none"
    selection_reason: str = ""
    repair_verified: bool = False
    api_success: bool = False
    response_parse_success: bool = False
    raw_api_response: str = ""
    api_call_attempts: list[APICallAttempt] = Field(default_factory=list)
    api_retry_count: int = 0
    anomaly_status: Literal[
        "none", "api_empty_response_after_retry", "api_error",
        "response_parse_error", "generated_proofs_failed_verification",
        "original_attempt_already_valid",
    ] = "none"
    failure_stage: Literal["none", "api", "parse", "verification", "input"] = "none"
    failure_reason: str = ""
    environment: EnvironmentContract
    prompt_version: Literal["repair_prompt_v2"] = "repair_prompt_v2"
    prompt_sha256: str
    model_name: str
    source_metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def enforce_promotion_contract(self) -> "RepairData":
        if self.api_retry_count > 1:
            raise ValueError("API retries must be capped at one")
        if self.repair_verified:
            if not self.api_success or not self.response_parse_success:
                raise ValueError("verified repair requires successful API and response parsing")
            if not self.selected_verified_proof or self.selected_strategy == "none":
                raise ValueError("verified repair requires a selected proof and strategy")
            selected = self.minimal_repair if self.selected_strategy == "minimal_repair" else self.clean_repair
            if selected is None or not selected.verification.verified:
                raise ValueError("selected repair must have a successful Pantograph receipt")
            if selected.normalized_proof != self.selected_verified_proof:
                raise ValueError("selected proof must equal the selected verified candidate")
            if self.failure_stage != "none" or self.anomaly_status != "none":
                raise ValueError("verified repair cannot carry a failure/anomaly status")
        elif self.selected_verified_proof or self.selected_strategy != "none":
            raise ValueError("unverified records cannot promote a proof")
        return self
