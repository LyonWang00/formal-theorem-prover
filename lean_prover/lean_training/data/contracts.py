"""Versioned contracts for audited Lean training data and attestations."""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, model_validator

from .preparation import ProofFormat


class DataState(StrEnum):
    RAW = "raw"
    VERIFIED = "verified"
    QUARANTINED = "quarantined"


class SourceSpan(BaseModel):
    start_row: int | None = None
    end_row: int | None = None
    start_line: int | None = None
    end_line: int | None = None


def make_attestation_id(
    *,
    record_id: str,
    environment_hash: str,
    assembler_version: str,
    normalization_version: str,
    assembled_source_hash: str,
) -> str:
    """Bind a proof attestation to its source, environment, and assemblers."""

    payload = "|".join(
        (
            record_id,
            environment_hash,
            assembler_version,
            normalization_version,
            assembled_source_hash,
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class LeanDataRecord(BaseModel):
    """Self-describing Lean record whose attestation is environment-bound."""

    record_id: str
    statement_id: str
    data_state: DataState = DataState.RAW

    source_dataset: str
    # Public aliases used by the context-aware cross-dataset schema.  Existing
    # audited artifacts keep record_id/source_dataset for backward compatibility.
    id: str | None = None
    source: str | None = None
    source_file: str | None = None
    source_module: str | None = None
    source_declaration: str | None = None
    source_commit: str | None = None
    source_span: SourceSpan | None = None

    imports: list[str] = Field(default_factory=list)
    namespace: str | None = None
    namespaces: list[str] = Field(default_factory=list)
    open_namespaces: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    variables: list[dict[str, Any]] = Field(default_factory=list)
    hypotheses: list[dict[str, Any]] = Field(default_factory=list)
    local_instances: list[str] = Field(default_factory=list)
    open_declarations: list[str] = Field(default_factory=list)
    open_scoped_declarations: list[str] = Field(default_factory=list)
    section_context: str | dict[str, Any] | None = None
    variable_context: str | None = None
    local_context: str | None = None
    local_notations: list[str] = Field(default_factory=list)
    local_attributes: list[str] = Field(default_factory=list)
    context_recovered: bool = False
    context_recovery_method: str = ""
    context_recovery_status: str = "unavailable"
    context_recovery_sources: list[str] = Field(default_factory=list)
    context_warnings: list[str] = Field(default_factory=list)

    informal_statement: str = ""
    statement: str
    proof: str | None = None
    proof_format: ProofFormat = ProofFormat.FULL_PROOF
    raw_declaration: str | None = None
    raw_source_context: str | None = None

    statement_verified: bool = False
    proof_verified: bool = False
    pantograph_verified: bool = False
    reference_proof_verified: bool = False

    verification_status: str = "raw"
    verification_error_type: str | None = None
    verification_error_message: str | None = None

    lean_version: str | None = None
    mathlib_commit: str | None = None
    environment_hash: str | None = None
    assembler_version: str
    normalization_version: str

    attestation_id: str | None = None
    attested_at: str | None = None
    assembled_source_hash: str | None = None

    context_recovery_version: str | None = None
    recovered_source_hash: str | None = None
    migration_applied: bool = False
    migration_rules: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_state_contract(self) -> "LeanDataRecord":
        if self.context_recovery_status not in {"full", "partial", "unavailable"}:
            raise ValueError(
                "context_recovery_status must be full, partial, or unavailable"
            )
        if self.context_recovered and self.context_recovery_status != "full":
            raise ValueError(
                "context_recovered=true requires context_recovery_status=full"
            )
        if self.proof_verified and not self.statement_verified:
            raise ValueError("proof_verified=true requires statement_verified=true")
        if self.data_state is DataState.VERIFIED:
            required = {
                "statement_verified": self.statement_verified,
                "pantograph_verified": self.pantograph_verified,
                "environment_hash": bool(self.environment_hash),
                "attestation_id": bool(self.attestation_id),
                "assembled_source_hash": bool(self.assembled_source_hash),
            }
            missing = [key for key, value in required.items() if not value]
            if missing:
                raise ValueError(f"verified record lacks attestation fields: {missing}")
        if self.data_state is DataState.QUARANTINED and self.proof_verified:
            raise ValueError("quarantined records cannot be proof_verified")
        return self

    def attestation_is_current(
        self,
        *,
        environment_hash: str,
        assembler_version: str,
        normalization_version: str,
        assembled_source_hash: str | None = None,
    ) -> bool:
        return bool(
            self.attestation_id
            and self.pantograph_verified
            and self.environment_hash == environment_hash
            and self.assembler_version == assembler_version
            and self.normalization_version == normalization_version
            and (
                assembled_source_hash is None
                or self.assembled_source_hash == assembled_source_hash
            )
        )

    def can_enter_sft(self) -> bool:
        return bool(
            self.data_state is DataState.VERIFIED
            and self.statement_verified
            and self.proof_verified
            and self.pantograph_verified
            and self.proof
            and self.attestation_id
        )

    def deterministic_context_lines(self) -> tuple[str, ...]:
        """Render only source-backed context fields; never infer missing context."""

        lines: list[str] = []
        namespaces = self.namespaces or ([self.namespace] if self.namespace else [])
        for namespace in namespaces:
            lines.append(f"namespace {namespace}")
        if not namespaces and self.namespace:
            lines.append(f"namespace {self.namespace}")
        open_namespaces = self.open_namespaces or self.open_declarations
        lines.extend(f"open {name}" for name in open_namespaces)
        scopes = self.scopes or self.open_scoped_declarations
        lines.extend(
            f"open scoped {name}" for name in scopes
        )
        section_block = (
            self.section_context
            if isinstance(self.section_context, str)
            else None
        )
        for block in (section_block, self.variable_context, self.local_context):
            if block:
                lines.extend(block.splitlines())
        lines.extend(self.local_notations)
        lines.extend(self.local_attributes)
        return tuple(line for line in lines if line.strip())


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
