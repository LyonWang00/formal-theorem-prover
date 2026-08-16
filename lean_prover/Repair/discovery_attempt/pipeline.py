"""EI discovery API -> normalization -> dual Pantograph verification pipeline."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Protocol, Sequence

from .client import RepairClientError
from .parser import RepairParseError, normalize_repaired_proof, parse_repair_response
from .prompting import prompt_sha256
from .schema import (
    APICallAttempt,
    ProofCandidateResult,
    RepairClientResponse,
    RepairData,
    RepairRequest,
    VerificationReceipt,
)


class RepairClient(Protocol):
    @property
    def model_name(self) -> str: ...

    def repair(self, request: RepairRequest) -> RepairClientResponse: ...


class ProofVerifier(Protocol):
    def check_source(
        self, source: str, *, timeout: int | None = None, reject_forbidden: bool = True
    ) -> Any: ...


@dataclass(frozen=True)
class RepairRunResult:
    data: RepairData
    assembled_sources: tuple[str, ...] = ()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _proof_token_count(proof: str) -> int:
    return len(re.findall(r"[A-Za-z_][\w']*|\S", proof, flags=re.UNICODE))


def assemble_repaired_source(
    request: RepairRequest, proof: str, *, include_imports: bool = False
) -> str:
    import_block = "\n".join(
        line if str(line).strip().startswith("import ") else f"import {line}"
        for line in request.imports
    )
    context = "\n".join(request.context_lines).strip()
    theorem = request.lean_statement.strip()
    declaration = f"{theorem} {proof}" if theorem.endswith(":=") else f"{theorem} := {proof}"
    pieces = [import_block] if include_imports else []
    if context:
        pieces.append(context)
    pieces.append(declaration)
    return "\n\n".join(pieces) + "\n"


def _receipt(result: Any, source: str, environment_hash: str) -> VerificationReceipt:
    success = bool(getattr(result, "success", False))
    return VerificationReceipt(
        status="verified" if success else str(getattr(result, "error_type", None) or "failed"),
        verified=success,
        check_seconds=float(getattr(result, "check_seconds", 0.0)),
        timed_out=bool(getattr(result, "timed_out", False)),
        diagnostics=str(getattr(result, "diagnostics", "") or ""),
        errors=[str(x) for x in (getattr(result, "errors", ()) or ())],
        warnings=[str(x) for x in (getattr(result, "warnings", ()) or ())],
        assembled_source_sha256=_sha256_text(source),
        environment_hash=environment_hash,
    )


class DiscoveryAttemptRepairPipeline:
    def __init__(self, *, client: RepairClient, verifier: ProofVerifier, timeout: int = 60) -> None:
        self.client = client
        self.verifier = verifier
        self.timeout = timeout

    def _base(self, request: RepairRequest, api_calls: Sequence[APICallAttempt]) -> dict[str, Any]:
        return {
            "source": request.source,
            "theorem_name": request.theorem_name,
            "lean_statement": request.lean_statement,
            "statement_id": request.statement_id,
            "aggregation_group_id": request.aggregation_group_id,
            "target_attempt_id": request.target_attempt_id,
            "target_attempt_rank": request.target_attempt_rank,
            "attempt_fingerprint": request.attempt_fingerprint,
            "peer_attempt_ids": [row.attempt_id for row in request.aggregated_attempts],
            "target_failure_proof": request.target_failure_proof,
            "target_error_message": request.target_error_message,
            "target_failure_class": request.target_failure_class,
            "target_proof_state": request.target_proof_state,
            "api_call_attempts": list(api_calls),
            "api_retry_count": max(0, len(api_calls) - 1),
            "environment": request.environment,
            "prompt_sha256": prompt_sha256(request),
            "model_name": self.client.model_name,
            "source_metadata": request.source_metadata,
        }

    def _verify_candidate(
        self, request: RepairRequest, raw_proof: str, strategy: str
    ) -> tuple[ProofCandidateResult, str]:
        normalized = normalize_repaired_proof(raw_proof)
        source = assemble_repaired_source(request, normalized.proof)
        result = self.verifier.check_source(
            source, timeout=self.timeout, reject_forbidden=True
        )
        receipt = _receipt(result, source, request.environment.environment_hash)
        return (
            ProofCandidateResult(
                strategy=strategy,
                raw_proof=raw_proof,
                normalized_proof=normalized.proof,
                normalization_actions=list(normalized.actions),
                proof_sha256=_sha256_text(normalized.proof),
                proof_character_count=len(normalized.proof),
                proof_token_count=_proof_token_count(normalized.proof),
                parse_success=True,
                verification=receipt,
            ),
            source,
        )

    def run_one(
        self,
        request: RepairRequest,
        *,
        api_call_attempts: Sequence[APICallAttempt] = (),
    ) -> RepairRunResult:
        base = self._base(request, api_call_attempts)
        try:
            client_result = self.client.repair(request)
        except RepairClientError as error:
            anomaly = (
                "api_empty_response_after_retry"
                if error.kind == "empty_response" and len(api_call_attempts) >= 2
                else "api_error"
            )
            return RepairRunResult(
                RepairData(
                    **base,
                    anomaly_status=anomaly,
                    failure_stage="api",
                    failure_reason=str(error),
                )
            )

        raw = client_result.raw_content
        try:
            proposal = parse_repair_response(raw, expected_attempt_id=request.target_attempt_id)
        except RepairParseError as error:
            return RepairRunResult(
                RepairData(
                    **base,
                    api_success=True,
                    raw_api_response=raw,
                    anomaly_status="response_parse_error",
                    failure_stage="parse",
                    failure_reason=str(error),
                )
            )

        original_source = assemble_repaired_source(request, request.target_failure_proof)
        original_result = self.verifier.check_source(
            original_source, timeout=self.timeout, reject_forbidden=True
        )
        original_receipt = _receipt(
            original_result, original_source, request.environment.environment_hash
        )

        minimal, minimal_source = self._verify_candidate(
            request, proposal.minimal_repair_proof, "minimal_repair"
        )
        clean, clean_source = self._verify_candidate(
            request, proposal.clean_proof, "clean_from_scratch"
        )

        if original_receipt.verified:
            resolved = "already_valid"
            resolution_reason = "The original attempt compiled in the frozen local environment."
        elif request.target_failure_class == "unsolved_goals":
            resolved = "incomplete_completable"
            resolution_reason = "Local source taxonomy reports unsolved goals."
        else:
            resolved = proposal.attempt_assessment
            resolution_reason = "Resolved from the model assessment plus the reproduced local failure."

        verified = [row for row in (minimal, clean) if row.verification.verified]
        # Shortest token count is primary; characters break ties. A clean proof
        # wins an exact tie so inherited failed syntax is not preferred by default.
        verified.sort(
            key=lambda row: (
                row.proof_token_count,
                row.proof_character_count,
                0 if row.strategy == "clean_from_scratch" else 1,
            )
        )
        if original_receipt.verified:
            selected = None
            anomaly = "original_attempt_already_valid"
            failure_stage = "input"
            failure_reason = "original attempt unexpectedly passed current Pantograph verification"
        else:
            selected = verified[0] if verified else None
            anomaly = "none" if selected else "generated_proofs_failed_verification"
            failure_stage = "none" if selected else "verification"
            failure_reason = "" if selected else "neither generated proof passed local Pantograph"

        data = RepairData(
            **base,
            original_attempt_verification=original_receipt,
            model_attempt_assessment=proposal.attempt_assessment,
            resolved_attempt_assessment=resolved,
            assessment_evidence=proposal.assessment_evidence,
            assessment_resolution_reason=resolution_reason,
            minimal_repair=minimal,
            clean_repair=clean,
            selected_verified_proof=selected.normalized_proof if selected else "",
            selected_strategy=selected.strategy if selected else "none",
            selection_reason=(
                "Selected the shortest Pantograph-verified normalized proof by token count, then character count."
                if selected else "No proof was promoted."
            ),
            repair_verified=bool(selected),
            api_success=True,
            response_parse_success=True,
            raw_api_response=raw,
            anomaly_status=anomaly,
            failure_stage=failure_stage,
            failure_reason=failure_reason,
        )
        return RepairRunResult(
            data,
            assembled_sources=(original_source, minimal_source, clean_source),
        )


# Historical name retained for existing experiment scripts.
RepairPipeline = DiscoveryAttemptRepairPipeline
