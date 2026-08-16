"""Problem-level Prover result aggregation and success/failure persistence."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Protocol

from lean_prover.Data import (
    LeanFailureDetail,
    LeanVerificationStatus,
    PlannerDataStatus,
    ProverDataStatus,
    ProverNodeResult,
    ProverProblemResult,
    ProverRootResult,
)
from lean_prover.Planner.schemas import (
    Blueprint,
    BlueprintNode,
    NodeStatus,
    TheoremProblem,
)


class ProverResultStore(Protocol):
    def record_result(self, result: ProverProblemResult) -> None:
        ...


class JsonlProverResultStore:
    """Route complete problem records into independent success/failure banks."""

    def __init__(
        self,
        *,
        success_path: str | Path = "outputs/prover/success.jsonl",
        failure_path: str | Path = "outputs/prover/failure.jsonl",
    ) -> None:
        self.success_path = Path(success_path)
        self.failure_path = Path(failure_path)

    def record_result(self, result: ProverProblemResult) -> None:
        path = (
            self.success_path
            if result.prover_status == ProverDataStatus.SUCCESS
            else self.failure_path
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(result.model_dump_json() + "\n")


def _enum_or_default(enum_type, value: str, default):
    try:
        return enum_type(value)
    except ValueError:
        return default


def collect_prover_result(
    *,
    problem: TheoremProblem,
    blueprint: Blueprint,
) -> ProverProblemResult:
    """Apply the all-subproblems gate independently of Planner success."""

    root_payload = blueprint.metadata.get("root_node")
    if not isinstance(root_payload, dict):
        raise ValueError("proved Blueprint is missing the ROOT proof result")
    root = BlueprintNode.model_validate(root_payload)

    node_results: list[ProverNodeResult] = []
    for node in blueprint.nodes:
        latest_attempt = node.proof_attempts[-1] if node.proof_attempts else None
        latest_feedback = node.lean_feedback[-1] if node.lean_feedback else None
        if node.status == NodeStatus.PROVED and node.verified_proof:
            verification_status = LeanVerificationStatus.SUCCESS
            failure_detail = LeanFailureDetail.NONE
        else:
            raw_status = (
                latest_attempt.error_type
                if latest_attempt is not None
                else (
                    latest_feedback.error_type
                    if latest_feedback is not None
                    else LeanVerificationStatus.INTERNAL_ERROR.value
                )
            )
            raw_detail = (
                latest_attempt.error_detail
                if latest_attempt is not None
                else (
                    latest_feedback.error_detail
                    if latest_feedback is not None
                    else LeanFailureDetail.INTERNAL_EXCEPTION.value
                )
            )
            verification_status = _enum_or_default(
                LeanVerificationStatus,
                raw_status,
                LeanVerificationStatus.INTERNAL_ERROR,
            )
            failure_detail = _enum_or_default(
                LeanFailureDetail,
                raw_detail,
                LeanFailureDetail.INTERNAL_EXCEPTION,
            )
        diagnostics = (
            latest_attempt.diagnostics
            if latest_attempt is not None
            else (latest_feedback.diagnostics if latest_feedback is not None else "")
        )
        node_results.append(
            ProverNodeResult(
                node_id=node.id,
                status=node.status,
                verification_status=verification_status,
                failure_detail=failure_detail,
                attempt_count=len(node.proof_attempts),
                has_verified_proof=bool(node.verified_proof),
                latest_diagnostics=diagnostics,
                skipped_reason=str(node.metadata.get("skipped_reason") or ""),
            )
        )

    all_verified = all(
        node.status == NodeStatus.PROVED
        and node.verification_status == LeanVerificationStatus.SUCCESS
        and node.has_verified_proof
        for node in node_results
    )
    failures = Counter(
        node.verification_status
        for node in node_results
        if node.verification_status != LeanVerificationStatus.SUCCESS
    )
    root_latest_attempt = (
        root.proof_attempts[-1] if root.proof_attempts else None
    )
    root_latest_feedback = (
        root.lean_feedback[-1] if root.lean_feedback else None
    )
    if root.status == NodeStatus.PROVED and root.verified_proof:
        root_verification_status = LeanVerificationStatus.SUCCESS
        root_failure_detail = LeanFailureDetail.NONE
    else:
        root_verification_status = _enum_or_default(
            LeanVerificationStatus,
            (
                root_latest_attempt.error_type
                if root_latest_attempt is not None
                else (
                    root_latest_feedback.error_type
                    if root_latest_feedback is not None
                    else LeanVerificationStatus.INTERNAL_ERROR.value
                )
            ),
            LeanVerificationStatus.INTERNAL_ERROR,
        )
        root_failure_detail = _enum_or_default(
            LeanFailureDetail,
            (
                root_latest_attempt.error_detail
                if root_latest_attempt is not None
                else (
                    root_latest_feedback.error_detail
                    if root_latest_feedback is not None
                    else LeanFailureDetail.INTERNAL_EXCEPTION.value
                )
            ),
            LeanFailureDetail.INTERNAL_EXCEPTION,
        )
        failures[root_verification_status] += 1
    root_verified = (
        root.status == NodeStatus.PROVED
        and root_verification_status == LeanVerificationStatus.SUCCESS
        and bool(root.verified_proof)
    )
    attempt_failures: Counter[LeanVerificationStatus] = Counter()
    failure_details: Counter[LeanFailureDetail] = Counter()
    for node in blueprint.nodes:
        if node.proof_attempts:
            for attempt in node.proof_attempts:
                status = _enum_or_default(
                    LeanVerificationStatus,
                    attempt.error_type,
                    LeanVerificationStatus.INTERNAL_ERROR,
                )
                detail = _enum_or_default(
                    LeanFailureDetail,
                    attempt.error_detail,
                    LeanFailureDetail.INTERNAL_EXCEPTION,
                )
                if status != LeanVerificationStatus.SUCCESS:
                    attempt_failures[status] += 1
                    failure_details[detail] += 1
        elif node.lean_feedback:
            feedback = node.lean_feedback[-1]
            status = _enum_or_default(
                LeanVerificationStatus,
                feedback.error_type,
                LeanVerificationStatus.INTERNAL_ERROR,
            )
            detail = _enum_or_default(
                LeanFailureDetail,
                feedback.error_detail,
                LeanFailureDetail.INTERNAL_EXCEPTION,
            )
            if status != LeanVerificationStatus.SUCCESS:
                attempt_failures[status] += 1
                failure_details[detail] += 1
    for attempt in root.proof_attempts:
        status = _enum_or_default(
            LeanVerificationStatus,
            attempt.error_type,
            LeanVerificationStatus.INTERNAL_ERROR,
        )
        detail = _enum_or_default(
            LeanFailureDetail,
            attempt.error_detail,
            LeanFailureDetail.INTERNAL_EXCEPTION,
        )
        if status != LeanVerificationStatus.SUCCESS:
            attempt_failures[status] += 1
            failure_details[detail] += 1
    return ProverProblemResult(
        input_hash=problem.input_hash,
        problem_hash=problem.problem_hash,
        planner_status=PlannerDataStatus.SUCCESS,
        prover_status=(
            ProverDataStatus.SUCCESS
            if all_verified and root_verified
            else ProverDataStatus.FAILURE
        ),
        all_subproblems_verified=all_verified,
        problem=problem,
        blueprint=blueprint,
        node_results=node_results,
        root_result=ProverRootResult(
            dependencies=list(root.depends_on),
            target_lean_decl=problem.target_lean_decl,
            status=root.status,
            verification_status=root_verification_status,
            failure_detail=root_failure_detail,
            proof_attempts=root.proof_attempts,
            lean_feedback=root.lean_feedback,
            verified_proof=root.verified_proof,
            latest_diagnostics=(
                root_latest_attempt.diagnostics
                if root_latest_attempt is not None
                else (
                    root_latest_feedback.diagnostics
                    if root_latest_feedback is not None
                    else ""
                )
            ),
            skipped_reason=str(root.metadata.get("skipped_reason") or ""),
        ),
        failure_counts=dict(failures),
        attempt_failure_counts=dict(attempt_failures),
        failure_detail_counts=dict(failure_details),
        metadata={
            "success_definition": (
                "all Blueprint subproblems and ROOT Pantograph-verified"
            ),
            "planner_and_prover_outcomes_are_independent": True,
        },
    )


__all__ = [
    "JsonlProverResultStore",
    "ProverResultStore",
    "collect_prover_result",
]
