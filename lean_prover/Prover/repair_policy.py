"""One proof-repair budget and eligibility contract for every architecture."""

from __future__ import annotations

from dataclasses import dataclass

from lean_prover.Data import LeanFailureDetail, LeanVerificationStatus
from lean_prover.Planner.schemas import LeanFeedback, ProofAttempt


REFERENCE_REPAIR_DETAILS = frozenset(
    {
        LeanFailureDetail.UNKNOWN_IDENTIFIER.value,
        LeanFailureDetail.UNKNOWN_CONSTANT.value,
        LeanFailureDetail.INVALID_FIELD.value,
    }
)

PROOF_REPAIR_DETAILS = frozenset(
    {
        *REFERENCE_REPAIR_DETAILS,
        LeanFailureDetail.PARSER_ERROR.value,
        LeanFailureDetail.UNEXPECTED_TOKEN.value,
        LeanFailureDetail.TYPE_MISMATCH.value,
        LeanFailureDetail.APPLICATION_TYPE_MISMATCH.value,
        LeanFailureDetail.FAILED_TO_SYNTHESIZE.value,
        LeanFailureDetail.TACTIC_EXECUTION_FAILED.value,
        LeanFailureDetail.UNSOLVED_GOALS.value,
        LeanFailureDetail.UNCLASSIFIED_ELABORATION.value,
    }
)


@dataclass(frozen=True)
class ControlledProofBudget:
    """Small defaults used by the controlled three-architecture experiment."""

    pass_k: int = 2
    repair_rounds_per_node: int = 2
    repair_candidates_per_node: int = 2
    candidates_per_generation: int = 1

    def __post_init__(self) -> None:
        if self.pass_k < 1:
            raise ValueError("pass_k must be positive")
        if self.repair_rounds_per_node < 0:
            raise ValueError("repair_rounds_per_node must be nonnegative")
        if self.repair_candidates_per_node < 0:
            raise ValueError("repair_candidates_per_node must be nonnegative")
        if self.candidates_per_generation != 1:
            raise ValueError("controlled repair generates exactly one candidate")


DEFAULT_CONTROLLED_PROOF_BUDGET = ControlledProofBudget()


def proof_repair_eligible(value: LeanFeedback | ProofAttempt) -> bool:
    """Repair bounded proof-body failures, never environment/statement failures."""

    detail = getattr(value, "error_detail", "")
    return str(detail) in PROOF_REPAIR_DETAILS


def reference_repair_eligible(value: LeanFeedback | ProofAttempt) -> bool:
    """Backward-compatible alias for callers migrated with older experiments."""

    return proof_repair_eligible(value)


def retrieval_grounding_required(value: LeanFeedback | ProofAttempt) -> bool:
    """Old/missing declaration repairs must cite a pinned-index candidate."""

    detail = getattr(value, "error_detail", "")
    return str(detail) in REFERENCE_REPAIR_DETAILS


def enforce_repair_grounding(
    feedback: LeanFeedback,
    metadata: dict[str, object],
) -> LeanFeedback:
    """Reject compiler success lacking required pinned-index evidence."""

    if not feedback.success or metadata.get("declarations_grounded") is not False:
        return feedback
    ungrounded = metadata.get("ungrounded_declared_names") or []
    if metadata.get("missing_required_index_evidence"):
        policy_message = (
            "program grounding gate rejected reference repair without an "
            "exact declaration cited from the pinned local environment index"
        )
    else:
        policy_message = (
            "program grounding gate rejected declarations outside the local "
            "retrieval/context allow-list: "
            + ", ".join(str(item) for item in ungrounded)
        )
    diagnostics = feedback.diagnostics or feedback.message
    return LeanFeedback(
        stage="proof_repair_grounding",
        success=False,
        message=policy_message,
        diagnostics=(
            diagnostics + "\n" + policy_message if diagnostics else policy_message
        ),
        error_type=LeanVerificationStatus.FORBIDDEN_TOKEN.value,
        error_detail=LeanFailureDetail.FORBIDDEN_DECLARATION.value,
    )


__all__ = [
    "ControlledProofBudget",
    "DEFAULT_CONTROLLED_PROOF_BUDGET",
    "PROOF_REPAIR_DETAILS",
    "REFERENCE_REPAIR_DETAILS",
    "proof_repair_eligible",
    "reference_repair_eligible",
    "retrieval_grounding_required",
    "enforce_repair_grounding",
]
