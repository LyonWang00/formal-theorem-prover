"""Shared Lean/Pantograph verification taxonomy.

The top-level values intentionally match
``lean_training.expert_iteration.schemas.VerificationStatus``.  More precise
diagnostic details are kept separately so consumers can aggregate with the
training contract without losing useful repair signals.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class LeanVerificationStatus(StrEnum):
    SUCCESS = "success"
    EXTRACTION_ERROR = "extraction_error"
    SYNTAX_ERROR = "syntax_error"
    ELABORATION_ERROR = "elaboration_error"
    TACTIC_ERROR = "tactic_error"
    UNSOLVED_GOALS = "unsolved_goals"
    TIMEOUT = "timeout"
    FORBIDDEN_TOKEN = "forbidden_token"
    ENVIRONMENT_ERROR = "environment_error"
    INTERNAL_ERROR = "internal_error"


class LeanFailureDetail(StrEnum):
    NONE = "none"
    EMPTY_GENERATION = "empty_generation"
    PROOF_EXTRACTION_FAILED = "proof_extraction_failed"
    FORBIDDEN_DECLARATION = "forbidden_declaration"
    UNKNOWN_IDENTIFIER = "unknown_identifier"
    UNKNOWN_CONSTANT = "unknown_constant"
    MISSING_IMPORT = "missing_import"
    PARSER_ERROR = "parser_error"
    UNEXPECTED_TOKEN = "unexpected_token"
    TYPE_MISMATCH = "type_mismatch"
    APPLICATION_TYPE_MISMATCH = "application_type_mismatch"
    FAILED_TO_SYNTHESIZE = "failed_to_synthesize"
    INVALID_FIELD = "invalid_field"
    TACTIC_EXECUTION_FAILED = "tactic_execution_failed"
    UNSOLVED_GOALS = "unsolved_goals"
    TIMEOUT = "timeout"
    ENVIRONMENT_UNAVAILABLE = "environment_unavailable"
    INTERNAL_EXCEPTION = "internal_exception"
    UNCLASSIFIED_ELABORATION = "unclassified_elaboration"


@dataclass(frozen=True)
class LeanErrorClassification:
    status: LeanVerificationStatus
    detail: LeanFailureDetail


_UNKNOWN_IDENTIFIER = re.compile(
    r"unknown\s+(?:identifier|constant)|invalid\s+field\s+notation",
    re.IGNORECASE,
)
_SYNTAX = re.compile(
    r"syntax\s+error|parser|unexpected\s+(?:token|end\s+of\s+input)|expected\s+.+\s+token",
    re.IGNORECASE,
)
_TACTIC = re.compile(
    r"tactic\s+.+failed|tactic\s+failure|no\s+goals\s+to\s+be\s+solved",
    re.IGNORECASE,
)
_ENVIRONMENT = re.compile(
    r"environment|failed\s+to\s+import|unknown\s+module|module\s+.+not\s+found|pantograph\s+server",
    re.IGNORECASE,
)


def classify_lean_diagnostics(
    diagnostics: str,
    *,
    success: bool = False,
    timed_out: bool = False,
    extraction_error: bool = False,
    forbidden_token: bool = False,
    environment_error: bool = False,
    internal_error: bool = False,
) -> LeanErrorClassification:
    """Classify compiler output using the shared training-compatible contract."""

    message = diagnostics.strip()
    lowered = message.casefold()
    if success:
        return LeanErrorClassification(
            LeanVerificationStatus.SUCCESS,
            LeanFailureDetail.NONE,
        )
    if timed_out:
        return LeanErrorClassification(
            LeanVerificationStatus.TIMEOUT,
            LeanFailureDetail.TIMEOUT,
        )
    if extraction_error:
        detail = (
            LeanFailureDetail.EMPTY_GENERATION
            if not message
            else LeanFailureDetail.PROOF_EXTRACTION_FAILED
        )
        return LeanErrorClassification(LeanVerificationStatus.EXTRACTION_ERROR, detail)
    if forbidden_token:
        return LeanErrorClassification(
            LeanVerificationStatus.FORBIDDEN_TOKEN,
            LeanFailureDetail.FORBIDDEN_DECLARATION,
        )
    if environment_error or _ENVIRONMENT.search(message):
        detail = (
            LeanFailureDetail.MISSING_IMPORT
            if "import" in lowered or "module" in lowered
            else LeanFailureDetail.ENVIRONMENT_UNAVAILABLE
        )
        return LeanErrorClassification(LeanVerificationStatus.ENVIRONMENT_ERROR, detail)
    if internal_error:
        return LeanErrorClassification(
            LeanVerificationStatus.INTERNAL_ERROR,
            LeanFailureDetail.INTERNAL_EXCEPTION,
        )
    if "unsolved goal" in lowered:
        return LeanErrorClassification(
            LeanVerificationStatus.UNSOLVED_GOALS,
            LeanFailureDetail.UNSOLVED_GOALS,
        )
    if _SYNTAX.search(message):
        detail = (
            LeanFailureDetail.UNEXPECTED_TOKEN
            if "unexpected" in lowered
            else LeanFailureDetail.PARSER_ERROR
        )
        return LeanErrorClassification(LeanVerificationStatus.SYNTAX_ERROR, detail)
    if _UNKNOWN_IDENTIFIER.search(message):
        if "invalid field" in lowered:
            detail = LeanFailureDetail.INVALID_FIELD
        elif "constant" in lowered:
            detail = LeanFailureDetail.UNKNOWN_CONSTANT
        else:
            detail = LeanFailureDetail.UNKNOWN_IDENTIFIER
        # Training's VerificationStatus folds unknown identifiers into
        # elaboration_error.  The secondary detail preserves the distinction.
        return LeanErrorClassification(LeanVerificationStatus.ELABORATION_ERROR, detail)
    if _TACTIC.search(message) or "tactic" in lowered:
        return LeanErrorClassification(
            LeanVerificationStatus.TACTIC_ERROR,
            LeanFailureDetail.TACTIC_EXECUTION_FAILED,
        )
    if "application type mismatch" in lowered:
        return LeanErrorClassification(
            LeanVerificationStatus.ELABORATION_ERROR,
            LeanFailureDetail.APPLICATION_TYPE_MISMATCH,
        )
    if "type mismatch" in lowered:
        return LeanErrorClassification(
            LeanVerificationStatus.ELABORATION_ERROR,
            LeanFailureDetail.TYPE_MISMATCH,
        )
    if "failed to synthesize" in lowered:
        return LeanErrorClassification(
            LeanVerificationStatus.ELABORATION_ERROR,
            LeanFailureDetail.FAILED_TO_SYNTHESIZE,
        )
    return LeanErrorClassification(
        LeanVerificationStatus.ELABORATION_ERROR,
        LeanFailureDetail.UNCLASSIFIED_ELABORATION,
    )
