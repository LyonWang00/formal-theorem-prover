"""Failure taxonomy and proof-distribution analysis for verified Lean data."""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Iterable

from .contracts import LeanDataRecord
from .preparation import normalize_proof_rhs


VERSION_INCOMPATIBLE_IDENTIFIERS = {"exp", "one_le_rpow"}


def classify_verification_failure(result: dict[str, Any]) -> str:
    diagnostics = str(result.get("diagnostics") or "")
    lowered = diagnostics.lower()
    if result.get("timed_out") or "timeout" in lowered or "timed out" in lowered:
        return "timeout"
    if "pantograph" in lowered and any(
        marker in lowered for marker in ("server", "connection", "environment")
    ):
        return "environment_error"
    unknown = re.search(r"Unknown (?:identifier|constant) `([^`]+)`", diagnostics)
    if unknown:
        identifier = unknown.group(1)
        if identifier in VERSION_INCOMPATIBLE_IDENTIFIERS:
            return "version_incompatible_identifier"
        if identifier.startswith("h") or identifier in {"this"}:
            return "missing_hypothesis"
        if identifier[:1].islower():
            return "missing_local_variable"
        return "missing_import"
    if "unexpected token" in lowered or "expected" in lowered and "syntax" in lowered:
        return "reference_syntax_error"
    if "invalid `" in lowered and "notation" in lowered:
        return "source_extraction_corruption"
    if "unsolved goals" in lowered:
        return "reference_unsolved_goals"
    if any(
        marker in lowered
        for marker in (
            "linarith failed",
            "made no progress",
            "tactic",
            "no goals to be solved",
        )
    ):
        return "reference_tactic_failure"
    if "unknown namespace" in lowered:
        return "missing_namespace"
    if "unknown parser" in lowered or "unknown syntax" in lowered:
        return "missing_local_notation"
    if result.get("compile_errors"):
        return "reference_elaboration_error"
    return "unknown"


TACTIC_NAMES = (
    "simp",
    "norm_num",
    "linarith",
    "nlinarith",
    "aesop",
    "omega",
    "ring",
    "ring_nf",
    "exact",
    "refine",
    "rw",
    "intro",
    "rintro",
    "constructor",
    "cases",
    "induction",
    "decide",
)


def normalized_proof_body(proof: str) -> str:
    value = normalize_proof_rhs(proof).strip()
    value = re.sub(r"^by\s*", "", value)
    return re.sub(r"\s+", " ", value).strip()


def tactic_signature(proof: str) -> tuple[str, ...]:
    body = normalized_proof_body(proof)
    tokens = re.findall(r"[A-Za-z_][A-Za-z0-9_']*", body)
    selected = [token for token in tokens if token in TACTIC_NAMES]
    return tuple(selected) or ((tokens[0],) if tokens else ())


def proof_distribution(records: Iterable[LeanDataRecord]) -> dict[str, Any]:
    usable = [record for record in records if record.proof]
    bodies = [normalized_proof_body(record.proof or "") for record in usable]
    exact = Counter(bodies)
    signatures = Counter(";".join(tactic_signature(record.proof or "")) for record in usable)
    token_lengths = sorted(len(body.split()) for body in bodies)

    def percentile(ratio: float) -> int:
        if not token_lengths:
            return 0
        return token_lengths[min(len(token_lengths) - 1, int((len(token_lengths) - 1) * ratio))]

    single_tactic = sum(len(tactic_signature(record.proof or "")) == 1 for record in usable)
    tactic_presence = {
        tactic: sum(
            tactic in tactic_signature(record.proof or "") for record in usable
        )
        for tactic in ("simp", "norm_num", "linarith", "aesop", "omega")
    }
    return {
        "records": len(usable),
        "unique_normalized_proofs": len(exact),
        "exact_duplicate_records": sum(count - 1 for count in exact.values()),
        "top_exact_proofs": exact.most_common(50),
        "top_tactic_signatures": signatures.most_common(50),
        "top_10_proof_coverage": sum(count for _, count in exact.most_common(10))
        / max(1, len(usable)),
        "top_20_proof_coverage": sum(count for _, count in exact.most_common(20))
        / max(1, len(usable)),
        "top_50_proof_coverage": sum(count for _, count in exact.most_common(50))
        / max(1, len(usable)),
        "single_tactic_records": single_tactic,
        "single_tactic_fraction": single_tactic / max(1, len(usable)),
        "multi_step_records": len(usable) - single_tactic,
        "tactic_presence": tactic_presence,
        "proof_token_lengths": {
            "min": min(token_lengths, default=0),
            "p50": percentile(0.50),
            "p90": percentile(0.90),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": max(token_lengths, default=0),
            "mean": sum(token_lengths) / max(1, len(token_lengths)),
        },
    }
