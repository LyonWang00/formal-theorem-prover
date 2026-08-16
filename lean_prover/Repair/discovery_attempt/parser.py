"""Strict EI discovery response parsing and proof normalization."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

from pydantic import ValidationError

from .schema import RepairProposal


class RepairParseError(ValueError):
    pass


_FORBIDDEN = re.compile(
    r"(?i)(```|\bsorry\b|\badmit\b|\baxiom\b|\bplaceholder\b|\bTODO\b)"
)
_PROSE_PREFIX = re.compile(
    r"(?i)^\s*(here(?:'s| is)|the repaired proof|explanation|solution)\b"
)
_TACTIC_PREFIX = re.compile(
    r"^(exact|apply|refine|intro|intros|constructor|cases|rcases|induction|"
    r"simp|simpa|rw|rfl|norm_num|ring|ring_nf|linarith|nlinarith|omega|aesop|"
    r"tauto|decide|native_decide|contradiction|assumption|trivial|subst|"
    r"have|show|suffices|obtain)\b"
)


@dataclass(frozen=True)
class NormalizedProof:
    proof: str
    actions: tuple[str, ...]


def normalize_repaired_proof(raw: str) -> NormalizedProof:
    """Normalize only unambiguous transport/formatting defects."""

    proof = str(raw or "").replace("\r\n", "\n").replace("\r", "\n").strip()
    actions: list[str] = []
    if not proof:
        raise RepairParseError("empty proof field")
    if _FORBIDDEN.search(proof):
        raise RepairParseError("proof contains Markdown or a forbidden proof token")
    if _PROSE_PREFIX.search(proof):
        raise RepairParseError("proof starts with explanatory prose")
    if len(proof) > 20_000:
        raise RepairParseError("proof exceeds the repair proof size limit")
    if re.match(r"^(theorem|lemma|example)\b", proof):
        raise RepairParseError("proof repeated or modified a declaration")
    if "\x00" in proof:
        raise RepairParseError("proof contains NUL bytes")

    # `rfl` and direct terms are valid proof terms. Other leading tactic commands
    # require a `by` wrapper when the model omitted it.
    if not proof.startswith("by") and proof != "rfl" and _TACTIC_PREFIX.match(proof):
        proof = "by\n  " + proof.replace("\n", "\n  ")
        actions.append("wrapped_leading_tactic_in_by_block")
    return NormalizedProof(proof=proof, actions=tuple(actions))


def parse_repaired_proof(raw: str) -> str:
    """Compatibility helper returning one normalized proof string."""

    return normalize_repaired_proof(raw).proof


def parse_repair_response(raw: str, *, expected_attempt_id: str) -> RepairProposal:
    content = str(raw or "").strip()
    if not content:
        raise RepairParseError("empty model response")
    if content.startswith("```") or content.endswith("```"):
        raise RepairParseError("response must be raw JSON without Markdown fences")
    try:
        value = json.loads(content)
    except json.JSONDecodeError as error:
        raise RepairParseError(f"response is not valid JSON: {error.msg}") from error
    if not isinstance(value, dict):
        raise RepairParseError("response must be one JSON object")
    try:
        proposal = RepairProposal.model_validate(value)
    except ValidationError as error:
        raise RepairParseError(f"response schema validation failed: {error}") from error
    if proposal.target_attempt_id != expected_attempt_id:
        raise RepairParseError("response target_attempt_id does not match the request")
    # Validate both proof fields before any verifier work. The normalized values
    # are deliberately materialized later so raw and normalized forms are kept.
    normalize_repaired_proof(proposal.minimal_repair_proof)
    normalize_repaired_proof(proposal.clean_proof)
    return proposal
