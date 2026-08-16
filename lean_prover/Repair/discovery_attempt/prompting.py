"""Version-pinned, statement-grouped EI discovery repair prompts."""

from __future__ import annotations

import hashlib
import json

from .schema import RepairRequest


PROMPT_VERSION = "repair_prompt_v2"
REPAIR_SYSTEM_PROMPT = """You are a Lean 4 proof-repair system. Obey the exact
Lean/Mathlib environment contract in the user message; do not assume a newer or
different library. Repair only the designated target attempt. Other attempts for
the same immutable theorem are evidence, not proof targets.

Return exactly one JSON object and no Markdown or prose outside it. It must have
exactly these keys:
schema_version, target_attempt_id, attempt_assessment, assessment_evidence,
minimal_repair_proof, minimal_repair_change_summary, clean_proof.

schema_version must be "repair_api_response_v2". attempt_assessment must be one
of "fully_incorrect", "partially_incorrect", or "incomplete_completable".
minimal_repair_proof must be a complete Lean proof obtained by the smallest
sound change to the target attempt. clean_proof must be an independently written,
complete proof from scratch. Do not preserve a failed step in clean_proof merely
because it appeared in an attempt. Both proof fields must contain only proof terms
(normally starting with `by`), never theorem declarations, Markdown, sorry,
admit, axioms, placeholders, or comments."""


def _trim(text: str, limit: int) -> str:
    value = text.strip()
    if len(value) <= limit:
        return value
    half = limit // 2
    return value[:half] + "\n...[diagnostics truncated]...\n" + value[-half:]


def build_repair_prompt(request: RepairRequest) -> str:
    environment = request.environment
    attempts = []
    for attempt in request.aggregated_attempts:
        attempts.append(
            {
                "attempt_id": attempt.attempt_id,
                "attempt_rank": attempt.attempt_rank,
                "is_target": attempt.attempt_id == request.target_attempt_id,
                "failure_proof": _trim(attempt.failure_proof, 12_000),
                "error_message": _trim(attempt.error_message, 8_000),
                "failure_class": attempt.failure_class,
                "proof_state": _trim(attempt.proof_state or "Unavailable", 4_000),
                "original_verification_status": attempt.original_verification_status,
            }
        )
    payload = {
        "prompt_version": PROMPT_VERSION,
        "environment_contract": {
            "lean_version": environment.lean_version,
            "lean_commit": environment.lean_commit,
            "mathlib_commit": environment.mathlib_commit,
            "pantograph_version": environment.pantograph_version,
            "imports": environment.imports,
            "environment_hash": environment.environment_hash,
        },
        "immutable_theorem": request.lean_statement,
        "target_attempt_id": request.target_attempt_id,
        "target_failure_class": request.target_failure_class,
        "statement_group_attempts": attempts,
        "classification_guide": {
            "fully_incorrect": "The approach has no useful proof progress or is based on an invalid premise/identifier.",
            "partially_incorrect": "Some steps are useful, but at least one step or term is wrong and must be changed.",
            "incomplete_completable": "The attempt made valid progress and can be completed by adding missing steps.",
        },
        "selection_note": "Local Pantograph compiles both proofs and promotes only a verified proof, preferring the shorter normalized proof.",
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2)


def prompt_sha256(request: RepairRequest) -> str:
    combined = REPAIR_SYSTEM_PROMPT + "\n\n" + build_repair_prompt(request)
    return hashlib.sha256(combined.encode("utf-8")).hexdigest()
