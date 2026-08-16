"""Dynamic theorem difficulty derived from verified EI discovery candidates."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping, Sequence


SCHEMA_VERSION = "ei-dynamic-difficulty-v1"
CLASSIFICATION_VERSION = "success-rate-thresholds-v1"
DIFFICULTIES = ("too_easy", "easy", "medium", "hard", "impossible")
FAILURE_TYPES = (
    "unsolved_goals",
    "tactic_error",
    "elaboration_error",
    "syntax_error",
    "unknown_identifier",
    "environment_error",
    "timeout",
    "extraction_error",
    "data_error",
    "other",
)
DATA_ENVIRONMENT_FAILURES = {
    "syntax_error",
    "unknown_identifier",
    "environment_error",
    "timeout",
    "extraction_error",
    "data_error",
}
CAPABILITY_FAILURES = {"unsolved_goals", "tactic_error", "elaboration_error"}
DEFAULT_EFFECTIVE_DIFFICULTIES = {"easy", "medium", "hard"}


class DynamicDifficultyError(ValueError):
    """Raised when discovery inputs violate the classification contract."""


def iter_jsonl(path: str | Path) -> Iterable[dict[str, Any]]:
    with Path(path).open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise DynamicDifficultyError(
                    f"invalid JSON at {path}:{line_number}: {error}"
                ) from error
            if not isinstance(row, dict):
                raise DynamicDifficultyError(
                    f"JSONL row must be an object at {path}:{line_number}"
                )
            yield row


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def normalized_domain(value: Any) -> str:
    domain = str(value or "unknown").strip().lower().replace(" ", "_")
    return domain or "unknown"


def normalize_failure_type(candidate: Mapping[str, Any]) -> str:
    taxonomy = str(
        candidate.get("failure_taxonomy")
        or candidate.get("pantograph_status")
        or candidate.get("status")
        or "other"
    ).strip().lower().replace("-", "_").replace(" ", "_")
    message = str(candidate.get("error_message") or "").lower()
    if "unknown identifier" in message or "unknown constant" in message:
        return "unknown_identifier"
    if "unexpected token" in message or "parser" in message or taxonomy in {
        "parse_error",
        "syntax",
    }:
        return "syntax_error"
    if bool(candidate.get("timed_out")) or taxonomy in {"timed_out", "timeout_only"}:
        return "timeout"
    aliases = {
        "verified": "verified",
        "success": "verified",
        "compile_success": "verified",
        "unsolved_goal": "unsolved_goals",
        "goals_remaining": "unsolved_goals",
        "unknown_ident": "unknown_identifier",
        "env_error": "environment_error",
        "extraction_failed": "extraction_error",
        "empty_extraction": "extraction_error",
    }
    taxonomy = aliases.get(taxonomy, taxonomy)
    return taxonomy if taxonomy in FAILURE_TYPES or taxonomy == "verified" else "other"


def classify_success_rate(success_count: int, attempt_count: int) -> str:
    if attempt_count <= 0:
        raise DynamicDifficultyError("attempt_count must be positive")
    if not 0 <= success_count <= attempt_count:
        raise DynamicDifficultyError(
            f"success_count must be in [0, attempt_count], got {success_count}/{attempt_count}"
        )
    if success_count == 0:
        return "impossible"
    rate = success_count / attempt_count
    if rate >= 0.75:
        return "too_easy"
    if rate >= 0.50:
        return "easy"
    if rate >= 0.25:
        return "medium"
    return "hard"


def zero_success_subtype(failures: Mapping[str, int]) -> str | None:
    total = sum(int(value) for value in failures.values())
    if total == 0:
        return "mixed_or_unknown"
    invalid = sum(int(failures.get(name, 0)) for name in DATA_ENVIRONMENT_FAILURES)
    capability = sum(int(failures.get(name, 0)) for name in CAPABILITY_FAILURES)
    if invalid:
        return "data_environment_hard"
    if capability == total:
        return "capability_hard"
    return "mixed_or_unknown"


def _required_text(row: Mapping[str, Any], field: str, *, context: str) -> str:
    value = str(row.get(field) or "").strip()
    if not value:
        raise DynamicDifficultyError(f"missing {field} in {context}")
    return value


def _manifest_indexes(
    rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]]]:
    by_theorem: dict[str, dict[str, Any]] = {}
    by_record: dict[str, dict[str, Any]] = {}
    for row in rows:
        record_id = _required_text(row, "id", context="discovery manifest")
        theorem_id = str(
            row.get("theorem_group_id") or row.get("statement_id") or record_id
        ).strip()
        if record_id in by_record:
            raise DynamicDifficultyError(f"duplicate record_id: {record_id}")
        if theorem_id in by_theorem:
            raise DynamicDifficultyError(f"duplicate theorem_id: {theorem_id}")
        by_record[record_id] = row
        by_theorem[theorem_id] = row
    return by_theorem, by_record


def _candidate_groups(
    rows: Sequence[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_candidates: set[str] = set()
    for row in rows:
        candidate_id = _required_text(row, "candidate_id", context="candidate results")
        statement_id = _required_text(row, "statement_id", context=candidate_id)
        if candidate_id in seen_candidates:
            raise DynamicDifficultyError(f"duplicate candidate_id: {candidate_id}")
        seen_candidates.add(candidate_id)
        groups[statement_id].append(row)
    return groups


def classify_theorem(
    manifest_row: Mapping[str, Any], candidates: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    if not candidates:
        raise DynamicDifficultyError("theorem has no candidate results")
    theorem_group_id = str(
        manifest_row.get("theorem_group_id")
        or manifest_row.get("statement_id")
        or manifest_row.get("id")
        or ""
    ).strip()
    candidate_statement_ids = {
        str(row.get("statement_id") or "").strip() for row in candidates
    }
    if len(candidate_statement_ids) != 1 or "" in candidate_statement_ids:
        raise DynamicDifficultyError(
            f"candidates must have exactly one non-empty statement_id: "
            f"{sorted(candidate_statement_ids)}"
        )
    theorem_id = next(iter(candidate_statement_ids))
    record_id = _required_text(manifest_row, "id", context=theorem_group_id)
    statement = str(
        manifest_row.get("statement") or manifest_row.get("lean_statement") or ""
    ).strip()
    if not statement:
        raise DynamicDifficultyError(f"missing statement for {theorem_id}")
    ordered = sorted(candidates, key=lambda row: int(row.get("candidate_rank") or 0))
    ranks = [int(row.get("candidate_rank") or 0) for row in ordered]
    if ranks != list(range(1, len(ordered) + 1)):
        raise DynamicDifficultyError(
            f"candidate ranks for {theorem_id} must be contiguous 1..N, got {ranks}"
        )
    source_ids = {str(row.get("source_id") or "") for row in ordered}
    if source_ids != {record_id}:
        raise DynamicDifficultyError(
            f"candidate source_id mismatch for {theorem_id}: {sorted(source_ids)} != {record_id}"
        )

    successes = [row for row in ordered if bool(row.get("pantograph_verified"))]
    attempt_count = len(ordered)
    success_count = len(successes)
    difficulty = classify_success_rate(success_count, attempt_count)
    failures = Counter(
        normalize_failure_type(row)
        for row in ordered
        if not bool(row.get("pantograph_verified"))
    )
    failure_distribution = {
        name: int(failures.get(name, 0)) for name in FAILURE_TYPES
    }
    subtype = zero_success_subtype(failure_distribution) if success_count == 0 else None
    candidate_lengths = [int(row.get("generation_length") or 0) for row in ordered]
    success_lengths = [int(row.get("generation_length") or 0) for row in successes]
    pass_at: dict[str, int | None] = {}
    for k in (1, 2, 4, 8):
        pass_at[f"pass_at_{k}"] = (
            int(any(bool(row.get("pantograph_verified")) for row in ordered[:k]))
            if attempt_count >= k
            else None
        )
    success_proofs = {
        str(row.get("generated_proof") or "").strip() for row in successes
    }
    success_proofs.discard("")
    effective = difficulty in DEFAULT_EFFECTIVE_DIFFICULTIES
    tags: list[str] = []
    if difficulty in {"easy", "medium"}:
        tags.append("ei_foundation")
    if difficulty in {"medium", "hard"}:
        tags.append("frontier_learning")
    if difficulty == "hard" or (
        difficulty == "impossible" and subtype == "capability_hard"
    ):
        tags.append("exploration")
    return {
        "schema_version": SCHEMA_VERSION,
        "classification_version": CLASSIFICATION_VERSION,
        "theorem_id": theorem_id,
        "theorem_group_id": theorem_group_id,
        "record_id": record_id,
        "source": str(manifest_row.get("source") or "unknown"),
        "domain": normalized_domain(
            manifest_row.get("category") or manifest_row.get("domain")
        ),
        "statement": statement,
        "attempt_count": attempt_count,
        "success_count": success_count,
        "success_rate": success_count / attempt_count,
        **pass_at,
        "difficulty": difficulty,
        "zero_success_subtype": subtype,
        "failure_distribution": failure_distribution,
        "failure_family_distribution": {
            "capability": sum(failure_distribution[name] for name in CAPABILITY_FAILURES),
            "data_environment": sum(
                failure_distribution[name] for name in DATA_ENVIRONMENT_FAILURES
            ),
            "other": failure_distribution["other"],
        },
        "proof_statistics": {
            "successful_proof_count": success_count,
            "unique_successful_proof_count": len(success_proofs),
            "mean_success_tokens": mean(success_lengths) if success_lengths else 0.0,
            "mean_candidate_tokens": mean(candidate_lengths),
        },
        "candidate_ids": [str(row["candidate_id"]) for row in ordered],
        "successful_candidate_ids": [str(row["candidate_id"]) for row in successes],
        "candidate_seed_min": min(int(row.get("candidate_seed") or 0) for row in ordered),
        "candidate_seed_max": max(int(row.get("candidate_seed") or 0) for row in ordered),
        "checkpoint_hashes": sorted(
            {str(row.get("checkpoint_hash") or "") for row in ordered}
        ),
        "generation_config_hashes": sorted(
            {str(row.get("generation_config_hash") or "") for row in ordered}
        ),
        "effective_for_ei_default": effective,
        "selection_tags": tags,
    }


def classify_discovery(
    manifest_rows: Sequence[dict[str, Any]],
    candidate_rows: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    by_theorem, by_record = _manifest_indexes(manifest_rows)
    candidate_groups = _candidate_groups(candidate_rows)
    records: list[dict[str, Any]] = []
    matched_candidate_groups: set[str] = set()
    for theorem_id, manifest_row in by_theorem.items():
        candidates = candidate_groups.get(theorem_id)
        candidate_key = theorem_id
        if candidates is None:
            record_id = str(manifest_row["id"])
            matching = [
                (key, rows)
                for key, rows in candidate_groups.items()
                if rows and str(rows[0].get("source_id") or "") == record_id
            ]
            if len(matching) != 1:
                raise DynamicDifficultyError(
                    f"cannot uniquely join candidates for theorem {theorem_id}"
                )
            candidate_key, candidates = matching[0]
        matched_candidate_groups.add(candidate_key)
        records.append(classify_theorem(manifest_row, candidates))
    extra = set(candidate_groups) - matched_candidate_groups
    if extra:
        raise DynamicDifficultyError(
            f"candidate results contain {len(extra)} unmatched statement groups"
        )
    if len(records) != len(by_record):
        raise DynamicDifficultyError("difficulty coverage is not 100%")
    return sorted(records, key=lambda row: row["theorem_id"])


def _aggregate_group(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    attempts = sum(int(row["attempt_count"]) for row in rows)
    successes = sum(int(row["success_count"]) for row in rows)
    return {
        "theorems": len(rows),
        "attempts": attempts,
        "successes": successes,
        "candidate_success_rate": successes / attempts if attempts else 0.0,
        "mean_theorem_success_rate": mean(float(row["success_rate"]) for row in rows)
        if rows
        else 0.0,
        "difficulty_distribution": dict(
            sorted(Counter(str(row["difficulty"]) for row in rows).items())
        ),
        "zero_success_subtypes": dict(
            sorted(
                Counter(
                    str(row["zero_success_subtype"])
                    for row in rows
                    if row.get("zero_success_subtype")
                ).items()
            )
        ),
    }


def build_statistics(
    records: Sequence[Mapping[str, Any]],
    *,
    manifest_sha256: str,
    candidate_results_sha256: str,
) -> dict[str, Any]:
    theorem_ids = [str(row["theorem_id"]) for row in records]
    record_ids = [str(row["record_id"]) for row in records]
    if len(theorem_ids) != len(set(theorem_ids)):
        raise DynamicDifficultyError("theorem duplicate gate failed")
    if len(record_ids) != len(set(record_ids)):
        raise DynamicDifficultyError("record duplicate gate failed")
    by_source: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_domain: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    failures: Counter[str] = Counter()
    hard_failures: Counter[str] = Counter()
    for row in records:
        by_source[str(row["source"])].append(row)
        by_domain[str(row["domain"])].append(row)
        for name, count in row["failure_distribution"].items():
            failures[name] += int(count)
            if row["difficulty"] in {"hard", "impossible"}:
                hard_failures[name] += int(count)
    success_bank_statements = [row for row in records if int(row["success_count"]) > 0]
    success_bank_candidates = Counter()
    for row in success_bank_statements:
        success_bank_candidates[str(row["difficulty"])] += int(row["success_count"])
    overall = _aggregate_group(records)
    effective = [row for row in records if bool(row["effective_for_ei_default"])]
    return {
        "schema_version": SCHEMA_VERSION,
        "classification_version": CLASSIFICATION_VERSION,
        "status": "DYNAMIC_DIFFICULTY_CLASSIFICATION_COMPLETED",
        "input_hashes": {
            "discovery_manifest_sha256": manifest_sha256,
            "candidate_results_sha256": candidate_results_sha256,
        },
        "thresholds": {
            "too_easy": "success_rate >= 0.75",
            "easy": "0.50 <= success_rate < 0.75",
            "medium": "0.25 <= success_rate < 0.50",
            "hard": "0 < success_rate < 0.25",
            "impossible": "success_count = 0, subtype determined from failures",
        },
        "overall": overall,
        "by_source": {
            key: _aggregate_group(value) for key, value in sorted(by_source.items())
        },
        "by_domain": {
            key: _aggregate_group(value) for key, value in sorted(by_domain.items())
        },
        "failure_distribution": {
            name: int(failures.get(name, 0)) for name in FAILURE_TYPES
        },
        "hard_impossible_failure_distribution": {
            name: int(hard_failures.get(name, 0)) for name in FAILURE_TYPES
        },
        "success_bank": {
            "contributing_theorems": len(success_bank_statements),
            "statement_difficulty_distribution": dict(
                sorted(
                    Counter(
                        str(row["difficulty"]) for row in success_bank_statements
                    ).items()
                )
            ),
            "verified_candidate_distribution": dict(sorted(success_bank_candidates.items())),
        },
        "effective_default": {
            "definition": ["easy", "medium", "hard"],
            "theorems": len(effective),
            "source_distribution": dict(
                sorted(Counter(str(row["source"]) for row in effective).items())
            ),
        },
        "audit": {
            "rows": len(records),
            "theorem_duplicates": len(theorem_ids) - len(set(theorem_ids)),
            "record_duplicates": len(record_ids) - len(set(record_ids)),
            "difficulty_coverage": sum(
                str(row.get("difficulty") or "") in DIFFICULTIES for row in records
            )
            / len(records)
            if records
            else 0.0,
        },
    }
