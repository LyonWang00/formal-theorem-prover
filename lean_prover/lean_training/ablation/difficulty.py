"""Transparent deterministic proxy features for Lean proof difficulty."""

from __future__ import annotations

import hashlib
import math
import re
from collections import Counter
from typing import Any


STATIC_DIFFICULTY_VERSION = "anchor_static_v2"
PRIMARY_TACTICS = (
    "simp_all", "native_decide", "norm_num", "nlinarith", "ring_nf",
    "linarith", "omega", "aesop", "exact", "simp", "ring", "rfl", "decide",
)


def normalized_proof(proof: str) -> str:
    return re.sub(r"\s+", " ", proof.strip())


def percentile(values: list[int | float], fraction: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _tactic_steps(proof: str, trajectory_steps: int | None) -> int:
    body = re.sub(r"^\s*by\b", "", proof, count=1).strip()
    lines = [line.strip() for line in body.splitlines() if line.strip() and not line.strip().startswith("--")]
    syntactic = sum(max(1, len(re.findall(r"(?:<;>|;)", line)) + 1) for line in lines)
    return max(1, syntactic, int(trajectory_steps or 0))


def _lemma_reference_count(proof: str) -> int:
    bracket_references = re.findall(r"\[([^\]]+)\]", proof)
    bracket_names = sum(len(re.findall(r"[A-Za-z_][\w.']*", value)) for value in bracket_references)
    command_refs = re.findall(r"\b(?:exact|apply|refine|using|rw|have)\s+\(?([A-Za-z_][\w.']*)", proof)
    return bracket_names + len(command_refs)


def extract_static_features(row: dict[str, Any], *, tokenizer, compile_time_ms: float | None) -> dict[str, Any]:
    statement = str(row.get("lean_statement") or row.get("statement") or "")
    context = "\n".join(str(value) for value in row.get("context_lines") or [])
    proof = str(row.get("proof") or row.get("completion") or "")
    tactics = [tactic for tactic in PRIMARY_TACTICS if re.search(rf"\b{re.escape(tactic)}\b", proof)]
    trajectory_steps = (row.get("metadata") or {}).get("trajectory_steps")
    step_count = _tactic_steps(proof, trajectory_steps)
    branch_count = proof.count("<;>") + len(re.findall(r"(?m)^\s*[·|]\s+", proof)) + len(re.findall(r"\b(?:case|by_cases|constructor|cases')\b", proof))
    return {
        "statement_id": str(row.get("statement_id") or row.get("id") or ""),
        "record_id": str(row.get("record_id") or row.get("id") or ""),
        "statement_tokens": len(tokenizer(statement, add_special_tokens=False)["input_ids"]),
        "context_tokens": len(tokenizer(context, add_special_tokens=False)["input_ids"]),
        "proof_tokens": len(tokenizer(proof, add_special_tokens=False)["input_ids"]),
        "tactic_step_count": step_count,
        "single_tactic": step_count == 1,
        "branch_count": branch_count,
        "have_count": len(re.findall(r"\bhave\b", proof)),
        "calc_count": len(re.findall(r"\bcalc\b", proof)),
        "lemma_reference_count": _lemma_reference_count(proof),
        "compile_time_ms": compile_time_ms,
        "normalized_proof_hash": hashlib.sha256(normalized_proof(proof).encode()).hexdigest(),
        "tactic_signature": "+".join(tactics) if tactics else "other",
        "primary_tactics": tactics or ["other"],
        "category": str(row.get("category") or (tactics[0] if tactics else "other")),
        "static_difficulty_version": STATIC_DIFFICULTY_VERSION,
    }


def assign_static_buckets(records: list[dict[str, Any]]) -> dict[str, float]:
    thresholds = {
        "proof_tokens_p50": percentile([row["proof_tokens"] for row in records], 0.50),
        "proof_tokens_p75": percentile([row["proof_tokens"] for row in records], 0.75),
        "tactic_steps_p75": percentile([row["tactic_step_count"] for row in records], 0.75),
        "lemma_references_p75": percentile([row["lemma_reference_count"] for row in records], 0.75),
        "compile_time_p50": percentile([row["compile_time_ms"] or 0 for row in records], 0.50),
        "compile_time_p75": percentile([row["compile_time_ms"] or 0 for row in records], 0.75),
    }
    for row in records:
        hard_flags = {
            "long_proof": row["proof_tokens"] >= thresholds["proof_tokens_p75"],
            "many_steps": row["tactic_step_count"] >= thresholds["tactic_steps_p75"] and row["tactic_step_count"] > 1,
            "branching": row["branch_count"] > 0,
            "has_have": row["have_count"] > 0,
            "has_calc": row["calc_count"] > 0,
            "many_lemma_refs": row["lemma_reference_count"] >= thresholds["lemma_references_p75"] and row["lemma_reference_count"] > 0,
            "slow_compile": (row["compile_time_ms"] or 0) >= thresholds["compile_time_p75"],
        }
        easy_conditions = (
            row["single_tactic"], row["proof_tokens"] <= thresholds["proof_tokens_p50"],
            row["branch_count"] == 0, row["have_count"] == 0, row["calc_count"] == 0,
            (row["compile_time_ms"] or 0) <= thresholds["compile_time_p50"],
        )
        if any(hard_flags.values()):
            bucket = "static_hard"
        elif sum(easy_conditions) == 6:
            bucket = "static_easy"
        else:
            bucket = "static_medium"
        row["static_bucket"] = bucket
        row["static_hard_reasons"] = [key for key, value in hard_flags.items() if value]
        row["static_easy_condition_count"] = sum(easy_conditions)
        row["static_score"] = sum(hard_flags.values()) - sum(easy_conditions) / 6
    return thresholds


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    proof_hashes = Counter(row["normalized_proof_hash"] for row in records)
    tactics = Counter(tactic for row in records for tactic in row["primary_tactics"])
    signatures = Counter(row["tactic_signature"] for row in records)

    def distribution(field: str) -> dict[str, float]:
        values = [float(row[field] or 0) for row in records]
        return {"mean": sum(values) / max(1, len(values)), "p50": percentile(values, .50), "p75": percentile(values, .75), "p90": percentile(values, .90), "p95": percentile(values, .95), "p99": percentile(values, .99), "max": max(values, default=0)}

    return {
        "records": len(records), "bucket_counts": dict(Counter(row["static_bucket"] for row in records)),
        "proof_tokens": distribution("proof_tokens"), "statement_tokens": distribution("statement_tokens"),
        "compile_time_ms": distribution("compile_time_ms"), "tactic_steps": distribution("tactic_step_count"),
        "single_tactic_ratio": sum(row["single_tactic"] for row in records) / max(1, len(records)),
        "multi_step_ratio": sum(not row["single_tactic"] for row in records) / max(1, len(records)),
        "primary_tactic_counts": dict(tactics), "distinct_tactic_signatures": len(signatures),
        "exact_duplicate_proof_ratio": 1 - len(proof_hashes) / max(1, len(records)),
        "top10_proof_coverage": sum(count for _, count in proof_hashes.most_common(10)) / max(1, len(records)),
        "top20_proof_coverage": sum(count for _, count in proof_hashes.most_common(20)) / max(1, len(records)),
        "top50_proof_coverage": sum(count for _, count in proof_hashes.most_common(50)) / max(1, len(records)),
    }
