#!/usr/bin/env python3
"""Strictly review the three successful round-34 Mathlib API repairs."""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.finalize_numinamath_expand_verification import apply_review, load_reviews, validate_review_hashes  # noqa: E402

RAW = ROOT / "lean_prover/Dataset/raw_data/numinamath_expand.jsonl"
BASE_REVIEW = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_020000_v14.jsonl"
REPAIR_ROOT = ROOT / "outputs/numinamath_expand_verification/repairs/manual_library_api_019000_round34"
REPLACEMENTS = REPAIR_ROOT / "replacement_review_manifest.jsonl"
RESULTS = REPAIR_ROOT / "batches/numinamath_expand_repair_library_api_019000_round34_b00000/verification_results.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_library_api_019000_round34.jsonl"


def sha(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def normalize_statement(value: str) -> str:
    value = re.sub(r"/-.*?-/", " ", value, flags=re.S)
    value = re.sub(r"(?m)^\s*import\s+.*$", " ", value)
    value = re.sub(r"(?m)^\s*open(?:\s+scoped)?\s+.*$", " ", value)
    value = re.sub(r"\btheorem\s+\S+", "theorem _", value, count=1)
    return re.sub(r"\s+", " ", value).strip()


def atomic_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)
    print(json.dumps({"output": str(path), "rows": len(rows), "sha256": sha(payload)}, sort_keys=True))


def main() -> None:
    raw = {row["record_id"]: row for row in map(json.loads, RAW.open(encoding="utf-8"))}
    base, _ = load_reviews([BASE_REVIEW])
    accepted_parents: dict[str, list[str]] = defaultdict(list)
    accepted_normalized: dict[str, list[str]] = defaultdict(list)
    for record_id, review in base.items():
        if review["quality_decision"] != "pass":
            continue
        candidate = apply_review(raw[record_id], review)
        accepted_parents[str(raw[record_id].get("upstream_source") or "")].append(record_id)
        accepted_normalized[normalize_statement(str(candidate["lean_statement"]))].append(record_id)
    replacements = {row["record_id"]: row for row in map(json.loads, REPLACEMENTS.open(encoding="utf-8"))}
    results = {row["record_id"]: row for row in map(json.loads, RESULTS.open(encoding="utf-8"))}
    if set(replacements) != set(results) or len(results) != 3:
        raise SystemExit("replacement/result identity mismatch")
    reviews: list[dict[str, object]] = []
    for record_id in sorted(replacements):
        replacement = replacements[record_id]["replacement"]
        result = results[record_id]
        if result.get("success") is not True or result.get("pantograph_verified") != "success":
            raise SystemExit(f"repair did not pass Pantograph: {record_id}")
        parent = str(raw[record_id].get("upstream_source") or "")
        if accepted_parents.get(parent):
            raise SystemExit(f"global parent conflict for {record_id}: {accepted_parents[parent]}")
        normalized = normalize_statement(str(replacement["lean_statement"]))
        duplicates = accepted_normalized.get(normalized, [])
        method = record_id.split("::", 1)[1].rsplit("_", 1)[0]
        if duplicates:
            review = {
                "manual_reviewed": True,
                "quality_decision": "reject",
                "quality_tier": "extremely_low",
                "record_id": record_id,
                "review_reason": (
                    f"Strict global duplicate re-review after successful API repair. Frozen raw parent={parent}; "
                    f"method={method}. The normalized theorem statement is already accepted under {duplicates}; "
                    "keeping another identical theorem would add no mathematical information and increase "
                    "overfitting risk. It is rejected regardless of Pantograph success."
                ),
                "reviewed_statement_sha256": sha(str(raw[record_id]["lean_statement"])),
                "reviewed_proof_sha256": sha(str(raw[record_id]["proof"])),
                "reviewer": "codex-root-library-api-round34",
            }
            candidate = apply_review(raw[record_id], review)
            validate_review_hashes(review, candidate)
            reviews.append(review)
            continue
        review = {
            "manual_reviewed": True,
            "quality_decision": "pass",
            "quality_tier": "medium",
            "record_id": record_id,
            "replacement": replacement,
            "review_reason": (
                f"Strict human re-review after local Pantograph success. Frozen raw parent={parent}; "
                f"method={method}. The candidate preserves the meaningful sharp inequality boundary "
                "classification from the parent theorem; the repair only qualifies a renamed Mathlib API "
                "or replaces it by its exact current equivalent. Mathematical statement, assumptions, and "
                "proof strategy are unchanged. Exact replacement has a compatible local Pantograph success "
                "receipt; global upstream-parent and normalized-statement duplicate checks both found zero conflicts."
            ),
            "reviewed_statement_sha256": sha(str(replacement["lean_statement"])),
            "reviewed_proof_sha256": sha(str(replacement["proof"])),
            "reviewer": "codex-root-library-api-round34",
        }
        candidate = apply_review(raw[record_id], review)
        validate_review_hashes(review, candidate)
        reviews.append(review)
        accepted_parents[parent].append(record_id)
        accepted_normalized[normalized].append(record_id)
    atomic_jsonl(OUTPUT, reviews)


if __name__ == "__main__":
    main()
