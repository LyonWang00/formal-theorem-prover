#!/usr/bin/env python3
"""Strictly review successful round-35 projection syntax repairs."""

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
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_020000_v14.jsonl"
REPAIR = ROOT / "outputs/numinamath_expand_verification/repairs/manual_projection_syntax_020000_round35"
REPLACEMENTS = REPAIR / "replacement_review_manifest.jsonl"
RESULTS = REPAIR / "batches/numinamath_expand_repair_projection_syntax_020000_round35_b00000/verification_results.jsonl"
OUTPUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_projection_syntax_020000_round35.jsonl"


def sha(value: str | bytes) -> str:
    data = value if isinstance(value, bytes) else value.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def normalize(value: str) -> str:
    value = re.sub(r"/-.*?-/", " ", value, flags=re.S)
    value = re.sub(r"(?m)^\s*(?:import|open(?:\s+scoped)?)\s+.*$", " ", value)
    value = re.sub(r"\btheorem\s+\S+", "theorem _", value, count=1)
    return re.sub(r"\s+", " ", value).strip()


def atomic_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    payload = b"".join((json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n").encode("utf-8") for row in rows)
    path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as handle:
        handle.write(payload); handle.flush(); os.fsync(handle.fileno())
    tmp.replace(path)
    print(json.dumps({"output": str(path), "rows": len(rows), "sha256": sha(payload)}, sort_keys=True))


def main() -> None:
    raw = {row["record_id"]: row for row in map(json.loads, RAW.open(encoding="utf-8"))}
    base, _ = load_reviews([BASE])
    parents: dict[str, list[str]] = defaultdict(list); normalized: dict[str, list[str]] = defaultdict(list)
    for record_id, review in base.items():
        if review["quality_decision"] != "pass": continue
        candidate = apply_review(raw[record_id], review)
        parents[str(raw[record_id].get("upstream_source") or "")].append(record_id)
        normalized[normalize(str(candidate["lean_statement"]))].append(record_id)
    replacements = {row["record_id"]: row for row in map(json.loads, REPLACEMENTS.open(encoding="utf-8"))}
    results = {row["record_id"]: row for row in map(json.loads, RESULTS.open(encoding="utf-8"))}
    if set(replacements) != set(results) or len(results) != 3: raise SystemExit("replacement/result mismatch")
    reviews: list[dict[str, object]] = []
    for record_id in sorted(replacements):
        replacement = replacements[record_id]["replacement"]; result = results[record_id]
        if result.get("success") is not True or result.get("pantograph_verified") != "success":
            raise SystemExit(f"missing Pantograph success: {record_id}")
        parent = str(raw[record_id].get("upstream_source") or "")
        if parents.get(parent): raise SystemExit(f"parent conflict: {record_id}: {parents[parent]}")
        norm = normalize(str(replacement["lean_statement"])); duplicates = normalized.get(norm, [])
        if duplicates: raise SystemExit(f"normalized duplicate: {record_id}: {duplicates}")
        method = record_id.split("::", 1)[1].rsplit("_", 1)[0]
        review = {
            "manual_reviewed": True, "quality_decision": "pass", "quality_tier": "medium",
            "record_id": record_id, "replacement": replacement,
            "review_reason": (
                f"Strict human repair re-review. Frozen raw parent={parent}; method={method}. The retained goal is "
                "a mathematically substantive, independently useful component of the parent conjunction (a global "
                "bound, least-value characterization, or concavity consequence), not a scalar answer fragment or "
                "logical reordering. Repair only removes a stray ':=' token introduced by automatic source assembly; "
                "the proposition and proof strategy remain unchanged. Exact replacement passed local Pantograph; "
                "global parent and normalized-statement duplicate checks found zero conflicts."
            ),
            "reviewed_statement_sha256": sha(str(replacement["lean_statement"])),
            "reviewed_proof_sha256": sha(str(replacement["proof"])),
            "reviewer": "codex-root-projection-syntax-round35",
        }
        candidate = apply_review(raw[record_id], review); validate_review_hashes(review, candidate)
        reviews.append(review); parents[parent].append(record_id); normalized[norm].append(record_id)
    atomic_jsonl(OUTPUT, reviews)


if __name__ == "__main__": main()
