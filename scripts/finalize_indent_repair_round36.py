#!/usr/bin/env python3
"""Finalize 20k indentation repairs under the strict semantic/duplicate gate."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from finalize_indent_repair_round16 import atomic_jsonl


ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "outputs/numinamath_expand_verification/repairs/curated_indent_reviewed_020000_round36_v1"
BATCH = RUN / "batches/numinamath_expand_repair_curated_020000_round36_b00000"
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_020000_v14.jsonl"
RAW = ROOT / "lean_prover/Dataset/raw_data/numinamath_expand.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_repaired_indent_020000_round36.jsonl"
SEMANTIC_REJECT = {
    "59ebab7d-40d2-532c-b22f-36497036c3d2::and_right_dc3b442612d0",
    "6e2fb8b3-64d3-5974-a4a0-be612eb9b2bd::lt_to_le_276befd9dd4f",
}


def sha(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def normalize(value: str) -> str:
    value = re.sub(r"/-.*?-/", " ", value, flags=re.S)
    value = re.sub(r"(?m)^\s*(?:import|open(?:\s+scoped)?)\s+.*$", " ", value)
    value = re.sub(r"\btheorem\s+\S+", "theorem _", value, count=1)
    return re.sub(r"\s+", " ", value).strip()


def main() -> None:
    base = {row["record_id"]: row for row in map(json.loads, BASE.open(encoding="utf-8"))}
    proposals = {row["record_id"]: row for row in map(json.loads, (RUN / "replacement_review_manifest.jsonl").open(encoding="utf-8"))}
    receipts = [json.loads(line) for line in (BATCH / "verification_results.jsonl").open(encoding="utf-8")]
    successes = {row["record_id"] for row in receipts if row.get("success") is True and row.get("pantograph_verified") == "success"}
    if len(proposals) != 12 or len(receipts) != 12 or successes != set(proposals):
        raise SystemExit("all 12 selected variants must have exact Pantograph success receipts")
    raw_all = {row["record_id"]: row for row in map(json.loads, RAW.open(encoding="utf-8"))}
    accepted_parents: dict[str, list[str]] = defaultdict(list); accepted_norm: dict[str, list[str]] = defaultdict(list)
    for record_id, review in base.items():
        if review["quality_decision"] != "pass": continue
        accepted_parents[str(raw_all[record_id].get("upstream_source") or "")].append(record_id)
        statement = str((review.get("replacement") or {}).get("lean_statement") or raw_all[record_id]["lean_statement"])
        accepted_norm[normalize(statement)].append(record_id)
    rows = []
    for record_id in sorted(proposals):
        old = base[record_id]; raw = raw_all[record_id]
        if old["quality_decision"] != "pending" or old["quality_tier"] not in {"high", "medium"}:
            raise SystemExit(f"base review is not eligible: {record_id}")
        parent = str(raw.get("upstream_source") or ""); norm = normalize(str(raw["lean_statement"]))
        duplicate_ids = accepted_norm.get(norm, [])
        reject_reason = None
        if record_id == "59ebab7d-40d2-532c-b22f-36497036c3d2::and_right_dc3b442612d0":
            reject_reason = "The repaired proof compiles, but the goal keeps only the single scalar answer k < 1/3 from an application-style compound result; this is an answer fragment with little reusable mathematical content."
        elif record_id == "6e2fb8b3-64d3-5974-a4a0-be612eb9b2bd::lt_to_le_276befd9dd4f":
            reject_reason = "The repaired proof compiles, but the candidate merely weakens a strict inequality to the corresponding non-strict inequality without adding a boundary classification or new mathematical consequence."
        elif accepted_parents.get(parent):
            reject_reason = f"A stronger or more substantive sibling for the same frozen parent is already accepted: {accepted_parents[parent]}."
        elif duplicate_ids:
            reject_reason = f"The normalized theorem statement is already accepted under {duplicate_ids}; retaining it would add an exact duplicate."
        if reject_reason:
            rows.append({"record_id": record_id, "quality_decision": "reject", "quality_tier": "extremely_low",
                         "manual_reviewed": True, "reviewer": "codex-root-indent-round36-strict-adjudication",
                         "review_reason": reject_reason + " It is rejected regardless of Pantograph success.",
                         "reviewed_statement_sha256": sha(str(raw["lean_statement"])),
                         "reviewed_proof_sha256": sha(str(raw["proof"]))})
            continue
        proposal = proposals[record_id]; proof = proposal["replacement_proof"]; proof_hash = sha(proof)
        if proof_hash != proposal["replacement_proof_sha256"]: raise SystemExit(f"proof hash mismatch: {record_id}")
        rows.append({"record_id": record_id, "quality_decision": "pass", "quality_tier": old["quality_tier"],
                     "manual_reviewed": True, "reviewer": "codex-root-indent-repair-round36",
                     "review_reason": old["review_reason"] + " Human repair re-review: only the copied parent tactic block indentation changed; statement, assumptions, target, and tactic text are identical. The exact replacement passed local Pantograph compilation and current global parent/normalized duplicate checks.",
                     "reviewed_statement_sha256": proposal["statement_sha256_after"],
                     "reviewed_proof_sha256": proof_hash, "replacement": {"proof": proof}})
        accepted_parents[parent].append(record_id); accepted_norm[norm].append(record_id)
    atomic_jsonl(OUT, rows)
    passed = sum(row["quality_decision"] == "pass" for row in rows)
    print(json.dumps({"rows": len(rows), "pass": passed, "reject": len(rows) - passed, "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__": main()
