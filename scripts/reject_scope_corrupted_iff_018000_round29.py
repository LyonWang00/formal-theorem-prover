#!/usr/bin/env python3
"""Reject Iff expansions whose parent Iff was split inside a binder or And."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from finalize_indent_repair_round16 import atomic_jsonl


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "lean_prover/Dataset/raw_data/numinamath_expand.jsonl"
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_018000_v12.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_scope_reject_018000_round29.jsonl"
IDS = {
    "70402ffc-352f-55cd-9823-bcbc1476d706::iff_forward_39acd30851e4",
    "2892300e-2945-581b-9c49-4ebc7cab9b45::iff_forward_7ea1ed12f080",
    "4bb68724-a225-5e0c-b510-afca00aa437c::iff_forward_d7d1a263640c",
    "7731fc41-b3cd-5f22-a6c9-0d409605c071::iff_forward_e90734c59229",
    "42891135-6d1d-5e12-bd02-f8092adbfa20::iff_forward_0b8c460e2ba2",
    "6045389c-9eeb-580f-9252-90e1657d33e4::iff_forward_f045853c9c6e",
    "67d31e65-ddc6-5fa0-8d5a-01fc488e6577::iff_forward_47b9224bc970",
    "4318684b-52ce-5564-a835-cf2763791254::iff_forward_1d218eab9ed5",
    "32f2f8a2-3905-5fb9-8a72-0af4134ad3d6::iff_forward_74c3a85746c5",
    "10a75152-b335-5cb7-ba46-580f1cdfde02::iff_forward_497afa8adbdc",
    "6732a16a-2f37-51b4-97c7-690366843e95::iff_forward_a0bbe3294b15",
    "55814029-9837-5025-9730-018c2682045e::iff_forward_a3c35c17a249",
    "96520dbc-cf4e-5253-ac7c-898ff1a914f4::iff_forward_1af09ff13f60",
}


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    base = {row["record_id"]: row for row in map(json.loads, BASE.open(encoding="utf-8")) if row["record_id"] in IDS}
    raw = {row["record_id"]: row for row in map(json.loads, RAW.open(encoding="utf-8")) if row["record_id"] in IDS}
    if set(base) != IDS or set(raw) != IDS:
        raise SystemExit("scope-reject IDs do not match the frozen checkpoint/raw source")
    if any(row["quality_decision"] != "pending" for row in base.values()):
        raise SystemExit("scope-reject override may only replace pending reviews")
    rows = []
    for record_id in sorted(IDS):
        extra = ""
        if record_id.startswith("4bb68724-"):
            extra = " The parent assumptions are also inconsistent at zero, so the copied proof obtains the result vacuously."
        rows.append(
            {
                "record_id": record_id,
                "quality_decision": "reject",
                "quality_tier": "extremely_low",
                "manual_reviewed": True,
                "reviewer": "codex-root-scope-adjudication-round29",
                "review_reason": (
                    "Strict re-review found that the generator split an Iff inside an Exists, Forall, or And scope. "
                    "Lean therefore types h_parent as an existential/conjunction containing an Iff rather than as the intended top-level equivalence; "
                    "the candidate proof cannot derive the classification from an arbitrary candidate premise. This is an AST-scope corruption, not a repairable proof typo, and is rejected."
                    + extra
                ),
                "reviewed_statement_sha256": sha(raw[record_id]["lean_statement"]),
                "reviewed_proof_sha256": sha(raw[record_id]["proof"]),
            }
        )
    atomic_jsonl(OUT, rows)
    print(json.dumps({"rows": len(rows), "reject": len(rows), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
