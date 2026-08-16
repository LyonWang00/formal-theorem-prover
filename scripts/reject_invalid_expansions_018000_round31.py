#!/usr/bin/env python3
"""Reject remaining 18k expansions with scope, order, or declaration corruption."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from finalize_indent_repair_round16 import atomic_jsonl


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "lean_prover/Dataset/raw_data/numinamath_expand.jsonl"
BASE = ROOT / "outputs/numinamath_expand_verification/manual_review_clean/reviewed_018000_v12a.jsonl"
OUT = ROOT / "outputs/numinamath_expand_verification/manual_review_repairs/review_invalid_reject_018000_round31.jsonl"

SCOPE_IFF = {
    "0788d074-fc89-5e78-b792-25cb8ff0aa9b::iff_forward_da055dc12c80",
    "0e8f4f45-e753-5f9f-a645-8c30eb56ce2d::iff_forward_4df20471993b",
    "4d86ae4c-b6d3-5f90-8c39-ffa23c3fa637::iff_forward_c2e386f3596d",
    "61409410-f17b-55b7-8a5a-14f7709b52f4::iff_forward_1bcd442e49da",
    "736402f7-adb2-5afb-95fe-d2bfb073b17a::iff_forward_0b42066a20d0",
    "87b940b0-89ea-5d64-adda-aac010eff146::iff_forward_2437059bef35",
    "aecff845-2cf0-58ef-be11-ce7a544a1511::iff_forward_f3eac0a045d5",
    "c1a5fd69-70d6-5abb-b998-75da37a119ab::iff_forward_e3730fc31638",
    "cb35d3b1-04b5-5a17-a88b-d0d2cf2ccb28::iff_forward_55df6b6c2e23",
    "cc59c3ae-9a88-5c5a-b5ed-b22e7873554b::iff_forward_8032c0ec9875",
    "cd78a7ff-2a7a-52a3-a9df-c8eec7722459::iff_forward_3faa9aa771cb",
    "e6da95f8-5344-51e0-9980-88be2c4f20b8::iff_forward_db8a4845edeb",
    "f9517006-3bc9-5149-9af9-477cbf5abe9b::iff_forward_c2c1589a8ef4",
    "fbb8da51-a539-5c05-96f7-289594c73334::iff_forward_975090e3506e",
}
FREE_OR_PROP_ORDER = {
    "1d9f0c11-9dae-59ff-806c-9bc822057875::le_to_lt_or_eq_1b34900c229e",
    "47f36b41-9eb4-5002-8563-074ea30d8607::le_to_lt_or_eq_6e690d21c252",
    "5ab8cc75-ae73-589a-b69d-d9c6c96d0237::le_to_lt_or_eq_57eb8fb2795b",
    "7161ac2c-71fa-59f3-a16f-245bf0d94331::le_to_lt_or_eq_c221c471e345",
    "7b33f878-514e-5f3d-a3fe-9aad2cdbb636::le_to_lt_or_eq_947248cc5c6a",
    "a7826ac5-683e-5ab7-ae83-e150e349255d::le_to_lt_or_eq_7bee72eeb62c",
    "d666b94b-3eb9-5629-b9d4-5dae46a7effe::le_to_lt_or_eq_74b155351c31",
}
DECLARATION_POLLUTION = {
    "3c3795c7-78b9-559b-876a-fd50fb0b0059::and_left_f19f74032517",
    "4b5b6abf-6043-5642-bf18-9ffeced35cdf::and_left_46a8a10fa81d",
    "7d53e48b-3873-5b97-b7b4-1ab992610ce2::and_left_7739d98ddaed",
    "d13efd3b-9267-5af5-aae9-c0920eb88ebf::and_right_274c910eaaa4",
    "da6c1e4b-d060-5bd3-b87d-894db7ea7a67::and_left_1579afcfa37c",
    "f83545a0-afa3-5589-8f91-34ef8d4e622f::and_left_de281b7269c7",
}
IDS = SCOPE_IFF | FREE_OR_PROP_ORDER | DECLARATION_POLLUTION


def sha(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def reason(record_id: str) -> str:
    if record_id in SCOPE_IFF:
        return (
            "Strict re-review found that the generator split the parent Iff inside an existential, quantified, or conjunction scope. "
            "The copied h_parent is therefore not the intended top-level equivalence, so the training pair is structurally mismatched and is rejected."
        )
    if record_id in FREE_OR_PROP_ORDER:
        return (
            "Strict re-review found a relation cut inside a quantifier or proposition. The generated target contains free identifiers or places a Prop-valued fragment in numeric order, so it is malformed and rejected."
        )
    return (
        "Strict re-review found a stray ':= by' or adjacent declaration embedded in the generated theorem header/target. "
        "This is declaration-concatenation pollution rather than a meaningful theorem and is rejected."
    )


def main() -> None:
    base = {row["record_id"]: row for row in map(json.loads, BASE.open(encoding="utf-8")) if row["record_id"] in IDS}
    raw = {row["record_id"]: row for row in map(json.loads, RAW.open(encoding="utf-8")) if row["record_id"] in IDS}
    if set(base) != IDS or set(raw) != IDS:
        raise SystemExit("invalid-expansion IDs do not match checkpoint/raw source")
    if any(row["quality_decision"] != "pending" for row in base.values()):
        raise SystemExit("invalid-expansion override may only replace pending reviews")
    rows = [
        {
            "record_id": record_id,
            "quality_decision": "reject",
            "quality_tier": "extremely_low",
            "manual_reviewed": True,
            "reviewer": "codex-root-invalid-expansion-adjudication-round31",
            "review_reason": reason(record_id),
            "reviewed_statement_sha256": sha(raw[record_id]["lean_statement"]),
            "reviewed_proof_sha256": sha(raw[record_id]["proof"]),
        }
        for record_id in sorted(IDS)
    ]
    atomic_jsonl(OUT, rows)
    print(json.dumps({"rows": len(rows), "reject": len(rows), "sha256": hashlib.sha256(OUT.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
