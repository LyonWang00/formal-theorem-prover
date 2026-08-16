#!/usr/bin/env python3
"""Build a provenance-rich manifest for four manually written Lean candidates.

This serializes human-authored theorem blocks.  It does not invoke Lean,
Pantograph, a model, or any expansion worker.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = (
    ROOT / "outputs/numinamath_expand_verification/manual_expansion_candidates"
)
SOURCE = CANDIDATE_DIR / "numinamath_manual_candidates_round01.lean"
OUT = CANDIDATE_DIR / "numinamath_manual_candidates_round01.jsonl"
REPORT = CANDIDATE_DIR / "numinamath_manual_candidates_round01.report.json"
PARENT_SOURCE = (
    ROOT / "lean_prover/Dataset/verified_data/numinamath_verified_success.jsonl"
)


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


METADATA: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "manual_round01_c01_cyclic_four",
        "marker": "C01",
        "theorem_name": "numinamath_manual_cyclic_four",
        "parent_record_id": "84f26e70-3dfd-589b-b7d0-7792576f0cc9::row_000000",
        "parent_theorem_name": "algebra_4013",
        "parent_statement_sha256": "61466565883b47c4dd3ae2c85031a6227060bd20da2026cc6600f8daad105aab",
        "parent_proof_sha256": "03d6cf2c4e7401adca0a134f8e648341a676feeac732aa5bdd36b183812205ed",
        "manual_method": "structural_cyclic_extension_three_to_four_variables",
        "problem": (
            "Let nonzero real numbers a,b,c,d satisfy abcd=1, and assume the displayed "
            "cyclic denominators are nonzero. Prove that the four cyclic fractions with "
            "denominators 1+a+ab+abc, 1+b+bc+bcd, 1+c+cd+cda, and 1+d+da+dab sum to 1."
        ),
        "quality_rationale": (
            "A genuine four-variable cyclic extension.  The proof transports the parent's "
            "common-denominator mechanism through three different nonzero scale factors; it "
            "does not alter a constant or split a conclusion."
        ),
        "manual_risk": (
            "Highest proof risk in this round: field_simp side conditions and multiplication "
            "associativity must be checked by Pantograph before promotion."
        ),
    },
    {
        "candidate_id": "manual_round01_c02_divisibility_equiv",
        "marker": "C02",
        "theorem_name": "numinamath_manual_divisibility_equiv",
        "parent_record_id": "aff8e6ca-7909-54cc-8414-b258fc4d1459::row_073657",
        "parent_theorem_name": "number_theory_177235",
        "parent_statement_sha256": "7eb6d132b113a33fa130058b79ee772f1ebdcd5c82b44113d5cd0afa9e55b0d5",
        "parent_proof_sha256": "59f29bfc5b58ee190937cce5749abf6bcac2021cc9a6ee0c8a20c0d66a8fc34d",
        "manual_method": "strengthen_one_way_modular_relation_to_equivalence",
        "problem": (
            "For all integers a,b, prove that 19 divides 11a+2b if and only if "
            "19 divides 18a+5b."
        ),
        "quality_rationale": (
            "Strictly strengthens the one-way parent and exposes that the two coefficient "
            "vectors differ by units modulo 19. Both directions use explicit integer identities."
        ),
        "manual_risk": "Low; verify witness orientation for Lean's divisibility constructor.",
    },
    {
        "candidate_id": "manual_round01_c03_amgm_equality_case",
        "marker": "C03",
        "theorem_name": "numinamath_manual_amgm_eq_iff",
        "parent_record_id": "c1c9e6b8-6063-5442-b21c-aa989b5d4262::row_099759",
        "parent_theorem_name": "inequalities_114406",
        "parent_statement_sha256": "3c2dfdc8800883ec07c9623e2705e17303246b6dc82b051a865fa019543379f8",
        "parent_proof_sha256": "d80c4d7718165b705daff4ba15401406b0da9d7fc2c7c167a9f2a0bf131d3052",
        "manual_method": "derive_exact_equality_case_from_nonnegative_square_proof",
        "problem": (
            "For positive real numbers a,b, prove that their arithmetic mean equals "
            "sqrt(ab) if and only if a=b."
        ),
        "quality_rationale": (
            "Adds the exact equality classification omitted by the parent inequality. The proof "
            "uses the vanishing of (sqrt a - sqrt b)^2 rather than a weaker order projection."
        ),
        "manual_risk": "Low to medium; check the current sqrt rewrite theorem names.",
    },
    {
        "candidate_id": "manual_round01_c04_cauchy_three_equality_case",
        "marker": "C04",
        "theorem_name": "numinamath_manual_cauchy_three_eq_iff",
        "parent_record_id": "90c7b6e2-ecb6-5b74-92f9-1b896847502f::row_090490",
        "parent_theorem_name": "inequalities_143050",
        "parent_statement_sha256": "3a70abf2f8a9a874921093edf4a2f4bcbd595d3a5ec5d5fcc6526fa2b232f994",
        "parent_proof_sha256": "0be1718e2cd3408c03ef1c7900f90656a5e61b63c1f9e59bbff4217cc639985c",
        "manual_method": "classify_equality_via_sum_of_three_nonnegative_terms",
        "problem": (
            "For positive real numbers a,b,c, prove that "
            "(a+b+c)(1/a+1/b+1/c)=9 if and only if a=b and b=c."
        ),
        "quality_rationale": (
            "Completes the three-variable inequality with a full equality-case classification. "
            "It requires a rational sum-of-squares identity and zero-sum reasoning."
        ),
        "manual_risk": "Medium; check div_eq_zero_iff and field_simp behavior before promotion.",
    },
)


def theorem_block(source: str, marker: str) -> str:
    start_token = f"-- CANDIDATE {marker} START\n"
    end_token = f"\n-- CANDIDATE {marker} END"
    if source.count(start_token) != 1 or source.count(end_token) != 1:
        raise ValueError(f"marker drift for {marker}")
    return source.split(start_token, 1)[1].split(end_token, 1)[0].strip()


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = SOURCE.read_text(encoding="utf-8")
    source_sha = digest(source)
    wanted_parents = {meta["parent_record_id"] for meta in METADATA}
    parents: dict[str, dict[str, Any]] = {}
    with PARENT_SOURCE.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record_id = str(row.get("record_id") or "")
            if record_id in wanted_parents:
                parents[record_id] = row
    if set(parents) != wanted_parents:
        raise ValueError(f"missing canonical parents: {sorted(wanted_parents - set(parents))}")
    rows: list[dict[str, Any]] = []
    for meta in METADATA:
        parent = parents[meta["parent_record_id"]]
        if parent.get("statement_sha256") != meta["parent_statement_sha256"]:
            raise ValueError(f"parent statement hash drift for {meta['candidate_id']}")
        if parent.get("proof_sha256") != meta["parent_proof_sha256"]:
            raise ValueError(f"parent proof hash drift for {meta['candidate_id']}")
        if parent.get("pantograph_verified") != "success":
            raise ValueError(f"canonical parent is not verified success: {meta['candidate_id']}")
        block = theorem_block(source, meta["marker"])
        delimiter = " := by\n"
        if delimiter not in block:
            raise ValueError(f"cannot split statement/proof for {meta['candidate_id']}")
        statement_body, proof_body = block.split(delimiter, 1)
        if not statement_body.startswith(f"theorem {meta['theorem_name']}"):
            raise ValueError(f"theorem name drift for {meta['candidate_id']}")
        lean_statement = "import Mathlib\n\n" + statement_body.strip()
        proof = "by\n" + proof_body.rstrip() + "\n"
        formal_proof = "import Mathlib\n\n" + block.rstrip() + "\n"
        row = {
            "manifest_schema_version": "numinamath_manual_candidate_v1",
            "candidate_kind": "human_authored_uncompiled_expansion",
            "candidate_id": meta["candidate_id"],
            "status": "manual_proof_written_pantograph_not_started",
            "dataset_record_ready": False,
            "pantograph_started": False,
            "parent_record_id": meta["parent_record_id"],
            "parent_theorem_name": meta["parent_theorem_name"],
            "parent_statement_sha256": meta["parent_statement_sha256"],
            "parent_proof_sha256": meta["parent_proof_sha256"],
            "manual_method": meta["manual_method"],
            "problem": meta["problem"],
            "theorem_name": meta["theorem_name"],
            "lean_statement": lean_statement,
            "proof": proof,
            "formal_proof": formal_proof,
            "statement_sha256": digest(lean_statement.strip()),
            "proof_sha256": digest(proof.strip()),
            "formal_proof_sha256": digest(formal_proof.strip()),
            "quality_decision": "candidate_high_pending_compile_and_second_review",
            "quality_rationale": meta["quality_rationale"],
            "manual_risk": meta["manual_risk"],
            "candidate_source": str(SOURCE.relative_to(ROOT)),
            "candidate_source_sha256": source_sha,
        }
        rows.append(row)
    if len(rows) != 4 or len({row["candidate_id"] for row in rows}) != 4:
        raise ValueError("candidate coverage/uniqueness failure")
    report = {
        "schema_version": "numinamath_manual_candidate_round_report_v1",
        "round": "round01",
        "candidate_count": len(rows),
        "parent_count": len({row["parent_record_id"] for row in rows}),
        "quality_candidate_high": len(rows),
        "low_quality_candidates": 0,
        "pantograph_jobs_started": 0,
        "pantograph_verified_rows": 0,
        "dataset_rows_written": 0,
        "canonical_parent_records_validated": len(parents),
        "canonical_parent_source": str(PARENT_SOURCE.relative_to(ROOT)),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_sha,
        "candidate_ids": [row["candidate_id"] for row in rows],
        "parent_record_ids": [row["parent_record_id"] for row in rows],
    }
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows, report = build()
    manifest_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    report["manifest"] = str(OUT.relative_to(ROOT))
    report["manifest_sha256"] = digest(manifest_text)
    report_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.check:
        if not OUT.exists() or not REPORT.exists():
            raise FileNotFoundError("manifest/report missing")
        if OUT.read_text(encoding="utf-8") != manifest_text:
            raise ValueError("manifest is not a deterministic rebuild")
        if REPORT.read_text(encoding="utf-8") != report_text:
            raise ValueError("report is not a deterministic rebuild")
    else:
        if not args.force and (OUT.exists() or REPORT.exists()):
            raise FileExistsError("refusing to overwrite without --force")
        OUT.write_text(manifest_text, encoding="utf-8")
        REPORT.write_text(report_text, encoding="utf-8")
    print(json.dumps({
        "mode": "check" if args.check else "write",
        "candidate_count": len(rows),
        "manifest": str(OUT.relative_to(ROOT)),
        "manifest_sha256": digest(manifest_text),
        "report": str(REPORT.relative_to(ROOT)),
        "report_sha256": digest(report_text),
        "source_sha256": report["source_sha256"],
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
