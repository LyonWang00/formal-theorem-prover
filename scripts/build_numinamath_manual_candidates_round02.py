#!/usr/bin/env python3
"""Serialize and validate human-authored NuminaMath round02 candidates.

The script performs provenance, hash, marker, and format checks only.  It does
not generate mathematics, invoke Lean/Pantograph, or write verified data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "outputs/numinamath_expand_verification/manual_expansion_candidates"
SOURCE = CANDIDATE_DIR / "numinamath_manual_candidates_round02.lean"
OUT = CANDIDATE_DIR / "numinamath_manual_candidates_round02.jsonl"
REPORT = CANDIDATE_DIR / "numinamath_manual_candidates_round02.report.json"
PARENT_SOURCE = ROOT / "lean_prover/Dataset/verified_data/numinamath_verified_success.jsonl"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


METADATA: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "manual_round02_c01_punctured_affine_range",
        "marker": "C01",
        "theorem_name": "numinamath_manual_punctured_affine_range",
        "parent_record_id": "7a8b11a9-6c2b-545b-a91e-2860e926c9c0::row_046693",
        "parent_theorem_name": "algebra_15386",
        "parent_statement_sha256": "64ea34b5aa01bdb806ad1af257a6466b55b364177a889016cba7836ea961af79",
        "parent_proof_sha256": "95fcf02ad68a66666137d41b7539de8af2b14b545a957a0d712c035101cfa227",
        "manual_method": "generalize_removable_hole_range_to_arbitrary_affine_map",
        "problem": (
            "Let m,c,r be real numbers with m nonzero. Prove that the image of the punctured "
            "domain R\\{r} under x ↦ mx+c is exactly R\\{mr+c}."
        ),
        "quality_rationale": (
            "Abstracts the parent's one numerical canceled-factor range computation into the "
            "general theorem for every nonconstant affine map, including both soundness and "
            "surjectivity onto the punctured codomain."
        ),
        "manual_risk": "Low; check Set.image witness orientation and field_simp normalization.",
    },
    {
        "candidate_id": "manual_round02_c02_trig_equality_cases",
        "marker": "C02",
        "theorem_name": "numinamath_manual_trig_range_equality_cases",
        "parent_record_id": "d06e68ac-c4dd-5c31-8a6b-ac4942db9f34::row_037268",
        "parent_theorem_name": "inequalities_285163",
        "parent_statement_sha256": "c2e36d37d6930b9b8bb610dd09ce879cd9b5d74346ef38294384e2c682a99ac4",
        "parent_proof_sha256": "fe210f72e606390eedce9996f74f5fd58242d90bf607676b107a4fae5ae87cef",
        "manual_method": "add_exact_equality_classification_for_both_extremal_bounds",
        "problem": (
            "For real t, characterize equality at both endpoints of "
            "-5 ≤ 4 sin t + cos(2t) ≤ 3: the upper endpoint occurs exactly when sin t=1, "
            "and the lower endpoint exactly when sin t=-1."
        ),
        "quality_rationale": (
            "Completes both sharp endpoint cases rather than projecting either inequality. "
            "The lower endpoint proof additionally uses the range of sine to exclude the "
            "extraneous algebraic root."
        ),
        "manual_risk": "Low; check current names of the standard sine bounds.",
    },
    {
        "candidate_id": "manual_round02_c03_power_sum_exact_remainder",
        "marker": "C03",
        "theorem_name": "numinamath_manual_power_sum_mod_five",
        "parent_record_id": "59a633c1-9a3c-50ef-970a-5268a5d42f76::row_041265",
        "parent_theorem_name": "number_theory_257025",
        "parent_statement_sha256": "d23ca9cc430dea7eac76c584780af2a58c2c5b4ee4253f28422e561a4432263d",
        "parent_proof_sha256": "309d41fbe928b8023b46a891665216e7504e91ffc7cf719414c237357c2f0fd7",
        "manual_method": "strengthen_divisibility_iff_to_complete_modulo_five_remainder",
        "problem": (
            "For every natural n, determine the exact remainder modulo 5 of "
            "1^n+2^n+3^n+4^n: it is 4 when 4 divides n and 0 otherwise."
        ),
        "quality_rationale": (
            "Strictly strengthens the parent's positive-n divisibility equivalence to an exact "
            "four-residue classification, also handling n=0 correctly. The proof treats every "
            "exponent class modulo 4 explicitly."
        ),
        "manual_risk": "Medium; simp normalization of powers in the four induction lemmas needs compilation.",
    },
    {
        "candidate_id": "manual_round02_c04_two_point_domain",
        "marker": "C04",
        "theorem_name": "numinamath_manual_two_point_domain_decomposition",
        "parent_record_id": "7506be5d-071e-5b14-a58a-1b4fa03751cb::row_043178",
        "parent_theorem_name": "algebra_10507",
        "parent_statement_sha256": "2c62ead543acb276a9d89d6716e537aeb271f873a0d9cbb2d8d5a1e705507235",
        "parent_proof_sha256": "72a1d9999cd2a78568dbb420ba7d5a0ca045dab69f179f0407c9f1110598682e",
        "manual_method": "generalize_numeric_denominator_domain_to_two_arbitrary_ordered_roots",
        "problem": (
            "For real r<s, prove that the nonzero set of (x-r)(x-s) is the disjoint-order "
            "decomposition x<r, r<x<s, or s<x."
        ),
        "quality_rationale": (
            "Replaces the parent's fixed roots -2 and 2 by arbitrary ordered roots and proves "
            "the complete complement-of-two-points decomposition using order trichotomy."
        ),
        "manual_risk": "Low; check sub_ne_zero orientation in the three reverse branches.",
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
    wanted = {meta["parent_record_id"] for meta in METADATA}
    parents: dict[str, dict[str, Any]] = {}
    with PARENT_SOURCE.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record_id = str(row.get("record_id") or "")
            if record_id in wanted:
                parents[record_id] = row
    if set(parents) != wanted:
        raise ValueError(f"missing canonical parents: {sorted(wanted - set(parents))}")
    rows: list[dict[str, Any]] = []
    for meta in METADATA:
        parent = parents[meta["parent_record_id"]]
        if parent.get("statement_sha256") != meta["parent_statement_sha256"]:
            raise ValueError(f"parent statement hash drift for {meta['candidate_id']}")
        if parent.get("proof_sha256") != meta["parent_proof_sha256"]:
            raise ValueError(f"parent proof hash drift for {meta['candidate_id']}")
        if parent.get("pantograph_verified") != "success":
            raise ValueError(f"parent is not verified success: {meta['candidate_id']}")
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
        rows.append({
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
        })
    if len(rows) != 4 or len({row["candidate_id"] for row in rows}) != 4:
        raise ValueError("candidate coverage/uniqueness failure")
    round01_report = CANDIDATE_DIR / "numinamath_manual_candidates_round01.report.json"
    prior_parents: set[str] = set()
    if round01_report.exists():
        prior = json.loads(round01_report.read_text(encoding="utf-8"))
        prior_parents = set(prior.get("parent_record_ids", []))
    overlap = prior_parents & wanted
    if overlap:
        raise ValueError(f"round01 parent overlap: {sorted(overlap)}")
    report = {
        "schema_version": "numinamath_manual_candidate_round_report_v1",
        "round": "round02",
        "candidate_count": len(rows),
        "parent_count": len(wanted),
        "quality_candidate_high": len(rows),
        "low_quality_candidates": 0,
        "pantograph_jobs_started": 0,
        "pantograph_verified_rows": 0,
        "dataset_rows_written": 0,
        "canonical_parent_records_validated": len(parents),
        "round01_parent_overlap": 0,
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
