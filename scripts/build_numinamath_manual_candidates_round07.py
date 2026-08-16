#!/usr/bin/env python3
"""Serialize and validate human-authored NuminaMath round07 candidates only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "outputs/numinamath_expand_verification/manual_expansion_candidates"
SOURCE = CANDIDATE_DIR / "numinamath_manual_candidates_round07.lean"
OUT = CANDIDATE_DIR / "numinamath_manual_candidates_round07.jsonl"
REPORT = CANDIDATE_DIR / "numinamath_manual_candidates_round07.report.json"
PARENT_SOURCE = ROOT / "lean_prover/Dataset/verified_data/numinamath_verified_success.jsonl"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def normalized_statement(text: str) -> str:
    text = re.sub(r"^\s*import\s+Mathlib\s*", "", text)
    text = re.sub(r"theorem\s+[A-Za-z0-9_']+", "theorem", text, count=1)
    return re.sub(r"\s+", "", text).lower()


METADATA: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "manual_round07_c01_antiperiod_period",
        "marker": "C01",
        "theorem_name": "numinamath_manual_antiperiod_implies_period",
        "parent_record_id": "a2fc2712-3e37-5acf-bbb1-18915ad73a4a::row_000346",
        "parent_theorem_name": "algebra_9042",
        "parent_statement_sha256": "3d0be6dac05f9e51021c14f91c9323a085af57c19885b1db827aab076967a7ee",
        "parent_proof_sha256": "9d141c0a7420656a53495e25f9f8c188325bff9664664f0436f4a8564e2c5240",
        "manual_method": "generalize_fixed_antiperiod_to_parameterized_period_structure",
        "problem": (
            "If a real-valued function satisfies f(x)=-f(x+p) for every real x, prove "
            "that 2p is a period: f(x+2p)=f(x) for every x."
        ),
        "quality_rationale": (
            "Extracts and parameterizes the structural mechanism behind the parent's repeated "
            "two-unit shifts. It removes the irrelevant parity and interval formula assumptions "
            "and proves a global functional law rather than another point evaluation."
        ),
        "manual_risk": "Low; two instances of antiperiodicity and linear arithmetic suffice.",
    },
    {
        "candidate_id": "manual_round07_c02_quadratic_form_endpoints",
        "marker": "C02",
        "theorem_name": "numinamath_manual_quadratic_form_endpoint_classification",
        "parent_record_id": "359f2631-afce-56e1-975d-19ab577cbbb6::row_007283",
        "parent_theorem_name": "algebra_305870",
        "parent_statement_sha256": "d266a5f9dd0bf8dadb13b879c2c73f71a20e08de796ed8d5941adc5ef032287d",
        "parent_proof_sha256": "ea531bcc1484d52d6a078e0cf523fbd501362f6bbd094dfe3f3219101a34fb64",
        "manual_method": "classify_both_endpoint_equality_cases_of_quadratic_form_range",
        "problem": (
            "Under a^2+ab+b^2=1, classify both endpoint cases of "
            "a^2-ab+b^2: the lower endpoint 1/3 occurs exactly at a=b, and the upper "
            "endpoint 3 exactly at a=-b."
        ),
        "quality_rationale": (
            "Completes the parent's range bounds by identifying every equality case at both "
            "sharp endpoints. The two classifications arise from distinct square certificates "
            "and are not restatements of the inequalities."
        ),
        "manual_risk": "Low; each direction is polynomial arithmetic from the constraint and one square.",
    },
    {
        "candidate_id": "manual_round07_c03_nonnegative_triple_sum",
        "marker": "C03",
        "theorem_name": "numinamath_manual_nonnegative_triple_sum_formula",
        "parent_record_id": "464663e7-2107-5bf1-9f58-ee5abf820ed1::row_007738",
        "parent_theorem_name": "algebra_20734",
        "parent_statement_sha256": "4bcffaa61f5cab8c871f7114fc6bb40cd5a749da3fa06b0c0d78a9ca1e19bd1d",
        "parent_proof_sha256": "5a637ca245b0ecc529e323533736c384f58a9b094c95796f6c9031e8834b1e08",
        "manual_method": "parameterize_numeric_sum_problem_as_exact_square_root_formula",
        "problem": (
            "For nonnegative real a,b,c with square sum S and pair-product sum P, prove "
            "the exact formula a+b+c=sqrt(S+2P)."
        ),
        "quality_rationale": (
            "Generalizes both numerical invariants in the parent and returns the complete closed "
            "formula. Nonnegativity selects the positive square root, so the statement records "
            "the essential branch information rather than merely squaring the target."
        ),
        "manual_risk": "Low; an exact expansion plus the nonnegative sqrt-square identity proves the result.",
    },
    {
        "candidate_id": "manual_round07_c04_alternating_cubes_sum",
        "marker": "C04",
        "theorem_name": "numinamath_manual_alternating_cubes_sum",
        "parent_record_id": "7c49bfd6-c046-5f93-9e6d-f363ec338a90::row_002567",
        "parent_theorem_name": "algebra_95560",
        "parent_statement_sha256": "ac1484ec02910d781e2c30b7f11fb30565e022048915b09554460a44dd4593d1",
        "parent_proof_sha256": "7c7fce6f4579fc2bd506ec01b88b2a370059a499334d9cb8bb404480a10cda05",
        "manual_method": "generalize_fixed_nine_term_computation_to_closed_formula_for_all_lengths",
        "problem": (
            "For every natural n, evaluate the first n differences of consecutive odd/even "
            "cubes and prove their sum is n^2(4n+3)."
        ),
        "quality_rationale": (
            "Replaces the parent's one-off nine-term normalization with a closed theorem for "
            "every length and an induction proof. The integer codomain avoids truncated "
            "subtraction and makes the algebraic structure explicit."
        ),
        "manual_risk": "Low to medium; later elaboration should confirm cast normalization in the induction step.",
    },
)


def theorem_block(source: str, marker: str) -> str:
    start = f"-- CANDIDATE {marker} START\n"
    end = f"\n-- CANDIDATE {marker} END"
    if source.count(start) != 1 or source.count(end) != 1:
        raise ValueError(f"marker drift for {marker}")
    return source.split(start, 1)[1].split(end, 1)[0].strip()


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = SOURCE.read_text(encoding="utf-8")
    source_sha = digest(source)
    wanted = {meta["parent_record_id"] for meta in METADATA}
    parents: dict[str, dict[str, Any]] = {}
    with PARENT_SOURCE.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            if row.get("record_id") in wanted:
                parents[str(row["record_id"])] = row
    if set(parents) != wanted:
        raise ValueError(f"missing canonical parents: {sorted(wanted - set(parents))}")

    rows: list[dict[str, Any]] = []
    for meta in METADATA:
        parent = parents[meta["parent_record_id"]]
        if parent.get("statement_sha256") != meta["parent_statement_sha256"]:
            raise ValueError(f"parent statement hash drift: {meta['candidate_id']}")
        if parent.get("proof_sha256") != meta["parent_proof_sha256"]:
            raise ValueError(f"parent proof hash drift: {meta['candidate_id']}")
        if parent.get("pantograph_verified") != "success":
            raise ValueError(f"parent is not Pantograph success: {meta['candidate_id']}")
        block = theorem_block(source, meta["marker"])
        statement_body, proof_body = block.split(" := by\n", 1)
        if not statement_body.startswith(f"theorem {meta['theorem_name']}"):
            raise ValueError(f"theorem name drift: {meta['candidate_id']}")
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
    if len({row["parent_record_id"] for row in rows}) != 4:
        raise ValueError("round07 parent uniqueness failure")

    prior_parents: set[str] = set()
    prior_normalized: set[str] = set()
    prior_locks: list[dict[str, Any]] = []
    for index in range(1, 7):
        round_name = f"round{index:02d}"
        manifest = CANDIDATE_DIR / f"numinamath_manual_candidates_{round_name}.jsonl"
        report_path = CANDIDATE_DIR / f"numinamath_manual_candidates_{round_name}.report.json"
        payload = manifest.read_bytes()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        actual_sha = digest_bytes(payload)
        if actual_sha != report.get("manifest_sha256"):
            raise ValueError(f"prior manifest/report SHA drift: {round_name}")
        prior_locks.append({"round": round_name, "path": str(manifest.relative_to(ROOT)),
                            "rows": len(payload.splitlines()), "sha256": actual_sha})
        prior_parents.update(report.get("parent_record_ids", []))
        for line in payload.splitlines():
            prior_normalized.add(normalized_statement(json.loads(line)["lean_statement"]))
    overlap = prior_parents & wanted
    if overlap:
        raise ValueError(f"prior-round parent overlap: {sorted(overlap)}")
    normalized = [normalized_statement(row["lean_statement"]) for row in rows]
    if len(set(normalized)) != 4 or set(normalized) & prior_normalized:
        raise ValueError("normalized statement duplicate")

    report = {
        "schema_version": "numinamath_manual_candidate_round_report_v1",
        "round": "round07", "candidate_count": 4, "parent_count": 4,
        "quality_candidate_high": 4, "low_quality_candidates": 0,
        "lake_jobs_started": 0, "pantograph_jobs_started": 0,
        "pantograph_verified_rows": 0, "dataset_rows_written": 0,
        "canonical_parent_records_validated": 4,
        "prior_round_parent_overlap": 0,
        "prior_round_normalized_statement_overlap": 0,
        "within_round_normalized_statement_duplicates": 0,
        "canonical_parent_source": str(PARENT_SOURCE.relative_to(ROOT)),
        "canonical_parent_source_sha256": digest_bytes(PARENT_SOURCE.read_bytes()),
        "source": str(SOURCE.relative_to(ROOT)), "source_sha256": source_sha,
        "candidate_ids": [row["candidate_id"] for row in rows],
        "parent_record_ids": [row["parent_record_id"] for row in rows],
        "statement_sha256": [row["statement_sha256"] for row in rows],
        "proof_sha256": [row["proof_sha256"] for row in rows],
        "formal_proof_sha256": [row["formal_proof_sha256"] for row in rows],
        "prior_round_manifests": prior_locks,
    }
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    rows, report = build()
    manifest_text = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows)
    report["manifest"] = str(OUT.relative_to(ROOT))
    report["manifest_sha256"] = digest(manifest_text)
    report_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.check:
        if OUT.read_text(encoding="utf-8") != manifest_text or REPORT.read_text(encoding="utf-8") != report_text:
            raise ValueError("round07 artifacts are not a deterministic rebuild")
    else:
        if not args.force and (OUT.exists() or REPORT.exists()):
            raise FileExistsError("refusing to overwrite without --force")
        OUT.write_text(manifest_text, encoding="utf-8")
        REPORT.write_text(report_text, encoding="utf-8")
    print(json.dumps({"mode": "check" if args.check else "write", "candidate_count": 4,
        "manifest": str(OUT.relative_to(ROOT)), "manifest_sha256": digest(manifest_text),
        "report": str(REPORT.relative_to(ROOT)), "report_sha256": digest(report_text),
        "source_sha256": report["source_sha256"]}, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
