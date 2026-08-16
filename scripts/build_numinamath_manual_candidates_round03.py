#!/usr/bin/env python3
"""Serialize and validate human-authored NuminaMath round03 candidates only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "outputs/numinamath_expand_verification/manual_expansion_candidates"
SOURCE = CANDIDATE_DIR / "numinamath_manual_candidates_round03.lean"
OUT = CANDIDATE_DIR / "numinamath_manual_candidates_round03.jsonl"
REPORT = CANDIDATE_DIR / "numinamath_manual_candidates_round03.report.json"
PARENT_SOURCE = ROOT / "lean_prover/Dataset/verified_data/numinamath_verified_success.jsonl"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


METADATA: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "manual_round03_c01_sharp_quadratic_equality",
        "marker": "C01",
        "theorem_name": "numinamath_manual_sharp_quadratic_equality",
        "parent_record_id": "00c57796-38f4-5198-a835-ecee67c26697::row_038934",
        "parent_theorem_name": "inequalities_265748",
        "parent_statement_sha256": "71919ea21d2a942a402a2440aeb578d3e038c73f32416ff56720e12e37296472",
        "parent_proof_sha256": "e1e00a4df434e89deace93160727074b828fefaac8d7907ffd74eb566449eb00",
        "manual_method": "classify_equality_for_the_sharp_universal_constant",
        "problem": (
            "For arbitrary real a,b, prove that equality in the sharp k=5/2 quadratic "
            "inequality holds if and only if a=b."
        ),
        "quality_rationale": (
            "Adds the exact equality case for the parent's optimal constant and strengthens the "
            "domain from positive reals to all reals. The conclusion follows from a genuine "
            "positive multiple of (a-b)^2."
        ),
        "manual_risk": "Low; the proof is a direct normalized-square argument.",
    },
    {
        "candidate_id": "manual_round03_c02_power_difference_divisibility",
        "marker": "C02",
        "theorem_name": "numinamath_manual_power_difference_dvd",
        "parent_record_id": "30302a97-007d-541d-9580-c59f4df8b632::row_032008",
        "parent_theorem_name": "number_theory_234852",
        "parent_statement_sha256": "07ae8f6867ff1e5614364b8c0d5c2b4941efc9ee4ce011d3ca1073a28a2332ed",
        "parent_proof_sha256": "652b6af02c9180c88ac3a5e6e138b4c415b56dc601b9df8c797260fdfa064187",
        "manual_method": "general_divisibility_lift_from_base_power_to_all_multiples",
        "problem": (
            "If an integer m divides x^k-y^k, prove that for every natural n it divides "
            "x^(kn)-y^(kn)."
        ),
        "quality_rationale": (
            "Extracts the reusable induction mechanism behind the parent's two numerical power "
            "divisibilities. The result is parameterized over modulus, bases, base exponent, and "
            "all exponent multiples."
        ),
        "manual_risk": "Low to medium; check pow_add normalization in the induction step.",
    },
    {
        "candidate_id": "manual_round03_c03_iunion_absorption_iff",
        "marker": "C03",
        "theorem_name": "numinamath_manual_inter_iUnion_absorption_iff",
        "parent_record_id": "a1613fba-6853-5d13-918e-9f21b8c9717b::row_037979",
        "parent_theorem_name": "algebra_12636",
        "parent_statement_sha256": "78ca398e3de0463b6cb54f75a1f834b9d9c7f1f46703b8876f675819b1b7e50a",
        "parent_proof_sha256": "999d8fe23f21a9d9600c35d474014f6fb6d3e1d2569c15dd287be4c03eb132bb",
        "manual_method": "generalize_two_set_absorption_equivalence_to_arbitrary_indexed_union",
        "problem": (
            "For a set A and an arbitrary indexed family s, prove that A intersect the union of "
            "the family equals that union if and only if every s_i is a subset of A."
        ),
        "quality_rationale": (
            "Generalizes the parent's two concrete subsets over real numbers to any element type, "
            "any index type, and an arbitrary family while retaining both logical directions."
        ),
        "manual_risk": "Low; verify iUnion membership theorem names after Mathlib rebuild.",
    },
    {
        "candidate_id": "manual_round03_c04_functional_equation_full_classification",
        "marker": "C04",
        "theorem_name": "numinamath_manual_even_functional_equation_classification",
        "parent_record_id": "3273c12a-cede-5578-9335-68f23a28090f::row_039477",
        "parent_theorem_name": "algebra_201896",
        "parent_statement_sha256": "6964684dc58a1ef6122dadd9c1df6409fad17141e5e5ae1d3280f179b2176bc7",
        "parent_proof_sha256": "00945369486cf2426b073c75c8467fc8f93042fcc3b7ef20cf49ee8a7824dc78",
        "manual_method": "strengthen_zero_set_answer_to_complete_function_classification",
        "problem": (
            "Classify every even real function satisfying f(x+y)=f(x)+f(y)+xy+2014: "
            "prove f(x)=x^2/2-2014 for all x."
        ),
        "quality_rationale": (
            "Strictly strengthens the parent's two-root answer to an explicit formula for the "
            "entire function. The proof first determines f(0), then uses the symmetric pair x,-x."
        ),
        "manual_risk": "Low; only arithmetic normalization and the evenness rewrite are required.",
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
    prior_parents: set[str] = set()
    for round_name in ("round01", "round02"):
        prior_report = CANDIDATE_DIR / f"numinamath_manual_candidates_{round_name}.report.json"
        if prior_report.exists():
            prior = json.loads(prior_report.read_text(encoding="utf-8"))
            prior_parents.update(prior.get("parent_record_ids", []))
    overlap = prior_parents & wanted
    if overlap:
        raise ValueError(f"prior-round parent overlap: {sorted(overlap)}")
    report = {
        "schema_version": "numinamath_manual_candidate_round_report_v1",
        "round": "round03",
        "candidate_count": len(rows),
        "parent_count": len(wanted),
        "quality_candidate_high": len(rows),
        "low_quality_candidates": 0,
        "lake_jobs_started": 0,
        "pantograph_jobs_started": 0,
        "pantograph_verified_rows": 0,
        "dataset_rows_written": 0,
        "canonical_parent_records_validated": len(parents),
        "prior_round_parent_overlap": 0,
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
