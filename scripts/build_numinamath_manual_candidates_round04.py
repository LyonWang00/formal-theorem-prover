#!/usr/bin/env python3
"""Serialize and validate human-authored NuminaMath round04 candidates only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "outputs/numinamath_expand_verification/manual_expansion_candidates"
SOURCE = CANDIDATE_DIR / "numinamath_manual_candidates_round04.lean"
OUT = CANDIDATE_DIR / "numinamath_manual_candidates_round04.jsonl"
REPORT = CANDIDATE_DIR / "numinamath_manual_candidates_round04.report.json"
PARENT_SOURCE = ROOT / "lean_prover/Dataset/verified_data/numinamath_verified_success.jsonl"


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def normalized_statement(text: str) -> str:
    return re.sub(r"\s+", "", text.replace("importMathlib", "")).lower()


METADATA: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "manual_round04_c01_even_quadratic_abs_classification",
        "marker": "C01",
        "theorem_name": "numinamath_manual_even_quadratic_abs_classification",
        "parent_record_id": "a5124340-730a-54b9-aec0-f8cdad6c0c92::row_000384",
        "parent_theorem_name": "algebra_9349",
        "parent_statement_sha256": "e327f6ed155bd9bec8d47a2b762f3642e14ed3f77e4c4649eb41c9d967ecde19",
        "parent_proof_sha256": "47d7e4018ba28a82b3cc3aec1b724da9b9eedc5ece64b5e00353a0e6f5722061",
        "manual_method": "strengthen_single_value_to_full_nonzero_domain_abs_classification",
        "problem": (
            "Classify the nonzero values of an even real function whose negative branch is "
            "f(x)=x^2+x: prove f(x)=x^2-|x| for every x != 0."
        ),
        "quality_rationale": (
            "Strictly strengthens one evaluated value to the complete formula on both open "
            "half-lines. The exclusion of zero is mathematically necessary because the parent "
            "hypotheses do not determine f(0)."
        ),
        "manual_risk": "Low; the proof splits by sign and applies evenness exactly once.",
    },
    {
        "candidate_id": "manual_round04_c02_cubic_three_roots_coefficients",
        "marker": "C02",
        "theorem_name": "numinamath_manual_cubic_three_roots_coefficients",
        "parent_record_id": "9e1c4689-2c2a-5f69-ae8f-d5328c668823::row_000588",
        "parent_theorem_name": "algebra_12970",
        "parent_statement_sha256": "5f05255fa068d35ba5da54c9d8175dd74be7a13df4f3227ef75e91541b4e97f2",
        "parent_proof_sha256": "c5b0ce4c4aabb81b1056589d39a1e1241d8f038844ae83defb444441a406c670",
        "manual_method": "strengthen_one_coefficient_ratio_to_complete_coefficient_classification",
        "problem": (
            "If a real cubic a*x^3+b*x^2+c*x+d vanishes at 1, 2, and 3, determine "
            "all lower coefficients: b=-6a, c=11a, and d=-6a."
        ),
        "quality_rationale": (
            "Replaces the parent's single ratio with the exact three-coefficient classification "
            "and removes the unnecessary nonzero-leading-coefficient assumption. This is the "
            "full coefficient content of the three root equations."
        ),
        "manual_risk": "Low; normalization produces three linear equations in the coefficients.",
    },
    {
        "candidate_id": "manual_round04_c03_recurrence_exact_quotient",
        "marker": "C03",
        "theorem_name": "numinamath_manual_recurrence_exact_divisibility_quotient",
        "parent_record_id": "0a7fe00b-28bf-5b83-ad0d-33b31dbc023e::row_002770",
        "parent_theorem_name": "number_theory_111051",
        "parent_statement_sha256": "8b68afb9274c978bc8a69389656ae26a794374e680de6fe28e7dd335ae8dc568",
        "parent_proof_sha256": "4bcc57a5748eca573d44156ff711c44fc14ae461d23cc0affb8b5fae74bc45f2",
        "manual_method": "strengthen_divisibility_to_exact_factor_and_quotient_identity",
        "problem": (
            "For the recurrence a_(n+1)=a_n^2+a_n+1, prove the exact factorization "
            "a_(n+1)^2+1=(a_n^2+1)(a_n^2+2a_n+2) for every n >= 1."
        ),
        "quality_rationale": (
            "Identifies the exact quotient hidden by the parent's divisibility conclusion. "
            "The result is a strictly stronger recurrence invariant and no longer uses the "
            "irrelevant initial-value hypothesis."
        ),
        "manual_risk": "Low; recurrence substitution reduces the theorem to a polynomial identity.",
    },
    {
        "candidate_id": "manual_round04_c04_coprime_unimodular_progression",
        "marker": "C04",
        "theorem_name": "numinamath_manual_coprime_unimodular_progression",
        "parent_record_id": "7171d14f-5372-51ca-8b42-27ac5ffdb523::row_026678",
        "parent_theorem_name": "number_theory_256072",
        "parent_statement_sha256": "8fa6c390bf34c1203385dcaae0c42225aae13b8d215bffe1c930a53440fa37f1",
        "parent_proof_sha256": "ebac009865d08146541265e1bd512cf0a050050e1db187d94c4efad0a960bc3b",
        "manual_method": "generalize_specific_pairwise_coprime_triple_by_unimodular_basis_change",
        "problem": (
            "Given coprime integers a and b, prove that a, a+b, and 2a+b are pairwise "
            "coprime: (a,a+b), (a+b,2a+b), and (2a+b,a)."
        ),
        "quality_rationale": (
            "Generalizes the parent's b=1 numerical pattern to every coprime integer basis. "
            "The proof transports an arbitrary Bezout certificate through three determinant-one "
            "linear changes, exposing the structural reason for coprimality."
        ),
        "manual_risk": "Low to medium; the three explicit Bezout coefficient identities need later elaboration checking.",
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
    if len({row["parent_record_id"] for row in rows}) != 4:
        raise ValueError("round04 parent uniqueness failure")

    prior_parents: set[str] = set()
    prior_normalized: set[str] = set()
    for round_name in ("round01", "round02", "round03"):
        prior_report = CANDIDATE_DIR / f"numinamath_manual_candidates_{round_name}.report.json"
        prior_manifest = CANDIDATE_DIR / f"numinamath_manual_candidates_{round_name}.jsonl"
        if not prior_report.exists() or not prior_manifest.exists():
            raise FileNotFoundError(f"prior round artifacts missing: {round_name}")
        prior = json.loads(prior_report.read_text(encoding="utf-8"))
        prior_parents.update(prior.get("parent_record_ids", []))
        with prior_manifest.open("r", encoding="utf-8") as handle:
            for line in handle:
                prior_normalized.add(normalized_statement(json.loads(line)["lean_statement"]))
    overlap = prior_parents & wanted
    if overlap:
        raise ValueError(f"prior-round parent overlap: {sorted(overlap)}")
    normalized = [normalized_statement(row["lean_statement"]) for row in rows]
    if len(set(normalized)) != len(normalized):
        raise ValueError("within-round normalized statement duplicate")
    prior_statement_overlap = set(normalized) & prior_normalized
    if prior_statement_overlap:
        raise ValueError("prior-round normalized statement duplicate")

    report = {
        "schema_version": "numinamath_manual_candidate_round_report_v1",
        "round": "round04",
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
        "prior_round_normalized_statement_overlap": 0,
        "within_round_normalized_statement_duplicates": 0,
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
