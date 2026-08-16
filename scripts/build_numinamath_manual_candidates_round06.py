#!/usr/bin/env python3
"""Serialize and validate human-authored NuminaMath round06 candidates only."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
CANDIDATE_DIR = ROOT / "outputs/numinamath_expand_verification/manual_expansion_candidates"
SOURCE = CANDIDATE_DIR / "numinamath_manual_candidates_round06.lean"
OUT = CANDIDATE_DIR / "numinamath_manual_candidates_round06.jsonl"
REPORT = CANDIDATE_DIR / "numinamath_manual_candidates_round06.report.json"
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
        "candidate_id": "manual_round06_c01_cube_difference_invariants",
        "marker": "C01",
        "theorem_name": "numinamath_manual_cube_difference_from_two_invariants",
        "parent_record_id": "c72335a3-b841-56e5-af25-3155a35c54b7::row_001085",
        "parent_theorem_name": "algebra_20393",
        "parent_statement_sha256": "7f1fec2f6a620c078ac8507792c607a21d713a768fc0dec25846e09ced1448ab",
        "parent_proof_sha256": "ae2beeaaa1d788a08aa8f5ba5b84e60565f3dd744b1149a98a03a09a77115c16",
        "manual_method": "parameterize_cube_difference_by_difference_and_square_sum_invariants",
        "problem": (
            "For real x,y with x-y=d and x^2+y^2=s, determine x^3-y^3 exactly "
            "as d(3s-d^2)/2."
        ),
        "quality_rationale": (
            "Generalizes the parent's two fixed numerical invariants to arbitrary real parameters "
            "and derives the complete closed formula. The proof exposes both the difference-of-"
            "cubes factorization and the recovery of xy from the two invariants."
        ),
        "manual_risk": "Low; every step is an explicit polynomial identity or a direct rewrite.",
    },
    {
        "candidate_id": "manual_round06_c02_list_lcm_divisibility",
        "marker": "C02",
        "theorem_name": "numinamath_manual_list_lcm_divisibility_iff",
        "parent_record_id": "11766c6a-f015-5412-90ee-12696d757c4c::row_000098",
        "parent_theorem_name": "number_theory_4583",
        "parent_statement_sha256": "0a286f051afa8b464df9e884e67acae2097caddc48ed914c08bace47f3c4d410",
        "parent_proof_sha256": "06274660dc68a29d5814c2d52dd0efe0bc86cf10a237b98957c6cf4c64d06aa7",
        "manual_method": "generalize_binary_lcm_divisibility_to_arbitrary_finite_lists",
        "problem": (
            "For any finite list of natural numbers, prove that its iterated lcm divides d "
            "if and only if every list member divides d."
        ),
        "quality_rationale": (
            "Lifts the parent's binary criterion to an arbitrary finite family with the correct "
            "empty-list identity. This is a reusable induction theorem rather than one extra "
            "hard-coded factor."
        ),
        "manual_risk": "Low to medium; later elaboration should confirm the foldr simplification lemma set.",
    },
    {
        "candidate_id": "manual_round06_c03_quadratic_root_interval",
        "marker": "C03",
        "theorem_name": "numinamath_manual_quadratic_between_roots_solution_set",
        "parent_record_id": "45521fff-bad3-5eed-a946-174b357a3df3::row_006907",
        "parent_theorem_name": "algebra_18648",
        "parent_statement_sha256": "7c7f385c7d42966a2e42e94adc39eb42c1b2efad6242f519bca5807a0933626e",
        "parent_proof_sha256": "8f9888654be2673ea1cdc13025363c732f7dd69f9454e94f4700822378e414f7",
        "manual_method": "generalize_fixed_quadratic_solution_set_to_arbitrary_ordered_roots",
        "problem": (
            "For arbitrary real roots a<=b, prove that (x-a)(x-b)<=0 holds exactly for "
            "x in the closed interval [a,b]."
        ),
        "quality_rationale": (
            "Turns a fixed numerical quadratic exercise into the complete parameterized sign "
            "classification for a monic quadratic. Both exclusions outside the roots and the "
            "entire converse interval are proved explicitly."
        ),
        "manual_risk": "Low; the proof uses only order cases and signs of two factors.",
    },
    {
        "candidate_id": "manual_round06_c04_quartic_equality_classification",
        "marker": "C04",
        "theorem_name": "numinamath_manual_quartic_inequality_equality_iff",
        "parent_record_id": "9ff9bf39-07fb-585e-b3df-cac8bd67a4ae::row_011523",
        "parent_theorem_name": "inequalities_4918",
        "parent_statement_sha256": "325443478bf236610c8396e0c5b82ccc91b15d842d2cea36c4ee8b890b257ac1",
        "parent_proof_sha256": "7fe71ea4a7a891427033710cd3de85d1e37a9d77513416e447911483fd5ed3a5",
        "manual_method": "classify_all_equality_cases_of_sharp_quartic_inequality",
        "problem": (
            "Classify equality in 8(a^4+b^4)>=(a+b)^4 over the reals: prove equality "
            "holds if and only if a=b."
        ),
        "quality_rationale": (
            "Adds the exact equality locus omitted by the parent. The proof factors the gap into "
            "(a-b)^2 times a positive-definite quadratic and separately resolves both possible "
            "zero factors."
        ),
        "manual_risk": "Low; the equality classification is backed by an explicit exact factorization.",
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
        delimiter = " := by\n"
        if delimiter not in block:
            raise ValueError(f"cannot split statement/proof: {meta['candidate_id']}")
        statement_body, proof_body = block.split(delimiter, 1)
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
        raise ValueError("round06 parent uniqueness failure")

    prior_parents: set[str] = set()
    prior_normalized: set[str] = set()
    prior_locks: list[dict[str, Any]] = []
    for index in range(1, 6):
        round_name = f"round{index:02d}"
        manifest = CANDIDATE_DIR / f"numinamath_manual_candidates_{round_name}.jsonl"
        report_path = CANDIDATE_DIR / f"numinamath_manual_candidates_{round_name}.report.json"
        payload = manifest.read_bytes()
        report = json.loads(report_path.read_text(encoding="utf-8"))
        actual_sha = digest_bytes(payload)
        if actual_sha != report.get("manifest_sha256"):
            raise ValueError(f"prior manifest/report SHA drift: {round_name}")
        prior_locks.append({
            "round": round_name,
            "path": str(manifest.relative_to(ROOT)),
            "rows": len(payload.splitlines()),
            "sha256": actual_sha,
        })
        prior_parents.update(report.get("parent_record_ids", []))
        for line in payload.splitlines():
            prior_normalized.add(normalized_statement(json.loads(line)["lean_statement"]))
    overlap = prior_parents & wanted
    if overlap:
        raise ValueError(f"prior-round parent overlap: {sorted(overlap)}")
    normalized = [normalized_statement(row["lean_statement"]) for row in rows]
    if len(set(normalized)) != len(normalized):
        raise ValueError("within-round normalized statement duplicate")
    if set(normalized) & prior_normalized:
        raise ValueError("prior-round normalized statement duplicate")

    report = {
        "schema_version": "numinamath_manual_candidate_round_report_v1",
        "round": "round06",
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
        "canonical_parent_source_sha256": digest_bytes(PARENT_SOURCE.read_bytes()),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_sha,
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
