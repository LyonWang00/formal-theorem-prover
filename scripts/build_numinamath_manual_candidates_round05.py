#!/usr/bin/env python3
"""Serialize and hash-lock the four human-authored NuminaMath round05 candidates.

The command only builds candidate artifacts.  It does not call Lake,
Pantograph, or an external API, and it never writes a verified dataset.
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
SOURCE = CANDIDATE_DIR / "numinamath_manual_candidates_round05.lean"
OUT = CANDIDATE_DIR / "numinamath_manual_candidates_round05.jsonl"
REPORT = CANDIDATE_DIR / "numinamath_manual_candidates_round05.report.json"
PARENT_SOURCE = (
    ROOT / "lean_prover/Dataset/verified_data/numinamath_verified_success.jsonl"
)
PRIOR_ROUNDS = ("round01", "round02", "round03", "round04")


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


METADATA: tuple[dict[str, Any], ...] = (
    {
        "candidate_id": "manual_round05_c01_recurrence_gcd_invariant",
        "marker": "C01",
        "theorem_name": "numinamath_manual_recurrence_gcd_invariant",
        "parent_record_id": "3e47a6ad-c134-5114-a9f2-b13326c6bd53::row_000091",
        "parent_theorem_name": "number_theory_4532",
        "parent_statement_sha256": "0bc849f13bfdcae768445d386d4697ce8d545d3d1a19c330732494f4abc34dd8",
        "parent_proof_sha256": "e9f5a59f8dc889fc60678d759eb18865a1f79db7905c816be2371c42dfc401cf",
        "manual_method": "generalize_fibonacci_coprimality_to_gcd_invariant_for_arbitrary_initial_values",
        "problem": (
            "Let a natural-number sequence satisfy u_(n+2)=u_(n+1)+u_n with "
            "arbitrary initial values. Prove that gcd(u_(n+1),u_n) is independent "
            "of n and equals gcd(u_1,u_0)."
        ),
        "quality_rationale": (
            "Removes both fixed Fibonacci initial conditions and the hard-coded gcd 1. "
            "The new theorem identifies the exact Euclidean invariant for every sequence "
            "with the same recurrence, so the parent is one corollary rather than a "
            "renumbered instance."
        ),
        "manual_risk": (
            "Low; the proof follows the parent's verified Euclidean rewrite but leaves "
            "the initial gcd symbolic."
        ),
    },
    {
        "candidate_id": "manual_round05_c02_abs_distance_sum_range",
        "marker": "C02",
        "theorem_name": "numinamath_manual_abs_distance_sum_range",
        "parent_record_id": "76b2f707-94ad-57e7-8217-13d50224eca6::row_008076",
        "parent_theorem_name": "algebra_21803",
        "parent_statement_sha256": "afef3a8286823699b0937de93aecdc8234ce3a6dfb606e0d3b4ca7fdb90f015a",
        "parent_proof_sha256": "d64abb133c5087343a4d72c8832e1767aee9a6f52d579b699bdcc539d45e289c",
        "manual_method": "generalize_two_fixed_foci_to_complete_parameterized_range_theorem",
        "problem": (
            "For real a<=b, determine the complete range of "
            "x |-> |x-a|+|x-b| and prove that it is exactly [b-a,infinity)."
        ),
        "quality_rationale": (
            "Replaces the parent's two numerical foci by arbitrary ordered endpoints and "
            "proves both the sharp lower bound and surjectivity onto every larger value. "
            "The explicit far-right witness makes this a complete range theorem."
        ),
        "manual_risk": (
            "Low; only exhaustive position cases for x and one explicit range witness are used."
        ),
    },
    {
        "candidate_id": "manual_round05_c03_prefix_mean_sequence_classification",
        "marker": "C03",
        "theorem_name": "numinamath_manual_prefix_mean_sequence_classification",
        "parent_record_id": "0971334a-3534-5199-8b94-2a8d587f02ca::row_058898",
        "parent_theorem_name": "algebra_97406",
        "parent_statement_sha256": "06fe18893c36ae2c76066813dd83d12098a02b1c70a1feea78c67c2d07217165",
        "parent_proof_sha256": "fcab51627cfdfcc0e7b4d29ac7d951f14e0bbcad70e593841e469dc8cc546211",
        "manual_method": "strengthen_one_requested_term_to_complete_sequence_classification",
        "problem": (
            "If the mean of the first n terms of a real sequence is n for every positive "
            "n, prove the closed formula a_n=2n+1 for every natural index n."
        ),
        "quality_rationale": (
            "Strictly strengthens the parent's isolated 2008th-term answer to a formula "
            "for the entire sequence. The proof derives every prefix sum and differences "
            "successive prefixes, including the n=0 boundary rather than specializing a number."
        ),
        "manual_risk": (
            "Low; the main proof obligations are natural-to-real coercions and the zero prefix."
        ),
    },
    {
        "candidate_id": "manual_round05_c04_concave_quadratic_maximum",
        "marker": "C04",
        "theorem_name": "numinamath_manual_concave_quadratic_maximum",
        "parent_record_id": "1f31e1b9-79b8-5076-9cb5-ceb79ebc4259::row_008710",
        "parent_theorem_name": "algebra_14014",
        "parent_statement_sha256": "45b713320d93d8b177ac7e3e6070119bd99e0ae91df3df2be41be264edb9db40",
        "parent_proof_sha256": "680901faa2e1631b093d2ccc4de517ef604c3412d89c6c7778e4e0f4b07521cd",
        "manual_method": "generalize_numeric_vertex_computation_to_all_strictly_concave_quadratics",
        "problem": (
            "For real coefficients with a<0, prove that ax^2+bx+c has greatest value "
            "c-b^2/(4a), attained at x=-b/(2a)."
        ),
        "quality_rationale": (
            "Turns one numerical completing-the-square exercise into the full sharp vertex "
            "theorem for every strictly concave real quadratic. It proves attainability and "
            "the global upper bound with an exact square certificate."
        ),
        "manual_risk": (
            "Medium; Pantograph must later check field_simp normalization and the sign lemma "
            "for a quotient of two nonpositive real numbers."
        ),
    },
)


def theorem_block(source: str, marker: str) -> str:
    start_token = f"-- CANDIDATE {marker} START\n"
    end_token = f"\n-- CANDIDATE {marker} END"
    if source.count(start_token) != 1 or source.count(end_token) != 1:
        raise ValueError(f"marker drift for {marker}")
    return source.split(start_token, 1)[1].split(end_token, 1)[0].strip()


def load_prior_manifests() -> tuple[set[str], list[dict[str, Any]]]:
    parent_ids: set[str] = set()
    locks: list[dict[str, Any]] = []
    for round_name in PRIOR_ROUNDS:
        path = CANDIDATE_DIR / f"numinamath_manual_candidates_{round_name}.jsonl"
        if not path.is_file():
            raise FileNotFoundError(
                f"prior-round manifest must be frozen before round05: {path}"
            )
        payload = path.read_bytes()
        rows = [json.loads(line) for line in payload.decode("utf-8").splitlines()]
        if not rows:
            raise ValueError(f"empty prior-round manifest: {path}")
        round_parents = [str(row.get("parent_record_id") or "") for row in rows]
        if any(not value for value in round_parents):
            raise ValueError(f"missing parent_record_id in prior manifest: {path}")
        if len(round_parents) != len(set(round_parents)):
            raise ValueError(f"duplicate parent within prior manifest: {path}")
        parent_ids.update(round_parents)
        locks.append(
            {
                "round": round_name,
                "path": str(path.relative_to(ROOT)),
                "sha256": digest_bytes(payload),
                "rows": len(rows),
            }
        )
    return parent_ids, locks


def build() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = SOURCE.read_text(encoding="utf-8")
    source_sha = digest(source)
    wanted = {meta["parent_record_id"] for meta in METADATA}
    if len(wanted) != len(METADATA):
        raise ValueError("round05 metadata repeats a parent")

    prior_parents, prior_locks = load_prior_manifests()
    overlap = sorted(prior_parents & wanted)
    if overlap:
        raise ValueError(f"prior-round parent overlap: {overlap}")

    parents: dict[str, dict[str, Any]] = {}
    with PARENT_SOURCE.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            record_id = str(row.get("record_id") or "")
            if record_id in wanted:
                if record_id in parents:
                    raise ValueError(f"duplicate canonical parent: {record_id}")
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
        if any(token in block.lower().split() for token in ("sorry", "admit", "axiom")):
            raise ValueError(f"forbidden placeholder in {meta['candidate_id']}")

        lean_statement = "import Mathlib\n\n" + statement_body.strip()
        proof = "by\n" + proof_body.rstrip() + "\n"
        formal_proof = "import Mathlib\n\n" + block.rstrip() + "\n"
        rows.append(
            {
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
        )

    if len(rows) != 4 or len({row["candidate_id"] for row in rows}) != 4:
        raise ValueError("candidate coverage/uniqueness failure")
    if len({row["formal_proof_sha256"] for row in rows}) != 4:
        raise ValueError("duplicate formal proof in round05")
    report = {
        "schema_version": "numinamath_manual_candidate_round_report_v1",
        "round": "round05",
        "candidate_count": len(rows),
        "parent_count": len(wanted),
        "quality_candidate_high": len(rows),
        "low_quality_candidates": 0,
        "lake_jobs_started": 0,
        "pantograph_jobs_started": 0,
        "pantograph_verified_rows": 0,
        "dataset_rows_written": 0,
        "canonical_parent_records_validated": len(parents),
        "prior_round_parent_overlap": len(overlap),
        "prior_round_manifests": prior_locks,
        "canonical_parent_source": str(PARENT_SOURCE.relative_to(ROOT)),
        "canonical_parent_source_sha256": digest_bytes(PARENT_SOURCE.read_bytes()),
        "source": str(SOURCE.relative_to(ROOT)),
        "source_sha256": source_sha,
        "candidate_ids": [row["candidate_id"] for row in rows],
        "parent_record_ids": [row["parent_record_id"] for row in rows],
        "statement_sha256": [row["statement_sha256"] for row in rows],
        "proof_sha256": [row["proof_sha256"] for row in rows],
        "formal_proof_sha256": [row["formal_proof_sha256"] for row in rows],
    }
    return rows, report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    if args.check and args.force:
        raise SystemExit("--check and --force are mutually exclusive")

    rows, report = build()
    manifest_text = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    report["manifest"] = str(OUT.relative_to(ROOT))
    report["manifest_sha256"] = digest(manifest_text)
    report_text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"

    if args.check:
        if not OUT.exists() or not REPORT.exists():
            raise FileNotFoundError("round05 manifest/report missing")
        if OUT.read_text(encoding="utf-8") != manifest_text:
            raise ValueError("round05 manifest is not a deterministic rebuild")
        if REPORT.read_text(encoding="utf-8") != report_text:
            raise ValueError("round05 report is not a deterministic rebuild")
    else:
        if not args.force and (OUT.exists() or REPORT.exists()):
            raise FileExistsError("refusing to overwrite round05 artifacts without --force")
        OUT.write_text(manifest_text, encoding="utf-8")
        REPORT.write_text(report_text, encoding="utf-8")

    print(
        json.dumps(
            {
                "mode": "check" if args.check else "write",
                "candidate_count": len(rows),
                "manifest": str(OUT.relative_to(ROOT)),
                "manifest_sha256": digest(manifest_text),
                "report": str(REPORT.relative_to(ROOT)),
                "report_sha256": digest(report_text),
                "source_sha256": report["source_sha256"],
                "prior_round_parent_overlap": report["prior_round_parent_overlap"],
                "lake_jobs_started": 0,
                "pantograph_jobs_started": 0,
            },
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
