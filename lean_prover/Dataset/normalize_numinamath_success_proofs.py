"""Populate missing proof fields in already verified complete NuminaMath rows."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from lean_prover.Dataset.build_verified_datasets import (
    dataset_record_hash,
    sha256_file,
    sha256_text,
)
from lean_prover.Dataset.repair_numinamath_failures import (
    _write_json,
    read_jsonl,
    write_jsonl,
)
from lean_prover.Dataset.verify_external_datasets import NUMINA_SOURCE, split_imports
from lean_prover.lean_training.expert_iteration.utils import environment_identity
from lean_prover.lean_training.verification.pantograph import PantographTheoremVerifier


FORBIDDEN = re.compile(r"\b(?:sorry|admit|axiom)\b", re.I)
NORMALIZATION_VERSION = "numinamath_success_proof_field_v1"


def extract_final_proof(source: str) -> str:
    if ":=" not in source:
        raise ValueError("complete source has no final `:=` proof separator")
    proof = source.rsplit(":=", 1)[1].strip()
    if not proof:
        raise ValueError("final proof is empty")
    if FORBIDDEN.search(proof):
        raise ValueError("final proof contains a forbidden token")
    return proof


def main() -> None:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--success-file",
        type=Path,
        default=root / "verified_data" / "numinamath_verified_success.jsonl",
    )
    parser.add_argument(
        "--output-report",
        type=Path,
        default=Path("outputs/numinamath_repair/proof_field_normalization_report.json"),
    )
    parser.add_argument("--lean-project", type=Path, default=Path("lean_project"))
    parser.add_argument("--timeout", type=int, default=30)
    args = parser.parse_args()

    rows = read_jsonl(args.success_file)
    targets = [
        row
        for row in rows
        if not str(row.get("proof") or row.get("formal_proof") or "").strip()
    ]
    identity = environment_identity(args.lean_project, ("Mathlib",))
    verifier = PantographTheoremVerifier(
        args.lean_project,
        imports=("Mathlib",),
        timeout=args.timeout,
        startup_timeout=900,
    )
    results = []
    try:
        warmup = verifier.warmup(timeout=180)
        if not warmup.success:
            raise RuntimeError(f"Pantograph warmup failed: {warmup.diagnostics}")
        for row in targets:
            source = str(row.get("formal_ground_truth") or "").strip()
            proof = extract_final_proof(source)
            _, source_body = split_imports(source)
            checked = verifier.check_source(
                source_body,
                timeout=args.timeout,
                reject_forbidden=True,
            )
            results.append({
                "record_id": row["record_id"],
                "success": checked.success,
                "diagnostics": checked.diagnostics,
                "proof_sha256": sha256_text(proof),
                "assembled_source_hash": sha256_text(source),
            })
            if not checked.success:
                raise RuntimeError(
                    f"proof-field normalization failed for {row['record_id']}: "
                    f"{checked.diagnostics}"
                )
            row.update({
                "proof": proof,
                "formal_proof": proof,
                "proof_present": True,
                "proof_source": "formal_ground_truth_normalized",
                "proof_sha256": sha256_text(proof),
                "assembled_source_hash": sha256_text(source),
                "lean_version": identity["lean_version"],
                "mathlib_commit": identity["mathlib_commit"],
                "environment_hash": identity["environment_hash"],
                "proof_field_normalization": {
                    "version": NORMALIZATION_VERSION,
                    "pantograph_reverified": True,
                },
            })
            row["record_hash"] = dataset_record_hash(row, source=NUMINA_SOURCE)
    finally:
        verifier.close()

    write_jsonl(args.success_file, rows)
    report = {
        "schema_version": "numinamath_success_proof_normalization_report_v1",
        "normalization_version": NORMALIZATION_VERSION,
        "target_rows": len(targets),
        "normalized_rows": len(results),
        "all_pantograph_verified": all(item["success"] for item in results),
        "environment": identity,
        "results": results,
        "success_sha256": sha256_file(args.success_file),
    }
    _write_json(args.output_report, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
