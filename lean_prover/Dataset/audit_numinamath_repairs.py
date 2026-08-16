"""Audit the frozen NuminaMath success/fail split after repair batches."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from lean_prover.Dataset.build_verified_datasets import dataset_record_hash, sha256_file
from lean_prover.Dataset.repair_numinamath_failures import iter_jsonl
from lean_prover.Dataset.verify_external_datasets import NUMINA_SOURCE


FORBIDDEN = re.compile(r"\b(?:sorry|admit|axiom)\b", re.I)


def audit(success_file: Path, fail_file: Path, *, original_success: int) -> dict[str, Any]:
    success_ids: set[str] = set()
    fail_ids: set[str] = set()
    duplicate_success = 0
    duplicate_fail = 0
    success_contract_errors = Counter()
    fail_contract_errors = Counter()
    repair_batches = Counter()
    repair_strategies = Counter()
    repaired_rows = 0
    empty_success_proof_ids: list[str] = []

    for row in iter_jsonl(success_file):
        record_id = str(row.get("record_id") or "")
        if record_id in success_ids:
            duplicate_success += 1
        success_ids.add(record_id)
        proof = str(row.get("proof") or row.get("formal_proof") or "").strip()
        if row.get("pantograph_verified") != "success":
            success_contract_errors["pantograph_verified"] += 1
        if row.get("source") != NUMINA_SOURCE:
            success_contract_errors["source"] += 1
        if "sourcce" in row:
            success_contract_errors["legacy_sourcce"] += 1
        if not proof:
            success_contract_errors["empty_proof"] += 1
            empty_success_proof_ids.append(record_id)
        if FORBIDDEN.search(proof):
            success_contract_errors["forbidden_proof_token"] += 1
        if row.get("record_hash") != dataset_record_hash(row, source=NUMINA_SOURCE):
            success_contract_errors["record_hash"] += 1
        repair = row.get("repair")
        if isinstance(repair, dict):
            repaired_rows += 1
            repair_batches[str(repair.get("batch_id") or "missing")] += 1
            repair_strategies[str(repair.get("selected_strategy") or "missing")] += 1

    for row in iter_jsonl(fail_file):
        record_id = str(row.get("record_id") or "")
        if record_id in fail_ids:
            duplicate_fail += 1
        fail_ids.add(record_id)
        if row.get("pantograph_verified") != "fail":
            fail_contract_errors["pantograph_verified"] += 1
        if row.get("source") != NUMINA_SOURCE:
            fail_contract_errors["source"] += 1
        if "sourcce" in row:
            fail_contract_errors["legacy_sourcce"] += 1
        if not str(row.get("error_message") or "").strip():
            fail_contract_errors["empty_error_message"] += 1
        if row.get("record_hash") != dataset_record_hash(row, source=NUMINA_SOURCE):
            fail_contract_errors["record_hash"] += 1

    overlap = success_ids & fail_ids
    success_count = len(success_ids)
    fail_count = len(fail_ids)
    return {
        "schema_version": "numinamath_repair_audit_v1",
        "original_success_baseline": original_success,
        "success_rows": success_count,
        "fail_rows": fail_count,
        "total_rows": success_count + fail_count,
        "net_promoted_since_baseline": success_count - original_success,
        "rows_with_repair_metadata": repaired_rows,
        "duplicate_success_ids": duplicate_success,
        "duplicate_fail_ids": duplicate_fail,
        "success_fail_id_overlap": len(overlap),
        "success_contract_errors": dict(success_contract_errors),
        "empty_success_proof_ids": empty_success_proof_ids,
        "fail_contract_errors": dict(fail_contract_errors),
        "repair_batch_counts": dict(sorted(repair_batches.items())),
        "repair_strategy_counts": dict(repair_strategies.most_common()),
        "success_sha256": sha256_file(success_file),
        "fail_sha256": sha256_file(fail_file),
        "passed": not (
            duplicate_success
            or duplicate_fail
            or overlap
            or success_contract_errors
            or fail_contract_errors
            or success_count + fail_count != 104155
            or success_count - original_success < 1000
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    root = Path(__file__).resolve().parent
    parser.add_argument(
        "--success-file",
        type=Path,
        default=root / "verified_data" / "numinamath_verified_success.jsonl",
    )
    parser.add_argument(
        "--fail-file",
        type=Path,
        default=root / "verified_data" / "numinamath_verified_fail.jsonl",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/numinamath_repair/final_audit.json"),
    )
    parser.add_argument("--original-success", type=int, default=24202)
    args = parser.parse_args()
    payload = audit(
        args.success_file,
        args.fail_file,
        original_success=args.original_success,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
