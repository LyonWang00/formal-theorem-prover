#!/usr/bin/env python3
"""Compile the locked 28-row manual v4 batch in one Pantograph worker session.

This command writes only per-candidate Pantograph receipts and compilation
reports inside the isolated v4 batch directory.  It makes no quality decision
and never writes raw or verified datasets.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from lean_prover.Dataset.numinamath_pantograph_worker import (
    _safe_close,
    start_verifier,
    verify_job,
)
from lean_prover.lean_training.expert_iteration.utils import environment_identity


ROOT = Path(__file__).resolve().parents[1]
BATCH_ID = "numinamath_manual_rounds01_07_v4_b00000"
BATCH_DIR = (
    ROOT
    / "outputs/numinamath_expand_verification/manual_expansion_candidates"
    / "pantograph_prepared_batches"
    / BATCH_ID
)
MANIFEST = BATCH_DIR / "candidate_manifest.jsonl"
RESULTS = BATCH_DIR / "verification_results.jsonl"
WORKER_REPORT = BATCH_DIR / "candidate_verification_report.json"
COMPILE_REPORT = BATCH_DIR / "pantograph_compile_only_report.json"
STATE_DIR = BATCH_DIR / "one_shot_state"
LEAN_PROJECT = ROOT / "lean_project"
MATHLIB_OLEAN = (
    LEAN_PROJECT
    / ".lake/packages/mathlib/.lake/build/lib/lean/Mathlib.olean"
)

EXPECTED_MANIFEST_SHA256 = (
    "141be823df692c62e786724cebff1246b91670b11d69c146782ff284fbfb442a"
)
EXPECTED_CANDIDATES = 28
EXPECTED_PREPARER_VERSION = "numinamath_manual_rounds01_07_pantograph_prepare_v4"


class OneShotError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def rows(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                raise OneShotError(f"blank line at {path}:{line_number}")
            value = json.loads(line)
            if not isinstance(value, dict):
                raise OneShotError(f"non-object row at {path}:{line_number}")
            yield value


def validate_inputs() -> list[dict[str, Any]]:
    if not MATHLIB_OLEAN.is_file() or MATHLIB_OLEAN.stat().st_size <= 0:
        raise OneShotError(f"missing/non-empty Mathlib.olean: {MATHLIB_OLEAN}")
    actual_manifest_sha = sha256_file(MANIFEST)
    if actual_manifest_sha != EXPECTED_MANIFEST_SHA256:
        raise OneShotError(
            f"v4 manifest SHA drift: expected {EXPECTED_MANIFEST_SHA256}, "
            f"got {actual_manifest_sha}"
        )
    candidates = list(rows(MANIFEST))
    if len(candidates) != EXPECTED_CANDIDATES:
        raise OneShotError(
            f"expected {EXPECTED_CANDIDATES} candidates, got {len(candidates)}"
        )
    for candidate in candidates:
        if candidate.get("batch_id") != BATCH_ID:
            raise OneShotError("candidate batch_id is not locked v4")
        if candidate.get("preparer_version") != EXPECTED_PREPARER_VERSION:
            raise OneShotError("candidate preparer_version is not locked v4")
        if candidate.get("contains_forbidden_token") is not False:
            raise OneShotError("candidate has forbidden-token flag")
        if len(candidate.get("variants") or []) != 1:
            raise OneShotError("one-shot requires exactly one proof per candidate")
    for key, values in {
        "record_id": [str(row["record_id"]) for row in candidates],
        "candidate_hash": [str(row["candidate_hash"]) for row in candidates],
        "parent_record_id": [
            str(row["manual_provenance"]["parent_record_id"])
            for row in candidates
        ],
        "statement_sha256": [
            str(row["manual_provenance"]["statement_sha256"])
            for row in candidates
        ],
        "proof_sha256": [
            str(row["manual_provenance"]["proof_sha256"])
            for row in candidates
        ],
    }.items():
        if len(set(values)) != EXPECTED_CANDIDATES:
            raise OneShotError(f"v4 input is not globally unique by {key}")
    return candidates


def build_compile_report(
    candidates: list[Mapping[str, Any]],
    *,
    environment: Mapping[str, Any],
) -> dict[str, Any]:
    receipts = list(rows(RESULTS))
    candidate_by_id = {str(row["record_id"]): row for row in candidates}
    receipt_by_id: dict[str, dict[str, Any]] = {}
    for receipt in receipts:
        record_id = str(receipt.get("record_id") or "")
        if record_id not in candidate_by_id:
            raise OneShotError(f"receipt for unknown v4 candidate: {record_id}")
        if record_id in receipt_by_id:
            raise OneShotError(f"multiple receipts for one-shot candidate: {record_id}")
        candidate = candidate_by_id[record_id]
        if receipt.get("batch_id") != BATCH_ID:
            raise OneShotError(f"receipt has wrong batch_id: {record_id}")
        if receipt.get("candidate_hash") != candidate.get("candidate_hash"):
            raise OneShotError(f"receipt candidate_hash mismatch: {record_id}")
        receipt_by_id[record_id] = receipt
    missing = sorted(set(candidate_by_id) - set(receipt_by_id))
    if missing:
        raise OneShotError(f"missing one-shot receipts: {missing}")

    by_round: dict[str, Counter[str]] = defaultdict(Counter)
    error_types: Counter[str] = Counter()
    per_record: list[dict[str, Any]] = []
    for record_id in sorted(candidate_by_id):
        candidate = candidate_by_id[record_id]
        receipt = receipt_by_id[record_id]
        success = bool(receipt.get("success"))
        status = "success" if success else "fail"
        round_name = str(candidate["manual_provenance"]["round"])
        by_round[round_name][status] += 1
        error_type = str(receipt.get("error_type") or "")
        if not success:
            error_types[error_type or "unspecified"] += 1
        per_record.append({
            "record_id": record_id,
            "round": round_name,
            "candidate_hash": receipt["candidate_hash"],
            "variant_hash": receipt["variant_hash"],
            "status": status,
            "pantograph_verified": receipt.get("pantograph_verified"),
            "timed_out": bool(receipt.get("timed_out")),
            "error_type": error_type,
            "verification_seconds": receipt.get("verification_seconds"),
            "pantograph_restart_count": receipt.get("pantograph_restart_count"),
            "assembled_source_hash": receipt.get("assembled_source_hash"),
            "diagnostics": receipt.get("diagnostics"),
            "errors": receipt.get("errors") or [],
            "warnings": receipt.get("warnings") or [],
        })
    counts = Counter(row["status"] for row in per_record)
    return {
        "schema_version": "numinamath_manual_v4_pantograph_compile_only_report_v1",
        "batch_id": BATCH_ID,
        "candidate_manifest": str(MANIFEST.relative_to(ROOT)),
        "candidate_manifest_sha256": sha256_file(MANIFEST),
        "candidate_count": len(candidates),
        "verification_results": str(RESULTS.relative_to(ROOT)),
        "verification_results_sha256": sha256_file(RESULTS),
        "receipt_count": len(receipts),
        "compile_status_counts": dict(sorted(counts.items())),
        "compile_status_by_round": {
            key: dict(sorted(value.items())) for key, value in sorted(by_round.items())
        },
        "failure_error_types": dict(sorted(error_types.items())),
        "environment": dict(environment),
        "mathlib_olean": str(MATHLIB_OLEAN.relative_to(ROOT)),
        "mathlib_olean_size": MATHLIB_OLEAN.stat().st_size,
        "mathlib_olean_sha256": sha256_file(MATHLIB_OLEAN),
        "per_record_receipts": per_record,
        "quality_decisions_made": 0,
        "independent_manual_review_required": True,
        "pantograph_one_shot": True,
        "raw_dataset_mutated": False,
        "verified_dataset_mutated": False,
        "external_api_called": False,
    }


def write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(dict(value), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="validate v4/Mathlib locks without starting Pantograph",
    )
    args = parser.parse_args()
    candidates = validate_inputs()
    if args.preflight_only:
        print(json.dumps({
            "mode": "preflight-only",
            "batch_id": BATCH_ID,
            "candidate_count": len(candidates),
            "candidate_manifest_sha256": sha256_file(MANIFEST),
            "mathlib_olean_size": MATHLIB_OLEAN.stat().st_size,
            "mathlib_olean_sha256": sha256_file(MATHLIB_OLEAN),
            "pantograph_started": False,
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    verifier = None
    environment: Mapping[str, Any]
    worker_restarts = 0
    try:
        verifier_args = argparse.Namespace(
            lean_project=LEAN_PROJECT,
            timeout=args.timeout,
        )
        verifier = start_verifier(verifier_args)
        environment = environment_identity(LEAN_PROJECT, ("Mathlib",))
        job = {
            "batch_id": BATCH_ID,
            "batch_dir": str(BATCH_DIR),
            "lean_project": str(LEAN_PROJECT),
            "timeout": args.timeout,
            "candidate_manifest_sha256": EXPECTED_MANIFEST_SHA256,
        }
        verifier, worker_restarts = verify_job(
            job=job,
            verifier=verifier,
            identity=environment,
            queue_dir=STATE_DIR,
        )
    finally:
        if verifier is not None:
            _safe_close(verifier)

    report = build_compile_report(candidates, environment=environment)
    report["pantograph_restarts"] = worker_restarts
    write_json_atomic(COMPILE_REPORT, report)
    print(json.dumps({
        "batch_id": BATCH_ID,
        "candidate_count": len(candidates),
        "compile_status_counts": report["compile_status_counts"],
        "failure_error_types": report["failure_error_types"],
        "verification_results_sha256": report["verification_results_sha256"],
        "compile_report": str(COMPILE_REPORT.relative_to(ROOT)),
        "compile_report_sha256": sha256_file(COMPILE_REPORT),
        "pantograph_restarts": worker_restarts,
        "quality_decisions_made": 0,
    }, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
