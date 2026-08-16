"""Strictly audit an externally verified Numina repair batch before ingestion.

The audit is read-only.  It checks file hashes, environment identity, candidate
and result linkage, replay completeness, success semantics, forbidden proof
tokens, assembled source hashes, and current fail-set ownership.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

from lean_prover.Dataset.build_verified_datasets import sha256_file
from lean_prover.Dataset.repair_numinamath_failures import (
    REPAIR_VERSION,
    canonical_hash,
    complete_source,
    with_imports,
)
from lean_prover.lean_training.expert_iteration.utils import environment_identity


FORBIDDEN = re.compile(r"\b(?:sorry|admit|axiom)\b", re.I)


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def audit_batch(
    batch_dir: Path,
    fail_rows: Mapping[str, Mapping[str, Any]],
    expected_environment: Mapping[str, Any],
) -> dict[str, Any]:
    candidate_path = batch_dir / "candidate_manifest.jsonl"
    result_path = batch_dir / "verification_results.jsonl"
    report_path = batch_dir / "candidate_verification_report.json"
    for path in (candidate_path, result_path, report_path):
        check(path.exists(), f"missing required cloud artifact: {path}")
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    batch_id = str(report.get("batch_id") or batch_dir.name)
    check(report.get("schema_version") == "numinamath_candidate_verification_report_v1", "unexpected report schema")
    check(report.get("repair_version") == REPAIR_VERSION, "repair version mismatch")
    check(report.get("master_dataset_mutated") is False, "cloud run reports master dataset mutation")
    check(report.get("candidate_manifest_sha256") == sha256_file(candidate_path), "candidate manifest hash mismatch")
    check(report.get("verification_results_sha256") == sha256_file(result_path), "verification result hash mismatch")
    check(report.get("environment") == dict(expected_environment), "report environment mismatch")

    candidates = list(iter_jsonl(candidate_path))
    candidate_ids = [str(row.get("record_id") or "") for row in candidates]
    check(all(candidate_ids), "candidate missing record_id")
    check(len(candidate_ids) == len(set(candidate_ids)), "duplicate candidate record_id")
    check(len(candidates) == int(report.get("input_candidates", -1)), "candidate count mismatch")
    missing_fail = sorted(set(candidate_ids) - set(fail_rows))
    check(not missing_fail, f"candidate records no longer belong to current fail set: {missing_fail[:5]}")

    expected_variants: dict[str, dict[str, Any]] = {}
    candidate_variant_order: dict[str, list[str]] = {}
    for candidate in candidates:
        record_id = str(candidate["record_id"])
        check(candidate.get("batch_id") == batch_id, f"candidate batch mismatch: {record_id}")
        check(candidate.get("repair_version") == REPAIR_VERSION, f"candidate repair version mismatch: {record_id}")
        source_candidate = dict(candidate)
        claimed_candidate_hash = str(source_candidate.pop("candidate_hash", ""))
        check(claimed_candidate_hash == canonical_hash(source_candidate), f"candidate hash mismatch: {record_id}")
        check(
            str(candidate.get("original_record_hash") or "") == str(fail_rows[record_id].get("record_hash") or ""),
            f"current fail record hash mismatch: {record_id}",
        )
        variants = candidate.get("variants") or []
        check(bool(variants), f"candidate has no variants: {record_id}")
        candidate_variant_order[record_id] = []
        for variant in variants:
            proof = str(variant.get("proof") or "")
            strategy = str(variant.get("strategy") or "")
            check(proof.startswith("by") and not FORBIDDEN.search(proof), f"invalid or forbidden proof: {record_id}")
            variant_hash = canonical_hash({
                "candidate_hash": claimed_candidate_hash,
                "strategy": strategy,
                "proof": proof,
                "environment_hash": expected_environment["environment_hash"],
            })
            check(variant_hash not in expected_variants, f"duplicate expected variant hash: {variant_hash}")
            expected_variants[variant_hash] = {
                "record_id": record_id,
                "candidate_hash": claimed_candidate_hash,
                "strategy": strategy,
                "proof": proof,
                "assembled_source_hash": sha256_text(
                    with_imports((), complete_source(str(candidate["source_body"]), proof))
                ),
            }
            candidate_variant_order[record_id].append(variant_hash)

    results = list(iter_jsonl(result_path))
    result_by_hash: dict[str, dict[str, Any]] = {}
    duplicate_results = 0
    for result in results:
        variant_hash = str(result.get("variant_hash") or "")
        check(variant_hash in expected_variants, f"unexpected result variant hash: {variant_hash}")
        if variant_hash in result_by_hash:
            duplicate_results += 1
            check(result_by_hash[variant_hash] == result, f"conflicting duplicate result: {variant_hash}")
            continue
        expected = expected_variants[variant_hash]
        check(result.get("schema_version") == "numinamath_repair_verification_result_v1", "unexpected result schema")
        check(result.get("repair_version") == REPAIR_VERSION, "result repair version mismatch")
        check(result.get("batch_id") == batch_id, "result batch mismatch")
        for field in ("record_id", "candidate_hash", "strategy", "proof", "assembled_source_hash"):
            check(str(result.get(field)) == str(expected[field]), f"result {field} mismatch: {variant_hash}")
        check(result.get("environment") == dict(expected_environment), f"result environment mismatch: {variant_hash}")
        success = bool(result.get("success"))
        check(result.get("pantograph_verified") == ("success" if success else "fail"), "success/status mismatch")
        if success:
            check(not result.get("timed_out"), "successful result marked timed out")
            check(not result.get("errors"), "successful result contains errors")
            check(not FORBIDDEN.search(str(result.get("proof") or "")), "successful result has forbidden proof token")
        result_by_hash[variant_hash] = result

    chosen: dict[str, dict[str, Any]] = {}
    missing_required: list[str] = []
    for record_id, variant_hashes in candidate_variant_order.items():
        last: dict[str, Any] | None = None
        for variant_hash in variant_hashes:
            result = result_by_hash.get(variant_hash)
            if result is None:
                missing_required.append(variant_hash)
                break
            last = result
            if result.get("success"):
                break
        if last is not None:
            chosen[record_id] = last
    check(not missing_required, f"missing required sequential results: {missing_required[:5]}")
    check(len(chosen) == len(candidates), "not every candidate has a selected result")
    solved = sum(bool(row.get("success")) for row in chosen.values())
    check(solved == int(report.get("records_solved", -1)), "solved count mismatch")
    check(len(chosen) - solved == int(report.get("records_failed", -1)), "failed count mismatch")
    check(len(chosen) == int(report.get("records_completed", -1)), "completed count mismatch")
    return {
        "batch_id": batch_id,
        "audit_status": "pass",
        "candidate_rows": len(candidates),
        "unique_result_variants": len(result_by_hash),
        "duplicate_identical_result_rows": duplicate_results,
        "records_solved": solved,
        "records_failed": len(chosen) - solved,
        "candidate_manifest_sha256": sha256_file(candidate_path),
        "verification_results_sha256": sha256_file(result_path),
        "environment_hash": expected_environment["environment_hash"],
        "current_fail_ownership": "pass",
        "forbidden_success_proof_count": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-dir", type=Path, action="append", required=True)
    parser.add_argument("--fail-file", type=Path, required=True)
    parser.add_argument("--lean-project", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, required=True)
    args = parser.parse_args()
    fail_rows = {str(row["record_id"]): row for row in iter_jsonl(args.fail_file)}
    identity = environment_identity(args.lean_project, ("Mathlib",))
    audits = [audit_batch(path, fail_rows, identity) for path in args.batch_dir]
    all_ids: list[str] = []
    for path in args.batch_dir:
        all_ids.extend(str(row["record_id"]) for row in iter_jsonl(path / "candidate_manifest.jsonl"))
    overlap_count = len(all_ids) - len(set(all_ids))
    check(overlap_count == 0, "record overlap between audited cloud batches")
    report = {
        "schema_version": "numinamath_cloud_ingest_audit_v1",
        "audit_status": "pass",
        "environment": identity,
        "batches": audits,
        "cross_batch_record_overlap": overlap_count,
        "total_candidates": sum(row["candidate_rows"] for row in audits),
        "total_solved": sum(row["records_solved"] for row in audits),
        "total_failed": sum(row["records_failed"] for row in audits),
    }
    args.report_file.parent.mkdir(parents=True, exist_ok=True)
    args.report_file.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
