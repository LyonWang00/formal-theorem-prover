#!/usr/bin/env python3
"""Read-only readiness audit for final NuminaMath expansion materialization.

The audit exercises the production finalizer's parsers, content locks, receipt
selection, parent restoration, and output field contracts entirely in memory.
It does not create staged final outputs and never invokes Lean or Pantograph.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from scripts.finalize_numinamath_expand_verification import (
    FAIL,
    SUCCESS,
    apply_review,
    compatible_receipts,
    decision_of,
    iter_jsonl,
    load_candidate_manifests,
    load_parent_source,
    load_receipts,
    load_reviews,
    materialize_row,
    normalize_newlines,
    resolve_jsonl_inputs,
    select_compatible_receipt,
    sha256_file,
    sha256_text,
    synchronize_formal_content,
    validate_review_hashes,
)


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/numinamath_expand_verification"
RAW = ROOT / "lean_prover/Dataset/raw_data/numinamath_expand.jsonl"
RAW_SHA256 = "dd66edc02676dc3d459eef84304e02bc4354c3c63a5d4bd8791c78b1ee79d60e"
PARENT = ROOT / "lean_prover/Dataset/verified_data/numinamath_verified_success.jsonl"
ENVIRONMENT_HASH = "46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a"
CURRENT_REVIEWS = (
    OUT / "manual_review_clean/reviewed_045723_v40d.jsonl",
    OUT / "manual_review_shards/pre_review_s00_b00045_adjudicated.jsonl",
    OUT / "manual_review_shards/pre_review_s00_b00046_adjudicated.jsonl",
    OUT / "manual_review_shards/pre_review_s00_b00047.jsonl",
)
RECEIPT_ROOTS = (OUT / "batches", OUT / "repairs")
BATCH_RE = re.compile(r"numinamath_expand_s(?P<shard>\d\d)_b(?P<batch>\d{5})$")


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(ROOT)),
        "rows": sum(1 for _ in iter_jsonl(path)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report",
        type=Path,
        default=OUT / "finalization_readiness_v40d.report.json",
    )
    parser.add_argument(
        "--quality-review",
        type=Path,
        action="append",
        default=[],
        help=(
            "Quality-review JSONL input; repeatable. Defaults to the current "
            "v40d plus three s00 tail pre-review files. For the final gate, "
            "pass exactly one hash-locked 46729-row checkpoint."
        ),
    )
    parser.add_argument("--write-report", action="store_true")
    args = parser.parse_args()

    review_paths = (
        [path.expanduser().resolve() for path in args.quality_review]
        if args.quality_review
        else list(CURRENT_REVIEWS)
    )
    missing_review_paths = [path for path in review_paths if not path.is_file()]
    if missing_review_paths:
        raise FileNotFoundError(f"missing quality review inputs: {missing_review_paths}")

    if sha256_file(RAW) != RAW_SHA256:
        raise RuntimeError("frozen raw SHA drift")
    raw_rows = [row for _line, row in iter_jsonl(RAW)]
    raw_by_id = {str(row["record_id"]): row for row in raw_rows}
    if len(raw_rows) != 46_729 or len(raw_by_id) != 46_729:
        raise RuntimeError("raw row/identity count drift")

    # Verify the two-shard dynamic batch partition, including short terminal batches.
    batch_manifest_files = sorted((OUT / "batches").glob("*/candidate_manifest.jsonl"))
    batch_rows: dict[str, int] = {}
    batch_ids: dict[str, set[str]] = {}
    for path in batch_manifest_files:
        match = BATCH_RE.fullmatch(path.parent.name)
        if match is None:
            continue
        rows = [row for _line, row in iter_jsonl(path)]
        key = f"s{match.group('shard')}_b{match.group('batch')}"
        ids = {str(row["record_id"]) for row in rows}
        if len(ids) != len(rows):
            raise RuntimeError(f"duplicate IDs in {path}")
        batch_rows[key] = len(rows)
        batch_ids[key] = ids
    all_batched_ids: set[str] = set()
    overlapping_batch_ids: set[str] = set()
    for ids in batch_ids.values():
        overlapping_batch_ids |= all_batched_ids & ids
        all_batched_ids |= ids
    if overlapping_batch_ids or all_batched_ids != set(raw_by_id):
        raise RuntimeError("initial shard manifests do not partition frozen raw")
    expected_terminal = {"s00_b00047": 6, "s01_b00046": 223}
    if any(batch_rows.get(key) != count for key, count in expected_terminal.items()):
        raise RuntimeError("terminal dynamic batch length drift")
    unexpected_short = {
        key: count
        for key, count in batch_rows.items()
        if count != 500 and key not in expected_terminal
    }
    if unexpected_short:
        raise RuntimeError(f"unexpected short batches: {unexpected_short}")

    reviews, review_stats = load_reviews(review_paths)
    if set(reviews) != set(raw_by_id):
        raise RuntimeError("current review inputs do not cover exactly frozen raw")

    candidate_paths = resolve_jsonl_inputs(
        list(RECEIPT_ROOTS), exact_basename="candidate_manifest.jsonl"
    )
    receipt_paths = resolve_jsonl_inputs(
        list(RECEIPT_ROOTS), basename_contains="verification_results"
    )
    candidates, candidate_stats = load_candidate_manifests(candidate_paths)
    receipts, receipt_stats = load_receipts(receipt_paths, candidates)
    unknown_receipts = sorted(set(receipts) - set(raw_by_id))
    unknown_candidates = sorted({key[0] for key in candidates} - set(raw_by_id))
    if unknown_receipts or unknown_candidates:
        raise RuntimeError("receipt/candidate inputs contain IDs outside frozen raw")

    parents, parent_stats = load_parent_source(PARENT)
    ledger: Counter[str] = Counter()
    field_contracts: Counter[str] = Counter()
    mismatch_counts: Counter[str] = Counter()
    selected_environments: Counter[str] = Counter()
    conflicts: list[str] = []
    review_hashes_validated = 0
    for record_id, raw in raw_by_id.items():
        review = reviews[record_id]
        candidate = synchronize_formal_content(apply_review(raw, review))
        # Audit hashes even for provisional pending rows; production validates
        # pass/reject rows at finalization time.
        validate_review_hashes(review, candidate)
        review_hashes_validated += 1
        parent_ref = str(raw.get("upstream_source") or "")
        if not parent_ref.startswith("parent:"):
            raise RuntimeError(f"invalid parent provenance: {record_id}")
        parent_id = parent_ref[len("parent:") :]
        parent = parents.get(parent_id)
        if parent is None:
            raise RuntimeError(f"missing canonical parent for {record_id}")
        decision = decision_of(review)
        if decision == "reject":
            ledger["quality_rejected"] += 1
            continue
        if decision == "pending":
            ledger["pending_review"] += 1
            continue
        compatible = compatible_receipts(
            receipts.get(record_id, ()),
            candidate,
            expected_environment_hash=ENVIRONMENT_HASH,
            mismatch_counts=mismatch_counts,
            frozen_formal_statement_sha256=(
                sha256_text(str(raw.get("formal_statement") or "").strip())
                if normalize_newlines(str(candidate.get("lean_statement") or "")).strip()
                == normalize_newlines(str(raw.get("lean_statement") or "")).strip()
                else ""
            ),
        )
        if not compatible:
            ledger["pending_verification"] += 1
            continue
        try:
            receipt, _superseded = select_compatible_receipt(compatible)
        except RuntimeError:
            conflicts.append(record_id)
            continue
        status = SUCCESS if receipt.success else FAIL
        candidate["question_type"] = parent["question_type"]
        output = materialize_row(candidate, receipt, status=status)
        expected_fields = set(raw) | {"pantograph_verified"}
        if status == SUCCESS:
            expected_fields -= {"error_message", "repair_error"}
            if {"error_message", "repair_error"} & set(output):
                raise RuntimeError(f"success error-field contamination: {record_id}")
            ledger["verified_success"] += 1
        else:
            expected_fields |= {"error_message", "repair_error"}
            if not output.get("error_message") or output.get("repair_error") != output.get("error_message"):
                raise RuntimeError(f"fail diagnostics contract mismatch: {record_id}")
            ledger["verified_fail"] += 1
        if set(output) != expected_fields:
            raise RuntimeError(f"final field-set contract mismatch: {record_id}")
        if output.get("pantograph_verified") != status:
            raise RuntimeError(f"pantograph status contract mismatch: {record_id}")
        field_contracts[status] += 1
        selected_environments[str(output.get("environment_hash") or "")] += 1

    if conflicts:
        raise RuntimeError(f"conflicting compatible receipt outcomes: {conflicts[:10]}")
    if sum(ledger.values()) != len(raw_rows):
        raise RuntimeError("in-memory readiness ledger does not cover raw")

    raw_with_any_pantograph_receipt = {
        record_id
        for record_id, rows in receipts.items()
        if any("pantograph" in row.backend.lower() for row in rows)
    }
    raw_without_any_receipt = set(raw_by_id) - raw_with_any_pantograph_receipt
    missing_receipt_by_batch = {
        key: len(ids & raw_without_any_receipt)
        for key, ids in sorted(batch_ids.items())
        if ids & raw_without_any_receipt
    }
    tail_rows = [reviews[record_id] for key in ("s00_b00045", "s00_b00046", "s00_b00047") for record_id in batch_ids[key]]
    pre_decisions = Counter(str(row.get("pre_quality_decision") or "") for row in tail_rows)
    tail_candidate_ids = {
        str(row["record_id"])
        for row in tail_rows
        if row.get("pre_quality_decision") == "candidate"
    }
    tail_reject_ids = {
        str(row["record_id"])
        for row in tail_rows
        if row.get("pre_quality_decision") == "reject"
    }
    tail_candidate_receipt = Counter()
    for record_id in tail_candidate_ids:
        rows = receipts.get(record_id, ())
        if not rows:
            tail_candidate_receipt["missing"] += 1
        elif any(row.success for row in rows):
            tail_candidate_receipt["has_success"] += 1
        else:
            tail_candidate_receipt["fail_only"] += 1

    current_review_files = [file_info(path) for path in review_paths]
    receipt_file_info = [file_info(path) for path in receipt_paths]
    candidate_file_info = [file_info(path) for path in candidate_paths]
    report: dict[str, Any] = {
        "schema_version": "numinamath_finalization_readiness_v40d_v1",
        # This readiness gate is intentionally stricter than the production
        # finalizer's --require-complete check. The finalizer does not need a
        # receipt for quality-rejected rows, but the dataset contract requires
        # evidence that every expansion was submitted to Pantograph.
        "ready_for_require_complete": not (
            ledger["pending_review"]
            or ledger["pending_verification"]
            or raw_without_any_receipt
        ),
        "frozen_raw": {
            "path": str(RAW.relative_to(ROOT)),
            "rows": len(raw_rows),
            "sha256": RAW_SHA256,
        },
        "dynamic_shards": {
            "batches": len(batch_rows),
            "partitioned_rows": len(all_batched_ids),
            "terminal_batch_lengths": expected_terminal,
            "missing_receipts_by_batch": missing_receipt_by_batch,
        },
        "current_review_inputs": {
            "files": current_review_files,
            "coverage": len(reviews),
            "production_decisions": review_stats["decisions"],
            "tail_pre_decisions": dict(sorted(pre_decisions.items())),
            "tail_candidate_receipts": dict(sorted(tail_candidate_receipt.items())),
            "tail_reject_without_receipt": len(tail_reject_ids & raw_without_any_receipt),
            "hashes_validated_against_raw_or_replacement": review_hashes_validated,
        },
        "candidate_manifests": {
            **candidate_stats,
            "file_count": len(candidate_paths),
            "files": candidate_file_info,
        },
        "verification_receipts": {
            **receipt_stats,
            "file_count": len(receipt_paths),
            "files": receipt_file_info,
            "raw_ids_without_any_receipt": len(raw_without_any_receipt),
            "all_raw_ids_have_pantograph_receipt": not raw_without_any_receipt,
            "required_raw_id_coverage": len(raw_rows),
            "covered_raw_ids": len(raw_rows) - len(raw_without_any_receipt),
            "coverage_backend_rule": "receipt backend name must contain pantograph",
            "environment_hashes_selected": dict(selected_environments),
            "compatibility_mismatches": dict(mismatch_counts),
        },
        "parent_source": {
            **parent_stats,
            "path": str(PARENT.relative_to(ROOT)),
            "sha256": sha256_file(PARENT),
        },
        "in_memory_finalizer_ledger": dict(sorted(ledger.items())),
        "field_contracts_validated": dict(sorted(field_contracts.items())),
        "override_semantics": {
            "finalizer_last_write_wins": False,
            "duplicate_review_record_id_is_error": True,
            "required_workflow": (
                "Merge base shards and all overlapping corrections/repair reviews into one "
                "unique final checkpoint before invoking the finalizer."
            ),
            "receipt_selection": (
                "Only exact content/environment-compatible receipts participate; same-outcome "
                "duplicates select the last, timeout failures may be superseded by success, "
                "and any non-timeout fail/success conflict is fatal."
            ),
        },
        "current_gaps": {
            "pending_review_rows": ledger["pending_review"],
            "pending_verification_for_current_pass_rows": ledger["pending_verification"],
            "v40d_quality_accepted_pending_second_review": 161,
            "tail_candidates_not_yet_formalized": len(tail_candidate_ids),
            "tail_candidates_with_success_receipt": tail_candidate_receipt["has_success"],
            "tail_candidates_with_fail_only_receipt": tail_candidate_receipt["fail_only"],
            "tail_candidates_missing_receipt": tail_candidate_receipt["missing"],
            "raw_rows_missing_any_pantograph_receipt": len(raw_without_any_receipt),
            "tail_pre_rejects_ready_to_formalize_after_receipt_gate": len(tail_reject_ids),
            "manual_v4_supplement_optional_and_not_yet_compiled": True,
        },
        "finalizer_field_contract": {
            "success": (
                "exact frozen raw field set plus pantograph_verified; error_message and "
                "repair_error absent; hashes/record_hash recomputed; question_type restored "
                "from canonical parent"
            ),
            "fail": (
                "exact frozen raw field set plus pantograph_verified, error_message and "
                "repair_error; both diagnostics nonempty/equal; hashes/record_hash recomputed; "
                "question_type restored from canonical parent"
            ),
        },
        "exact_readiness_command_after_gaps_close": [
            "PYTHONPATH=.",
            ".venv/bin/python",
            "scripts/audit_numinamath_finalization_readiness.py",
            "--quality-review", "<hash-locked-full-46729-row-final-review-checkpoint.jsonl>",
            "--report", "outputs/numinamath_expand_verification/finalization_readiness_final.report.json",
            "--write-report",
        ],
        "exact_final_command_after_gaps_close": [
            "PYTHONPATH=.",
            ".venv/bin/python",
            "scripts/finalize_numinamath_expand_verification.py",
            "--raw", "lean_prover/Dataset/raw_data/numinamath_expand.jsonl",
            "--expected-raw-sha256", RAW_SHA256,
            "--verification-results", "outputs/numinamath_expand_verification/batches",
            "--verification-results", "outputs/numinamath_expand_verification/repairs",
            "--candidate-manifests", "outputs/numinamath_expand_verification/batches",
            "--candidate-manifests", "outputs/numinamath_expand_verification/repairs",
            "--quality-review", "<hash-locked-full-46729-row-final-review-checkpoint.jsonl>",
            "--parent-source", "lean_prover/Dataset/verified_data/numinamath_verified_success.jsonl",
            "--verified-supplement", "<rounds01-07-v4-verified-supplement.jsonl>",
            "--expected-supplement-sha256", "<exact-lowercase-supplement-sha256>",
            "--expected-environment-hash", ENVIRONMENT_HASH,
            "--success-output", "lean_prover/Dataset/verified_data/numinamath_expand_verified_success.jsonl",
            "--fail-output", "lean_prover/Dataset/verified_data/numinamath_expand_verified_fail.jsonl",
            "--quality-rejected-output", "lean_prover/Dataset/verified_data/numinamath_expand_quality_rejected.jsonl",
            "--pending-output", "lean_prover/Dataset/verified_data/numinamath_expand_pending.jsonl",
            "--audit-output", "lean_prover/Dataset/verified_data/numinamath_expand_finalization_audit.json",
            "--max-low-quality-ratio", "0.05",
            "--require-complete",
        ],
        "lake_lean_pantograph_started": False,
        "raw_or_final_verified_written": False,
    }
    payload = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if args.write_report:
        path = args.report.expanduser().resolve()
        if path.exists() and path.read_text(encoding="utf-8") != payload:
            raise FileExistsError(f"refusing to overwrite non-identical report: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(payload, encoding="utf-8")
        print(json.dumps({"report": str(path), "sha256": sha256_file(path)}, indent=2))
    else:
        print(payload, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
