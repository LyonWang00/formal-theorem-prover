#!/usr/bin/env python3
"""Build the one production NuminaMath review checkpoint, after all gates pass.

The output is deterministic raw-order JSONL.  This script deliberately refuses
to write it unless all 46,729 frozen raw records have a Pantograph receipt,
every review input is explicitly SHA-256 locked, all provisional decisions have
been resolved, and every quality-pass row has an exact content/environment-
compatible receipt.  It never invokes Lean/Pantograph and never writes raw or
verified datasets.

Override precedence is fixed and explicit:

    v40d/base < round58 final < round59 final < b45 reviewed resolutions
    < compile-fail quality override

The three official s00 tail files are disjoint additions, not overrides.  An
override may only replace a row that is still pending at that precedence level;
therefore no later input can silently change an accepted or rejected decision.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

from scripts.finalize_numinamath_expand_verification import (
    apply_review,
    compatible_receipts,
    decision_of,
    iter_jsonl,
    load_candidate_manifests,
    load_receipts,
    load_reviews,
    normalize_newlines,
    resolve_jsonl_inputs,
    select_compatible_receipt,
    sha256_file,
    sha256_text,
    synchronize_formal_content,
    valid_sha256,
    validate_review_hashes,
)


PROJECT = Path(__file__).resolve().parents[1]
OUT = PROJECT / "outputs/numinamath_expand_verification"
RAW = PROJECT / "lean_prover/Dataset/raw_data/numinamath_expand.jsonl"
RAW_SHA256 = "dd66edc02676dc3d459eef84304e02bc4354c3c63a5d4bd8791c78b1ee79d60e"
BASE = OUT / "manual_review_clean/reviewed_045723_v40d.jsonl"
BASE_SHA256 = "802b83749f7caec105bd13c6197c764db301c20135a9ffa3999b2d90b5bd154a"
ENVIRONMENT_HASH = "46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a"
TAIL_BATCHES = ("s00_b00045", "s00_b00046", "s00_b00047")
TAIL_LENGTHS = (500, 500, 6)
TAIL_MANIFEST_SHA256 = (
    "8cb6a5c28a181770dfc166cd7a12ae9465ff0e0b13fac64019042f5c361f591c",
    "863d0e990303be2eb46982906c0fc5ebb2c870d33bab7f8ce6145fa0c97d2260",
    "1410f2b2f3c1751d47567f3c3ffbb7076b19148c11e10fa8635514f2c9f429e1",
)
RECEIPT_ROOTS = (OUT / "batches", OUT / "repairs")
DEFAULT_OUTPUT = OUT / "manual_review_clean/reviewed_046729_final.jsonl"


def locked_file(path: Path, expected_sha256: str, role: str) -> tuple[Path, str]:
    resolved = path.expanduser().resolve()
    expected = str(expected_sha256 or "").strip()
    if not valid_sha256(expected):
        raise ValueError(f"{role}: an explicit lowercase SHA-256 lock is required")
    if not resolved.is_file():
        raise FileNotFoundError(f"{role}: missing input {resolved}")
    actual = sha256_file(resolved)
    if actual != expected:
        raise ValueError(f"{role}: SHA mismatch; expected {expected}, got {actual}")
    return resolved, actual


def load_locked_review(
    path: Path, expected_sha256: str, role: str
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    resolved, actual = locked_file(path, expected_sha256, role)
    rows, stats = load_reviews([resolved])
    return rows, {
        "role": role,
        "path": str(resolved.relative_to(PROJECT)),
        "sha256": actual,
        "rows": len(rows),
        "decisions": stats["decisions"],
    }


def atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(f"refusing to overwrite {path}")
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            for row in rows:
                handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True))
                handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def inventory(paths: Sequence[Path]) -> tuple[list[dict[str, Any]], str]:
    entries = [
        {
            "path": str(path.resolve().relative_to(PROJECT)),
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
        for path in sorted(path.resolve() for path in paths)
    ]
    payload = json.dumps(entries, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    actual = sha256_text(payload)
    return entries, actual


def locked_inventory(
    paths: Sequence[Path], expected_sha256: str, role: str
) -> tuple[list[dict[str, Any]], str]:
    expected = str(expected_sha256 or "").strip()
    if not valid_sha256(expected):
        raise ValueError(f"{role}: an explicit inventory SHA-256 lock is required")
    entries, actual = inventory(paths)
    if actual != expected:
        raise ValueError(f"{role}: inventory SHA mismatch; expected {expected}, got {actual}")
    return entries, actual


def apply_pending_only_override(
    current: dict[str, dict[str, Any]],
    override: Mapping[str, Mapping[str, Any]],
    *,
    role: str,
) -> set[str]:
    applied: set[str] = set()
    for record_id, row in override.items():
        if record_id not in current:
            raise ValueError(f"{role}: override ID is absent from frozen review universe: {record_id}")
        if decision_of(current[record_id]) != "pending":
            raise ValueError(
                f"{role}: may only resolve a currently pending row, got "
                f"{decision_of(current[record_id])}: {record_id}"
            )
        current[record_id] = dict(row)
        applied.add(record_id)
    return applied


def require_full_pantograph_coverage(
    raw_ids: set[str], receipts: Mapping[str, Sequence[Any]]
) -> set[str]:
    covered = {
        record_id
        for record_id, rows in receipts.items()
        if record_id in raw_ids
        and any("pantograph" in str(row.backend).lower() for row in rows)
    }
    missing = raw_ids - covered
    if missing:
        sample = sorted(missing)[:10]
        raise RuntimeError(
            "full checkpoint is blocked: Pantograph receipt coverage is "
            f"{len(covered)}/{len(raw_ids)}; missing={len(missing)}, sample={sample}"
        )
    return covered


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--raw", type=Path, default=RAW)
    result.add_argument("--expected-raw-sha256", default=RAW_SHA256)
    result.add_argument("--base-review", type=Path, default=BASE)
    result.add_argument("--expected-base-review-sha256", default=BASE_SHA256)
    result.add_argument(
        "--tail-review",
        type=Path,
        action="append",
        default=[],
        help="Official final s00 tail review; repeat in b45, b46, b47 order.",
    )
    result.add_argument(
        "--expected-tail-review-sha256",
        action="append",
        default=[],
        help="SHA lock paired positionally with --tail-review.",
    )
    for round_name in ("round58", "round59"):
        result.add_argument(f"--{round_name}-final-review", type=Path)
        result.add_argument(
            f"--expected-{round_name}-final-review-sha256"
        )
    result.add_argument("--compile-fail-quality-override", type=Path)
    result.add_argument(
        "--expected-compile-fail-quality-override-sha256"
    )
    result.add_argument("--b45-success-review", type=Path)
    result.add_argument("--expected-b45-success-review-sha256")
    result.add_argument("--b45-reject-review", type=Path)
    result.add_argument("--expected-b45-reject-review-sha256")
    result.add_argument(
        "--verification-results",
        type=Path,
        action="append",
        default=[],
        help="Receipt JSONL/directory; defaults to batches and repairs.",
    )
    result.add_argument(
        "--candidate-manifests",
        type=Path,
        action="append",
        default=[],
        help="Manifest JSONL/directory; defaults to batches and repairs.",
    )
    result.add_argument("--expected-receipt-inventory-sha256")
    result.add_argument("--expected-candidate-inventory-sha256")
    result.add_argument(
        "--print-input-inventory-locks",
        action="store_true",
        help="Print current receipt/manifest inventory hashes and exit without reviews/writes.",
    )
    result.add_argument("--expected-environment-hash", default=ENVIRONMENT_HASH)
    result.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    result.add_argument("--report", type=Path)
    result.add_argument("--write", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    if not valid_sha256(str(args.expected_environment_hash or "").strip()):
        raise ValueError("--expected-environment-hash must be an explicit SHA-256")
    receipt_roots = args.verification_results or list(RECEIPT_ROOTS)
    candidate_roots = args.candidate_manifests or list(RECEIPT_ROOTS)
    receipt_paths = resolve_jsonl_inputs(
        [path.expanduser().resolve() for path in receipt_roots],
        basename_contains="verification_results",
    )
    candidate_paths = resolve_jsonl_inputs(
        [path.expanduser().resolve() for path in candidate_roots],
        exact_basename="candidate_manifest.jsonl",
    )
    if args.print_input_inventory_locks:
        receipt_entries, receipt_sha = inventory(receipt_paths)
        candidate_entries, candidate_sha = inventory(candidate_paths)
        print(
            json.dumps(
                {
                    "verification_receipts": {
                        "files": len(receipt_entries),
                        "inventory_sha256": receipt_sha,
                    },
                    "candidate_manifests": {
                        "files": len(candidate_entries),
                        "inventory_sha256": candidate_sha,
                    },
                    "lake_lean_pantograph_started": False,
                    "files_written": False,
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    required_named_inputs = {
        "--round58-final-review": args.round58_final_review,
        "--expected-round58-final-review-sha256": args.expected_round58_final_review_sha256,
        "--round59-final-review": args.round59_final_review,
        "--expected-round59-final-review-sha256": args.expected_round59_final_review_sha256,
        "--b45-success-review": args.b45_success_review,
        "--expected-b45-success-review-sha256": (
            args.expected_b45_success_review_sha256
        ),
        "--b45-reject-review": args.b45_reject_review,
        "--expected-b45-reject-review-sha256": (
            args.expected_b45_reject_review_sha256
        ),
        "--compile-fail-quality-override": args.compile_fail_quality_override,
        "--expected-compile-fail-quality-override-sha256": (
            args.expected_compile_fail_quality_override_sha256
        ),
        "--expected-receipt-inventory-sha256": args.expected_receipt_inventory_sha256,
        "--expected-candidate-inventory-sha256": args.expected_candidate_inventory_sha256,
    }
    missing_named_inputs = [name for name, value in required_named_inputs.items() if not value]
    if missing_named_inputs:
        raise ValueError(f"missing required locked inputs: {missing_named_inputs}")
    raw_path, actual_raw_sha = locked_file(
        args.raw, args.expected_raw_sha256, "frozen raw"
    )
    base_rows, base_lock = load_locked_review(
        args.base_review, args.expected_base_review_sha256, "v40d base"
    )
    if len(args.tail_review) != 3 or len(args.expected_tail_review_sha256) != 3:
        raise ValueError("exactly three tail review/SHA pairs are required: s00 b45,b46,b47")

    raw_list = [row for _line, row in iter_jsonl(raw_path)]
    raw_by_id = {str(row["record_id"]): row for row in raw_list}
    if len(raw_list) != 46_729 or len(raw_by_id) != 46_729:
        raise ValueError("frozen raw must contain 46729 unique record_ids")

    # Lock the initial shard manifests to the raw partition and use them to
    # define the exact three tail ID sets, including the dynamic six-row end.
    tail_id_sets: list[set[str]] = []
    tail_manifest_locks: list[dict[str, Any]] = []
    for batch, expected_length, expected_manifest_sha in zip(
        TAIL_BATCHES, TAIL_LENGTHS, TAIL_MANIFEST_SHA256
    ):
        manifest = OUT / "batches" / f"numinamath_expand_{batch}" / "candidate_manifest.jsonl"
        manifest, actual_manifest_sha = locked_file(
            manifest, expected_manifest_sha, f"frozen {batch} candidate manifest"
        )
        ids = {str(row["record_id"]) for _line, row in iter_jsonl(manifest)}
        if len(ids) != expected_length:
            raise ValueError(f"{batch}: expected {expected_length} unique manifest IDs")
        tail_id_sets.append(ids)
        tail_manifest_locks.append(
            {
                "role": f"frozen {batch} candidate manifest",
                "path": str(manifest.relative_to(PROJECT)),
                "sha256": actual_manifest_sha,
                "rows": len(ids),
            }
        )
    tail_ids = set().union(*tail_id_sets)
    expected_base_ids = set(raw_by_id) - tail_ids
    if set(base_rows) != expected_base_ids or len(base_rows) != 45_723:
        raise ValueError("v40d base IDs do not equal frozen raw minus s00 b45-b47")

    input_locks: list[dict[str, Any]] = [base_lock, *tail_manifest_locks]
    current = {record_id: dict(row) for record_id, row in base_rows.items()}
    for index, (path, expected_sha, expected_ids) in enumerate(
        zip(args.tail_review, args.expected_tail_review_sha256, tail_id_sets), start=45
    ):
        role = f"official s00 b{index} tail"
        rows, lock = load_locked_review(path, expected_sha, role)
        if set(rows) != expected_ids:
            missing = sorted(expected_ids - set(rows))[:5]
            extra = sorted(set(rows) - expected_ids)[:5]
            raise ValueError(f"{role}: exact batch ID mismatch; missing={missing}, extra={extra}")
        current.update(rows)
        input_locks.append(lock)
    if set(current) != set(raw_by_id):
        raise AssertionError("base plus official tails do not cover exactly frozen raw")

    original_pending = {
        record_id for record_id, row in current.items() if decision_of(row) == "pending"
    }
    if len(original_pending) < 161:
        raise ValueError("expected at least the 161 v40d pending rows before final overrides")

    override_specs = (
        (
            "round58 final review",
            args.round58_final_review,
            args.expected_round58_final_review_sha256,
        ),
        (
            "round59 final review",
            args.round59_final_review,
            args.expected_round59_final_review_sha256,
        ),
    )
    applied_by_role: dict[str, set[str]] = {}
    override_rows_by_role: dict[str, dict[str, dict[str, Any]]] = {}
    for role, path, expected_sha in override_specs:
        rows, lock = load_locked_review(path, expected_sha, role)
        applied_by_role[role] = apply_pending_only_override(current, rows, role=role)
        override_rows_by_role[role] = rows
        input_locks.append(lock)

    # The safe-repair sidecar intentionally records both attempted rows.  Only
    # the exact successful pass is allowed to resolve the official b45
    # baseline; its still-pending sibling must not replace the baseline proof.
    safe_rows, safe_lock = load_locked_review(
        args.b45_success_review,
        args.expected_b45_success_review_sha256,
        "b45 successful safe repair",
    )
    safe_pass_rows = {
        record_id: row
        for record_id, row in safe_rows.items()
        if decision_of(row) == "pass"
    }
    safe_pending_ids = {
        record_id
        for record_id, row in safe_rows.items()
        if decision_of(row) == "pending"
    }
    if len(safe_rows) != 2 or len(safe_pass_rows) != 1 or len(safe_pending_ids) != 1:
        raise ValueError(
            "b45 successful safe repair must contain exactly one pass and one "
            "audit-only pending row"
        )
    safe_role = "b45 successful safe repair"
    applied_by_role[safe_role] = apply_pending_only_override(
        current, safe_pass_rows, role=safe_role
    )
    override_rows_by_role[safe_role] = safe_pass_rows
    input_locks.append(safe_lock)

    reject_rows, reject_lock = load_locked_review(
        args.b45_reject_review,
        args.expected_b45_reject_review_sha256,
        "b45 pending second-review rejects",
    )
    if len(reject_rows) != 2 or any(
        decision_of(row) != "reject" for row in reject_rows.values()
    ):
        raise ValueError(
            "b45 pending second-review reject input must contain exactly two rejects"
        )
    reject_role = "b45 pending second-review rejects"
    applied_by_role[reject_role] = apply_pending_only_override(
        current, reject_rows, role=reject_role
    )
    override_rows_by_role[reject_role] = reject_rows
    input_locks.append(reject_lock)

    compile_role = "compile-fail quality override"
    compile_rows, compile_lock = load_locked_review(
        args.compile_fail_quality_override,
        args.expected_compile_fail_quality_override_sha256,
        compile_role,
    )
    applied_by_role[compile_role] = apply_pending_only_override(
        current, compile_rows, role=compile_role
    )
    override_rows_by_role[compile_role] = compile_rows
    input_locks.append(compile_lock)

    final_decisions = Counter(decision_of(row) for row in current.values())
    if final_decisions["pending"]:
        pending = sorted(
            record_id for record_id, row in current.items() if decision_of(row) == "pending"
        )
        raise RuntimeError(
            f"full checkpoint is blocked: {len(pending)} review decisions remain pending; "
            f"sample={pending[:10]}"
        )

    compile_fail_ids = applied_by_role["compile-fail quality override"]
    if any(decision_of(current[record_id]) != "pass" for record_id in compile_fail_ids):
        raise ValueError("compile-fail quality override may contain only final quality passes")

    receipt_inventory, receipt_inventory_sha = locked_inventory(
        receipt_paths,
        args.expected_receipt_inventory_sha256,
        "verification receipt inventory",
    )
    candidate_inventory, candidate_inventory_sha = locked_inventory(
        candidate_paths,
        args.expected_candidate_inventory_sha256,
        "candidate manifest inventory",
    )
    candidates, candidate_stats = load_candidate_manifests(candidate_paths)
    receipts, receipt_stats = load_receipts(receipt_paths, candidates)
    unknown_receipts = sorted(set(receipts) - set(raw_by_id))
    if unknown_receipts:
        raise ValueError(f"receipt inputs contain IDs outside frozen raw: {unknown_receipts[:10]}")
    covered = require_full_pantograph_coverage(set(raw_by_id), receipts)

    mismatch_counts: Counter[str] = Counter()
    selected_outcomes: Counter[str] = Counter()
    round_success_ids = (
        applied_by_role["round58 final review"]
        | applied_by_role["round59 final review"]
    ) - compile_fail_ids
    for record_id, raw in raw_by_id.items():
        review = current[record_id]
        candidate = synchronize_formal_content(apply_review(raw, review))
        validate_review_hashes(review, candidate)
        if decision_of(review) != "pass":
            continue
        compatible = compatible_receipts(
            receipts.get(record_id, ()),
            candidate,
            expected_environment_hash=args.expected_environment_hash,
            mismatch_counts=mismatch_counts,
            frozen_formal_statement_sha256=(
                sha256_text(str(raw.get("formal_statement") or "").strip())
                if normalize_newlines(str(candidate.get("lean_statement") or "")).strip()
                == normalize_newlines(str(raw.get("lean_statement") or "")).strip()
                else ""
            ),
        )
        if not compatible:
            raise RuntimeError(f"quality-pass row lacks an exact compatible receipt: {record_id}")
        selected, _superseded_timeout = select_compatible_receipt(compatible)
        selected_outcomes["success" if selected.success else "fail"] += 1
        if not selected.success and record_id not in compile_fail_ids:
            raise ValueError(
                f"an exact failed receipt requires the explicit compile-fail quality "
                f"override role: {record_id}"
            )
        if record_id in compile_fail_ids and selected.success:
            raise ValueError(
                f"compile-fail override points at a currently successful exact receipt: {record_id}"
            )
        if record_id in round_success_ids and not selected.success:
            raise ValueError(
                f"round58/59 pass must have an exact successful receipt; use the "
                f"compile-fail quality override for failures: {record_id}"
            )

    ordered_rows = [current[str(raw["record_id"])] for raw in raw_list]
    if len({str(row["record_id"]) for row in ordered_rows}) != 46_729:
        raise AssertionError("final review ordering lost record identity")
    report: dict[str, Any] = {
        "schema_version": "numinamath_full_review_checkpoint_builder_v1",
        "ready": True,
        "raw": {
            "path": str(raw_path.relative_to(PROJECT)),
            "rows": len(raw_list),
            "sha256": actual_raw_sha,
        },
        "input_precedence": [
            "v40d base plus disjoint official tails",
            "round58 final review",
            "round59 final review",
            "b45 successful safe repair",
            "b45 pending second-review rejects",
            "compile-fail quality override",
        ],
        "input_locks": input_locks,
        "rows": len(ordered_rows),
        "unique_record_ids": len({str(row["record_id"]) for row in ordered_rows}),
        "decisions": dict(sorted(final_decisions.items())),
        "overrides_applied": {
            role: len(ids) for role, ids in applied_by_role.items()
        },
        "original_pending_ids": len(original_pending),
        "pantograph_receipt_coverage": {
            "covered": len(covered),
            "required": len(raw_by_id),
            "complete": True,
            "receipt_files": len(receipt_paths),
        },
        "receipt_inventory": {
            "sha256": receipt_inventory_sha,
            "files": receipt_inventory,
        },
        "candidate_inventory": {
            "sha256": candidate_inventory_sha,
            "files": candidate_inventory,
        },
        "candidate_manifests": {
            **candidate_stats,
            "files": len(candidate_paths),
        },
        "receipt_stats": receipt_stats,
        "selected_quality_pass_receipts": dict(sorted(selected_outcomes.items())),
        "compatibility_mismatches_ignored_for_other_candidates": dict(
            sorted(mismatch_counts.items())
        ),
        "write_requested": args.write,
        "lake_lean_pantograph_started": False,
        "raw_or_verified_files_written": False,
    }

    if not args.write:
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    output = args.output.expanduser().resolve()
    report_path = (
        args.report.expanduser().resolve()
        if args.report
        else output.with_suffix(".report.json")
    )
    # Check both destinations before creating either one.
    if output.exists() or report_path.exists():
        raise FileExistsError(f"refusing to overwrite output/report: {output}, {report_path}")
    atomic_write_jsonl(output, ordered_rows)
    report.update(
        {
            "output": str(output.relative_to(PROJECT)),
            "output_sha256": sha256_file(output),
        }
    )
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
