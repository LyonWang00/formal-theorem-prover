"""Materialize project datasets and finish LeanWorkbook Pantograph coverage.

The module deliberately reuses the training package's LeanWorkbook trajectory
reconstruction, theorem assembly, environment identity, and persistent
Pantograph pool.  It adds only the storage contract requested for the project
level ``Dataset`` directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

from lean_prover.lean_training.data.lean_workbook import (
    reconstruct_lean_workbook_records,
    render_tactic_proof,
)
from lean_prover.lean_training.data.preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
    ProofFormat,
    compose_lean_theorem,
)
from lean_prover.lean_training.expert_iteration.utils import environment_identity
from lean_prover.lean_training.verification.pantograph import build_labeled_lean_code
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)
from lean_prover.lean_training.verification.schema import VerificationTask


SUCCESS = "success"
FAIL = "fail"
HASH_SCHEMA_VERSION = "dataset-record-sha256-v1"
LEANWORKBOOK_SOURCE = "InternLM/Lean-Workbook"
LEANDOJO_SOURCE = "LeanDojo-v2/current-mathlib"
DEFAULT_SOURCE_COMMIT = "2e066e310b2c6d2c27616927ae131f82901c8f1c"


def _json_text(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _first_text(row: Mapping[str, Any], keys: Sequence[str]) -> str:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def dataset_record_hash(row: Mapping[str, Any], *, source: str) -> str:
    """Hash stable theorem content, excluding mutable verification metadata."""

    payload = {
        "source": source,
        "record_id": _first_text(row, ("record_id", "id", "statement_id")),
        "statement": _first_text(
            row,
            ("statement", "lean_statement", "formal_statement"),
        ),
        "proof": _first_text(
            row,
            ("proof", "completion", "reference_proof", "generated_proof"),
        ),
    }
    return sha256_text(_json_text(payload))


def add_dataset_contract(
    row: Mapping[str, Any],
    *,
    source: str,
    status: str,
    error_message: str | None = None,
) -> dict[str, Any]:
    """Return one row with the common source, verification, and hash fields.

    The deprecated misspelling ``sourcce`` is removed if it is present in an
    older materialized row; ``source`` is the only canonical origin field.
    """

    if status not in {SUCCESS, FAIL}:
        raise ValueError(f"unsupported Pantograph status: {status!r}")
    payload = dict(row)
    statement = _first_text(
        payload,
        ("statement", "lean_statement", "formal_statement"),
    )
    proof = _first_text(
        payload,
        ("proof", "completion", "reference_proof", "generated_proof"),
    )
    payload.update(
        {
            "pantograph_verified": status,
            "source": source,
            "hash_schema_version": HASH_SCHEMA_VERSION,
            "statement_sha256": sha256_text(statement),
            "proof_sha256": sha256_text(proof),
        }
    )
    payload.pop("sourcce", None)
    payload["record_hash"] = dataset_record_hash(payload, source=source)
    if status == FAIL:
        message = str(error_message or payload.get("error_message") or "").strip()
        if not message:
            message = "Pantograph verification failed without diagnostic text."
        payload["error_message"] = message
    else:
        payload.pop("error_message", None)
    validate_contract_row(payload, expected_status=status)
    return payload


def validate_contract_row(
    row: Mapping[str, Any],
    *,
    expected_status: str | None = None,
) -> None:
    status = str(row.get("pantograph_verified") or "")
    if status not in {SUCCESS, FAIL}:
        raise ValueError("pantograph_verified must be 'success' or 'fail'")
    if expected_status is not None and status != expected_status:
        raise ValueError(f"expected {expected_status!r}, found {status!r}")
    source = str(row.get("source") or "")
    if not source:
        raise ValueError("source must be a non-empty string")
    if "sourcce" in row:
        raise ValueError("deprecated misspelled field sourcce must not be present")
    for key in ("record_hash", "statement_sha256", "proof_sha256"):
        value = str(row.get(key) or "")
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"{key} must be a lowercase SHA-256 digest")
    if status == FAIL and not str(row.get("error_message") or "").strip():
        raise ValueError("failed rows require a non-empty error_message")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def append_jsonl(path: Path, row: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
        handle.flush()


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            count += 1
    return count


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )


def materialize_frozen_datasets(
    *,
    dataset_root: Path,
    leanworkbook_raw: Path,
    leandojo_raw: Path,
    leanworkbook_verified: Path,
    leandojo_verified: Path,
) -> dict[str, Any]:
    """Copy raw inputs and write contract-normalized frozen success pools."""

    raw_dir = dataset_root / "raw_data"
    verified_dir = dataset_root / "verified_data"
    raw_dir.mkdir(parents=True, exist_ok=True)
    verified_dir.mkdir(parents=True, exist_ok=True)

    raw_targets = {
        "leanworkbook": raw_dir / "leanworkbook_raw.parquet",
        "leandojo": raw_dir / "leandojo_raw.jsonl",
    }
    for key, source_path in (
        ("leanworkbook", leanworkbook_raw),
        ("leandojo", leandojo_raw),
    ):
        target = raw_targets[key]
        if source_path.resolve() != target.resolve():
            shutil.copy2(source_path, target)

    outputs = {
        "leanworkbook": verified_dir / "leanworkbook_verified_success_1.jsonl",
        "leandojo": verified_dir / "leandojo_verified_success_1.jsonl",
    }
    counts: dict[str, int] = {}
    for key, input_path, source_label in (
        ("leanworkbook", leanworkbook_verified, LEANWORKBOOK_SOURCE),
        ("leandojo", leandojo_verified, LEANDOJO_SOURCE),
    ):
        seen_ids: set[str] = set()

        def normalized_rows() -> Iterable[dict[str, Any]]:
            with input_path.open("r", encoding="utf-8-sig") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    record_id = _first_text(row, ("record_id", "id", "statement_id"))
                    if not record_id:
                        raise ValueError(f"{input_path}:{line_number}: missing ID")
                    if record_id in seen_ids:
                        raise ValueError(f"{input_path}:{line_number}: duplicate {record_id}")
                    seen_ids.add(record_id)
                    yield add_dataset_contract(
                        row,
                        source=source_label,
                        status=SUCCESS,
                    )

        counts[key] = write_jsonl(outputs[key], normalized_rows())

    manifest = {
        "schema_version": HASH_SCHEMA_VERSION,
        "raw": {
            key: {
                "path": str(path.resolve()),
                "sha256": sha256_file(path),
                "bytes": path.stat().st_size,
            }
            for key, path in raw_targets.items()
        },
        "verified_success_1": {
            key: {
                "path": str(outputs[key].resolve()),
                "rows": counts[key],
                "sha256": sha256_file(outputs[key]),
            }
            for key in outputs
        },
        "contract": {
            "pantograph_verified": [SUCCESS, FAIL],
            "source_fields": ["source"],
            "hash_fields": [
                "record_hash",
                "statement_sha256",
                "proof_sha256",
            ],
        },
    }
    _write_json(dataset_root / "dataset_manifest.json", manifest)
    return manifest


def audit_materialized_datasets(*, dataset_root: Path) -> dict[str, Any]:
    """Validate every materialized row and the WB partition invariants."""

    raw_dir = dataset_root / "raw_data"
    verified_dir = dataset_root / "verified_data"
    specifications = (
        (
            "leanworkbook_success_1",
            verified_dir / "leanworkbook_verified_success_1.jsonl",
            LEANWORKBOOK_SOURCE,
            SUCCESS,
        ),
        (
            "leanworkbook_success_2",
            verified_dir / "leanworkbook_verified_success_2.jsonl",
            LEANWORKBOOK_SOURCE,
            SUCCESS,
        ),
        (
            "leanworkbook_fail_2",
            verified_dir / "leanworkbook_verified_fail_2.jsonl",
            LEANWORKBOOK_SOURCE,
            FAIL,
        ),
        (
            "leandojo_success_1",
            verified_dir / "leandojo_verified_success_1.jsonl",
            LEANDOJO_SOURCE,
            SUCCESS,
        ),
    )
    file_reports: dict[str, Any] = {}
    ids_by_name: dict[str, set[str]] = {}
    for name, path, source, status in specifications:
        rows = read_jsonl(path)
        ids: set[str] = set()
        for line_number, row in enumerate(rows, start=1):
            try:
                validate_contract_row(row, expected_status=status)
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: {error}") from error
            if row.get("source") != source:
                raise ValueError(
                    f"{path}:{line_number}: source must be {source!r}"
                )
            if row["record_hash"] != dataset_record_hash(row, source=source):
                raise ValueError(f"{path}:{line_number}: record_hash mismatch")
            statement = _first_text(
                row,
                ("statement", "lean_statement", "formal_statement"),
            )
            proof = _first_text(
                row,
                ("proof", "completion", "reference_proof", "generated_proof"),
            )
            if row["statement_sha256"] != sha256_text(statement):
                raise ValueError(f"{path}:{line_number}: statement_sha256 mismatch")
            if row["proof_sha256"] != sha256_text(proof):
                raise ValueError(f"{path}:{line_number}: proof_sha256 mismatch")
            record_id = _first_text(row, ("record_id", "id", "statement_id"))
            if not record_id or record_id in ids:
                raise ValueError(f"{path}:{line_number}: missing or duplicate ID")
            ids.add(record_id)
        ids_by_name[name] = ids
        file_reports[name] = {
            "path": str(path.resolve()),
            "rows": len(rows),
            "unique_ids": len(ids),
            "pantograph_verified": status,
            "source": source,
            "sha256": sha256_file(path),
            "contract_errors": 0,
        }

    wb_names = (
        "leanworkbook_success_1",
        "leanworkbook_success_2",
        "leanworkbook_fail_2",
    )
    wb_sum = sum(len(ids_by_name[name]) for name in wb_names)
    wb_union = set().union(*(ids_by_name[name] for name in wb_names))
    wb_duplicate_ids = wb_sum - len(wb_union)
    if wb_duplicate_ids or len(wb_union) != 13_517:
        raise ValueError(
            "LeanWorkbook partition must contain 13,517 disjoint theorem IDs; "
            f"found union={len(wb_union)}, duplicates={wb_duplicate_ids}"
        )

    import pyarrow.parquet as pq

    wb_raw_path = raw_dir / "leanworkbook_raw.parquet"
    ld_raw_path = raw_dir / "leandojo_raw.jsonl"
    wb_raw_rows = pq.read_metadata(wb_raw_path).num_rows
    with ld_raw_path.open("r", encoding="utf-8-sig") as handle:
        ld_raw_rows = sum(1 for line in handle if line.strip())
    report = {
        "status": "passed",
        "schema_version": HASH_SCHEMA_VERSION,
        "raw": {
            "leanworkbook_trajectory_rows": wb_raw_rows,
            "leandojo_candidate_rows": ld_raw_rows,
        },
        "files": file_reports,
        "leanworkbook_partition": {
            "union_ids": len(wb_union),
            "duplicate_ids_across_files": wb_duplicate_ids,
            "complete": len(wb_union) == 13_517 and wb_duplicate_ids == 0,
        },
    }
    _write_json(dataset_root / "dataset_audit.json", report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def _proof_from_record(record: Mapping[str, Any]) -> str:
    proof = _first_text(record, ("proof", "completion", "reference_proof"))
    if proof:
        return proof
    raw_context = record.get("raw_source_context")
    if not raw_context:
        return ""
    trajectory = json.loads(str(raw_context))
    return render_tactic_proof(str(step.get("tactic") or "") for step in trajectory)


def _verification_task(
    row: Mapping[str, Any],
    *,
    problem_index: int,
    environment_hash: str,
) -> VerificationTask:
    record_id = _first_text(row, ("record_id", "id"))
    statement = _first_text(row, ("statement", "lean_statement", "formal_statement"))
    proof = _proof_from_record(row)
    if not record_id or not statement or not proof:
        raise ValueError("verification requires record ID, statement, and proof")
    declaration = compose_lean_theorem(
        statement,
        proof,
        proof_format=ProofFormat.FULL_PROOF,
    )
    task = VerificationTask(
        priority=problem_index,
        problem_index=problem_index,
        attempt_index=0,
        problem_id=record_id,
        prompt="",
        generated_proof=proof,
        raw_completion=proof,
        lean_code=declaration,
        imports=tuple(row.get("imports") or ("Mathlib",)),
        payload={
            "record_id": record_id,
            "environment_hash": environment_hash,
            "assembler_version": ASSEMBLER_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
        },
        reject_forbidden=True,
    )
    assembled = build_labeled_lean_code(task, include_imports=True)
    task.payload["assembled_source_hash"] = sha256_text(assembled)
    return task


def _result_error(result: Mapping[str, Any]) -> str:
    parts: list[str] = []
    for value in (
        result.get("diagnostics"),
        result.get("rejected_reason"),
        result.get("status"),
    ):
        text = str(value or "").strip()
        if text and text not in parts:
            parts.append(text)
    for value in result.get("compile_errors") or ():
        text = str(value or "").strip()
        if text and text not in parts:
            parts.append(text)
    return "\n".join(parts) or "Pantograph verification failed."


def _compatible_result(
    result: Mapping[str, Any],
    task: VerificationTask,
    *,
    environment_hash: str,
) -> bool:
    return (
        str(result.get("record_id") or result.get("problem_id") or "")
        == task.problem_id
        and result.get("environment_hash") == environment_hash
        and result.get("assembler_version") == ASSEMBLER_VERSION
        and result.get("normalization_version") == NORMALIZATION_VERSION
        and result.get("assembled_source_hash")
        == (task.payload or {}).get("assembled_source_hash")
    )


def _load_result_index(paths: Iterable[Path]) -> dict[str, list[dict[str, Any]]]:
    index: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        for row in read_jsonl(path):
            record_id = str(row.get("record_id") or row.get("problem_id") or "")
            if record_id:
                index.setdefault(record_id, []).append(row)
    return index


def verify_remaining_leanworkbook(
    *,
    dataset_root: Path,
    lean_project: Path,
    source_commit: str,
    prior_results: Sequence[Path],
    workers: int,
    timeout: int,
    batch_size: int,
    max_new_compilations: int | None = None,
    reports_only: bool = False,
) -> dict[str, Any]:
    """Finish missing theorem-level verification and split remaining WB rows."""

    import pyarrow.parquet as pq

    raw_path = dataset_root / "raw_data" / "leanworkbook_raw.parquet"
    verified_dir = dataset_root / "verified_data"
    frozen_path = verified_dir / "leanworkbook_verified_success_1.jsonl"
    local_results_path = verified_dir / "leanworkbook_verification_results_2.jsonl"
    success_path = verified_dir / "leanworkbook_verified_success_2.jsonl"
    fail_path = verified_dir / "leanworkbook_verified_fail_2.jsonl"
    report_path = verified_dir / "leanworkbook_verification_report_2.json"

    raw_rows = pq.read_table(raw_path).to_pylist()
    reconstructed, reconstruction_report = reconstruct_lean_workbook_records(
        raw_rows,
        source_file=raw_path,
        source_commit=source_commit,
    )
    reconstructed_rows = [row.model_dump(mode="json") for row in reconstructed]
    frozen_rows = read_jsonl(frozen_path)
    frozen_ids = {
        _first_text(row, ("record_id", "id", "statement_id")) for row in frozen_rows
    }
    all_ids = {_first_text(row, ("record_id", "id")) for row in reconstructed_rows}
    if len(all_ids) != len(reconstructed_rows):
        raise RuntimeError("LeanWorkbook reconstruction contains duplicate record IDs")
    missing_frozen = frozen_ids - all_ids
    if missing_frozen:
        raise RuntimeError(
            f"frozen WB rows are absent from raw data: {sorted(missing_frozen)[:5]}"
        )

    identity = environment_identity(lean_project, ("Mathlib",))
    result_index = _load_result_index([local_results_path, *prior_results])
    pending: list[VerificationTask] = []
    tasks_by_id: dict[str, VerificationTask] = {}
    records_by_id: dict[str, dict[str, Any]] = {}
    results_by_id: dict[str, dict[str, Any]] = {}
    assembly_failures: dict[str, dict[str, Any]] = {}

    # The historical full-WB audit numbered only its compilable ``proved``
    # candidates.  Recreate that index exactly so its namespace-bearing
    # assembled-source hashes remain reusable.  Rows that the old pipeline
    # skipped keep their raw reconstruction index for the new compilation.
    proved_problem_index = 0
    for raw_problem_index, row in enumerate(reconstructed_rows):
        record_id = _first_text(row, ("record_id", "id"))
        raw_status = str((row.get("metadata") or {}).get("raw_status") or "")
        if raw_status == "proved":
            problem_index = proved_problem_index
            proved_problem_index += 1
        else:
            problem_index = raw_problem_index
        if record_id in frozen_ids:
            continue
        records_by_id[record_id] = row
        try:
            task = _verification_task(
                row,
                problem_index=problem_index,
                environment_hash=str(identity["environment_hash"]),
            )
        except Exception as error:
            assembly_failures[record_id] = {
                "record_id": record_id,
                "problem_id": record_id,
                "success": False,
                "status": "assembly_error",
                "diagnostics": f"{type(error).__name__}: {error}",
                "compile_errors": [str(error)],
                "environment_hash": identity["environment_hash"],
                "assembler_version": ASSEMBLER_VERSION,
                "normalization_version": NORMALIZATION_VERSION,
                "assembled_source_hash": None,
            }
            continue
        tasks_by_id[record_id] = task
        compatible = next(
            (
                result
                for result in result_index.get(record_id, ())
                if _compatible_result(
                    result,
                    task,
                    environment_hash=str(identity["environment_hash"]),
                )
            ),
            None,
        )
        if compatible is not None:
            results_by_id[record_id] = compatible
        else:
            pending.append(task)

    initially_missing = len(pending)
    if max_new_compilations is not None:
        pending = pending[:max_new_compilations]
    if reports_only and pending:
        raise RuntimeError(f"{len(pending)} compatible Pantograph results are missing")

    def persist(result: dict[str, Any]) -> None:
        append_jsonl(local_results_path, result)
        record_id = str(result.get("record_id") or result.get("problem_id") or "")
        results_by_id[record_id] = result
        completed = len(results_by_id)
        if completed % 100 == 0:
            print(
                json.dumps(
                    {
                        "phase": "leanworkbook_remaining_verification",
                        "results_available": completed,
                        "new_results_required": initially_missing,
                    }
                ),
                flush=True,
            )

    runtime: dict[str, Any] = {}
    if pending:
        pool = VerificationPool(
            VerificationPoolConfig(
                lean_project_path=str(lean_project),
                imports=("Mathlib",),
                timeout=timeout,
                warmup_timeout=3600,
                num_workers=workers,
                queue_maxsize=max(8, workers * 4),
                max_worker_restarts=3,
                max_task_retries=1,
                shutdown_timeout=15,
                task_spool_dir=str(
                    verified_dir / "runtime" / "leanworkbook_task_spool"
                ),
            )
        )
        with pool:
            for start in range(0, len(pending), batch_size):
                run = pool.run_batch(
                    pending[start : start + batch_size],
                    on_result=persist,
                )
                runtime = run.runtime_stats
                if run.fatal_errors:
                    raise RuntimeError(f"Pantograph pool failed: {run.fatal_errors}")

    unresolved = [
        task.problem_id
        for task in pending
        if task.problem_id not in results_by_id
    ]
    if unresolved:
        raise RuntimeError(f"Pantograph results missing after run: {unresolved[:5]}")

    output_success: list[dict[str, Any]] = []
    output_fail: list[dict[str, Any]] = []
    missing_due_to_limit: list[str] = []
    failure_statuses: Counter[str] = Counter()
    for row in reconstructed_rows:
        record_id = _first_text(row, ("record_id", "id"))
        if record_id in frozen_ids:
            continue
        result = results_by_id.get(record_id) or assembly_failures.get(record_id)
        if result is None:
            missing_due_to_limit.append(record_id)
            continue
        enriched = {
            **row,
            "proof": _proof_from_record(row),
            "lean_version": identity["lean_version"],
            "mathlib_commit": identity["mathlib_commit"],
            "environment_hash": identity["environment_hash"],
            "assembled_source_hash": result.get("assembled_source_hash"),
            "verification_status": result.get("status"),
            "verification_error_type": result.get("rejected_reason"),
            "verification_backend": result.get("verifier_backend") or "pantograph",
        }
        if result.get("success"):
            output_success.append(
                add_dataset_contract(
                    enriched,
                    source=LEANWORKBOOK_SOURCE,
                    status=SUCCESS,
                )
            )
        else:
            failure_statuses[str(result.get("status") or "unknown")] += 1
            output_fail.append(
                add_dataset_contract(
                    enriched,
                    source=LEANWORKBOOK_SOURCE,
                    status=FAIL,
                    error_message=_result_error(result),
                )
            )

    write_jsonl(success_path, output_success)
    write_jsonl(fail_path, output_fail)
    complete = not missing_due_to_limit and (
        len(frozen_rows) + len(output_success) + len(output_fail)
        == len(reconstructed_rows)
    )
    report = {
        "source": LEANWORKBOOK_SOURCE,
        "source_commit": source_commit,
        "environment": identity,
        "reconstruction": reconstruction_report,
        "raw_trajectory_rows": len(raw_rows),
        "unique_theorem_records": len(reconstructed_rows),
        "frozen_success_1": len(frozen_rows),
        "remaining_records": len(records_by_id),
        "compatible_prior_results_reused": len(records_by_id) - initially_missing,
        "new_compilations_required": initially_missing,
        "new_compilations_requested_this_run": len(pending),
        "success_2": len(output_success),
        "fail_2": len(output_fail),
        "failure_statuses": dict(failure_statuses),
        "missing_due_to_limit_count": len(missing_due_to_limit),
        "missing_due_to_limit_sample": missing_due_to_limit[:20],
        "complete_original_theorem_coverage": complete,
        "outputs": {
            "success_2": {
                "path": str(success_path.resolve()),
                "sha256": sha256_file(success_path),
            },
            "fail_2": {
                "path": str(fail_path.resolve()),
                "sha256": sha256_file(fail_path),
            },
            "results_cache": str(local_results_path.resolve()),
        },
        "runtime": runtime,
    }
    _write_json(report_path, report)
    dataset_manifest_path = dataset_root / "dataset_manifest.json"
    dataset_manifest = (
        json.loads(dataset_manifest_path.read_text(encoding="utf-8"))
        if dataset_manifest_path.exists()
        else {"schema_version": HASH_SCHEMA_VERSION}
    )
    dataset_manifest["leanworkbook_full_verification"] = {
        "raw_trajectory_rows": len(raw_rows),
        "unique_theorem_records": len(reconstructed_rows),
        "success_1": len(frozen_rows),
        "success_2": len(output_success),
        "fail_2": len(output_fail),
        "complete_original_theorem_coverage": complete,
        "environment_hash": identity["environment_hash"],
        "report_path": str(report_path.resolve()),
        "files": {
            "success_1": {
                "path": str(frozen_path.resolve()),
                "sha256": sha256_file(frozen_path),
            },
            "success_2": report["outputs"]["success_2"],
            "fail_2": report["outputs"]["fail_2"],
        },
    }
    _write_json(dataset_manifest_path, dataset_manifest)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return report


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(__file__).resolve().parent,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    materialize = subparsers.add_parser("materialize")
    materialize.add_argument("--leanworkbook-raw", type=Path, required=True)
    materialize.add_argument("--leandojo-raw", type=Path, required=True)
    materialize.add_argument("--leanworkbook-verified", type=Path, required=True)
    materialize.add_argument("--leandojo-verified", type=Path, required=True)

    subparsers.add_parser("audit")

    verify = subparsers.add_parser("verify-remaining-leanworkbook")
    verify.add_argument("--lean-project", type=Path, required=True)
    verify.add_argument("--source-commit", default=DEFAULT_SOURCE_COMMIT)
    verify.add_argument("--prior-results", type=Path, action="append", default=[])
    verify.add_argument("--workers", type=int, default=1)
    verify.add_argument("--timeout", type=int, default=30)
    verify.add_argument("--batch-size", type=int, default=128)
    verify.add_argument("--max-new-compilations", type=int)
    verify.add_argument("--reports-only", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "materialize":
        report = materialize_frozen_datasets(
            dataset_root=args.dataset_root,
            leanworkbook_raw=args.leanworkbook_raw,
            leandojo_raw=args.leandojo_raw,
            leanworkbook_verified=args.leanworkbook_verified,
            leandojo_verified=args.leandojo_verified,
        )
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    if args.command == "audit":
        audit_materialized_datasets(dataset_root=args.dataset_root)
        return
    verify_remaining_leanworkbook(
        dataset_root=args.dataset_root,
        lean_project=args.lean_project,
        source_commit=args.source_commit,
        prior_results=args.prior_results,
        workers=args.workers,
        timeout=args.timeout,
        batch_size=args.batch_size,
        max_new_compilations=args.max_new_compilations,
        reports_only=args.reports_only,
    )


if __name__ == "__main__":
    main()
