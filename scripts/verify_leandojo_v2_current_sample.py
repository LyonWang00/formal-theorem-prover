"""Verify the fixed current-mathlib LeanDojo-v2 sample with Pantograph."""

from __future__ import annotations

import argparse
import json
import math
import re
import statistics
import time
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from lean_prover.lean_training.data.leandojo_v2_current import (
    ADAPTER_VERSION,
    CURRENT_ENVIRONMENT_HASH,
    CURRENT_MATHLIB_COMMIT,
    LEANDOJO_V2_COMMIT,
    sha256_text,
)
from lean_prover.lean_training.verification.cache import (
    VerificationCache,
    make_context_cache_key,
)
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)
from lean_prover.lean_training.verification.schema import VerificationTask


CACHE_NAMESPACE = "leandojo_v2_current_mathlib_5e932f97_sample500_v3"
VERIFIER_VERSION = "Pantograph-0.3.15"
NORMALIZATION_VERSION = "source-faithful-v1"
CACHE_VERIFIER_IDENTITY = (
    f"{VERIFIER_VERSION};mathlib={CURRENT_MATHLIB_COMMIT};"
    f"leandojo_v2={LEANDOJO_V2_COMMIT}"
)
FAILURE_CATEGORIES = (
    "source_reconstruction_error",
    "missing_same_file_dependency",
    "missing_import",
    "missing_namespace",
    "missing_scope",
    "missing_local_declaration",
    "unknown_identifier",
    "syntax_error",
    "notation_error",
    "elaboration_error",
    "application_type_mismatch",
    "typeclass_error",
    "tactic_error",
    "unsolved_goals",
    "environment_error",
    "timeout",
    "worker_error",
    "cache_error",
    "unknown",
)


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return float(
        ordered[lower] * (upper - position)
        + ordered[upper] * (position - lower)
    )


def _time_distribution(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values) if values else 0.0,
        "p50": _percentile(values, 0.50),
        "p90": _percentile(values, 0.90),
        "max": max(values, default=0.0),
    }


def _rss_kib() -> int | None:
    status = Path("/proc/self/status")
    if not status.exists():
        return None
    match = re.search(
        r"^VmRSS:\s*(\d+)\s+kB", status.read_text(encoding="utf-8"), re.M
    )
    return int(match.group(1)) if match else None


def classify_failure(result: dict[str, Any]) -> str:
    if result.get("timed_out"):
        return "timeout"
    text = "\n".join(
        [
            str(result.get("diagnostics") or ""),
            *[str(value) for value in result.get("compile_errors") or []],
        ]
    ).lower()
    rules = (
        (
            "source_reconstruction_error",
            (
                "invalid name after `end`",
                "missing name after `end`",
                "expected the current scope name",
            ),
        ),
        ("worker_error", ("worker ", "server not running", "broken pipe")),
        ("environment_error", ("pantograph", "environment", "no such file")),
        ("missing_import", ("unknown module", "invalid import")),
        ("missing_namespace", ("unknown namespace",)),
        ("missing_scope", ("unknown scope",)),
        ("unknown_identifier", ("unknown identifier", "unknown constant")),
        ("notation_error", ("unexpected token", "unknown parser", "notation")),
        ("syntax_error", ("parser", "syntax error", "unexpected end of input")),
        ("application_type_mismatch", ("application type mismatch",)),
        (
            "typeclass_error",
            ("failed to synthesize", "typeclass instance problem"),
        ),
        ("unsolved_goals", ("unsolved goals", "no goals to be solved")),
        (
            "tactic_error",
            ("tactic", "did not find instance", "simp made no progress"),
        ),
        ("elaboration_error", ("type mismatch", "application mismatch")),
    )
    for category, markers in rules:
        if any(marker in text for marker in markers):
            return category
    return "unknown"


def _remove_import_commands(source: str) -> str:
    """Remove imports already loaded by the Pantograph server."""

    return "".join(
        line
        for line in source.splitlines(keepends=True)
        if not re.match(
            r"^\s*(?:(?:public|private|protected|meta)\s+)*import\s+\S+",
            line,
        )
        and not re.match(r"^\s*module\s*$", line)
    ).lstrip("\n")


def _cache_key(row: dict[str, Any]) -> str:
    return make_context_cache_key(
        statement_hash=sha256_text(str(row["statement"])),
        proof_hash=sha256_text(str(row["proof"])),
        imports_hash=sha256_text(
            json.dumps(row.get("imports") or [], sort_keys=True)
        ),
        namespace_hash=sha256_text(
            json.dumps(row.get("namespace_stack") or [], sort_keys=True)
        ),
        scope_hash=sha256_text(
            json.dumps(row.get("open_scopes") or [], sort_keys=True)
        ),
        variable_context_hash=sha256_text(
            json.dumps(row.get("variables") or [], sort_keys=True)
        ),
        context_hash=sha256_text(
            json.dumps(
                {
                    "open_namespaces": row.get("open_namespaces") or [],
                    "sections": row.get("section_context") or [],
                    "local_instances": row.get("local_instances") or [],
                    "local_notations": row.get("local_notations") or [],
                    "local_attributes": row.get("local_attributes") or [],
                },
                sort_keys=True,
            )
        ),
        assembled_source_hash=str(row["assembled_source_hash"]),
        environment_hash=str(row["environment_hash"]),
        verifier_version=CACHE_VERIFIER_IDENTITY,
        assembler_version=ADAPTER_VERSION,
        normalization_version=NORMALIZATION_VERSION,
    )


def _tasks(
    rows: list[dict[str, Any]],
    *,
    assembled_dir: Path,
    broad_import: bool,
    cache_namespace: str = CACHE_NAMESPACE,
) -> list[VerificationTask]:
    tasks: list[VerificationTask] = []
    assembled_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows):
        fidelity_source = str(row["metadata"]["assembled_source"])
        mode = "broad_import_diagnostic" if broad_import else "fidelity"
        if broad_import:
            complete_source = "import Mathlib\n\n" + _remove_import_commands(
                fidelity_source
            )
            task_imports = ("Mathlib",)
        else:
            complete_source = fidelity_source
            task_imports = tuple(str(value) for value in row.get("imports") or ())
        source_body = _remove_import_commands(complete_source)
        source_hash = sha256_text(complete_source)
        source_path = assembled_dir / f"{row['id']}.lean"
        source_path.write_text(complete_source, encoding="utf-8")
        task = VerificationTask(
            priority=index,
            problem_index=int(row["_verification_index"]),
            attempt_index=1 if broad_import else 0,
            problem_id=str(row["id"]),
            prompt="",
            generated_proof=str(row["proof"]),
            raw_completion=str(row["proof"]),
            lean_code=source_body,
            imports=task_imports,
            context_lines=(),
            payload={
                "generation_id": f"{mode}:{row['id']}",
                "sample_id": row["id"],
                "qualified_name": row["qualified_name"],
                "source_file": row["source_file"],
                "verification_mode": mode,
                "assembled_source_hash": source_hash,
                "pantograph_source_body_hash": sha256_text(source_body),
                "assembled_source_path": str(source_path),
                "environment_hash": CURRENT_ENVIRONMENT_HASH,
                "mathlib_commit": CURRENT_MATHLIB_COMMIT,
                "leandojo_v2_commit": LEANDOJO_V2_COMMIT,
                "assembler_version": ADAPTER_VERSION,
                "cache_namespace": cache_namespace,
                "cache_hit": False,
                "preassembled_source": True,
            },
            reject_forbidden=True,
        )
        tasks.append(task)
    return tasks


def _normalize_results(
    raw_results: list[dict[str, Any]],
    rows_by_id: dict[str, dict[str, Any]],
    *,
    cache_namespace: str = CACHE_NAMESPACE,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    for result in sorted(
        raw_results,
        key=lambda value: int(
            rows_by_id[
                str(value.get("sample_id") or value.get("problem_id"))
            ]["_verification_index"]
        ),
    ):
        sample_id = str(result.get("sample_id") or result.get("problem_id"))
        row = rows_by_id[sample_id]
        success = bool(result.get("success"))
        normalized.append(
            {
                "sample_id": sample_id,
                "qualified_name": row["qualified_name"],
                "source_file": row["source_file"],
                "compile_success": success,
                "cache_hit": False,
                "verification_mode": result["verification_mode"],
                "assembled_source_hash": result["assembled_source_hash"],
                "assembled_source_path": result["assembled_source_path"],
                "error_category": "" if success else classify_failure(result),
                "error_message": str(result.get("diagnostics") or ""),
                "stdout": "\n".join(result.get("compile_messages") or []),
                "stderr": "\n".join(result.get("compile_errors") or []),
                "elapsed_seconds": float(
                    result.get("verification_seconds") or 0.0
                ),
                "total_seconds": float(result.get("total_seconds") or 0.0),
                "worker_id": str(result.get("worker_id")),
                "environment_hash": CURRENT_ENVIRONMENT_HASH,
                "timed_out": bool(result.get("timed_out")),
                "verifier_backend": result.get("verifier_backend"),
                "pantograph_version": "0.3.15",
                "mathlib_commit": CURRENT_MATHLIB_COMMIT,
                "leandojo_v2_commit": LEANDOJO_V2_COMMIT,
                "cache_namespace": cache_namespace,
            }
        )
    return normalized


def _run_pool(
    tasks: list[VerificationTask],
    *,
    lean_project: Path,
    output_root: Path,
    workers: int,
    timeout: int,
    spool_name: str,
    imports: tuple[str, ...],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config = VerificationPoolConfig(
        lean_project_path=str(lean_project.resolve()),
        imports=imports or ("Init",),
        timeout=timeout,
        warmup_timeout=max(120, timeout),
        num_workers=workers,
        queue_maxsize=max(32, workers * 16),
        max_task_retries=1,
        max_worker_restarts=3,
        task_spool_dir=str(output_root / "runtime" / spool_name),
        save_full_source_on_failure_only=True,
    )
    with VerificationPool(config) as pool:
        run = pool.run_batch(tasks)
    if run.fatal_errors:
        raise RuntimeError("; ".join(run.fatal_errors))
    runtime = {
        **run.runtime_stats,
        "warmup_reports": run.warmup_reports,
        "recovered_worker_failures": run.recovered_worker_failures,
        "fatal_errors": run.fatal_errors,
    }
    return run.results, runtime


def _run_grouped_pools(
    rows: list[dict[str, Any]],
    *,
    lean_project: Path,
    output_root: Path,
    workers: int,
    timeout: int,
    assembled_dir: Path,
    broad_import: bool,
    cache_namespace: str = CACHE_NAMESPACE,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    for row in rows:
        imports = (
            ("Mathlib",)
            if broad_import
            else tuple(str(value) for value in row.get("imports") or ())
        )
        grouped.setdefault(imports, []).append(row)
    all_results: list[dict[str, Any]] = []
    group_reports: list[dict[str, Any]] = []
    for group_index, (imports, group_rows) in enumerate(sorted(grouped.items())):
        tasks = _tasks(
            group_rows,
            assembled_dir=assembled_dir,
            broad_import=broad_import,
            cache_namespace=cache_namespace,
        )
        results, runtime = _run_pool(
            tasks,
            lean_project=lean_project,
            output_root=output_root,
            workers=workers,
            timeout=timeout,
            spool_name=(
                f"{'broad' if broad_import else 'fidelity'}_"
                f"group_{group_index:03d}_tasks"
            ),
            imports=imports,
        )
        all_results.extend(results)
        group_reports.append(
            {
                "group_index": group_index,
                "imports": list(imports),
                "task_count": len(group_rows),
                "runtime": runtime,
            }
        )
    return all_results, {
        "group_count": len(group_reports),
        "groups": group_reports,
    }


def _cache_consistency_check(
    *,
    rows: list[dict[str, Any]],
    results: list[dict[str, Any]],
    cache_path: Path,
    repeat_count: int,
    cache_namespace: str = CACHE_NAMESPACE,
) -> dict[str, Any]:
    result_by_id = {row["sample_id"]: row for row in results}
    keys: dict[str, str] = {}
    with VerificationCache(cache_path) as cache:
        for row in rows:
            key = _cache_key(row)
            keys[str(row["id"])] = key
            result = result_by_id[str(row["id"])]
            cache.put(
                cache_key=key,
                proof_hash=sha256_text(str(row["proof"])),
                statement_hash=sha256_text(str(row["statement"])),
                environment_hash=CURRENT_ENVIRONMENT_HASH,
                assembler_version=ADAPTER_VERSION,
                normalization_version=NORMALIZATION_VERSION,
                status="success" if result["compile_success"] else "failed",
                error_type=result["error_category"] or None,
                source_path=result["assembled_source_path"],
                verification=result,
            )
        repeated = rows[:repeat_count]
        hits = 0
        mismatches = 0
        for row in repeated:
            cached = cache.get(
                keys[str(row["id"])],
                environment_hash=CURRENT_ENVIRONMENT_HASH,
                assembler_version=ADAPTER_VERSION,
                normalization_version=NORMALIZATION_VERSION,
            )
            if cached is not None:
                hits += 1
                if cached["compile_success"] != result_by_id[str(row["id"])][
                    "compile_success"
                ]:
                    mismatches += 1
        if not repeated:
            return {
                "formal_run_cache_hit_count": 0,
                "repeat_count": 0,
                "repeat_cache_hits": 0,
                "repeat_result_mismatches": 0,
                "identical_request_hit": None,
                "source_change_miss": None,
                "environment_change_miss": None,
                "cache_entry_count": cache.count(),
                "cache_path": str(cache.path),
                "cache_namespace": cache_namespace,
            }
        probe = repeated[0]
        base_key = keys[str(probe["id"])]
        source_changed = dict(probe)
        source_changed["assembled_source_hash"] = sha256_text(
            str(probe["metadata"]["assembled_source"]) + "\n-- cache probe\n"
        )
        source_change_miss = (
            cache.get(
                _cache_key(source_changed),
                environment_hash=CURRENT_ENVIRONMENT_HASH,
                assembler_version=ADAPTER_VERSION,
                normalization_version=NORMALIZATION_VERSION,
            )
            is None
        )
        environment_change_miss = (
            cache.get(
                base_key,
                environment_hash=CURRENT_ENVIRONMENT_HASH + "-probe",
                assembler_version=ADAPTER_VERSION,
                normalization_version=NORMALIZATION_VERSION,
            )
            is None
        )
        return {
            "formal_run_cache_hit_count": 0,
            "repeat_count": repeat_count,
            "repeat_cache_hits": hits,
            "repeat_result_mismatches": mismatches,
            "identical_request_hit": hits == repeat_count,
            "source_change_miss": source_change_miss,
            "environment_change_miss": environment_change_miss,
            "cache_entry_count": cache.count(),
            "cache_path": str(cache.path),
            "cache_namespace": cache_namespace,
        }


def verify_sample(
    *,
    manifest: Path,
    lean_project: Path,
    output_root: Path,
    workers: int,
    timeout: int,
    expected_size: int,
    cache_repeat_count: int,
    limit: int | None = None,
    sample_ids: set[str] | None = None,
    cache_namespace: str = CACHE_NAMESPACE,
) -> dict[str, Any]:
    rows = list(_iter_jsonl(manifest))
    if sample_ids:
        rows = [row for row in rows if str(row["id"]) in sample_ids]
    if limit is not None:
        rows = rows[:limit]
    for index, row in enumerate(rows):
        row["_verification_index"] = index
    if len(rows) != expected_size:
        raise RuntimeError(f"expected {expected_size} rows, found {len(rows)}")
    if len({row["id"] for row in rows}) != expected_size:
        raise RuntimeError("sample contains duplicate ids")
    if any(row["repository_commit"] != CURRENT_MATHLIB_COMMIT for row in rows):
        raise RuntimeError("sample mathlib commit mismatch")
    rows_by_id = {str(row["id"]): row for row in rows}
    verification_dir = output_root / "verification"
    started_at = datetime.now(UTC).isoformat()
    coordinator_rss_before = _rss_kib()
    started = time.monotonic()

    raw_fidelity, fidelity_runtime = _run_grouped_pools(
        rows,
        lean_project=lean_project,
        output_root=output_root,
        workers=workers,
        timeout=timeout,
        assembled_dir=verification_dir / "assembled_sources" / "fidelity",
        broad_import=False,
        cache_namespace=cache_namespace,
    )
    fidelity = _normalize_results(
        raw_fidelity,
        rows_by_id,
        cache_namespace=cache_namespace,
    )
    if len(fidelity) != expected_size:
        raise RuntimeError(
            f"Pantograph returned {len(fidelity)} of {expected_size} results"
        )
    _write_jsonl(verification_dir / "results.jsonl", fidelity)
    successful = [row for row in fidelity if row["compile_success"]]
    failed = [row for row in fidelity if not row["compile_success"]]
    _write_jsonl(
        verification_dir / "verified_high_confidence.jsonl",
        successful,
    )
    _write_jsonl(verification_dir / "failed.jsonl", failed)

    broad: list[dict[str, Any]] = []
    broad_runtime: dict[str, Any] = {}
    if failed:
        failed_rows = [rows_by_id[row["sample_id"]] for row in failed]
        raw_broad, broad_runtime = _run_grouped_pools(
            failed_rows,
            lean_project=lean_project,
            output_root=output_root,
            workers=workers,
            timeout=timeout,
            assembled_dir=verification_dir
            / "assembled_sources"
            / "broad_import_diagnostic",
            broad_import=True,
            cache_namespace=cache_namespace,
        )
        broad = _normalize_results(
            raw_broad,
            rows_by_id,
            cache_namespace=cache_namespace,
        )
    _write_jsonl(verification_dir / "broad_import_diagnostic.jsonl", broad)

    cache_report = _cache_consistency_check(
        rows=rows,
        results=fidelity,
        cache_path=verification_dir / "cache" / f"{cache_namespace}.sqlite",
        repeat_count=min(cache_repeat_count, len(rows)),
        cache_namespace=cache_namespace,
    )
    taxonomy = Counter(
        row["error_category"] for row in failed if row["error_category"]
    )
    taxonomy_payload = {
        "categories": {
            category: taxonomy.get(category, 0)
            for category in FAILURE_CATEGORIES
        },
        "total_failures": len(failed),
        "timeout_count": sum(row["timed_out"] for row in failed),
    }
    _write_json(verification_dir / "failure_taxonomy.json", taxonomy_payload)
    elapsed_values = [row["elapsed_seconds"] for row in fidelity]
    elapsed_total = time.monotonic() - started
    coordinator_rss_after = _rss_kib()
    worker_restart_count = sum(
        int(worker.get("restart_count") or 0)
        for group in fidelity_runtime.get("groups", [])
        for worker in group.get("runtime", {}).get("workers", [])
    )
    runtime_payload = {
        "stage": "pantograph_sample500",
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "wall_seconds": round(elapsed_total, 4),
        "coordinator_rss_before_kib": coordinator_rss_before,
        "coordinator_rss_after_kib": coordinator_rss_after,
        "worker_count": workers,
        "timeout_seconds": timeout,
        "worker_restart_count": worker_restart_count,
        "fidelity": fidelity_runtime,
        "broad_import_diagnostic": broad_runtime,
    }
    runtime_dir = output_root / "runtime"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    with (runtime_dir / "verification_resources.jsonl").open(
        "a", encoding="utf-8"
    ) as handle:
        handle.write(json.dumps(runtime_payload, sort_keys=True) + "\n")

    report = {
        "total": expected_size,
        "fidelity_success": len(successful),
        "fidelity_success_ratio": len(successful) / expected_size,
        "broad_import_diagnostic_total": len(broad),
        "broad_import_diagnostic_success": sum(
            row["compile_success"] for row in broad
        ),
        "failure_taxonomy": taxonomy_payload,
        "cache": cache_report,
        "verification_time_seconds": _time_distribution(elapsed_values),
        "worker_restart_count": worker_restart_count,
        "runtime": runtime_payload,
        "training_started": False,
        "thresholds": {
            "minimum_70_percent": len(successful) / expected_size >= 0.70,
            "engineering_85_percent": len(successful) / expected_size >= 0.85,
            "excellent_90_percent": len(successful) / expected_size >= 0.90,
        },
    }
    _write_json(verification_dir / "verification_summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--lean-project", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--expected-size", type=int, default=500)
    parser.add_argument("--cache-repeat-count", type=int, default=25)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample-id", action="append", default=[])
    parser.add_argument("--cache-namespace", default=CACHE_NAMESPACE)
    args = parser.parse_args()
    summary = verify_sample(
        manifest=args.manifest,
        lean_project=args.lean_project,
        output_root=args.output_root,
        workers=args.workers,
        timeout=args.timeout,
        expected_size=args.expected_size,
        cache_repeat_count=args.cache_repeat_count,
        limit=args.limit,
        sample_ids=set(args.sample_id),
        cache_namespace=args.cache_namespace,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
