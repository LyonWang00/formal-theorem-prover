"""Resumable full-pool Pantograph verification for LeanDojo-v2 candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Iterable

from lean_prover.lean_training.verification.cache import VerificationCache
try:
    from scripts.verify_leandojo_v2_current_sample import (
        FAILURE_CATEGORIES,
        _cache_key,
        verify_sample,
    )
except ModuleNotFoundError:
    from verify_leandojo_v2_current_sample import (
        FAILURE_CATEGORIES,
        _cache_key,
        verify_sample,
    )


CACHE_NAMESPACE = "leandojo_v2_current_mathlib_5e932f97_full5366_v1"
EXPECTED_COUNT = 5366


def reclassify_known_failures(row: dict[str, Any]) -> dict[str, Any]:
    """Refine stored diagnostics without changing or rerunning verification."""

    if row.get("compile_success") or row.get("timed_out"):
        return row
    if row.get("error_category") not in (None, "", "unknown"):
        return row
    text = "\n".join(
        str(row.get(key) or "")
        for key in ("error_message", "stderr", "stdout")
    ).lower()
    if any(
        marker in text
        for marker in (
            "invalid name after `end`",
            "missing name after `end`",
            "expected the current scope name",
        )
    ):
        return {**row, "error_category": "source_reconstruction_error"}
    return row


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8-sig") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def partition_rows(
    rows: list[dict[str, Any]], max_batch_size: int
) -> list[list[dict[str, Any]]]:
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(str(value) for value in row.get("imports") or ())].append(row)
    batches: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for _, group in sorted(groups.items()):
        if len(group) > max_batch_size:
            raise RuntimeError(
                f"one import group has {len(group)} rows, exceeding "
                f"batch size {max_batch_size}"
            )
        if current and len(current) + len(group) > max_batch_size:
            batches.append(current)
            current = []
        current.extend(group)
    if current:
        batches.append(current)
    return batches


def prepare(
    *, candidate: Path, output_root: Path, max_batch_size: int
) -> dict[str, Any]:
    rows = list(iter_jsonl(candidate))
    if len(rows) != EXPECTED_COUNT:
        raise RuntimeError(f"expected {EXPECTED_COUNT} rows, found {len(rows)}")
    ids = [str(row["id"]) for row in rows]
    if len(set(ids)) != len(ids):
        raise RuntimeError("candidate input contains duplicate record IDs")
    batches = partition_rows(rows, max_batch_size)
    manifest_dir = output_root / "verification/batches/manifests"
    plan_rows: list[dict[str, Any]] = []
    accounted: list[str] = []
    for index, batch in enumerate(batches):
        manifest = manifest_dir / f"batch_{index:03d}.jsonl"
        write_jsonl(manifest, batch)
        batch_ids = [str(row["id"]) for row in batch]
        accounted.extend(batch_ids)
        plan_rows.append(
            {
                "batch_index": index,
                "manifest": str(manifest.resolve()),
                "row_count": len(batch),
                "import_group_count": len(
                    {
                        tuple(
                            str(value) for value in row.get("imports") or ()
                        )
                        for row in batch
                    }
                ),
                "first_id": batch_ids[0],
                "last_id": batch_ids[-1],
            }
        )
    if len(accounted) != EXPECTED_COUNT or set(accounted) != set(ids):
        raise RuntimeError("batch plan does not account for every candidate exactly once")
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "candidate_path": str(candidate.resolve()),
        "candidate_count": len(rows),
        "cache_namespace": CACHE_NAMESPACE,
        "max_batch_size": max_batch_size,
        "batch_count": len(batches),
        "batches": plan_rows,
        "all_candidates_accounted_once": True,
    }
    write_json(output_root / "audit/batch_plan.json", payload)
    return payload


def run_batch(
    *,
    output_root: Path,
    lean_project: Path,
    batch_index: int,
    workers: int,
    timeout: int,
) -> dict[str, Any]:
    plan = json.loads(
        (output_root / "audit/batch_plan.json").read_text(encoding="utf-8")
    )
    batch = plan["batches"][batch_index]
    manifest = Path(batch["manifest"])
    expected_size = int(batch["row_count"])
    batch_root = output_root / f"verification/batches/batch_{batch_index:03d}"
    status_path = batch_root / "batch_status.json"
    results_path = batch_root / "verification/results.jsonl"
    if status_path.exists() and results_path.exists():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        if (
            status.get("complete")
            and sum(1 for _ in iter_jsonl(results_path)) == expected_size
        ):
            return {**status, "resumed_without_reverification": True}

    started_at = datetime.now(UTC).isoformat()
    summary = verify_sample(
        manifest=manifest,
        lean_project=lean_project,
        output_root=batch_root,
        workers=workers,
        timeout=timeout,
        expected_size=expected_size,
        cache_repeat_count=0,
        cache_namespace=CACHE_NAMESPACE,
    )
    result_count = sum(1 for _ in iter_jsonl(results_path))
    if result_count != expected_size:
        raise RuntimeError(
            f"batch {batch_index} returned {result_count}/{expected_size}"
        )
    status = {
        "batch_index": batch_index,
        "started_at": started_at,
        "finished_at": datetime.now(UTC).isoformat(),
        "complete": True,
        "expected_count": expected_size,
        "result_count": result_count,
        "fidelity_success": int(summary["fidelity_success"]),
        "fidelity_success_ratio": float(summary["fidelity_success_ratio"]),
        "timeout_count": int(
            summary["failure_taxonomy"].get("timeout_count") or 0
        ),
        "compile_failure_count": (
            expected_size
            - int(summary["fidelity_success"])
            - int(summary["failure_taxonomy"].get("timeout_count") or 0)
        ),
        "first_run_cache_hits": int(
            summary["cache"].get("formal_run_cache_hit_count") or 0
        ),
        "cache_namespace": CACHE_NAMESPACE,
        "worker_restart_count": int(summary.get("worker_restart_count") or 0),
        "quality_gate_95_percent": (
            float(summary["fidelity_success_ratio"]) >= 0.95
        ),
        "resumed_without_reverification": False,
    }
    write_json(status_path, status)
    return status


def cache_repeat_audit(
    *, output_root: Path, batch_plan: dict[str, Any], repeat_count: int
) -> dict[str, Any]:
    selected: list[tuple[dict[str, Any], dict[str, Any], Path]] = []
    for batch in batch_plan["batches"]:
        batch_index = int(batch["batch_index"])
        batch_root = (
            output_root / f"verification/batches/batch_{batch_index:03d}"
        )
        rows = list(iter_jsonl(Path(batch["manifest"])))
        results = {
            str(row["sample_id"]): row
            for row in iter_jsonl(batch_root / "verification/results.jsonl")
        }
        cache_path = (
            batch_root
            / "verification/cache"
            / f"{CACHE_NAMESPACE}.sqlite"
        )
        for row in rows:
            selected.append((row, results[str(row["id"])], cache_path))
            if len(selected) == repeat_count:
                break
        if len(selected) == repeat_count:
            break
    hits = 0
    mismatches = 0
    audited: list[dict[str, Any]] = []
    caches: dict[Path, VerificationCache] = {}
    try:
        for row, result, cache_path in selected:
            cache = caches.setdefault(cache_path, VerificationCache(cache_path))
            cached = cache.get(
                _cache_key(row),
                environment_hash=str(row["environment_hash"]),
                assembler_version=str(
                    row.get("metadata", {}).get("adapter_version") or ""
                ),
                normalization_version="source-faithful-v1",
            )
            hit = cached is not None
            hits += int(hit)
            mismatch = bool(
                hit
                and bool(cached["compile_success"])
                != bool(result["compile_success"])
            )
            mismatches += int(mismatch)
            audited.append(
                {
                    "sample_id": row["id"],
                    "cache_hit": hit,
                    "result_match": hit and not mismatch,
                    "cache_path": str(cache_path),
                }
            )
    finally:
        for cache in caches.values():
            cache.close()
    payload = {
        "repeat_count": len(selected),
        "cache_hits": hits,
        "result_mismatches": mismatches,
        "all_identical_requests_hit": hits == len(selected),
        "all_results_match": mismatches == 0,
        "records": audited,
    }
    write_json(output_root / "audit/cache_repeat50.json", payload)
    return payload


def finalize(*, output_root: Path, repeat_count: int) -> dict[str, Any]:
    plan = json.loads(
        (output_root / "audit/batch_plan.json").read_text(encoding="utf-8")
    )
    all_results: list[dict[str, Any]] = []
    broad_results: list[dict[str, Any]] = []
    statuses: list[dict[str, Any]] = []
    candidates_by_id: dict[str, dict[str, Any]] = {}
    for batch in plan["batches"]:
        index = int(batch["batch_index"])
        manifest_path = Path(batch["manifest"])
        candidates_by_id.update(
            {str(row["id"]): row for row in iter_jsonl(manifest_path)}
        )
        batch_root = output_root / f"verification/batches/batch_{index:03d}"
        status_path = batch_root / "batch_status.json"
        results_path = batch_root / "verification/results.jsonl"
        if not status_path.exists() or not results_path.exists():
            raise RuntimeError(f"batch {index} is incomplete")
        status = json.loads(status_path.read_text(encoding="utf-8"))
        rows = [
            reclassify_known_failures(row)
            for row in iter_jsonl(results_path)
        ]
        if not status.get("complete") or len(rows) != int(batch["row_count"]):
            raise RuntimeError(f"batch {index} is incomplete or inconsistent")
        statuses.append(status)
        all_results.extend({**row, "batch_index": index} for row in rows)
        broad_path = (
            batch_root / "verification/broad_import_diagnostic.jsonl"
        )
        if broad_path.exists():
            broad_results.extend(
                {**row, "batch_index": index}
                for row in iter_jsonl(broad_path)
            )
    ids = [str(row["sample_id"]) for row in all_results]
    if len(all_results) != EXPECTED_COUNT or len(set(ids)) != EXPECTED_COUNT:
        raise RuntimeError(
            f"full results do not account for {EXPECTED_COUNT} unique rows"
        )

    verified = [row for row in all_results if row["compile_success"]]
    timeouts = [row for row in all_results if row.get("timed_out")]
    failures = [
        row
        for row in all_results
        if not row["compile_success"] and not row.get("timed_out")
    ]
    rejected: list[dict[str, Any]] = []
    verification = output_root / "verification"
    write_jsonl(verification / "all_results.jsonl", all_results)
    write_jsonl(verification / "verified_all.jsonl", verified)
    write_jsonl(verification / "verified_default_timeout.jsonl", verified)
    write_jsonl(verification / "slow_verified.jsonl", [])
    write_jsonl(verification / "timeout_quarantine.jsonl", timeouts)
    write_jsonl(verification / "compile_failure_quarantine.jsonl", failures)
    write_jsonl(verification / "rejected.jsonl", rejected)
    write_jsonl(
        verification / "broad_import_diagnostic.jsonl", broad_results
    )
    taxonomy = Counter(
        str(row.get("error_category") or "unknown")
        for row in all_results
        if not row["compile_success"]
    )
    taxonomy_payload = {
        "categories": {
            category: taxonomy.get(category, 0)
            for category in FAILURE_CATEGORIES
        },
        "total_failures": len(all_results) - len(verified),
        "timeout_count": len(timeouts),
    }
    write_json(verification / "failure_taxonomy.json", taxonomy_payload)
    failed_results = [row for row in all_results if not row["compile_success"]]
    failures_by_source = Counter(
        str(row.get("source_file") or "unknown") for row in failed_results
    )
    failures_by_proof_style = Counter(
        str(
            candidates_by_id.get(str(row["sample_id"]), {}).get("proof_style")
            or "unknown"
        )
        for row in failed_results
    )
    cache = cache_repeat_audit(
        output_root=output_root,
        batch_plan=plan,
        repeat_count=repeat_count,
    )
    success_ratio = len(verified) / EXPECTED_COUNT
    payload = {
        "created_at": datetime.now(UTC).isoformat(),
        "total_candidates": EXPECTED_COUNT,
        "total_results": len(all_results),
        "unique_result_ids": len(set(ids)),
        "verified_default_timeout": len(verified),
        "success_ratio": success_ratio,
        "slow_verified": 0,
        "timeouts": len(timeouts),
        "compile_failures": len(failures),
        "failure_concentration": {
            "by_source_file": dict(failures_by_source.most_common()),
            "by_proof_style": dict(failures_by_proof_style.most_common()),
        },
        "rejected": len(rejected),
        "broad_import_diagnostic_total": len(broad_results),
        "broad_import_diagnostic_success": sum(
            bool(row["compile_success"]) for row in broad_results
        ),
        "batch_statuses": statuses,
        "all_candidates_accounted_once": True,
        "all_first_run_cache_hits_zero": all(
            int(status.get("first_run_cache_hits") or 0) == 0
            for status in statuses
        ),
        "cache_repeat": cache,
        "worker_restart_count": sum(
            int(status.get("worker_restart_count") or 0)
            for status in statuses
        ),
        "quality_gate_95_percent": success_ratio >= 0.95,
        "training_started": False,
    }
    write_json(verification / "verification_summary.json", payload)
    report = f"""# LeanDojo-v2 full-pool Pantograph verification

- Total candidates/results: `{EXPECTED_COUNT}` / `{len(all_results)}`
- Default-timeout verified: `{len(verified)}`
- Fidelity success ratio: `{success_ratio:.2%}`
- Timeout quarantine: `{len(timeouts)}`
- Compile-failure quarantine: `{len(failures)}`
- Failure source files: `{json.dumps(dict(failures_by_source.most_common()), ensure_ascii=False)}`
- Failure proof styles: `{json.dumps(dict(failures_by_proof_style.most_common()), ensure_ascii=False)}`
- Rejected before verification: `{len(rejected)}`
- First-run cache hits: `0`
- Repeat-cache hits: `{cache["cache_hits"]}/{cache["repeat_count"]}`
- Repeat-cache mismatches: `{cache["result_mismatches"]}`
- Worker restarts: `{payload["worker_restart_count"]}`
- 95% quality gate: `{"PASS" if payload["quality_gate_95_percent"] else "FAIL"}`
- Training started: `False`
"""
    (verification / "verification_report.md").write_text(
        report, encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "command", choices=("prepare", "run-batch", "finalize")
    )
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--lean-project", type=Path)
    parser.add_argument("--max-batch-size", type=int, default=900)
    parser.add_argument("--batch-index", type=int)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--timeout", type=int, default=30)
    parser.add_argument("--repeat-count", type=int, default=50)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    if args.command == "prepare":
        if args.candidate is None:
            parser.error("--candidate is required for prepare")
        print(
            json.dumps(
                prepare(
                    candidate=args.candidate.resolve(),
                    output_root=output_root,
                    max_batch_size=args.max_batch_size,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    elif args.command == "run-batch":
        if args.lean_project is None or args.batch_index is None:
            parser.error(
                "--lean-project and --batch-index are required for run-batch"
            )
        print(
            json.dumps(
                run_batch(
                    output_root=output_root,
                    lean_project=args.lean_project.resolve(),
                    batch_index=args.batch_index,
                    workers=args.workers,
                    timeout=args.timeout,
                ),
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        finalize(output_root=output_root, repeat_count=args.repeat_count)


if __name__ == "__main__":
    main()
