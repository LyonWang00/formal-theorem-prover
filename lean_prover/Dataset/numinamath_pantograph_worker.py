"""Single persistent Pantograph worker for staged NuminaMath repair batches."""

from __future__ import annotations

import argparse
import json
import os
import signal
import shutil
import threading
import time
import traceback
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

from lean_prover.Dataset.build_verified_datasets import FAIL, SUCCESS, _write_json, sha256_file, sha256_text
from lean_prover.Dataset.repair_numinamath_failures import (
    REPAIR_VERSION,
    append_jsonl,
    canonical_hash,
    complete_source,
    read_jsonl,
    start_verifier,
)
from lean_prover.lean_training.expert_iteration.utils import environment_identity


WORKER_SCHEMA = "numinamath_pantograph_worker_v1"


class _PantographHardTimeout(TimeoutError):
    pass


def _pantograph_process_pid(verifier: Any) -> int | None:
    """Return the current Pantograph REPL PID without depending on its public API."""
    backend = getattr(verifier, "backend", None)
    server = getattr(backend, "_server", None)
    process = getattr(server, "proc", None)
    pid = getattr(process, "pid", None)
    return int(pid) if isinstance(pid, int) and pid > 0 else None


def _kill_pantograph_repl(verifier: Any) -> None:
    """Force-release a timed-out REPL before constructing its replacement."""
    repl_pid = _pantograph_process_pid(verifier)
    if repl_pid is None:
        return
    try:
        os.kill(repl_pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def _safe_close(verifier: Any) -> None:
    """Close a verifier even when its independently watched REPL already exited."""
    try:
        verifier.close()
    except (OSError, ProcessLookupError):
        pass


def _check_with_hard_timeout(verifier: Any, source: str, *, timeout: int) -> Any:
    """Enforce signal and independent process guards around a Pantograph check."""
    started = time.monotonic()
    repl_pid = _pantograph_process_pid(verifier)
    watchdog_fired = threading.Event()

    def kill_stalled_repl() -> None:
        watchdog_fired.set()
        if repl_pid is None:
            return
        try:
            os.kill(repl_pid, signal.SIGKILL)
        except ProcessLookupError:
            pass

    watchdog = threading.Timer(timeout + 10, kill_stalled_repl)
    watchdog.daemon = True
    watchdog.start()
    if not hasattr(signal, "SIGALRM") or not hasattr(signal, "setitimer"):
        try:
            return verifier.check_source(source, timeout=timeout, reject_forbidden=True)
        except Exception:
            if not watchdog_fired.is_set():
                raise
        finally:
            watchdog.cancel()
        message = f"Pantograph REPL exceeded watchdog timeout of {timeout + 10} seconds"
        return SimpleNamespace(
            success=False,
            diagnostics=message,
            check_seconds=round(time.monotonic() - started, 4),
            timed_out=True,
            error_type="pantograph_process_watchdog_timeout",
            errors=(message,),
            warnings=(),
        )
    previous_handler = signal.getsignal(signal.SIGALRM)

    def raise_timeout(_signum: int, _frame: Any) -> None:
        raise _PantographHardTimeout(
            f"Pantograph check exceeded hard timeout of {timeout + 5} seconds"
        )

    signal.signal(signal.SIGALRM, raise_timeout)
    signal.setitimer(signal.ITIMER_REAL, timeout + 5)
    try:
        return verifier.check_source(source, timeout=timeout, reject_forbidden=True)
    except _PantographHardTimeout as error:
        message = str(error)
        return SimpleNamespace(
            success=False,
            diagnostics=message,
            check_seconds=round(time.monotonic() - started, 4),
            timed_out=True,
            error_type="pantograph_hard_timeout",
            errors=(message,),
            warnings=(),
        )
    except Exception:
        if not watchdog_fired.is_set():
            raise
        message = f"Pantograph REPL exceeded watchdog timeout of {timeout + 10} seconds"
        return SimpleNamespace(
            success=False,
            diagnostics=message,
            check_seconds=round(time.monotonic() - started, 4),
            timed_out=True,
            error_type="pantograph_process_watchdog_timeout",
            errors=(message,),
            warnings=(),
        )
    finally:
        watchdog.cancel()
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)


def _write_json_atomic(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(dict(value), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _source(candidate: Mapping[str, Any], proof: str) -> str:
    # PantographTheoremVerifier starts a session with Mathlib pre-imported.
    # Interactive `check_source` snippets therefore contain declarations only;
    # replaying import commands inside the session is rejected by Lean.
    return complete_source(str(candidate["source_body"]), proof).strip()


def _heartbeat(queue_dir: Path, *, status: str, **extra: Any) -> None:
    _write_json_atomic(queue_dir / "heartbeat.json", {
        "schema_version": WORKER_SCHEMA,
        "pid": os.getpid(),
        "status": status,
        "updated_unix": int(time.time()),
        **extra,
    })


def verify_job(
    *, job: Mapping[str, Any], verifier: Any, identity: Mapping[str, Any], queue_dir: Path,
) -> tuple[Any, int]:
    batch_dir = Path(str(job["batch_dir"])).resolve()
    timeout = int(job.get("timeout", 30))
    candidates = read_jsonl(batch_dir / "candidate_manifest.jsonl")
    if not candidates:
        raise RuntimeError(f"empty candidate manifest: {batch_dir}")
    result_path = batch_dir / "verification_results.jsonl"
    cached = {
        str(row["variant_hash"]): row
        for row in read_jsonl(result_path)
        if row.get("repair_version") == REPAIR_VERSION and row.get("variant_hash")
    }
    completed = 0
    solved = 0
    new_attempts = 0
    restarts = 0
    for candidate in candidates:
        chosen: Mapping[str, Any] | None = None
        for variant in candidate["variants"]:
            proof = str(variant["proof"])
            source = _source(candidate, proof)
            variant_hash = canonical_hash({
                "candidate_hash": candidate["candidate_hash"],
                "strategy": variant["strategy"],
                "proof": proof,
                "environment_hash": identity["environment_hash"],
            })
            result = cached.get(variant_hash)
            if result is None:
                checked = _check_with_hard_timeout(verifier, source, timeout=timeout)
                restart_count = 0
                if checked.error_type == "pantograph_error" or checked.timed_out:
                    if checked.timed_out:
                        _kill_pantograph_repl(verifier)
                    _safe_close(verifier)
                    verifier = start_verifier(argparse.Namespace(
                        lean_project=Path(str(job["lean_project"])), timeout=timeout
                    ))
                    restarts += 1
                    restart_count = 1
                if checked.error_type == "pantograph_error":
                    checked = _check_with_hard_timeout(verifier, source, timeout=timeout)
                result = {
                    "schema_version": "numinamath_repair_verification_result_v1",
                    "repair_version": REPAIR_VERSION,
                    "batch_id": candidate["batch_id"],
                    "record_id": candidate["record_id"],
                    "candidate_hash": candidate["candidate_hash"],
                    "variant_hash": variant_hash,
                    "strategy": variant["strategy"],
                    "proof": proof,
                    "success": checked.success,
                    "pantograph_verified": SUCCESS if checked.success else FAIL,
                    "diagnostics": checked.diagnostics,
                    "errors": list(checked.errors),
                    "warnings": list(checked.warnings),
                    "timed_out": checked.timed_out,
                    "error_type": checked.error_type,
                    "verification_seconds": checked.check_seconds,
                    "pantograph_restart_count": restart_count,
                    "assembled_source_hash": sha256_text(source),
                    "environment": dict(identity),
                    "worker_pid": os.getpid(),
                }
                append_jsonl(result_path, result)
                cached[variant_hash] = result
                new_attempts += 1
            chosen = result
            if result.get("success"):
                break
        if chosen is None:
            raise RuntimeError(f"candidate has no proof variants: {candidate['record_id']}")
        completed += 1
        solved += int(bool(chosen.get("success")))
        if completed % 10 == 0 or completed == len(candidates):
            _heartbeat(
                queue_dir,
                status="running",
                batch_id=str(candidate["batch_id"]),
                records_completed=completed,
                records_total=len(candidates),
                records_solved=solved,
                new_variant_attempts=new_attempts,
                pantograph_restarts=restarts,
            )
    report = {
        "schema_version": "numinamath_candidate_verification_report_v1",
        "repair_version": REPAIR_VERSION,
        "batch_id": candidates[0]["batch_id"],
        "input_candidates": len(candidates),
        "records_completed": completed,
        "records_solved": solved,
        "records_failed": completed - solved,
        "new_variant_attempts": new_attempts,
        "pantograph_restarts": restarts,
        "candidate_manifest_sha256": sha256_file(batch_dir / "candidate_manifest.jsonl"),
        "verification_results_sha256": sha256_file(result_path),
        "environment": dict(identity),
        "master_dataset_mutated": False,
        "external_api_called": any(
            candidate.get("lane") == "deepseek_v4_flash_assisted" for candidate in candidates
        ),
        "persistent_worker_pid": os.getpid(),
    }
    _write_json(batch_dir / "candidate_verification_report.json", report)
    return verifier, restarts


def run_worker(args: argparse.Namespace) -> None:
    queue_dir = args.queue_dir.resolve()
    pending = queue_dir / "pending"
    running = queue_dir / "running"
    done = queue_dir / "done"
    failed = queue_dir / "failed"
    for path in (pending, running, done, failed):
        path.mkdir(parents=True, exist_ok=True)
    lock = queue_dir / "worker.lock"
    if lock.exists():
        try:
            old_pid = int(json.loads(lock.read_text(encoding="utf-8"))["pid"])
            os.kill(old_pid, 0)
        except (OSError, ValueError, KeyError, json.JSONDecodeError):
            lock.unlink(missing_ok=True)
        else:
            raise RuntimeError(f"a worker is already alive with pid {old_pid}")
    _write_json_atomic(lock, {"pid": os.getpid(), "started_unix": int(time.time())})
    verifier = None
    try:
        verifier = start_verifier(args)
        identity = environment_identity(args.lean_project, ("Mathlib",))
        _heartbeat(queue_dir, status="ready", environment=identity)
        while not (queue_dir / "STOP").exists():
            jobs = sorted(pending.glob("*.json"))
            if not jobs:
                _heartbeat(queue_dir, status="idle", environment=identity)
                time.sleep(args.poll_seconds)
                continue
            job_path = jobs[0]
            claimed = running / job_path.name
            job_path.replace(claimed)
            try:
                job = json.loads(claimed.read_text(encoding="utf-8"))
                job.setdefault("lean_project", str(args.lean_project.resolve()))
                _heartbeat(queue_dir, status="running", batch_id=job.get("batch_id"))
                verifier, _ = verify_job(
                    job=job, verifier=verifier, identity=identity, queue_dir=queue_dir
                )
                claimed.replace(done / claimed.name)
                _heartbeat(queue_dir, status="idle", last_completed=job.get("batch_id"))
            except Exception as error:
                failure = {
                    "schema_version": WORKER_SCHEMA,
                    "job_file": str(claimed),
                    "error": f"{type(error).__name__}: {error}",
                    "traceback": traceback.format_exc(),
                    "failed_unix": int(time.time()),
                }
                _write_json(failed / f"{claimed.stem}.error.json", failure)
                claimed.replace(failed / claimed.name)
                _heartbeat(queue_dir, status="job_failed", error=failure["error"])
    finally:
        if verifier is not None:
            _safe_close(verifier)
        lock.unlink(missing_ok=True)
        _heartbeat(queue_dir, status="stopped")


def enqueue(args: argparse.Namespace) -> None:
    queue_dir = args.queue_dir.resolve()
    pending = queue_dir / "pending"
    pending.mkdir(parents=True, exist_ok=True)
    batch_dir = args.batch_dir.resolve()
    report = json.loads((batch_dir / "prepare_report.json").read_text(encoding="utf-8"))
    batch_id = str(report["batch_id"])
    job = {
        "schema_version": WORKER_SCHEMA,
        "batch_id": batch_id,
        "batch_dir": str(batch_dir),
        "lean_project": str(args.lean_project.resolve()),
        "timeout": args.timeout,
        "candidate_manifest_sha256": sha256_file(batch_dir / "candidate_manifest.jsonl"),
        "enqueued_unix": int(time.time()),
    }
    path = pending / f"{batch_id}.json"
    if path.exists() or any((queue_dir / state / path.name).exists() for state in ("running", "done", "failed")):
        raise FileExistsError(f"job already exists: {batch_id}")
    _write_json_atomic(path, job)
    print(json.dumps(job, ensure_ascii=False, indent=2))


def status(args: argparse.Namespace) -> None:
    heartbeat = args.queue_dir.resolve() / "heartbeat.json"
    if not heartbeat.exists():
        print(json.dumps({"status": "not_started"}))
        return
    print(heartbeat.read_text(encoding="utf-8"), end="")


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(description=__doc__)
    sub = value.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run")
    run.add_argument("--queue-dir", type=Path, default=Path("outputs/numinamath_repair/worker_queue"))
    run.add_argument("--lean-project", type=Path, default=Path("lean_project"))
    run.add_argument("--timeout", type=int, default=30)
    run.add_argument("--poll-seconds", type=float, default=1.0)
    enqueue_parser = sub.add_parser("enqueue")
    enqueue_parser.add_argument("--queue-dir", type=Path, default=Path("outputs/numinamath_repair/worker_queue"))
    enqueue_parser.add_argument("--batch-dir", type=Path, required=True)
    enqueue_parser.add_argument("--lean-project", type=Path, default=Path("lean_project"))
    enqueue_parser.add_argument("--timeout", type=int, default=30)
    status_parser = sub.add_parser("status")
    status_parser.add_argument("--queue-dir", type=Path, default=Path("outputs/numinamath_repair/worker_queue"))
    return value


def main() -> None:
    args = parser().parse_args()
    if args.command == "run":
        run_worker(args)
    elif args.command == "enqueue":
        enqueue(args)
    else:
        status(args)


if __name__ == "__main__":
    main()
