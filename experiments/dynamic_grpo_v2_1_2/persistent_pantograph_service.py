#!/usr/bin/env python3
"""Node-local persistent Pantograph workers shared across pipeline phases.

One service process owns one ``VerificationPool`` containing one warm Lean
worker.  A launcher starts four services serially, then generation verification
and the four DDP training ranks attach to the same Unix sockets.  Client close
only disconnects; workers are stopped solely by the allocation-level owner.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
from multiprocessing.connection import Client, Listener
import os
from pathlib import Path
import signal
import threading
import time
from typing import Iterable

from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
    VerificationPoolRun,
)
from lean_prover.lean_training.verification.schema import VerificationTask


def _authkey() -> bytes:
    value = os.environ.get("PANTOGRAPH_POOL_AUTHKEY")
    if not value:
        raise RuntimeError("PANTOGRAPH_POOL_AUTHKEY is required")
    return value.encode("utf-8")


def _task_payload(task: VerificationTask) -> dict:
    return asdict(task)


def _restore_task(payload: dict) -> VerificationTask:
    value = dict(payload)
    for key in ("imports", "context_lines"):
        if key in value and value[key] is not None:
            value[key] = tuple(value[key])
    return VerificationTask(**value)


class WorkerSocketClient:
    """Synchronous client for one persistent worker service."""

    def __init__(self, socket_path: Path) -> None:
        self.socket_path = socket_path

    def request(self, payload: dict) -> dict:
        connection = Client(str(self.socket_path), family="AF_UNIX", authkey=_authkey())
        try:
            connection.send(payload)
            response = connection.recv()
        finally:
            connection.close()
        if not response.get("ok"):
            raise RuntimeError(response.get("error", "persistent Pantograph request failed"))
        return response

    def run_batch(self, tasks: list[VerificationTask]) -> VerificationPoolRun:
        response = self.request(
            {"op": "run_batch", "tasks": [_task_payload(task) for task in tasks]}
        )
        return VerificationPoolRun(**response["run"])

    def snapshot(self) -> dict:
        return self.request({"op": "snapshot"})["snapshot"]


class PersistentVerificationPool:
    """Drop-in pool facade backed by allocation-lifetime Unix services."""

    def __init__(self, config: VerificationPoolConfig) -> None:
        del config
        socket_dir = os.environ.get("PANTOGRAPH_POOL_SOCKET_DIR")
        if not socket_dir:
            raise RuntimeError("PANTOGRAPH_POOL_SOCKET_DIR is required")
        self.socket_dir = Path(socket_dir)
        mode = os.environ.get("PANTOGRAPH_POOL_CLIENT_MODE", "global")
        if mode == "rank":
            rank = int(os.environ.get("LOCAL_RANK", os.environ.get("RANK", "0")))
            indices = [rank % int(os.environ.get("PANTOGRAPH_POOL_WORKERS", "4"))]
        elif mode == "global":
            indices = list(range(int(os.environ.get("PANTOGRAPH_POOL_WORKERS", "4"))))
        else:
            raise ValueError(f"unknown PANTOGRAPH_POOL_CLIENT_MODE={mode!r}")
        self.clients = {
            index: WorkerSocketClient(self.socket_dir / f"worker-{index}.sock")
            for index in indices
        }
        self.mode = mode
        self.warmup_reports: list[dict] = []
        self.fatal_errors: list[str] = []
        self.recovered_worker_failures: list[str] = []
        self.started = False

    def start(self) -> None:
        if self.started:
            return
        snapshots = [client.snapshot() for client in self.clients.values()]
        for snapshot in snapshots:
            self.warmup_reports.extend(snapshot.get("warmup_reports", []))
            self.fatal_errors.extend(snapshot.get("fatal_errors", []))
            self.recovered_worker_failures.extend(
                snapshot.get("recovered_worker_failures", [])
            )
        if self.fatal_errors:
            raise RuntimeError(f"persistent Pantograph services are unhealthy: {self.fatal_errors}")
        self.started = True

    def run_batch(self, tasks: Iterable[VerificationTask], *, on_result=None) -> VerificationPoolRun:
        self.start()
        task_list = list(tasks)
        if self.mode == "rank":
            client = next(iter(self.clients.values()))
            run = client.run_batch(task_list)
        else:
            indices = sorted(self.clients)
            partitions = {index: [] for index in indices}
            for task in task_list:
                index = indices[int(task.problem_index) % len(indices)]
                partitions[index].append(task)
            with ThreadPoolExecutor(max_workers=len(indices)) as executor:
                futures = {
                    index: executor.submit(self.clients[index].run_batch, batch)
                    for index, batch in partitions.items()
                    if batch
                }
                runs = [future.result() for future in futures.values()]
            run = VerificationPoolRun()
            for part in runs:
                run.results.extend(part.results)
                run.warmup_reports.extend(part.warmup_reports)
                run.fatal_errors.extend(part.fatal_errors)
                run.recovered_worker_failures.extend(part.recovered_worker_failures)
                run.runtime_stats[str(len(run.runtime_stats))] = part.runtime_stats
        if on_result:
            for result in run.results:
                on_result(result)
        self.fatal_errors.extend(run.fatal_errors)
        self.recovered_worker_failures.extend(run.recovered_worker_failures)
        return run

    def runtime_snapshot(self) -> dict:
        return {
            "backend": "allocation_persistent_unix_service",
            "mode": self.mode,
            "services": {
                str(index): client.snapshot()
                for index, client in self.clients.items()
            },
        }

    def close(self) -> None:
        # The allocation-level owner, not phase clients, owns worker lifetime.
        self.started = False


def serve(args: argparse.Namespace) -> None:
    socket_path = Path(args.socket)
    ready_path = Path(args.ready_file)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    ready_path.unlink(missing_ok=True)
    pool = VerificationPool(
        VerificationPoolConfig(
            lean_project_path=args.lean_project,
            imports=("Mathlib",),
            timeout=args.timeout,
            warmup_timeout=args.warmup_timeout,
            num_workers=1,
            queue_maxsize=args.queue_maxsize,
            task_spool_dir=args.task_spool_dir,
            cancel_on_success=False,
        )
    )
    listener = None
    stopping = threading.Event()

    def stop(_signum=None, _frame=None) -> None:
        stopping.set()
        if listener is not None:
            try:
                listener.close()
            except Exception:
                pass

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        pool.start()
        listener = Listener(str(socket_path), family="AF_UNIX", authkey=_authkey())
        snapshot = pool.runtime_snapshot()
        ready_path.write_text(
            json.dumps(
                {
                    "status": "ready",
                    "service_pid": os.getpid(),
                    "worker_index": args.worker_index,
                    "warmup_reports": list(pool.warmup_reports),
                    "runtime": snapshot,
                    "ready_at_unix": time.time(),
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        while not stopping.is_set():
            try:
                connection = listener.accept()
            except (OSError, EOFError):
                if stopping.is_set():
                    break
                raise
            try:
                request = connection.recv()
                operation = request.get("op")
                if operation == "run_batch":
                    run = pool.run_batch(_restore_task(row) for row in request["tasks"])
                    connection.send({"ok": True, "run": asdict(run)})
                elif operation == "snapshot":
                    connection.send(
                        {
                            "ok": True,
                            "snapshot": {
                                "worker_index": args.worker_index,
                                "service_pid": os.getpid(),
                                "warmup_reports": list(pool.warmup_reports),
                                "fatal_errors": list(pool.fatal_errors),
                                "recovered_worker_failures": list(pool.recovered_worker_failures),
                                "runtime": pool.runtime_snapshot(),
                            },
                        }
                    )
                elif operation == "shutdown":
                    connection.send({"ok": True})
                    stopping.set()
                else:
                    connection.send({"ok": False, "error": f"unknown op {operation!r}"})
            except Exception as error:
                try:
                    connection.send({"ok": False, "error": repr(error)})
                except Exception:
                    pass
            finally:
                connection.close()
    finally:
        if listener is not None:
            try:
                listener.close()
            except Exception:
                pass
        pool.close()
        socket_path.unlink(missing_ok=True)
        ready_path.unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--worker-index", type=int, required=True)
    parser.add_argument("--lean-project", required=True)
    parser.add_argument("--task-spool-dir", required=True)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--warmup-timeout", type=int, default=1800)
    parser.add_argument("--queue-maxsize", type=int, default=256)
    return parser.parse_args()


if __name__ == "__main__":
    serve(parse_args())
