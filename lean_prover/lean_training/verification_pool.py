"""Multiprocess Pantograph verification pool."""

from __future__ import annotations

import multiprocessing as mp
import queue
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

from lean_prover.lean_training.pantograph_verifier import PantographTaskVerifier
from lean_prover.lean_training.verification_schema import (
    VerificationResult,
    VerificationTask,
    VerificationWarmupReport,
    make_verification_result,
)


@dataclass
class VerificationPoolConfig:
    lean_project_path: str
    imports: tuple[str, ...] = ("Mathlib",)
    timeout: int = 120
    warmup_timeout: int = 120
    num_workers: int = 1
    queue_maxsize: int = 128
    disable_warmup: bool = False
    cancel_on_success: bool = False


@dataclass
class VerificationPoolRun:
    results: list[dict] = field(default_factory=list)
    warmup_reports: list[dict] = field(default_factory=list)
    fatal_errors: list[str] = field(default_factory=list)
    recovered_worker_failures: list[str] = field(default_factory=list)


def run_verification_pool(
    tasks: Iterable[VerificationTask],
    config: VerificationPoolConfig,
    *,
    on_result: Callable[[dict], None] | None = None,
) -> VerificationPoolRun:
    pool = VerificationPool(config)
    run = VerificationPoolRun()
    try:
        pool.start()
        run.warmup_reports.extend(pool.warmup_reports)
        run.fatal_errors.extend(pool.fatal_errors)
        if run.fatal_errors:
            return run
        for task in tasks:
            pool.submit(task)
            for result in pool.drain():
                run.results.append(result)
                if on_result:
                    on_result(result)
        pool.finish()
        for result in pool.drain_until_done():
            run.results.append(result)
            if on_result:
                on_result(result)
        run.recovered_worker_failures.extend(pool.recovered_worker_failures)
        run.fatal_errors.extend(pool.fatal_errors)
    finally:
        pool.close()
    return run


class VerificationPool:
    def __init__(self, config: VerificationPoolConfig) -> None:
        self.config = config
        self.ctx = mp.get_context("spawn")
        self.result_queue = self.ctx.Queue()
        self.queues = [
            self.ctx.Queue(maxsize=config.queue_maxsize)
            for _ in range(config.num_workers)
        ]
        self.workers: list[mp.Process] = []
        self.warmup_reports: list[dict] = []
        self.fatal_errors: list[str] = []
        self.recovered_worker_failures: list[str] = []
        self.in_flight: dict[int, VerificationTask] = {}
        self.next_worker = 0
        self.shutting_down = False

    def start(self) -> None:
        self.workers = [self._make_worker(worker_id) for worker_id in range(self.config.num_workers)]
        for worker in self.workers:
            worker.start()
        while len(self.warmup_reports) + len(self.fatal_errors) < self.config.num_workers:
            self.drain(block=True, timeout=1.0)
            self._monitor_workers(warmup_phase=True)

    def submit(self, task: VerificationTask) -> None:
        worker_id = task.problem_index % self.config.num_workers
        self.queues[worker_id].put((task.priority, task.attempt_index, task))

    def finish(self) -> None:
        self.shutting_down = True
        for task_queue in self.queues:
            task_queue.put(None)

    def drain(self, *, block: bool = False, timeout: float = 0.0) -> list[dict]:
        results: list[dict] = []
        while True:
            try:
                if block and not results:
                    message = self.result_queue.get(timeout=timeout)
                else:
                    message = self.result_queue.get_nowait()
            except queue.Empty:
                break
            message_type = message.get("type")
            if message_type == "warmup":
                self.warmup_reports.append(message["report"])
            elif message_type == "started":
                self.in_flight[int(message["worker_id"])] = message["task"]
            elif message_type == "result":
                worker_id = message.get("worker_id")
                if worker_id is not None:
                    self.in_flight.pop(int(worker_id), None)
                results.append(message["result"])
            elif message_type == "fatal":
                worker_id = int(message.get("worker_id", -1))
                error = str(message.get("error", "unknown worker error"))
                if self.shutting_down or len(self.warmup_reports) < self.config.num_workers:
                    self.fatal_errors.append(error)
                elif worker_id >= 0:
                    self._restart_worker(worker_id, error)
                else:
                    self.fatal_errors.append(error)
            elif message_type == "done":
                pass
        self._monitor_workers(warmup_phase=False)
        return results

    def drain_until_done(self) -> list[dict]:
        results: list[dict] = []
        while any(worker.is_alive() for worker in self.workers):
            results.extend(self.drain(block=True, timeout=1.0))
        for worker in self.workers:
            worker.join()
        results.extend(self.drain())
        return results

    def close(self) -> None:
        self.shutting_down = True
        for task_queue in self.queues:
            try:
                task_queue.put_nowait(None)
            except Exception:
                pass
        for worker in self.workers:
            if worker.is_alive():
                worker.join(timeout=5)
                if worker.is_alive():
                    worker.terminate()
        for task_queue in self.queues:
            try:
                task_queue.close()
            except Exception:
                pass

    def _make_worker(self, worker_id: int):
        return self.ctx.Process(
            target=worker_process_loop,
            kwargs={
                "worker_id": worker_id,
                "task_queue": self.queues[worker_id],
                "result_queue": self.result_queue,
                "config": self.config,
            },
            daemon=True,
        )

    def _restart_worker(self, worker_id: int, reason: str) -> None:
        task = self.in_flight.pop(worker_id, None)
        if task is not None:
            retry_task = VerificationTask(
                priority=task.priority,
                problem_index=task.problem_index,
                attempt_index=task.attempt_index,
                problem_id=task.problem_id,
                prompt=task.prompt,
                generated_proof=task.generated_proof,
                raw_completion=task.raw_completion,
                lean_code=task.lean_code,
                imports=task.imports,
                context_lines=task.context_lines,
                enqueue_time=time.monotonic(),
                generation_seconds=task.generation_seconds,
                payload=task.payload,
                reject_forbidden=task.reject_forbidden,
            )
            self.queues[worker_id].put(
                (retry_task.priority, retry_task.attempt_index, retry_task)
            )
        replacement = self._make_worker(worker_id)
        self.workers[worker_id] = replacement
        replacement.start()
        self.recovered_worker_failures.append(
            f"restarted worker {worker_id}: {reason}"
        )

    def _monitor_workers(self, *, warmup_phase: bool) -> None:
        if self.shutting_down:
            return
        for worker_id, worker in enumerate(self.workers):
            if worker.is_alive() or worker.exitcode in (None, 0):
                continue
            if warmup_phase:
                self.fatal_errors.append(
                    f"worker {worker_id} exited before warmup: exitcode={worker.exitcode}"
                )
            else:
                self._restart_worker(worker_id, f"process exitcode={worker.exitcode}")


def worker_process_loop(
    *,
    worker_id: int,
    task_queue,
    result_queue,
    config: VerificationPoolConfig,
) -> None:
    verifier = None
    canceled_problem_ids: set[str] = set()
    try:
        verifier = PantographTaskVerifier(
            Path(config.lean_project_path),
            timeout=config.timeout,
            default_imports=config.imports,
            worker_id=worker_id,
        )
        if config.disable_warmup:
            report = VerificationWarmupReport(
                worker_id=worker_id,
                backend=verifier.backend_name,
                enabled=False,
                success=True,
                post_startup_warmup_seconds=0.0,
                diagnostics="warmup disabled",
            )
        else:
            report = verifier.warmup(config.warmup_timeout)
            report.worker_id = worker_id
        result_queue.put(
            {"type": "warmup", "worker_id": worker_id, "report": report.to_json()}
        )
        if not report.success:
            result_queue.put(
                {
                    "type": "fatal",
                    "worker_id": worker_id,
                    "error": f"worker {worker_id} warmup failed: {report.diagnostics}",
                }
            )
            return
        while True:
            item = task_queue.get()
            if item is None:
                break
            _, _, task = item
            if task is None:
                break
            dequeue_time = time.monotonic()
            result_queue.put(
                {"type": "started", "worker_id": worker_id, "task": task}
            )
            if config.cancel_on_success and task.problem_id in canceled_problem_ids:
                result = make_verification_result(
                    task,
                    worker_id=worker_id,
                    status="canceled",
                    success=False,
                    diagnostics="canceled after another attempt succeeded",
                    verifier_backend=verifier.backend_name,
                    queue_wait_seconds=round(dequeue_time - task.enqueue_time, 4),
                    total_seconds=round(dequeue_time - task.enqueue_time, 4),
                    compile_messages=("canceled after another attempt succeeded",),
                )
                result_queue.put(
                    {"type": "result", "worker_id": worker_id, "result": result.to_json()}
                )
                continue
            result_payload = verifier.verify(task)
            done_time = time.monotonic()
            result = make_verification_result(
                task,
                worker_id=worker_id,
                status="success" if result_payload["success"] else "failed",
                success=bool(result_payload["success"]),
                diagnostics=str(result_payload.get("diagnostics", "")),
                verification_seconds=float(
                    result_payload.get("verification_seconds", 0.0)
                ),
                queue_wait_seconds=round(dequeue_time - task.enqueue_time, 4),
                total_seconds=round(done_time - task.enqueue_time, 4),
                timed_out=bool(result_payload.get("timed_out", False)),
                verifier_backend=verifier.backend_name,
                compile_messages=result_payload.get("messages", ()),
                compile_errors=result_payload.get("errors", ()),
                compile_warnings=result_payload.get("warnings", ()),
            )
            if config.cancel_on_success and result.success:
                canceled_problem_ids.add(task.problem_id)
            result_queue.put(
                {"type": "result", "worker_id": worker_id, "result": result.to_json()}
            )
    except Exception as error:
        result_queue.put(
            {
                "type": "fatal",
                "worker_id": worker_id,
                "error": f"worker {worker_id} failed: {error}",
            }
        )
    finally:
        if verifier is not None:
            verifier.close()
        result_queue.put({"type": "done", "worker_id": worker_id})
