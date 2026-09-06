"""Persistent multiprocess Pantograph verification pool."""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import os
import queue
import signal
import tempfile
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Callable, Iterable

from lean_prover.lean_training.verification.pantograph import PantographTaskVerifier
from lean_prover.lean_training.verification.schema import (
    VerificationTask,
    VerificationTaskRef,
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
    heartbeat_interval: float = 5.0
    heartbeat_timeout: float = 30.0
    max_worker_restarts: int = 3
    max_task_retries: int = 2
    shutdown_timeout: float = 10.0
    task_spool_dir: str | None = None
    save_full_source_on_failure_only: bool = True


@dataclass
class VerificationPoolRun:
    results: list[dict] = field(default_factory=list)
    warmup_reports: list[dict] = field(default_factory=list)
    fatal_errors: list[str] = field(default_factory=list)
    recovered_worker_failures: list[str] = field(default_factory=list)
    runtime_stats: dict = field(default_factory=dict)


def run_verification_pool(
    tasks: Iterable[VerificationTask],
    config: VerificationPoolConfig,
    *,
    on_result: Callable[[dict], None] | None = None,
) -> VerificationPoolRun:
    """Run one batch with a temporary pool, preserving the legacy API."""

    pool = VerificationPool(config)
    try:
        return pool.run_batch(tasks, on_result=on_result)
    finally:
        pool.close()


class VerificationPool:
    """Keep independent Pantograph workers alive across verification batches."""

    def __init__(self, config: VerificationPoolConfig) -> None:
        if config.num_workers <= 0 or config.queue_maxsize <= 0:
            raise ValueError("worker count and queue size must be positive")
        if config.heartbeat_interval <= 0 or config.heartbeat_timeout <= 0:
            raise ValueError("heartbeat intervals must be positive")
        if config.heartbeat_timeout <= config.heartbeat_interval:
            raise ValueError("heartbeat_timeout must exceed heartbeat_interval")
        self.config = config
        self.task_spool_dir = Path(
            config.task_spool_dir
            or tempfile.mkdtemp(prefix="lean-verification-tasks-")
        )
        self.task_spool_dir.mkdir(parents=True, exist_ok=True)
        self.ctx = mp.get_context("spawn")
        self.result_queue = self.ctx.Queue(maxsize=config.queue_maxsize)
        self.queues = [
            self.ctx.Queue(maxsize=config.queue_maxsize)
            for _ in range(config.num_workers)
        ]
        self.workers: list[mp.Process | None] = [None] * config.num_workers
        self.worker_generations = [0] * config.num_workers
        self.worker_restart_counts = [0] * config.num_workers
        self.worker_pids: dict[int, int] = {}
        self.worker_states: dict[int, str] = {}
        self.last_heartbeats: dict[int, float] = {}
        self.ready_generations: dict[int, int] = {}
        self.warmup_reports: list[dict] = []
        self.fatal_errors: list[str] = []
        self.recovered_worker_failures: list[str] = []
        self.in_flight: dict[int, VerificationTaskRef] = {}
        self.in_flight_started: dict[int, float] = {}
        self.task_retries: dict[str, int] = {}
        self.buffered_results: list[dict] = []
        self.started = False
        self.shutting_down = False

    def __enter__(self) -> "VerificationPool":
        self.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def start(self) -> None:
        """Start all workers and block until every Pantograph server is warm."""

        if self.started:
            return
        self.shutting_down = False
        self.started = True
        startup_timeout = max(
            self.config.timeout + self.config.warmup_timeout + 30,
            self.config.heartbeat_timeout * 2,
        )
        # This 16-core node suffers severe Mathlib cold-start contention when
        # four Pantograph servers initialize at once.  Warm workers one at a
        # time, then keep all four resident for the parallel verification run.
        for worker_id in range(self.config.num_workers):
            self._spawn_worker(worker_id)
            deadline = time.monotonic() + startup_timeout
            while (
                self.ready_generations.get(worker_id)
                != self.worker_generations[worker_id]
            ):
                self.drain(block=True, timeout=1.0, monitor_heartbeats=False)
                self._monitor_workers(
                    warmup_phase=True,
                    monitor_heartbeats=False,
                )
                if self.fatal_errors:
                    raise RuntimeError(
                        f"Pantograph pool startup failed: {self.fatal_errors}"
                    )
                if time.monotonic() >= deadline:
                    raise TimeoutError(
                        "Pantograph worker did not finish serial startup before "
                        f"the pool deadline: [{worker_id}]"
                    )
        deadline = time.monotonic() + startup_timeout
        while not self._all_workers_ready():
            self.drain(block=True, timeout=1.0, monitor_heartbeats=False)
            self._monitor_workers(warmup_phase=True, monitor_heartbeats=False)
            if self.fatal_errors:
                raise RuntimeError(
                    f"Pantograph pool startup failed: {self.fatal_errors}"
                )
            if time.monotonic() >= deadline:
                pending = [
                    worker_id
                    for worker_id in range(self.config.num_workers)
                    if self.ready_generations.get(worker_id)
                    != self.worker_generations[worker_id]
                ]
                raise TimeoutError(
                    "Pantograph workers did not finish startup before the pool "
                    f"deadline: {pending}"
                )

    def run_batch(
        self,
        tasks: Iterable[VerificationTask],
        *,
        on_result: Callable[[dict], None] | None = None,
    ) -> VerificationPoolRun:
        """Submit a bounded batch without stopping the persistent workers."""

        self.start()
        run = VerificationPoolRun()
        warmup_offset = len(self.warmup_reports)
        failure_offset = len(self.recovered_worker_failures)
        expected: set[str] = set()
        completed: set[str] = set()

        def accept(results: Iterable[dict]) -> None:
            for result in results:
                key = _result_key(result)
                if key not in expected or key in completed:
                    continue
                completed.add(key)
                run.results.append(result)
                if on_result:
                    on_result(result)

        for task in tasks:
            expected.add(_task_key(task))
            self.submit(task)
            buffered = self.buffered_results
            self.buffered_results = []
            accept(buffered)
            accept(self.drain(block=False))
        while completed != expected and not self.fatal_errors:
            accept(self.drain(block=True, timeout=1.0))
        run.warmup_reports.extend(self.warmup_reports[warmup_offset:])
        run.recovered_worker_failures.extend(
            self.recovered_worker_failures[failure_offset:]
        )
        run.fatal_errors.extend(self.fatal_errors)
        run.runtime_stats = self.runtime_snapshot()
        return run

    def submit(self, task: VerificationTask) -> None:
        """Submit a task to its stable worker shard using a bounded queue."""

        if not self.started or self.shutting_down:
            raise RuntimeError("verification pool is not accepting tasks")
        worker_id = task.problem_index % self.config.num_workers
        queued_task = replace(task, enqueue_time=time.monotonic())
        task_ref = self._spool_task(queued_task)
        item = (task_ref.priority, task_ref.attempt_index, task_ref)
        while True:
            try:
                self.queues[worker_id].put(item, block=True, timeout=1.0)
                return
            except queue.Full:
                self.buffered_results.extend(
                    self.drain(block=False, monitor_heartbeats=True)
                )
                if self.fatal_errors:
                    raise RuntimeError(
                        f"verification pool failed while submitting: {self.fatal_errors}"
                    )

    def drain(
        self,
        *,
        block: bool = False,
        timeout: float = 0.0,
        monitor_heartbeats: bool = True,
    ) -> list[dict]:
        results: list[dict] = []
        received_any = False
        while True:
            try:
                if block and not received_any:
                    message = self.result_queue.get(timeout=timeout)
                else:
                    message = self.result_queue.get_nowait()
            except queue.Empty:
                break
            received_any = True
            message_type = message.get("type")
            worker_id = int(message.get("worker_id", -1))
            generation = int(message.get("generation", -1))
            stale_generation = (
                worker_id >= 0 and generation != self.worker_generations[worker_id]
            )
            # A worker can finish several queued tasks just before its heartbeat
            # is judged stale.  Those already-completed verification results are
            # still valid and must reach the coordinator; only stale lifecycle
            # messages are obsolete.  Consumers deduplicate by task/result key
            # if the in-flight task was also retried on the replacement worker.
            if stale_generation and message_type != "result":
                continue
            if message_type in {"booting", "heartbeat"}:
                self.worker_pids[worker_id] = int(message["pid"])
                self.worker_states[worker_id] = str(message.get("state", "unknown"))
                self.last_heartbeats[worker_id] = time.monotonic()
            if message_type == "warmup":
                report = dict(message["report"])
                report.update(
                    {
                        "pid": int(message["pid"]),
                        "generation": generation,
                        "pool_restart_count": self.worker_restart_counts[worker_id],
                    }
                )
                self.warmup_reports.append(report)
                self.ready_generations[worker_id] = generation
                self.worker_states[worker_id] = "idle"
                self.last_heartbeats[worker_id] = time.monotonic()
            elif message_type == "started":
                self.in_flight[worker_id] = message["task"]
                self.in_flight_started[worker_id] = time.monotonic()
                self.worker_states[worker_id] = "verifying"
            elif message_type == "result":
                current_task = self.in_flight.get(worker_id)
                incoming_result = dict(message["result"])
                if (
                    not stale_generation
                    and current_task is not None
                    and _task_key(current_task) == _result_key(incoming_result)
                ):
                    self.in_flight.pop(worker_id, None)
                    self.in_flight_started.pop(worker_id, None)
                    self.worker_states[worker_id] = "idle"
                result = incoming_result
                result.update(
                    {
                        "worker_pid": int(message["pid"]),
                        "worker_generation": generation,
                        "stale_worker_generation": stale_generation,
                        "worker_restart_count": self.worker_restart_counts[worker_id],
                        "pantograph_restart_count": int(
                            message.get("pantograph_restart_count", 0)
                        ),
                        "source_path": (
                            current_task.task_path if current_task is not None else None
                        ),
                    }
                )
                results.append(result)
                if result.get("success") and current_task is not None:
                    Path(current_task.task_path).unlink(missing_ok=True)
            elif message_type == "fatal":
                self._restart_worker(
                    worker_id,
                    str(message.get("error", "unknown worker error")),
                )
            elif message_type == "done":
                self.worker_states[worker_id] = "stopped"
        self._monitor_workers(
            warmup_phase=not self._all_workers_ready(),
            monitor_heartbeats=monitor_heartbeats,
        )
        if self.buffered_results:
            results.extend(self.buffered_results)
            self.buffered_results = []
        return results

    def runtime_snapshot(self) -> dict:
        now = time.monotonic()
        latest_warmups = {
            int(report["worker_id"]): report for report in self.warmup_reports
        }
        workers = []
        for worker_id, process in enumerate(self.workers):
            report = latest_warmups.get(worker_id, {})
            heartbeat = self.last_heartbeats.get(worker_id)
            workers.append(
                {
                    "worker_id": worker_id,
                    "pid": self.worker_pids.get(worker_id),
                    "alive": bool(process and process.is_alive()),
                    "state": self.worker_states.get(worker_id, "starting"),
                    "generation": self.worker_generations[worker_id],
                    "restart_count": self.worker_restart_counts[worker_id],
                    "last_heartbeat_age_seconds": (
                        round(now - heartbeat, 4) if heartbeat is not None else None
                    ),
                    "server_startup_seconds": report.get("server_startup_seconds"),
                    "post_startup_warmup_seconds": report.get(
                        "post_startup_warmup_seconds"
                    ),
                }
            )
        return {
            "start_method": self.ctx.get_start_method(),
            "queue_maxsize": self.config.queue_maxsize,
            "heartbeat_interval": self.config.heartbeat_interval,
            "heartbeat_timeout": self.config.heartbeat_timeout,
            "workers": workers,
            "recovered_worker_failures": list(self.recovered_worker_failures),
            "fatal_errors": list(self.fatal_errors),
        }

    def close(self) -> None:
        """Stop every worker and close all IPC handles owned by the coordinator."""

        if self.shutting_down:
            return
        self.shutting_down = True
        for task_queue in self.queues:
            try:
                task_queue.put_nowait(None)
            except Exception:
                pass
        for process in self.workers:
            if process is None:
                continue
            if process.is_alive():
                process.join(timeout=self.config.shutdown_timeout)
                if process.is_alive():
                    _terminate_process_tree(process)
                    process.join(timeout=self.config.shutdown_timeout)
                if process.is_alive() and hasattr(process, "kill"):
                    _kill_process_tree(process)
                    process.join(timeout=2)
            try:
                process.close()
            except Exception:
                pass
        for task_queue in self.queues:
            try:
                task_queue.close()
                task_queue.join_thread()
            except Exception:
                pass
        try:
            self.result_queue.close()
            self.result_queue.join_thread()
        except Exception:
            pass
        self.workers = [None] * self.config.num_workers
        self.in_flight.clear()
        self.in_flight_started.clear()
        self.buffered_results.clear()
        self.started = False

    def _all_workers_ready(self) -> bool:
        return all(
            self.ready_generations.get(worker_id)
            == self.worker_generations[worker_id]
            for worker_id in range(self.config.num_workers)
        )

    def _make_worker(self, worker_id: int) -> mp.Process:
        generation = self.worker_generations[worker_id]
        return self.ctx.Process(
            target=worker_process_loop,
            kwargs={
                "worker_id": worker_id,
                "generation": generation,
                "task_queue": self.queues[worker_id],
                "result_queue": self.result_queue,
                "config": self.config,
            },
            name=f"pantograph-worker-{worker_id}-g{generation}",
            daemon=True,
        )

    def _spawn_worker(self, worker_id: int) -> None:
        process = self._make_worker(worker_id)
        self.workers[worker_id] = process
        self.worker_states[worker_id] = "starting"
        self.ready_generations.pop(worker_id, None)
        previous_cuda_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try:
            process.start()
        finally:
            if previous_cuda_devices is None:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else:
                os.environ["CUDA_VISIBLE_DEVICES"] = previous_cuda_devices

    def _restart_worker(
        self,
        worker_id: int,
        reason: str,
        *,
        retry_in_flight: bool = True,
        enforce_restart_limit: bool = True,
    ) -> None:
        if self.shutting_down or worker_id < 0:
            return
        process = self.workers[worker_id]
        if process is not None and process.is_alive():
            _terminate_process_tree(process)
            process.join(timeout=self.config.shutdown_timeout)
        task = self.in_flight.pop(worker_id, None)
        self.in_flight_started.pop(worker_id, None)
        retry_item = None
        if task is not None and retry_in_flight:
            key = _task_key(task)
            retry_count = self.task_retries.get(key, 0) + 1
            self.task_retries[key] = retry_count
            if retry_count > self.config.max_task_retries:
                self.fatal_errors.append(
                    f"task {key} exceeded max_task_retries after worker {worker_id} failed"
                )
                return
            retry_task = replace(task, enqueue_time=time.monotonic())
            if isinstance(retry_task, VerificationTask):
                retry_task = self._spool_task(retry_task)
            retry_item = (
                retry_task.priority,
                retry_task.attempt_index,
                retry_task,
            )
        self.worker_restart_counts[worker_id] += 1
        if (
            enforce_restart_limit
            and self.worker_restart_counts[worker_id] > self.config.max_worker_restarts
        ):
            self.fatal_errors.append(
                f"worker {worker_id} exceeded max_worker_restarts: {reason}"
            )
            return
        self.worker_generations[worker_id] += 1
        self.recovered_worker_failures.append(
            f"restarted worker {worker_id}: {reason}"
        )
        self._spawn_worker(worker_id)
        # Start the replacement before requeueing its in-flight task.  Putting
        # into a full bounded queue before a consumer exists deadlocks the
        # coordinator precisely when a worker fails under load.
        if retry_item is not None:
            while True:
                try:
                    self.queues[worker_id].put(
                        retry_item,
                        block=True,
                        timeout=1.0,
                    )
                    break
                except queue.Full:
                    replacement = self.workers[worker_id]
                    if replacement is None or not replacement.is_alive():
                        self.fatal_errors.append(
                            f"replacement worker {worker_id} exited before "
                            "the in-flight task could be requeued"
                        )
                        break

    def _hard_timeout_worker(self, worker_id: int, elapsed: float) -> None:
        """End a wedged Pantograph RPC and emit a normal timeout receipt.

        Pantograph's transport timeout does not always interrupt a Lean RPC.
        The worker heartbeat runs on a separate thread, so heartbeat liveness
        alone cannot detect this state.  The coordinator owns the worker
        process group and can therefore enforce the configured per-task limit
        without abandoning the resident-pool architecture.
        """

        task_ref = self.in_flight.get(worker_id)
        if task_ref is None:
            return
        task = _load_task(task_ref)
        generation = self.worker_generations[worker_id]
        worker_pid = self.worker_pids.get(worker_id)
        reason = (
            f"hard task timeout after {elapsed:.1f}s "
            f"(configured timeout={self.config.timeout}s)"
        )
        result = make_verification_result(
            task,
            worker_id=worker_id,
            status="failed",
            success=False,
            diagnostics=reason,
            verification_seconds=float(self.config.timeout),
            queue_wait_seconds=round(
                max(0.0, self.in_flight_started.get(worker_id, time.monotonic()) - task.enqueue_time),
                4,
            ),
            total_seconds=round(time.monotonic() - task.enqueue_time, 4),
            timed_out=True,
            verifier_backend="pantograph",
            compile_messages=(reason,),
            compile_errors=(reason,),
        ).to_json(compact_success=self.config.save_full_source_on_failure_only)
        result.update(
            {
                "worker_pid": worker_pid,
                "worker_generation": generation,
                "stale_worker_generation": False,
                "worker_restart_count": self.worker_restart_counts[worker_id] + 1,
                "pantograph_restart_count": 0,
                "source_path": task_ref.task_path,
            }
        )
        self._restart_worker(
            worker_id,
            reason,
            retry_in_flight=False,
            enforce_restart_limit=False,
        )
        self.recovered_worker_failures.append(
            f"recorded timeout for task {_task_key(task_ref)}"
        )
        self.buffered_results.append(result)

    def _spool_task(self, task: VerificationTask) -> VerificationTaskRef:
        key = _task_key(task)
        safe_key = "".join(character if character.isalnum() or character in "-_" else "_" for character in key)
        path = self.task_spool_dir / f"{safe_key}.json"
        temporary = path.with_suffix(".json.tmp")
        payload = asdict(task)
        payload["imports"] = list(task.imports)
        payload["context_lines"] = list(task.context_lines)
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
        return VerificationTaskRef(
            priority=task.priority,
            attempt_index=task.attempt_index,
            problem_index=task.problem_index,
            task_key=key,
            task_path=str(path),
            enqueue_time=task.enqueue_time,
        )

    def _monitor_workers(
        self,
        *,
        warmup_phase: bool,
        monitor_heartbeats: bool,
    ) -> None:
        if self.shutting_down:
            return
        now = time.monotonic()
        for worker_id, process in enumerate(self.workers):
            if process is None:
                continue
            if not process.is_alive() and process.exitcode is not None:
                self._restart_worker(
                    worker_id,
                    f"process exitcode={process.exitcode}"
                    + (" during warmup" if warmup_phase else ""),
                )
                continue
            heartbeat = self.last_heartbeats.get(worker_id)
            is_ready = (
                self.ready_generations.get(worker_id)
                == self.worker_generations[worker_id]
            )
            task_started = self.in_flight_started.get(worker_id)
            hard_deadline = self.config.timeout + max(
                5.0, self.config.heartbeat_interval * 2
            )
            if (
                is_ready
                and task_started is not None
                and now - task_started > hard_deadline
            ):
                self._hard_timeout_worker(worker_id, now - task_started)
                continue
            if (
                monitor_heartbeats
                and is_ready
                and heartbeat is not None
                and now - heartbeat > self.config.heartbeat_timeout
            ):
                self._restart_worker(
                    worker_id,
                    f"heartbeat timeout after {now - heartbeat:.1f}s",
                )


def worker_process_loop(
    *,
    worker_id: int,
    generation: int,
    task_queue,
    result_queue,
    config: VerificationPoolConfig,
) -> None:
    """Own exactly one Pantograph server and report liveness continuously."""

    if os.name != "nt":
        try:
            os.setsid()
        except PermissionError:
            pass
    asyncio.set_event_loop(asyncio.new_event_loop())

    verifier = None
    stop_heartbeat = threading.Event()
    state = {"value": "starting"}

    def emit(message_type: str, **values) -> None:
        result_queue.put(
            {
                "type": message_type,
                "worker_id": worker_id,
                "generation": generation,
                "pid": os.getpid(),
                **values,
            }
        )

    def heartbeat_loop() -> None:
        while not stop_heartbeat.wait(config.heartbeat_interval):
            emit(
                "heartbeat",
                state=state["value"],
                pantograph_restart_count=(
                    verifier.restart_count if verifier is not None else 0
                ),
            )

    heartbeat = threading.Thread(
        target=heartbeat_loop,
        name=f"pantograph-heartbeat-{worker_id}",
        daemon=True,
    )
    emit("booting", state="starting")
    heartbeat.start()
    canceled_problem_ids: set[str] = set()
    try:
        verifier = PantographTaskVerifier(
            Path(config.lean_project_path),
            timeout=config.timeout,
            default_imports=config.imports,
            worker_id=worker_id,
            startup_timeout=config.warmup_timeout,
        )
        state["value"] = "warming"
        if config.disable_warmup:
            report = VerificationWarmupReport(
                worker_id=worker_id,
                backend=verifier.backend_name,
                enabled=False,
                success=True,
                post_startup_warmup_seconds=0.0,
                diagnostics="warmup disabled",
                server_startup_seconds=verifier.server_startup_seconds,
            )
        else:
            report = verifier.warmup(config.warmup_timeout)
            report.worker_id = worker_id
        emit("warmup", report=report.to_json())
        if not report.success:
            raise RuntimeError(f"warmup failed: {report.diagnostics}")
        state["value"] = "idle"
        while True:
            item = task_queue.get()
            if item is None:
                break
            _, _, task_ref = item
            if task_ref is None:
                break
            task = _load_task(task_ref)
            dequeue_time = time.monotonic()
            state["value"] = "verifying"
            emit("started", state="verifying", task=task_ref)
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
            else:
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
            state["value"] = "idle"
            emit(
                "result",
                state="idle",
                result=result.to_json(
                    compact_success=config.save_full_source_on_failure_only
                ),
                pantograph_restart_count=verifier.restart_count,
            )
    except Exception as error:
        emit("fatal", state="failed", error=f"worker {worker_id} failed: {error}")
    finally:
        state["value"] = "stopping"
        stop_heartbeat.set()
        heartbeat.join(timeout=max(1.0, config.heartbeat_interval * 2))
        if verifier is not None:
            verifier.close()
        emit("done", state="stopped")


def _task_key(task: VerificationTask | VerificationTaskRef) -> str:
    if isinstance(task, VerificationTaskRef):
        return task.task_key
    if task.payload and task.payload.get("generation_id"):
        return str(task.payload["generation_id"])
    return f"{task.problem_id}:{task.problem_index}:{task.attempt_index}"


def _load_task(reference: VerificationTaskRef) -> VerificationTask:
    payload = json.loads(Path(reference.task_path).read_text(encoding="utf-8-sig"))
    payload["imports"] = tuple(payload.get("imports") or ())
    payload["context_lines"] = tuple(payload.get("context_lines") or ())
    payload["enqueue_time"] = reference.enqueue_time
    return VerificationTask(**payload)


def _result_key(result: dict) -> str:
    if result.get("generation_id"):
        return str(result["generation_id"])
    return (
        f"{result.get('problem_id')}:{result.get('problem_index')}:"
        f"{result.get('attempt_index')}"
    )


def _terminate_process_tree(process: mp.Process) -> None:
    """Terminate a worker and the Pantograph server in its process group."""

    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGTERM)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    process.terminate()


def _kill_process_tree(process: mp.Process) -> None:
    if os.name != "nt":
        try:
            os.killpg(process.pid, signal.SIGKILL)
            return
        except ProcessLookupError:
            return
        except OSError:
            pass
    process.kill()
