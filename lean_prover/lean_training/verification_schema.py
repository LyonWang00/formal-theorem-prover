"""Shared schemas for Pantograph-backed Lean verification."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable


@dataclass(frozen=True)
class VerificationTask:
    priority: int
    problem_index: int
    attempt_index: int
    problem_id: str
    prompt: str
    generated_proof: str
    raw_completion: str
    lean_code: str
    imports: tuple[str, ...]
    context_lines: tuple[str, ...] = ()
    enqueue_time: float = 0.0
    generation_seconds: float | None = None
    payload: dict[str, Any] | None = None
    reject_forbidden: bool = True


@dataclass
class VerificationResult:
    problem_id: str
    attempt_id: str
    problem_index: int
    attempt_index: int
    prompt: str
    generated_proof: str
    raw_completion: str
    lean_code: str
    imports: tuple[str, ...]
    context_lines: tuple[str, ...]
    worker_id: int | None
    priority: int
    status: str
    success: bool
    diagnostics: str
    generation_seconds: float | None = None
    queue_wait_seconds: float | None = None
    verification_seconds: float | None = None
    total_seconds: float | None = None
    verifier_backend: str = ""
    rejected_reason: str | None = None
    timed_out: bool = False
    compile_messages: tuple[str, ...] = ()
    compile_errors: tuple[str, ...] = ()
    compile_warnings: tuple[str, ...] = ()
    payload: dict[str, Any] | None = None

    def to_json(self) -> dict[str, Any]:
        data = {
            "problem_id": self.problem_id,
            "attempt_id": self.attempt_id,
            "problem_index": self.problem_index,
            "attempt_index": self.attempt_index,
            "prompt": self.prompt,
            "generated_proof": self.generated_proof,
            "raw_completion": self.raw_completion,
            "lean_code": self.lean_code,
            "imports": list(self.imports),
            "context_lines": list(self.context_lines),
            "worker_id": self.worker_id,
            "priority": self.priority,
            "status": self.status,
            "success": self.success,
            "diagnostics": self.diagnostics,
            "generation_seconds": self.generation_seconds,
            "queue_wait_seconds": self.queue_wait_seconds,
            "verification_seconds": self.verification_seconds,
            "total_seconds": self.total_seconds,
            "verifier_backend": self.verifier_backend,
            "rejected_reason": self.rejected_reason,
            "timed_out": self.timed_out,
            "compile_messages": list(self.compile_messages),
            "compile_errors": list(self.compile_errors),
            "compile_warnings": list(self.compile_warnings),
            "has_compile_errors": bool(self.compile_errors),
            "has_compile_warnings": bool(self.compile_warnings),
        }
        if self.payload:
            data.update(self.payload)
        return data


@dataclass
class VerificationWarmupReport:
    worker_id: int
    backend: str
    enabled: bool
    success: bool
    post_startup_warmup_seconds: float
    diagnostics: str
    server_startup_seconds: float | None = None
    timed_out: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "worker_id": self.worker_id,
            "backend": self.backend,
            "enabled": self.enabled,
            "success": self.success,
            "post_startup_warmup_seconds": self.post_startup_warmup_seconds,
            "diagnostics": self.diagnostics,
            "server_startup_seconds": self.server_startup_seconds,
            "timed_out": self.timed_out,
        }


def make_verification_result(
    task: VerificationTask,
    *,
    worker_id: int | None,
    status: str,
    success: bool,
    diagnostics: str,
    verifier_backend: str,
    generation_seconds: float | None = None,
    verification_seconds: float | None = None,
    queue_wait_seconds: float | None = None,
    total_seconds: float | None = None,
    timed_out: bool = False,
    rejected_reason: str | None = None,
    compile_messages: Iterable[str] = (),
    compile_errors: Iterable[str] = (),
    compile_warnings: Iterable[str] = (),
) -> VerificationResult:
    return VerificationResult(
        problem_id=task.problem_id,
        attempt_id=f"{task.problem_index},{task.attempt_index}",
        problem_index=task.problem_index,
        attempt_index=task.attempt_index,
        prompt=task.prompt,
        generated_proof=task.generated_proof,
        raw_completion=task.raw_completion,
        lean_code=task.lean_code,
        imports=task.imports,
        context_lines=task.context_lines,
        worker_id=worker_id,
        priority=task.priority,
        status=status,
        success=success,
        diagnostics=diagnostics,
        generation_seconds=(
            task.generation_seconds if generation_seconds is None else generation_seconds
        ),
        queue_wait_seconds=queue_wait_seconds,
        verification_seconds=verification_seconds,
        total_seconds=total_seconds,
        verifier_backend=verifier_backend,
        rejected_reason=rejected_reason,
        timed_out=timed_out,
        compile_messages=tuple(compile_messages),
        compile_errors=tuple(compile_errors),
        compile_warnings=tuple(compile_warnings),
        payload=task.payload,
    )
