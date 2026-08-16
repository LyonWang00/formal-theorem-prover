"""Pantograph-backed theorem compilation checks for SFT benchmark."""

from __future__ import annotations

import time
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from lean_prover.backends.pantograph_backend import (
    PantographBackend,
    PantographOptions,
    PantographUnavailableError,
)
from lean_prover.lean_training.prepare_datasets import (
    build_lean_source_with_preamble,
    contains_forbidden_proof_token,
)
from lean_prover.lean_training.verification_schema import (
    VerificationTask,
    VerificationWarmupReport,
)


@dataclass
class PantographCheckResult:
    success: bool
    diagnostics: str
    check_seconds: float
    timed_out: bool = False
    error_type: str | None = None
    messages: tuple[str, ...] = ()
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def to_json(self) -> dict[str, Any]:
        return {
            "success": self.success,
            "diagnostics": self.diagnostics,
            "check_seconds": self.check_seconds,
            "timed_out": self.timed_out,
            "error_type": self.error_type,
            "messages": list(self.messages),
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


class PantographTheoremVerifier:
    """Keep one Pantograph server alive and check completed Lean declarations."""

    backend_name = "pantograph"

    def __init__(
        self,
        project_path: str | Path,
        *,
        imports: Sequence[str] = ("Mathlib",),
        timeout: int = 120,
        core_options: Sequence[str] = (),
        server_options: dict[str, Any] | None = None,
    ) -> None:
        self.project_path = Path(project_path).expanduser().resolve()
        self.imports = tuple(imports)
        self.timeout = timeout
        self.server_startup_seconds = 0.0
        start = time.monotonic()
        self.backend = PantographBackend(
            PantographOptions(
                project_path=self.project_path,
                imports=self.imports,
                timeout=timeout,
                core_options=tuple(core_options),
                server_options=server_options,
            )
        )
        self.server = self.backend.ensure_server()
        self.server_startup_seconds = round(time.monotonic() - start, 4)

    def warmup(self, timeout: int | None = None) -> PantographCheckResult:
        return self.check_source(
            "example : True := by\n  trivial",
            timeout=timeout,
            reject_forbidden=False,
        )

    def check_source(
        self,
        source: str,
        *,
        timeout: int | None = None,
        reject_forbidden: bool = True,
    ) -> PantographCheckResult:
        start = time.monotonic()
        if reject_forbidden and contains_forbidden_proof_token(source):
            return PantographCheckResult(
                success=False,
                diagnostics="rejected source containing sorry/admit",
                check_seconds=round(time.monotonic() - start, 4),
                error_type="forbidden_proof_token",
                errors=("rejected source containing sorry/admit",),
            )

        original_timeout = getattr(self.server, "timeout", None)
        if timeout is not None:
            self.server.timeout = timeout
        try:
            units = self.server.check_compile(source)
            raw_messages: list[Any] = []
            for unit in units:
                raw_messages.extend(getattr(unit, "messages", []))
            messages, errors, warnings = classify_pantograph_messages(raw_messages)
            diagnostics = "\n".join(messages)
            sorry_warnings = tuple(
                message
                for message in warnings
                if "sorry" in message.lower() or "sorryax" in message.lower()
            )
            errors = tuple(errors) + sorry_warnings
            return PantographCheckResult(
                success=not errors,
                diagnostics=diagnostics,
                check_seconds=round(time.monotonic() - start, 4),
                error_type=None if not errors else "lean_compilation",
                messages=tuple(messages),
                errors=errors,
                warnings=warnings,
            )
        except Exception as error:
            message = str(error)
            return PantographCheckResult(
                success=False,
                diagnostics=message,
                check_seconds=round(time.monotonic() - start, 4),
                timed_out="timeout" in message.lower(),
                error_type="pantograph_error",
                errors=(message,),
            )
        finally:
            if timeout is not None and original_timeout is not None:
                self.server.timeout = original_timeout

    def close(self) -> None:
        self.backend.close()


def require_pantograph(project_path: str | Path) -> None:
    try:
        PantographTheoremVerifier(project_path, imports=("Init",), timeout=30).close()
    except PantographUnavailableError:
        raise


def classify_pantograph_messages(
    raw_messages: Sequence[Any],
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    messages: list[str] = []
    errors: list[str] = []
    warnings: list[str] = []
    for raw in raw_messages:
        text = str(raw)
        messages.append(text)
        severity = _message_severity(raw)
        if severity == "error":
            errors.append(text)
        elif severity == "warning":
            warnings.append(text)
        elif severity is None:
            lowered = text.lower()
            if re.search(r"\berror\s*:", lowered) or lowered.startswith("error"):
                errors.append(text)
            elif re.search(r"\bwarning\s*:", lowered) or lowered.startswith("warning"):
                warnings.append(text)
    return tuple(messages), tuple(errors), tuple(warnings)


def _message_severity(message: Any) -> str | None:
    for field in ("severity", "level", "kind", "type"):
        value = getattr(message, field, None)
        if value is None and isinstance(message, dict):
            value = message.get(field)
        if value is None:
            continue
        lowered = str(value).lower()
        if "error" in lowered:
            return "error"
        if "warning" in lowered or "warn" in lowered:
            return "warning"
        if "info" in lowered:
            return "info"
    return None


class PantographTaskVerifier:
    """Single long-lived Pantograph worker for verification tasks."""

    backend_name = "pantograph"

    def __init__(
        self,
        project_path: str | Path,
        *,
        timeout: int,
        default_imports: Sequence[str],
        worker_id: int,
    ) -> None:
        self.project_path = Path(project_path).resolve()
        self.timeout = timeout
        self.default_imports = tuple(default_imports)
        self.worker_id = worker_id
        self.verifier = PantographTheoremVerifier(
            self.project_path,
            imports=self.default_imports,
            timeout=timeout,
        )
        self.server_startup_seconds = self.verifier.server_startup_seconds
        self.restart_count = 0

    def warmup(self, timeout: int) -> VerificationWarmupReport:
        result = self.verifier.warmup(timeout=timeout)
        return VerificationWarmupReport(
            worker_id=self.worker_id,
            backend=self.backend_name,
            enabled=True,
            success=result.success,
            post_startup_warmup_seconds=result.check_seconds,
            diagnostics=result.diagnostics,
            server_startup_seconds=self.server_startup_seconds,
            timed_out=result.timed_out,
        )

    def verify(self, task: VerificationTask) -> dict[str, Any]:
        server_imports = tuple(self.default_imports)
        task_imports = tuple(task.imports)
        if not imports_are_covered(task_imports, server_imports):
            message = (
                "task imports differ from the active Pantograph server imports: "
                f"task={task_imports}, server={server_imports}"
            )
            return {
                "success": False,
                "diagnostics": message,
                "verification_seconds": 0.0,
                "timed_out": False,
                "messages": (message,),
                "errors": (message,),
                "warnings": (),
            }
        code = build_labeled_lean_code(task, include_imports=False)
        result = self.verifier.check_source(
            code,
            timeout=self.timeout,
            reject_forbidden=task.reject_forbidden,
        )
        if self._should_restart(result):
            restart_note = self._restart_after_failure()
            retry = self.verifier.check_source(
                code,
                timeout=self.timeout,
                reject_forbidden=task.reject_forbidden,
            )
            if retry.diagnostics:
                retry.diagnostics = f"{restart_note}\n{retry.diagnostics}"
            else:
                retry.diagnostics = restart_note
            result = retry
        return {
            "success": result.success,
            "diagnostics": result.diagnostics,
            "verification_seconds": result.check_seconds,
            "timed_out": result.timed_out,
            "messages": result.messages,
            "errors": result.errors,
            "warnings": result.warnings,
        }

    @staticmethod
    def _should_restart(result: PantographCheckResult) -> bool:
        if result.error_type != "pantograph_error":
            return False
        diagnostics = result.diagnostics.lower()
        restart_markers = (
            "server not running",
            "connection reset",
            "connection refused",
            "broken pipe",
            "transport endpoint",
            "server closed",
        )
        return any(marker in diagnostics for marker in restart_markers)

    def _restart_after_failure(self) -> str:
        self.restart_count += 1
        try:
            self.verifier.close()
        except Exception:
            pass
        start = time.monotonic()
        self.verifier = PantographTheoremVerifier(
            self.project_path,
            imports=self.default_imports,
            timeout=self.timeout,
        )
        restart_seconds = round(time.monotonic() - start, 4)
        self.server_startup_seconds = self.verifier.server_startup_seconds
        warmup = self.verifier.warmup(timeout=self.timeout)
        return (
            f"restarted Pantograph worker {self.worker_id} "
            f"(restart_count={self.restart_count}, "
            f"restart_seconds={restart_seconds}, "
            f"warmup_success={warmup.success}, "
            f"warmup_seconds={warmup.check_seconds})"
        )

    def close(self) -> None:
        self.verifier.close()


def build_labeled_lean_code(
    task: VerificationTask,
    *,
    include_imports: bool = True,
) -> str:
    import_block = "\n".join(f"import {module}" for module in task.imports)
    namespace = f"EvalProblem_{task.problem_index}_Attempt_{task.attempt_index}"
    source = build_lean_source_with_preamble(
        task.lean_code,
        context_lines=task.context_lines,
        namespace=namespace,
        label=f"({task.problem_index},{task.attempt_index})",
    )
    return (f"{import_block}\n\n" if include_imports else "") + source


def imports_are_covered(
    task_imports: Sequence[str],
    server_imports: Sequence[str],
) -> bool:
    server_set = set(server_imports)
    for module in task_imports:
        if module in server_set:
            continue
        if "Mathlib" in server_set and (
            module == "Mathlib" or module.startswith("Mathlib.")
        ):
            continue
        return False
    return True

