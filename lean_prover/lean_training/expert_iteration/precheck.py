"""Fail-fast, non-training checks required before expert iteration."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Iterable

from lean_prover.lean_training.data.preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
    ProofFormat,
    compose_lean_theorem,
    normalize_proof_for_assembly,
)
from lean_prover.lean_training.verification.pantograph import build_labeled_lean_code
from lean_prover.lean_training.verification.schema import VerificationTask

from .schemas import DataRole, StatementRecord, stable_hash
from .utils import environment_identity


FIXED_SMOKE_CASES = (
    ("norm_num", "example : (1 : Nat) + 1 = 2", "norm_num"),
    ("simp", "example (x : Nat) : x = x", "simp"),
    ("linarith", "example (x y : Rat) (h : x ≤ y) : x - y ≤ 0", "linarith"),
)


def run_direct_lean_smoke(
    lean_project_path: str | Path,
    output_dir: str | Path,
    *,
    imports: Iterable[str],
) -> dict[str, Any]:
    """Compile the fixed smoke source with the same Lake project used by workers."""

    project = Path(lean_project_path).expanduser().resolve()
    # The Lean command runs with ``cwd=project`` so every generated source
    # passed to it must be absolute.  Keeping a caller-relative path here makes
    # an existing file appear missing as soon as the subprocess changes cwd.
    destination = Path(output_dir).expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=True)
    source_path = destination / "lean_smoke.lean"
    source = "\n".join(f"import {module}" for module in imports) + "\n\n"
    source += "\n\n".join(
        f"{statement} := by\n  {proof}" for _, statement, proof in FIXED_SMOKE_CASES
    )
    source_path.write_text(source + "\n", encoding="utf-8")
    command = ["lake", "env", "lean", str(source_path)]
    try:
        completed = subprocess.run(
            command,
            cwd=project,
            capture_output=True,
            text=True,
            timeout=180,
            check=False,
        )
        return_code = completed.returncode
        stdout = completed.stdout
        stderr = completed.stderr
    except (OSError, subprocess.TimeoutExpired) as error:
        return_code = -1
        stdout = ""
        stderr = f"{type(error).__name__}: {error}"
    identity = environment_identity(project, imports)
    return {
        "success": return_code == 0,
        "passed": 3 if return_code == 0 else 0,
        "total": 3,
        "command": command,
        "working_directory": str(project),
        "lean_path": os.environ.get("LEAN_PATH"),
        "stdout": stdout,
        "stderr": stderr,
        "return_code": return_code,
        "source_path": str(source_path),
        "source": source,
        **identity,
    }


def build_fixed_smoke_tasks(imports: Iterable[str]) -> list[VerificationTask]:
    tasks: list[VerificationTask] = []
    normalized_imports = tuple(imports)
    for index, (name, statement, proof) in enumerate(FIXED_SMOKE_CASES):
        lean_code = compose_lean_theorem(
            statement,
            proof,
            proof_format=ProofFormat.TACTIC_BODY,
        )
        task = VerificationTask(
            priority=index,
            problem_index=index,
            attempt_index=0,
            problem_id=f"precheck-{name}",
            prompt="",
            generated_proof=proof,
            raw_completion=proof,
            lean_code=lean_code,
            imports=normalized_imports,
            payload={"precheck_case": name},
            reject_forbidden=True,
        )
        tasks.append(task)
    return tasks


def select_reference_records(
    datasets: dict[DataRole, list[StatementRecord]],
    *,
    train_samples: int,
    eval_samples: int,
    benchmark_samples: int,
) -> tuple[list[StatementRecord], list[str]]:
    """Select disjoint known proofs, documenting benchmark-proof fallbacks."""

    notes: list[str] = []

    def with_proofs(role: DataRole) -> list[StatementRecord]:
        return [row for row in datasets.get(role, []) if row.reference_proof]

    train = with_proofs(DataRole.TRAIN)
    evaluation = with_proofs(DataRole.EVAL)
    benchmark: list[StatementRecord] = []
    for role in (DataRole.BENCHMARK, DataRole.BENCHMARK_DEV, DataRole.BENCHMARK_TEST):
        benchmark.extend(with_proofs(role))
    selected = train[:train_samples] + evaluation[:eval_samples]
    if len(benchmark) >= benchmark_samples:
        selected.extend(benchmark[:benchmark_samples])
    else:
        selected.extend(benchmark)
        missing = benchmark_samples - len(benchmark)
        extras = train[train_samples : train_samples + missing]
        selected.extend(extras)
        notes.append(
            "benchmark rows contain no usable reference proofs; used disjoint "
            f"additional train references for {len(extras)}/{missing} slots"
        )
    expected = train_samples + eval_samples + benchmark_samples
    if len(selected) != expected:
        notes.append(f"only {len(selected)}/{expected} requested references are available")
    return selected, notes


def build_reference_tasks(
    records: Iterable[StatementRecord],
    *,
    default_imports: Iterable[str],
) -> list[VerificationTask]:
    tasks: list[VerificationTask] = []
    for index, record in enumerate(records):
        proof = record.reference_proof or ""
        normalized, proof_format = normalize_proof_for_assembly(proof)
        lean_code = compose_lean_theorem(
            record.statement,
            normalized,
            proof_format=proof_format,
        )
        imports = tuple(record.imports or default_imports)
        payload = {
            "statement_id": record.statement_id,
            "data_role": record.data_role.value,
            "statement": record.statement,
            "normalized_proof": normalized,
            "proof_format": proof_format.value,
            "assembler_version": ASSEMBLER_VERSION,
            "normalization_version": NORMALIZATION_VERSION,
        }
        task = VerificationTask(
            priority=index,
            problem_index=index,
            attempt_index=0,
            problem_id=record.statement_id,
            prompt=record.generation_prompt(),
            generated_proof=normalized,
            raw_completion=proof,
            lean_code=lean_code,
            imports=imports,
            context_lines=tuple(record.context_lines),
            payload=payload,
            reject_forbidden=True,
        )
        assembled = build_labeled_lean_code(task, include_imports=True)
        payload["assembled_source"] = assembled
        payload["assembled_source_hash"] = stable_hash(assembled)
        tasks.append(task)
    return tasks


def summarize_pool_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    passed = sum(bool(row.get("success")) for row in results)
    return {
        "success": passed == len(results) and bool(results),
        "passed": passed,
        "total": len(results),
        "pass_rate": passed / max(1, len(results)),
        "results": results,
    }
