"""Discovery verification adapter over the persistent Pantograph pool."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Iterable

from lean_prover.lean_training.data.preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
    compose_lean_theorem,
    lean_code_tokens,
    normalize_proof_for_assembly,
)
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)
from lean_prover.lean_training.verification.cache import VerificationCache
from lean_prover.lean_training.verification.pantograph import build_labeled_lean_code
from lean_prover.lean_training.verification.schema import VerificationTask

from .config import VerificationConfig
from .schemas import (
    DataRole,
    GenerationRecord,
    StatementRecord,
    VerificationRecord,
    VerificationStatus,
    normalized_proof,
    stable_hash,
)
from .utils import append_jsonl, environment_identity, iter_jsonl


def _remove_preloaded_import_commands(source: str) -> str:
    """Strip commands already loaded by the Pantograph worker environment."""

    return "".join(
        line
        for line in source.splitlines(keepends=True)
        if not re.match(
            r"^\s*(?:(?:public|private|protected|meta)\s+)*import\s+\S+",
            line,
        )
        and not re.match(r"^\s*module\s*$", line)
    ).lstrip("\n")


def verify_candidates(
    generations: Iterable[GenerationRecord],
    statements: dict[str, StatementRecord],
    *,
    config: VerificationConfig,
    output_path: str | Path,
    cache_path: str | Path,
    force: bool = False,
    verification_pool: VerificationPool | None = None,
    collect_results: bool = True,
) -> list[VerificationRecord]:
    """Stream resumable verification through bounded batches and SQLite cache."""

    output = Path(output_path)
    if force and output.exists():
        output.unlink()
    existing_ids = {
        str(row["generation_id"])
        for row in iter_jsonl(output)
        if row.get("generation_id")
    }
    collected = (
        [VerificationRecord.model_validate(row) for row in iter_jsonl(output)]
        if collect_results
        else []
    )
    environment = environment_identity(config.lean_project_path, config.imports)
    cache = VerificationCache(cache_path) if config.use_cache else None
    tasks: list[VerificationTask] = []
    task_generations: dict[str, GenerationRecord] = {}
    owned_pool: VerificationPool | None = None

    def persist(record: VerificationRecord) -> None:
        if record.generation_id in existing_ids:
            return
        append_jsonl(output, (record.model_dump(mode="json"),))
        existing_ids.add(record.generation_id)
        if collect_results:
            collected.append(record)

    def flush_tasks() -> None:
        nonlocal owned_pool
        if not tasks:
            return
        pool = verification_pool
        if pool is None:
            if owned_pool is None:
                owned_pool = VerificationPool(_pool_config(config, output))
                owned_pool.start()
            pool = owned_pool
        pool_run = pool.run_batch(tasks)
        if pool_run.fatal_errors:
            raise RuntimeError(f"discovery verification failed: {pool_run.fatal_errors}")
        for result in pool_run.results:
            generation_id = str(result["generation_id"])
            generation = task_generations[generation_id]
            statement = statements[generation.statement_id]
            status = _status_from_result(result)
            success = bool(result.get("success"))
            errors = list(result.get("compile_errors") or [])
            source = result.get("assembled_source")
            record = VerificationRecord(
                generation_id=generation_id,
                statement_id=generation.statement_id,
                data_role=DataRole.DISCOVERY,
                iteration=generation.iteration,
                verified=success,
                status=status,
                error_type=None if success else status.value,
                error_message=errors[0] if errors else str(result.get("diagnostics") or "") or None,
                compile_time_ms=round(float(result.get("verification_seconds") or 0) * 1000),
                timed_out=bool(result.get("timed_out")),
                environment_hash=str(environment["environment_hash"]),
                lean_version=str(environment["lean_version"]),
                mathlib_commit=environment["mathlib_commit"],
                assembler_version=ASSEMBLER_VERSION,
                normalization_version=NORMALIZATION_VERSION,
                statement=None if success and config.save_full_source_on_failure_only else statement.statement,
                normalized_proof=str(result.get("normalized_proof") or ""),
                proof_format=str(result.get("proof_format") or generation.proof_format or "full_proof"),
                assembled_source=(None if success and config.save_full_source_on_failure_only else str(source or "")),
                assembled_source_hash=str(result.get("assembled_source_hash") or ""),
                imports=[] if success and config.save_full_source_on_failure_only else list(statement.imports or config.imports),
                namespace=None if success and config.save_full_source_on_failure_only else statement.namespace,
                context=None if success and config.save_full_source_on_failure_only else statement.context,
                context_lines=[] if success and config.save_full_source_on_failure_only else list(statement.context_lines),
                metadata={
                    "worker_id": result.get("worker_id"),
                    "worker_pid": result.get("worker_pid"),
                    "worker_generation": result.get("worker_generation"),
                    "worker_restart_count": result.get("worker_restart_count"),
                    "pantograph_restart_count": result.get("pantograph_restart_count"),
                    "compile_messages": [] if success else result.get("compile_messages") or [],
                    "compile_errors": errors,
                    "compile_warnings": [] if success else result.get("compile_warnings") or [],
                    "cache_hit": False,
                    "cache_key": result.get("cache_key"),
                    "source_path": None if success else result.get("source_path"),
                },
            )
            persist(record)
            if cache is not None:
                cache.put(
                    cache_key=str(result["cache_key"]),
                    proof_hash=stable_hash(record.normalized_proof or ""),
                    statement_hash=statement.statement_hash,
                    environment_hash=str(environment["environment_hash"]),
                    assembler_version=ASSEMBLER_VERSION,
                    normalization_version=NORMALIZATION_VERSION,
                    status=record.status.value,
                    error_type=record.error_type,
                    source_path=record.metadata.get("source_path"),
                    verification=record.model_dump(mode="json"),
                )
        tasks.clear()
        task_generations.clear()

    try:
        for problem_index, generation in enumerate(generations):
            if generation.generation_id in existing_ids:
                continue
            statement = statements[generation.statement_id]
            prechecked = _precheck(generation, statement, environment, config, problem_index=problem_index)
            if prechecked is not None:
                persist(prechecked)
                continue
            proof = generation.extracted_proof or ""
            normalized, proof_format = normalize_proof_for_assembly(proof, proof_format=generation.proof_format)
            cache_key = stable_hash(
                statement.statement_hash, normalized_proof(normalized),
                "\n".join(statement.imports or config.imports), statement.namespace or "",
                statement.context or "", "\n".join(statement.context_lines),
                environment["environment_hash"],
            )
            cached_row = (
                cache.get(
                    cache_key,
                    environment_hash=str(environment["environment_hash"]),
                    assembler_version=ASSEMBLER_VERSION,
                    normalization_version=NORMALIZATION_VERSION,
                )
                if cache is not None and not force
                else None
            )
            if cached_row is not None:
                cached = VerificationRecord.model_validate(cached_row).model_copy(
                    update={
                        "generation_id": generation.generation_id,
                        "statement_id": generation.statement_id,
                        "iteration": generation.iteration,
                        "metadata": {**cached_row.get("metadata", {}), "cache_hit": True},
                    }
                )
                persist(cached)
                continue
            source_template = str(
                statement.metadata.get("preassembled_source_template") or ""
            )
            source_placeholder = str(
                statement.metadata.get("preassembled_source_placeholder")
                or "__CODEX_GENERATED_PROOF__"
            )
            if source_template:
                source_template = _remove_preloaded_import_commands(source_template)
                if source_template.count(source_placeholder) != 1:
                    persist(
                        _failed_precheck(
                            generation,
                            environment,
                            VerificationStatus.EXTRACTION_ERROR,
                            "source-faithful template must contain exactly one proof placeholder",
                            statement=statement,
                            config=config,
                            problem_index=problem_index,
                        )
                    )
                    continue
                lean_code = source_template.replace(source_placeholder, normalized, 1)
            else:
                try:
                    lean_code = compose_lean_theorem(
                        statement.statement, normalized, proof_format=proof_format
                    )
                except ValueError as error:
                    persist(
                        _failed_precheck(
                            generation,
                            environment,
                            VerificationStatus.EXTRACTION_ERROR,
                            str(error),
                            statement=statement,
                            config=config,
                            problem_index=problem_index,
                        )
                    )
                    continue
            payload = {
                "generation_id": generation.generation_id,
                "cache_key": cache_key,
                "statement": statement.statement,
                "normalized_proof": normalized,
                "proof_format": proof_format.value,
                "assembler_version": ASSEMBLER_VERSION,
                "normalization_version": NORMALIZATION_VERSION,
                "namespace": statement.namespace,
                "context": statement.context,
                "preassembled_source": bool(source_template),
            }
            task = VerificationTask(
                priority=problem_index, problem_index=problem_index,
                attempt_index=generation.sample_index, problem_id=statement.statement_id,
                prompt=generation.prompt or "", generated_proof=proof,
                raw_completion=generation.raw_output, lean_code=lean_code,
                imports=tuple(
                    config.imports
                    if source_template
                    else statement.imports or config.imports
                ),
                context_lines=tuple(statement.context_lines), payload=payload,
                reject_forbidden=False,
            )
            assembled_source = (
                lean_code
                if source_template
                else build_labeled_lean_code(task, include_imports=True)
            )
            payload["assembled_source"] = assembled_source
            payload["assembled_source_hash"] = stable_hash(assembled_source)
            tasks.append(task)
            task_generations[generation.generation_id] = generation
            if len(tasks) >= config.queue_maxsize:
                flush_tasks()
        flush_tasks()
    finally:
        if owned_pool is not None:
            owned_pool.close()
        if cache is not None:
            cache.close()
    return sorted(collected, key=lambda item: item.generation_id)


def _pool_config(config: VerificationConfig, output: Path) -> VerificationPoolConfig:
    return VerificationPoolConfig(
        lean_project_path=config.lean_project_path,
        imports=tuple(config.imports),
        timeout=config.timeout_seconds,
        warmup_timeout=config.warmup_timeout_seconds,
        num_workers=config.num_workers,
        queue_maxsize=config.queue_maxsize,
        heartbeat_interval=config.heartbeat_interval_seconds,
        heartbeat_timeout=config.heartbeat_timeout_seconds,
        max_worker_restarts=config.max_worker_restarts,
        max_task_retries=config.max_task_retries,
        shutdown_timeout=config.shutdown_timeout_seconds,
        task_spool_dir=str(output.parent / "verification_tasks"),
        save_full_source_on_failure_only=config.save_full_source_on_failure_only,
    )


def _precheck(
    generation: GenerationRecord,
    statement: StatementRecord,
    environment: dict[str, str | None],
    config: VerificationConfig,
    *,
    problem_index: int,
) -> VerificationRecord | None:
    proof = (generation.extracted_proof or "").strip()
    if not proof:
        return _failed_precheck(
            generation,
            environment,
            VerificationStatus.EXTRACTION_ERROR,
            "empty extracted proof",
            statement=statement,
            config=config,
            problem_index=problem_index,
        )
    tokens = set(lean_code_tokens(proof))
    forbidden_tokens: set[str] = set()
    if config.reject_sorry:
        forbidden_tokens.update({"sorry", "sorryAx"})
    if config.reject_admit:
        forbidden_tokens.add("admit")
    if config.reject_axiom:
        forbidden_tokens.add("axiom")
    forbidden = tokens & forbidden_tokens
    if forbidden:
        record = _failed_precheck(
            generation,
            environment,
            VerificationStatus.FORBIDDEN_TOKEN,
            f"forbidden Lean token(s): {sorted(forbidden)}",
            statement=statement,
            config=config,
            problem_index=problem_index,
        )
        record.contains_sorry = bool(tokens & {"sorry", "sorryAx"})
        record.contains_admit = "admit" in tokens
        record.contains_axiom = "axiom" in tokens
        return record
    return None


def _failed_precheck(
    generation: GenerationRecord,
    environment: dict[str, str | None],
    status: VerificationStatus,
    message: str,
    *,
    statement: StatementRecord | None = None,
    config: VerificationConfig | None = None,
    problem_index: int = 0,
) -> VerificationRecord:
    normalized = None
    proof_format = None
    assembled_source = None
    proof = (generation.extracted_proof or "").strip()
    if proof:
        normalized, resolved_format = normalize_proof_for_assembly(
            proof,
            proof_format=generation.proof_format,
        )
        proof_format = resolved_format.value
        if statement is not None and config is not None:
            try:
                lean_code = compose_lean_theorem(
                    statement.statement,
                    normalized,
                    proof_format=resolved_format,
                )
                diagnostic_task = VerificationTask(
                    priority=problem_index,
                    problem_index=problem_index,
                    attempt_index=generation.sample_index,
                    problem_id=statement.statement_id,
                    prompt=generation.prompt or "",
                    generated_proof=normalized,
                    raw_completion=generation.raw_output,
                    lean_code=lean_code,
                    imports=tuple(statement.imports or config.imports),
                    context_lines=tuple(statement.context_lines),
                )
                assembled_source = build_labeled_lean_code(
                    diagnostic_task,
                    include_imports=True,
                )
            except ValueError:
                assembled_source = None
    return VerificationRecord(
        generation_id=generation.generation_id,
        statement_id=generation.statement_id,
        data_role=DataRole.DISCOVERY,
        iteration=generation.iteration,
        verified=False,
        status=status,
        error_type=status.value,
        error_message=message,
        environment_hash=str(environment["environment_hash"]),
        lean_version=str(environment["lean_version"]),
        mathlib_commit=environment["mathlib_commit"],
        assembler_version=str(environment.get("assembler_version") or ASSEMBLER_VERSION),
        normalization_version=str(
            environment.get("normalization_version") or NORMALIZATION_VERSION
        ),
        statement=statement.statement if statement is not None else None,
        normalized_proof=normalized,
        proof_format=proof_format,
        assembled_source=assembled_source,
        assembled_source_hash=(
            stable_hash(assembled_source) if assembled_source is not None else None
        ),
        imports=list(
            (statement.imports or config.imports)
            if statement is not None and config is not None
            else []
        ),
        namespace=statement.namespace if statement is not None else None,
        context=statement.context if statement is not None else None,
        context_lines=list(statement.context_lines) if statement is not None else [],
    )


def _status_from_result(result: dict) -> VerificationStatus:
    from lean_prover.Data.compile_errors import classify_lean_diagnostics

    message = " ".join(
        str(item) for item in result.get("compile_errors") or []
    )
    classification = classify_lean_diagnostics(
        message,
        success=bool(result.get("success")),
        timed_out=bool(result.get("timed_out")),
    )
    return VerificationStatus(classification.status.value)
