"""Role-aware monitor and final benchmark adapters over the existing pipeline."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Any, Protocol

from lean_prover.lean_training.evaluation.benchmark import run_pipeline
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)

from .config import ExpertIterationConfig
from .datasets import index_statements, load_statements
from .discovery_verifier import verify_candidates
from .isolated_stage import expose_environment_cuda_toolkit
from .isolated_stage import IsolatedStageRunner
from .schemas import DataRole, DiscoveryStatementState, GenerationRecord
from .utils import iter_jsonl, read_jsonl, write_json_atomic, write_jsonl_atomic
from lean_prover.lean_training.runtime import resolve_runtime_profile


class EvaluationRunner(Protocol):
    def run(
        self,
        *,
        role: str,
        dataset_path: str,
        base_model: str,
        adapter_path: str | None,
        output_dir: str,
        pass_k: list[int],
        samples_per_statement: int,
        seed: int,
    ) -> dict[str, Any]: ...


class BenchmarkPipelineAdapter:
    def __init__(
        self,
        config: ExpertIterationConfig,
        verification_pool: VerificationPool | None = None,
    ) -> None:
        self.config = config
        self.verification_pool = verification_pool

    def set_verification_pool(self, pool: VerificationPool) -> None:
        """Reuse the coordinator-owned Pantograph pool for every evaluation."""

        self.verification_pool = pool

    def run(
        self,
        *,
        role: str,
        dataset_path: str,
        base_model: str,
        adapter_path: str | None,
        output_dir: str,
        pass_k: list[int],
        samples_per_statement: int,
        seed: int,
    ) -> dict[str, Any]:
        if role not in {
            "monitor",
            "benchmark",
            "benchmark_dev",
            "benchmark_test",
            "discovery_replay",
        }:
            raise ValueError(f"unsupported generation-evaluation role: {role}")
        generation = self.config.discovery.generation
        verification = self.config.verification
        args = argparse.Namespace(
            model_name_or_path=base_model,
            adapter_path=adapter_path,
            benchmark_file=dataset_path,
            output_dir=output_dir,
            generation_backend=generation.backend,
            generation_batch_size=generation.batch_size,
            pass_k=max(max(pass_k), samples_per_statement),
            num_workers=verification.num_workers,
            queue_maxsize=verification.queue_maxsize,
            max_new_tokens=generation.max_new_tokens,
            temperature=generation.temperature,
            top_p=generation.top_p,
            lean_timeout=verification.timeout_seconds,
            warmup_timeout=verification.warmup_timeout_seconds,
            lean_project_path=verification.lean_project_path,
            imports=tuple(verification.imports),
            load_in_4bit=generation.load_in_4bit,
            vllm_max_model_len=generation.max_model_len,
            vllm_gpu_memory_utilization=generation.gpu_memory_utilization,
            vllm_enforce_eager=generation.enforce_eager,
            vllm_kv_cache_memory_bytes=generation.kv_cache_memory_bytes,
            generation_contract_path=generation.generation_contract_path,
            generation_contract_sha256=generation.generation_contract_sha256,
            num_benchmark_samples=None,
            generation_seed=seed,
            resume=True,
            force_resume=False,
            disable_warmup=False,
        )
        expose_environment_cuda_toolkit(os.environ)
        if resolve_runtime_profile(self.config.runtime).name == "laptop":
            summary = self._run_staged_laptop_evaluation(
                role=role,
                dataset_path=dataset_path,
                base_model=base_model,
                adapter_path=adapter_path,
                output_dir=output_dir,
                samples_per_statement=samples_per_statement,
                seed=seed,
            )
        else:
            summary = run_pipeline(args, verification_pool=self.verification_pool)
        pass_at = summary.get("pass_at") or {}
        renamed = {"data_role": role, **summary}
        for k in pass_k:
            renamed[f"{role}_pass_at_{k}"] = pass_at.get(f"pass@{k}")
        attempt_results = summary.get("recorded_attempt_results") or {}
        recorded_attempts = (
            sum(int(value) for value in attempt_results.values())
            if isinstance(attempt_results, dict)
            else int(attempt_results)
        )
        attempts = max(1, recorded_attempts)
        problems = max(1, int(summary.get("num_benchmark_samples") or 0))
        attempt_rows = read_jsonl(Path(output_dir) / "attempts.jsonl")
        if attempt_rows:
            attempts = len(attempt_rows)
            compile_successes = sum(bool(row.get("success")) for row in attempt_rows)
            parse_successes = sum(
                bool(str(row.get("generated_proof") or "").strip())
                for row in attempt_rows
            )
            timeouts = sum(bool(row.get("timed_out")) for row in attempt_rows)
        else:
            compile_successes = int(summary.get("successes") or 0)
            parse_successes = recorded_attempts
            timeouts = int(summary.get("timeout_attempts") or 0)
        renamed[f"{role}_problem_solve_rate"] = float(summary.get("successes") or 0) / problems
        renamed[f"{role}_parse_success_rate"] = parse_successes / attempts
        renamed[f"{role}_compile_success_rate"] = compile_successes / attempts
        renamed[f"{role}_timeout_rate"] = timeouts / attempts
        write_json_atomic(Path(output_dir) / f"{role}_metrics.json", renamed)
        return renamed

    def _run_staged_laptop_evaluation(
        self,
        *,
        role: str,
        dataset_path: str,
        base_model: str,
        adapter_path: str | None,
        output_dir: str,
        samples_per_statement: int,
        seed: int,
    ) -> dict[str, Any]:
        """Evaluate with no vLLM/Pantograph lifetime overlap on a laptop."""

        output = Path(output_dir)
        output.mkdir(parents=True, exist_ok=True)
        role_enum = {
            "monitor": DataRole.MONITOR,
            "discovery_replay": DataRole.DISCOVERY,
        }.get(role, DataRole.BENCHMARK)
        statements = load_statements(dataset_path, role_enum)
        states = {
            statement.statement_id: DiscoveryStatementState(
                statement_id=statement.statement_id
            )
            for statement in statements
        }
        generation_path = output / "generations.jsonl"
        verification_path = output / "verifications.jsonl"
        runner = IsolatedStageRunner(
            output / "runtime",
            flashinfer_sampler=self.config.execution.flashinfer_sampler,
        )
        generation_start = time.monotonic()
        try:
            generation_result = runner.run(
                "generation",
                {
                    "config": self.config.model_dump(mode="json"),
                    "base_model": base_model,
                    "adapter_path": adapter_path,
                    "selected": [
                        statement.model_dump(
                            mode="json", exclude={"reference_proof"}
                        )
                        for statement in statements
                    ],
                    "statement_states": {
                        key: value.model_dump(mode="json")
                        for key, value in states.items()
                    },
                    "iteration": -10,
                    "checkpoint": str(adapter_path or base_model),
                    "output_path": str(generation_path),
                    "force": False,
                    "sampling_budget": {
                        "new": samples_per_statement,
                        "frontier": samples_per_statement,
                        "unsolved": samples_per_statement,
                        "audit": samples_per_statement,
                    },
                    "generation_seed": seed,
                },
                timeout=self.config.execution.generation_timeout_seconds,
            )
        finally:
            runner.close()
        generation_seconds = round(time.monotonic() - generation_start, 4)

        generations = [
            GenerationRecord.model_validate(row) for row in iter_jsonl(generation_path)
        ]
        verification = self.config.verification
        pool = VerificationPool(
            VerificationPoolConfig(
                lean_project_path=verification.lean_project_path,
                imports=tuple(verification.imports),
                timeout=verification.timeout_seconds,
                warmup_timeout=verification.warmup_timeout_seconds,
                num_workers=verification.num_workers,
                queue_maxsize=verification.queue_maxsize,
                heartbeat_interval=verification.heartbeat_interval_seconds,
                heartbeat_timeout=verification.heartbeat_timeout_seconds,
                max_worker_restarts=verification.max_worker_restarts,
                max_task_retries=verification.max_task_retries,
                shutdown_timeout=verification.shutdown_timeout_seconds,
                task_spool_dir=str(output / "verification_tasks"),
                save_full_source_on_failure_only=(
                    verification.save_full_source_on_failure_only
                ),
            )
        )
        verification_start = time.monotonic()
        try:
            pool.start()
            verifications = verify_candidates(
                generations,
                index_statements(statements),
                config=verification,
                output_path=verification_path,
                cache_path=output / "verification_cache.sqlite",
                force=False,
                verification_pool=pool,
            )
            pool_runtime = pool.runtime_snapshot()
        finally:
            pool.close()
        verification_seconds = round(time.monotonic() - verification_start, 4)

        verification_by_generation = {
            row.generation_id: row for row in verifications
        }
        attempts = []
        successes_by_statement: dict[str, set[int]] = {
            statement.statement_id: set() for statement in statements
        }
        for generation in generations:
            result = verification_by_generation.get(generation.generation_id)
            success = bool(result and result.verified)
            if success:
                successes_by_statement[generation.statement_id].add(
                    generation.sample_index
                )
            attempts.append(
                {
                    "problem_id": generation.statement_id,
                    "attempt_index": generation.sample_index,
                    "generation_id": generation.generation_id,
                    "generated_proof": generation.extracted_proof,
                    "success": success,
                    "status": result.status.value if result else "missing",
                    "timed_out": bool(result and result.timed_out),
                    "cache_hit": bool(
                        result and result.metadata.get("cache_hit")
                    ),
                }
            )
        write_jsonl_atomic(output / "attempts.jsonl", attempts)
        pass_at = {
            f"pass@{k}": sum(
                bool(successes & set(range(k)))
                for successes in successes_by_statement.values()
            )
            / max(1, len(statements))
            for k in range(1, samples_per_statement + 1)
        }
        max_k = samples_per_statement
        solved = sum(bool(values) for values in successes_by_statement.values())
        worker_restarts = sum(
            int(worker.get("restart_count") or 0)
            for worker in pool_runtime.get("workers", [])
        )
        summary = {
            "model_name_or_path": base_model,
            "adapter_path": adapter_path,
            "generation_backend": generation_result.get("backend"),
            "verifier_backend": "pantograph",
            "execution_mode": "staged_sequential",
            "num_workers": verification.num_workers,
            "num_benchmark_samples": len(statements),
            "pass_k": max_k,
            "successes": solved,
            f"pass@{max_k}": pass_at[f"pass@{max_k}"],
            "pass_at": pass_at,
            "early_stop_on_success": False,
            "queued_attempts": len(generations),
            "rejected_attempts": sum(
                row.status.value in {"forbidden_token", "extraction_error"}
                for row in verifications
            ),
            "recorded_attempt_results": len(verifications),
            "attempts_with_compile_errors": sum(not row.verified for row in verifications),
            "timeout_attempts": sum(row.timed_out for row in verifications),
            "warmup_seconds": max(
                (
                    float(worker.get("server_startup_seconds") or 0)
                    for worker in pool_runtime.get("workers", [])
                ),
                default=0.0,
            ),
            "fatal_errors": list(pool.fatal_errors),
            "recovered_worker_failures": list(pool.recovered_worker_failures),
            "generation_seconds": generation_seconds,
            "verification_seconds": verification_seconds,
            "total_seconds": round(generation_seconds + verification_seconds, 4),
            "generation_subprocess_pid": generation_result.get("pid"),
            "pantograph_runtime": pool_runtime,
            "pantograph_worker_restart_count": worker_restarts,
            "cache_hits": sum(
                bool(row.metadata.get("cache_hit")) for row in verifications
            ),
            "cache_misses": sum(
                not bool(row.metadata.get("cache_hit")) for row in verifications
            ),
        }
        write_json_atomic(output / "benchmark_summary.json", summary)
        return summary
