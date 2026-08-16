#!/usr/bin/env python3
"""Run one paired ablation evaluation, including source-faithful LD holdout."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from lean_prover.lean_training.expert_iteration.config import (
    load_expert_iteration_config,
)
from lean_prover.lean_training.expert_iteration.datasets import (
    index_statements,
    load_statements,
)
from lean_prover.lean_training.expert_iteration.discovery_verifier import (
    verify_candidates,
)
from lean_prover.lean_training.expert_iteration.evaluation_adapter import (
    BenchmarkPipelineAdapter,
)
from lean_prover.lean_training.expert_iteration.isolated_stage import (
    IsolatedStageRunner,
    expose_environment_cuda_toolkit,
)
from lean_prover.lean_training.expert_iteration.schemas import (
    DataRole,
    DiscoveryStatementState,
    GenerationRecord,
    VerificationRecord,
)
from lean_prover.lean_training.expert_iteration.utils import (
    iter_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)


def _ld_source_faithful(
    *,
    config_path: str,
    model: str,
    adapter: str | None,
    dataset: str,
    output_dir: str,
    seed: int,
    samples: int,
) -> dict[str, Any]:
    config = load_expert_iteration_config(config_path)
    statements = load_statements(dataset, DataRole.BENCHMARK)
    if not statements:
        raise ValueError(f"LD holdout dataset is empty or unreadable: {dataset}")
    if not all(
        statement.metadata.get("preassembled_source_template")
        for statement in statements
    ):
        raise ValueError("LD source-faithful evaluation requires source templates")
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    generation_path = output / "generations.jsonl"
    verification_path = output / "verifications.jsonl"
    states = {
        statement.statement_id: DiscoveryStatementState(
            statement_id=statement.statement_id
        )
        for statement in statements
    }
    expose_environment_cuda_toolkit(os.environ)
    runner = IsolatedStageRunner(
        output / "runtime",
        flashinfer_sampler=config.execution.flashinfer_sampler,
    )
    generation_start = time.monotonic()
    try:
        generation_result = runner.run(
            "generation",
            {
                "config": config.model_dump(mode="json"),
                "base_model": model,
                "adapter_path": adapter,
                "selected": [
                    statement.model_dump(mode="json", exclude={"reference_proof"})
                    for statement in statements
                ],
                "statement_states": {
                    key: value.model_dump(mode="json") for key, value in states.items()
                },
                "iteration": -20,
                "checkpoint": str(adapter or model),
                "output_path": str(generation_path),
                "force": False,
                "sampling_budget": {
                    "new": samples,
                    "frontier": samples,
                    "unsolved": samples,
                    "audit": samples,
                },
                "generation_seed": seed,
            },
            timeout=config.execution.generation_timeout_seconds,
        )
    finally:
        runner.close()
    generation_seconds = round(time.monotonic() - generation_start, 4)
    generations = [
        GenerationRecord.model_validate(row) for row in iter_jsonl(generation_path)
    ]
    by_statement = index_statements(statements)
    generation_groups: dict[tuple[str, ...], list[GenerationRecord]] = defaultdict(list)
    for generation in generations:
        statement = by_statement[generation.statement_id]
        generation_groups[tuple(statement.imports)].append(generation)

    verification_start = time.monotonic()
    group_runtime: list[dict[str, Any]] = []
    for group_index, (imports, group_generations) in enumerate(
        sorted(generation_groups.items())
    ):
        group_config = config.verification.model_copy(
            update={"imports": list(imports)}
        )
        pool = VerificationPool(
            VerificationPoolConfig(
                lean_project_path=group_config.lean_project_path,
                imports=imports,
                timeout=group_config.timeout_seconds,
                warmup_timeout=group_config.warmup_timeout_seconds,
                num_workers=group_config.num_workers,
                queue_maxsize=group_config.queue_maxsize,
                heartbeat_interval=group_config.heartbeat_interval_seconds,
                heartbeat_timeout=group_config.heartbeat_timeout_seconds,
                max_worker_restarts=group_config.max_worker_restarts,
                max_task_retries=group_config.max_task_retries,
                shutdown_timeout=group_config.shutdown_timeout_seconds,
                task_spool_dir=str(
                    output / "verification_tasks" / f"group_{group_index:03d}"
                ),
                save_full_source_on_failure_only=(
                    group_config.save_full_source_on_failure_only
                ),
            )
        )
        try:
            pool.start()
            verify_candidates(
                group_generations,
                by_statement,
                config=group_config,
                output_path=verification_path,
                cache_path=output / "verification_cache.sqlite",
                force=False,
                verification_pool=pool,
            )
            group_runtime.append(
                {
                    "group_index": group_index,
                    "imports": list(imports),
                    "generation_count": len(group_generations),
                    "runtime": pool.runtime_snapshot(),
                }
            )
        finally:
            pool.close()
    verification_seconds = round(time.monotonic() - verification_start, 4)

    verifications = [
        VerificationRecord.model_validate(row) for row in iter_jsonl(verification_path)
    ]
    verification_by_generation = {
        row.generation_id: row for row in verifications
    }
    successes_by_statement: dict[str, set[int]] = {
        statement.statement_id: set() for statement in statements
    }
    attempts: list[dict[str, Any]] = []
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
                "source_id": by_statement[generation.statement_id].source_id,
                "attempt_index": generation.sample_index,
                "generation_id": generation.generation_id,
                "raw_output": generation.raw_output,
                "generated_proof": generation.extracted_proof,
                "finish_reason": generation.finish_reason,
                "success": success,
                "status": result.status.value if result else "missing",
                "timed_out": bool(result and result.timed_out),
                "cache_hit": bool(result and result.metadata.get("cache_hit")),
            }
        )
    write_jsonl_atomic(output / "attempts.jsonl", attempts)
    pass_at = {
        f"pass@{k}": sum(
            bool(successes & set(range(k)))
            for successes in successes_by_statement.values()
        )
        / max(1, len(statements))
        for k in range(1, samples + 1)
    }
    solved = sum(bool(value) for value in successes_by_statement.values())
    worker_restarts = sum(
        int(worker.get("restart_count") or 0)
        for group in group_runtime
        for worker in group["runtime"].get("workers", [])
    )
    summary = {
        "data_role": "ld_holdout",
        "model_name_or_path": model,
        "adapter_path": adapter,
        "generation_backend": generation_result.get("backend"),
        "verifier_backend": "pantograph",
        "execution_mode": "staged_sequential_grouped_source_faithful",
        "num_workers": config.verification.num_workers,
        "import_group_count": len(group_runtime),
        "num_benchmark_samples": len(statements),
        "queued_attempts": len(generations),
        "recorded_attempt_results": len(verifications),
        "successes": solved,
        "candidate_success_rate": sum(row["success"] for row in attempts)
        / max(1, len(attempts)),
        "pass_at": pass_at,
        "ld_holdout_pass_at_1": pass_at["pass@1"],
        "ld_holdout_pass_at_2": pass_at["pass@2"],
        "ld_holdout_pass_at_4": pass_at[f"pass@{samples}"],
        "generation_seconds": generation_seconds,
        "verification_seconds": verification_seconds,
        "total_seconds": round(generation_seconds + verification_seconds, 4),
        "generation_subprocess_pid": generation_result.get("pid"),
        "pantograph_import_groups": group_runtime,
        "pantograph_worker_restart_count": worker_restarts,
        "cache_hits": sum(row["cache_hit"] for row in attempts),
        "cache_misses": sum(not row["cache_hit"] for row in attempts),
    }
    write_json_atomic(output / "ld_holdout_metrics.json", summary)
    write_json_atomic(output / "benchmark_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--role",
        required=True,
        choices=("monitor", "benchmark", "discovery_replay", "ld_holdout"),
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", default=None)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--samples-per-statement", type=int, default=4)
    args = parser.parse_args()
    if args.role == "ld_holdout":
        result = _ld_source_faithful(
            config_path=args.config,
            model=args.model,
            adapter=args.adapter,
            dataset=args.dataset,
            output_dir=args.output,
            seed=args.seed,
            samples=args.samples_per_statement,
        )
    else:
        config = load_expert_iteration_config(args.config)
        result = BenchmarkPipelineAdapter(config).run(
            role=args.role,
            dataset_path=args.dataset,
            base_model=args.model,
            adapter_path=args.adapter,
            output_dir=args.output,
            pass_k=sorted({1, 2, args.samples_per_statement}),
            samples_per_statement=args.samples_per_statement,
            seed=args.seed,
        )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
