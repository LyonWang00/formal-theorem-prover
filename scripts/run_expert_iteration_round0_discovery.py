"""Run resumable H0 generation and Pantograph verification for EI Round 0."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

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


EXPECTED_H0_HASH = (
    "6b0ea36dcc8dfc71f9b85220797c29703660679dac214af368de084e7a176431"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/expert_iteration/round0"),
    )
    parser.add_argument(
        "--verification-workers",
        type=int,
        default=None,
        help="Resume-time Pantograph concurrency override; does not alter verification semantics.",
    )
    parser.add_argument(
        "--verification-queue-maxsize",
        type=int,
        default=None,
        help="Resume-time persistence batch size override.",
    )
    parser.add_argument(
        "--verification-order",
        choices=("manifest", "shortest_first"),
        default="manifest",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_status(root: Path, **changes: Any) -> None:
    path = root / "status.json"
    payload = (
        json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    )
    payload.update(changes)
    write_json_atomic(path, payload)


def failure_taxonomy(record: VerificationRecord) -> tuple[str, str]:
    status = record.status.value
    message = str(record.error_message or "").lower()
    if "unknown identifier" in message or "unknown constant" in message:
        return "repair_candidate", "unknown_identifier"
    if status in {"unsolved_goals", "tactic_error"}:
        return "near_miss", status
    if status in {"elaboration_error", "syntax_error"}:
        return "repair_candidate", status
    return "invalid", status


def repetition_ratio(text: str, n: int = 4) -> float:
    tokens = text.split()
    if len(tokens) < n:
        return 0.0
    grams = [tuple(tokens[index : index + n]) for index in range(len(tokens) - n + 1)]
    return 1.0 - len(set(grams)) / max(1, len(grams))


def main() -> None:
    args = parse_args()
    project = args.project.resolve()
    root = args.root if args.root.is_absolute() else project / args.root
    manifest = root / "discovery/discovery_manifest.jsonl"
    config_path = root / "discovery/runtime_config.json"
    contract_path = root / "discovery/generation_contract.json"
    if not all(path.is_file() for path in (manifest, config_path, contract_path)):
        raise FileNotFoundError("Round 0 discovery preparation is incomplete")
    config = load_expert_iteration_config(config_path)
    contract = json.loads(contract_path.read_text(encoding="utf-8"))
    if contract.get("checkpoint_model_sha256") != EXPECTED_H0_HASH:
        raise RuntimeError("H0 identity in resolved generation contract drifted")
    if sha256(Path(contract["checkpoint_path"]) / "model.safetensors") != EXPECTED_H0_HASH:
        raise RuntimeError("H0 checkpoint changed after discovery freeze")
    statements = load_statements(manifest, DataRole.DISCOVERY)
    expected_rows = len(statements)
    expected_candidates = expected_rows * 8
    if expected_rows <= 0:
        raise RuntimeError("discovery manifest is empty")
    if any(statement.reference_proof for statement in statements):
        raise RuntimeError("discovery loader recovered a forbidden reference proof")
    statement_by_id = index_statements(statements)
    states = {
        statement.statement_id: DiscoveryStatementState(
            statement_id=statement.statement_id
        )
        for statement in statements
    }

    generation_path = root / "discovery/candidate_generations.jsonl"
    verification_path = root / "discovery/candidate_verifications.jsonl"
    existing_generation_count = sum(1 for _ in iter_jsonl(generation_path))
    if existing_generation_count == expected_candidates:
        previous_status = json.loads(
            (root / "status.json").read_text(encoding="utf-8")
        )
        generation_result = {
            "backend": "transformers",
            "resumed_complete_generation": True,
        }
        generation_seconds = float(previous_status.get("generation_seconds") or 0.0)
    else:
        expose_environment_cuda_toolkit(os.environ)
        runner = IsolatedStageRunner(
            root / "runtime/discovery_generation",
            flashinfer_sampler=config.execution.flashinfer_sampler,
        )
        generation_started = time.monotonic()
        write_status(
            root,
            status="DISCOVERY_GENERATION_RUNNING",
            generation_started=True,
            trainer_started=False,
            grpo_started=False,
        )
        try:
            generation_result = runner.run(
                "generation",
                {
                    "config": config.model_dump(mode="json"),
                    "base_model": str(contract["checkpoint_path"]),
                    "adapter_path": None,
                    "selected": [
                        statement.model_dump(mode="json", exclude={"reference_proof"})
                        for statement in statements
                    ],
                    "statement_states": {
                        key: value.model_dump(mode="json")
                        for key, value in states.items()
                    },
                    "iteration": 0,
                    "checkpoint": f"H0-No-Hard:{EXPECTED_H0_HASH}",
                    "output_path": str(generation_path),
                    "force": False,
                    "sampling_budget": {
                        "new": 8,
                        "frontier": 8,
                        "unsolved": 8,
                        "audit": 8,
                    },
                    "generation_seed": int(
                        json.loads(
                            (root / "audit/discovery_audit.json").read_text(
                                encoding="utf-8"
                            )
                        )["candidate_seed"]
                    ),
                },
                timeout=config.execution.generation_timeout_seconds,
            )
        finally:
            runner.close()
        generation_seconds = round(time.monotonic() - generation_started, 4)
    generations = [
        GenerationRecord.model_validate(row) for row in iter_jsonl(generation_path)
    ]
    if len(generations) != expected_candidates:
        raise RuntimeError(
            f"generation incomplete: {len(generations)}/{expected_candidates}"
        )
    if len({row.generation_id for row in generations}) != expected_candidates:
        raise RuntimeError("duplicate generation IDs")
    per_statement = Counter(row.statement_id for row in generations)
    if set(per_statement.values()) != {8}:
        raise RuntimeError("not every discovery statement has eight candidates")
    if any(row.metadata.get("backend") != "transformers" for row in generations):
        raise RuntimeError("non-Transformers generation entered Round 0")
    if any(row.metadata.get("adapter_path") for row in generations):
        raise RuntimeError("adapter/unmerged generation entered Round 0")
    if any(
        row.metadata.get("generation_contract_sha256")
        != contract["generation_config_sha256"]
        for row in generations
    ):
        raise RuntimeError("generation contract hash mismatch in candidate records")
    write_status(
        root,
        status="DISCOVERY_GENERATION_COMPLETED",
        generated_candidates=len(generations),
        generation_seconds=generation_seconds,
    )

    worker_count = args.verification_workers or config.verification.num_workers
    queue_maxsize = (
        args.verification_queue_maxsize or config.verification.queue_maxsize
    )
    if worker_count <= 0:
        raise ValueError("verification worker count must be positive")
    if queue_maxsize <= 0:
        raise ValueError("verification queue size must be positive")
    verification_generations = list(generations)
    if args.verification_order == "shortest_first":
        verification_generations.sort(
            key=lambda row: (
                int(row.metadata.get("completion_tokens") or 10**9),
                row.statement_id,
                row.sample_index,
            )
        )
    existing_verifications = [
        VerificationRecord.model_validate(row)
        for row in iter_jsonl(verification_path)
    ]
    if len(existing_verifications) == expected_candidates:
        previous_summary_path = root / "discovery/discovery_summary.json"
        previous_summary = (
            json.loads(previous_summary_path.read_text(encoding="utf-8"))
            if previous_summary_path.exists()
            else {}
        )
        pool_runtimes: list[dict[str, Any]] = []
        verification_seconds = float(
            previous_summary.get("verification_seconds") or 0.0
        )
        verifications = existing_verifications
    else:
        verification_started = time.monotonic()
        write_status(root, status="DISCOVERY_VERIFICATION_RUNNING")
        existing_verification_ids = {
            row.generation_id for row in existing_verifications
        }
        generation_groups: dict[tuple[str, ...], list[GenerationRecord]] = defaultdict(list)
        for generation in verification_generations:
            statement = statement_by_id[generation.statement_id]
            imports = tuple(statement.imports or config.verification.imports)
            generation_groups[imports].append(generation)
        pool_runtimes = []
        for group_index, (imports, group_generations) in enumerate(
            sorted(generation_groups.items())
        ):
            pending_generations = [
                generation
                for generation in group_generations
                if generation.generation_id not in existing_verification_ids
            ]
            if not pending_generations:
                continue
            verify_config = config.verification.model_copy(
                update={
                    "imports": list(imports),
                    "num_workers": worker_count,
                    "queue_maxsize": queue_maxsize,
                }
            )
            pool = VerificationPool(
                VerificationPoolConfig(
                    lean_project_path=verify_config.lean_project_path,
                    imports=imports,
                    timeout=verify_config.timeout_seconds,
                    warmup_timeout=verify_config.warmup_timeout_seconds,
                    num_workers=verify_config.num_workers,
                    queue_maxsize=verify_config.queue_maxsize,
                    heartbeat_interval=verify_config.heartbeat_interval_seconds,
                    heartbeat_timeout=verify_config.heartbeat_timeout_seconds,
                    max_worker_restarts=verify_config.max_worker_restarts,
                    max_task_retries=verify_config.max_task_retries,
                    shutdown_timeout=verify_config.shutdown_timeout_seconds,
                    task_spool_dir=str(
                        root / "discovery/verification_tasks" / f"group_{group_index:03d}"
                    ),
                    save_full_source_on_failure_only=(
                        verify_config.save_full_source_on_failure_only
                    ),
                )
            )
            try:
                pool.start()
                verify_candidates(
                    pending_generations,
                    statement_by_id,
                    config=verify_config,
                    output_path=verification_path,
                    cache_path=root / "discovery/verification_cache.sqlite",
                    force=False,
                    verification_pool=pool,
                )
                pool_runtimes.append(
                    {
                        "group_index": group_index,
                        "imports": list(imports),
                        "generation_count": len(pending_generations),
                        "runtime": pool.runtime_snapshot(),
                    }
                )
            finally:
                pool.close()
        verification_seconds = round(time.monotonic() - verification_started, 4)
        verifications = [
            VerificationRecord.model_validate(row)
            for row in iter_jsonl(verification_path)
        ]
    if len(verifications) != expected_candidates:
        raise RuntimeError(
            f"verification incomplete: {len(verifications)}/{expected_candidates}"
        )
    verification_by_id = {row.generation_id: row for row in verifications}
    if len(verification_by_id) != expected_candidates:
        raise RuntimeError("duplicate verification records")
    ld_statement_ids = {
        row.statement_id
        for row in statements
        if str(row.source).lower().startswith("ld")
    }
    contaminated = [
        row.generation_id
        for row in verifications
        if row.statement_id in ld_statement_ids
        and "already been declared" in str(row.error_message or "").lower()
    ]
    if contaminated:
        raise RuntimeError(
            f"source-faithful LD verification contamination: {len(contaminated)} candidates"
        )

    successes_by_statement: dict[str, list[GenerationRecord]] = defaultdict(list)
    candidate_rows: list[dict[str, Any]] = []
    success_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    taxonomy = Counter()
    failure_layers = Counter()
    for generation in sorted(
        generations, key=lambda row: (row.statement_id, row.sample_index)
    ):
        statement_record = statement_by_id[generation.statement_id]
        verification = verification_by_id[generation.generation_id]
        verified = bool(verification.verified)
        if verified:
            successes_by_statement[generation.statement_id].append(generation)
            layer, failure_type = "success", "success"
        else:
            layer, failure_type = failure_taxonomy(verification)
            taxonomy[failure_type] += 1
            failure_layers[layer] += 1
        base = {
            "statement_id": generation.statement_id,
            "source_id": statement_record.source_id,
            "source": statement_record.source,
            "candidate_id": generation.generation_id,
            "candidate_rank": generation.sample_index + 1,
            "candidate_seed": generation.generation_seed,
            "checkpoint_hash": EXPECTED_H0_HASH,
            "generation_config_hash": contract["generation_config_sha256"],
            "generated_proof": generation.extracted_proof,
            "raw_output": generation.raw_output,
            "extraction_success": bool(generation.extracted_proof),
            "finish_reason": generation.finish_reason,
            "generation_length": generation.metadata.get("completion_tokens"),
            "pantograph_verified": verified,
            "pantograph_status": verification.status.value,
            "timed_out": verification.timed_out,
            "failure_layer": layer,
            "failure_taxonomy": failure_type,
            "error_message": verification.error_message,
            "compile_time_ms": verification.compile_time_ms,
            "repeated_4gram_ratio": repetition_ratio(generation.raw_output),
            "generation_metadata": generation.metadata,
            "verification_metadata": verification.metadata,
        }
        candidate_rows.append(base)
        if verified:
            success_rows.append(
                {
                    **base,
                    "theorem": statement_record.statement,
                    "proof": generation.extracted_proof,
                    "no_sorry": not verification.contains_sorry,
                    "valid_proof_extraction": True,
                }
            )
        else:
            failure_rows.append(
                {
                    **base,
                    "theorem": statement_record.statement,
                    "extracted_proof": generation.extracted_proof,
                }
            )

    statement_rows: list[dict[str, Any]] = []
    pass_at = {index: 0 for index in range(1, 9)}
    for statement_record in statements:
        successful = successes_by_statement[statement_record.statement_id]
        successful_indices = {row.sample_index for row in successful}
        for index in range(1, 9):
            pass_at[index] += bool(successful_indices & set(range(index)))
        count = len(successful)
        statement_rows.append(
            {
                "statement_id": statement_record.statement_id,
                "source_id": statement_record.source_id,
                "source": statement_record.source,
                "success_count": count,
                "candidate_count": 8,
                "discovery_bucket": (
                    "solved_easy" if count == 8 else "frontier" if count else "unsolved"
                ),
                "successful_candidate_ids": [row.generation_id for row in successful],
            }
        )

    write_jsonl_atomic(root / "discovery/candidate_results.jsonl", candidate_rows)
    write_jsonl_atomic(root / "discovery/statement_outcomes.jsonl", statement_rows)
    write_jsonl_atomic(root / "success_bank/success_bank.jsonl", success_rows)
    write_jsonl_atomic(root / "failure_bank/failure_bank.jsonl", failure_rows)
    worker_restarts = sum(
        int(worker.get("restart_count") or 0)
        for group in pool_runtimes
        for worker in group["runtime"].get("workers", [])
    )
    worker_process_ids = sorted(
        {
            int(row.metadata["worker_pid"])
            for row in verifications
            if row.metadata.get("worker_pid") is not None
        }
    )
    reported_restart_max = max(
        (
            int(row.metadata.get("worker_restart_count") or 0)
            for row in verifications
        ),
        default=0,
    )
    summary = {
        "status": "EI_ROUND0_DISCOVERY_COMPLETED",
        "statements": len(statements),
        "generated_candidates": len(generations),
        "verified_candidates": len(verifications),
        "pantograph_successes": len(success_rows),
        "pantograph_success_rate": len(success_rows) / expected_candidates,
        "success_bank_rows": len(success_rows),
        "success_bank_statements": sum(
            bool(rows) for rows in successes_by_statement.values()
        ),
        "failure_bank_rows": len(failure_rows),
        "failure_taxonomy": dict(taxonomy),
        "failure_layers": dict(failure_layers),
        "statement_buckets": dict(
            Counter(row["discovery_bucket"] for row in statement_rows)
        ),
        "pass_at": {
            f"pass@{index}": {
                "solved": pass_at[index],
                "rate": pass_at[index] / expected_rows,
            }
            for index in range(1, 9)
        },
        "generation_seconds": generation_seconds,
        "verification_seconds": verification_seconds,
        "checkpoint_hash": EXPECTED_H0_HASH,
        "generation_config_hash": contract["generation_config_sha256"],
        "candidate_seed_min": min(row.generation_seed or 0 for row in generations),
        "candidate_seed_max": max(row.generation_seed or 0 for row in generations),
        "fatal_errors": 0,
        "worker_restarts": worker_restarts,
        "worker_process_lifecycles": len(worker_process_ids),
        "worker_process_ids": worker_process_ids,
        "reported_worker_restart_max": reported_restart_max,
        "verification_workers": worker_count,
        "verification_queue_maxsize": queue_maxsize,
        "verification_execution_mode": "staged_sequential_grouped_source_faithful",
        "verification_import_group_count": len(pool_runtimes),
        "verification_import_groups": pool_runtimes,
        "verification_resume_order": args.verification_order,
        "verification_concurrency_override": (
            args.verification_workers is not None
        ),
        "verification_concurrency_override_reason": (
            "two Mathlib workers exceeded the 14 GiB WSL memory envelope; "
            "resumed incomplete verification with one worker"
            if args.verification_workers == 1
            else None
        ),
        "mean_generation_length": sum(
            int(row.metadata.get("completion_tokens") or 0) for row in generations
        )
        / expected_candidates,
        "max_length_finish_rate": sum(
            row.finish_reason == "length" for row in generations
        )
        / expected_candidates,
        "pathological_repetition_rate": sum(
            repetition_ratio(row.raw_output) >= 0.30 for row in generations
        )
        / expected_candidates,
    }
    write_json_atomic(root / "discovery/discovery_summary.json", summary)
    write_status(
        root,
        status="DISCOVERY_COMPLETED",
        discovery_summary=str(root / "discovery/discovery_summary.json"),
        generated_candidates=expected_candidates,
        verified_candidates=expected_candidates,
        trainer_started=False,
        grpo_started=False,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
