"""Resumable stage orchestrator for multi-round expert iteration."""

from __future__ import annotations

import atexit
import hashlib
import json
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable

from .banks import FailureBank, ProofBank
from .config import ExpertIterationConfig
from .datasets import index_statements, load_statements, validate_role_isolation
from .discovery_generator import create_generator_backend, generate_candidates
from .discovery_pool import initialize_statement_states, select_discovery_pool
from .discovery_verifier import verify_candidates
from .evaluation_adapter import BenchmarkPipelineAdapter, EvaluationRunner
from .isolated_stage import IsolatedStageRunner
from .precheck import (
    build_fixed_smoke_tasks,
    build_reference_tasks,
    run_direct_lean_smoke,
    select_reference_records,
    summarize_pool_results,
)
from .schemas import (
    DataRole,
    DiscoveryStatementState,
    GenerationRecord,
    IterationStage,
    IterationState,
    StatementRecord,
    VerificationRecord,
)
from .state import StateStore
from .train_dataset_builder import build_iteration_train_dataset
from .trainer_adapter import IterationTrainer, SFTTrainerAdapter, prepare_fixed_anchor
from .utils import (
    environment_identity,
    file_sha256,
    count_jsonl,
    append_jsonl,
    iter_jsonl,
    read_jsonl,
    write_json_atomic,
    write_jsonl_atomic,
)
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)
from lean_prover.lean_training.runtime import (
    GenerationVerificationEngine,
    ResourceManager,
    resolve_runtime_profile,
)


class ExpertIterationOrchestrator:
    """Coordinate role-isolated discovery, Banks, SFT, monitor, and benchmark."""

    def __init__(
        self,
        config: ExpertIterationConfig,
        *,
        generator_factory: Callable[..., Any] | None = None,
        verifier: Callable[..., list[VerificationRecord]] | None = None,
        trainer: IterationTrainer | None = None,
        evaluation_runner: EvaluationRunner | None = None,
    ) -> None:
        self.config = config
        self.run_dir = Path(config.output_dir).expanduser()
        self.state_store = StateStore(self.run_dir)
        self.managed_generator = generator_factory is None
        self.generator_factory = generator_factory or create_generator_backend
        self.managed_verifier = verifier is None
        self.verifier = verifier or verify_candidates
        self.managed_trainer = trainer is None
        self.trainer = trainer or SFTTrainerAdapter(config)
        self.evaluation_runner = evaluation_runner or BenchmarkPipelineAdapter(config)
        self.datasets: dict[DataRole, list[StatementRecord]] = {}
        self.statement_states: dict[str, DiscoveryStatementState] = {}
        self.proof_bank = ProofBank(
            self.run_dir / "banks" / "proof_bank.jsonl",
            max_per_statement=config.proof_bank.max_verified_proofs_per_statement,
            max_training_per_statement=config.proof_bank.max_training_proofs_per_statement,
            selection_strategy=config.proof_bank.selection_strategy,
            seed=config.seed,
        )
        self.failure_bank = FailureBank(self.run_dir / "banks" / "failure_bank.jsonl")
        self.eval_hash = ""
        self.initial_adapter = config.initial_model.sft_adapter
        self.anchor_model = (
            config.initial_model.merged_sft0_path
            if config.initial_model.checkpoint_strategy == "fixed_anchor"
            else config.initial_model.base_model
        ) or config.initial_model.base_model
        self.verification_pool_identity: str | None = None
        self.isolated_runner = IsolatedStageRunner(
            self.run_dir,
            flashinfer_sampler=config.execution.flashinfer_sampler,
        )
        self.runtime_profile = resolve_runtime_profile(config.runtime)
        self.resource_manager = ResourceManager(
            self.runtime_profile,
            run_dir=self.run_dir,
            generation_stopper=self.isolated_runner.close,
            pantograph_factory=self._create_verification_pool,
        )
        self.execution_engine = GenerationVerificationEngine(
            self.runtime_profile,
            self.resource_manager,
        )
        self._closed = False
        self._last_pool_runtime_write = 0.0
        self._last_memory_check = 0.0
        self.run_started_at = time.time()
        atexit.register(self.close)

    def _elapsed_hours(self) -> float:
        return max(0.0, time.time() - self.run_started_at) / 3600.0

    def _effective_discovery_config(self, iteration: int):
        """Apply elapsed-time statement reductions without mutating the run config."""

        discovery = self.config.discovery.model_copy(deep=True)
        budget = self.config.runtime_budget
        threshold = (
            budget.reduce_iteration_1_if_elapsed_hours_above
            if iteration == 0
            else budget.reduce_iteration_2_if_elapsed_hours_above
        )
        reduced = (
            budget.reduced_iteration_1_statements
            if iteration == 0
            else budget.reduced_iteration_2_statements
        )
        if threshold is None or reduced is None or self._elapsed_hours() <= threshold:
            return discovery
        counts = discovery.iteration_bucket_counts.get(iteration)
        if counts:
            total = sum(counts.values())
            if total > reduced:
                scaled = {
                    key: int(value * reduced / total)
                    for key, value in counts.items()
                }
                remainder = reduced - sum(scaled.values())
                for key in sorted(counts, key=counts.get, reverse=True):
                    if remainder <= 0:
                        break
                    scaled[key] += 1
                    remainder -= 1
                discovery.iteration_bucket_counts[iteration] = scaled
        else:
            discovery.statements_per_iteration = reduced
        return discovery

    def __enter__(self) -> "ExpertIterationOrchestrator":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()

    def close(self) -> None:
        """Release the active GPU subprocess and all persistent Pantograph workers."""

        if self._closed:
            return
        self._closed = True
        pool = self.resource_manager.pantograph_pool
        snapshot = pool.runtime_snapshot() if pool is not None else None
        self.resource_manager.cleanup_all()
        if snapshot is not None:
            for worker in snapshot.get("workers", []):
                worker["alive"] = False
                worker["state"] = "stopped"
            snapshot["status"] = "closed"
            write_json_atomic(self.run_dir / "runtime" / "pantograph_pool.json", snapshot)

    def _create_verification_pool(self) -> VerificationPool:
        """Create a pool; the runtime layer owns when it starts and stops."""

        verification = self.config.verification
        pool = VerificationPool(
            VerificationPoolConfig(
                lean_project_path=verification.lean_project_path,
                imports=tuple(verification.imports),
                timeout=verification.timeout_seconds,
                warmup_timeout=verification.warmup_timeout_seconds,
                num_workers=self.runtime_profile.pantograph_workers,
                queue_maxsize=self.runtime_profile.max_queue_size,
                heartbeat_interval=verification.heartbeat_interval_seconds,
                heartbeat_timeout=verification.heartbeat_timeout_seconds,
                max_worker_restarts=verification.max_worker_restarts,
                max_task_retries=verification.max_task_retries,
                shutdown_timeout=verification.shutdown_timeout_seconds,
                task_spool_dir=str(self.run_dir / "runtime" / "verification_tasks"),
                save_full_source_on_failure_only=(
                    self.runtime_profile.save_full_source_on_failure_only
                ),
            )
        )
        pool.start()
        return pool

    def _ensure_verification_pool(self) -> VerificationPool:
        """Start or refresh the persistent pool when Lean/config identity changes."""

        identity, environment = self._verification_identity()
        verification = self.config.verification
        restart_reason = (
            "verification config or Lean/mathlib identity changed"
            if self.verification_pool_identity not in {None, identity}
            else None
        )
        pool = self.resource_manager.start_verification(identity=identity)
        self.verification_pool_identity = identity
        snapshot = pool.runtime_snapshot()
        snapshot.update(
            {
                "status": "running",
                "identity": identity,
                "environment": environment,
                "restart_reason": restart_reason,
            }
        )
        write_json_atomic(
            self.run_dir / "runtime" / "pantograph_pool.json",
            snapshot,
        )
        return pool

    def _verification_identity(self) -> tuple[str, dict[str, Any]]:
        verification = self.config.verification
        environment = environment_identity(
            verification.lean_project_path,
            verification.imports,
        )
        identity_payload = {
            "verification": verification.model_dump(mode="json"),
            "environment": environment,
        }
        identity = hashlib.sha256(
            json.dumps(
                identity_payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()
        return identity, environment

    def _execute_training(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.resource_manager.before_training()
        try:
            if self.managed_trainer and self.config.execution.isolate_gpu_stages:
                result = self.isolated_runner.run(
                    "training",
                    {
                        "config": self.config.model_dump(mode="json"),
                        "arguments": arguments,
                    },
                    timeout=self.config.execution.training_timeout_seconds,
                    service_callback=self._service_verification_pool,
                )
                metrics = dict(result["metrics"])
                metrics["training_subprocess_pid"] = result.get("pid")
                return metrics
            return self.trainer.train_iteration(**arguments)
        finally:
            self.resource_manager.after_training()

    def _service_verification_pool(self) -> None:
        """Drain heartbeats while the coordinator waits on an isolated GPU stage."""

        now = time.monotonic()
        if now - self._last_memory_check >= 5.0:
            self.resource_manager.check_memory(
                "GPU_STAGE_HEARTBEAT",
                raise_on_critical=True,
            )
            self._last_memory_check = now
        pool = self.resource_manager.pantograph_pool
        if pool is None:
            return
        unexpected_results = pool.drain(block=False, monitor_heartbeats=True)
        if unexpected_results:
            raise RuntimeError(
                "Pantograph pool produced results while no verification batch was active"
            )
        if pool.fatal_errors:
            raise RuntimeError(f"Pantograph pool failed: {pool.fatal_errors}")
        if (
            now - self._last_pool_runtime_write
            >= self.config.verification.heartbeat_interval_seconds
        ):
            snapshot = pool.runtime_snapshot()
            snapshot.update(
                {
                    "status": "running",
                    "identity": self.verification_pool_identity,
                }
            )
            write_json_atomic(
                self.run_dir / "runtime" / "pantograph_pool.json",
                snapshot,
            )
            self._last_pool_runtime_write = now

    def initialize(
        self,
        *,
        dry_run: bool = False,
        force: bool = False,
    ) -> dict[str, Any]:
        self.config.validate_paths()
        roles = {
            DataRole.TRAIN: self.config.data.train_path,
            DataRole.EVAL: self.config.data.eval_path,
            DataRole.DISCOVERY: self.config.data.discovery_path,
        }
        if self.config.data.monitor_path:
            roles[DataRole.MONITOR] = self.config.data.monitor_path
        if self.config.data.benchmark_path:
            roles[DataRole.BENCHMARK] = self.config.data.benchmark_path
        if self.config.data.benchmark_dev_path:
            roles[DataRole.BENCHMARK_DEV] = self.config.data.benchmark_dev_path
        if self.config.data.benchmark_test_path:
            roles[DataRole.BENCHMARK_TEST] = self.config.data.benchmark_test_path
        self.datasets = {role: load_statements(path, role) for role, path in roles.items()}
        empty_required = [
            role.value
            for role in (DataRole.TRAIN, DataRole.EVAL, DataRole.DISCOVERY)
            if not self.datasets.get(role)
        ]
        if empty_required:
            raise ValueError(f"required expert-iteration datasets are empty: {empty_required}")
        for records in self.datasets.values():
            index_statements(records)
        overlap = validate_role_isolation(
            self.datasets,
            report_path=self.run_dir / "data_overlap_report.json",
            policy=self.config.data.overlap_policy,
        )
        self.eval_hash = file_sha256(self.config.data.eval_path)
        global_state = self.state_store.load_global()
        if global_state.get("run_started_at") is not None:
            self.run_started_at = float(global_state["run_started_at"])
        previous_config_hash = global_state.get("config_hash")
        if (
            previous_config_hash
            and previous_config_hash != self.config.config_hash()
            and not force
        ):
            raise ValueError("run config changed; use a new output_dir or --force")
        previous_eval_hash = global_state.get("eval_dataset_hash")
        if previous_eval_hash and previous_eval_hash != self.eval_hash:
            raise ValueError("fixed eval dataset changed for an existing run")
        self.statement_states = initialize_statement_states(
            self.datasets[DataRole.DISCOVERY],
            self.state_store.load_statement_states(),
        )
        if not dry_run:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            global_state.update(
                {
                    "run_name": self.config.run_name,
                    "config_hash": self.config.config_hash(),
                    "eval_dataset_hash": self.eval_hash,
                    "run_started_at": self.run_started_at,
                }
            )
            self.state_store.save_global(global_state)
            self._write_config_snapshot()
            write_jsonl_atomic(
                self.run_dir / "statements.jsonl",
                (
                    record.model_dump(mode="json")
                    for role in self.datasets.values()
                    for record in role
                ),
            )
            self.state_store.save_statement_states(self.statement_states)
        report = {
            "config_hash": self.config.config_hash(),
            "eval_dataset_hash": self.eval_hash,
            "records_by_role": {role.value: len(rows) for role, rows in self.datasets.items()},
            "overlap_report": overlap,
        }
        return report

    def dry_run(self) -> dict[str, Any]:
        report = self.initialize(dry_run=True)
        sample_pool = select_discovery_pool(
            self.datasets[DataRole.DISCOVERY],
            self.statement_states,
            self._effective_discovery_config(0),
            iteration=0,
            seed=self.config.seed,
            category_sampling=self.config.category_sampling,
        )[: min(10, self._effective_discovery_config(0).statements_per_iteration)]
        expected = sum(
            self.config.discovery.sampling_budget.get("new", self.config.discovery.generation.samples_per_statement)
            for _ in sample_pool
        )
        report.update(
            {
                "dry_run": True,
                "sample_discovery_pool": [record.statement_id for record in sample_pool],
                "estimated_candidates_for_sample": expected,
                "train_seed_rows": len(self.datasets[DataRole.TRAIN]),
                "fixed_eval_rows": len(self.datasets[DataRole.EVAL]),
            }
        )
        write_json_atomic(self.run_dir / "dry_run_report.json", report)
        return report

    def run(
        self,
        *,
        stage: IterationStage | None = None,
        iteration: int | None = None,
        resume: bool = True,
        force: bool = False,
    ) -> dict[str, Any]:
        self._closed = False
        try:
            if iteration is not None and not 0 <= iteration < self.config.iterations.max_iterations:
                raise ValueError(
                    f"iteration must be in [0, {self.config.iterations.max_iterations - 1}]"
                )
            self.initialize(force=force)
            if stage is IterationStage.BENCHMARK:
                precheck_state = self._get_or_create_iteration_state(
                    0,
                    resume=resume,
                    force=force,
                )
                self.run_stage(
                    precheck_state,
                    IterationStage.PRECHECK,
                    force=force,
                )
                self._prepare_initial_model()
                return self.run_benchmark(force=force)
            iterations = (
                [iteration]
                if iteration is not None
                else list(range(self.config.iterations.max_iterations))
            )
            final: dict[str, Any] = {}
            for current in iterations:
                if current is None:
                    continue
                state = self._get_or_create_iteration_state(
                    current,
                    resume=resume,
                    force=force,
                )
                if stage is None and state.status == "completed" and resume and not force:
                    if self.config.precheck.enabled and not state.precheck_completed:
                        self.run_stage(state, IterationStage.PRECHECK, force=False)
                    final = state.metrics
                    continue
                if stage is None:
                    for current_stage in (
                        IterationStage.PRECHECK,
                        IterationStage.SELECT_DISCOVERY_POOL,
                        IterationStage.GENERATE_DISCOVERY,
                        IterationStage.VERIFY_DISCOVERY,
                        IterationStage.UPDATE_BANKS,
                        IterationStage.BUILD_TRAIN_DATASET,
                        IterationStage.TRAIN,
                        IterationStage.EVAL,
                        IterationStage.MONITOR,
                        IterationStage.FINALIZE_ITERATION,
                    ):
                        self.run_stage(state, current_stage, force=force)
                        if (
                            current_stage is IterationStage.PRECHECK
                            and current == 0
                            and iteration is None
                            and self.config.monitor.enabled
                            and self.config.monitor.evaluate_initial_checkpoint
                        ):
                            self._prepare_initial_model()
                            self._run_initial_monitor(force=force)
                else:
                    self.run_stage(state, stage, force=force)
                final = state.metrics
                if stage is None and self._should_stop(current):
                    break
            if (
                stage is None
                and iteration is None
                and self.config.benchmark.enabled
                and self.config.benchmark.run_mode == "final_only"
            ):
                self.run_benchmark(force=force)
            return final
        finally:
            self.close()

    def run_stage(self, state: IterationState, stage: IterationStage, *, force: bool) -> None:
        hard_limit = self.config.runtime_budget.hard_limit_hours
        if hard_limit is not None and self._elapsed_hours() >= hard_limit:
            raise TimeoutError(
                f"expert iteration exceeded hard runtime limit of {hard_limit} hours"
            )
        self._assert_stage_prerequisites(state, stage)
        if force:
            self._invalidate_from_stage(state, stage)
        state.current_stage = stage
        self.state_store.save_iteration(state)
        if stage in {
            IterationStage.GENERATE_DISCOVERY,
            IterationStage.TRAIN,
            IterationStage.MONITOR,
        }:
            self._prepare_initial_model()
        dispatch = {
            IterationStage.PRECHECK: self._precheck,
            IterationStage.SELECT_DISCOVERY_POOL: self._select_pool,
            IterationStage.GENERATE_DISCOVERY: self._generate,
            IterationStage.VERIFY_DISCOVERY: self._verify,
            IterationStage.UPDATE_BANKS: self._update_banks,
            IterationStage.BUILD_TRAIN_DATASET: self._build_train,
            IterationStage.TRAIN: self._train,
            IterationStage.EVAL: self._eval,
            IterationStage.MONITOR: self._monitor,
            IterationStage.FINALIZE_ITERATION: self._finalize,
        }
        if stage not in dispatch:
            raise ValueError(f"stage {stage} is not an iteration stage")
        callback = lambda: dispatch[stage](state, force=force)
        if stage is IterationStage.GENERATE_DISCOVERY:
            self.execution_engine.run_generation_stage(callback)
        elif (
            stage is IterationStage.VERIFY_DISCOVERY
            and self.managed_verifier
            and not (
                self.config.execution.verify_after_each_generation_batch
                and state.verification_completed
                and (
                    self._discovery_dir(state.iteration) / "verifications.jsonl"
                ).exists()
                and not force
            )
        ):
            identity, _ = self._verification_identity()
            self.execution_engine.run_verification_stage(callback, identity=identity)
        else:
            callback()
        self.state_store.save_iteration(state)

    def _invalidate_from_stage(
        self,
        state: IterationState,
        stage: IterationStage,
    ) -> None:
        ordered_flags = [
            (IterationStage.PRECHECK, "precheck_completed"),
            (IterationStage.SELECT_DISCOVERY_POOL, "pool_selected"),
            (IterationStage.GENERATE_DISCOVERY, "generation_completed"),
            (IterationStage.VERIFY_DISCOVERY, "verification_completed"),
            (IterationStage.UPDATE_BANKS, "banks_updated"),
            (IterationStage.BUILD_TRAIN_DATASET, "train_dataset_built"),
            (IterationStage.TRAIN, "training_completed"),
            (IterationStage.EVAL, "eval_completed"),
            (IterationStage.MONITOR, "monitor_completed"),
        ]
        stage_order = [item[0] for item in ordered_flags]
        if stage in stage_order:
            start = stage_order.index(stage)
            for _, flag in ordered_flags[start:]:
                setattr(state, flag, False)
        if (
            stage in stage_order
            and stage_order.index(stage) <= stage_order.index(IterationStage.TRAIN)
        ):
            state.training_skipped = False
            state.output_checkpoint = None
        state.metrics.pop("stop_reason", None)
        if (
            stage in stage_order
            and stage_order.index(stage) <= stage_order.index(IterationStage.EVAL)
        ):
            state.metrics.pop("eval_loss_degradation_warning", None)
        if (
            stage in stage_order
            and stage_order.index(stage) <= stage_order.index(IterationStage.MONITOR)
        ):
            for key in list(state.metrics):
                if key.startswith("monitor_") or key.startswith("benchmark_dev_"):
                    state.metrics.pop(key, None)
        state.status = "running"

    def _assert_stage_prerequisites(
        self,
        state: IterationState,
        stage: IterationStage,
    ) -> None:
        required_flags = {
            IterationStage.SELECT_DISCOVERY_POOL: ("precheck_completed",),
            IterationStage.GENERATE_DISCOVERY: ("pool_selected",),
            IterationStage.VERIFY_DISCOVERY: ("generation_completed",),
            IterationStage.UPDATE_BANKS: ("verification_completed",),
            IterationStage.BUILD_TRAIN_DATASET: ("banks_updated",),
            IterationStage.TRAIN: ("train_dataset_built",),
            IterationStage.EVAL: ("training_completed",),
            IterationStage.MONITOR: ("training_completed",),
            IterationStage.FINALIZE_ITERATION: ("training_completed",),
        }
        missing = [
            flag
            for flag in required_flags.get(stage, ())
            if not getattr(state, flag)
        ]
        if stage is IterationStage.FINALIZE_ITERATION:
            if not state.training_skipped and not state.eval_completed:
                missing.append("eval_completed")
            if self.config.monitor.enabled and not state.monitor_completed:
                missing.append("monitor_completed")
        if missing:
            raise RuntimeError(
                f"iteration={state.iteration} stage={stage.value} requires {sorted(set(missing))}"
            )

    def _precheck(self, state: IterationState, *, force: bool) -> None:
        """Run hard environment, assembler, checkpoint, and small-generation gates."""

        output_dir = self.run_dir / "precheck"
        report_path = output_dir / "precheck_report.json"
        if state.precheck_completed and report_path.exists() and not force:
            return
        if not self.config.precheck.enabled:
            report = {"enabled": False, "success": True, "checks": {}}
            write_json_atomic(report_path, report)
            state.precheck_completed = True
            state.metrics["precheck_success"] = True
            return

        precheck = self.config.precheck
        report: dict[str, Any] = {"enabled": True, "success": False, "checks": {}}

        def record(name: str, value: dict[str, Any], *, required_rate: float) -> None:
            report["checks"][name] = value
            pass_rate = float(value.get("pass_rate", int(bool(value.get("success")))))
            if pass_rate < required_rate:
                write_json_atomic(report_path, report)
                raise RuntimeError(
                    f"PRECHECK failed: {name} pass_rate={pass_rate:.4f} "
                    f"< required={required_rate:.4f}; see {report_path}"
                )

        if precheck.lean_smoke_test.enabled:
            lean_smoke = run_direct_lean_smoke(
                self.config.verification.lean_project_path,
                output_dir,
                imports=self.config.verification.imports,
            )
            lean_smoke["pass_rate"] = lean_smoke["passed"] / lean_smoke["total"]
            record(
                "lean_smoke_test",
                lean_smoke,
                required_rate=precheck.lean_smoke_test.required_pass_rate,
            )

        pool = self._ensure_verification_pool()
        if precheck.pantograph_smoke_test.enabled:
            pool_result = pool.run_batch(
                build_fixed_smoke_tasks(self.config.verification.imports)
            )
            pantograph_smoke = summarize_pool_results(pool_result.results)
            pantograph_smoke["warmup_reports"] = pool_result.warmup_reports
            pantograph_smoke["runtime_stats"] = pool_result.runtime_stats
            record(
                "pantograph_smoke_test",
                pantograph_smoke,
                required_rate=precheck.pantograph_smoke_test.required_pass_rate,
            )

        reference_config = precheck.reference_roundtrip
        if reference_config.enabled:
            references, notes = select_reference_records(
                self.datasets,
                train_samples=reference_config.train_samples,
                eval_samples=reference_config.eval_samples,
                benchmark_samples=reference_config.benchmark_samples,
            )
            reference_result = pool.run_batch(
                build_reference_tasks(
                    references,
                    default_imports=self.config.verification.imports,
                )
            )
            reference_report = summarize_pool_results(reference_result.results)
            reference_report["notes"] = notes
            reference_report["requested"] = (
                reference_config.train_samples
                + reference_config.eval_samples
                + reference_config.benchmark_samples
            )
            if reference_report["total"] != reference_report["requested"]:
                reference_report["pass_rate"] = 0.0
                reference_report["success"] = False
            record(
                "reference_roundtrip",
                reference_report,
                required_rate=reference_config.required_pass_rate,
            )

        checkpoint_report = self._checkpoint_precheck_report()
        if precheck.checkpoint_check.enabled:
            record(
                "checkpoint_check",
                checkpoint_report,
                required_rate=1.0,
            )

        prompt_rows = self.datasets[DataRole.DISCOVERY][:
            min(10, len(self.datasets[DataRole.DISCOVERY]))
        ]
        prompt_success = all(
            row.generation_prompt().endswith("### Lean proof\n")
            and not (
                row.reference_proof
                and row.reference_proof.strip() in row.generation_prompt()
            )
            for row in prompt_rows
        )
        record(
            "prompt_template_check",
            {
                "success": prompt_success,
                "pass_rate": float(prompt_success),
                "prompt_format": "plain_text_lean_sections_v1",
                "samples_checked": len(prompt_rows),
            },
            required_rate=1.0,
        )

        if precheck.generation_smoke_test.enabled:
            self.resource_manager.stop_verification()
            generation_report = self._generation_precheck(force=force)
            report["checks"]["generation_smoke_test"] = generation_report
            if not generation_report["success"]:
                write_json_atomic(report_path, report)
                raise RuntimeError(
                    "PRECHECK failed: generation_smoke_test; "
                    f"see {report_path}"
                )

        self.resource_manager.stop_verification()
        report["success"] = True
        write_json_atomic(report_path, report)
        state.precheck_completed = True
        state.metrics["precheck_success"] = True

    def _checkpoint_precheck_report(self) -> dict[str, Any]:
        model = self.config.initial_model

        def inspect_path(value: str | None) -> dict[str, Any] | None:
            if not value:
                return None
            path = Path(value).expanduser()
            exists = path.exists()
            resolved = str(path.resolve()) if exists else value
            files = []
            if path.is_dir():
                files = sorted(
                    {
                        child.name
                        for pattern in (
                            "config.json",
                            "adapter_config.json",
                            "*.safetensors",
                            "*.index.json",
                            "tokenizer_config.json",
                        )
                        for child in path.glob(pattern)
                    }
                )
            return {
                "configured": value,
                "resolved": resolved,
                "exists": exists,
                "files": files,
            }

        base = inspect_path(model.base_model)
        adapter = inspect_path(model.sft_adapter)
        merged = inspect_path(model.merged_sft0_path)
        tokenizer = inspect_path(model.tokenizer or model.merged_sft0_path or model.base_model)
        selected = merged if model.checkpoint_strategy == "fixed_anchor" else adapter or base
        base_is_remote_id = bool(base and not base["exists"] and "/" in model.base_model)
        selected_ok = bool(selected and selected["exists"])
        adapter_ok = bool(
            adapter is None
            or (
                adapter["exists"]
                and "adapter_config.json" in adapter["files"]
                and any(name.endswith(".safetensors") for name in adapter["files"])
            )
        )
        merged_ok = bool(
            merged is None
            or (
                merged["exists"]
                and "config.json" in merged["files"]
                and any(
                    name.endswith(".safetensors") or name.endswith(".index.json")
                    for name in merged["files"]
                )
            )
        )
        success = bool(
            (base and (base["exists"] or base_is_remote_id))
            and selected_ok
            and adapter_ok
            and merged_ok
        )
        return {
            "success": success,
            "pass_rate": float(success),
            "base_model_path": base,
            "sft_adapter_path": adapter,
            "merged_checkpoint_path": merged,
            "selected_checkpoint_path": selected,
            "tokenizer_path": tokenizer,
            "checkpoint_strategy": model.checkpoint_strategy,
            "adapter_enabled": bool(
                model.checkpoint_strategy == "continue_adapter" and model.sft_adapter
            ),
        }

    def _generation_precheck(
        self,
        *,
        force: bool,
    ) -> dict[str, Any]:
        """Generate and verify at most 20 prompts without touching iteration artifacts."""

        smoke = self.config.precheck.generation_smoke_test
        selected = self.datasets[DataRole.DISCOVERY][: smoke.statements]
        output_dir = self.run_dir / "precheck" / "generation_smoke"
        generation_path = output_dir / "generations.jsonl"
        state_rows = {
            row.statement_id: DiscoveryStatementState(statement_id=row.statement_id)
            for row in selected
        }
        budget = {
            "new": smoke.samples_per_statement,
            "frontier": smoke.samples_per_statement,
            "unsolved": smoke.samples_per_statement,
            "audit": smoke.samples_per_statement,
        }
        base = self.anchor_model
        adapter = (
            self.initial_adapter
            if self.config.initial_model.checkpoint_strategy == "continue_adapter"
            else None
        )
        self.resource_manager.start_generation()
        try:
            if self.managed_generator and self.config.execution.isolate_gpu_stages:
                subprocess_result = self.isolated_runner.run(
                "generation",
                {
                    "config": self.config.model_dump(mode="json"),
                    "base_model": base,
                    "adapter_path": adapter,
                    "selected": [
                        row.model_dump(mode="json", exclude={"reference_proof"})
                        for row in selected
                    ],
                    "statement_states": {
                        key: value.model_dump(mode="json")
                        for key, value in state_rows.items()
                    },
                    "iteration": -1,
                    "checkpoint": str(adapter or base),
                    "output_path": str(generation_path),
                    "force": True,
                    "sampling_budget": budget,
                    "generation_seed": self.config.seed,
                },
                timeout=self.config.execution.generation_timeout_seconds,
                    service_callback=self._service_verification_pool,
                )
                backend = subprocess_result.get("backend")
                subprocess_pid = subprocess_result.get("pid")
            else:
                generator = self.generator_factory(
                    base,
                    adapter,
                    self.config.discovery.generation,
                )
                generate_candidates(
                selected,
                state_rows,
                iteration=-1,
                checkpoint=str(adapter or base),
                generator=generator,
                config=self.config.discovery.generation,
                sampling_budget=budget,
                zero_success_backoff_after_rounds=(
                    self.config.discovery.max_consecutive_zero_success_rounds
                ),
                output_path=generation_path,
                seed=self.config.seed,
                    force=True,
                )
                backend = generator.backend_name
                subprocess_pid = None
        finally:
            self.resource_manager.stop_generation()
        generations = [
            GenerationRecord.model_validate(row) for row in read_jsonl(generation_path)
        ]
        pool = self._ensure_verification_pool()
        try:
            verifications = verify_candidates(
                generations,
                index_statements(selected),
                config=self.config.verification,
                output_path=output_dir / "verifications.jsonl",
                cache_path=output_dir / "verification_cache.sqlite",
                force=True,
                verification_pool=pool,
            )
        finally:
            self.resource_manager.stop_verification()
        extraction_rate = sum(bool(row.extracted_proof) for row in generations) / max(
            1, len(generations)
        )
        length_ratio = sum(
            (row.finish_reason or str(row.metadata.get("finish_reason") or "")).lower()
            == "length"
            for row in generations
        ) / max(1, len(generations))
        success = bool(
            len(generations) == len(selected) * smoke.samples_per_statement
            and extraction_rate >= smoke.minimum_extraction_success_rate
            and length_ratio <= smoke.max_length_finish_ratio
        )
        return {
            "success": success,
            "statements": len(selected),
            "candidates": len(generations),
            "extraction_success_rate": extraction_rate,
            "length_finish_ratio": length_ratio,
            "finish_reason_distribution": dict(
                Counter(
                    row.finish_reason
                    or str(row.metadata.get("finish_reason") or "unknown")
                    for row in generations
                )
            ),
            "verification_successes": sum(row.verified for row in verifications),
            "verification_error_distribution": dict(
                Counter(row.status.value for row in verifications if not row.verified)
            ),
            "backend": backend,
            "subprocess_pid": subprocess_pid,
            "generation_path": str(generation_path),
            "verification_path": str(output_dir / "verifications.jsonl"),
        }

    def _select_pool(self, state: IterationState, *, force: bool) -> None:
        path = self._discovery_dir(state.iteration) / "selected_pool.jsonl"
        if state.pool_selected and path.exists() and not force:
            return
        selected = select_discovery_pool(
            self.datasets[DataRole.DISCOVERY],
            self.statement_states,
            self._effective_discovery_config(state.iteration),
            iteration=state.iteration,
            seed=self.config.seed,
            category_sampling=self.config.category_sampling,
        )
        write_jsonl_atomic(
            path,
            (record.model_dump(mode="json", exclude={"reference_proof"}) for record in selected),
        )
        state.pool_selected = True
        state.metrics["statements_attempted"] = len(selected)
        state.metrics["selected_discovery_by_category"] = dict(
            Counter(record.category or "unknown" for record in selected)
        )

    def _generate(self, state: IterationState, *, force: bool) -> None:
        path = self._discovery_dir(state.iteration) / "generations.jsonl"
        if state.generation_completed and path.exists() and not force:
            return
        selected = self._selected_statements(state.iteration)
        discovery = self._effective_discovery_config(state.iteration)
        sampling_budget = discovery.iteration_sampling_budget.get(
            state.iteration,
            discovery.sampling_budget,
        )
        generation_seed = discovery.iteration_generation_seeds.get(
            state.iteration,
            self.config.seed,
        )
        base, adapter = self._discovery_model(state.iteration)
        if self.managed_generator and self.config.execution.isolate_gpu_stages:
            batch_size = int(discovery.batch_statements or len(selected))
            chunks = (
                [selected]
                if self.runtime_profile.vllm_persistent
                else [selected[offset : offset + batch_size] for offset in range(0, len(selected), batch_size)]
            )
            results = []
            for batch_index, batch in enumerate(chunks):
                result = self.isolated_runner.run(
                    "generation",
                    {
                        "config": self.config.model_dump(mode="json"),
                        "base_model": base,
                        "adapter_path": adapter,
                        "selected": [
                            record.model_dump(mode="json", exclude={"reference_proof"})
                            for record in batch
                        ],
                        "statement_states": {
                            record.statement_id: self.statement_states[record.statement_id].model_dump(mode="json")
                            for record in batch
                        },
                        "iteration": state.iteration,
                        "checkpoint": state.discovery_checkpoint,
                        "output_path": str(path),
                        "force": bool(force and batch_index == 0),
                        "sampling_budget": sampling_budget,
                        "generation_seed": generation_seed,
                    },
                    timeout=self.config.execution.generation_timeout_seconds,
                    service_callback=self._service_verification_pool,
                )
                results.append(result)
                self.resource_manager.check_memory(
                    f"GENERATE_BATCH_{batch_index:04d}_END",
                    raise_on_critical=True,
                )
                if self.config.execution.verify_after_each_generation_batch:
                    # The laptop profile deliberately tears down vLLM before
                    # starting Pantograph, then closes Pantograph before the
                    # next isolated generation subprocess. Verification and
                    # Bank updates are cumulative and resumable, so completed
                    # candidates are never regenerated or recompiled.
                    self.resource_manager.stop_generation()
                    identity, _ = self._verification_identity()
                    self.resource_manager.start_verification(identity=identity)
                    try:
                        state.verification_completed = False
                        self._verify(
                            state,
                            force=bool(force and batch_index == 0),
                        )
                    finally:
                        self.resource_manager.stop_verification()
                    state.banks_updated = False
                    self._update_banks(state, force=True)
                    self.state_store.save_iteration(state)
                    if batch_index + 1 < len(chunks):
                        self.resource_manager.start_generation()
            result = results[-1] if results else {}
            state.metrics["generation_subprocess_pid"] = result.get("pid")
            state.metrics["generation_subprocess_pids"] = [item.get("pid") for item in results]
            state.metrics["generation_batches"] = len(results)
            state.metrics["generation_backend"] = result.get("backend")
            state.metrics["flashinfer_sampler_enabled"] = bool(
                result.get("flashinfer_sampler_enabled")
            )
            state.metrics["generation_cuda_home"] = result.get("cuda_home")
            state.metrics["generation_nvcc"] = result.get("nvcc")
            state.metrics["generation_model_load"] = {
                key: result.get(key)
                for key in (
                    "base_model_path",
                    "adapter_path",
                    "tokenizer_path",
                    "vllm_model_path",
                    "eos_token_id",
                    "pad_token_id",
                    "stop_token_ids",
                )
            }
        else:
            generator = self.generator_factory(
                base,
                adapter,
                discovery.generation,
            )
            generations = generate_candidates(
                selected,
                self.statement_states,
                iteration=state.iteration,
                checkpoint=state.discovery_checkpoint,
                generator=generator,
                config=discovery.generation,
                sampling_budget=sampling_budget,
                zero_success_backoff_after_rounds=(
                    self.config.discovery.max_consecutive_zero_success_rounds
                ),
                output_path=path,
                seed=generation_seed,
                force=force,
            )
        state.generation_completed = True
        state.metrics["candidates_generated"] = count_jsonl(path)

    def _verify(self, state: IterationState, *, force: bool) -> None:
        output = self._discovery_dir(state.iteration) / "verifications.jsonl"
        if state.verification_completed and output.exists() and not force:
            return
        statement_index = index_statements(self.datasets[DataRole.DISCOVERY])
        if self.managed_verifier:
            pool = self._ensure_verification_pool()
            verifications = self.verifier(
                (
                    GenerationRecord.model_validate(row)
                    for row in iter_jsonl(self._discovery_dir(state.iteration) / "generations.jsonl")
                ),
                statement_index,
                config=self.config.verification,
                output_path=output,
                cache_path=self.run_dir / "verification_cache.sqlite",
                force=force,
                verification_pool=pool,
                collect_results=False,
            )
            snapshot = pool.runtime_snapshot()
            snapshot["environment_identity"] = self.verification_pool_identity
            snapshot["status"] = "running"
            write_json_atomic(
                self.run_dir / "runtime" / "pantograph_pool.json",
                snapshot,
            )
            state.metrics["pantograph_worker_pids"] = sorted(
                worker["pid"]
                for worker in snapshot["workers"]
                if worker.get("pid") is not None
            )
            state.metrics["pantograph_worker_restart_counts"] = {
                str(worker["worker_id"]): worker["restart_count"]
                for worker in snapshot["workers"]
            }
            state.metrics["pantograph_server_startup_seconds"] = {
                str(worker["worker_id"]): worker["server_startup_seconds"]
                for worker in snapshot["workers"]
            }
        else:
            generations = [
                GenerationRecord.model_validate(row)
                for row in iter_jsonl(self._discovery_dir(state.iteration) / "generations.jsonl")
            ]
            verifications = self.verifier(
                generations,
                statement_index,
                config=self.config.verification,
                output_path=output,
                cache_path=self.run_dir / "verification_cache.sqlite",
                force=force,
            )
        state.verification_completed = True
        state.metrics["candidates_verified"] = count_jsonl(output)
        persisted_verifications = (
            VerificationRecord.model_validate(row) for row in iter_jsonl(output)
        )
        worker_ids = sorted(
            {
                int(record.metadata["worker_id"])
                for record in persisted_verifications
                if record.metadata.get("worker_id") is not None
            }
        )
        state.metrics["pantograph_worker_ids"] = worker_ids
        state.metrics["pantograph_workers_observed"] = len(worker_ids)

    def _update_banks(self, state: IterationState, *, force: bool) -> None:
        metrics_path = self._discovery_dir(state.iteration) / "metrics.json"
        if state.banks_updated and metrics_path.exists() and not force:
            return
        statements = index_statements(self.datasets[DataRole.DISCOVERY])
        discovery_dir = self._discovery_dir(state.iteration)
        generation_path = discovery_dir / "generations.jsonl"
        verification_path = discovery_dir / "verifications.jsonl"
        index_path = discovery_dir / "generation_index.sqlite"
        connection = sqlite3.connect(index_path)
        connection.execute(
            "CREATE TABLE IF NOT EXISTS generations "
            "(generation_id TEXT PRIMARY KEY, record_json TEXT NOT NULL)"
        )
        with connection:
            connection.executemany(
                "INSERT OR REPLACE INTO generations(generation_id, record_json) VALUES (?, ?)",
                (
                    (str(row["generation_id"]), json.dumps(row, ensure_ascii=False))
                    for row in iter_jsonl(generation_path)
                ),
            )
        verification_environments = {
            str(row["environment_hash"])
            for row in iter_jsonl(verification_path)
            if row.get("environment_hash")
        }
        if len(verification_environments) > 1:
            raise RuntimeError(
                "one discovery round produced multiple Lean environment hashes: "
                f"{sorted(verification_environments)}"
            )
        if verification_environments:
            state.metrics["proofs_marked_for_reverification"] = (
                self.proof_bank.mark_for_environment(next(iter(verification_environments)))
            )
        counts: dict[str, list[int]] = defaultdict(lambda: [0, 0])
        category_metrics: dict[str, dict[str, Any]] = defaultdict(
            lambda: {
                "statement_ids": set(),
                "solved_statement_ids": set(),
                "candidates": 0,
                "successful_candidates": 0,
                "new_proofs": 0,
            }
        )
        successful_path = discovery_dir / "successful_candidates.jsonl"
        failed_path = discovery_dir / "failed_candidates.jsonl"
        successful_path.unlink(missing_ok=True)
        failed_path.unlink(missing_ok=True)
        error_distribution: Counter = Counter()
        cache_hits = 0
        verification_count = 0
        for verification_row in iter_jsonl(verification_path):
            verification = VerificationRecord.model_validate(verification_row)
            generation_row = connection.execute(
                "SELECT record_json FROM generations WHERE generation_id = ?",
                (verification.generation_id,),
            ).fetchone()
            if generation_row is None:
                connection.close()
                raise RuntimeError(
                    f"verification references missing generation {verification.generation_id}"
                )
            generation = GenerationRecord.model_validate(json.loads(generation_row[0]))
            statement = statements[generation.statement_id]
            verification_count += 1
            cache_hits += int(bool(verification.metadata.get("cache_hit")))
            category = statement.category or "unknown"
            category_metrics[category]["statement_ids"].add(statement.statement_id)
            category_metrics[category]["candidates"] += 1
            counts[statement.statement_id][1] += 1
            if verification.verified:
                counts[statement.statement_id][0] += 1
                category_metrics[category]["successful_candidates"] += 1
                category_metrics[category]["solved_statement_ids"].add(
                    statement.statement_id
                )
                self.proof_bank.insert_verified(statement, generation, verification)
                append_jsonl(
                    successful_path,
                    ({
                        "generation_id": generation.generation_id,
                        "statement_id": generation.statement_id,
                        "proof": generation.extracted_proof,
                        "status": verification.status.value,
                        "compile_time_ms": verification.compile_time_ms,
                    },),
                )
            else:
                self.failure_bank.insert_failed(statement, generation, verification)
                error_distribution[verification.status.value] += 1
                append_jsonl(failed_path, ({"generation": generation.model_dump(mode="json"), "verification": verification.model_dump(mode="json")},))
        connection.close()
        proofs_found_this_iteration = (
            proof for proof in self.proof_bank.records if proof.iteration_found == state.iteration
        )
        new_proofs = 0
        for proof in proofs_found_this_iteration:
            new_proofs += 1
            category = statements[proof.statement_id].category or "unknown"
            category_metrics[category]["new_proofs"] += 1
        for statement_id, (success_count, sample_count) in counts.items():
            if (
                self.statement_states[statement_id].current_bucket.value == "solved_easy"
                and self.statement_states[statement_id].last_iteration_attempted
                != state.iteration
            ):
                self.statement_states[statement_id].audit_count += 1
            self.statement_states[statement_id].update(
                iteration=state.iteration,
                sample_count=sample_count,
                success_count=success_count,
                hard_archive_after_rounds=self.config.discovery.hard_archive_after_rounds,
                solved_easy_success_ratio=self.config.discovery.solved_easy_success_ratio,
            )
        for proof in self.proof_bank.records:
            state_row = self.statement_states.get(proof.statement_id)
            if state_row and proof.proof_id not in state_row.proof_bank_ids:
                state_row.proof_bank_ids.append(proof.proof_id)
        self.state_store.save_statement_states(self.statement_states)
        newly_solved_ids = sorted(
            statement_id
            for statement_id in counts
            if self.statement_states[statement_id].first_solved_iteration == state.iteration
        )
        write_jsonl_atomic(
            self._discovery_dir(state.iteration) / "newly_solved_statements.jsonl",
            (statements[statement_id].model_dump(mode="json", exclude={"reference_proof"}) for statement_id in newly_solved_ids),
        )
        total_statements = max(1, len(counts))
        bucket_counts = Counter(self.statement_states[key].current_bucket.value for key in counts)
        metrics = {
            "iteration": state.iteration,
            "discovery_checkpoint": state.discovery_checkpoint,
            "statements_attempted": len(counts),
            "candidates_generated": count_jsonl(generation_path),
            "candidates_verified": verification_count,
            "successful_candidates": sum(value[0] for value in counts.values()),
            "discovery_success_at_k": sum(
                value[0] > 0 for value in counts.values()
            )
            / total_statements,
            "verified_candidate_success_rate": sum(
                value[0] for value in counts.values()
            )
            / max(1, verification_count),
            "discovery_new_proofs": new_proofs,
            "newly_solved_statements": len(newly_solved_ids),
            "cumulative_solved_statements": sum(value.ever_solved for value in self.statement_states.values()),
            "zero_success_statement_ratio": bucket_counts["unsolved"] / total_statements,
            "frontier_statement_ratio": bucket_counts["frontier"] / total_statements,
            "all_success_statement_ratio": bucket_counts["solved_easy"] / total_statements,
            "proof_bank_size": len(self.proof_bank.records),
            "failure_bank_size": len(self.failure_bank),
            "discovery_error_distribution": dict(error_distribution),
            "discovery_cache_hit_rate": cache_hits / max(1, verification_count),
            "discovery_by_category": {
                category: {
                    "statements": len(values["statement_ids"]),
                    "candidates": values["candidates"],
                    "successful_candidates": values["successful_candidates"],
                    "new_proofs": values["new_proofs"],
                    "success_at_k": len(values["solved_statement_ids"])
                    / max(1, len(values["statement_ids"])),
                    "candidate_success_rate": values["successful_candidates"]
                    / max(1, values["candidates"]),
                }
                for category, values in sorted(category_metrics.items())
            },
        }
        write_json_atomic(metrics_path, metrics)
        state.metrics.update(metrics)
        state.banks_updated = True

    def _build_train(self, state: IterationState, *, force: bool) -> None:
        train_dir = self._iteration_dir(state.iteration) / "train"
        output = train_dir / "train_dataset.jsonl"
        if state.train_dataset_built and output.exists() and not force:
            return
        mix = self.config.train_mix_first_iteration if state.iteration == 0 else self.config.train_mix
        protected = {
            record.statement_hash
            for role in (DataRole.EVAL, DataRole.MONITOR, DataRole.BENCHMARK, DataRole.BENCHMARK_DEV, DataRole.BENCHMARK_TEST)
            for record in self.datasets.get(role, [])
        }
        stats = build_iteration_train_dataset(
            train_seed_path=self.config.data.train_path,
            selected_proofs=self.proof_bank.selected_for_training(),
            iteration=state.iteration,
            mix=mix,
            category_sampling=self.config.category_sampling,
            output_path=output,
            manifest_path=train_dir / "train_manifest.jsonl",
            stats_path=train_dir / "dataset_stats.json",
            protected_statement_hashes=protected,
        )
        state.metrics.update({"train_examples_by_source": stats["examples_by_source"], "actual_train_mix": stats["actual_sampling_mix"]})
        state.train_dataset_built = True

    def _train(self, state: IterationState, *, force: bool) -> None:
        checkpoint_dir = self._iteration_dir(state.iteration) / "checkpoint"
        if state.training_completed and checkpoint_dir.exists() and not force:
            return
        current_count = sum(
            proof.iteration_found == state.iteration
            for proof in self.proof_bank.selected_for_training()
        )
        if current_count < self.config.iterations.minimum_new_proofs_to_train:
            for key in (
                "train_loss",
                "best_eval_loss",
                "best_eval_step",
                "final_eval_loss",
                "eval_dataset_hash",
                "resumed_from_checkpoint",
            ):
                state.metrics.pop(key, None)
            state.training_skipped = True
            state.training_completed = True
            state.eval_completed = False
            state.output_checkpoint = state.discovery_checkpoint
            state.metrics["training_skipped"] = True
            state.metrics["training_skip_reason"] = (
                f"current proofs {current_count} < minimum "
                f"{self.config.iterations.minimum_new_proofs_to_train}"
            )
            return
        _, previous_adapter = self._discovery_model(state.iteration)
        metrics = self._execute_training(
            {
                "iteration": state.iteration,
                "train_path": str(
                    self._iteration_dir(state.iteration)
                    / "train"
                    / "train_dataset.jsonl"
                ),
                "eval_path": self.config.data.eval_path,
                "output_dir": str(checkpoint_dir),
                "initialization_checkpoint": state.train_initialization_checkpoint,
                "previous_adapter": previous_adapter,
                "expected_eval_hash": self.eval_hash,
            }
        )
        state.metrics.update(metrics)
        state.output_checkpoint = str(checkpoint_dir)
        state.training_completed = True
        state.eval_completed = False

    def _eval(self, state: IterationState, *, force: bool) -> None:
        if state.training_skipped:
            return
        if not state.training_completed:
            raise RuntimeError(f"iteration={state.iteration} stage=eval requires completed training")
        if state.metrics.get("eval_dataset_hash") != self.eval_hash:
            raise ValueError(f"iteration={state.iteration} stage=eval fixed eval hash mismatch")
        state.eval_completed = True

    def _monitor(self, state: IterationState, *, force: bool) -> None:
        base, _ = self._training_initialization(state.iteration)
        adapter = state.output_checkpoint if not state.training_skipped else self._discovery_model(state.iteration)[1]
        if self.config.monitor.enabled:
            output = self._iteration_dir(state.iteration) / "monitor"
            if not (state.monitor_completed and output.exists() and not force):
                metrics = self.evaluation_runner.run(
                    role="monitor",
                    dataset_path=str(self.config.data.monitor_path),
                    base_model=base,
                    adapter_path=adapter,
                    output_dir=str(output),
                    pass_k=self.config.monitor.pass_k,
                    samples_per_statement=self.config.monitor.samples_per_statement,
                    seed=(
                        (self.config.monitor.generation_seed or self.config.seed)
                        if self.config.monitor.fixed_generation_seed
                        else self.config.seed + state.iteration
                    ),
                )
                state.metrics.update(
                    {key: value for key, value in metrics.items() if key.startswith("monitor_")}
                )
                state.monitor_completed = True
        if (
            self.config.data.benchmark_dev_path
            and self.config.checkpoint_selection.metric == "best_benchmark_dev_pass_at_k"
        ):
            output = self._iteration_dir(state.iteration) / "benchmark_dev"
            metrics = self.evaluation_runner.run(
                role="benchmark_dev",
                dataset_path=self.config.data.benchmark_dev_path,
                base_model=base,
                adapter_path=adapter,
                output_dir=str(output),
                pass_k=self.config.benchmark.pass_k,
                samples_per_statement=self.config.benchmark.samples_per_statement,
                seed=self.config.seed,
            )
            state.metrics.update(
                {key: value for key, value in metrics.items() if key.startswith("benchmark_dev_")}
            )

    def _finalize(self, state: IterationState, *, force: bool) -> None:
        state.status = "completed"
        self.state_store.save_iteration(state)
        write_json_atomic(self._iteration_dir(state.iteration) / "metrics.json", state.metrics)
        (self._iteration_dir(state.iteration) / "summary.md").write_text(
            _format_iteration_summary(state),
            encoding="utf-8",
        )
        global_state = self.state_store.load_global()
        iterations = global_state.setdefault("iterations", {})
        iterations[str(state.iteration)] = {
            "status": state.status,
            "checkpoint": state.output_checkpoint,
            "metrics": state.metrics,
        }
        global_state.update(
            {
                "run_name": self.config.run_name,
                "config_hash": self.config.config_hash(),
                "eval_dataset_hash": self.eval_hash,
                "latest_completed_iteration": state.iteration,
            }
        )
        completed_states = [
            saved
            for index in range(self.config.iterations.max_iterations)
            if (
                saved := (
                    state
                    if index == state.iteration
                    else self.state_store.load_iteration(index)
                )
            )
            and saved.status == "completed"
        ]
        trained_expert_iterations = sum(
            item.training_completed and not item.training_skipped
            for item in completed_states
        )
        selected_checkpoint = self._select_checkpoint()
        global_state.update(
            {
                "initial_sft_completed": bool(
                    Path(self.anchor_model).expanduser().exists()
                    or self.config.initial_model.sft_adapter
                ),
                "expert_training_completed": trained_expert_iterations > 0,
                "trained_expert_iterations": trained_expert_iterations,
                "selected_checkpoint": selected_checkpoint,
                "selected_checkpoint_role": (
                    f"M{trained_expert_iterations}"
                    if trained_expert_iterations > 0
                    else "M0"
                ),
                "expert_train_skipped_reason": (
                    state.metrics.get("training_skip_reason")
                    if state.training_skipped
                    else None
                ),
            }
        )
        self.state_store.save_global(global_state)

    def run_benchmark(self, *, force: bool = False) -> dict[str, Any]:
        if not self.config.benchmark.enabled or not self.config.data.benchmark_path:
            return {"benchmark_skipped": True}
        selected = self._select_checkpoint()
        output = self.run_dir / "benchmark"
        metrics_path = output / "benchmark_metrics.json"
        if metrics_path.exists() and not force:
            return json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        base, _ = self._training_initialization(0)
        adapter = selected if selected != base else None
        write_json_atomic(output / "selected_checkpoint.json", {"selected_checkpoint": selected})
        samples_per_statement = self.config.benchmark.samples_per_statement
        budget = self.config.runtime_budget
        if (
            budget.reduce_benchmark_k_if_elapsed_hours_above is not None
            and budget.reduced_benchmark_samples_per_statement is not None
            and self._elapsed_hours() > budget.reduce_benchmark_k_if_elapsed_hours_above
        ):
            samples_per_statement = budget.reduced_benchmark_samples_per_statement
        metrics = self.evaluation_runner.run(
            role="benchmark",
            dataset_path=self.config.data.benchmark_path,
            base_model=base,
            adapter_path=adapter,
            output_dir=str(output),
            pass_k=self.config.benchmark.pass_k,
            samples_per_statement=samples_per_statement,
            seed=self.config.benchmark.generation_seed or self.config.seed,
        )
        write_json_atomic(metrics_path, metrics)
        return metrics

    def _run_initial_monitor(self, *, force: bool = False) -> dict[str, Any]:
        output = self.run_dir / "monitor" / "M0"
        metrics_path = output / "monitor_metrics.json"
        if metrics_path.exists() and not force:
            return json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        base, adapter = self._discovery_model(0)
        metrics = self.evaluation_runner.run(
            role="monitor",
            dataset_path=str(self.config.data.monitor_path),
            base_model=base,
            adapter_path=adapter,
            output_dir=str(output),
            pass_k=self.config.monitor.pass_k,
            samples_per_statement=self.config.monitor.samples_per_statement,
            seed=self.config.monitor.generation_seed or self.config.seed,
        )
        write_json_atomic(metrics_path, metrics)
        return metrics

    def _prepare_initial_model(self) -> None:
        if not self.config.initial_model.skip_initial_sft and not self.initial_adapter:
            output = self.run_dir / "initial_sft" / "checkpoint"
            if not (
                (output / "adapter_model.safetensors").is_file()
                and (output / "trainer_state.json").is_file()
            ):
                self._execute_training(
                    {
                        "iteration": -1,
                        "train_path": self.config.data.train_path,
                        "eval_path": self.config.data.eval_path,
                        "output_dir": str(output),
                        "initialization_checkpoint": self.config.initial_model.base_model,
                        "previous_adapter": None,
                        "expected_eval_hash": self.eval_hash,
                    }
                )
            self.initial_adapter = str(output)
        if self.config.initial_model.checkpoint_strategy == "fixed_anchor":
            self.anchor_model = prepare_fixed_anchor(
                self.config,
                sft_adapter=self.initial_adapter,
            )
        else:
            self.anchor_model = self.config.initial_model.base_model

    def _get_or_create_iteration_state(
        self,
        iteration: int,
        *,
        resume: bool,
        force: bool,
    ) -> IterationState:
        existing = self.state_store.load_iteration(iteration) if resume else None
        if existing:
            if existing.config_hash != self.config.config_hash():
                if not force:
                    raise ValueError(
                        f"iteration={iteration} config hash changed; use a new run or --force"
                    )
                existing.config_hash = self.config.config_hash()
            if existing.eval_dataset_hash != self.eval_hash:
                raise ValueError(f"iteration={iteration} fixed eval dataset changed")
            return existing
        base, previous_adapter = self._discovery_model(iteration)
        train_base, _ = self._training_initialization(iteration)
        return IterationState(
            iteration=iteration,
            discovery_checkpoint=previous_adapter or base,
            train_initialization_checkpoint=train_base,
            config_hash=self.config.config_hash(),
            eval_dataset_hash=self.eval_hash,
        )

    def _discovery_model(self, iteration: int) -> tuple[str, str | None]:
        if iteration == 0:
            if self.config.initial_model.checkpoint_strategy == "fixed_anchor":
                return self.anchor_model, None
            return self.config.initial_model.base_model, self.initial_adapter
        previous = self.state_store.load_iteration(iteration - 1)
        if not previous or previous.status != "completed":
            raise RuntimeError(f"iteration={iteration} requires completed iteration={iteration - 1}")
        if self.config.initial_model.checkpoint_strategy == "fixed_anchor":
            adapter = (
                None
                if previous.training_skipped
                or previous.output_checkpoint == self.anchor_model
                else previous.output_checkpoint
            )
            return self.anchor_model, adapter
        return self.config.initial_model.base_model, previous.output_checkpoint

    def _training_initialization(self, iteration: int) -> tuple[str, str | None]:
        if self.config.initial_model.checkpoint_strategy == "fixed_anchor":
            return self.anchor_model, None
        _, adapter = self._discovery_model(iteration)
        return self.config.initial_model.base_model, adapter

    def _selected_statements(self, iteration: int) -> list[StatementRecord]:
        ids = {row["statement_id"] for row in read_jsonl(self._discovery_dir(iteration) / "selected_pool.jsonl")}
        return [record for record in self.datasets[DataRole.DISCOVERY] if record.statement_id in ids]

    def _select_checkpoint(self) -> str:
        completed = [
            state
            for iteration in range(self.config.iterations.max_iterations)
            if (state := self.state_store.load_iteration(iteration)) and state.status == "completed"
        ]
        if not completed:
            return self.initial_adapter or self.anchor_model
        metric = self.config.checkpoint_selection.metric
        if metric == "best_eval_loss":
            eligible = [state for state in completed if state.metrics.get("best_eval_loss") is not None]
            if eligible:
                return min(eligible, key=lambda item: item.metrics["best_eval_loss"]).output_checkpoint or self.anchor_model
        metric_key = None
        if metric == "best_monitor_pass_at_1":
            metric_key = "monitor_pass_at_1"
        elif metric == "best_monitor_pass_at_k":
            metric_key = f"monitor_pass_at_{max(self.config.monitor.pass_k)}"
        elif metric == "best_benchmark_dev_pass_at_k":
            metric_key = f"benchmark_dev_pass_at_{max(self.config.benchmark.pass_k)}"
        if metric_key:
            eligible = [state for state in completed if state.metrics.get(metric_key) is not None]
            if eligible:
                selected = max(eligible, key=lambda item: item.metrics[metric_key])
                return selected.output_checkpoint or selected.discovery_checkpoint
        return completed[-1].output_checkpoint or completed[-1].discovery_checkpoint

    def _should_stop(self, iteration: int) -> bool:
        state = self.state_store.load_iteration(iteration)
        if not state:
            return False
        global_state = self.state_store.load_global()
        global_state.pop("stop_reason", None)
        global_state.pop("stopped_after_iteration", None)

        def record_stop(reason: str) -> bool:
            global_state["stop_reason"] = reason
            global_state["stopped_after_iteration"] = iteration
            state.metrics["stop_reason"] = reason
            if str(iteration) in global_state.get("iterations", {}):
                global_state["iterations"][str(iteration)]["metrics"] = state.metrics
            self.state_store.save_iteration(state)
            write_json_atomic(self._iteration_dir(iteration) / "metrics.json", state.metrics)
            (self._iteration_dir(iteration) / "summary.md").write_text(
                _format_iteration_summary(state),
                encoding="utf-8",
            )
            self.state_store.save_global(global_state)
            return True

        terminate_below = self.config.iterations.terminate_if_new_proofs_below
        if terminate_below and int(state.metrics.get("discovery_new_proofs") or 0) < terminate_below:
            return record_stop("new_proofs_below_terminate_threshold")

        completed_prior = [
            prior
            for index in range(iteration)
            if (prior := self.state_store.load_iteration(index))
            and prior.status == "completed"
        ]
        completed_all = [*completed_prior, state]
        low_rounds = 0
        for completed_state in reversed(completed_all):
            completed_ratio = completed_state.metrics.get(
                "newly_solved_statements", 0
            ) / max(1, completed_state.metrics.get("statements_attempted", 0))
            if completed_ratio >= self.config.stopping.min_discovery_new_solved_ratio:
                break
            low_rounds += 1
        global_state["low_discovery_rounds"] = low_rounds
        if low_rounds >= self.config.stopping.patience:
            return record_stop("discovery_new_solved_ratio_below_threshold")
        best_eval: float | None = None
        eval_rounds = 0
        eval_degraded = False
        for completed_state in completed_all:
            value = completed_state.metrics.get("final_eval_loss")
            if value is None:
                eval_rounds = 0
                eval_degraded = False
                continue
            eval_degraded = bool(
                best_eval is not None
                and value
                > best_eval * (1 + self.config.stopping.max_eval_loss_degradation)
            )
            eval_rounds = eval_rounds + 1 if eval_degraded else 0
            best_eval = float(value) if best_eval is None else min(best_eval, float(value))
        global_state["eval_degradation_rounds"] = eval_rounds
        if eval_degraded:
            state.metrics["eval_loss_degradation_warning"] = True
            self.state_store.save_iteration(state)
            write_json_atomic(self._iteration_dir(iteration) / "metrics.json", state.metrics)
        if global_state["eval_degradation_rounds"] >= self.config.stopping.patience:
            return record_stop("eval_loss_degradation")
        if self.config.monitor.enabled:
            monitor_key = f"monitor_pass_at_{max(self.config.monitor.pass_k)}"
            best_monitor: float | None = None
            monitor_rounds = 0
            for completed_state in completed_all:
                value = completed_state.metrics.get(monitor_key)
                if value is None:
                    monitor_rounds = 0
                    continue
                improvement = (
                    float(value) - best_monitor if best_monitor is not None else None
                )
                stalled = (
                    improvement is not None
                    and improvement < self.config.stopping.min_monitor_improvement
                )
                monitor_rounds = monitor_rounds + 1 if stalled else 0
                best_monitor = (
                    float(value)
                    if best_monitor is None
                    else max(best_monitor, float(value))
                )
            global_state["monitor_stall_rounds"] = monitor_rounds
            if global_state["monitor_stall_rounds"] >= self.config.stopping.patience:
                return record_stop("monitor_improvement_below_threshold")
        if state.training_skipped and self.config.iterations.stop_when_training_skipped:
            return record_stop("training_skipped")
        self.state_store.save_global(global_state)
        return False

    def _write_config_snapshot(self) -> None:
        path = self.run_dir / "run_config.yaml"
        payload = self.config.model_dump(mode="json")
        try:
            import yaml

            text = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True)
        except ImportError:
            text = json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        path.write_text(text, encoding="utf-8")

    def _iteration_dir(self, iteration: int) -> Path:
        path = self.state_store.iteration_dir(iteration)
        path.mkdir(parents=True, exist_ok=True)
        return path

    def _discovery_dir(self, iteration: int) -> Path:
        path = self._iteration_dir(iteration) / "discovery"
        path.mkdir(parents=True, exist_ok=True)
        return path


def _format_iteration_summary(state: IterationState) -> str:
    metrics = state.metrics
    lines = [
        f"# Expert iteration {state.iteration:03d}",
        "",
        f"- Status: {state.status}",
        f"- Discovery checkpoint: `{state.discovery_checkpoint}`",
        f"- Output checkpoint: `{state.output_checkpoint or state.discovery_checkpoint}`",
        f"- Statements attempted: {metrics.get('statements_attempted', 0)}",
        f"- Candidates verified: {metrics.get('candidates_verified', 0)}",
        f"- Newly solved statements: {metrics.get('newly_solved_statements', 0)}",
        f"- Proof Bank size: {metrics.get('proof_bank_size', 0)}",
        f"- Failure Bank size: {metrics.get('failure_bank_size', 0)}",
        f"- Best eval loss: {metrics.get('best_eval_loss', 'n/a')}",
        f"- Training skipped: {metrics.get('training_skipped', False)}",
    ]
    monitor_keys = sorted(key for key in metrics if key.startswith("monitor_pass_at_"))
    lines.extend(f"- {key}: {metrics[key]}" for key in monitor_keys)
    return "\n".join(lines) + "\n"
