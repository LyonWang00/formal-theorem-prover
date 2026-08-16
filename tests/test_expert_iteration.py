import json
import os
import queue
import sys
import time
import types
from pathlib import Path

import pytest
from datasets import Dataset
from pydantic import ValidationError

from lean_prover.lean_training.data.contracts import make_attestation_id
from lean_prover.lean_training.data.preparation import (
    ASSEMBLER_VERSION,
    NORMALIZATION_VERSION,
)
from lean_prover.lean_training.expert_iteration.banks import FailureBank, ProofBank
from lean_prover.lean_training.expert_iteration.config import (
    CategorySamplingConfig,
    ExpertIterationConfig,
    GenerationConfig,
    TrainMixConfig,
    VerificationConfig,
)
from lean_prover.lean_training.expert_iteration.discovery_generator import generate_candidates
from lean_prover.lean_training.expert_iteration.datasets import (
    assert_prompt_has_no_reference_proof,
    load_statements,
    validate_role_isolation,
)
from lean_prover.lean_training.expert_iteration.discovery_verifier import verify_candidates
from lean_prover.lean_training.expert_iteration.orchestrator import ExpertIterationOrchestrator
from lean_prover.lean_training.expert_iteration.isolated_stage import IsolatedStageRunner
from lean_prover.lean_training.expert_iteration.schemas import (
    DataRole,
    DiscoveryBucket,
    DiscoveryStatementState,
    GenerationRecord,
    StatementRecord,
    VerificationRecord,
    VerificationStatus,
    stable_statement_id,
)
from lean_prover.lean_training.expert_iteration.train_dataset_builder import (
    build_iteration_train_dataset,
)
from lean_prover.lean_training.expert_iteration.utils import (
    environment_identity,
    file_sha256,
    write_jsonl_atomic,
)
from lean_prover.lean_training.evaluation.benchmark import VllmGenerator
from lean_prover.lean_training.sft_pipeline.trainer import (
    WeightedSFTTrainer,
    validate_pantograph_attestation,
)
from lean_prover.lean_training.verification.pool import (
    VerificationPool,
    VerificationPoolConfig,
)
from lean_prover.lean_training.verification.schema import VerificationTask
from lean_prover.lean_training.verification.schema import VerificationTaskRef
from lean_prover.lean_training.verification.cache import VerificationCache
from lean_prover.lean_training.runtime.config import RuntimeConfig
from lean_prover.lean_training.runtime.engine import GenerationVerificationEngine
from lean_prover.lean_training.runtime.memory_monitor import MemoryMonitor, MemoryPressureError
from lean_prover.lean_training.runtime.profile import resolve_runtime_profile
from lean_prover.lean_training.runtime.resource_manager import ResourceManager


def write_rows(path: Path, rows):
    write_jsonl_atomic(path, rows)
    return str(path)


def prepared_row(row_id, statement, proof=None):
    row = {
        "id": row_id,
        "source": "test",
        "lean_statement": statement,
        "prompt": f"### Lean statement\n{statement}\n\n### Lean proof\n",
        "imports": ["Mathlib"],
    }
    if proof is not None:
        row.update({"proof": proof, "completion": proof})
    return row


def make_statement(role=DataRole.DISCOVERY):
    return StatementRecord.from_prepared(
        prepared_row("s", "theorem s : True", "by trivial"),
        role,
    )


def make_generation(statement, *, generation_id="gen_1", raw="by trivial", iteration=0):
    return GenerationRecord(
        generation_id=generation_id,
        statement_id=statement.statement_id,
        data_role=DataRole.DISCOVERY,
        iteration=iteration,
        checkpoint="checkpoint",
        sample_index=0,
        raw_output=raw,
        extracted_proof=raw,
        temperature=0.9,
        top_p=0.95,
        max_new_tokens=32,
    )


def make_verification(generation, *, success=True):
    return VerificationRecord(
        generation_id=generation.generation_id,
        statement_id=generation.statement_id,
        data_role=DataRole.DISCOVERY,
        iteration=generation.iteration,
        verified=success,
        status=VerificationStatus.SUCCESS if success else VerificationStatus.TACTIC_ERROR,
        error_type=None if success else "tactic_error",
        environment_hash="env",
        assembler_version=ASSEMBLER_VERSION,
        normalization_version=NORMALIZATION_VERSION,
        assembled_source_hash=f"assembled-{generation.generation_id}",
    )


def test_statement_id_role_and_prompt_isolation():
    first = stable_statement_id("source", "42", "theorem a : True")
    second = stable_statement_id("source", "42", "theorem a : False")
    assert first == second
    record = make_statement(DataRole.MONITOR)
    prompt = record.generation_prompt()
    assert record.reference_proof == "by trivial"
    assert "by trivial" not in prompt
    assert_prompt_has_no_reference_proof(record, prompt)
    with pytest.raises(ValidationError):
        StatementRecord.model_validate({**record.model_dump(), "data_role": "test"})
    for role in (DataRole.DISCOVERY, DataRole.MONITOR, DataRole.BENCHMARK):
        role_record = record.model_copy(update={"data_role": role})
        assert "by trivial" not in role_record.generation_prompt()


def test_role_overlap_is_strict_by_statement_hash(tmp_path):
    train = StatementRecord.from_prepared(
        prepared_row("train-id", "theorem duplicate : True", "by trivial"),
        DataRole.TRAIN,
    )
    discovery = StatementRecord.from_prepared(
        prepared_row("different-id", "theorem duplicate : True"),
        DataRole.DISCOVERY,
    )
    with pytest.raises(ValueError, match="overlap"):
        validate_role_isolation(
            {DataRole.TRAIN: [train], DataRole.DISCOVERY: [discovery]},
            report_path=tmp_path / "overlap.json",
        )
    assert json.loads((tmp_path / "overlap.json").read_text())["valid"] is False


def test_banks_enforce_roles_success_and_idempotency(tmp_path):
    statement = make_statement()
    generation = make_generation(statement)
    success = make_verification(generation, success=True)
    failure = make_verification(generation, success=False)
    proof_bank = ProofBank(tmp_path / "proofs.jsonl")
    inserted = proof_bank.insert_verified(statement, generation, success)
    assert inserted is not None and inserted.selected_as_primary
    assert proof_bank.insert_verified(statement, generation, success) is None
    equivalent = make_generation(
        statement,
        generation_id="gen_2",
        raw="by\n  trivial",
    )
    assert proof_bank.insert_verified(statement, equivalent, make_verification(equivalent)) is None
    alternative = make_generation(statement, generation_id="gen_3", raw="by exact True.intro")
    assert proof_bank.insert_verified(statement, alternative, make_verification(alternative))
    assert len(proof_bank.selected_for_training()) == 1
    failure_bank = FailureBank(tmp_path / "failures.jsonl")
    assert failure_bank.insert_failed(statement, generation, failure) is not None
    assert failure_bank.insert_failed(statement, generation, failure) is None
    with pytest.raises(ValueError):
        proof_bank.insert_verified(statement, generation, failure)
    monitor = statement.model_copy(update={"data_role": DataRole.MONITOR})
    with pytest.raises(ValueError):
        failure_bank.insert_failed(monitor, generation, failure)
    benchmark = statement.model_copy(update={"data_role": DataRole.BENCHMARK})
    with pytest.raises(ValueError):
        proof_bank.insert_verified(benchmark, generation, success)


def test_environment_change_marks_proof_for_reverification(tmp_path):
    statement = make_statement()
    generation = make_generation(statement)
    bank = ProofBank(tmp_path / "proofs.jsonl")
    bank.insert_verified(statement, generation, make_verification(generation))
    assert len(bank.selected_for_training()) == 1
    assert bank.mark_for_environment("different-env") == 1
    assert bank.records[0].needs_reverification
    assert bank.selected_for_training() == []
    reverified = make_verification(generation)
    reverified.environment_hash = "different-env"
    assert bank.insert_verified(statement, generation, reverified) is None
    assert not bank.records[0].needs_reverification
    assert len(bank.selected_for_training()) == 1


def test_train_mix_weights_and_protected_roles(tmp_path):
    anchor = prepared_row("anchor", "theorem anchor : True", "by trivial")
    anchor["statement_hash"] = StatementRecord.from_prepared(
        anchor,
        DataRole.TRAIN,
    ).statement_hash
    anchor["category"] = "algebra"
    train_path = write_rows(tmp_path / "train.jsonl", [anchor])
    statement = make_statement()
    statement.category = "logic"
    generation = make_generation(statement)
    bank = ProofBank(tmp_path / "proofs.jsonl")
    bank.insert_verified(statement, generation, make_verification(generation))
    output = tmp_path / "mixed.jsonl"
    stats = build_iteration_train_dataset(
        train_seed_path=train_path,
        selected_proofs=bank.selected_for_training(),
        iteration=0,
        mix=TrainMixConfig(anchor=0.5, historical_expert=0.0, current_expert=0.5),
        category_sampling=CategorySamplingConfig(enabled=True),
        output_path=output,
        manifest_path=tmp_path / "manifest.jsonl",
        stats_path=tmp_path / "stats.json",
        protected_statement_hashes=set(),
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    weights = {
        source: sum(row["sample_weight"] for row in rows if row["expert_source"] == source)
        for source in ("anchor", "current_expert")
    }
    assert weights == pytest.approx({"anchor": 0.5, "current_expert": 0.5})
    assert stats["examples_by_category"] == {"algebra": 1, "logic": 1}
    assert all(row["id"] != "eval" for row in rows)
    with pytest.raises(ValueError, match="protected"):
        build_iteration_train_dataset(
            train_seed_path=train_path,
            selected_proofs=[],
            iteration=0,
            mix=TrainMixConfig(anchor=1.0, historical_expert=0.0, current_expert=0.0),
            category_sampling=CategorySamplingConfig(),
            output_path=tmp_path / "bad.jsonl",
            manifest_path=tmp_path / "bad_manifest.jsonl",
            stats_path=tmp_path / "bad_stats.json",
            protected_statement_hashes={anchor["statement_hash"]},
        )


def test_configuration_rejects_invalid_mix_and_missing_benchmark():
    with pytest.raises(ValidationError):
        TrainMixConfig(anchor=0.5, historical_expert=0.2, current_expert=0.2)
    with pytest.raises(ValidationError, match="benchmark_path"):
        ExpertIterationConfig.model_validate(
            {
                "run_name": "invalid",
                "output_dir": "output",
                "data": {"train_path": "a", "eval_path": "b", "discovery_path": "c"},
                "initial_model": {"base_model": "base"},
            }
        )


def test_configuration_preserves_precheck_and_requested_packing(tmp_path):
    config = ExpertIterationConfig.model_validate(
        {
            "run_name": "contract",
            "output_dir": str(tmp_path / "output"),
            "data": {
                "train_path": "train.jsonl",
                "eval_path": "eval.jsonl",
                "discovery_path": "discovery.jsonl",
            },
            "initial_model": {
                "base_model": "base",
                "checkpoint_strategy": "fixed_anchor",
                "merged_sft0_path": "merged",
            },
            "training": {"packing": True},
            "precheck": {"enabled": True},
            "benchmark": {"enabled": False},
        }
    )
    assert config.training.packing is True
    assert config.precheck.enabled is True
    assert config.precheck.generation_smoke_test.statements == 10


def test_expert_iteration_defaults_to_two_independent_spawn_workers(tmp_path):
    expert_verification = VerificationConfig(lean_project_path=str(tmp_path))
    assert expert_verification.num_workers == 2
    pool = VerificationPool(
        VerificationPoolConfig(
            lean_project_path=str(tmp_path),
            num_workers=expert_verification.num_workers,
        )
    )
    try:
        workers = [pool._make_worker(worker_id) for worker_id in range(2)]
        assert pool.ctx.get_start_method() == "spawn"
        assert len(workers) == 2
        assert workers[0] is not workers[1]
        assert workers[0].name != workers[1].name
    finally:
        pool.close()


def test_isolated_stage_uses_a_distinct_close_fds_process(tmp_path):
    runner = IsolatedStageRunner(tmp_path)
    result = runner.run("probe", {"token": "ok"}, timeout=30)
    assert result["pid"] != os.getpid()
    assert result["parent_pid"] == os.getpid()
    assert result["payload"] == {"token": "ok"}
    events = [
        json.loads(line)
        for line in (tmp_path / "runtime/isolated_processes.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert events[0]["event"] == "started"
    assert events[0]["close_fds"] is True
    assert events[-1]["event"] == "finished"
    assert events[-1]["return_code"] == 0


def test_vllm_generator_honors_low_memory_options(monkeypatch):
    captured = {}
    fake_vllm = types.ModuleType("vllm")

    class FakeLLM:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    fake_vllm.LLM = FakeLLM
    monkeypatch.setitem(sys.modules, "vllm", fake_vllm)
    VllmGenerator(
        "model",
        adapter_path=None,
        max_model_len=256,
        gpu_memory_utilization=0.55,
        enforce_eager=True,
        load_in_4bit=True,
        kv_cache_memory_bytes=33554432,
    )
    assert captured["enforce_eager"] is True
    assert captured["quantization"] == "bitsandbytes"
    assert captured["load_format"] == "bitsandbytes"
    assert captured["kv_cache_memory_bytes"] == 33554432


def test_worker_restart_requeues_only_the_in_flight_task(tmp_path):
    pool = VerificationPool(
        VerificationPoolConfig(
            lean_project_path=str(tmp_path),
            num_workers=1,
            queue_maxsize=2,
            max_worker_restarts=2,
            max_task_retries=1,
        )
    )

    class DeadProcess:
        def is_alive(self):
            return False

    task = VerificationTask(
        priority=0,
        problem_index=0,
        attempt_index=0,
        problem_id="retry",
        prompt="",
        generated_proof="by trivial",
        raw_completion="by trivial",
        lean_code="theorem retry : True := by trivial",
        imports=("Mathlib",),
        payload={"generation_id": "gen_retry"},
    )
    local_queue = queue.Queue(maxsize=2)
    pool.queues[0] = local_queue
    pool.workers[0] = DeadProcess()
    pool.in_flight[0] = task
    spawned = []
    pool._spawn_worker = lambda worker_id: spawned.append(worker_id)
    try:
        pool._restart_worker(0, "simulated crash")
        _, _, retried = local_queue.get_nowait()
        assert isinstance(retried, VerificationTaskRef)
        assert retried.task_key == "gen_retry"
        assert retried.enqueue_time > 0
        assert pool.worker_restart_counts == [1]
        assert pool.worker_generations == [1]
        assert pool.task_retries == {"gen_retry": 1}
        assert spawned == [0]
        assert pool.fatal_errors == []
    finally:
        pool.close()


def test_blocking_drain_stops_after_a_heartbeat_message(tmp_path):
    pool = VerificationPool(
        VerificationPoolConfig(lean_project_path=str(tmp_path), num_workers=1)
    )
    pool.result_queue = queue.Queue()
    pool.result_queue.put(
        {
            "type": "heartbeat",
            "worker_id": 0,
            "generation": 0,
            "pid": 123,
            "state": "starting",
        }
    )
    start = time.monotonic()
    try:
        assert pool.drain(block=True, timeout=1.0) == []
        assert time.monotonic() - start < 0.2
    finally:
        pool.close()


def test_full_pipeline_stage_vocabulary_preserves_legacy_iteration_aliases():
    from lean_prover.lean_training.expert_iteration.schemas import (
        FULL_PIPELINE_STAGE_ORDER,
        IterationStage,
    )

    assert [stage.name for stage in FULL_PIPELINE_STAGE_ORDER] == [
        "DATA_AUDIT",
        "DATA_BUILD",
        "PRECHECK",
        "TRAIN_INITIAL",
        "CHECKPOINT_VERIFY",
        "TRAIN_MEMORIZATION",
        "DISCOVERY_BOOTSTRAP",
        "SELECT_POOL",
        "GENERATE",
        "VERIFY",
        "UPDATE_BANKS",
        "BUILD_DATASET",
        "TRAIN_EXPERT",
        "EVAL",
        "MONITOR",
        "FINALIZE",
        "BENCHMARK",
    ]
    assert IterationStage.SELECT_POOL is IterationStage.SELECT_DISCOVERY_POOL
    assert IterationStage.GENERATE is IterationStage.GENERATE_DISCOVERY
    assert IterationStage.VERIFY is IterationStage.VERIFY_DISCOVERY
    assert IterationStage.BUILD_DATASET is IterationStage.BUILD_TRAIN_DATASET
    assert IterationStage.TRAIN_EXPERT is IterationStage.TRAIN
    assert IterationStage.FINALIZE is IterationStage.FINALIZE_ITERATION


def test_drain_keeps_completed_result_from_restarted_worker_generation(tmp_path):
    pool = VerificationPool(
        VerificationPoolConfig(lean_project_path=str(tmp_path), num_workers=1)
    )
    pool.result_queue = queue.Queue()
    pool.worker_generations[0] = 1
    pool.result_queue.put(
        {
            "type": "result",
            "worker_id": 0,
            "generation": 0,
            "pid": 123,
            "pantograph_restart_count": 0,
            "result": {
                "problem_id": "problem-1",
                "problem_index": 0,
                "attempt_index": 2,
                "success": False,
            },
        }
    )
    try:
        results = pool.drain(block=False)
        assert len(results) == 1
        assert results[0]["attempt_index"] == 2
        assert results[0]["stale_worker_generation"] is True
    finally:
        pool.close()


def test_weighted_sampler_survives_trainer_column_removal():
    trainer = object.__new__(WeightedSFTTrainer)
    trainer.sample_weight_field = "sample_weight"
    trainer.sample_weights = [0.1, 0.9]
    trainer.train_dataset = Dataset.from_dict({"input_ids": [[1], [2]]})
    trainer._train_batch_size = 1
    trainer._get_dataloader = lambda **kwargs: kwargs["sampler_fn"](
        kwargs["dataset"]
    )
    sampler = trainer.get_train_dataloader()
    assert list(sampler.weights) == pytest.approx([0.1, 0.9])


def test_sft_rejects_rows_without_pantograph_attestation():
    dataset = Dataset.from_dict(
        {"prompt": ["p", "p"], "completion": ["c", "c"], "pantograph_verified": [True, False]}
    )
    with pytest.raises(ValueError, match="without successful Pantograph"):
        validate_pantograph_attestation(dataset, name="train")


def test_sft_attestation_is_bound_to_source_and_versions():
    source_hash = "assembled-v1"
    environment_hash = "environment-v1"
    attestation_id = make_attestation_id(
        record_id="record-1",
        environment_hash=environment_hash,
        assembler_version=ASSEMBLER_VERSION,
        normalization_version=NORMALIZATION_VERSION,
        assembled_source_hash=source_hash,
    )
    row = {
        "record_id": "record-1",
        "data_state": "verified",
        "statement_verified": True,
        "proof_verified": True,
        "pantograph_verified": True,
        "environment_hash": environment_hash,
        "assembler_version": ASSEMBLER_VERSION,
        "normalization_version": NORMALIZATION_VERSION,
        "attestation_id": attestation_id,
        "assembled_source_hash": source_hash,
    }
    validate_pantograph_attestation(Dataset.from_list([row]), name="train")
    row["assembled_source_hash"] = "tampered"
    with pytest.raises(ValueError, match="attestation_id"):
        validate_pantograph_attestation(Dataset.from_list([row]), name="train")


def test_statement_bucket_transitions_are_resumable():
    state = DiscoveryStatementState(statement_id="s")
    state.update(
        iteration=0,
        sample_count=4,
        success_count=1,
        hard_archive_after_rounds=2,
        solved_easy_success_ratio=1.0,
    )
    assert state.current_bucket is DiscoveryBucket.FRONTIER
    state.update(
        iteration=1,
        sample_count=4,
        success_count=0,
        hard_archive_after_rounds=2,
        solved_easy_success_ratio=1.0,
    )
    state.update(
        iteration=2,
        sample_count=4,
        success_count=0,
        hard_archive_after_rounds=2,
        solved_easy_success_ratio=1.0,
    )
    assert state.current_bucket is DiscoveryBucket.HARD_ARCHIVE
    attempts = state.total_attempts
    zero_rounds = state.consecutive_zero_success_rounds
    state.update(
        iteration=2,
        sample_count=4,
        success_count=0,
        hard_archive_after_rounds=2,
        solved_easy_success_ratio=1.0,
    )
    assert state.total_attempts == attempts
    assert state.consecutive_zero_success_rounds == zero_rounds


def test_forbidden_axiom_is_rejected_without_pantograph(tmp_path):
    statement = make_statement()
    generation = make_generation(statement, raw="by\n  axiom bad : False")
    output = tmp_path / "verification.jsonl"
    records = verify_candidates(
        [generation],
        {statement.statement_id: statement},
        config=VerificationConfig(
            lean_project_path=str(tmp_path),
            imports=["Mathlib"],
            num_workers=1,
            timeout_seconds=1,
            warmup_timeout_seconds=1,
        ),
        output_path=output,
        cache_path=tmp_path / "cache.jsonl",
    )
    assert records[0].status is VerificationStatus.FORBIDDEN_TOKEN
    assert records[0].contains_axiom
    assert records[0].statement == statement.statement
    assert records[0].normalized_proof == generation.extracted_proof
    assert records[0].proof_format == "full_proof"
    assert "import Mathlib" in records[0].assembled_source
    assert records[0].assembler_version
    assert records[0].normalization_version


def test_assembler_version_changes_verification_environment_hash(tmp_path, monkeypatch):
    from lean_prover.lean_training.expert_iteration import utils

    monkeypatch.setattr(utils, "command_output", lambda *args, **kwargs: "Lean 4.test")
    monkeypatch.setattr(utils, "_mathlib_commit", lambda project: "mathlib-test")
    first = environment_identity(tmp_path, ["Mathlib"])
    monkeypatch.setattr(utils, "ASSEMBLER_VERSION", "test-next")
    second = environment_identity(tmp_path, ["Mathlib"])
    assert first["environment_hash"] != second["environment_hash"]
    assert second["assembler_version"] == "test-next"


class FakeGenerator:
    backend_name = "fake"

    def __init__(self, round_index):
        self.round_index = round_index

    def generate_batch(self, prompts, *, k, **kwargs):
        rows = []
        for problem_index, prompt in enumerate(prompts):
            statement_name = next(
                name for name in ("a", "b", "c") if f"theorem {name} :" in prompt
            )
            for sample_index in range(k):
                if self.round_index == 0:
                    success_limit = {"a": 4, "b": 1, "c": 0}[statement_name]
                    success = sample_index < success_limit
                    proof = (
                        f"by exact True.intro -- ok_{statement_name}"
                        if success
                        else f"by fail_{statement_name}_{sample_index}"
                    )
                else:
                    success_limit = {"a": 4, "b": 3, "c": 1}[statement_name]
                    success = sample_index < success_limit
                    proof = (
                        "by exact True.intro -- ok_a"
                        if success and statement_name == "a"
                        else "by trivial -- ok"
                        if success
                        else f"by fail_{statement_name}_{sample_index}"
                    )
                rows.append(
                    {
                        "problem_batch_index": problem_index,
                        "local_attempt_index": sample_index,
                        "raw_completion": proof,
                        "completion_tokens": len(proof.split()),
                        "generation_problem_seconds": 0.01,
                        "finish_reason": "stop",
                    }
                )
        return rows


class FakeTrainer:
    def __init__(self):
        self.calls = []

    def train_iteration(self, **kwargs):
        self.calls.append(kwargs)
        Path(kwargs["output_dir"]).mkdir(parents=True, exist_ok=True)
        assert file_sha256(kwargs["eval_path"]) == kwargs["expected_eval_hash"]
        return {
            "eval_dataset_hash": kwargs["expected_eval_hash"],
            "train_loss": 1.0,
            "best_eval_loss": 0.5 - 0.1 * kwargs["iteration"],
            "final_eval_loss": 0.5 - 0.1 * kwargs["iteration"],
            "best_eval_step": 1,
            "training_skipped": False,
        }


def fake_verify(generations, statements, *, output_path, **kwargs):
    records = []
    for generation in generations:
        success = "ok" in (generation.extracted_proof or "")
        records.append(
            VerificationRecord(
                generation_id=generation.generation_id,
                statement_id=generation.statement_id,
                data_role=DataRole.DISCOVERY,
                iteration=generation.iteration,
                verified=success,
                status=VerificationStatus.SUCCESS if success else VerificationStatus.TACTIC_ERROR,
                error_type=None if success else "tactic_error",
                error_message=None if success else "mock failure",
                environment_hash="mock-env",
                assembler_version=ASSEMBLER_VERSION,
                normalization_version=NORMALIZATION_VERSION,
                assembled_source_hash=f"assembled-{generation.generation_id}",
            )
        )
    write_jsonl_atomic(output_path, (record.model_dump(mode="json") for record in records))
    return records


def test_mock_two_round_orchestration_reuses_fixed_eval(tmp_path):
    train_path = write_rows(
        tmp_path / "train.jsonl",
        [prepared_row("train", "theorem train : True", "by trivial")],
    )
    eval_path = write_rows(
        tmp_path / "eval.jsonl",
        [prepared_row("eval", "theorem eval : True", "by trivial")],
    )
    discovery_path = write_rows(
        tmp_path / "discovery.jsonl",
        [
            prepared_row("a", "theorem a : True", "by trivial"),
            prepared_row("b", "theorem b : True", "by trivial"),
            prepared_row("c", "theorem c : True", "by trivial"),
        ],
    )
    config = ExpertIterationConfig.model_validate(
        {
            "run_name": "mock",
            "output_dir": str(tmp_path / "output"),
            "seed": 7,
            "data": {
                "train_path": train_path,
                "eval_path": eval_path,
                "discovery_path": discovery_path,
            },
            "initial_model": {
                "base_model": "base",
                "sft_adapter": "sft0",
                "skip_initial_sft": True,
                "checkpoint_strategy": "continue_adapter",
            },
            "iterations": {"max_iterations": 2, "minimum_new_proofs_to_train": 1},
            "discovery": {
                "statements_per_iteration": 3,
                "pool_mix": {"new": 0.3, "frontier": 0.4, "unsolved": 0.2, "audit": 0.1},
                "generation": {"samples_per_statement": 4, "batch_size": 8},
                "sampling_budget": {"new": 4, "frontier": 4, "unsolved": 4, "audit": 2},
            },
            "verification": {"lean_project_path": str(tmp_path)},
            "proof_bank": {
                "max_verified_proofs_per_statement": 3,
                "max_training_proofs_per_statement": 1,
                "selection_strategy": "shortest",
            },
            "monitor": {"enabled": False},
            "benchmark": {"enabled": False},
            "stopping": {"min_discovery_new_solved_ratio": 0.0, "patience": 2},
        }
    )
    trainer = FakeTrainer()
    round_counter = {"value": 0}

    def generator_factory(*args):
        current = round_counter["value"]
        round_counter["value"] += 1
        return FakeGenerator(current)

    orchestrator = ExpertIterationOrchestrator(
        config,
        generator_factory=generator_factory,
        verifier=fake_verify,
        trainer=trainer,
    )
    dry_run = orchestrator.dry_run()
    assert dry_run["dry_run"] is True
    assert dry_run["records_by_role"] == {"train": 1, "eval": 1, "discovery": 3}
    assert trainer.calls == []
    orchestrator.run()

    assert len(trainer.calls) == 2
    assert trainer.calls[0]["train_path"] != trainer.calls[1]["train_path"]
    assert trainer.calls[0]["eval_path"] == trainer.calls[1]["eval_path"] == eval_path
    assert trainer.calls[0]["expected_eval_hash"] == trainer.calls[1]["expected_eval_hash"]
    proof_rows = json.loads(
        "[" + ",".join((tmp_path / "output/banks/proof_bank.jsonl").read_text().splitlines()) + "]"
    )
    failure_rows = (tmp_path / "output/banks/failure_bank.jsonl").read_text().splitlines()
    assert proof_rows
    assert failure_rows
    assert len({row["proof_id"] for row in proof_rows}) == len(proof_rows)
    second_manifest = (tmp_path / "output/iteration_001/train/train_manifest.jsonl").read_text()
    assert "historical_expert" in second_manifest
    assert "current_expert" in second_manifest
    assert "eval" not in second_manifest
    assert not (tmp_path / "output/benchmark").exists()
    first_round = [
        json.loads(line)
        for line in (tmp_path / "output/iteration_000/discovery/verifications.jsonl")
        .read_text()
        .splitlines()
    ]
    first_counts = {}
    generation_rows = {
        row["generation_id"]: row
        for row in (
            json.loads(line)
            for line in (tmp_path / "output/iteration_000/discovery/generations.jsonl")
            .read_text()
            .splitlines()
        )
    }
    for row in first_round:
        statement_id = row["statement_id"]
        counts = first_counts.setdefault(statement_id, [0, 0])
        counts[1] += 1
        counts[0] += int(row["verified"])
        assert generation_rows[row["generation_id"]]["data_role"] == "discovery"
    assert sorted(first_counts.values()) == [[0, 4], [1, 4], [4, 4]]
    orchestrator.run(resume=True)
    assert len(trainer.calls) == 2


def test_final_benchmark_is_not_run_for_a_single_intermediate_iteration(tmp_path):
    train_path = write_rows(
        tmp_path / "train.jsonl",
        [prepared_row("train", "theorem train : True", "by trivial")],
    )
    eval_path = write_rows(
        tmp_path / "eval.jsonl",
        [prepared_row("eval", "theorem eval : True", "by trivial")],
    )
    discovery_path = write_rows(
        tmp_path / "discovery.jsonl",
        [prepared_row("a", "theorem a : True", "by trivial")],
    )
    benchmark_path = write_rows(
        tmp_path / "benchmark.jsonl",
        [prepared_row("bench", "theorem bench : True")],
    )
    config = ExpertIterationConfig.model_validate(
        {
            "run_name": "benchmark-guard",
            "output_dir": str(tmp_path / "output"),
            "data": {
                "train_path": train_path,
                "eval_path": eval_path,
                "discovery_path": discovery_path,
                "benchmark_path": benchmark_path,
            },
            "initial_model": {
                "base_model": "base",
                "sft_adapter": "sft0",
                "checkpoint_strategy": "continue_adapter",
            },
            "iterations": {"max_iterations": 2, "minimum_new_proofs_to_train": 100},
            "discovery": {
                "statements_per_iteration": 1,
                "generation": {"samples_per_statement": 4},
            },
            "verification": {"lean_project_path": str(tmp_path)},
            "monitor": {"enabled": False},
            "benchmark": {"enabled": True, "run_mode": "final_only"},
            "stopping": {"min_discovery_new_solved_ratio": 0.0},
        }
    )
    orchestrator = ExpertIterationOrchestrator(
        config,
        generator_factory=lambda *args: FakeGenerator(0),
        verifier=fake_verify,
        trainer=FakeTrainer(),
    )
    benchmark_calls = []

    def record_benchmark(*, force=False):
        benchmark_calls.append(force)
        return {}

    orchestrator.run_benchmark = record_benchmark
    orchestrator.run(iteration=0)
    assert benchmark_calls == []


def test_runtime_profiles_resolve_laptop_and_server_defaults():
    laptop = resolve_runtime_profile(RuntimeConfig(profile="laptop"))
    assert laptop.generation_verification_mode == "sequential"
    assert laptop.vllm_persistent is False
    assert laptop.pantograph_persistent is False
    assert laptop.pantograph_workers == 2
    assert laptop.max_queue_size == 32
    server = resolve_runtime_profile(RuntimeConfig(profile="server"))
    assert server.generation_verification_mode == "pipeline"
    assert server.vllm_persistent is True
    assert server.pantograph_persistent is True
    assert server.pantograph_workers == 16
    assert server.max_queue_size == 1000


class _LifecyclePool:
    def __init__(self, events):
        self.events = events
        self.started = True

    def close(self):
        self.events.append("pool_close")
        self.started = False

    def runtime_snapshot(self):
        return {"workers": []}


def _resource_manager(tmp_path, profile, events):
    return ResourceManager(
        profile,
        run_dir=tmp_path,
        generation_stopper=lambda: events.append("vllm_stop_callback"),
        pantograph_factory=lambda: events.append("pool_start") or _LifecyclePool(events),
    )


def test_laptop_lifecycle_stops_generation_before_stage_pantograph(tmp_path):
    events = []
    resources = _resource_manager(
        tmp_path,
        resolve_runtime_profile(RuntimeConfig(profile="laptop", memory_guard={"enabled": False})),
        events,
    )
    resources.start_generation()
    resources.stop_generation()
    resources.start_verification(identity="env")
    resources.stop_verification()
    assert events.index("vllm_stop_callback") < events.index("pool_start")
    assert events[-1] == "pool_close"


def test_server_lifecycle_keeps_pantograph_until_cleanup(tmp_path):
    events = []
    resources = _resource_manager(
        tmp_path,
        resolve_runtime_profile(RuntimeConfig(profile="server", memory_guard={"enabled": False})),
        events,
    )
    first = resources.start_verification(identity="env")
    resources.stop_verification()
    assert first.started is True
    assert "pool_close" not in events
    resources.cleanup_all()
    assert "pool_close" in events


def test_pipeline_interface_exists_without_implementing_async_grpo(tmp_path):
    events = []
    profile = resolve_runtime_profile(RuntimeConfig(profile="server", memory_guard={"enabled": False}))
    resources = _resource_manager(tmp_path, profile, events)
    engine = GenerationVerificationEngine(profile, resources)
    assert engine.run_pipeline(lambda: "generated", lambda: "verified") == ("generated", "verified")
    with pytest.raises(NotImplementedError, match="GRPO"):
        engine.run_async(lambda: None, lambda: None)
    resources.cleanup_all()


def test_memory_monitor_records_warning_and_cooperative_critical_cleanup(tmp_path, monkeypatch):
    from lean_prover.lean_training.runtime import memory_monitor as module

    ratios = iter((0.86, 0.96))
    monkeypatch.setattr(
        module,
        "_memory_info",
        lambda: {
            "process_rss_bytes": 123,
            "system_total_bytes": 1000,
            "system_available_bytes": 100,
            "system_used_bytes": 900,
            "system_used_ratio": next(ratios),
        },
    )
    monkeypatch.setattr(module, "_gpu_memory", lambda: [])
    cleaned = []
    monitor = MemoryMonitor(tmp_path / "memory.jsonl", warning_ratio=0.85, critical_ratio=0.95)
    assert monitor.snapshot("WARNING")["level"] == "warning"
    with pytest.raises(MemoryPressureError):
        monitor.snapshot("CRITICAL", on_critical=lambda: cleaned.append(True), raise_on_critical=True)
    assert cleaned == [True]
    assert len((tmp_path / "memory.jsonl").read_text().splitlines()) == 2


def test_sqlite_verification_cache_is_environment_scoped(tmp_path):
    with VerificationCache(tmp_path / "cache.sqlite") as cache:
        cache.put(
            cache_key="key", proof_hash="proof", statement_hash="statement",
            environment_hash="env-a", assembler_version="2", normalization_version="2",
            status="success", error_type=None, source_path=None,
            verification={"generation_id": "g", "status": "success"},
        )
        assert cache.get("key", environment_hash="env-a", assembler_version="2", normalization_version="2") is not None
        assert cache.get("key", environment_hash="env-b", assembler_version="2", normalization_version="2") is None
        assert cache.count() == 1


def test_verification_queue_contains_only_spooled_task_reference(tmp_path):
    pool = VerificationPool(
        VerificationPoolConfig(
            lean_project_path=str(tmp_path), num_workers=1, queue_maxsize=2,
            task_spool_dir=str(tmp_path / "spool"),
        )
    )
    class CapturingQueue:
        def __init__(self):
            self.item = None
        def put(self, item, **kwargs):
            self.item = item
    capturing = CapturingQueue()
    original_queue = pool.queues[0]
    pool.queues[0] = capturing
    pool.started = True
    task = VerificationTask(
        priority=0, problem_index=0, attempt_index=0, problem_id="p",
        prompt="large prompt", generated_proof="by trivial", raw_completion="raw",
        lean_code="theorem p : True := by trivial", imports=("Mathlib",),
    )
    pool.submit(task)
    reference = capturing.item[2]
    assert isinstance(reference, VerificationTaskRef)
    assert not hasattr(reference, "lean_code")
    assert Path(reference.task_path).exists()
    pool.close()
    original_queue.close()
    original_queue.join_thread()


def test_generation_interruption_keeps_jsonl_and_resume_deduplicates(tmp_path):
    statements = [
        StatementRecord.from_prepared(
            prepared_row(str(index), f"theorem t{index} : True", "by trivial"),
            DataRole.DISCOVERY,
        )
        for index in range(3)
    ]
    states = {row.statement_id: DiscoveryStatementState(statement_id=row.statement_id) for row in statements}

    class InterruptingGenerator:
        backend_name = "fake"
        def __init__(self, fail_after):
            self.calls = 0
            self.fail_after = fail_after
        def generate_batch(self, prompts, *, k, **kwargs):
            self.calls += 1
            if self.fail_after and self.calls > self.fail_after:
                raise RuntimeError("simulated interruption")
            return [
                {"problem_batch_index": index, "local_attempt_index": 0, "raw_completion": "by trivial", "finish_reason": "stop"}
                for index in range(len(prompts))
            ]

    output = tmp_path / "generations.jsonl"
    generation_config = GenerationConfig(samples_per_statement=1, batch_size=1)
    with pytest.raises(RuntimeError, match="interruption"):
        generate_candidates(
            statements, states, iteration=0, checkpoint="m0",
            generator=InterruptingGenerator(fail_after=1), config=generation_config,
            sampling_budget={"new": 1}, zero_success_backoff_after_rounds=2,
            output_path=output, seed=1, force=True,
        )
    assert len(output.read_text().splitlines()) == 1
    run = generate_candidates(
        statements, states, iteration=0, checkpoint="m0",
        generator=InterruptingGenerator(fail_after=None), config=generation_config,
        sampling_budget={"new": 1}, zero_success_backoff_after_rounds=2,
        output_path=output, seed=1, force=False,
    )
    assert len(run) == 3
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len({row["generation_id"] for row in rows}) == 3
    assert {
        row["statement_id"]: row["generation_seed"] for row in rows
    } == {
        statement.statement_id: 1 + index
        for index, statement in enumerate(statements)
    }


def test_verification_interruption_keeps_jsonl_and_resume_deduplicates(tmp_path):
    statement = make_statement()
    first = make_generation(
        statement,
        generation_id="gen_first",
        raw="by axiom forbidden_first : False",
    )
    second = make_generation(
        statement,
        generation_id="gen_second",
        raw="by axiom forbidden_second : False",
    )

    def interrupted():
        yield first
        raise RuntimeError("simulated verification interruption")

    output = tmp_path / "verifications.jsonl"
    config = VerificationConfig(
        lean_project_path=str(tmp_path),
        imports=["Mathlib"],
        num_workers=1,
        use_cache=False,
    )
    with pytest.raises(RuntimeError, match="verification interruption"):
        verify_candidates(
            interrupted(),
            {statement.statement_id: statement},
            config=config,
            output_path=output,
            cache_path=tmp_path / "cache.sqlite",
            force=True,
        )
    assert len(output.read_text().splitlines()) == 1
    resumed = verify_candidates(
        [first, second],
        {statement.statement_id: statement},
        config=config,
        output_path=output,
        cache_path=tmp_path / "cache.sqlite",
        force=False,
    )
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert len(rows) == len({row["generation_id"] for row in rows}) == 2
    assert len(resumed) == 2
