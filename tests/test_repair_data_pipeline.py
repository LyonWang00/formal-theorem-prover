from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import pytest
from pydantic import ValidationError

from lean_prover.lean_training.repair_pipeline import (
    AggregatedAttempt, EnvironmentContract, RepairClientConfig, RepairClientError,
    RepairClientResponse, RepairData,
    RepairParseError, RepairPipeline, RepairRequest, build_repair_prompt,
    load_environment_contract, normalize_repaired_proof, parse_repair_response,
)
from lean_prover.lean_training.repair_pipeline.pipeline import assemble_repaired_source
from lean_prover.lean_training.repair_pipeline.schema import APIObservation
from lean_prover.lean_training.verification.pantograph import PantographTheoremVerifier
from scripts.run_repair_data_pipeline import api_call, build_requests, eligible_attempts


def environment() -> EnvironmentContract:
    return EnvironmentContract(
        lean_version="4.29.1",
        lean_commit="f72c35b3f637c8c6571d353742168ab66cc22c00",
        mathlib_commit="5e932f97dd25535344f80f9dd8da3aab83df0fe6",
        pantograph_version="0.3.15",
        imports=["Mathlib"],
        lean_toolchain_sha256="1" * 64,
        lake_manifest_sha256="2" * 64,
    )


def request() -> RepairRequest:
    attempts = [
        AggregatedAttempt(attempt_id="candidate-1", attempt_rank=0,
            failure_proof="by exact True.intro", error_message="type mismatch",
            failure_class="elaboration_error"),
        AggregatedAttempt(attempt_id="candidate-2", attempt_rank=1,
            failure_proof="by simp", error_message="unsolved goals",
            failure_class="unsolved_goals"),
    ]
    return RepairRequest(
        source="WB", theorem_name="demo",
        lean_statement="theorem demo (p : Prop) (h : p) : p",
        target_attempt_id="candidate-1", target_attempt_rank=0,
        target_failure_proof="by exact True.intro",
        target_error_message="type mismatch",
        target_failure_class="elaboration_error",
        statement_id="statement-1", aggregation_group_id="statement-1",
        aggregated_attempts=attempts, attempt_fingerprint="f" * 64,
        environment=environment(),
    )


def observation() -> APIObservation:
    return APIObservation(provider="deepseek", base_url="https://api.deepseek.com",
        model="deepseek-v4-flash", latency_ms=10, response_content_empty=False)


def response_json(*, minimal: str = "by\n  simpa using h", clean: str = "exact h") -> str:
    return json.dumps({
        "schema_version": "repair_api_response_v2",
        "target_attempt_id": "candidate-1",
        "attempt_assessment": "partially_incorrect",
        "assessment_evidence": "The target uses a proof of True where p is required.",
        "minimal_repair_proof": minimal,
        "minimal_repair_change_summary": "Replace the incorrect term.",
        "clean_proof": clean,
    })


class FakeClient:
    model_name = "deepseek-v4-flash"

    def __init__(self, response: str) -> None:
        self.response = response

    def repair(self, value: RepairRequest) -> RepairClientResponse:
        assert value.target_attempt_id == "candidate-1"
        return RepairClientResponse(raw_content=self.response, observation=observation())


@dataclass
class CheckResult:
    success: bool
    diagnostics: str = ""
    check_seconds: float = 0.1
    error_type: str | None = None
    timed_out: bool = False
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()


class FakeVerifier:
    def __init__(self) -> None:
        self.sources: list[str] = []

    def check_source(self, source: str, **_: object) -> CheckResult:
        self.sources.append(source)
        success = "True.intro" not in source and ("exact h" in source or "simpa using h" in source)
        return CheckResult(success=success, diagnostics="" if success else "type mismatch",
            error_type=None if success else "lean_compilation",
            errors=() if success else ("type mismatch",))


def test_prompt_pins_environment_and_aggregates_attempts() -> None:
    prompt = build_repair_prompt(request())
    assert "4.29.1" in prompt
    assert "f72c35b3f637c8c6571d353742168ab66cc22c00" in prompt
    assert "5e932f97dd25535344f80f9dd8da3aab83df0fe6" in prompt
    assert environment().environment_hash in prompt
    assert "candidate-1" in prompt and "candidate-2" in prompt


def test_strict_json_parser_and_proof_normalization() -> None:
    proposal = parse_repair_response(response_json(), expected_attempt_id="candidate-1")
    assert proposal.attempt_assessment == "partially_incorrect"
    normalized = normalize_repaired_proof(proposal.clean_proof)
    assert normalized.proof == "by\n  exact h"
    assert normalized.actions == ("wrapped_leading_tactic_in_by_block",)


@pytest.mark.parametrize("raw", ["", "```json\n{}\n```", json.dumps({"schema_version": "repair_api_response_v2"})])
def test_parser_rejects_empty_wrapped_or_incomplete_json(raw: str) -> None:
    with pytest.raises(RepairParseError):
        parse_repair_response(raw, expected_attempt_id="candidate-1")


def test_source_assembly_preserves_statement() -> None:
    source = assemble_repaired_source(request(), "by exact h", include_imports=True)
    assert "import Mathlib" in source
    assert "theorem demo (p : Prop) (h : p) : p := by exact h" in source


def test_pipeline_verifies_both_and_selects_shorter_clean_proof() -> None:
    verifier = FakeVerifier()
    result = RepairPipeline(client=FakeClient(response_json()), verifier=verifier).run_one(request())
    assert result.data.repair_verified
    assert result.data.minimal_repair and result.data.minimal_repair.verification.verified
    assert result.data.clean_repair and result.data.clean_repair.verification.verified
    assert result.data.selected_strategy == "clean_from_scratch"
    assert result.data.selected_verified_proof == "by\n  exact h"
    assert len(verifier.sources) == 3


def test_pipeline_promotes_nothing_when_generated_proofs_fail() -> None:
    result = RepairPipeline(client=FakeClient(response_json(
        minimal="by exact missing", clean="by exact absent")), verifier=FakeVerifier()).run_one(request())
    assert not result.data.repair_verified
    assert result.data.selected_verified_proof == ""
    assert result.data.anomaly_status == "generated_proofs_failed_verification"


def test_attempt_dedup_keeps_distinct_attempts_for_same_statement() -> None:
    base = {"statement_id": "s", "theorem": "theorem t (p : Prop) (h : p) : p",
        "error_message": "type mismatch", "failure_taxonomy": "elaboration_error",
        "failure_layer": "repair_candidate", "extraction_success": True,
        "timed_out": False, "source": "WB"}
    rows = [
        {**base, "candidate_id": "c1", "candidate_rank": 0, "generated_proof": "by exact True.intro"},
        {**base, "candidate_id": "c2", "candidate_rank": 1, "generated_proof": "by exact h"},
        {**base, "candidate_id": "c3", "candidate_rank": 2, "generated_proof": "by exact h"},
    ]
    eligible, excluded = eligible_attempts(rows)
    assert [row["candidate_id"] for row in eligible] == ["c1", "c2"]
    assert excluded[0]["exclusion_reason"] == "duplicate_attempt"
    requests = build_requests(eligible, environment(), count=0, seed=7)
    assert len(requests) == 2
    assert all(len(row.aggregated_attempts) == 2 for row in requests)


def test_schema_forbids_ambiguous_correct_proof_and_excess_retry() -> None:
    payload = {"source": "WB", "lean_statement": "theorem t : True",
        "statement_id": "s", "aggregation_group_id": "s", "target_attempt_id": "c",
        "attempt_fingerprint": "f" * 64, "peer_attempt_ids": ["c"],
        "target_failure_proof": "by contradiction", "target_error_message": "failed",
        "target_failure_class": "tactic_error", "environment": environment().model_dump(mode="json"),
        "prompt_sha256": "p" * 64, "model_name": "m", "correct_proof": "by trivial"}
    with pytest.raises(ValidationError):
        RepairData.model_validate(payload)
    payload.pop("correct_proof")
    payload["api_retry_count"] = 2
    with pytest.raises(ValidationError):
        RepairData.model_validate(payload)


class EmptyThenSuccessClient:
    def __init__(self, *, always_empty: bool = False) -> None:
        self.calls = 0
        self.always_empty = always_empty

    def repair(self, _: RepairRequest) -> RepairClientResponse:
        self.calls += 1
        if self.always_empty or self.calls == 1:
            empty = observation().model_copy(update={"response_content_empty": True})
            raise RepairClientError("empty", kind="empty_response", observation=empty)
        return RepairClientResponse(raw_content=response_json(), observation=observation())


def test_empty_api_response_is_retried_once_and_only_once() -> None:
    config = RepairClientConfig(api_key="not-used")
    recovering = EmptyThenSuccessClient()
    recovered = api_call(request(), config, client=recovering)
    assert recovered["status"] == "success"
    assert len(recovered["api_call_attempts"]) == 2
    assert recovering.calls == 2

    empty = EmptyThenSuccessClient(always_empty=True)
    failed = api_call(request(), config, client=empty)
    assert failed["status"] == "empty_response_after_retry"
    assert len(failed["api_call_attempts"]) == 2
    assert empty.calls == 2


@pytest.mark.skipif(
    os.environ.get("RUN_PANTOGRAPH_INTEGRATION") != "1",
    reason="set RUN_PANTOGRAPH_INTEGRATION=1 inside the WSL Lean project",
)
def test_real_pantograph_gate_verifies_both_generated_proofs() -> None:
    project = Path.cwd() / "lean_project"
    exact_environment = load_environment_contract(project, imports=("Mathlib",))
    value = request().model_copy(update={"environment": exact_environment})
    verifier = PantographTheoremVerifier(
        project, imports=("Mathlib",), timeout=60, startup_timeout=900
    )
    try:
        assert verifier.warmup(timeout=120).success
        result = RepairPipeline(
            client=FakeClient(response_json()), verifier=verifier, timeout=60
        ).run_one(value)
    finally:
        verifier.close()
    assert result.data.original_attempt_verification is not None
    assert not result.data.original_attempt_verification.verified
    assert result.data.minimal_repair is not None
    assert result.data.minimal_repair.verification.verified
    assert result.data.clean_repair is not None
    assert result.data.clean_repair.verification.verified
    assert result.data.repair_verified
    assert result.data.environment.environment_hash == exact_environment.environment_hash
