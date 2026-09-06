import json

import pytest

from lean_prover.lean_training.evaluation.benchmark import (
    ProblemState,
    augment_summary,
    standardized_result_rows,
    write_standard_artifacts,
)
from lean_prover.lean_training.evaluation.rollout import read_rollout_records


def test_sft_rollout_uses_prompt_but_never_reference_completion(tmp_path):
    path = tmp_path / "sft_data.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "import Mathlib\n\ntheorem target : True := by sorry\n",
                "completion": "by\n  trivial",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    records = read_rollout_records(path, rollout_kind="sft")
    assert len(records) == 1
    assert records[0].lean_statement == "theorem target : True"
    assert records[0].imports == ("Mathlib",)
    assert "trivial" not in records[0].prompt


def test_grpo_final_data_rejects_any_proof_field(tmp_path):
    path = tmp_path / "grpo_data.jsonl"
    path.write_text(
        json.dumps(
            {
                "prompt": "import Mathlib\n\ntheorem target : True := by sorry\n",
                "proof": "by trivial",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="contains proof fields"):
        read_rollout_records(path, rollout_kind="grpo")


def test_standard_summary_has_exact_k_histogram_and_error_counts(tmp_path):
    states = {
        "a": ProblemState(
            problem_id="a",
            prompt="p",
            expected_attempts=2,
            success=True,
            results=[
                {
                    "attempt_id": "0,0",
                    "attempt_index": 0,
                    "success": True,
                    "status": "success",
                    "statement_hash": "s1",
                    "assembled_source_hash": "a1",
                },
                {
                    "attempt_id": "0,1",
                    "attempt_index": 1,
                    "success": False,
                    "status": "failed",
                    "compile_errors": ["bad tactic"],
                    "statement_hash": "s1",
                    "assembled_source_hash": "a2",
                },
            ],
        ),
        "b": ProblemState(
            problem_id="b",
            prompt="p",
            expected_attempts=2,
            success=False,
            results=[
                {
                    "attempt_id": "1,0",
                    "attempt_index": 0,
                    "success": False,
                    "status": "failed",
                    "timed_out": True,
                },
                {
                    "attempt_id": "1,1",
                    "attempt_index": 1,
                    "success": False,
                    "status": "rejected",
                    "rejected_reason": "empty proof",
                },
            ],
        ),
    }
    summary = augment_summary({}, states, pass_k=2)
    assert summary["problem_success_rate"] == 0.5
    assert summary["attempt_success_rate"] == 0.25
    assert summary["problem_attempt_success_histogram"] == {
        "0/2": 1,
        "1/2": 1,
        "2/2": 0,
    }
    assert summary["error_type_counts"] == {
        "lean_compilation": 1,
        "rejected": 1,
        "timeout": 1,
    }
    rows = standardized_result_rows(states)
    assert rows[0]["problem_attempt_success_rate"] == 0.5
    report = write_standard_artifacts(
        tmp_path, "benchmark", states, {}, pass_k=2
    )
    assert (tmp_path / "benchmark_results.jsonl").is_file()
    assert (tmp_path / "benchmark_report.json").is_file()
    assert report["problem_attempt_success_histogram"]["1/2"] == 1
