import json
from types import SimpleNamespace

from scripts.run_minif2f_planner_prover_pass4 import (
    lexical_proof_token_count,
    node_attempt_statistics,
    selected_rows,
    tactic_command_count,
    write_node_attempt_statistics_report,
    system_failure_message,
)


def attempt(
    attempt_id: str,
    proof: str,
    *,
    completion_tokens: int | None = None,
    prompt_type: str = "api_proof_generation",
    success: bool = False,
) -> SimpleNamespace:
    metadata = (
        {"api_usage": {"completion_tokens": completion_tokens}}
        if completion_tokens is not None
        else {}
    )
    return SimpleNamespace(
        attempt_id=attempt_id,
        proof=proof,
        prompt_type=prompt_type,
        success=success,
        metadata=metadata,
    )


def test_tactic_count_ignores_layout_and_counts_commands() -> None:
    proof = """by
  constructor
  · intro h
    exact h
  · omega
"""

    assert tactic_command_count(proof) == 4
    assert lexical_proof_token_count("by exact hp") == 3


def test_node_attempt_statistics_prefer_api_tokens_and_include_root() -> None:
    node = SimpleNamespace(
        id="L1",
        proof_attempts=[
            attempt("A-1", "by exact hp", completion_tokens=9),
            attempt("A-2", "by\n  trivial"),
        ],
    )
    root = SimpleNamespace(proof_attempts=[])

    statistics = node_attempt_statistics([node], root)

    assert statistics[0]["node_id"] == "L1"
    assert statistics[0]["attempt_count"] == 2
    assert statistics[0]["base_attempt_count"] == 2
    assert statistics[0]["repair_attempt_count"] == 0
    assert statistics[0]["base_pass_at_4"] is False
    assert statistics[0]["proof_token_count_total"] == 11
    assert statistics[0]["average_proof_token_count"] == 5.5
    assert statistics[0]["tactic_count_total"] == 2
    assert statistics[0]["average_tactic_count"] == 1.0
    assert statistics[0]["attempts"][0]["proof_token_count_source"] == (
        "api_completion_tokens"
    )
    assert statistics[0]["attempts"][1]["proof_token_count_source"] == (
        "lean_lexical"
    )
    assert statistics[1]["node_id"] == "ROOT"
    assert statistics[1]["attempt_count"] == 0
    assert statistics[1]["average_proof_token_count"] is None
    assert statistics[1]["average_tactic_count"] is None


def test_repair_candidate_token_count_does_not_duplicate_shared_json_usage() -> None:
    node = SimpleNamespace(
        id="L1",
        proof_attempts=[
            attempt(
                "A-r1",
                "exact Nat.add_zero n",
                completion_tokens=999,
                prompt_type="api_proof_repair_retrieval",
                success=True,
            )
        ],
    )

    statistics = node_attempt_statistics([node], None)[0]

    assert statistics["base_attempt_count"] == 0
    assert statistics["repair_attempt_count"] == 1
    assert statistics["repair_rescued"] is True
    assert statistics["proof_token_count_total"] == 5
    assert statistics["attempts"][0]["proof_token_count_source"] == (
        "lean_lexical_repair_candidate"
    )


def test_write_node_attempt_statistics_report(tmp_path) -> None:
    report_path = tmp_path / "node_attempt_statistics.jsonl"
    write_node_attempt_statistics_report(
        [
            {
                "sample_index": 3,
                "source_id": "demo",
                "source_split": "test",
                "planner_success": True,
                "prover_success": False,
                "node_attempt_statistics": [
                    {
                        "node_id": "L1",
                        "is_root": False,
                        "attempt_count": 4,
                        "average_proof_token_count": 12.5,
                        "average_tactic_count": 2.0,
                    }
                ],
            }
        ],
        report_path,
    )

    report = report_path.read_text(encoding="utf-8")
    assert '"source_id": "demo"' in report
    assert '"node_id": "L1"' in report
    assert '"average_proof_token_count": 12.5' in report


def test_missing_header_module_is_a_system_failure() -> None:
    issue = SimpleNamespace(
        message=(
            "object file '/tmp/Mathlib/Old.olean' of module "
            "Mathlib.Old does not exist"
        )
    )
    result = SimpleNamespace(
        planner_result=SimpleNamespace(issues=[issue])
    )

    assert system_failure_message(result) == (
        "Lean/header environment incompatibility detected"
    )


def test_unrelated_403_number_does_not_fake_global_api_failure() -> None:
    issue = SimpleNamespace(
        message=(
            "BluePrintRepair JSON was truncated after a mathematical payload "
            "containing the number 403"
        )
    )
    result = SimpleNamespace(
        planner_result=SimpleNamespace(issues=[issue])
    )

    assert system_failure_message(result) is None


def test_http_403_is_a_global_api_failure() -> None:
    issue = SimpleNamespace(
        message="HTTP status 403 Forbidden returned by the model endpoint"
    )
    result = SimpleNamespace(
        planner_result=SimpleNamespace(issues=[issue])
    )

    assert system_failure_message(result) == "global model API failure detected"


def test_selected_rows_exact_source_ids_do_not_expand_to_dataset(tmp_path) -> None:
    dataset = tmp_path / "data.jsonl"
    rows = [
        {
            "id": f"p{index}",
            "split": "test",
            "informal_stmt": f"problem {index}",
            "formal_statement": f"theorem p{index} : True",
            "header": "import Mathlib",
        }
        for index in range(5)
    ]
    dataset.write_text(
        "".join(json.dumps(row) + "\n" for row in rows),
        encoding="utf-8",
    )

    selected = selected_rows(
        [dataset],
        tmp_path / "manifest.jsonl",
        2,
        7,
        ["p3", "p1"],
    )

    assert [row["source_id"] for row in selected] == ["p3", "p1"]
