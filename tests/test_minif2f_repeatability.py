import json

from scripts.run_minif2f_repeatability import (
    blueprint_signature,
    dag_structure_signature,
    recover_node_local_verify_failure,
    summarize,
)


def record(source_id, *, planner, pipeline, nodes, signature_suffix):
    return {
        "source_id": source_id,
        "planner_success": planner,
        "pipeline_success": pipeline,
        "prover_success": pipeline,
        "planner_stage": "completed" if planner else "lean_validation",
        "pipeline_stage": "prover_success" if pipeline else "planner_lean_validation",
        "node_count": nodes,
        "attempt_count": 4,
        "base_attempt_count": 4,
        "repair_attempt_count": 0,
        "repair_rescued_node_count": 0,
        "root_repair_rescued": False,
        "attempt_proof_token_total": 20,
        "attempt_tactic_total": 8,
        "elapsed_seconds": 1,
        "pipeline_result": {
            "blueprint": {
                "nodes": [{
                    "id": "L1",
                    "depends_on": [],
                    "informal_statement": signature_suffix,
                    "lean_decl": f"lemma L1 : {signature_suffix}",
                }]
            }
        },
    }


def test_repeatability_summary_detects_mixed_results_and_blueprints(tmp_path):
    for repetition, row in enumerate(
        [
            record("p", planner=True, pipeline=True, nodes=1, signature_suffix="True"),
            record("p", planner=False, pipeline=False, nodes=2, signature_suffix="False"),
        ],
        start=1,
    ):
        directory = tmp_path / f"rep_{repetition:02d}"
        directory.mkdir()
        (directory / "problem_results.jsonl").write_text(
            json.dumps(row) + "\n", encoding="utf-8"
        )

    report = summarize(tmp_path, ["p"], 2)

    assert report["completed_run_count"] == 2
    problem = report["per_problem"][0]
    assert problem["planner_success_rate"] == 0.5
    assert problem["mixed_pipeline_outcome"] is True
    assert problem["unique_blueprint_count"] == 2
    assert problem["unique_dag_structure_count"] == 1
    assert blueprint_signature(record(
        "p", planner=True, pipeline=True, nodes=1, signature_suffix="True"
    ))
    assert dag_structure_signature(record(
        "p", planner=True, pipeline=True, nodes=1, signature_suffix="True"
    ))


def test_node_local_verify_crash_is_preserved_without_resampling(tmp_path):
    fatal = {
        "error_type": "RuntimeError",
        "error_message": (
            "Verify nodes failed after all parallel calls completed (L2): "
            "LLM did not return valid JSON after 5 attempts"
        ),
    }
    (tmp_path / "fatal_error.json").write_text(json.dumps(fatal), encoding="utf-8")
    sources = [{
        "sample_index": 0,
        "source_id": "p",
        "source_split": "test",
        "informal_stmt": "True",
        "reference_formal_statement": "theorem p : True",
    }]

    assert recover_node_local_verify_failure(tmp_path, sources) is True

    result = json.loads((tmp_path / "problem_results.jsonl").read_text())
    assert result["source_id"] == "p"
    assert result["pipeline_stage"] == "verify_node_failure"
    assert result["pipeline_result"]["exception"] == fatal
    assert not (tmp_path / "fatal_error.json").exists()
    assert (tmp_path / "sample_failure_p.json").is_file()
