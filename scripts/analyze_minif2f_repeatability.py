#!/usr/bin/env python3
"""Create a compact audit of a completed repeatability experiment."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def rows(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def mean_std(values: list[float]) -> dict[str, float]:
    mean = sum(values) / len(values) if values else 0.0
    std = (
        sum((value - mean) ** 2 for value in values) / len(values)
    ) ** 0.5 if values else 0.0
    return {"mean": mean, "population_std": std, "min": min(values), "max": max(values)}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("experiment_dir", type=Path)
    args = parser.parse_args()
    root = args.experiment_dir
    summary = json.loads((root / "repeatability_summary.json").read_text())
    problem_rows = [
        row
        for directory in sorted(root.glob("rep_*"))
        for row in rows(directory / "problem_results.jsonl")
    ]
    round_summaries = [
        json.loads((directory / "summary.json").read_text())
        for directory in sorted(root.glob("rep_*"))
    ]
    verify_records = [
        record
        for directory in sorted(root.glob("rep_*"))
        for record in rows(directory / "verify_results.jsonl")
    ]
    blueprint_records = [
        record
        for directory in sorted(root.glob("rep_*"))
        for record in rows(directory / "blueprint_repair_results.jsonl")
    ]
    repair_attempts = [
        attempt
        for directory in sorted(root.glob("rep_*"))
        for attempt in rows(directory / "attempts.jsonl")
        if attempt.get("prompt_type") == "api_proof_repair_retrieval"
    ]
    per_problem = [
        {
            "source_id": item["source_id"],
            "planner_success": f"{item['planner_success_count']}/10",
            "pipeline_success": f"{item['pipeline_success_count']}/10",
            "conditional_success_given_planner": item[
                "conditional_prover_success_rate_given_planner_success"
            ],
            "node_count_range": [item["node_count_min"], item["node_count_max"]],
            "node_count_population_std": item["node_count_population_std"],
            "unique_full_blueprints": item["unique_blueprint_count"],
            "unique_dag_structures": item["unique_dag_structure_count"],
            "dominant_dag_frequency": item["most_common_dag_structure_frequency"],
            "repair_attempts": item["repair_attempt_count"],
            "repair_rescued_nodes": item["repair_rescued_node_count"],
            "repair_rescued_roots": item["repair_rescued_root_count"],
            "average_attempt_tokens": item["average_attempt_proof_token_count"],
            "average_attempt_tactics": item["average_attempt_tactic_count"],
        }
        for item in summary["per_problem"]
    ]
    report = {
        "schema_version": "minif2f_repeatability_audit_v1",
        "integrity": {
            "run_count": len(problem_rows),
            "unique_source_repetition_pairs": sum(
                item["completed_repetitions"] for item in summary["per_problem"]
            ),
            "lean_native_route_count": sum(
                bool(row.get("lean_native_route_verified")) for row in problem_rows
            ),
            "unhandled_fatal_errors": [
                str(path.relative_to(root)) for path in root.rglob("fatal_error.json")
            ],
            "preserved_sample_failures": [
                str(path.relative_to(root))
                for path in root.rglob("sample_failure_*.json")
            ],
        },
        "overall": summary["overall"],
        "round_variation": {
            "planner_successes_per_10": mean_std(
                [row["planner_success_count"] for row in round_summaries]
            ),
            "pipeline_successes_per_10": mean_std(
                [row["prover_success_count"] for row in round_summaries]
            ),
            "nodes_per_round": mean_std(
                [row["total_node_count"] for row in round_summaries]
            ),
            "repair_attempts_per_round": mean_std(
                [row["repair_attempt_count"] for row in round_summaries]
            ),
        },
        "stage_distribution": {
            "planner": dict(sorted(Counter(row["planner_stage"] for row in problem_rows).items())),
            "pipeline": dict(sorted(Counter(row["pipeline_stage"] for row in problem_rows).items())),
        },
        "verify": {
            "record_types": dict(sorted(Counter(row.get("record_type") for row in verify_records).items())),
            "error_types": dict(sorted(Counter(
                row.get("error_type")
                for row in verify_records
                if row.get("record_type") == "verification_error"
            ).items())),
            "semantic_mismatch_result_count": sum(
                row.get("record_type") == "verification_result"
                and not row.get("formal_statement_correct", True)
                for row in verify_records
            ),
        },
        "blueprint_repair_states": dict(
            sorted(Counter(str(row.get("state")) for row in blueprint_records).items())
        ),
        "prover_repair_trigger_details": Counter(
            str((row.get("metadata") or {}).get("repaired_failure_detail"))
            for row in repair_attempts
        ).most_common(),
        "attempts": {
            "base": sum(row.get("base_attempt_count", 0) for row in problem_rows),
            "repair": sum(row.get("repair_attempt_count", 0) for row in problem_rows),
            "ordinary_nodes_proved": sum(row.get("proved_node_count", 0) for row in problem_rows),
            "ordinary_nodes_base_pass4": sum(row.get("base_proved_node_count", 0) for row in problem_rows),
            "ordinary_nodes_repair_rescued": sum(row.get("repair_rescued_node_count", 0) for row in problem_rows),
            "roots_repair_rescued": sum(bool(row.get("root_repair_rescued")) for row in problem_rows),
        },
        "per_problem": per_problem,
    }
    output = root / "repeatability_audit.json"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
