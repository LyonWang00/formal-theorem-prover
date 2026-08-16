"""Aggregate the fixed two-batch, three-architecture comparison."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ARCH_DIRS = {
    "original": ("batch02_original", "batch03_original"),
    "logical_ideas": ("batch02_logical_ideas", "batch03_logical_ideas"),
    "rootfirst": ("batch02_rootfirst", "batch03_rootfirst"),
}
REFERENCE_DETAILS = {"unknown_identifier", "unknown_constant", "invalid_field"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def summaries(root: Path, architecture: str) -> list[dict[str, Any]]:
    return [
        json.loads((root / name / "summary.json").read_text(encoding="utf-8"))
        for name in ARCH_DIRS[architecture]
    ]


def current_records(root: Path, architecture: str) -> list[dict[str, Any]]:
    filename = "results.jsonl" if architecture == "rootfirst" else "problem_results.jsonl"
    return [
        row
        for name in ARCH_DIRS[architecture]
        for row in load_jsonl(root / name / filename)
    ]


def proof_attempts(records: list[dict[str, Any]], architecture: str):
    for row in records:
        if architecture == "rootfirst":
            for node in row["result"]["blueprint"]["nodes"]:
                for attempt in node.get("attempts", []):
                    yield row["problem_id"], node["id"], attempt
        else:
            for node in row.get("node_attempt_statistics", []):
                for attempt in node.get("attempts", []):
                    yield row["source_id"], node["node_id"], attempt


def is_repair(attempt: dict[str, Any], architecture: str) -> bool:
    if architecture == "rootfirst":
        return attempt.get("kind") in {"proof_repair", "disproof_repair"}
    return attempt.get("prompt_type") == "api_proof_repair_retrieval"


def current_aggregate(root: Path, architecture: str) -> dict[str, Any]:
    records = current_records(root, architecture)
    batch_summaries = summaries(root, architecture)
    success_key = "success" if architecture == "rootfirst" else "prover_success"
    elapsed = [float(row["elapsed_seconds"]) for row in records]
    if architecture == "rootfirst":
        attempts = list(proof_attempts(records, architecture))
    else:
        attempts = [
            (str(attempt["problem_id"]), str(attempt["node_id"]), attempt)
            for name in ARCH_DIRS[architecture]
            for attempt in load_jsonl(root / name / "attempts.jsonl")
        ]
    repairs = [item for item in attempts if is_repair(item[2], architecture)]
    repair_details = Counter()
    repair_success = Counter()
    grounding = {
        "reference_repair_attempt_count": 0,
        "index_available_count": 0,
        "cited_index_name_count": 0,
        "proof_used_index_name_count": 0,
        "grounding_pass_count": 0,
        "missing_required_index_evidence_count": 0,
        "violations": [],
    }
    for problem_id, node_id, attempt in repairs:
        metadata = attempt.get("metadata") or {}
        detail = str(metadata.get("repaired_failure_detail") or "unknown")
        repair_details[detail] += 1
        if attempt.get("success"):
            repair_success[detail] += 1
        if detail not in REFERENCE_DETAILS and not metadata.get(
            "retrieval_grounding_required"
        ):
            continue
        grounding["reference_repair_attempt_count"] += 1
        retrieval = metadata.get("retrieval") or {}
        if retrieval.get("index_available"):
            grounding["index_available_count"] += 1
        if metadata.get("cited_retrieved_declaration_names"):
            grounding["cited_index_name_count"] += 1
        if metadata.get("used_retrieved_declaration_names"):
            grounding["proof_used_index_name_count"] += 1
        if metadata.get("declarations_grounded") is True:
            grounding["grounding_pass_count"] += 1
        if metadata.get("missing_required_index_evidence"):
            grounding["missing_required_index_evidence_count"] += 1
        if attempt.get("success") and not (
            retrieval.get("index_available")
            and metadata.get("used_retrieved_declaration_names")
            and metadata.get("declarations_grounded") is True
        ):
            grounding["violations"].append(
                {"problem_id": problem_id, "node_id": node_id}
            )

    result: dict[str, Any] = {
        "problem_count": len(records),
        "success_count": sum(bool(row[success_key]) for row in records),
        "success_rate": (
            sum(bool(row[success_key]) for row in records) / len(records)
            if records
            else 0.0
        ),
        "total_elapsed_seconds": sum(elapsed),
        "average_elapsed_seconds": sum(elapsed) / len(elapsed) if elapsed else 0.0,
        "repair_attempt_count": len(repairs),
        "repair_success_count": sum(bool(item[2].get("success")) for item in repairs),
        "repair_failure_detail_distribution": dict(repair_details),
        "repair_success_detail_distribution": dict(repair_success),
        "reference_repair_grounding_audit": grounding,
        "per_problem_success": {
            str(row["problem_id"] if architecture == "rootfirst" else row["source_id"]): bool(
                row[success_key]
            )
            for row in records
        },
    }
    summary_attempt_count = sum(
        int(
            item[
                "total_attempt_count"
                if architecture == "rootfirst"
                else "attempt_count"
            ]
        )
        for item in batch_summaries
    )
    result["attempt_count"] = summary_attempt_count
    result["average_attempt_proof_token_count"] = (
        sum(
            float(item["average_attempt_proof_token_count"])
            * int(
                item[
                    "total_attempt_count"
                    if architecture == "rootfirst"
                    else "attempt_count"
                ]
            )
            for item in batch_summaries
        )
        / summary_attempt_count
        if summary_attempt_count
        else 0.0
    )
    result["average_attempt_tactic_count"] = (
        sum(
            float(item["average_attempt_tactic_count"])
            * int(
                item[
                    "total_attempt_count"
                    if architecture == "rootfirst"
                    else "attempt_count"
                ]
            )
            for item in batch_summaries
        )
        / summary_attempt_count
        if summary_attempt_count
        else 0.0
    )
    if architecture == "rootfirst":
        effectiveness = [
            item
            for row in records
            for item in row.get("refinement_effectiveness", [])
        ]
        result.update(
            {
                "direct_root_success_count": sum(
                    bool(row["direct_root_success"]) for row in records
                ),
                "refined_success_count": sum(
                    row["stage"] == "refined_success" for row in records
                ),
                "helper_node_count": sum(
                    int(row["helper_node_count"]) for row in records
                ),
                "effective_helper_count": sum(
                    bool(item.get("effective_refinement"))
                    for item in effectiveness
                ),
                "helper_effectiveness_status_distribution": dict(
                    Counter(str(item.get("status")) for item in effectiveness)
                ),
            }
        )
    else:
        result.update(
            {
                "planner_success_count": sum(
                    bool(row["planner_success"]) for row in records
                ),
                "planner_generated_subgoal_count": sum(
                    int(row["planner_generated_subgoal_count"])
                    for row in records
                ),
                "proved_node_count": sum(
                    int(summary["proved_node_count"])
                    for summary in batch_summaries
                ),
                "repair_rescued_node_count": sum(
                    int(summary["repair_rescued_node_count"])
                    for summary in batch_summaries
                ),
                "root_repair_rescued_count": sum(
                    int(summary["root_repair_rescued_count"])
                    for summary in batch_summaries
                ),
            }
        )
    return result


def previous_aggregate(root: Path, architecture: str) -> dict[str, Any]:
    batch_summaries = [
        json.loads(
            (root / batch / architecture / "summary.json").read_text(
                encoding="utf-8"
            )
        )
        for batch in ("batch02", "batch03")
    ]
    if architecture == "rootfirst":
        return {
            "problem_count": sum(int(item["processed"]) for item in batch_summaries),
            "success_count": sum(int(item["success_count"]) for item in batch_summaries),
            "helper_node_count": sum(
                float(item["average_helper_node_count"]) * int(item["processed"])
                for item in batch_summaries
            ),
            "refined_success_count": sum(
                int(item["refined_success_count"]) for item in batch_summaries
            ),
        }
    return {
        "problem_count": sum(
            int(item["processed_problem_count"]) for item in batch_summaries
        ),
        "success_count": sum(
            int(item["prover_success_count"]) for item in batch_summaries
        ),
        "planner_success_count": sum(
            int(item["planner_success_count"]) for item in batch_summaries
        ),
        "planner_generated_subgoal_count": sum(
            int(item["total_node_count"]) for item in batch_summaries
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--previous-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current = {
        arch: current_aggregate(args.current_root, arch) for arch in ARCH_DIRS
    }
    previous = {
        arch: previous_aggregate(args.previous_root, arch) for arch in ARCH_DIRS
    }
    report = {
        "schema_version": "priority1_three_architecture_comparison_v1",
        "controlled_parameters": {
            "pass_k": 4,
            "proof_repair_rounds_per_node": 2,
            "proof_repair_candidates_per_node": 2,
            "proof_repair_candidates_per_generation": 1,
            "formalization_repair_rounds": 2,
            "max_total_graph_nodes": 4,
        },
        "current": current,
        "previous": previous,
        "success_delta": {
            arch: current[arch]["success_count"] - previous[arch]["success_count"]
            for arch in ARCH_DIRS
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
