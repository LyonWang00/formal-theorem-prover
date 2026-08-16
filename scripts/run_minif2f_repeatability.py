#!/usr/bin/env python3
"""Run repeated independent Planner/Prover samples and summarize variance."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def atomic_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def blueprint_signature(row: dict[str, Any]) -> str | None:
    blueprint = ((row.get("pipeline_result") or {}).get("blueprint") or {})
    nodes = blueprint.get("nodes") or []
    if not nodes:
        return None
    canonical = [
        {
            "id": node.get("id"),
            "depends_on": node.get("depends_on") or [],
            "informal_statement": node.get("informal_statement"),
            "informal_proof": node.get("informal_proof"),
            "logical_ideas": node.get("logical_ideas") or [],
            "lean_decl": node.get("lean_decl"),
        }
        for node in nodes
    ]
    payload = json.dumps(canonical, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def dag_structure_signature(row: dict[str, Any]) -> str | None:
    """Hash only node count/order, edges, and root dependencies."""

    blueprint = ((row.get("pipeline_result") or {}).get("blueprint") or {})
    nodes = blueprint.get("nodes") or []
    if not nodes:
        return None
    positions = {str(node.get("id")): index for index, node in enumerate(nodes)}
    canonical = {
        "node_count": len(nodes),
        "edges": sorted(
            (positions[dependency], index)
            for index, node in enumerate(nodes)
            for dependency in (node.get("depends_on") or [])
            if dependency in positions
        ),
        "unknown_dependencies": sorted(
            (str(dependency), index)
            for index, node in enumerate(nodes)
            for dependency in (node.get("depends_on") or [])
            if dependency not in positions
        ),
        "root_dependencies": sorted(
            positions[dependency]
            for dependency in (blueprint.get("root_dependencies") or [])
            if dependency in positions
        ),
    }
    payload = json.dumps(canonical, sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def population_std(values: list[float]) -> float:
    if not values:
        return 0.0
    mean = sum(values) / len(values)
    return math.sqrt(sum((value - mean) ** 2 for value in values) / len(values))


def rate(count: int, total: int) -> float:
    return count / total if total else 0.0


def recover_node_local_verify_failure(
    repetition_dir: Path,
    source_rows: list[dict[str, Any]],
) -> bool:
    """Preserve one interrupted sample instead of silently resampling it."""

    fatal_path = repetition_dir / "fatal_error.json"
    results_path = repetition_dir / "problem_results.jsonl"
    if not fatal_path.is_file():
        return False
    fatal = json.loads(fatal_path.read_text(encoding="utf-8"))
    message = str(fatal.get("error_message") or "")
    node_local_markers = (
        "Verify nodes failed after all parallel calls completed",
        "LLM did not return valid JSON after 5 attempts",
    )
    if not any(marker in message for marker in node_local_markers):
        return False
    completed = {row["source_id"] for row in load_jsonl(results_path)}
    source = next(
        (row for row in source_rows if row["source_id"] not in completed),
        None,
    )
    if source is None:
        return False
    record = {
        "schema_version": "minif2f_planner_prover_pass4_problem_v3",
        "sample_index": source["sample_index"],
        "source_id": source["source_id"],
        "source_split": source["source_split"],
        "source_informal_statement": source["informal_stmt"],
        "reference_formal_statement": source["reference_formal_statement"],
        "model": "deepseek-v4-flash",
        "attempts_per_node": 4,
        "elapsed_seconds": 0.0,
        "pipeline_success": False,
        "pipeline_stage": "verify_node_failure",
        "planner_success": False,
        "planner_stage": "verify_node_failure",
        "planner_mode": "lean",
        "lean_native_route_verified": True,
        "node_count": 0,
        "planner_generated_subgoal_count": 0,
        "proved_node_count": 0,
        "base_proved_node_count": 0,
        "repair_rescued_node_count": 0,
        "root_attempted": False,
        "root_proved": False,
        "root_base_proved": False,
        "root_repair_rescued": False,
        "root_attempt_count": 0,
        "node_attempt_statistics": [],
        "attempt_count": 0,
        "base_attempt_count": 0,
        "repair_attempt_count": 0,
        "attempt_proof_token_total": 0,
        "average_attempt_proof_token_count": None,
        "attempt_tactic_total": 0,
        "average_attempt_tactic_count": None,
        "prover_success": False,
        "pipeline_result": {
            "stage": "verify_node_failure",
            "success": False,
            "exception": fatal,
            "note": (
                "The independent sample was retained as failed and was not "
                "resampled. Per-node Verify results remain in verify_results.jsonl."
            ),
        },
    }
    with results_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
    archived = repetition_dir / f"sample_failure_{source['source_id']}.json"
    fatal_path.replace(archived)
    return True


def summarize(output_dir: Path, source_ids: list[str], repetitions: int) -> dict:
    tagged_rows: list[tuple[int, dict[str, Any]]] = []
    for repetition in range(1, repetitions + 1):
        path = output_dir / f"rep_{repetition:02d}" / "problem_results.jsonl"
        tagged_rows.extend((repetition, row) for row in load_jsonl(path))

    per_problem: list[dict[str, Any]] = []
    for source_id in source_ids:
        samples = [
            (repetition, row)
            for repetition, row in tagged_rows
            if row.get("source_id") == source_id
        ]
        rows = [row for _, row in samples]
        node_counts = [float(row.get("node_count", 0)) for row in rows]
        signatures = [
            signature
            for row in rows
            if (signature := blueprint_signature(row)) is not None
        ]
        dag_signatures = [
            signature
            for row in rows
            if (signature := dag_structure_signature(row)) is not None
        ]
        planner_success_count = sum(bool(row.get("planner_success")) for row in rows)
        pipeline_success_count = sum(bool(row.get("pipeline_success")) for row in rows)
        prover_success_count = sum(bool(row.get("prover_success")) for row in rows)
        attempt_count = sum(int(row.get("attempt_count", 0)) for row in rows)
        proof_tokens = sum(int(row.get("attempt_proof_token_total", 0)) for row in rows)
        tactics = sum(int(row.get("attempt_tactic_total", 0)) for row in rows)
        repair_attempts = sum(int(row.get("repair_attempt_count", 0)) for row in rows)
        rescued_nodes = sum(int(row.get("repair_rescued_node_count", 0)) for row in rows)
        rescued_roots = sum(bool(row.get("root_repair_rescued")) for row in rows)
        per_problem.append(
            {
                "source_id": source_id,
                "completed_repetitions": len(rows),
                "planner_success_count": planner_success_count,
                "planner_success_rate": rate(planner_success_count, len(rows)),
                "pipeline_success_count": pipeline_success_count,
                "pipeline_success_rate": rate(pipeline_success_count, len(rows)),
                "pipeline_success_bernoulli_std": (
                    math.sqrt(
                        rate(pipeline_success_count, len(rows))
                        * (1 - rate(pipeline_success_count, len(rows)))
                    )
                    if rows
                    else 0.0
                ),
                "conditional_prover_success_rate_given_planner_success": (
                    pipeline_success_count / planner_success_count
                    if planner_success_count
                    else 0.0
                ),
                "prover_success_count": prover_success_count,
                "prover_success_rate": rate(prover_success_count, len(rows)),
                "planner_stage_distribution": dict(
                    sorted(Counter(str(row.get("planner_stage")) for row in rows).items())
                ),
                "pipeline_stage_distribution": dict(
                    sorted(Counter(str(row.get("pipeline_stage")) for row in rows).items())
                ),
                "node_count_values": [int(value) for value in node_counts],
                "node_count_mean": sum(node_counts) / len(node_counts) if node_counts else 0.0,
                "node_count_population_std": population_std(node_counts),
                "node_count_min": int(min(node_counts)) if node_counts else 0,
                "node_count_max": int(max(node_counts)) if node_counts else 0,
                "unique_blueprint_count": len(set(signatures)),
                "most_common_blueprint_frequency": (
                    Counter(signatures).most_common(1)[0][1] if signatures else 0
                ),
                "unique_dag_structure_count": len(set(dag_signatures)),
                "most_common_dag_structure_frequency": (
                    Counter(dag_signatures).most_common(1)[0][1]
                    if dag_signatures
                    else 0
                ),
                "attempt_count": attempt_count,
                "base_attempt_count": sum(int(row.get("base_attempt_count", 0)) for row in rows),
                "repair_attempt_count": repair_attempts,
                "repair_rescued_node_count": rescued_nodes,
                "repair_rescued_root_count": rescued_roots,
                "average_attempt_proof_token_count": (
                    proof_tokens / attempt_count if attempt_count else None
                ),
                "average_attempt_tactic_count": tactics / attempt_count if attempt_count else None,
                "elapsed_seconds_mean": (
                    sum(float(row.get("elapsed_seconds", 0)) for row in rows) / len(rows)
                    if rows
                    else 0.0
                ),
                "mixed_planner_outcome": 0 < planner_success_count < len(rows),
                "mixed_pipeline_outcome": 0 < pipeline_success_count < len(rows),
                "repetition_results": [
                    {
                        "repetition": repetition,
                        "planner_success": bool(row.get("planner_success")),
                        "pipeline_success": bool(row.get("pipeline_success")),
                        "pipeline_stage": row.get("pipeline_stage"),
                        "node_count": row.get("node_count", 0),
                        "blueprint_signature": blueprint_signature(row),
                        "dag_structure_signature": dag_structure_signature(row),
                        "attempt_count": row.get("attempt_count", 0),
                        "repair_attempt_count": row.get("repair_attempt_count", 0),
                    }
                    for repetition, row in samples
                ],
            }
        )

    rows = [row for _, row in tagged_rows]
    node_count_values = [float(row.get("node_count", 0)) for row in rows]
    attempt_count = sum(int(row.get("attempt_count", 0)) for row in rows)
    report = {
        "schema_version": "minif2f_repeatability_v1",
        "requested_problem_count": len(source_ids),
        "requested_repetitions_per_problem": repetitions,
        "requested_run_count": len(source_ids) * repetitions,
        "completed_run_count": len(rows),
        "source_ids": source_ids,
        "overall": {
            "planner_success_count": sum(bool(row.get("planner_success")) for row in rows),
            "planner_success_rate": rate(
                sum(bool(row.get("planner_success")) for row in rows), len(rows)
            ),
            "pipeline_success_count": sum(bool(row.get("pipeline_success")) for row in rows),
            "pipeline_success_rate": rate(
                sum(bool(row.get("pipeline_success")) for row in rows), len(rows)
            ),
            "node_count_mean": (
                sum(node_count_values) / len(node_count_values) if node_count_values else 0.0
            ),
            "node_count_population_std": population_std(node_count_values),
            "attempt_count": attempt_count,
            "repair_attempt_count": sum(int(row.get("repair_attempt_count", 0)) for row in rows),
            "repair_rescued_node_count": sum(int(row.get("repair_rescued_node_count", 0)) for row in rows),
            "repair_rescued_root_count": sum(bool(row.get("root_repair_rescued")) for row in rows),
            "average_attempt_proof_token_count": (
                sum(int(row.get("attempt_proof_token_total", 0)) for row in rows) / attempt_count
                if attempt_count
                else None
            ),
            "average_attempt_tactic_count": (
                sum(int(row.get("attempt_tactic_total", 0)) for row in rows) / attempt_count
                if attempt_count
                else None
            ),
            "mixed_planner_outcome_problem_count": sum(
                item["mixed_planner_outcome"] for item in per_problem
            ),
            "mixed_pipeline_outcome_problem_count": sum(
                item["mixed_pipeline_outcome"] for item in per_problem
            ),
            "problem_with_multiple_blueprints_count": sum(
                item["unique_blueprint_count"] > 1 for item in per_problem
            ),
            "problem_with_multiple_dag_structures_count": sum(
                item["unique_dag_structure_count"] > 1 for item in per_problem
            ),
        },
        "per_problem": per_problem,
    }
    atomic_json(output_dir / "repeatability_summary.json", report)
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, nargs="+", required=True)
    parser.add_argument("--lean-project", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--problem-count", type=int, default=10)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--enable-prover-repair", action="store_true")
    parser.add_argument("--max-prover-repair-rounds", type=int, default=5)
    args = parser.parse_args()
    if not os.environ.get("DEEPSEEK_API_KEY"):
        raise RuntimeError("DEEPSEEK_API_KEY is required")
    reference = load_jsonl(args.reference_manifest)
    if len(reference) < args.problem_count:
        raise ValueError("reference manifest is shorter than --problem-count")
    source_ids = [str(row["source_id"]) for row in reference[: args.problem_count]]
    source_rows = reference[: args.problem_count]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    atomic_json(
        args.output_dir / "experiment_manifest.json",
        {
            "source_manifest": str(args.reference_manifest),
            "source_ids": source_ids,
            "problem_count": args.problem_count,
            "repetitions": args.repetitions,
            "independent_blueprints_per_problem": args.repetitions,
        },
    )
    runner = Path(__file__).with_name("run_minif2f_planner_prover_pass4.py")
    for repetition in range(1, args.repetitions + 1):
        repetition_dir = args.output_dir / f"rep_{repetition:02d}"
        if recover_node_local_verify_failure(repetition_dir, source_rows):
            summarize(args.output_dir, source_ids, args.repetitions)
            print(
                f"[repeatability {repetition}/{args.repetitions}] preserved "
                "one node-local Verify failure without resampling",
                flush=True,
            )
        command = [
            sys.executable,
            str(runner),
            "--dataset",
            *(str(path) for path in args.dataset),
            "--lean-project",
            str(args.lean_project),
            "--output-dir",
            str(repetition_dir),
            "--count",
            str(args.problem_count),
            "--seed",
            str(2026081300 + repetition),
            "--timeout",
            str(args.timeout),
            "--max-prover-repair-rounds",
            str(args.max_prover_repair_rounds),
        ]
        for source_id in source_ids:
            command.extend(("--source-id", source_id))
        if args.enable_prover_repair:
            command.append("--enable-prover-repair")
        print(
            f"[repeatability {repetition}/{args.repetitions}] starting/resuming",
            flush=True,
        )
        completed = subprocess.run(command, check=False)
        summarize(args.output_dir, source_ids, args.repetitions)
        if completed.returncode != 0:
            raise RuntimeError(
                f"repeatability repetition {repetition} failed with exit code "
                f"{completed.returncode}; partial results were summarized"
            )
    print(
        json.dumps(
            summarize(args.output_dir, source_ids, args.repetitions),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
