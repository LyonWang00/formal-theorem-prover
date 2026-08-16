"""Recompute attempt metrics from persisted proof bodies without API calls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from scripts.run_minif2f_planner_prover_pass4 import (
    node_attempt_statistics,
    summarize,
    write_node_attempt_statistics_report,
)


def _attempt(raw: dict) -> SimpleNamespace:
    return SimpleNamespace(
        attempt_id=raw.get("attempt_id", ""),
        prompt_type=raw.get("prompt_type", ""),
        proof=raw.get("proof", ""),
        success=bool(raw.get("success", False)),
        metadata=raw.get("metadata") or {},
    )


def _node(raw: dict) -> SimpleNamespace:
    return SimpleNamespace(
        id=raw["id"],
        proof_attempts=[_attempt(item) for item in raw.get("proof_attempts", [])],
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    results_path = args.output_dir / "problem_results.jsonl"
    records = [
        json.loads(line)
        for line in results_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    for record in records:
        pipeline = record.get("pipeline_result") or {}
        blueprint = pipeline.get("blueprint") or {}
        nodes = [_node(raw) for raw in blueprint.get("nodes", [])]
        prover_result = pipeline.get("prover_result") or {}
        raw_root = prover_result.get("root_result")
        root = None
        if isinstance(raw_root, dict):
            root = SimpleNamespace(
                proof_attempts=[
                    _attempt(item) for item in raw_root.get("proof_attempts", [])
                ]
            )
        statistics = node_attempt_statistics(nodes, root)
        attempt_count = sum(item["attempt_count"] for item in statistics)
        token_total = sum(item["proof_token_count_total"] for item in statistics)
        tactic_total = sum(item["tactic_count_total"] for item in statistics)
        record.update(
            {
                "schema_version": "minif2f_planner_prover_pass4_problem_v3",
                "node_attempt_statistics": statistics,
                "base_proved_node_count": sum(
                    item["base_pass_at_4"]
                    for item in statistics
                    if not item["is_root"]
                ),
                "repair_rescued_node_count": sum(
                    item["repair_rescued"]
                    for item in statistics
                    if not item["is_root"]
                ),
                "root_base_proved": any(
                    item["is_root"] and item["base_pass_at_4"]
                    for item in statistics
                ),
                "root_repair_rescued": any(
                    item["is_root"] and item["repair_rescued"]
                    for item in statistics
                ),
                "attempt_count": attempt_count,
                "base_attempt_count": sum(
                    item["base_attempt_count"] for item in statistics
                ),
                "repair_attempt_count": sum(
                    item["repair_attempt_count"] for item in statistics
                ),
                "attempt_proof_token_total": token_total,
                "average_attempt_proof_token_count": (
                    round(token_total / attempt_count, 4)
                    if attempt_count
                    else None
                ),
                "attempt_tactic_total": tactic_total,
                "average_attempt_tactic_count": (
                    round(tactic_total / attempt_count, 4)
                    if attempt_count
                    else None
                ),
            }
        )
    temporary = results_path.with_suffix(".metrics.tmp")
    temporary.write_text(
        "".join(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )
    temporary.replace(results_path)
    write_node_attempt_statistics_report(
        records,
        args.output_dir / "node_attempt_statistics.jsonl",
    )
    summary = summarize(results_path, args.output_dir / "summary.json")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
