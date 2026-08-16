"""Compare the fixed-manifest baseline with retrieval-guided Prover repair."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import sys


def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def main() -> int:
    old_root, new_root = map(Path, sys.argv[1:3])
    output_path = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    old = {row["source_id"]: row for row in rows(old_root / "problem_results.jsonl")}
    new = {row["source_id"]: row for row in rows(new_root / "problem_results.jsonl")}
    old_manifest = (old_root / "sample_manifest.jsonl").read_bytes()
    new_manifest = (new_root / "sample_manifest.jsonl").read_bytes()
    attempts = rows(new_root / "attempts.jsonl")
    repair = [row for row in attempts if row["prompt_type"] == "api_proof_repair_retrieval"]
    base = [row for row in attempts if row["prompt_type"] != "api_proof_repair_retrieval"]

    transitions = Counter(
        (
            bool(old[source_id]["pipeline_success"]),
            bool(new[source_id]["pipeline_success"]),
        )
        for source_id in sorted(set(old) & set(new))
    )
    planner_transitions = Counter(
        (
            bool(old[source_id]["planner_success"]),
            bool(new[source_id]["planner_success"]),
        )
        for source_id in sorted(set(old) & set(new))
    )
    repaired_nodes = {
        (row["problem_id"], row["node_id"])
        for row in repair
        if row.get("success")
    }
    repaired_problems = {item[0] for item in repaired_nodes}
    per_node = defaultdict(list)
    for row in repair:
        per_node[(row["problem_id"], row["node_id"])].append(row)
    rounds = Counter(int((row.get("metadata") or {}).get("repair_round") or 0) for row in repair)
    successful_rounds = Counter(
        int((row.get("metadata") or {}).get("repair_round") or 0)
        for row in repair
        if row.get("success")
    )
    trigger_errors = Counter(
        str((row.get("metadata") or {}).get("repaired_failure_detail") or "")
        for row in repair
    )
    retrieved_names: set[str] = set()
    indexed_attempts = 0
    ungrounded = 0
    for row in repair:
        metadata = row.get("metadata") or {}
        retrieval = metadata.get("retrieval") or {}
        candidates = retrieval.get("candidate_declarations") or []
        if retrieval.get("index_available"):
            indexed_attempts += 1
        for candidate in candidates:
            if isinstance(candidate, dict) and candidate.get("full_name"):
                retrieved_names.add(str(candidate["full_name"]))
        if metadata.get("declarations_grounded") is False:
            ungrounded += 1

    old_success = {source_id for source_id, row in old.items() if row["pipeline_success"]}
    new_success = {source_id for source_id, row in new.items() if row["pipeline_success"]}
    common_planner_success = {
        source_id
        for source_id in set(old) & set(new)
        if old[source_id]["planner_success"] and new[source_id]["planner_success"]
    }
    repaired_pipeline_success = sorted(repaired_problems & new_success)
    report = {
        "manifest": {
            "identical_bytes": old_manifest == new_manifest,
            "old_sha256": hashlib.sha256(old_manifest).hexdigest(),
            "new_sha256": hashlib.sha256(new_manifest).hexdigest(),
            "old_count": len(rows(old_root / "sample_manifest.jsonl")),
            "new_count": len(rows(new_root / "sample_manifest.jsonl")),
        },
        "summaries": {
            "old": json.loads((old_root / "summary.json").read_text()),
            "new": json.loads((new_root / "summary.json").read_text()),
        },
        "paired": {
            "common_problem_count": len(set(old) & set(new)),
            "planner_transitions": {str(key): value for key, value in planner_transitions.items()},
            "pipeline_transitions": {str(key): value for key, value in transitions.items()},
            "newly_solved": sorted(new_success - old_success),
            "regressed": sorted(old_success - new_success),
            "solved_in_both": sorted(old_success & new_success),
            "pipeline_transitions_when_both_planners_succeeded": {
                str(key): value
                for key, value in Counter(
                    (
                        bool(old[source_id]["pipeline_success"]),
                        bool(new[source_id]["pipeline_success"]),
                    )
                    for source_id in common_planner_success
                ).items()
            },
        },
        "repair": {
            "base_attempt_count": len(base),
            "repair_attempt_count": len(repair),
            "unique_triggered_nodes": len(per_node),
            "unique_triggered_problems": len({item[0] for item in per_node}),
            "successful_repair_attempt_count": sum(bool(row.get("success")) for row in repair),
            "rescued_node_count": len(repaired_nodes),
            "problems_with_rescued_node_count": len(repaired_problems),
            "pipeline_successes_with_rescued_nodes": repaired_pipeline_success,
            "pipeline_success_with_rescued_node_count": len(repaired_pipeline_success),
            "rescued_nodes": sorted([list(item) for item in repaired_nodes]),
            "attempts_by_round": dict(sorted(rounds.items())),
            "successes_by_round": dict(sorted(successful_rounds.items())),
            "trigger_failure_details": trigger_errors.most_common(),
            "attempts_with_environment_index": indexed_attempts,
            "unique_retrieved_declaration_names": len(retrieved_names),
            "ungrounded_declared_name_attempts": ungrounded,
            "nodes_exhausting_round_five": sum(
                any(int((row.get("metadata") or {}).get("repair_round") or 0) == 5 for row in node_rows)
                and not any(row.get("success") for row in node_rows)
                for node_rows in per_node.values()
            ),
        },
    }
    rendered = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    if output_path is not None:
        output_path.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
