#!/usr/bin/env python3
"""Audit source availability and hard constraints for EI difficulty ablation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


REQUESTS = {
    "A_medium_hard": {"medium": 300, "hard": 100, "easy": 50, "repair": 0, "replay": 50},
    "B_medium_only": {"medium": 400, "hard": 50, "easy": 0, "repair": 0, "replay": 50},
    "C_add_repair": {"medium": 300, "hard": 100, "easy": 50, "repair": 100, "replay": 50},
    "D_add_replay": {"medium": 300, "hard": 100, "easy": 50, "repair": 0, "replay": 150},
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8-sig") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    output = project / "outputs/expert_iteration/difficulty_ablation"

    dynamic = read_jsonl(project / "outputs/expert_iteration/round0/dynamic_difficulty/dynamic_difficulty_manifest.jsonl")
    repairs = read_jsonl(project / "outputs/expert_iteration/ei_ablation/frontier_repair_v2/repair_verified.jsonl")
    h0 = read_jsonl(project / "outputs/stage2_data_ratio_ablation/hard_a_ablation/H0_no_hard/manifest.jsonl")

    by_theorem_id = {str(row["theorem_id"]): row for row in dynamic}
    difficulty = Counter(str(row["difficulty"]).lower() for row in dynamic)
    difficulty_source = Counter(
        (str(row["difficulty"]).lower(), str(row.get("source") or "unknown")) for row in dynamic
    )
    valid_dynamic = [
        row for row in dynamic
        if str(row["difficulty"]).lower() in {"easy", "medium", "hard"}
        and int(row.get("success_count") or 0) > 0
        and not row.get("zero_success_subtype")
    ]

    repair_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in repairs:
        if (
            row.get("repair_verified") is True
            and row.get("selected_verified_proof")
            and not any(token in str(row["selected_verified_proof"]).lower() for token in ("sorry", "admit", "axiom"))
        ):
            repair_groups[str(row["aggregation_group_id"])].append(row)
    repair_difficulty = Counter()
    for group in repair_groups:
        source = by_theorem_id.get(group)
        repair_difficulty[str(source.get("difficulty") if source else "unmatched").lower()] += 1

    dynamic_theorem_ids = {str(row["theorem_id"]) for row in valid_dynamic}
    repair_group_ids = set(repair_groups)
    repair_outside_valid_dynamic = repair_group_ids - dynamic_theorem_ids
    repair_outside_breakdown = Counter(
        str(by_theorem_id[group].get("difficulty") if group in by_theorem_id else "unmatched").lower()
        for group in repair_outside_valid_dynamic
    )

    h0_source = Counter(str(row.get("sampling_source") or "unknown") for row in h0)
    h0_reference = sum(row.get("reference_proof_verified") is True for row in h0)
    h0_verified = sum(row.get("pantograph_verified") is True for row in h0)

    availability = {
        "medium": difficulty["medium"],
        "hard": sum(
            1 for row in dynamic
            if str(row["difficulty"]).lower() == "hard" and int(row.get("success_count") or 0) == 1
        ),
        "easy": difficulty["easy"],
        "repair_unique_theorem_groups": len(repair_groups),
        "repair_unique_groups_outside_valid_dynamic": len(repair_outside_valid_dynamic),
        "replay_wb": h0_source["WB"],
        "replay_ld": h0_source["LD"],
    }
    arms: dict[str, Any] = {}
    for name, requested in REQUESTS.items():
        realized = {
            "medium": min(requested["medium"], availability["medium"]),
            "hard": min(requested["hard"], availability["hard"]),
            "easy": min(requested["easy"], availability["easy"]),
            "repair": min(requested["repair"], availability["repair_unique_groups_outside_valid_dynamic"]),
            "replay": requested["replay"],
        }
        arms[name] = {
            "requested": requested,
            "upper_bound_without_replacement": realized,
            "upper_bound_rows": sum(realized.values()),
            "shortfalls": {
                role: requested[role] - realized[role]
                for role in requested if requested[role] > realized[role]
            },
        }

    audit = {
        "status": "FEASIBILITY_AUDITED",
        "dynamic_rows": len(dynamic),
        "dynamic_difficulty_counts": dict(difficulty),
        "dynamic_difficulty_source_counts": {
            f"{key[0]}:{key[1]}": value for key, value in sorted(difficulty_source.items())
        },
        "valid_dynamic_rows": len(valid_dynamic),
        "repair_verified_rows": len(repairs),
        "repair_eligible_unique_theorem_groups": len(repair_groups),
        "repair_difficulty_unique_groups": dict(repair_difficulty),
        "repair_overlap_with_valid_dynamic_groups": len(repair_group_ids & dynamic_theorem_ids),
        "repair_unique_groups_outside_valid_dynamic": len(repair_outside_valid_dynamic),
        "repair_outside_valid_dynamic_breakdown": dict(repair_outside_breakdown),
        "h0_foundation_rows": len(h0),
        "h0_foundation_source_counts": dict(h0_source),
        "h0_pantograph_verified_rows": h0_verified,
        "h0_reference_proof_verified_rows": h0_reference,
        "availability": availability,
        "arms": arms,
        "policy": {
            "category_shortfall": "use every eligible unique item up to requested count; never repeat or backfill from another role",
            "repair_overlap": "C may add only Repair theorem groups absent from its fixed A discovery subset",
            "replay_ratio": "select approximately WB:LD=2:1 inside Replay",
            "reference_target_interpretation": "reference proofs are forbidden for Dynamic/Repair targets; frozen H0 foundation targets are used only in the explicitly required Replay role",
            "arm_domain_warning": "Dynamic Discovery and Repair pools are WB-only; only Replay can add LD under the frozen source definitions",
        },
    }
    write_json(output / "feasibility_audit.json", audit)
    report = [
        "# EI Difficulty-aware Ablation Feasibility Audit",
        "",
        f"- Dynamic: {len(dynamic)} rows; TooEasy/Easy/Medium/Hard/Impossible = "
        f"{difficulty['tooeasy']}/{difficulty['easy']}/{difficulty['medium']}/{difficulty['hard']}/{difficulty['impossible']}.",
        f"- Eligible Repair: {len(repairs)} attempt rows, {len(repair_groups)} unique theorem groups; "
        f"{len(repair_outside_valid_dynamic)} groups are outside the Easy/Medium/Hard-success pool.",
        f"- H0 foundation Replay pool: {len(h0)} rows ({h0_source['WB']} WB, {h0_source['LD']} LD), "
        f"all Pantograph verified: {h0_verified == len(h0)}.",
        "- The requested category counts are not attainable without replacement. The experiment must use actual counts and report shortfalls, as authorized by the task.",
        "- The frozen Dynamic and Repair sources are WB-only. Replay itself can satisfy WB:LD≈2:1, but no arm can be globally domain-balanced under the prescribed source definitions.",
        "",
        "| Arm | Requested | Maximum without replacement | Shortfall |",
        "|---|---:|---:|---|",
    ]
    for name, value in arms.items():
        report.append(
            f"| {name} | {sum(value['requested'].values())} | {value['upper_bound_rows']} | {value['shortfalls']} |"
        )
    (output / "feasibility_audit.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
