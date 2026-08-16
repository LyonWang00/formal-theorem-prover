"""Preflight the modified Phase C2 quotas without starting training."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    out = project / "outputs/stage2_data_ratio_ablation/phaseC2_balanced_hard"
    classification_path = project / "outputs/task0b_light/classification_manifest.jsonl"
    hard_path = project / "outputs/stage2_data_ablation/phaseA_analysis/hard_reclassification.json"
    c1_audit_path = project / "outputs/stage2_data_ratio_ablation/phaseC1_new_wb/data_audit.json"
    classification = read_jsonl(classification_path)
    hard = json.loads(hard_path.read_text(encoding="utf-8"))
    c1_audit = json.loads(c1_audit_path.read_text(encoding="utf-8"))

    class_counts = collections.Counter(row["classification"] for row in classification)
    hard_a = [row for row in hard["records"] if row["hard_bucket"] == "Hard-A"]
    excluded_hard_a = set(c1_audit["excluded_by_protected_gate"]["Hard-A"])
    protected_clean_hard_a = [row for row in hard_a if row["record_id"] not in excluded_hard_a]
    hard_a_quality = {
        "raw_rows": len(hard_a),
        "protected_clean_rows": len(protected_clean_hard_a),
        "protected_conflicts": sorted(excluded_hard_a),
        "all_have_allowed_strong_near_miss": all(
            any(
                candidate.get("strong_near_miss")
                and candidate.get("failure_taxonomy") in {"unsolved_goals", "tactic_error"}
                for candidate in row["candidates"]
            )
            for row in hard_a
        ),
        "max_reference_proof_tokens": max(row["reference_proof_tokens"] for row in hard_a),
        "rows_with_complexity_flags": sum(any(row["complexity_flags"].values()) for row in hard_a),
        "planner_or_subgoal_text_hits": sum(
            "planner" in json.dumps(row, ensure_ascii=False).lower()
            or "subgoal" in json.dumps(row, ensure_ascii=False).lower()
            for row in hard_a
        ),
    }
    requested = {
        "New WB": 1200,
        "Old replay/Stable-Core": 100,
        "Old replay/Frontier": 200,
        "Old replay/Hard-A": 300,
        "Standalone Hard-A": 200,
    }
    available = {
        "New WB": 1200,
        "Stable/Core": int(class_counts["stable_core"]),
        "Frontier": int(class_counts["frontier"]),
        "Hard-A raw": len(hard_a),
        "Hard-A protected-clean": len(protected_clean_hard_a),
    }
    shortages = {
        "Stable/Core": max(0, 100 - available["Stable/Core"]),
        "Frontier": max(0, 200 - available["Frontier"]),
        "Hard-A": max(0, 500 - available["Hard-A protected-clean"]),
    }
    maximum_legal_non_foundation = available["Stable/Core"] + available["Frontier"] + available["Hard-A protected-clean"]
    audit = {
        "phase": "C2",
        "experiment": "S2-Balanced-Hard modified foundation-dominant recipe",
        "status": "BLOCKED_PRETRAINING_INFEASIBLE",
        "requested_total_rows": 2000,
        "requested": requested,
        "available_frozen_pools": available,
        "shortages": shortages,
        "requested_non_foundation_rows": 800,
        "maximum_legal_unique_non_foundation_rows": maximum_legal_non_foundation,
        "minimum_total_shortfall": 800 - maximum_legal_non_foundation,
        "composition_semantics": {
            "top_level_standalone_hard_a_rows": 200,
            "hard_a_inside_old_replay_rows": 300,
            "actual_total_hard_a_rows_requested": 500,
            "actual_total_hard_a_fraction": 0.25,
            "claimed_hard_a_fraction": 0.10,
            "claimed_10_percent_is_consistent_with_row_composition": False,
        },
        "hard_a_quality": hard_a_quality,
        "constraints_enforced": {
            "hard_b_used": 0,
            "hard_c_used": 0,
            "unclassified_hard_used": 0,
            "duplicate_sampling_used": False,
            "replacement_sampling_used": False,
            "proofs_modified": 0,
            "theorems_modified": 0,
        },
        "execution": {
            "manifest_created": False,
            "trainer_started": False,
            "gpu_training_started": False,
            "model_modified": False,
            "evaluation_started": False,
            "phase_c3_started": False,
        },
        "source_hashes": {
            str(classification_path): sha256_file(classification_path),
            str(hard_path): sha256_file(hard_path),
            str(c1_audit_path): sha256_file(c1_audit_path),
        },
        "blocking_reason": (
            "The frozen protected-clean pools contain only 12 Stable/Core, 47 Frontier, "
            "and 397 Hard-A rows for 800 requested non-foundation slots. The modified "
            "recipe requires 100, 200, and 500 respectively. Filling the 344-row gap "
            "would require a prohibited pool, duplicate sampling, new classification, "
            "or a user-approved composition change. In addition, the stated 10% "
            "Hard-A interpretation conflicts with the row composition, which contains "
            "500 Hard-A rows (25%) after counting the 300 Hard-A rows inside Old replay."
        ),
    }
    write_json(out / "phaseC2_feasibility_audit.json", audit)
    write_json(out / "status.json", {
        "status": audit["status"],
        "trainer_started": False,
        "gpu_training_started": False,
        "phase_c3_started": False,
        "minimum_total_shortfall": audit["minimum_total_shortfall"],
        "waiting_for_user_direction": True,
    })
    report = [
        "# Phase C2 modified recipe — preflight report", "",
        "## Outcome", "",
        "**BLOCKED BEFORE TRAINING: the requested unique frozen pools cannot satisfy the modified 2000-row composition.**", "",
        "No manifest was frozen, no Trainer or GPU training was started, M0 was not modified, and Phase C3 was not started.", "",
        "## Requested versus available", "",
        "| Pool | Requested | Protected-clean available | Shortfall |", "|---|---:|---:|---:|",
        f"| Stable/Core | 100 | {available['Stable/Core']} | {shortages['Stable/Core']} |",
        f"| Frontier | 200 | {available['Frontier']} | {shortages['Frontier']} |",
        f"| Hard-A (Old replay 300 + standalone 200) | 500 | {available['Hard-A protected-clean']} | {shortages['Hard-A']} |",
        f"| **Total non-New-WB** | **800** | **{maximum_legal_non_foundation}** | **{audit['minimum_total_shortfall']}** |", "",
        "The raw Hard-A pool has 399 rows. Two rows conflict with protected evaluation by qualified theorem name and remain excluded, leaving 397. All 399 raw Hard-A records have an allowed strong near-miss (`unsolved_goals` or `tactic_error`), have no Task0B complexity flag, contain no Planner/Subgoal marker, and have at most 101 reference-proof tokens. Therefore the shortage is a pool-size issue, not an avoidable quality-filter loss.", "",
        "## Composition interpretation conflict", "",
        "The top-level table labels 200 standalone Hard-A rows as 10%, but Old replay separately contains 300 Hard-A rows. The actual requested training content therefore contains 500 Hard-A rows, or 25% of all 2000 rows. This recipe cannot validly answer whether *total* Hard-A at 10% is stable unless the Old replay composition is revised or the intended percentage is explicitly defined as the standalone role only.", "",
        "## Enforced prohibitions", "",
        "- Hard-B used: 0", "- Hard-C used: 0", "- Unclassified hard used: 0",
        "- Duplicate/replacement sampling: none", "- Proof/theorem modifications: 0",
        "- Trainer/GPU training: not started", "- Phase C3: not started", "",
        "## Decisions that can make Phase C2 feasible", "",
        "A new explicit data decision is required. Examples include expanding and freezing additional M0 classifications, increasing New WB, relaxing Stable/Frontier quotas, or allowing another audited replay bucket. None was selected automatically because each changes the requested experiment.", "",
        "Phase C2 not started: preflight blocked.", "",
        "Waiting for user direction; Phase C3 remains prohibited.", "",
    ]
    (out / "phaseC2_preflight_report.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
