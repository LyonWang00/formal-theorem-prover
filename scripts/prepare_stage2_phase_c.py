"""Perform the immutable Phase-C data quota preflight without training."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", type=Path, required=True)
    args = parser.parse_args()
    project = args.project.resolve()
    out = project / "outputs/stage2_data_ablation/phaseC_new_easy"
    out.mkdir(parents=True, exist_ok=True)

    classification_path = project / "outputs/task0b_light/classification_manifest.jsonl"
    hard_path = project / "outputs/stage2_data_ablation/phaseA_analysis/hard_reclassification.json"
    pool_path = project / "outputs/stage2_sft_incremental_ablation/phase0/data_pool_summary.json"
    classification = read_jsonl(classification_path)
    bucket_counts = Counter(str(row.get("final_bucket")) for row in classification)
    hard = json.loads(hard_path.read_text(encoding="utf-8"))
    pool = json.loads(pool_path.read_text(encoding="utf-8"))

    available = {
        "Stable/Core": bucket_counts["stable_core"],
        "Frontier": bucket_counts["frontier"],
        "Hard-A": int(hard["counts"]["Hard-A"]),
        "New WB": int(pool["available_pools"]["new_wb"]["verified_unused_protected_clean"]),
        "New LD-easy": int(pool["available_pools"]["new_ld_easy"]["verified_unused_protected_clean"]),
    }
    required = {"Stable/Core": 100, "Frontier": 200, "Hard-A": 300, "New WB": 900, "New LD-easy": 500}
    shortages = {key: max(0, required[key] - available[key]) for key in required}
    selectable = {key: min(required[key], available[key]) for key in required}
    feasible = not any(shortages.values())
    result = {
        "phase": "C",
        "experiment": "S2-Balanced-NewEasy",
        "status": "READY" if feasible else "BLOCKED_INSUFFICIENT_FROZEN_DATA",
        "target_rows": 2000,
        "required": required,
        "available": available,
        "shortages": shortages,
        "maximum_rows_under_exact_recipe_without_replacement": sum(selectable.values()),
        "total_shortfall": 2000 - sum(selectable.values()),
        "constraints_preserved": {
            "replacement_sampling": False,
            "duplicate_rows": 0,
            "recipe_modified": False,
            "evaluation_data_used": False,
            "trainer_started": False,
            "gpu_training_started": False,
        },
        "source_identity": {
            "task0b_classification_manifest": {"path": str(classification_path), "sha256": sha256(classification_path), "rows": len(classification)},
            "phaseA_hard_reclassification": {"path": str(hard_path), "sha256": sha256(hard_path), "rows": sum(hard["counts"].values())},
            "phase0_pool_summary": {"path": str(pool_path), "sha256": sha256(pool_path)},
        },
        "blocking_reason": None if feasible else "The frozen recipe cannot be assembled uniquely: Stable/Core, Frontier, and especially unused New LD-easy are below quota. Filling it would require replacement, reuse, category substitution, or a recipe change, all outside the authorized experiment.",
        "next_authority_needed": None if feasible else "A revised Phase-C data composition or authorization to build/audit an additional New LD-easy pool.",
    }
    (out / "phaseC_preflight.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = [
        "# Phase C — Balanced Replay + New Easy preflight", "",
        f"**Status: {result['status']}**", "",
        "The requested 2,000-row manifest cannot be constructed from the frozen pools without violating the no-replacement/no-substitution constraints.", "",
        "| Bucket | Required | Available | Shortage |", "|---|---:|---:|---:|",
    ]
    for key in required:
        report.append(f"| {key} | {required[key]} | {available[key]} | {shortages[key]} |")
    report += ["", f"Maximum exact-recipe contribution currently selectable: **{sum(selectable.values())}/2000**; total shortfall **{result['total_shortfall']}**.", "", "The largest blocker is New LD-easy: only 4 verified unused protected-clean rows remain after round 1, versus 500 required. Stable/Core and Frontier are also short by 88 and 153 rows respectively. Hard-A and New WB have sufficient capacity.", "", "No manifest was fabricated, no replacement sampling was used, and no Trainer or GPU training was started.", "", "To continue Phase C, the experiment needs either a revised frozen composition or a newly sourced and fully audited unused LD-easy pool. The current constraints do not authorize either change.", ""]
    (out / "phaseC_preflight_report.md").write_text("\n".join(report), encoding="utf-8")

    summary = project / "outputs/stage2_data_ablation/final_summary.md"
    summary.write_text(
        "# Stage-2 data ablation status\n\n"
        "- Phase A: completed (analysis only).\n"
        "- Phase B: completed; hard-heavy replay gives a small pass@2 gain but lowers aggregate pass@1.\n"
        "- Phase C: started and stopped at immutable data preflight; exact 2,000-row recipe is short by 737 unique rows.\n"
        "- Phase D: not started.\n\n"
        "See `phaseB_hard_replay/phaseB_report.md` and `phaseC_new_easy/phaseC_preflight_report.md`.\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
