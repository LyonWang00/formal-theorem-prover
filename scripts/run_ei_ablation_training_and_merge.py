#!/usr/bin/env python3
"""Run the three EI ablation arms sequentially and merge each adapter."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ARMS = {
    "EI-A_success_only": "EI-A-MERGED",
    "EI-B_success_frontier": "EI-B-MERGED",
    "EI-C_success_frontier_replay": "EI-C-MERGED",
}


def run(command: list[str], cwd: Path) -> None:
    subprocess.run(command, cwd=cwd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True, type=Path)
    args = parser.parse_args()
    project = args.project.resolve()
    root = project / "outputs/expert_iteration/ei_ablation"
    base = project / "outputs/stage2_data_ratio_ablation/hard_a_ablation/H0_no_hard/checkpoints/H0-No-Hard-MERGED"
    completed: dict[str, str] = {}
    for arm, merged_name in ARMS.items():
        arm_root = root / arm
        audit = json.loads((arm_root / "data_audit.json").read_text(encoding="utf-8-sig"))
        if not audit.get("passed"):
            raise RuntimeError(f"data audit failed for {arm}")
        summary = arm_root / "checkpoint/training_summary.json"
        adapter = arm_root / "checkpoint/adapter"
        if not summary.is_file():
            run([
                sys.executable, "scripts/train_ei_data_composition_arm.py",
                "--project", str(project), "--arm", arm,
            ], project)
        if not summary.is_file() or not (adapter / "adapter_model.safetensors").is_file():
            raise RuntimeError(f"training output incomplete for {arm}")
        merged = arm_root / "checkpoint" / merged_name
        if not (merged / "model.safetensors").is_file():
            if merged.exists():
                raise FileExistsError(f"partial merged checkpoint requires manual audit: {merged}")
            run([
                sys.executable, "scripts/merge_lora_adapter.py",
                "--base-model", str(base), "--adapter", str(adapter), "--output", str(merged),
            ], project)
        completed[arm] = str(merged)
    print(json.dumps({"status": "completed", "arms": completed, "parallel_training": False}, indent=2))


if __name__ == "__main__":
    main()
