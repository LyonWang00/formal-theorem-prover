"""Run and select the fixed Early Retention Gate for every B/C checkpoint.

The runner is intentionally resumable: a checkpoint is skipped only when its
metrics and both 200-row candidate files are present.  It never changes the
fixed dataset, generation seed, or checkpoint-selection ordering.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ARM_CHECKPOINTS = {
    "B1": ("B1_anchor_1000", (0, 5, 10, 15, 20, 25, 40, 63)),
    "B2": ("B2_anchor_expert_1000", (0, 5, 10, 15, 20, 25, 40, 63)),
    "C1": ("C1_anchor_micro", (0, 1, 3, 5, 8, 10)),
    "C2": ("C2_expert_micro", (0, 1, 3, 5, 8, 10)),
    "C3": ("C3_expert_anchor_micro", (0, 1, 3, 5, 8, 10, 12)),
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.open(encoding="utf-8"))


def completed(output: Path) -> bool:
    return (
        (output / "discovery_replay_metrics.json").is_file()
        and jsonl_count(output / "generations.jsonl") == 200
        and jsonl_count(output / "verifications.jsonl") == 200
    )


def metric_summary(output: Path) -> dict[str, Any]:
    metric = read_json(output / "discovery_replay_metrics.json")
    return {
        "pass_at_4": metric["discovery_replay_pass_at_4"],
        "pass_at_1": metric["discovery_replay_pass_at_1"],
        "candidate_success_rate": metric["discovery_replay_compile_success_rate"],
        "solved_statements": metric["successes"],
        "generation_seconds": metric["generation_seconds"],
        "verification_seconds": metric["verification_seconds"],
        "total_seconds": metric["total_seconds"],
        "pantograph_worker_restart_count": metric[
            "pantograph_worker_restart_count"
        ],
    }


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("outputs/expert_sft_no_replacement_ablation"),
    )
    parser.add_argument(
        "--config", type=Path, default=Path("configs/expert_iteration.round1.yaml")
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(
            "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor"
        ),
    )
    parser.add_argument("--seed", type=int, default=42001)
    parser.add_argument("--samples-per-statement", type=int, default=4)
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    dataset = args.root / "manifests/early_retention_gate_50.jsonl"
    output_root = args.root / "evaluations/early_gate"
    progress_path = output_root / "progress.json"
    results: dict[str, Any] = {}
    total = sum(len(steps) for _, steps in ARM_CHECKPOINTS.values())
    current = 0

    for arm, (checkpoint_dir, steps) in ARM_CHECKPOINTS.items():
        results[arm] = {}
        for step in steps:
            current += 1
            name = f"step_{step}"
            adapter = args.root / "checkpoints" / checkpoint_dir / name
            output = output_root / arm / name
            if not adapter.is_dir():
                raise FileNotFoundError(f"missing checkpoint: {adapter}")
            was_complete = completed(output)
            print(
                f"EARLY_GATE {current}/{total} {arm} {name} "
                f"status={'reuse' if was_complete else 'run'}",
                flush=True,
            )
            if not was_complete:
                command = [
                    sys.executable,
                    "scripts/run_ablation_evaluation.py",
                    "--config",
                    str(args.config),
                    "--role",
                    "discovery_replay",
                    "--model",
                    str(args.model),
                    "--adapter",
                    str(adapter),
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(output),
                    "--seed",
                    str(args.seed),
                    "--samples-per-statement",
                    str(args.samples_per_statement),
                ]
                for attempt in range(1, args.max_attempts + 1):
                    result = subprocess.run(command, check=False)
                    if result.returncode == 0 and completed(output):
                        break
                    print(
                        f"EARLY_GATE_RETRY {arm} {name} attempt={attempt} "
                        f"return_code={result.returncode}",
                        flush=True,
                    )
                    if attempt == args.max_attempts:
                        raise subprocess.CalledProcessError(
                            result.returncode, command
                        )
                    time.sleep(10)
                if not completed(output):
                    raise RuntimeError(f"incomplete Early Gate output: {output}")
            results[arm][name] = metric_summary(output)
            write_json(
                progress_path,
                {
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                    "completed": current,
                    "total": total,
                    "last": {"arm": arm, "step": step},
                    "results": results,
                },
            )

    selection: dict[str, Any] = {
        "selection_order": [
            "early_gate_pass_at_4",
            "early_gate_pass_at_1",
            "early_gate_candidate_success_rate",
            "earlier_step",
        ],
        "generation_seed": args.seed,
        "dataset": str(dataset.resolve()),
        "arms": {},
    }
    for arm, (_, steps) in ARM_CHECKPOINTS.items():
        best = max(
            steps,
            key=lambda step: (
                results[arm][f"step_{step}"]["pass_at_4"],
                results[arm][f"step_{step}"]["pass_at_1"],
                results[arm][f"step_{step}"]["candidate_success_rate"],
                -step,
            ),
        )
        selection["arms"][arm] = {
            "best_step": best,
            "checkpoint": str(
                (
                    args.root
                    / "checkpoints"
                    / ARM_CHECKPOINTS[arm][0]
                    / f"step_{best}"
                ).resolve()
            ),
            "best_metrics": results[arm][f"step_{best}"],
            "candidates": results[arm],
        }
    write_json(args.root / "checkpoint_selection.json", selection)
    print(json.dumps(selection, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
