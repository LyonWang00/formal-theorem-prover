"""Run the post-selection Discovery Gate, Monitor, and gated full replay."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def jsonl_count(path: Path) -> int:
    if not path.is_file():
        return 0
    return sum(bool(line.strip()) for line in path.open(encoding="utf-8"))


def complete(output: Path, candidates: int) -> bool:
    metrics = list(output.glob("*_metrics.json"))
    return (
        bool(metrics)
        and jsonl_count(output / "generations.jsonl") == candidates
        and jsonl_count(output / "verifications.jsonl") == candidates
    )


def run_evaluation(
    *,
    config: Path,
    role: str,
    model: Path,
    adapter: Path | None,
    dataset: Path,
    output: Path,
    seed: int,
    statements: int,
    max_attempts: int,
) -> dict[str, Any]:
    candidates = statements * 4
    if not complete(output, candidates):
        command = [
            sys.executable,
            "scripts/run_ablation_evaluation.py",
            "--config",
            str(config),
            "--role",
            role,
            "--model",
            str(model),
            "--dataset",
            str(dataset),
            "--output",
            str(output),
            "--seed",
            str(seed),
            "--samples-per-statement",
            "4",
        ]
        if adapter is not None:
            command.extend(("--adapter", str(adapter)))
        for attempt in range(1, max_attempts + 1):
            result = subprocess.run(command, check=False)
            if result.returncode == 0 and complete(output, candidates):
                break
            print(
                f"RELEASE_GATE_RETRY role={role} output={output.name} "
                f"attempt={attempt} return_code={result.returncode}",
                flush=True,
            )
            if attempt == max_attempts:
                raise subprocess.CalledProcessError(result.returncode, command)
            time.sleep(10)
    metric_path = output / f"{role}_metrics.json"
    return read_json(metric_path)


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
        "--round1", type=Path, default=Path("outputs/expert_iteration_round1")
    )
    parser.add_argument("--max-attempts", type=int, default=3)
    args = parser.parse_args()

    selection = read_json(args.root / "checkpoint_selection.json")
    manifest = read_json(args.root / "manifests/experiment_manifest.json")
    gate_metadata = read_json(
        args.root / "manifests/early_retention_gate_metadata.json"
    )
    m0 = Path(manifest["m0"]["absolute_path"])
    discovery_gate = Path(gate_metadata["source_path"])
    discovery_seed = int(manifest["m0"]["generation_config"]["generation_seed"])
    models: dict[str, tuple[Path, Path | None]] = {
        "M0": (m0, None),
        "C0": (args.root / "checkpoints/C0_zero_step/merged", None),
        **{
            arm: (m0, Path(row["checkpoint"]))
            for arm, row in selection["arms"].items()
        },
    }

    discovery_metrics: dict[str, Any] = {}
    for index, (name, (base, adapter)) in enumerate(models.items(), start=1):
        print(f"DISCOVERY_GATE {index}/{len(models)} {name}", flush=True)
        discovery_metrics[name] = run_evaluation(
            config=args.config,
            role="discovery_replay",
            model=base,
            adapter=adapter,
            dataset=discovery_gate,
            output=args.root / "evaluations/discovery_gate" / name,
            seed=discovery_seed,
            statements=150,
            max_attempts=args.max_attempts,
        )
        write_json(
            args.root / "evaluations/discovery_gate/progress.json",
            discovery_metrics,
        )

    trained = [name for name in models if name not in {"M0", "C0"}]
    top_two = sorted(
        trained,
        key=lambda name: (
            discovery_metrics[name]["discovery_replay_pass_at_4"],
            discovery_metrics[name]["discovery_replay_pass_at_1"],
            discovery_metrics[name]["discovery_replay_compile_success_rate"],
        ),
        reverse=True,
    )[:2]
    monitor_names = ["M0", "C0", *top_two]
    monitor_dataset = args.round1 / "inputs/verified/monitor.jsonl"
    monitor_metrics: dict[str, Any] = {}
    for index, name in enumerate(monitor_names, start=1):
        base, adapter = models[name]
        print(f"MONITOR {index}/{len(monitor_names)} {name}", flush=True)
        monitor_metrics[name] = run_evaluation(
            config=args.config,
            role="monitor",
            model=base,
            adapter=adapter,
            dataset=monitor_dataset,
            output=args.root / "evaluations/monitor" / name,
            seed=3001,
            statements=64,
            max_attempts=args.max_attempts,
        )

    m0_monitor = monitor_metrics["M0"]
    monitor_floor = float(m0_monitor["monitor_pass_at_4"]) - 1 / 64
    qualified = [
        name
        for name in top_two
        if int(discovery_metrics[name]["successes"]) >= 98
        and float(monitor_metrics[name]["monitor_pass_at_4"]) >= monitor_floor
    ]
    full_replay: dict[str, Any] = {}
    for index, name in enumerate(qualified, start=1):
        base, adapter = models[name]
        print(f"FULL_REPLAY {index}/{len(qualified)} {name}", flush=True)
        full_replay[name] = run_evaluation(
            config=args.config,
            role="discovery_replay",
            model=base,
            adapter=adapter,
            dataset=Path("data/processed/lean_workbook_verified_v2/discovery.jsonl"),
            output=args.root / "evaluations/full_replay" / name,
            seed=discovery_seed,
            statements=500,
            max_attempts=args.max_attempts,
        )

    release = {
        "m0_baseline": {
            "discovery_gate": str(
                (args.root / "evaluations/discovery_gate/M0").resolve()
            ),
            "monitor": str((args.root / "evaluations/monitor/M0").resolve()),
            "full_replay_reuse_if_needed": str(
                (args.round1 / "iteration_000/discovery").resolve()
            ),
        },
        "discovery_gate": discovery_metrics,
        "monitor_evaluated": monitor_names,
        "monitor": monitor_metrics,
        "full_replay_threshold": {
            "discovery_solved_at_least": 98,
            "monitor_pass_at_4_at_least": monitor_floor,
        },
        "full_replay_qualified": qualified,
        "full_replay": full_replay,
        "benchmark_run": False,
        "benchmark_reason": (
            "Benchmark is not run by this stage; it remains gated on Discovery, "
            "Monitor, and Full Replay all matching or exceeding M0."
        ),
    }
    write_json(args.root / "release_gate_results.json", release)
    print(json.dumps(release, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
