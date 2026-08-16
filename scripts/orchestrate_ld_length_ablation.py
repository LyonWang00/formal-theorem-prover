#!/usr/bin/env python3
"""Run every remaining frozen model/length canary unit sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


MODELS = {
    "M0-ZERO": None,
    "MIX-A-WB100": "outputs/wb_ld_small_sft_ablation/checkpoints/MIX-A-WB100/final",
    "MIX-B-LD25": "outputs/wb_ld_small_sft_ablation/checkpoints/MIX-B-LD25/final",
    "MIX-C-LD50": "outputs/wb_ld_small_sft_ablation/checkpoints/MIX-C-LD50/final",
    "MIX-E-LD100": "outputs/wb_ld_small_sft_ablation/checkpoints/MIX-E-LD100/final",
}
LENGTHS = (256, 512, 1024)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_progress(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def is_transient_generation_start_failure(log_path: Path) -> bool:
    """Return whether a failed unit is safe to retry without changing inputs."""

    text = log_path.read_text(encoding="utf-8", errors="replace")
    signatures = (
        "Engine core initialization failed",
        "CUDA out of memory",
        "not enough GPU memory",
    )
    return any(signature in text for signature in signatures)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/length_ablation"),
    )
    parser.add_argument("--cooldown-seconds", type=int, default=60)
    parser.add_argument("--transient-retries", type=int, default=2)
    parser.add_argument("--retry-cooldown-seconds", type=int, default=120)
    args = parser.parse_args()
    root = args.root.resolve()
    output = (root / args.output).resolve()
    base_model = (
        root
        / "outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor"
    )
    dataset = output / "manifests/combined_length_canary_114.jsonl"
    progress_path = output / "runtime/orchestrator_progress.json"
    progress = {
        "started_at": utc_now(),
        "status": "running",
        "completed": [],
        "skipped_existing": [],
        "current": None,
    }
    write_progress(progress_path, progress)
    try:
        for model_name, adapter_relative in MODELS.items():
            for length in LENGTHS:
                unit = f"{model_name}/L{length}"
                unit_output = output / "generations" / model_name / f"L{length}"
                summary_path = unit_output / "run_summary.json"
                if summary_path.is_file():
                    progress["skipped_existing"].append(unit)
                    write_progress(progress_path, progress)
                    print(f"SKIP {unit}: complete", flush=True)
                    continue
                progress["current"] = unit
                write_progress(progress_path, progress)
                print(f"START {unit} {utc_now()}", flush=True)
                if args.cooldown_seconds:
                    print(
                        f"COOLDOWN {unit} {args.cooldown_seconds}s before GPU load",
                        flush=True,
                    )
                    time.sleep(args.cooldown_seconds)
                command = [
                    sys.executable,
                    str(root / "scripts/run_ld_length_canary.py"),
                    "--config",
                    str(output / f"manifests/config_L{length}.json"),
                    "--model-name",
                    model_name,
                    "--model",
                    str(base_model),
                    "--dataset",
                    str(dataset),
                    "--output",
                    str(unit_output),
                    "--seed",
                    "20260901",
                    "--max-new-tokens",
                    str(length),
                ]
                if adapter_relative:
                    command.extend(["--adapter", str(root / adapter_relative)])
                log_dir = output / "runtime/logs"
                log_dir.mkdir(parents=True, exist_ok=True)
                completed = None
                log_path = log_dir / f"{model_name}_L{length}.log"
                for attempt in range(args.transient_retries + 1):
                    if attempt:
                        print(
                            f"RETRY {unit} attempt={attempt + 1} after "
                            f"{args.retry_cooldown_seconds}s",
                            flush=True,
                        )
                        time.sleep(args.retry_cooldown_seconds)
                    attempt_log = (
                        log_path
                        if attempt == 0 and not log_path.exists()
                        else log_dir
                        / f"{model_name}_L{length}_attempt_{attempt + 1}.log"
                    )
                    with attempt_log.open("w", encoding="utf-8") as log:
                        completed = subprocess.run(
                            command,
                            cwd=root,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            text=True,
                            check=False,
                        )
                    log_path = attempt_log
                    if not completed.returncode:
                        break
                    if not is_transient_generation_start_failure(log_path):
                        break
                assert completed is not None
                if completed.returncode:
                    progress["status"] = "failed"
                    progress["failed"] = {
                        "unit": unit,
                        "return_code": completed.returncode,
                        "log": str(log_path),
                        "attempts": attempt + 1,
                    }
                    write_progress(progress_path, progress)
                    raise SystemExit(completed.returncode)
                progress["completed"].append(unit)
                progress["current"] = None
                write_progress(progress_path, progress)
                print(f"DONE {unit} {utc_now()}", flush=True)
        progress["status"] = "completed"
        progress["current"] = None
        progress["finished_at"] = utc_now()
        write_progress(progress_path, progress)
    except BaseException:
        if progress["status"] == "running":
            progress["status"] = "interrupted"
            progress["interrupted_at"] = utc_now()
            write_progress(progress_path, progress)
        raise


if __name__ == "__main__":
    main()
