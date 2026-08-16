#!/usr/bin/env python3
"""Run only the full length evaluations authorized by expansion_plan.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def transient_failure(path: Path) -> bool:
    text = path.read_text(encoding="utf-8", errors="replace")
    return any(
        signature in text
        for signature in (
            "Engine core initialization failed",
            "CUDA out of memory",
            "not enough GPU memory",
        )
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/ld_length_difficulty_pipeline/length_ablation"),
    )
    parser.add_argument("--cooldown-seconds", type=int, default=60)
    parser.add_argument("--retry-cooldown-seconds", type=int, default=120)
    parser.add_argument("--transient-retries", type=int, default=2)
    args = parser.parse_args()
    project = args.root.resolve()
    output = (project / args.output).resolve()
    plan = read_json(output / "manifests/expansion_plan.json")
    if not plan.get("authorized"):
        print("No full expansion is authorized.", flush=True)
        return
    progress_path = output / "runtime/expansion_progress.json"
    progress: dict[str, Any] = {
        "started_at": utc_now(),
        "status": "running",
        "completed": [],
        "skipped_existing": [],
        "current": None,
    }
    write_json(progress_path, progress)
    try:
        for run in plan["runs"]:
            unit = f"{run['dataset']}:{run['model']}:L{run['length']}"
            unit_output = Path(run["output"])
            if (unit_output / "run_summary.json").is_file():
                progress["skipped_existing"].append(unit)
                write_json(progress_path, progress)
                print(f"SKIP {unit}", flush=True)
                continue
            progress["current"] = unit
            write_json(progress_path, progress)
            print(f"START {unit} {utc_now()}", flush=True)
            time.sleep(args.cooldown_seconds)
            command = [
                sys.executable,
                str(project / "scripts/run_ld_length_canary.py"),
                "--config",
                str(output / f"manifests/config_L{run['length']}.json"),
                "--model-name",
                run["model"],
                "--model",
                run["base_model"],
                "--dataset",
                run["manifest"],
                "--output",
                run["output"],
                "--seed",
                str(run["seed"]),
                "--max-new-tokens",
                str(run["length"]),
                "--expected-wb",
                str(run["expected_wb"]),
                "--expected-ld",
                str(run["expected_ld"]),
            ]
            if run.get("adapter"):
                command.extend(["--adapter", run["adapter"]])
            log_dir = output / "runtime/expansion_logs"
            log_dir.mkdir(parents=True, exist_ok=True)
            completed = None
            log_path = log_dir / (
                f"{run['dataset']}_{run['model']}_L{run['length']}.log"
            )
            for attempt in range(args.transient_retries + 1):
                if attempt:
                    time.sleep(args.retry_cooldown_seconds)
                attempt_log = (
                    log_path
                    if attempt == 0 and not log_path.exists()
                    else log_path.with_name(
                        f"{log_path.stem}_attempt_{attempt + 1}.log"
                    )
                )
                with attempt_log.open("w", encoding="utf-8") as handle:
                    completed = subprocess.run(
                        command,
                        cwd=project,
                        stdout=handle,
                        stderr=subprocess.STDOUT,
                        text=True,
                        check=False,
                    )
                log_path = attempt_log
                if not completed.returncode or not transient_failure(log_path):
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
                write_json(progress_path, progress)
                raise SystemExit(completed.returncode)
            progress["completed"].append(unit)
            progress["current"] = None
            write_json(progress_path, progress)
            print(f"DONE {unit} {utc_now()}", flush=True)
        progress["status"] = "completed"
        progress["finished_at"] = utc_now()
        write_json(progress_path, progress)
    except BaseException:
        if progress["status"] == "running":
            progress["status"] = "interrupted"
            progress["interrupted_at"] = utc_now()
            write_json(progress_path, progress)
        raise


if __name__ == "__main__":
    main()
