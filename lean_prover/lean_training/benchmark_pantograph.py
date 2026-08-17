"""Benchmark Pantograph startup, import warmup, and theorem checking throughput."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

from lean_prover.lean_training.verification.pantograph import PantographTheoremVerifier


SIMPLE_THEOREM = "example : True := by\n  trivial"


def parse_args() -> argparse.Namespace:
    default_project = Path(__file__).resolve().parents[2] / "lean_project"
    parser = argparse.ArgumentParser(description="Benchmark Pantograph verifier.")
    parser.add_argument(
        "--lean_project_path",
        default=os.environ.get("LEAN_PROJECT_PATH", str(default_project)),
    )
    parser.add_argument("--imports", default="Mathlib")
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--warmup_timeout", type=int, default=1200)
    parser.add_argument("--output_json", default=None)
    args = parser.parse_args()
    args.imports = tuple(item.strip() for item in args.imports.split(",") if item.strip())
    return args


def throughput(
    verifier: PantographTheoremVerifier,
    *,
    count: int,
    timeout: int,
) -> dict[str, Any]:
    times: list[float] = []
    failures: list[dict[str, Any]] = []
    start = time.monotonic()
    for index in range(count):
        result = verifier.check_source(
            f"-- benchmark {count}/{index}\nexample : True := by\n  trivial",
            timeout=timeout,
        )
        times.append(result.check_seconds)
        if not result.success:
            failures.append(
                {
                    "index": index,
                    "diagnostics": result.diagnostics,
                    "timed_out": result.timed_out,
                }
            )
    total = round(time.monotonic() - start, 4)
    return {
        "count": count,
        "successes": count - len(failures),
        "failures": failures,
        "total_seconds": total,
        "avg_seconds": round(total / count, 4) if count else 0.0,
        "median_attempt_seconds": round(statistics.median(times), 4) if times else 0.0,
        "min_attempt_seconds": round(min(times), 4) if times else 0.0,
        "max_attempt_seconds": round(max(times), 4) if times else 0.0,
    }


def main() -> None:
    args = parse_args()
    verifier = PantographTheoremVerifier(
        args.lean_project_path,
        imports=args.imports,
        timeout=args.timeout,
    )
    try:
        warmup = verifier.warmup(timeout=args.warmup_timeout)
        single = verifier.check_source(SIMPLE_THEOREM, timeout=args.timeout)
        result = {
            "lean_project_path": str(Path(args.lean_project_path).resolve()),
            "imports": list(args.imports),
            "server_startup_seconds": verifier.server_startup_seconds,
            "post_startup_warmup": warmup.to_json(),
            "single_theorem": single.to_json(),
            "throughput_10": throughput(verifier, count=10, timeout=args.timeout),
            "throughput_100": throughput(verifier, count=100, timeout=args.timeout),
        }
    finally:
        verifier.close()

    text = json.dumps(result, ensure_ascii=False, indent=2)
    print(text)
    if args.output_json:
        path = Path(args.output_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
