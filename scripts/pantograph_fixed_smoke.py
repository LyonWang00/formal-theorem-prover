"""Verify the fixed Lean training smoke cases through Pantograph."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from lean_prover.lean_training.verification.pantograph import (
    PantographTheoremVerifier,
)


CASES = {
    "norm_num": "example : (1 : ℕ) + 1 = 2 := by\n  norm_num",
    "simp": "example (x : ℕ) : x = x := by\n  simp",
    "linarith": (
        "example (x y : ℚ) (h : x ≤ y) : x - y ≤ 0 := by\n"
        "  linarith"
    ),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", default="lean_project")
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--startup-timeout", type=int, default=600)
    args = parser.parse_args()

    verifier = PantographTheoremVerifier(
        Path(args.project),
        imports=("Mathlib",),
        timeout=args.timeout,
        startup_timeout=args.startup_timeout,
    )
    rows: list[dict[str, object]] = []
    try:
        for name, source in CASES.items():
            result = verifier.check_source(
                source,
                timeout=args.timeout,
                reject_forbidden=False,
            )
            rows.append({"name": name, "source": source, **result.to_json()})
    finally:
        verifier.close()

    report = {
        "project": str(Path(args.project).resolve()),
        "imports": ["Mathlib"],
        "passed": sum(bool(row["success"]) for row in rows),
        "total": len(rows),
        "cases": rows,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["passed"] == report["total"] else 1)


if __name__ == "__main__":
    main()
