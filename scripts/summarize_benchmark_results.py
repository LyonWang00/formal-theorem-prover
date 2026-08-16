from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    root = Path(sys.argv[1])
    results = root / "results"
    summary = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    attempts = [
        json.loads(line)
        for line in (results / "attempts.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    problems = [
        json.loads(line)
        for line in (results / "problem_results.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    benchmark_log = (root / "benchmark.log").read_text(encoding="utf-8", errors="replace").splitlines()
    outer_log = Path(str(root) + ".outer.log")
    outer_lines = (
        outer_log.read_text(encoding="utf-8", errors="replace").splitlines()
        if outer_log.exists()
        else []
    )
    report = {
        "summary": summary,
        "attempt_rows": len(attempts),
        "problem_rows": len(problems),
        "problem_result_count_benchmark_log": sum(
            1 for line in benchmark_log if line.startswith("PROBLEM_RESULT ")
        ),
        "problem_result_count_outer_log": sum(
            1 for line in outer_lines if line.startswith("PROBLEM_RESULT ")
        ),
        "status_counts": dict(Counter(row.get("status") for row in attempts)),
        "success_attempts": sum(1 for row in attempts if row.get("success")),
        "timed_out_attempts": sum(1 for row in attempts if row.get("timed_out")),
        "rejected_attempts": sum(
            1 for row in attempts if row.get("status") == "rejected"
        ),
        "import_mismatch_attempts": sum(
            1 for row in attempts if "imports differ" in row.get("diagnostics", "")
        ),
        "successful_problem_ids": [
            problem["problem_id"] for problem in problems if problem.get("success")
        ],
        "attempts_per_problem": [
            len(problem.get("attempts", [])) for problem in problems
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()


