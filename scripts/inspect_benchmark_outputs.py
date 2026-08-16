from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines()
        if line.strip()
    ]


def main() -> None:
    root = Path(sys.argv[1])
    results = root / "results"
    attempts = read_jsonl(results / "attempts.jsonl")
    problems = read_jsonl(results / "problem_results.jsonl")
    successes = read_jsonl(results / "success_attempts.jsonl")
    summary_path = results / "summary.json"

    report = {
        "root": str(root),
        "summary_exists": summary_path.exists(),
        "attempt_rows": len(attempts),
        "problem_rows": len(problems),
        "success_attempt_rows": len(successes),
        "problem_successes": sum(1 for row in problems if row.get("success")),
        "server_not_running_attempts": sum(
            1 for row in attempts if "Server not running" in str(row.get("diagnostics"))
        ),
        "status_counts": dict(Counter(row.get("status") for row in attempts)),
        "last_problem_id": problems[-1].get("problem_id") if problems else None,
        "successful_problem_ids": [
            row.get("problem_id") for row in problems if row.get("success")
        ],
    }
    if summary_path.exists():
        report["summary"] = json.loads(summary_path.read_text(encoding="utf-8"))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
