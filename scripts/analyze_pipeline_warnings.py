from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path


def main() -> None:
    results_dir = Path(sys.argv[1])
    attempts = [
        json.loads(line)
        for line in (results_dir / "attempts.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    warning_rows = [
        row for row in attempts if "warning:" in row.get("diagnostics", "")
    ]
    warning_only_rows = [
        row
        for row in warning_rows
        if "error:" not in row.get("diagnostics", "")
        and "unsolved goals" not in row.get("diagnostics", "")
    ]
    warning_heads: Counter[str] = Counter()
    for row in warning_rows:
        for line in row.get("diagnostics", "").splitlines():
            if "warning:" in line:
                warning_heads.update([re.sub(r"^\d+:\d+(?:-\d+:\d+)?:\s*", "", line)])
    report = {
        "attempts": len(attempts),
        "warning_attempts": len(warning_rows),
        "warning_only_attempts": len(warning_only_rows),
        "warning_heads": warning_heads.most_common(20),
        "warning_only_examples": [
            {
                "problem_id": row.get("problem_id"),
                "attempt_id": row.get("attempt_id"),
                "status": row.get("status"),
                "success": row.get("success"),
                "generated_proof": row.get("generated_proof"),
                "diagnostics": row.get("diagnostics"),
            }
            for row in warning_only_rows[:10]
        ],
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

