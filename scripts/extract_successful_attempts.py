from __future__ import annotations

import json
import sys
from pathlib import Path


def _messages_from_legacy_diagnostics(attempt: dict) -> tuple[list[str], list[str], list[str]]:
    messages = list(attempt.get("compile_messages") or [])
    errors = list(attempt.get("compile_errors") or [])
    warnings = list(attempt.get("compile_warnings") or [])
    if messages or errors or warnings:
        return messages, errors, warnings

    diagnostics = str(attempt.get("diagnostics") or "")
    if not diagnostics:
        return [], [], []
    messages = [diagnostics]
    errors = [diagnostics] if "error:" in diagnostics else []
    warnings = [
        line for line in diagnostics.splitlines() if "warning:" in line
    ]
    return messages, errors, warnings


def main() -> None:
    results_dir = Path(sys.argv[1])
    output_path = Path(sys.argv[2])
    problems = [
        json.loads(line)
        for line in (results_dir / "problem_results.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    successes = []
    for problem in problems:
        for attempt in problem.get("attempts", []):
            messages, errors, warnings = _messages_from_legacy_diagnostics(attempt)
            success = bool(attempt.get("success")) or (
                bool(warnings) and not errors and not attempt.get("timed_out")
            )
            if success:
                successes.append(
                    {
                        "problem_id": problem["problem_id"],
                        "attempt_id": attempt["attempt_id"],
                        "problem_index": attempt.get("problem_index"),
                        "attempt_index": attempt.get("attempt_index"),
                        "prompt": problem["prompt"],
                        "lean_code": attempt["lean_code"],
                        "generated_proof": attempt["generated_proof"],
                        "status": "success",
                        "success": True,
                        "diagnostics": attempt.get("diagnostics", ""),
                        "compile_messages": messages,
                        "compile_errors": errors,
                        "compile_warnings": warnings,
                        "has_compile_errors": bool(errors),
                        "has_compile_warnings": bool(warnings),
                        "generation_seconds": attempt.get("generation_seconds"),
                        "queue_wait_seconds": attempt.get("queue_wait_seconds"),
                        "verification_seconds": attempt.get("verification_seconds"),
                        "total_seconds": attempt.get("total_seconds"),
                        "worker_id": attempt.get("worker_id"),
                        "verifier_backend": attempt.get("verifier_backend"),
                    }
                )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".jsonl":
        output_path.write_text(
            "".join(
                json.dumps(row, ensure_ascii=False) + "\n" for row in successes
            ),
            encoding="utf-8",
        )
    else:
        output_path.write_text(
            json.dumps(successes, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print(len(successes))


if __name__ == "__main__":
    main()
