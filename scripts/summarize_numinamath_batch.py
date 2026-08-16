"""Print a compact, read-only summary of a staged NuminaMath repair batch."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path)
    parser.add_argument("--examples", type=int, default=10)
    parser.add_argument("--include-source", action="store_true")
    parser.add_argument("--failures-only", action="store_true")
    parser.add_argument("--source-preview-chars", type=int, default=1000)
    parser.add_argument("--diagnostic-chars", type=int, default=300)
    args = parser.parse_args()

    results = read_jsonl(args.batch_dir / "verification_results.jsonl")
    candidates = read_jsonl(args.batch_dir / "candidate_manifest.jsonl")
    candidate_by_id = {str(row.get("record_id") or ""): row for row in candidates}
    api_observations = read_jsonl(args.batch_dir / "api_observations.jsonl")
    by_record: dict[str, list[dict[str, object]]] = {}
    for row in results:
        by_record.setdefault(str(row.get("record_id") or ""), []).append(row)

    records_solved = 0
    final_error_types: Counter[str] = Counter()
    success_strategies: Counter[str] = Counter()
    diagnostic_heads: Counter[str] = Counter()
    examples: list[dict[str, object]] = []
    for record_id, attempts in by_record.items():
        success = next((row for row in attempts if row.get("success")), None)
        chosen = success or attempts[-1]
        records_solved += int(success is not None)
        if success is not None:
            success_strategies[str(success.get("strategy") or "unknown")] += 1
        error_type = str(chosen.get("error_type") or "success")
        final_error_types[error_type] += 1
        errors = chosen.get("errors") if isinstance(chosen.get("errors"), list) else []
        diagnostic_lines = str(errors[0] if errors else "").splitlines()
        diagnostic = str(errors[0] if errors else "")[: args.diagnostic_chars]
        if diagnostic:
            diagnostic_heads[diagnostic] += 1
        if len(examples) < args.examples and (not args.failures_only or success is None):
            example = {
                "record_id": record_id,
                "success": success is not None,
                "attempts": len(attempts),
                "final_strategy": chosen.get("strategy"),
                "error_type": error_type,
                "diagnostic_head": diagnostic,
            }
            if args.include_source:
                source = str(candidate_by_id.get(record_id, {}).get("original_source_body") or "")
                example["source_preview"] = source[: args.source_preview_chars]
            examples.append(example)

    print(json.dumps({
        "batch_dir": str(args.batch_dir.resolve()),
        "candidate_rows": len(candidates),
        "candidate_record_ids_preview": [
            str(row.get("record_id") or "") for row in candidates[: args.examples]
        ],
        "api_observation_rows": len(api_observations),
        "api_statuses": dict(Counter(
            str(row.get("status") or "unknown") for row in api_observations
        ).most_common()),
        "api_quota_error_rows": sum(
            str(row.get("status") or "") == "skipped_after_quota_exhaustion"
            or "quota" in str(row.get("error") or "").casefold()
            or "insufficient balance" in str(row.get("error") or "").casefold()
            for row in api_observations
        ),
        "result_rows": len(results),
        "records": len(by_record),
        "records_solved": records_solved,
        "records_failed": len(by_record) - records_solved,
        "success_strategies": dict(success_strategies.most_common()),
        "final_error_types": dict(final_error_types.most_common()),
        "diagnostic_heads": dict(diagnostic_heads.most_common(20)),
        "examples": examples,
        "read_only": True,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
