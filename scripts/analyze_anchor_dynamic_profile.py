"""Bind M0 verification outcomes to static anchor features and test correlation."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path


def _rank(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.0] * len(values)
    index = 0
    while index < len(order):
        end = index + 1
        while end < len(order) and values[order[end]] == values[order[index]]:
            end += 1
        rank = (index + end - 1) / 2 + 1
        for offset in range(index, end):
            ranks[order[offset]] = rank
        index = end
    return ranks


def _pearson(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        return 0.0
    lm, rm = sum(left) / len(left), sum(right) / len(right)
    numerator = sum((a - lm) * (b - rm) for a, b in zip(left, right))
    denominator = math.sqrt(sum((a - lm) ** 2 for a in left) * sum((b - rm) ** 2 for b in right))
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-map", required=True, type=Path)
    parser.add_argument("--verifications", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    profile = [json.loads(line) for line in args.profile_map.read_text(encoding="utf-8").splitlines() if line.strip()]
    verification = [json.loads(line) for line in args.verifications.read_text(encoding="utf-8").splitlines() if line.strip()]
    success_by_id: dict[str, int] = Counter()
    attempts_by_id: dict[str, int] = Counter()
    for row in verification:
        sid = str(row["statement_id"])
        attempts_by_id[sid] += 1
        success_by_id[sid] += int(bool(row.get("verified")))
    dynamic_rows = []
    for row in profile:
        sid = str(row["evaluation_statement_id"])
        count = int(success_by_id[sid])
        attempts = int(attempts_by_id[sid])
        if attempts != 4:
            raise ValueError(f"{sid} has {attempts}, expected 4 attempts")
        dynamic = "dynamic_hard" if count == 0 else "dynamic_frontier" if count <= 2 else "dynamic_medium" if count == 3 else "dynamic_easy"
        dynamic_rows.append({**row, "m0_success_count_at_4": count, "m0_success_rate": count / 4, "dynamic_bucket": dynamic, "frontier": count in (1, 2)})
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "anchor_dynamic_difficulty.jsonl").open("w", encoding="utf-8") as handle:
        for row in dynamic_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    static_order = {"static_easy": 0.0, "static_medium": 1.0, "static_hard": 2.0}
    x = [static_order[row["static_bucket"]] for row in dynamic_rows]
    y = [float(row["m0_success_count_at_4"]) for row in dynamic_rows]
    spearman = _pearson(_rank(x), _rank(y))
    by_static = {}
    for bucket in static_order:
        values = [row["m0_success_count_at_4"] for row in dynamic_rows if row["static_bucket"] == bucket]
        by_static[bucket] = {"records": len(values), "mean_success_count_at_4": sum(values) / max(1, len(values)), "success_at_4": sum(v > 0 for v in values) / max(1, len(values))}
    monotonic = by_static["static_easy"]["mean_success_count_at_4"] >= by_static["static_medium"]["mean_success_count_at_4"] >= by_static["static_hard"]["mean_success_count_at_4"]
    summary = {
        "records": len(dynamic_rows),
        "attempts": len(verification),
        "verified_candidates": sum(y),
        "candidate_success_rate": sum(y) / max(1, len(verification)),
        "dynamic_bucket_counts": dict(Counter(row["dynamic_bucket"] for row in dynamic_rows)),
        "success_count_distribution": dict(Counter(str(row["m0_success_count_at_4"]) for row in dynamic_rows)),
        "static_bucket_performance": by_static,
        "spearman_static_ordinal_vs_m0_success_count": spearman,
        "absolute_spearman": abs(spearman),
        "monotonic_easy_to_hard": monotonic,
        "expand_remaining_anchor": abs(spearman) < 0.20 or not monotonic,
    }
    (args.output_dir / "anchor_dynamic_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "anchor_dynamic_summary.md").write_text(
        "# Anchor Dynamic Difficulty Audit\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in summary.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
