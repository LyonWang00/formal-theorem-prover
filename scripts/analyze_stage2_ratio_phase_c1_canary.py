"""Analyze paired M0/S2-New-WB behavior canaries and enforce the Phase C1 gate."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any


DATASETS = ("wb_retention_canary30", "ld_easy_canary20", "monitor_canary20")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[int], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    pos = (len(ordered) - 1) * q
    lo, hi = int(pos), min(int(pos) + 1, len(ordered) - 1)
    return ordered[lo] * (hi - pos) + ordered[hi] * (pos - lo)


def repetition_ratio(text: str, n: int = 4) -> float:
    tokens = re.findall(r"\w+|[^\w\s]", text, flags=re.UNICODE)
    grams = [tuple(tokens[i:i+n]) for i in range(max(0, len(tokens) - n + 1))]
    return 0.0 if not grams else 1.0 - len(set(grams)) / len(grams)


def repeated_line(text: str) -> bool:
    lines = [re.sub(r"\s+", " ", line.strip()) for line in text.splitlines()]
    counts = Counter(line for line in lines if len(line) >= 8)
    return any(count >= 3 for count in counts.values())


def summarize(directory: Path) -> dict[str, Any]:
    generations = read_jsonl(directory / "generations.jsonl")
    attempts = read_jsonl(directory / "attempts.jsonl")
    summary_paths = list(directory.glob("*_summary.json")) + list(directory.glob("*_metrics.json"))
    summaries = [json.loads(path.read_text(encoding="utf-8")) for path in summary_paths]
    lengths = [int(row.get("metadata", {}).get("completion_tokens", 0)) for row in generations]
    reps = [repetition_ratio(str(row.get("raw_output", ""))) for row in generations]
    line_reps = [repeated_line(str(row.get("raw_output", ""))) for row in generations]
    nonempty = [bool(str(row.get("extracted_proof", "")).strip()) for row in generations]
    starts_by = [str(row.get("extracted_proof", "")).lstrip().startswith("by") for row in generations]
    max_finish = [
        row.get("finish_reason") in {"length", "max_length"}
        or int(row.get("metadata", {}).get("completion_tokens", 0)) >= int(row.get("max_new_tokens", 256))
        for row in generations
    ]
    return {
        "candidate_count": len(generations), "attempt_count": len(attempts),
        "successes": sum(bool(row.get("success")) for row in attempts),
        "timeouts": sum(bool(row.get("timed_out")) for row in attempts),
        "completion_tokens": {"mean": mean(lengths) if lengths else 0.0, "p50": percentile(lengths, .5), "p95": percentile(lengths, .95), "max": max(lengths, default=0)},
        "max_length_finish_ratio": sum(max_finish) / len(max_finish) if max_finish else 0.0,
        "extraction_success_rate": sum(nonempty) / len(nonempty) if nonempty else 0.0,
        "starts_with_by_rate": sum(starts_by) / len(starts_by) if starts_by else 0.0,
        "mean_repeated_4gram_ratio": mean(reps) if reps else 0.0,
        "pathological_repetition_rate": sum(line or ratio >= .35 for line, ratio in zip(line_reps, reps)) / len(reps) if reps else 0.0,
        "fatal_errors": sum(len(item.get("fatal_errors", [])) for item in summaries),
        "worker_restarts": sum(int(item.get("pantograph_worker_restart_count", 0)) for item in summaries),
    }


def aggregate(items: list[dict[str, Any]]) -> dict[str, Any]:
    n, attempts = sum(x["candidate_count"] for x in items), sum(x["attempt_count"] for x in items)
    weighted = lambda key: sum(x[key] * x["candidate_count"] for x in items) / n if n else 0.0
    return {
        "candidate_count": n, "attempt_count": attempts,
        "successes": sum(x["successes"] for x in items), "timeouts": sum(x["timeouts"] for x in items),
        "completion_tokens_mean": sum(x["completion_tokens"]["mean"] * x["candidate_count"] for x in items) / n if n else 0.0,
        "max_completion_tokens": max(x["completion_tokens"]["max"] for x in items),
        "max_length_finish_ratio": weighted("max_length_finish_ratio"),
        "extraction_success_rate": weighted("extraction_success_rate"),
        "starts_with_by_rate": weighted("starts_with_by_rate"),
        "mean_repeated_4gram_ratio": weighted("mean_repeated_4gram_ratio"),
        "pathological_repetition_rate": weighted("pathological_repetition_rate"),
        "fatal_errors": sum(x["fatal_errors"] for x in items), "worker_restarts": sum(x["worker_restarts"] for x in items),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase-root", type=Path, required=True)
    parser.add_argument("--challenger", default="S2-New-WB")
    args = parser.parse_args()
    root = args.phase_root.resolve()
    models = ("M0", args.challenger)
    canary = root / "evaluation/canary"
    per_dataset = {dataset: {model: summarize(canary / model / dataset) for model in models} for dataset in DATASETS}
    combined = {model: aggregate([per_dataset[d][model] for d in DATASETS]) for model in models}
    m0, s2 = combined["M0"], combined[args.challenger]
    checks = {
        "all_140_candidates_per_model_recorded": m0["candidate_count"] == 140 and s2["candidate_count"] == 140,
        "no_fatal_errors": s2["fatal_errors"] == 0,
        "no_worker_restarts": s2["worker_restarts"] == 0,
        "extraction_at_least_90pct": s2["extraction_success_rate"] >= .90,
        "proof_prefix_format_not_regressed": s2["starts_with_by_rate"] >= m0["starts_with_by_rate"] - .05,
        "max_length_finish_at_most_30pct": s2["max_length_finish_ratio"] <= .30,
        "mean_length_not_regressed": s2["completion_tokens_mean"] <= max(m0["completion_tokens_mean"] * 1.5, m0["completion_tokens_mean"] + 20),
        "pathological_repetition_not_regressed": s2["pathological_repetition_rate"] <= max(m0["pathological_repetition_rate"] + .10, .20),
        "timeout_rate_at_most_5pct": s2["timeouts"] / max(1, s2["attempt_count"]) <= .05,
    }
    result = {"gate_passed": all(checks.values()), "checks": checks, "combined": combined, "per_dataset": per_dataset}
    (canary / "canary_gate.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Phase C1 behavior canary gate", "", f"**Gate: {'PASS' if result['gate_passed'] else 'FAIL'}**", "", "| Check | Result |", "|---|---:|"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} |" for name, ok in checks.items()]
    lines += ["", "## Combined metrics", "", f"| Metric | M0 | {args.challenger} |", "|---|---:|---:|"]
    for key in ("candidate_count", "successes", "timeouts", "completion_tokens_mean", "max_completion_tokens", "max_length_finish_ratio", "extraction_success_rate", "starts_with_by_rate", "mean_repeated_4gram_ratio", "pathological_repetition_rate", "fatal_errors", "worker_restarts"):
        lines.append(f"| {key} | {m0[key]:.6g} | {s2[key]:.6g} |")
    lines += ["", "## Dataset details", ""]
    for dataset in DATASETS:
        lines += [f"### {dataset}", "", "| Model | Success | Timeout | Mean tokens | P95 | Max-length ratio | Extraction | Pathological repetition |", "|---|---:|---:|---:|---:|---:|---:|---:|"]
        for model in models:
            item = per_dataset[dataset][model]
            lines.append(f"| {model} | {item['successes']}/{item['attempt_count']} | {item['timeouts']} | {item['completion_tokens']['mean']:.2f} | {item['completion_tokens']['p95']:.2f} | {item['max_length_finish_ratio']:.2%} | {item['extraction_success_rate']:.2%} | {item['pathological_repetition_rate']:.2%} |")
        lines.append("")
    (canary / "canary_gate.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"gate_passed": result["gate_passed"], "checks": checks, "combined": combined}, indent=2))


if __name__ == "__main__":
    main()
