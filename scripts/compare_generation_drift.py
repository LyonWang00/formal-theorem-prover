"""Compare raw M0/M1 proof-generation behavior without regenerating candidates."""

from __future__ import annotations

import argparse
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from lean_prover.lean_training.evaluation.benchmark import classify_generation_output


TACTICS = (
    "simp",
    "simp_all",
    "norm_num",
    "linarith",
    "nlinarith",
    "ring",
    "ring_nf",
    "omega",
    "aesop",
    "exact",
    "rfl",
    "decide",
    "native_decide",
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    lower, upper = math.floor(position), math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def normalize_proof(row: dict[str, Any]) -> str:
    proof = str(row.get("normalized_proof") or row.get("extracted_proof") or "").strip()
    return re.sub(r"\s+", " ", proof)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = [
        int(row.get("metadata", {}).get("completion_tokens") or len(str(row.get("raw_output") or "").split()))
        for row in rows
    ]
    finish = Counter(str(row.get("finish_reason") or row.get("metadata", {}).get("finish_reason") or "unknown") for row in rows)
    classifications = Counter(classify_generation_output(str(row.get("raw_output") or "")) for row in rows)
    normalized = [normalize_proof(row) for row in rows]
    nonempty = [proof for proof in normalized if proof]
    proof_frequency = Counter(nonempty)
    by_statement: dict[str, set[str]] = defaultdict(set)
    for row, proof in zip(rows, normalized, strict=True):
        if proof:
            by_statement[str(row["statement_id"])].add(proof)
    tactic_counts = {
        tactic: sum(bool(re.search(rf"\b{re.escape(tactic)}\b", proof)) for proof in nonempty)
        for tactic in TACTICS
    }
    return {
        "candidates": len(rows),
        "statements": len(by_statement),
        "mean_output_tokens": sum(lengths) / max(1, len(lengths)),
        "p50_output_tokens": percentile(lengths, 0.50),
        "p90_output_tokens": percentile(lengths, 0.90),
        "p95_output_tokens": percentile(lengths, 0.95),
        "max_output_tokens": max(lengths, default=0),
        "finish_reason_distribution": dict(finish),
        "length_finish_ratio": finish.get("length", 0) / max(1, len(rows)),
        "repetitive_output_ratio": classifications.get("repetitive_output", 0) / max(1, len(rows)),
        "multiple_proof_ratio": classifications.get("multiple_proofs", 0) / max(1, len(rows)),
        "extraction_success_rate": len(nonempty) / max(1, len(rows)),
        "unique_normalized_proof_ratio": len(proof_frequency) / max(1, len(nonempty)),
        "mean_unique_candidates_per_statement": sum(map(len, by_statement.values())) / max(1, len(by_statement)),
        "top10_output_proof_coverage": sum(count for _, count in proof_frequency.most_common(10)) / max(1, len(nonempty)),
        "top50_output_proof_coverage": sum(count for _, count in proof_frequency.most_common(50)) / max(1, len(nonempty)),
        "classification_distribution": dict(classifications),
        "tactic_presence_counts": tactic_counts,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--m0", required=True, type=Path)
    parser.add_argument("--m1", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    result = {"M0": summarize(read_jsonl(args.m0)), "M1": summarize(read_jsonl(args.m1))}
    result["delta"] = {
        key: result["M1"][key] - result["M0"][key]
        for key in (
            "mean_output_tokens",
            "length_finish_ratio",
            "repetitive_output_ratio",
            "multiple_proof_ratio",
            "unique_normalized_proof_ratio",
            "mean_unique_candidates_per_statement",
            "top10_output_proof_coverage",
        )
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "m0_m1_generation_drift.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    lines = ["# M0/M1 Generation Drift", ""]
    for name in ("M0", "M1"):
        row = result[name]
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Mean output tokens: {row['mean_output_tokens']:.4f}",
                f"- P50/P90/P95/max: {row['p50_output_tokens']}/{row['p90_output_tokens']}/{row['p95_output_tokens']}/{row['max_output_tokens']}",
                f"- Length finish ratio: {row['length_finish_ratio']:.4%}",
                f"- Repetitive output ratio: {row['repetitive_output_ratio']:.4%}",
                f"- Multiple proof ratio: {row['multiple_proof_ratio']:.4%}",
                f"- Unique normalized proof ratio: {row['unique_normalized_proof_ratio']:.4%}",
                f"- Mean unique candidates/statement: {row['mean_unique_candidates_per_statement']:.4f}",
                f"- Top-10/Top-50 coverage: {row['top10_output_proof_coverage']:.4%}/{row['top50_output_proof_coverage']:.4%}",
                "",
            ]
        )
    (args.output_dir / "m0_m1_generation_drift.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
