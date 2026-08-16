"""Build static anchor features, deterministic buckets, and summaries."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from transformers import AutoTokenizer
from lean_prover.lean_training.ablation.difficulty import STATIC_DIFFICULTY_VERSION, assign_static_buckets, extract_static_features, summarize_records


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor", required=True, type=Path)
    parser.add_argument("--compile-results", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.anchor.read_text(encoding="utf-8").splitlines() if line.strip()]
    compile_rows = [json.loads(line) for line in args.compile_results.read_text(encoding="utf-8").splitlines() if line.strip()]
    compile_by_id = {str(row.get("record_id") or row.get("problem_id")): row for row in compile_rows}
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    features = []
    for row in rows:
        record_id = str(row.get("record_id") or row.get("id")); result = compile_by_id.get(record_id)
        if result is None or not result.get("success"): raise ValueError(f"missing successful compile audit for {record_id}")
        features.append(extract_static_features(row, tokenizer=tokenizer, compile_time_ms=float(result.get("verification_seconds") or 0) * 1000))
    thresholds = assign_static_buckets(features); args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "anchor_static_difficulty.jsonl").open("w", encoding="utf-8") as handle:
        for row in features: handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    subsets = {"all_anchor": summarize_records(features),
        "single_tactic_anchor": summarize_records([row for row in features if row["single_tactic"]]),
        "multi_step_anchor": summarize_records([row for row in features if not row["single_tactic"]])}
    summary = {"version": STATIC_DIFFICULTY_VERSION,
        "rules": {"easy": "hard flags absent and all six easy conditions hold", "hard": "any configured hard flag", "medium": "all remaining records"},
        "thresholds": thresholds, "subsets": subsets, "bucket_counts": dict(Counter(row["static_bucket"] for row in features))}
    (args.output_dir / "anchor_static_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Anchor Static Difficulty Audit", "", f"- Version: `{STATIC_DIFFICULTY_VERSION}`", f"- Records: {len(features)}",
        f"- Buckets: {summary['bucket_counts']}", f"- Single-tactic ratio: {subsets['all_anchor']['single_tactic_ratio']:.4%}",
        f"- Exact duplicate proof ratio: {subsets['all_anchor']['exact_duplicate_proof_ratio']:.4%}",
        f"- Distinct tactic signatures: {subsets['all_anchor']['distinct_tactic_signatures']}", "",
        "The buckets are deterministic proxy difficulty labels, not absolute mathematical difficulty."]
    (args.output_dir / "anchor_static_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__": main()
