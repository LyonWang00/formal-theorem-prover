"""Audit proof quality at all four Expert SFT selection/exposure layers."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import re
from pathlib import Path
from typing import Any

from transformers import AutoTokenizer


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
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def percentile(values: list[int], fraction: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * fraction
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def normalize(proof: str) -> str:
    return re.sub(r"\s+", " ", proof.strip())


def summarize(rows: list[dict[str, Any]], tokenizer: Any) -> dict[str, Any]:
    proofs = [str(row["proof"]) for row in rows]
    normalized = [normalize(proof) for proof in proofs]
    frequencies = Counter(normalized)
    lengths = [len(tokenizer(proof, add_special_tokens=False)["input_ids"]) for proof in proofs]
    tactics = {
        tactic: sum(bool(re.search(rf"\b{re.escape(tactic)}\b", proof)) for proof in normalized)
        for tactic in TACTICS
    }
    covered = sum(tactics.values())
    return {
        "records_or_draws": len(rows),
        "unique_normalized_proofs": len(frequencies),
        "duplicate_proof_ratio": 1 - len(frequencies) / len(rows),
        "top10_proof_coverage": sum(count for _, count in frequencies.most_common(10)) / len(rows),
        "top50_proof_coverage": sum(count for _, count in frequencies.most_common(50)) / len(rows),
        "proof_tokens": {
            "mean": sum(lengths) / len(lengths),
            "p50": percentile(lengths, 0.50),
            "p90": percentile(lengths, 0.90),
            "p95": percentile(lengths, 0.95),
            "max": max(lengths),
            "total": sum(lengths),
        },
        "tactic_presence_counts": tactics,
        "other_count": sum(not any(re.search(rf"\b{re.escape(tactic)}\b", proof) for tactic in TACTICS) for proof in normalized),
        "total_named_tactic_presences": covered,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round1", type=Path, default=Path("outputs/expert_iteration_round1"))
    parser.add_argument("--ablation", type=Path, default=Path("outputs/expert_sft_anchor_ablation"))
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("outputs/qwen25_1_5b_clean_m0_verified_v2/initial_sft/merged_anchor"),
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)

    successful = read_jsonl(args.round1 / "iteration_000/discovery/successful_candidates.jsonl")
    proof_bank = read_jsonl(args.round1 / "proof_bank/proof_bank.jsonl")
    selected = [
        row
        for row in proof_bank
        if int(row.get("iteration_found", -1)) == 0 and bool(row.get("selected_as_primary"))
    ]
    frontier = read_jsonl(args.ablation / "expert_frontier_pool.jsonl")
    trace = read_jsonl(args.ablation / "audit/current_recipe_sampling_trace.jsonl")
    proof_by_id = {str(row["proof_id"]): str(row["proof"]) for row in proof_bank}
    actual_draws = [
        {"proof": proof_by_id[str(row["record_id"])], **row}
        for row in trace
        if row["source"] == "expert"
    ]
    if (len(successful), len(selected), len(frontier), len(actual_draws)) != (456, 226, 153, 493):
        raise ValueError(
            "unexpected layer sizes: "
            f"{len(successful)}, {len(selected)}, {len(frontier)}, {len(actual_draws)}"
        )
    layers = {
        "all_verified_candidates": summarize(successful, tokenizer),
        "selected_expert_proofs": summarize(selected, tokenizer),
        "frontier_selected_proofs": summarize(frontier, tokenizer),
        "actual_current_recipe_expert_draws": summarize(actual_draws, tokenizer),
    }
    layers["actual_current_recipe_expert_draws"]["recorded_completion_label_tokens"] = sum(
        int(row["completion_label_tokens"]) for row in actual_draws
    )
    output = args.ablation / "expert_quality_layers.json"
    output.write_text(json.dumps(layers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Expert Proof Quality by Selection Layer", ""]
    for name, row in layers.items():
        tokens = row["proof_tokens"]
        lines.extend(
            [
                f"## {name}",
                "",
                f"- Records/draws: {row['records_or_draws']}",
                f"- Mean/P50/P90/P95/max proof tokens: {tokens['mean']:.4f}/{tokens['p50']}/{tokens['p90']}/{tokens['p95']}/{tokens['max']}",
                f"- Duplicate ratio: {row['duplicate_proof_ratio']:.4%}",
                f"- Top-10/Top-50 coverage: {row['top10_proof_coverage']:.4%}/{row['top50_proof_coverage']:.4%}",
                f"- Tactic presence: {json.dumps(row['tactic_presence_counts'], ensure_ascii=False, sort_keys=True)}",
                "",
            ]
        )
    (args.ablation / "expert_quality_layers.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(layers, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
