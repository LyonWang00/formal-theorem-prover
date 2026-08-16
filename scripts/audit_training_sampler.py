"""Rebuild and audit the current weighted Expert-SFT sampling contract."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoTokenizer


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def completion_contract(row: dict[str, Any], tokenizer, max_length: int) -> dict[str, Any]:
    prompt = str(row["prompt"])
    target = str(row["completion"])
    completion = target
    if tokenizer.eos_token and not completion.endswith(tokenizer.eos_token):
        completion += tokenizer.eos_token
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(prompt + completion, add_special_tokens=False)["input_ids"]
    prefix_ok = full_ids[: len(prompt_ids)] == prompt_ids
    end = min(len(full_ids), max_length)
    start = min(len(prompt_ids), end)
    label_ids = full_ids[start:end]
    labeled_eos = bool(label_ids and tokenizer.eos_token_id == label_ids[-1])
    return {
        "label_tokens": len(label_ids),
        "label_range": [start, end],
        "labeled_eos": labeled_eos,
        "prefix_matches": prefix_ok,
        "input_ids": full_ids[:end],
        "decoded_sequence": tokenizer.decode(full_ids[:end], skip_special_tokens=False),
        "eos_token": tokenizer.eos_token,
        "eos_token_id": tokenizer.eos_token_id,
    }


def source_name(row: dict[str, Any]) -> str:
    return "expert" if "expert" in str(row.get("expert_source") or "") else "anchor_data"


def trace_sampler(args: argparse.Namespace) -> dict[str, Any]:
    rows = read_jsonl(Path(args.train_file))
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    contracts = [completion_contract(row, tokenizer, args.max_seq_length) for row in rows]
    weights = torch.tensor([float(row["sample_weight"]) for row in rows], dtype=torch.double)
    if not torch.isfinite(weights).all() or torch.any(weights <= 0):
        raise ValueError("sample weights must be finite and positive")

    # Match the current trainer: global torch RNG, replacement sampling, and
    # one DataLoader iterator base-seed draw before the sampler is consumed.
    torch.manual_seed(args.seed)
    sampler = WeightedRandomSampler(weights, num_samples=len(rows), replacement=True)
    loader = DataLoader(list(range(len(rows))), batch_size=1, sampler=sampler)
    drawn_indices = [int(batch.item()) for batch in loader]
    if (len(drawn_indices) + args.gradient_accumulation - 1) // args.gradient_accumulation != args.optimizer_steps:
        raise ValueError("draw count does not reproduce the requested optimizer-step budget")

    trace: list[dict[str, Any]] = []
    draw_counts: Counter[int] = Counter()
    for draw_index, row_index in enumerate(drawn_indices):
        draw_counts[row_index] += 1
        row = rows[row_index]
        trace.append(
            {
                "draw_index": draw_index,
                "micro_batch_index": draw_index,
                "optimizer_step": draw_index // args.gradient_accumulation + 1,
                "row_index": row_index,
                "record_id": row.get("record_id") or row.get("id"),
                "source": source_name(row),
                "completion_label_tokens": contracts[row_index]["label_tokens"],
                "draw_number_for_record": draw_counts[row_index],
            }
        )
    output = Path(args.output_dir)
    write_jsonl(output / "current_recipe_sampling_trace.jsonl", trace)

    physical = Counter(source_name(row) for row in rows)
    draws = Counter(row["source"] for row in trace)
    label_tokens = Counter()
    indices_by_source: dict[str, set[int]] = {"anchor_data": set(), "expert": set()}
    repeats_by_source: dict[str, list[int]] = {"anchor_data": [], "expert": []}
    for item in trace:
        label_tokens[item["source"]] += int(item["completion_label_tokens"])
        indices_by_source[item["source"]].add(int(item["row_index"]))
    for source in ("anchor_data", "expert"):
        repeats_by_source[source] = [
            count for index, count in draw_counts.items() if source_name(rows[index]) == source
        ]
    total_draws = len(trace)
    total_labels = sum(label_tokens.values())
    summary = {
        "contract": "current_m1_weighted_sampler_reconstruction_v1",
        "seed": args.seed,
        "optimizer_steps": args.optimizer_steps,
        "gradient_accumulation": args.gradient_accumulation,
        "effective_batch_size": args.gradient_accumulation,
        "total_physical_rows": len(rows),
        "physical_rows": dict(physical),
        "physical_row_ratio": {key: value / len(rows) for key, value in physical.items()},
        "total_draws": total_draws,
        "anchor_draws": draws["anchor_data"],
        "expert_draws": draws["expert"],
        "anchor_draw_ratio": draws["anchor_data"] / total_draws,
        "expert_draw_ratio": draws["expert"] / total_draws,
        "anchor_unique_records_seen": len(indices_by_source["anchor_data"]),
        "expert_unique_records_seen": len(indices_by_source["expert"]),
        "anchor_coverage_ratio": len(indices_by_source["anchor_data"]) / physical["anchor_data"],
        "expert_coverage_ratio": len(indices_by_source["expert"]) / physical["expert"],
        "anchor_mean_draws_per_seen_record": draws["anchor_data"] / max(1, len(indices_by_source["anchor_data"])),
        "expert_mean_draws_per_seen_record": draws["expert"] / max(1, len(indices_by_source["expert"])),
        "anchor_max_draws_per_record": max(repeats_by_source["anchor_data"], default=0),
        "expert_max_draws_per_record": max(repeats_by_source["expert"], default=0),
        "anchor_completion_label_tokens": label_tokens["anchor_data"],
        "expert_completion_label_tokens": label_tokens["expert"],
        "anchor_label_token_share": label_tokens["anchor_data"] / max(1, total_labels),
        "expert_label_token_share": label_tokens["expert"] / max(1, total_labels),
        "requested_draw_mix": {"anchor_data": 0.85, "expert": 0.15},
        "mixture_within_one_percentage_point": abs(draws["expert"] / total_draws - 0.15) <= 0.01,
    }
    write_json(output / "current_recipe_sampling_trace.json", summary)

    rng = random.Random(args.seed)
    samples: list[dict[str, Any]] = []
    format_checks: dict[str, dict[str, int]] = {}
    for source in ("anchor_data", "expert"):
        candidates = [index for index, row in enumerate(rows) if source_name(row) == source]
        chosen = sorted(rng.sample(candidates, min(args.format_samples, len(candidates))))
        checks = Counter()
        for index in chosen:
            row = rows[index]
            contract = contracts[index]
            prompt = str(row["prompt"])
            target = str(row["completion"])
            row_checks = {
                "standard_prompt_sections": prompt.startswith("### Informal statement\n")
                and "\n\n### Lean statement\n" in prompt
                and prompt.endswith("\n\n### Lean proof\n"),
                "target_is_full_by_proof": target.lstrip().startswith("by"),
                "labeled_eos": contract["labeled_eos"],
                "prefix_matches": contract["prefix_matches"],
                "no_markdown_fence": "```" not in target,
                "no_repeated_theorem": not bool(re.search(r"\b(theorem|lemma|example)\b", target)),
                "no_reference_annotation": "reference proof" not in prompt.lower(),
            }
            checks.update({key: int(value) for key, value in row_checks.items()})
            samples.append(
                {
                    "source": source,
                    "index": index,
                    "record_id": row.get("record_id") or row.get("id"),
                    "raw_record": row,
                    "formatted_prompt": prompt,
                    "formatted_assistant_target": target,
                    "tokenizer_decoded_sequence": contract["decoded_sequence"],
                    "completion_label_token_range": contract["label_range"],
                    "completion_label_tokens": contract["label_tokens"],
                    "eos_token": contract["eos_token"],
                    "proof_format": row.get("proof_format") or "full_proof",
                    "checks": row_checks,
                }
            )
        format_checks[source] = dict(checks)
    comparison = {
        "sample_count_by_source": dict(Counter(row["source"] for row in samples)),
        "check_pass_counts": format_checks,
        "same_prompt_template": all(row["checks"]["standard_prompt_sections"] for row in samples),
        "same_full_by_proof_format": all(row["checks"]["target_is_full_by_proof"] for row in samples),
        "all_have_labeled_eos": all(row["checks"]["labeled_eos"] for row in samples),
        "all_prefixes_match": all(row["checks"]["prefix_matches"] for row in samples),
        "all_without_markdown_fence": all(row["checks"]["no_markdown_fence"] for row in samples),
        "all_without_repeated_theorem": all(row["checks"]["no_repeated_theorem"] for row in samples),
        "samples": samples,
    }
    write_json(output / "source_format_comparison.json", comparison)
    lines = [
        "# Anchor/Expert Source Format Audit",
        "",
        f"- Samples: {comparison['sample_count_by_source']}",
        f"- Same prompt template: {comparison['same_prompt_template']}",
        f"- Full `by ...` proof targets: {comparison['same_full_by_proof_format']}",
        f"- All have labeled EOS: {comparison['all_have_labeled_eos']}",
        f"- Prompt/full tokenization prefixes match: {comparison['all_prefixes_match']}",
        f"- No Markdown fences: {comparison['all_without_markdown_fence']}",
        f"- No repeated theorem declaration: {comparison['all_without_repeated_theorem']}",
    ]
    (output / "source_format_comparison.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"sampling": summary, "format": {key: value for key, value in comparison.items() if key != "samples"}}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-file", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--optimizer-steps", type=int, default=202)
    parser.add_argument("--gradient-accumulation", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--format-samples", type=int, default=20)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    print(json.dumps(trace_sampler(args), ensure_ascii=False, indent=2))
