"""Materialize the fixed 100-step B1/B2 Expert-SFT ablation datasets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

from transformers import AutoTokenizer


TOTAL_DRAWS = 1600
B2_ANCHOR_DRAWS = 1440
B2_EXPERT_DRAWS = 160
MAX_ANCHOR_REPEAT = 3
DIFFICULTY_RATIOS = {
    "dynamic_hard": 0.25,
    "dynamic_frontier": 0.40,
    "dynamic_medium": 0.20,
    "dynamic_easy": 0.15,
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(rows: Iterable[dict[str, Any]], path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            payload = (json.dumps(row, ensure_ascii=False) + "\n").encode("utf-8")
            digest.update(payload)
            handle.write(payload.decode("utf-8"))
    return digest.hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode("utf-8")).hexdigest()


def row_id(row: dict[str, Any]) -> str:
    value = str(row.get("record_id") or row.get("id") or "").strip()
    if not value:
        raise ValueError("training row has no record_id/id")
    return value


def quotas(total: int) -> dict[str, int]:
    raw = {name: ratio * total for name, ratio in DIFFICULTY_RATIOS.items()}
    result = {name: math.floor(value) for name, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(raw, key=lambda name: (-(raw[name] - result[name]), name))
    for name in order[:remainder]:
        result[name] += 1
    return result


def coverage_first(
    pool: list[dict[str, Any]],
    count: int,
    *,
    seed: int,
    max_repeat: int,
) -> list[dict[str, Any]]:
    unique = {row_id(row): row for row in pool}
    ordered = sorted(unique.values(), key=lambda row: stable_key(seed, row_id(row)))
    draws: list[dict[str, Any]] = []
    for repeat in range(max_repeat):
        cycle = sorted(
            ordered,
            key=lambda row: stable_key(seed + 1009 * repeat, row_id(row)),
        )
        draws.extend(cycle[: min(len(cycle), count - len(draws))])
        if len(draws) == count:
            return draws
    raise ValueError(
        f"pool capacity {len(ordered) * max_repeat} is below requested {count}"
    )


def annotate(
    row: dict[str, Any],
    *,
    arm: str,
    source: str,
    draw_index: int,
) -> dict[str, Any]:
    result = dict(row)
    result.update(
        {
            "ablation_arm": arm,
            "ablation_source": source,
            "ablation_draw_index": draw_index,
            "sample_weight": 1.0,
        }
    )
    return result


def source_stats(rows: list[dict[str, Any]], tokenizer) -> dict[str, Any]:
    counts = Counter(row_id(row) for row in rows)
    label_tokens = sum(
        len(
            tokenizer(
                str(row.get("completion") or row.get("proof") or ""),
                add_special_tokens=False,
            )["input_ids"]
        )
        for row in rows
    )
    return {
        "draws": len(rows),
        "label_tokens": label_tokens,
        "unique_seen": len(counts),
        "mean_repeat": len(rows) / len(counts) if counts else 0.0,
        "max_repeat": max(counts.values(), default=0),
        "repeat_histogram": dict(Counter(str(value) for value in counts.values())),
    }


def build_arm(
    *,
    arm: str,
    anchors: list[dict[str, Any]],
    experts: list[dict[str, Any]],
    output_root: Path,
    tokenizer,
    seed: int,
    requested_anchor_ratio: float,
    requested_expert_ratio: float,
) -> dict[str, Any]:
    combined = [(row, "anchor") for row in anchors] + [
        (row, "expert") for row in experts
    ]
    shuffled = sorted(
        enumerate(combined),
        key=lambda item: stable_key(seed + 700, f"{arm}:{item[0]}"),
    )
    rows = [
        annotate(row, arm=arm, source=source, draw_index=index)
        for index, (_, (row, source)) in enumerate(shuffled)
    ]
    if len(rows) != TOTAL_DRAWS:
        raise ValueError(f"{arm} has {len(rows)} draws, expected {TOTAL_DRAWS}")
    if not all(row.get("pantograph_verified") is True for row in rows):
        raise ValueError(f"{arm} contains a row without pantograph_verified=True")

    anchor_stats = source_stats(anchors, tokenizer)
    expert_stats = source_stats(experts, tokenizer)
    total_label_tokens = anchor_stats["label_tokens"] + expert_stats["label_tokens"]
    dataset_path = output_root / arm / "train" / "train_dataset.jsonl"
    dataset_hash = write_jsonl(rows, dataset_path)
    trace = {
        "arm": arm,
        "seed": seed,
        "total_draws": len(rows),
        "effective_batch_size": 16,
        "optimizer_steps": len(rows) // 16,
        "requested_anchor_ratio": requested_anchor_ratio,
        "requested_expert_ratio": requested_expert_ratio,
        "actual_anchor_draw_ratio": len(anchors) / len(rows),
        "actual_expert_draw_ratio": len(experts) / len(rows),
        "anchor_label_token_share": (
            anchor_stats["label_tokens"] / total_label_tokens
            if total_label_tokens
            else 0.0
        ),
        "expert_label_token_share": (
            expert_stats["label_tokens"] / total_label_tokens
            if total_label_tokens
            else 0.0
        ),
        "anchor_unique_seen": anchor_stats["unique_seen"],
        "expert_unique_seen": expert_stats["unique_seen"],
        "anchor_mean_repeat": anchor_stats["mean_repeat"],
        "expert_mean_repeat": expert_stats["mean_repeat"],
        "anchor_max_repeat": anchor_stats["max_repeat"],
        "expert_max_repeat": expert_stats["max_repeat"],
        "anchor_draws": anchor_stats["draws"],
        "expert_draws": expert_stats["draws"],
        "anchor_label_tokens": anchor_stats["label_tokens"],
        "expert_label_tokens": expert_stats["label_tokens"],
        "anchor_repeat_histogram": anchor_stats["repeat_histogram"],
        "expert_repeat_histogram": expert_stats["repeat_histogram"],
        "anchor_difficulty_draw_counts": dict(
            Counter(str(row["ablation_difficulty_bucket"]) for row in anchors)
        ),
        "train_dataset": str(dataset_path),
        "train_dataset_sha256": dataset_hash,
        "all_rows_pantograph_verified": True,
    }
    trace_path = output_root / arm / "sampling_trace.json"
    trace_path.write_text(
        json.dumps(trace, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--balanced-anchor", required=True, type=Path)
    parser.add_argument("--frontier-expert", required=True, type=Path)
    parser.add_argument("--anchor-gate", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    balanced_rows = read_jsonl(args.balanced_anchor)
    anchor_by_bucket: defaultdict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in balanced_rows:
        bucket = str(row.get("ablation_difficulty_bucket") or "")
        if bucket not in DIFFICULTY_RATIOS:
            raise ValueError(f"invalid balanced anchor bucket {bucket!r}")
        anchor_by_bucket[bucket].append(row)

    def balanced_draws(count: int, seed_offset: int) -> list[dict[str, Any]]:
        draws: list[dict[str, Any]] = []
        for offset, (bucket, required) in enumerate(quotas(count).items()):
            draws.extend(
                coverage_first(
                    anchor_by_bucket[bucket],
                    required,
                    seed=args.seed + seed_offset + offset,
                    max_repeat=MAX_ANCHOR_REPEAT,
                )
            )
        return draws

    frontier = read_jsonl(args.frontier_expert)
    if len({row_id(row) for row in frontier}) != len(frontier):
        raise ValueError("frontier expert pool has more than one proof per statement")
    expert_draws = coverage_first(
        frontier,
        B2_EXPERT_DRAWS,
        seed=args.seed + 500,
        max_repeat=2,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    args.output_root.mkdir(parents=True, exist_ok=True)
    traces = {
        "B1": build_arm(
            arm="B1_balanced_anchor_only",
            anchors=balanced_draws(TOTAL_DRAWS, 0),
            experts=[],
            output_root=args.output_root,
            tokenizer=tokenizer,
            seed=args.seed,
            requested_anchor_ratio=1.0,
            requested_expert_ratio=0.0,
        ),
        "B2": build_arm(
            arm="B2_balanced_anchor_frontier_90_10",
            anchors=balanced_draws(B2_ANCHOR_DRAWS, 100),
            experts=expert_draws,
            output_root=args.output_root,
            tokenizer=tokenizer,
            seed=args.seed,
            requested_anchor_ratio=0.9,
            requested_expert_ratio=0.1,
        ),
    }

    anchor_gate_rows = read_jsonl(args.anchor_gate)
    if len(anchor_gate_rows) != 150:
        raise ValueError(f"expected fixed 150-row anchor gate, got {len(anchor_gate_rows)}")
    retention_path = args.output_root / "gates" / "retention_gate_50.jsonl"
    retention_hash = write_jsonl(anchor_gate_rows[:50], retention_path)
    manifest = {
        "seed": args.seed,
        "initialization_checkpoint": args.tokenizer,
        "source_balanced_anchor": str(args.balanced_anchor),
        "source_balanced_anchor_sha256": file_sha256(args.balanced_anchor),
        "source_frontier_expert": str(args.frontier_expert),
        "source_frontier_expert_sha256": file_sha256(args.frontier_expert),
        "source_anchor_gate": str(args.anchor_gate),
        "source_anchor_gate_sha256": file_sha256(args.anchor_gate),
        "retention_gate": str(retention_path),
        "retention_gate_sha256": retention_hash,
        "retention_selection": "first_50_of_fixed_anchor_gate_150_in_source_order",
        "traces": traces,
    }
    (args.output_root / "train_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
