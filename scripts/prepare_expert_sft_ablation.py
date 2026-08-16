"""Materialize reproducible Anchor/Expert ablation draws and their token audits."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

from transformers import AutoTokenizer


TOTAL_DRAWS = 3232
MIXED_ANCHOR_DRAWS = 2910
MIXED_EXPERT_DRAWS = 322
MAX_BALANCED_REPEAT = 3
DIFFICULTY_RATIOS = {"dynamic_hard": .25, "dynamic_frontier": .40, "dynamic_medium": .20, "dynamic_easy": .15}


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_key(seed: int, value: str) -> str:
    return hashlib.sha256(f"{seed}:{value}".encode()).hexdigest()


def quota(total: int) -> dict[str, int]:
    raw = {key: total * ratio for key, ratio in DIFFICULTY_RATIOS.items()}
    result = {key: math.floor(value) for key, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - result[key]), key))
    for key in order[:remainder]:
        result[key] += 1
    return result


def coverage_first(pool: list[dict], count: int, seed: int, identity) -> list[dict]:
    ordered = sorted(pool, key=lambda row: stable_key(seed, identity(row)))
    draws = []
    for repeat in range(MAX_BALANCED_REPEAT):
        round_rows = sorted(ordered, key=lambda row: stable_key(seed + 1009 * repeat, identity(row)))
        take = min(len(round_rows), count - len(draws))
        draws.extend(round_rows[:take])
        if len(draws) == count:
            return draws
    raise ValueError(f"pool capacity {len(pool) * MAX_BALANCED_REPEAT} is below requested {count}")


def annotate(row: dict, *, arm: str, source: str, difficulty: str | None, draw_index: int) -> dict:
    result = dict(row)
    result.update({
        "ablation_arm": arm,
        "ablation_source": source,
        "ablation_difficulty_bucket": difficulty,
        "ablation_draw_index": draw_index,
        "sample_weight": 1.0,
    })
    return result


def tactic_signature(proof: str) -> str:
    tactics = [name for name in ("simp_all", "norm_num", "nlinarith", "ring_nf", "linarith", "omega", "aesop", "exact", "simp", "ring", "rfl", "decide") if re.search(rf"\b{name}\b", proof)]
    return "+".join(tactics) if tactics else "other"


def stats(rows: list[dict], tokenizer) -> dict:
    ids = [str(row.get("record_id") or row.get("id")) for row in rows]
    completions = [str(row.get("completion") or row.get("proof") or "") for row in rows]
    token_counts = [len(tokenizer(value, add_special_tokens=False)["input_ids"]) for value in completions]
    counts = Counter(ids)
    source_counts = Counter(str(row.get("ablation_source") or row.get("expert_source")) for row in rows)
    difficulty_counts = Counter(str(row.get("ablation_difficulty_bucket")) for row in rows if row.get("ablation_difficulty_bucket"))
    source_tokens = Counter()
    difficulty_tokens = Counter()
    for row, tokens in zip(rows, token_counts):
        source_tokens[str(row.get("ablation_source") or row.get("expert_source"))] += tokens
        if row.get("ablation_difficulty_bucket"):
            difficulty_tokens[str(row["ablation_difficulty_bucket"])] += tokens
    total_tokens = sum(token_counts)
    return {
        "draws": len(rows),
        "unique_records": len(counts),
        "coverage_ratio_against_anchor_3000": len(counts) / 3000,
        "max_repeat_per_record": max(counts.values(), default=0),
        "repeat_histogram": dict(Counter(str(value) for value in counts.values())),
        "source_draw_counts": dict(source_counts),
        "source_draw_ratios": {key: value / max(1, len(rows)) for key, value in source_counts.items()},
        "difficulty_draw_counts": dict(difficulty_counts),
        "completion_label_tokens": total_tokens,
        "source_label_tokens": dict(source_tokens),
        "source_label_token_ratios": {key: value / max(1, total_tokens) for key, value in source_tokens.items()},
        "difficulty_label_tokens": dict(difficulty_tokens),
        "mean_completion_tokens": total_tokens / max(1, len(rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current-train", required=True, type=Path)
    parser.add_argument("--static", required=True, type=Path)
    parser.add_argument("--dynamic", required=True, type=Path)
    parser.add_argument("--discovery-verifications", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    current = read_jsonl(args.current_train)
    anchors = [row for row in current if row.get("expert_source") == "anchor"]
    experts = [row for row in current if row.get("expert_source") != "anchor"]
    if len(anchors) != 3000:
        raise ValueError(f"expected 3000 anchor rows, got {len(anchors)}")
    static = read_jsonl(args.static)
    static_by_id = {str(row["record_id"]): row for row in static}
    dynamic = read_jsonl(args.dynamic)
    dynamic_by_id = {str(row["record_id"]): row for row in dynamic}

    # Preserve measured dynamic labels.  Assign the 2100 unprofiled rows by static
    # score so every target bucket has enough unique rows for the <=3 repeat cap.
    target_unique = {key: math.ceil(value / MAX_BALANCED_REPEAT) for key, value in quota(TOTAL_DRAWS).items()}
    combined_bucket = {record_id: row["dynamic_bucket"] for record_id, row in dynamic_by_id.items()}
    existing = Counter(combined_bucket.values())
    unprofiled = [row for row in static if str(row["record_id"]) not in combined_bucket]
    unprofiled.sort(key=lambda row: (float(row["static_score"]), stable_key(args.seed, str(row["record_id"]))))
    cursor = 0
    for bucket in ("dynamic_easy", "dynamic_medium", "dynamic_frontier"):
        needed = max(0, target_unique[bucket] - existing[bucket])
        for row in unprofiled[cursor:cursor + needed]:
            combined_bucket[str(row["record_id"])] = bucket
        cursor += needed
    for row in unprofiled[cursor:]:
        combined_bucket[str(row["record_id"])] = "dynamic_hard"
    combined_counts = Counter(combined_bucket.values())
    for bucket, minimum in target_unique.items():
        if combined_counts[bucket] < minimum:
            raise ValueError(f"{bucket} unique pool {combined_counts[bucket]} < required {minimum}")

    anchor_by_bucket: dict[str, list[dict]] = defaultdict(list)
    for row in anchors:
        record_id = str(row.get("record_id") or row.get("id"))
        anchor_by_bucket[combined_bucket[record_id]].append(row)

    verifications = read_jsonl(args.discovery_verifications)
    success = Counter()
    attempts = Counter()
    for row in verifications:
        sid = str(row["statement_id"])
        attempts[sid] += 1
        success[sid] += int(bool(row.get("verified")))
    if set(attempts.values()) != {4}:
        raise ValueError("discovery source does not have exactly four attempts per statement")
    expert_by_statement = {str(row["id"]): row for row in experts}
    frontier_ids = sorted((sid for sid in attempts if success[sid] in (1, 2)), key=lambda sid: stable_key(args.seed, sid))
    frontier_pool = [expert_by_statement[sid] for sid in frontier_ids if sid in expert_by_statement]
    if len(frontier_pool) != len(frontier_ids):
        raise ValueError(f"frontier proof coverage mismatch: ids={len(frontier_ids)} rows={len(frontier_pool)}")

    args.output_root.mkdir(parents=True, exist_ok=True)
    expert_layers = {
        "all_statements": len(attempts),
        "unsolved_statements": sum(success[sid] == 0 for sid in attempts),
        "solved_statements": sum(success[sid] > 0 for sid in attempts),
        "frontier_statements_1_or_2_of_4": len(frontier_ids),
        "medium_statements_3_of_4": sum(success[sid] == 3 for sid in attempts),
        "easy_statements_4_of_4": sum(success[sid] == 4 for sid in attempts),
        "success_count_distribution": dict(Counter(str(success[sid]) for sid in attempts)),
        "frontier_training_proofs": len(frontier_pool),
        "frontier_unique_statements": len({row["id"] for row in frontier_pool}),
        "frontier_tactic_signatures": dict(Counter(tactic_signature(str(row.get("completion") or "")) for row in frontier_pool)),
    }
    with (args.output_root / "expert_frontier_pool.jsonl").open("w", encoding="utf-8") as handle:
        for row in frontier_pool:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    (args.output_root / "expert_difficulty_summary.json").write_text(json.dumps(expert_layers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (args.output_root / "anchor_balanced_bucket_assignment.json").write_text(json.dumps({
        "measured_dynamic_records": len(dynamic_by_id),
        "static_proxy_records": len(combined_bucket) - len(dynamic_by_id),
        "bucket_counts": dict(combined_counts),
        "target_ratios": DIFFICULTY_RATIOS,
        "max_repeat": MAX_BALANCED_REPEAT,
        "proxy_rule": "preserve measured labels; sort unprofiled by static_score ascending and fill minimum easy, medium, frontier capacities; remainder hard",
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    rng = random.Random(args.seed)
    uniform_order = [rng.choice(anchors) for _ in range(TOTAL_DRAWS)]
    balanced_order = []
    for offset, (bucket, count) in enumerate(quota(TOTAL_DRAWS).items()):
        balanced_order.extend(coverage_first(anchor_by_bucket[bucket], count, args.seed + offset, lambda row: str(row.get("record_id") or row.get("id"))))
    balanced_order = [
        row
        for _, row in sorted(
            enumerate(balanced_order),
            key=lambda item: stable_key(
                args.seed + 9000,
                f"{item[1].get('record_id')}:{item[0]}",
            ),
        )
    ]

    mixed_uniform_anchor = uniform_order[:MIXED_ANCHOR_DRAWS]
    mixed_balanced_anchor = []
    for offset, (bucket, count) in enumerate(quota(MIXED_ANCHOR_DRAWS).items()):
        mixed_balanced_anchor.extend(coverage_first(anchor_by_bucket[bucket], count, args.seed + 100 + offset, lambda row: str(row.get("record_id") or row.get("id"))))
    expert_draws = coverage_first(frontier_pool, MIXED_EXPERT_DRAWS, args.seed + 500, lambda row: str(row["id"]))

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
    arm_specs = {
        "A1_unbalanced_anchor_only": [(row, "anchor", combined_bucket[str(row.get("record_id") or row.get("id"))]) for row in uniform_order],
        "A3_unbalanced_anchor_frontier_90_10": [(row, "anchor", combined_bucket[str(row.get("record_id") or row.get("id"))]) for row in mixed_uniform_anchor] + [(row, "frontier_expert", "dynamic_frontier") for row in expert_draws],
        "A4_balanced_anchor_only": [(row, "anchor", combined_bucket[str(row.get("record_id") or row.get("id"))]) for row in balanced_order],
        "A5_balanced_anchor_frontier_90_10": [(row, "anchor", combined_bucket[str(row.get("record_id") or row.get("id"))]) for row in mixed_balanced_anchor] + [(row, "frontier_expert", "dynamic_frontier") for row in expert_draws],
    }
    global_manifest = {"total_draws_per_trained_arm": TOTAL_DRAWS, "optimizer_steps": TOTAL_DRAWS // 16, "effective_batch_size": 16, "arms": {}}
    for arm, spec in arm_specs.items():
        shuffled = sorted(enumerate(spec), key=lambda item: stable_key(args.seed + 700, f"{arm}:{item[0]}"))
        rows = [annotate(row, arm=arm, source=source, difficulty=bucket, draw_index=index) for index, (_, (row, source, bucket)) in enumerate(shuffled)]
        arm_dir = args.output_root / arm / "train"
        arm_dir.mkdir(parents=True, exist_ok=True)
        dataset = arm_dir / "train_dataset.jsonl"
        with dataset.open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        audit = stats(rows, tokenizer)
        audit.update({"dataset": str(dataset), "sha256": sha256(dataset), "all_rows_pantograph_verified": all(row.get("pantograph_verified") is True for row in rows), "initialization_checkpoint": "clean_M0_merged", "adapter_initialization": None})
        (arm_dir / "train_manifest.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        global_manifest["arms"][arm] = audit
    global_manifest["arms"]["A0_clean_M0"] = {"training": "none", "reuse": "clean M0 merged checkpoint"}
    global_manifest["arms"]["A2_current_M1_85_15"] = {"training": "reuse", "reuse": "outputs/expert_iteration_round1/iteration_000/checkpoint", "sampler_audit": "audit/current_recipe_sampling_trace.json"}
    (args.output_root / "ablation_train_manifest.json").write_text(json.dumps(global_manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"expert": expert_layers, "combined_anchor_buckets": dict(combined_counts), "arms": {key: value["source_draw_counts"] for key, value in global_manifest["arms"].items() if "source_draw_counts" in value}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
