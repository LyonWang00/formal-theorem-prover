"""Create deterministic, difficulty-stratified gate datasets for SFT ablations."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Iterable


ANCHOR_QUOTAS = {"hard": 40, "frontier": 60, "medium": 30, "easy": 20}
DISCOVERY_QUOTAS = {0: 50, 1: 40, 2: 30, 3: 20, 4: 10}


def read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(rows: Iterable[dict[str, object]], target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    return hashlib.sha256(target.read_bytes()).hexdigest()


def anchor_bucket(success_count: int) -> str:
    if success_count == 0:
        return "hard"
    if success_count in (1, 2):
        return "frontier"
    if success_count == 3:
        return "medium"
    if success_count == 4:
        return "easy"
    raise ValueError(f"invalid success_count@4: {success_count}")


def select_anchor_gate(
    profile_rows: list[dict[str, object]],
    difficulty_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    difficulty_by_record = {str(row["record_id"]): row for row in difficulty_rows}
    bucketed_rows: defaultdict[str, list[dict[str, object]]] = defaultdict(list)
    for row in profile_rows:
        record_id = str(row.get("id") or row.get("record_id"))
        difficulty = difficulty_by_record.get(record_id)
        if difficulty is None:
            raise ValueError(f"missing dynamic difficulty for anchor record {record_id}")
        bucketed_rows[anchor_bucket(int(difficulty["m0_success_count_at_4"]))].append(row)

    effective_quotas = {
        bucket: min(quota, len(bucketed_rows[bucket])) for bucket, quota in ANCHOR_QUOTAS.items()
    }
    shortfall = sum(ANCHOR_QUOTAS.values()) - sum(effective_quotas.values())
    redistributed: Counter[str] = Counter()
    initial_shortfall = shortfall
    for index, bucket in enumerate(("hard", "frontier")):
        capacity = len(bucketed_rows[bucket]) - effective_quotas[bucket]
        desired = initial_shortfall // 2 + int(index < initial_shortfall % 2)
        addition = min(desired, capacity)
        effective_quotas[bucket] += addition
        redistributed[bucket] += addition
        shortfall -= addition
    for bucket in ("hard", "frontier", "medium", "easy"):
        if shortfall == 0:
            break
        capacity = len(bucketed_rows[bucket]) - effective_quotas[bucket]
        addition = min(shortfall, capacity)
        effective_quotas[bucket] += addition
        redistributed[bucket] += addition
        shortfall -= addition
    if shortfall:
        raise ValueError(f"anchor profile has only {len(profile_rows)} rows; cannot build 150-row gate")

    selected: list[dict[str, object]] = []
    counts: Counter[str] = Counter()
    for row in profile_rows:
        record_id = str(row.get("id") or row.get("record_id"))
        difficulty = difficulty_by_record[record_id]
        success_count = int(difficulty["m0_success_count_at_4"])
        bucket = anchor_bucket(success_count)
        if counts[bucket] >= effective_quotas[bucket]:
            continue
        selected.append(row)
        counts[bucket] += 1
    if dict(counts) != effective_quotas:
        raise ValueError(f"anchor gate quota shortfall: actual={dict(counts)}, target={effective_quotas}")
    return selected, {
        "selection": "profile_order_within_dynamic_difficulty_bucket",
        "requested_quota": ANCHOR_QUOTAS,
        "available": {bucket: len(bucketed_rows[bucket]) for bucket in ANCHOR_QUOTAS},
        "effective_quota": effective_quotas,
        "actual": dict(counts),
        "requested_shortfall": {
            bucket: max(0, ANCHOR_QUOTAS[bucket] - len(bucketed_rows[bucket]))
            for bucket in ANCHOR_QUOTAS
        },
        "redistributed_to": dict(redistributed),
    }


def select_discovery_gate(
    discovery_rows: list[dict[str, object]],
    generation_rows: list[dict[str, object]],
    verification_rows: list[dict[str, object]],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    successes: defaultdict[str, int] = defaultdict(int)
    attempts: Counter[str] = Counter()
    for row in verification_rows:
        statement_id = str(row["statement_id"])
        attempts[statement_id] += 1
        successes[statement_id] += int(bool(row.get("verified")))

    discovery_by_prompt = {str(row["prompt"]): row for row in discovery_rows}
    first_generation_by_id: dict[str, dict[str, object]] = {}
    for row in generation_rows:
        first_generation_by_id.setdefault(str(row["statement_id"]), row)
    evaluation_order = list(first_generation_by_id)
    if len(evaluation_order) != 500:
        raise ValueError(f"expected 500 fixed discovery statements, found {len(evaluation_order)}")

    selected: list[dict[str, object]] = []
    counts: Counter[int] = Counter()
    for statement_id in evaluation_order:
        prompt = str(first_generation_by_id[statement_id]["prompt"])
        row = discovery_by_prompt.get(prompt)
        if row is None:
            raise ValueError(f"fixed discovery prompt missing from source dataset: {statement_id}")
        if attempts[statement_id] != 4:
            raise ValueError(f"expected four M0 attempts for discovery statement {statement_id}")
        success_count = successes[statement_id]
        if success_count not in DISCOVERY_QUOTAS:
            raise ValueError(f"invalid discovery success_count@4: {success_count}")
        if counts[success_count] >= DISCOVERY_QUOTAS[success_count]:
            continue
        selected.append(row)
        counts[success_count] += 1
    if dict(counts) != DISCOVERY_QUOTAS:
        raise ValueError(f"discovery gate quota shortfall: actual={dict(counts)}, target={DISCOVERY_QUOTAS}")
    return selected, {
        "selection": "original_discovery_order_within_m0_success_count_bucket",
        "quota": {str(key): value for key, value in DISCOVERY_QUOTAS.items()},
        "actual": {str(key): counts[key] for key in DISCOVERY_QUOTAS},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--anchor-profile", required=True, type=Path)
    parser.add_argument("--anchor-difficulty", required=True, type=Path)
    parser.add_argument("--discovery", required=True, type=Path)
    parser.add_argument("--discovery-generations", required=True, type=Path)
    parser.add_argument("--discovery-verifications", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    anchor_rows, anchor_metadata = select_anchor_gate(
        read_jsonl(args.anchor_profile), read_jsonl(args.anchor_difficulty)
    )
    discovery_rows, discovery_metadata = select_discovery_gate(
        read_jsonl(args.discovery),
        read_jsonl(args.discovery_generations),
        read_jsonl(args.discovery_verifications),
    )
    anchor_path = args.output_dir / "anchor_gate_150.jsonl"
    discovery_path = args.output_dir / "discovery_gate_150.jsonl"
    anchor_hash = write_jsonl(anchor_rows, anchor_path)
    discovery_hash = write_jsonl(discovery_rows, discovery_path)
    manifest = {
        "anchor_gate": {
            "source": str(args.anchor_profile),
            "difficulty_source": str(args.anchor_difficulty),
            "path": str(anchor_path),
            "rows": len(anchor_rows),
            "sha256": anchor_hash,
            **anchor_metadata,
        },
        "discovery_gate": {
            "source": str(args.discovery),
            "m0_generation_source": str(args.discovery_generations),
            "m0_verification_source": str(args.discovery_verifications),
            "path": str(discovery_path),
            "rows": len(discovery_rows),
            "sha256": discovery_hash,
            **discovery_metadata,
        },
        "anchor_seed": 42001,
        "discovery_seed": 4401,
        "samples_per_statement": 4,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "gate_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
