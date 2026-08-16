"""Deterministic sampling interfaces over a dynamic difficulty manifest."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .dynamic_difficulty import DynamicDifficultyError, iter_jsonl


PROFILE_CATEGORIES = {
    "ei_foundation": ("easy", "medium"),
    "frontier_learning": ("medium", "hard"),
    "exploration": ("hard", "impossible_capability"),
}


def load_dynamic_manifest(path: str | Path) -> list[dict[str, Any]]:
    return list(iter_jsonl(path))


def _matches(row: Mapping[str, Any], category: str) -> bool:
    normalized = category.strip().lower()
    if normalized in {"impossible_capability", "capability_hard"}:
        return row.get("difficulty") == "impossible" and row.get(
            "zero_success_subtype"
        ) == "capability_hard"
    if normalized in {"impossible_invalid", "data_environment_hard"}:
        return row.get("difficulty") == "impossible" and row.get(
            "zero_success_subtype"
        ) == "data_environment_hard"
    return row.get("difficulty") == normalized


def _largest_remainder(total: int, ratio: Mapping[str, float]) -> dict[str, int]:
    weight = sum(float(value) for value in ratio.values())
    if total < 0 or weight <= 0:
        raise DynamicDifficultyError("total and ratio must be positive")
    raw = {key: total * float(value) / weight for key, value in ratio.items()}
    allocation = {key: math.floor(value) for key, value in raw.items()}
    remaining = total - sum(allocation.values())
    order = sorted(raw, key=lambda key: (-(raw[key] - allocation[key]), key))
    for key in order[:remaining]:
        allocation[key] += 1
    return allocation


def sample_by_difficulty(
    manifest: str | Path | Sequence[Mapping[str, Any]],
    *,
    categories: Sequence[str],
    ratio: Mapping[str, float] | None = None,
    total: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    if not categories:
        raise DynamicDifficultyError("categories cannot be empty")
    rows = (
        load_dynamic_manifest(manifest)
        if isinstance(manifest, (str, Path))
        else [dict(row) for row in manifest]
    )
    normalized = [value.strip().lower() for value in categories]
    if len(normalized) != len(set(normalized)):
        raise DynamicDifficultyError("categories must be unique")
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        matched = [category for category in normalized if _matches(row, category)]
        if len(matched) > 1:
            raise DynamicDifficultyError(
                f"overlapping category aliases for theorem {row.get('theorem_id')}"
            )
        if matched:
            buckets[matched[0]].append(row)
    rng = random.Random(seed)
    for category in normalized:
        buckets[category].sort(key=lambda row: str(row["theorem_id"]))
        rng.shuffle(buckets[category])

    available = sum(len(buckets[category]) for category in normalized)
    requested = available if total is None else total
    if requested < 0 or requested > available:
        raise DynamicDifficultyError(
            f"requested {requested} rows but only {available} are available"
        )
    if ratio is None:
        combined = [row for category in normalized for row in buckets[category]]
        rng.shuffle(combined)
        selected = combined[:requested]
    else:
        if set(ratio) != set(normalized):
            raise DynamicDifficultyError("ratio keys must exactly match categories")
        allocation = _largest_remainder(requested, ratio)
        selected = []
        deficits = 0
        for category in normalized:
            take = min(allocation[category], len(buckets[category]))
            selected.extend(buckets[category][:take])
            deficits += allocation[category] - take
            buckets[category] = buckets[category][take:]
        if deficits:
            remainder = [row for category in normalized for row in buckets[category]]
            rng.shuffle(remainder)
            if len(remainder) < deficits:
                raise DynamicDifficultyError("ratio allocation cannot be satisfied")
            selected.extend(remainder[:deficits])
        rng.shuffle(selected)
    theorem_ids = [str(row["theorem_id"]) for row in selected]
    if len(theorem_ids) != len(set(theorem_ids)):
        raise DynamicDifficultyError("sampler produced duplicate theorem IDs")
    return selected


def sample_profile(
    manifest: str | Path | Sequence[Mapping[str, Any]],
    *,
    profile: str,
    total: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    if profile not in PROFILE_CATEGORIES:
        raise DynamicDifficultyError(f"unknown sampling profile: {profile}")
    return sample_by_difficulty(
        manifest,
        categories=PROFILE_CATEGORIES[profile],
        total=total,
        seed=seed,
    )

