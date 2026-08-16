"""Deterministic discovery-pool selection and bucket scheduling."""

from __future__ import annotations

import random
from collections import Counter

from .config import CategorySamplingConfig, DiscoveryConfig
from .schemas import DiscoveryBucket, DiscoveryStatementState, StatementRecord


def initialize_statement_states(
    statements: list[StatementRecord],
    existing: dict[str, DiscoveryStatementState] | None = None,
) -> dict[str, DiscoveryStatementState]:
    states = dict(existing or {})
    for statement in statements:
        states.setdefault(
            statement.statement_id,
            DiscoveryStatementState(statement_id=statement.statement_id),
        )
    return states


def select_discovery_pool(
    statements: list[StatementRecord],
    states: dict[str, DiscoveryStatementState],
    config: DiscoveryConfig,
    *,
    iteration: int,
    seed: int,
    category_sampling: CategorySamplingConfig | None = None,
) -> list[StatementRecord]:
    """Select new/frontier/unsolved/audit rows without duplicating statements."""

    rng = random.Random(seed + iteration)
    by_bucket: dict[str, list[StatementRecord]] = {
        "new": [],
        "frontier": [],
        "unsolved": [],
        "audit": [],
    }
    for statement in statements:
        bucket = states[statement.statement_id].current_bucket
        if bucket is DiscoveryBucket.NEW:
            by_bucket["new"].append(statement)
        elif bucket is DiscoveryBucket.FRONTIER:
            by_bucket["frontier"].append(statement)
        elif bucket is DiscoveryBucket.UNSOLVED:
            by_bucket["unsolved"].append(statement)
        elif bucket is DiscoveryBucket.SOLVED_EASY:
            by_bucket["audit"].append(statement)
    for values in by_bucket.values():
        _order_by_category(values, rng, category_sampling)
    requested_counts = config.iteration_bucket_counts.get(iteration)
    total = min(
        sum(requested_counts.values()) if requested_counts else config.statements_per_iteration,
        sum(map(len, by_bucket.values())),
    )
    if iteration == 0 and by_bucket["new"]:
        requested_new = requested_counts["new"] if requested_counts else total
        return by_bucket["new"][: min(total, requested_new)]
    selected: list[StatementRecord] = []
    selected_ids: set[str] = set()
    for bucket, ratio in config.pool_mix.items():
        count = requested_counts[bucket] if requested_counts else round(total * ratio)
        for statement in by_bucket[bucket][:count]:
            selected.append(statement)
            selected_ids.add(statement.statement_id)
    if len(selected) < total:
        remainder = [
            statement
            for bucket in ("frontier", "new", "unsolved", "audit")
            for statement in by_bucket[bucket]
            if statement.statement_id not in selected_ids
        ]
        selected.extend(remainder[: total - len(selected)])
    return selected


def _order_by_category(
    values: list[StatementRecord],
    rng: random.Random,
    config: CategorySamplingConfig | None,
) -> None:
    if not config or not config.enabled:
        rng.shuffle(values)
        return
    counts = Counter(statement.category or "unknown" for statement in values)
    total = max(1, len(values))
    random_keys: dict[str, float] = {}
    for statement in values:
        factor = min(
            config.max_oversample_factor,
            (total / counts[statement.category or "unknown"]) ** 0.5,
        )
        random_keys[statement.statement_id] = rng.random() ** (1.0 / factor)
    values.sort(key=lambda statement: random_keys[statement.statement_id], reverse=True)


def samples_for_statement(
    state: DiscoveryStatementState,
    config: DiscoveryConfig,
) -> int:
    if state.current_bucket is DiscoveryBucket.SOLVED_EASY:
        key = "audit"
    else:
        key = state.current_bucket.value
    return config.sampling_budget.get(key, config.generation.samples_per_statement)
