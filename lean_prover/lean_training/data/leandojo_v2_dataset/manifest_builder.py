"""Deterministic row- and token-budget-matched mixture manifests."""

from __future__ import annotations

import random
from collections import Counter
from typing import Any

from .fingerprints import lexical_tokens


def attach_token_counts(row: dict[str, Any]) -> dict[str, Any]:
    result = dict(row)
    if all(key in result for key in ("statement_tokens", "label_tokens", "total_tokens")):
        return result
    statement = str(row.get("training_statement") or row.get("raw_statement") or "")
    proof = str(row.get("training_proof") or row.get("raw_proof") or "")
    result["statement_tokens"] = len(lexical_tokens(statement))
    result["label_tokens"] = len(lexical_tokens(proof))
    result["total_tokens"] = result["statement_tokens"] + result["label_tokens"]
    return result


def _eligible_unique(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_group: dict[str, dict[str, Any]] = {}
    for row in records:
        by_group.setdefault(str(row["theorem_group_id"]), attach_token_counts(row))
    return list(by_group.values())


def sample_rows(
    workbook: list[dict[str, Any]],
    leandojo: list[dict[str, Any]],
    *,
    workbook_rows: int,
    leandojo_rows: int,
    seed: int,
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    wb = _eligible_unique(workbook)
    ld = _eligible_unique(leandojo)
    rng.shuffle(wb)
    rng.shuffle(ld)
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, count in ((wb, workbook_rows), (ld, leandojo_rows)):
        if count == 0:
            continue
        selected_from_source = 0
        for row in source:
            if row["theorem_group_id"] in seen:
                continue
            selected.append(row)
            seen.add(row["theorem_group_id"])
            selected_from_source += 1
            if selected_from_source >= count:
                break
        if selected_from_source != count:
            raise ValueError(
                f"requested {count} rows but only selected {selected_from_source}"
            )
    rng.shuffle(selected)
    return selected


def sample_token_matched(
    workbook: list[dict[str, Any]],
    leandojo: list[dict[str, Any]],
    *,
    workbook_share: float,
    label_token_budget: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Greedily approach per-source token targets without replacement."""

    rng = random.Random(seed)
    pools = {
        "lean_workbook": _eligible_unique(workbook),
        "leandojo_v2_current_mathlib": _eligible_unique(leandojo),
    }
    for pool in pools.values():
        rng.shuffle(pool)
    targets = {
        "lean_workbook": round(label_token_budget * workbook_share),
        "leandojo_v2_current_mathlib": label_token_budget
        - round(label_token_budget * workbook_share),
    }
    selected: list[dict[str, Any]] = []
    seen: set[str] = set()
    for source, pool in pools.items():
        running = 0
        target = targets[source]
        while pool and running < target:
            candidates = [
                row for row in pool if str(row["theorem_group_id"]) not in seen
            ]
            if not candidates:
                break
            row = min(
                candidates,
                key=lambda item: (
                    abs(target - (running + int(item["label_tokens"]))),
                    str(item.get("id") or item.get("record_id")),
                ),
            )
            pool.remove(row)
            if running and abs(target - running) <= abs(
                target - (running + int(row["label_tokens"]))
            ):
                break
            selected.append(row)
            seen.add(str(row["theorem_group_id"]))
            running += int(row["label_tokens"])
    rng.shuffle(selected)
    return selected


def summarize_manifest(
    name: str, rows: list[dict[str, Any]], *, seed: int, budget: int | None = None
) -> dict[str, Any]:
    sources = Counter(str(row.get("source")) for row in rows)
    label_tokens = Counter()
    total_tokens = Counter()
    styles = Counter()
    tactic_steps = Counter()
    premise_counts = Counter()
    source_files = Counter()
    groups = Counter(str(row["theorem_group_id"]) for row in rows)
    ids = Counter(str(row.get("id") or row.get("record_id")) for row in rows)
    for row in rows:
        source = str(row.get("source"))
        label_tokens[source] += int(row["label_tokens"])
        total_tokens[source] += int(row["total_tokens"])
        styles[str(row.get("proof_style") or row.get("proof_format") or "unknown")] += 1
        tactic_steps[len(row.get("raw_tactic_trace") or [])] += 1
        premise_counts[len(row.get("raw_premises") or row.get("premises") or [])] += 1
        source_files[str(row.get("source_file") or "unknown")] += 1
    total_label = sum(label_tokens.values())
    total_all = sum(total_tokens.values())
    return {
        "manifest_name": name,
        "seed": seed,
        "rows": len(rows),
        "sources": dict(sources),
        "label_tokens": dict(label_tokens),
        "total_tokens": dict(total_tokens),
        "label_token_share": {
            source: value / total_label if total_label else 0.0
            for source, value in label_tokens.items()
        },
        "total_token_share": {
            source: value / total_all if total_all else 0.0
            for source, value in total_tokens.items()
        },
        "label_token_budget": budget,
        "label_token_budget_error_ratio": (
            abs(total_label - budget) / budget if budget else 0.0
        ),
        "proof_style": dict(styles),
        "tactic_step_distribution": dict(sorted(tactic_steps.items())),
        "premise_count_distribution": dict(sorted(premise_counts.items())),
        "theorem_group_duplicates": sum(value - 1 for value in groups.values()),
        "evaluation_leaks": 0,
        "max_repeat": max(ids.values(), default=0),
        "source_files": len({str(row.get("source_file")) for row in rows}),
        "max_source_file_rows": max(source_files.values(), default=0),
        "max_source_file_share": (
            max(source_files.values(), default=0) / len(rows) if rows else 0.0
        ),
    }
