"""Yield-aware scheduler interface for future EI discovery rounds."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from .dynamic_difficulty import DynamicDifficultyError


@dataclass
class DiscoveryScheduler:
    target_effective_by_source: dict[str, int]
    effective_difficulties: set[str] = field(
        default_factory=lambda: {"easy", "medium", "hard"}
    )
    collected_theorem_ids: set[str] = field(default_factory=set)
    effective_collected_by_source: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.target_effective_by_source or any(
            int(value) < 0 for value in self.target_effective_by_source.values()
        ):
            raise DynamicDifficultyError("scheduler targets must be non-negative")
        for source in self.target_effective_by_source:
            self.effective_collected_by_source.setdefault(source, 0)

    def ingest(self, records: Sequence[Mapping[str, Any]]) -> dict[str, int]:
        added: dict[str, int] = {source: 0 for source in self.target_effective_by_source}
        for row in records:
            theorem_id = str(row.get("theorem_id") or "")
            source = str(row.get("source") or "")
            if not theorem_id or theorem_id in self.collected_theorem_ids:
                continue
            self.collected_theorem_ids.add(theorem_id)
            if source in added and str(row.get("difficulty")) in self.effective_difficulties:
                self.effective_collected_by_source[source] += 1
                added[source] += 1
        return added

    def remaining_targets(self) -> dict[str, int]:
        return {
            source: max(
                0,
                int(target) - int(self.effective_collected_by_source.get(source, 0)),
            )
            for source, target in self.target_effective_by_source.items()
        }

    @property
    def complete(self) -> bool:
        return all(value == 0 for value in self.remaining_targets().values())

    def plan_next_batch(
        self,
        expected_effective_yield: Mapping[str, float],
        *,
        min_batch: int = 1,
        max_batch: int = 500,
    ) -> dict[str, int]:
        if min_batch <= 0 or max_batch < min_batch:
            raise DynamicDifficultyError("invalid scheduler batch bounds")
        plan: dict[str, int] = {}
        for source, remaining in self.remaining_targets().items():
            if remaining == 0:
                plan[source] = 0
                continue
            yield_rate = float(expected_effective_yield.get(source, 0.0))
            if not 0 < yield_rate <= 1:
                raise DynamicDifficultyError(
                    f"expected yield for {source} must be in (0, 1]"
                )
            required = math.ceil(remaining / yield_rate)
            plan[source] = min(max_batch, max(min_batch, required))
        return plan

    def state_dict(self) -> dict[str, Any]:
        return {
            "target_effective_by_source": dict(self.target_effective_by_source),
            "effective_difficulties": sorted(self.effective_difficulties),
            "effective_collected_by_source": dict(self.effective_collected_by_source),
            "remaining_targets": self.remaining_targets(),
            "complete": self.complete,
            "unique_theorems_seen": len(self.collected_theorem_ids),
        }

