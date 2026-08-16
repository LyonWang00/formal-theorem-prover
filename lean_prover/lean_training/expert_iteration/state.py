"""Persistent iteration and discovery-statement state."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .schemas import DiscoveryStatementState, IterationState, utc_now
from .utils import read_jsonl, write_json_atomic, write_jsonl_atomic


class StateStore:
    def __init__(self, run_dir: str | Path) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.global_path = self.run_dir / "global_state.json"
        self.statement_state_path = self.run_dir / "discovery_statement_states.jsonl"

    def iteration_dir(self, iteration: int) -> Path:
        return self.run_dir / f"iteration_{iteration:03d}"

    def load_iteration(self, iteration: int) -> IterationState | None:
        path = self.iteration_dir(iteration) / "state.json"
        if not path.exists():
            return None
        return IterationState.model_validate(json.loads(path.read_text(encoding="utf-8-sig")))

    def save_iteration(self, state: IterationState) -> None:
        state.updated_at = utc_now()
        write_json_atomic(
            self.iteration_dir(state.iteration) / "state.json",
            state.model_dump(mode="json"),
        )

    def load_statement_states(self) -> dict[str, DiscoveryStatementState]:
        return {
            row["statement_id"]: DiscoveryStatementState.model_validate(row)
            for row in read_jsonl(self.statement_state_path)
        }

    def save_statement_states(self, states: dict[str, DiscoveryStatementState]) -> None:
        write_jsonl_atomic(
            self.statement_state_path,
            (states[key].model_dump(mode="json") for key in sorted(states)),
        )

    def load_global(self) -> dict[str, Any]:
        if not self.global_path.exists():
            return {}
        return json.loads(self.global_path.read_text(encoding="utf-8-sig"))

    def save_global(self, value: dict[str, Any]) -> None:
        write_json_atomic(self.global_path, value)
