"""Small, backend-neutral proof-search framework."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Sequence

from .backends.proof_backend import (
    InteractiveProofBackend,
    ProofState,
    TacticProposer,
    TransitionStatus,
)


@dataclass(frozen=True)
class ProofResult:
    """Result of a proof-search run."""

    success: bool
    tactics: tuple[str, ...]
    explored_nodes: int
    message: str = ""

    def as_lean_proof(self) -> str:
        """Render successful tactics as a Lean ``by`` proof."""

        if not self.success:
            raise ValueError("cannot render an unsuccessful search result")
        body = "\n".join(f"  {tactic}" for tactic in self.tactics)
        return f"by\n{body}"


class StaticTacticProposer:
    """Baseline proposer useful for smoke tests and simple theorems."""

    def __init__(self, tactics: Sequence[str]) -> None:
        self.tactics = tuple(dict.fromkeys(tactics))

    def propose(
        self, state: ProofState, history: Sequence[str]
    ) -> Sequence[str]:
        del state, history
        return self.tactics


class BreadthFirstSearch:
    """Bounded breadth-first tactic search.

    Replace ``StaticTacticProposer`` with an LLM-backed implementation without
    changing the Pantograph integration or final verification code.
    """

    def __init__(self, max_depth: int = 8, max_nodes: int = 200) -> None:
        if max_depth <= 0 or max_nodes <= 0:
            raise ValueError("max_depth and max_nodes must be positive")
        self.max_depth = max_depth
        self.max_nodes = max_nodes

    def run(
        self,
        backend: InteractiveProofBackend,
        proposer: TacticProposer,
        target: str | None = None,
    ) -> ProofResult:
        explored = 0
        seen: set[str] = set()
        try:
            initial_state = backend.start(target)
            queue = deque([(initial_state, tuple())])
            seen.add(initial_state.text)

            while queue and explored < self.max_nodes:
                state, history = queue.popleft()
                if len(history) >= self.max_depth:
                    continue
                candidates = proposer.propose(state, history)
                for tactic in candidates:
                    if explored >= self.max_nodes:
                        break
                    tactic = tactic.strip()
                    if not tactic:
                        continue
                    explored += 1
                    transition = backend.apply_tactic(state, tactic)
                    next_history = history + (tactic,)
                    if transition.status is TransitionStatus.PROVED:
                        tactics = (
                            backend.tactic_path(transition.state)
                            if transition.state is not None
                            else next_history
                        )
                        return ProofResult(True, tactics, explored)
                    if (
                        transition.status is TransitionStatus.OPEN
                        and transition.state is not None
                        and transition.state.text not in seen
                    ):
                        seen.add(transition.state.text)
                        queue.append((transition.state, next_history))
                    elif transition.state is not None:
                        backend.release_state(transition.state)
                backend.release_state(state)
            return ProofResult(
                False,
                (),
                explored,
                "search limits exhausted or no candidates remain",
            )
        finally:
            backend.close()
