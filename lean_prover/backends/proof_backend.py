"""Backend-neutral types used by interactive proof search."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol, Sequence, runtime_checkable


class TransitionStatus(str, Enum):
    """Outcome of applying one tactic."""

    OPEN = "open"
    PROVED = "proved"
    ERROR = "error"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ProofState:
    """Backend-neutral search node.

    ``raw`` is private to the backend. Search code should use the stable
    metadata fields and never inspect it.
    """

    text: str
    raw: Any
    state_id: int | str | None = None
    goals: tuple[str, ...] = ()
    parent_id: int | str | None = None
    tactic: str | None = None
    depth: int = 0
    finished: bool = False


@dataclass(frozen=True)
class TacticTransition:
    """Result returned by an interactive backend."""

    status: TransitionStatus
    state: ProofState | None = None
    message: str = ""


@runtime_checkable
class InteractiveProofBackend(Protocol):
    """Uniform API for Pantograph and test backends."""

    def start(self, target: str | None = None) -> ProofState:
        """Start a session and return the initial proof state."""

    def get_goals(self, state: ProofState) -> tuple[str, ...]:
        """Return display strings for all open goals."""

    def apply_tactic(
        self, state: ProofState, tactic: str
    ) -> TacticTransition:
        """Apply one tactic to ``state``."""

    def release_state(self, state: ProofState) -> None:
        """Delete a state that the search will no longer revisit."""

    def tactic_path(self, state: ProofState) -> tuple[str, ...]:
        """Recover the tactic sequence from the root to ``state``."""

    def close(self) -> None:
        """Release the underlying Lean process and temporary resources."""


class TacticProposer(Protocol):
    """A model or heuristic that proposes tactics for one proof state."""

    def propose(
        self, state: ProofState, history: Sequence[str]
    ) -> Sequence[str]:
        """Return candidate tactics, best candidate first."""
