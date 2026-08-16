"""PyPantograph backend with explicit branching and state lifecycle."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any, Sequence

from .proof_backend import ProofState, TacticTransition, TransitionStatus


class PantographUnavailableError(RuntimeError):
    """Raised when PyPantograph is absent or cannot start."""


@dataclass(frozen=True)
class PantographOptions:
    project_path: str | Path
    imports: tuple[str, ...] = ("Mathlib",)
    timeout: int = 60
    core_options: tuple[str, ...] = ()
    server_options: dict[str, Any] | None = None
    lean_path: str | None = None


class PantographBackend:
    """Interactive Lean backend backed by one Pantograph server.

    Calling :meth:`apply_tactic` repeatedly with the same parent creates
    independent Pantograph states, which is the primitive needed by beam and
    best-first search.
    """

    def __init__(
        self,
        options: PantographOptions,
        *,
        _server: Any = None,
        _api: dict[str, Any] | None = None,
    ) -> None:
        self.options = options
        self._server: Any = _server
        self._api_override = _api
        self._states: dict[int | str, ProofState] = {}
        self._lineage: dict[int | str, ProofState] = {}
        self._released: set[int | str] = set()

    @staticmethod
    def _load_api() -> dict[str, Any]:
        try:
            from pantograph import Server
            from pantograph.message import ServerError, TacticFailure
            from pantograph.server import get_version
        except ImportError as error:
            vendor = Path(__file__).resolve().parents[2] / ".vendor" / "PyPantograph"
            if vendor.is_dir() and str(vendor) not in sys.path:
                sys.path.insert(0, str(vendor))
                try:
                    from pantograph import Server
                    from pantograph.message import ServerError, TacticFailure
                    from pantograph.server import get_version
                except ImportError as vendor_error:
                    raise PantographUnavailableError(
                        "PyPantograph is not installed in this Python environment"
                    ) from vendor_error
            else:
                raise PantographUnavailableError(
                    "PyPantograph is not installed in this Python environment"
                ) from error
        return {
            "Server": Server,
            "get_version": get_version,
            "ServerError": ServerError,
            "TacticFailure": TacticFailure,
        }

    @property
    def version(self) -> str:
        return str(self._api()["get_version"]())

    def _api(self) -> dict[str, Any]:
        return self._api_override or self._load_api()

    def _ensure_server(self) -> Any:
        if self._server is not None:
            return self._server
        api = self._api()
        project = Path(self.options.project_path).expanduser().resolve()
        if not (project / "lean-toolchain").is_file():
            raise FileNotFoundError(
                f"Pantograph project lacks lean-toolchain: {project}"
            )
        if not (
            (project / "lakefile.lean").is_file()
            or (project / "lakefile.toml").is_file()
        ):
            raise FileNotFoundError(
                f"Pantograph project lacks a lakefile: {project}"
            )
        try:
            self._server = api["Server"](
                imports=list(self.options.imports),
                project_path=str(project),
                lean_path=self.options.lean_path,
                options=self.options.server_options or {},
                core_options=list(self.options.core_options),
                timeout=self.options.timeout,
            )
        except Exception as error:
            raise PantographUnavailableError(
                f"failed to start Pantograph for {project}: {error}"
            ) from error
        return self._server

    def ensure_server(self) -> Any:
        """Return a live Pantograph server, starting it if needed."""

        return self._ensure_server()

    @staticmethod
    def _goal_texts(raw_state: Any) -> tuple[str, ...]:
        return tuple(str(goal) for goal in raw_state.goals)

    def _register(
        self,
        raw_state: Any,
        *,
        parent: ProofState | None = None,
        tactic: str | None = None,
    ) -> ProofState:
        state_id = raw_state.state_id
        goals = self._goal_texts(raw_state)
        state = ProofState(
            text="\n\n".join(goals),
            raw=raw_state,
            state_id=state_id,
            goals=goals,
            parent_id=None if parent is None else parent.state_id,
            tactic=tactic,
            depth=0 if parent is None else parent.depth + 1,
            finished=not goals,
        )
        self._states[state_id] = state
        self._lineage[state_id] = state
        return state

    def start(self, target: str | None = None) -> ProofState:
        if target is None or not target.strip():
            raise ValueError("PantographBackend.start requires a Lean target")
        raw_state = self._ensure_server().goal_start(target)
        return self._register(raw_state)

    def get_goals(self, state: ProofState) -> tuple[str, ...]:
        self._assert_live(state)
        return state.goals

    def _assert_live(self, state: ProofState) -> None:
        if state.state_id is None or state.state_id not in self._states:
            raise KeyError(f"unknown Pantograph state: {state.state_id}")
        if state.state_id in self._released:
            raise KeyError(f"released Pantograph state: {state.state_id}")

    def apply_tactic(
        self, state: ProofState, tactic: str
    ) -> TacticTransition:
        self._assert_live(state)
        if state.finished:
            return TacticTransition(
                TransitionStatus.ERROR, message="proof is already finished"
            )
        api = self._api()
        try:
            raw_next = self._ensure_server().goal_tactic(
                state.raw, tactic
            )
        except api["TacticFailure"] as error:
            return TacticTransition(
                TransitionStatus.ERROR, message=str(error)
            )
        except api["ServerError"] as error:
            message = str(error)
            status = (
                TransitionStatus.TIMEOUT
                if "timeout" in message.lower()
                else TransitionStatus.ERROR
            )
            return TacticTransition(status, message=message)
        next_state = self._register(
            raw_next, parent=state, tactic=tactic
        )
        return TacticTransition(
            TransitionStatus.PROVED
            if next_state.finished
            else TransitionStatus.OPEN,
            state=next_state,
        )

    def release_state(self, state: ProofState) -> None:
        state_id = state.state_id
        if (
            state_id is None
            or state_id in self._released
            or self._server is None
        ):
            return
        self._server.run("goal.delete", {"stateIds": [state_id]})
        self._released.add(state_id)
        self._states.pop(state_id, None)

    def tactic_path(self, state: ProofState) -> tuple[str, ...]:
        path: list[str] = []
        current = state
        while current.parent_id is not None:
            if current.tactic is None:
                raise RuntimeError("non-root state is missing its tactic")
            path.append(current.tactic)
            parent = self._lineage.get(current.parent_id)
            if parent is None:
                raise KeyError(
                    f"parent state was released: {current.parent_id}"
                )
            current = parent
        path.reverse()
        return tuple(path)

    def branch(
        self, state: ProofState, tactics: Sequence[str]
    ) -> tuple[TacticTransition, ...]:
        """Apply candidates independently to the same parent state."""

        return tuple(self.apply_tactic(state, tactic) for tactic in tactics)

    def close(self) -> None:
        if self._server is not None:
            self._server.__exit__(None, None, None)
        self._server = None
        self._states.clear()
        self._lineage.clear()
        self._released.clear()
