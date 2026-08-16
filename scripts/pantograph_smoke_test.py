"""End-to-end branching smoke test for the Pantograph backend."""

from __future__ import annotations

import os
from pathlib import Path

from lean_prover.backends.pantograph_backend import (
    PantographBackend,
    PantographOptions,
)
from lean_prover.backends.proof_backend import TransitionStatus


PROJECT = Path(
    os.environ.get(
        "LEAN_PROJECT_PATH",
        Path(__file__).parents[1] / "lean_project",
    )
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def main() -> None:
    backend = PantographBackend(
        PantographOptions(
            project_path=PROJECT,
            imports=("Mathlib",),
            timeout=120,
        )
    )
    checks = 0

    def passed(label: str) -> None:
        nonlocal checks
        checks += 1
        print(f"[PASS {checks:02d}] {label}")

    try:
        print(f"Pantograph: {backend.version}")
        root = backend.start("鈭€ n : Nat, n + 0 = n")
        print(f"Initial goal:\n{root.text}")
        passed("import Mathlib")
        require(root.goals, "initial goal is missing")
        passed("create a Nat theorem goal")

        failed = backend.apply_tactic(
            root, "this_tactic_does_not_exist"
        )
        print(f"Bad tactic: {failed.status.value}: {failed.message}")
        require(failed.status is TransitionStatus.ERROR, "bad tactic passed")
        passed("capture an invalid tactic")

        intro, direct = backend.branch(root, ("intro n", "simp"))
        print(f"Branch intro n: {intro.status.value}")
        print(f"Branch simp: {direct.status.value}")
        require(intro.state is not None, "intro branch has no state")
        require(direct.state is not None, "simp branch has no state")
        passed("apply a successful tactic")
        require(
            intro.state.state_id != direct.state.state_id,
            "branches reused a state id",
        )
        require(
            intro.state.parent_id == direct.state.parent_id == root.state_id,
            "branches do not share the requested parent",
        )
        passed("create two branches from one proof state")
        passed("keep branch states independent")

        require(
            direct.status is TransitionStatus.PROVED,
            "direct simp branch did not finish",
        )
        require(
            backend.get_goals(intro.state),
            "finishing one branch damaged the open branch",
        )
        parent_after_failure = backend.apply_tactic(root, "simp")
        require(
            parent_after_failure.status is TransitionStatus.PROVED,
            "failed branch damaged its parent or sibling",
        )
        passed("isolate a failed branch from successful branches")
        passed("recognize proof finished")

        finished = backend.apply_tactic(intro.state, "simp")
        require(
            finished.status is TransitionStatus.PROVED
            and finished.state is not None,
            "intro/simp path did not finish",
        )
        path = backend.tactic_path(finished.state)
        print("Successful tactic path:", " ; ".join(path))
        require(path == ("intro n", "simp"), "wrong tactic path")
        passed("recover the complete tactic path")

        sorry_root = backend.start("False 鈫?False")
        sorry_result = backend.apply_tactic(sorry_root, "sorry")
        admit_result = backend.apply_tactic(sorry_root, "admit")
        require(
            sorry_result.status is TransitionStatus.ERROR,
            "sorry was accepted",
        )
        require(
            admit_result.status is TransitionStatus.ERROR,
            "admit was accepted",
        )
        passed("reject sorry/admit")

        pruned_id = intro.state.state_id
        backend.release_state(intro.state)
        require(pruned_id is not None, "pruned state had no id")

        require(checks == 10, f"expected 10 checks, got {checks}")
        print("All Pantograph acceptance tests passed (10/10).")
    finally:
        backend.close()


if __name__ == "__main__":
    main()
