import os
import unittest
from pathlib import Path

from lean_prover.backends.pantograph_backend import (
    PantographBackend,
    PantographOptions,
)
from lean_prover.backends.proof_backend import TransitionStatus


@unittest.skipUnless(
    os.environ.get("RUN_PANTOGRAPH_TESTS") == "1",
    "set RUN_PANTOGRAPH_TESTS=1 for the real Lean integration test",
)
class PantographIntegrationTests(unittest.TestCase):
    def test_mathlib_branching(self):
        project = Path(
            os.environ.get(
                "LEAN_PROJECT_PATH",
                Path(__file__).parents[1] / "lean_project",
            )
        )
        backend = PantographBackend(
            PantographOptions(project, imports=("Mathlib",), timeout=120)
        )
        try:
            root = backend.start("鈭€ n : Nat, n + 0 = n")
            bad = backend.apply_tactic(root, "not_a_real_tactic")
            left, right = backend.branch(root, ("intro n", "simp"))

            self.assertEqual(bad.status, TransitionStatus.ERROR)
            self.assertEqual(left.status, TransitionStatus.OPEN)
            self.assertEqual(right.status, TransitionStatus.PROVED)
            self.assertNotEqual(
                left.state.state_id, right.state.state_id
            )
            done = backend.apply_tactic(left.state, "simp")
            self.assertEqual(done.status, TransitionStatus.PROVED)
            self.assertEqual(
                backend.tactic_path(done.state), ("intro n", "simp")
            )
        finally:
            backend.close()
