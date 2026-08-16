import unittest

from lean_prover.backends.proof_backend import (
    ProofState,
    TacticTransition,
    TransitionStatus,
)
from lean_prover.proof_search import BreadthFirstSearch, StaticTacticProposer


class FakeBackend:
    def __init__(self):
        self.closed = False
        self.states = {}

    def start(self, target=None):
        del target
        state = ProofState("goal", 0, state_id=0, goals=("goal",))
        self.states[0] = state
        return state

    def get_goals(self, state):
        return state.goals

    def apply_tactic(self, state, tactic):
        if state.raw == 0 and tactic == "intro":
            return TacticTransition(
                TransitionStatus.OPEN,
                ProofState(
                    "introduced",
                    1,
                    state_id=1,
                    goals=("introduced",),
                    parent_id=0,
                    tactic="intro",
                    depth=1,
                ),
            )
        if state.raw == 1 and tactic == "assumption":
            return TacticTransition(
                TransitionStatus.PROVED,
                ProofState(
                    "",
                    2,
                    state_id=2,
                    parent_id=1,
                    tactic="assumption",
                    depth=2,
                    finished=True,
                ),
            )
        return TacticTransition(TransitionStatus.ERROR, message="failed")

    def release_state(self, state):
        del state

    def tactic_path(self, state):
        if state.state_id == 2:
            return ("intro", "assumption")
        return ()

    def close(self):
        self.closed = True


class BreadthFirstSearchTests(unittest.TestCase):
    def test_finds_and_renders_proof(self):
        backend = FakeBackend()
        result = BreadthFirstSearch(max_depth=3, max_nodes=10).run(
            backend, StaticTacticProposer(["intro", "assumption"])
        )

        self.assertTrue(result.success)
        self.assertEqual(result.tactics, ("intro", "assumption"))
        self.assertEqual(
            result.as_lean_proof(), "by\n  intro\n  assumption"
        )
        self.assertTrue(backend.closed)

    def test_closes_backend_when_search_fails(self):
        backend = FakeBackend()
        result = BreadthFirstSearch(max_depth=1, max_nodes=1).run(
            backend, StaticTacticProposer(["bad"])
        )

        self.assertFalse(result.success)
        self.assertTrue(backend.closed)


if __name__ == "__main__":
    unittest.main()
