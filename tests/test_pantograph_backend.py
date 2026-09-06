import tempfile
import unittest
from pathlib import Path

from lean_prover.backends.pantograph_backend import (
    PantographBackend,
    PantographOptions,
)
from lean_prover.backends.proof_backend import TransitionStatus


class FakeTacticFailure(Exception):
    pass


class FakeServerError(Exception):
    pass


class RawGoalState:
    def __init__(self, state_id, goals):
        self.state_id = state_id
        self.goals = goals


class FakePantographServer:
    def __init__(self):
        self.next_id = 1
        self.deleted = []
        self.closed = False

    def goal_start(self, target):
        return RawGoalState(0, [f"鈯?{target}"])

    def goal_tactic(self, state, tactic):
        if tactic in {"bad", "sorry", "admit"}:
            raise FakeTacticFailure(f"rejected tactic: {tactic}")
        state_id = self.next_id
        self.next_id += 1
        if tactic in {"rfl", "exact h"}:
            return RawGoalState(state_id, [])
        return RawGoalState(state_id, [f"{tactic} branch"])

    def run(self, command, payload):
        assert command == "goal.delete"
        self.deleted.extend(payload["stateIds"])
        return {}

    def __exit__(self, exc_type, exc, traceback):
        self.closed = True


class PantographBackendTests(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        project = Path(self.tempdir.name)
        (project / "lean-toolchain").write_text(
            "leanprover/lean4:v4.30.0", encoding="utf-8"
        )
        (project / "lakefile.toml").write_text(
            'name = "Test"', encoding="utf-8"
        )
        self.server = FakePantographServer()
        self.backend = PantographBackend(
            PantographOptions(project),
            _server=self.server,
            _api={
                "ServerError": FakeServerError,
                "TacticFailure": FakeTacticFailure,
                "get_version": lambda: "fake",
            },
        )

    def tearDown(self):
        self.backend.close()
        self.tempdir.cleanup()

    def test_independent_branches_and_path_recovery(self):
        root = self.backend.start("鈭€ n : Nat, n = n")
        left, right = self.backend.branch(
            root, ("intro n", "simp")
        )

        self.assertEqual(left.status, TransitionStatus.OPEN)
        self.assertEqual(right.status, TransitionStatus.OPEN)
        self.assertNotEqual(left.state.state_id, right.state.state_id)
        self.assertEqual(left.state.parent_id, root.state_id)
        self.assertEqual(right.state.parent_id, root.state_id)

        finished = self.backend.apply_tactic(left.state, "rfl")
        self.assertEqual(finished.status, TransitionStatus.PROVED)
        self.assertEqual(
            self.backend.tactic_path(finished.state),
            ("intro n", "rfl"),
        )
        self.assertEqual(
            self.backend.get_goals(right.state), ("simp branch",)
        )

    def test_failed_branch_does_not_damage_parent(self):
        root = self.backend.start("1 = 1")
        failed = self.backend.apply_tactic(root, "bad")
        success = self.backend.apply_tactic(root, "rfl")

        self.assertEqual(failed.status, TransitionStatus.ERROR)
        self.assertEqual(success.status, TransitionStatus.PROVED)

    def test_rejects_sorry_and_admit(self):
        root = self.backend.start("False")
        for tactic in ("sorry", "admit"):
            with self.subTest(tactic=tactic):
                result = self.backend.apply_tactic(root, tactic)
                self.assertEqual(result.status, TransitionStatus.ERROR)

    def test_release_deletes_server_state(self):
        root = self.backend.start("True")
        child = self.backend.apply_tactic(root, "constructor").state
        self.backend.release_state(child)

        self.assertIn(child.state_id, self.server.deleted)
        with self.assertRaises(KeyError):
            self.backend.get_goals(child)

    def test_path_survives_releasing_ancestors(self):
        root = self.backend.start("鈭€ n : Nat, n = n")
        child = self.backend.apply_tactic(root, "intro n").state
        finished = self.backend.apply_tactic(child, "rfl").state

        self.backend.release_state(root)
        self.backend.release_state(child)

        self.assertEqual(
            self.backend.tactic_path(finished),
            ("intro n", "rfl"),
        )

    def test_explicit_lean_path_is_forwarded_to_server(self):
        captured = {}

        def make_server(**kwargs):
            captured.update(kwargs)
            return FakePantographServer()

        backend = PantographBackend(
            PantographOptions(
                Path(self.tempdir.name),
                lean_path="/workspace/lib/lean:/mathlib/lib/lean",
            ),
            _api={
                "Server": make_server,
                "ServerError": FakeServerError,
                "TacticFailure": FakeTacticFailure,
                "get_version": lambda: "fake",
            },
        )
        try:
            backend.ensure_server()
        finally:
            backend.close()

        self.assertEqual(
            captured["lean_path"],
            "/workspace/lib/lean:/mathlib/lib/lean",
        )


if __name__ == "__main__":
    unittest.main()
