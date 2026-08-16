from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from lean_prover.lean_training.expert_iteration import utils


class EnvironmentIdentityReadOnlyTests(unittest.TestCase):
    def test_does_not_invoke_lake_for_version_discovery(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            project = Path(directory)
            calls: list[list[str]] = []

            def fake_command_output(command, *, cwd=None):
                calls.append(command)
                return "Lean (version 4.29.1)"

            with (
                patch.object(utils, "command_output", side_effect=fake_command_output),
                patch.object(utils, "_mathlib_commit", return_value="abc123"),
            ):
                identity = utils.environment_identity(project, ("Mathlib",))

        self.assertEqual(calls, [["lean", "--version"]])
        self.assertEqual(identity["lean_version"], "Lean (version 4.29.1)")
        self.assertEqual(identity["mathlib_commit"], "abc123")
        self.assertNotIn("lake", " ".join(calls[0]))


if __name__ == "__main__":
    unittest.main()
