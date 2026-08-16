import os
import unittest
from pathlib import Path
from unittest.mock import patch

from lean_prover.backends import BackendConfig, create_backend
from lean_prover.backends.pantograph_backend import PantographBackend


class BackendFactoryTests(unittest.TestCase):
    def test_default_project_path_is_repository_relative(self):
        with patch.dict(os.environ, {}, clear=True):
            config = BackendConfig.from_environment()
        project = Path(config.project_path)
        self.assertEqual(project.name, "lean_project")
        self.assertTrue(project.is_absolute())

    def test_pantograph_is_default_backend(self):
        backend = create_backend(
            BackendConfig(project_path="/tmp/lean-project")
        )
        self.assertIsInstance(backend, PantographBackend)

    def test_unknown_backend_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown Lean backend"):
            create_backend(
                BackendConfig(
                    name="unknown",
                    project_path="/tmp/lean-project",
                )
            )

if __name__ == "__main__":
    unittest.main()
