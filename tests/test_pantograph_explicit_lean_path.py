from __future__ import annotations

import os
import json
import tempfile
import unittest
from pathlib import Path

from lean_prover.lean_training.verification.pantograph import (
    _explicit_workspace_lean_path,
)


class ExplicitWorkspaceLeanPathTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.project = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_requires_mathlib_umbrella(self) -> None:
        self.assertIsNone(_explicit_workspace_lean_path(self.project))

    def test_refuses_incomplete_local_mathlib(self) -> None:
        (self.project / ".lake" / "packages" / "mathlib").mkdir(parents=True)

        with self.assertRaisesRegex(
            FileNotFoundError, "refusing destructive Lake fallback"
        ):
            _explicit_workspace_lean_path(self.project)

    def test_refuses_declared_but_missing_mathlib_package(self) -> None:
        (self.project / "lake-manifest.json").write_text(
            json.dumps({"packages": [{"name": "mathlib"}]}),
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            FileNotFoundError, "package directory is missing"
        ):
            _explicit_workspace_lean_path(self.project)

    def test_refuses_malformed_manifest(self) -> None:
        (self.project / "lake-manifest.json").write_text(
            "{not-json",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(
            RuntimeError, "cannot validate local Mathlib"
        ):
            _explicit_workspace_lean_path(self.project)

    def test_uses_existing_local_artifacts(self) -> None:
        mathlib = (
            self.project
            / ".lake"
            / "packages"
            / "mathlib"
            / ".lake"
            / "build"
            / "lib"
            / "lean"
        )
        dependency = (
            self.project
            / ".lake"
            / "packages"
            / "batteries"
            / ".lake"
            / "build"
            / "lib"
            / "lean"
        )
        root_lib = self.project / ".lake" / "build" / "lib" / "lean"
        mathlib.mkdir(parents=True)
        dependency.mkdir(parents=True)
        root_lib.mkdir(parents=True)
        (mathlib / "Mathlib.olean").write_bytes(b"fixture")

        result = _explicit_workspace_lean_path(self.project)

        self.assertIsNotNone(result)
        paths = result.split(os.pathsep)
        self.assertEqual(paths[0], str(root_lib.resolve()))
        self.assertEqual(paths[1], str(mathlib.resolve()))
        self.assertIn(str(dependency.resolve()), paths)


if __name__ == "__main__":
    unittest.main()
