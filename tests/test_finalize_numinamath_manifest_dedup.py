from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.finalize_numinamath_expand_verification import (
    load_candidate_manifests,
    sha256_text,
)


def candidate(*, proof: str = "by\n  rfl") -> dict[str, object]:
    source_body = "theorem dedup_test (x : Nat) : x = x := by"
    return {
        "record_id": "dedup-record",
        "candidate_hash": "1" * 64,
        "source_body": source_body,
        "source_body_sha256": sha256_text(source_body),
        "original_record_hash": "2" * 64,
        "selected_source_sha256": "3" * 64,
        "imports": ["Mathlib"],
        "variants": [{"strategy": "test", "proof": proof}],
    }


class CandidateManifestDedupTest(unittest.TestCase):
    def write(self, directory: Path, name: str, row: dict[str, object]) -> Path:
        path = directory / name
        path.write_text(json.dumps(row) + "\n", encoding="utf-8")
        return path

    def test_identical_duplicate_is_deduplicated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            row = candidate()
            paths = [self.write(root, "a.jsonl", row), self.write(root, "b.jsonl", row)]
            indexed, stats = load_candidate_manifests(paths)
            self.assertEqual(len(indexed), 1)
            self.assertEqual(stats["identical_duplicate_rows_deduplicated"], 1)
            self.assertEqual(stats["variants"], 1)

    def test_conflicting_duplicate_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            paths = [
                self.write(root, "a.jsonl", candidate()),
                self.write(root, "b.jsonl", candidate(proof="by\n  exact rfl")),
            ]
            with self.assertRaisesRegex(ValueError, "conflicting candidate manifest index"):
                load_candidate_manifests(paths)


if __name__ == "__main__":
    unittest.main()
