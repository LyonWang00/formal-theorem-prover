from __future__ import annotations

import unittest
from types import SimpleNamespace

from scripts.build_numinamath_full_review_checkpoint import (
    apply_pending_only_override,
    require_full_pantograph_coverage,
)


def review(record_id: str, decision: str) -> dict[str, object]:
    return {
        "record_id": record_id,
        "quality_decision": decision,
    }


class FullCheckpointGateTests(unittest.TestCase):
    def test_override_can_only_resolve_current_pending_row(self) -> None:
        current = {"r1": review("r1", "pending")}
        applied = apply_pending_only_override(
            current, {"r1": review("r1", "pass")}, role="round58"
        )
        self.assertEqual(applied, {"r1"})
        self.assertEqual(current["r1"]["quality_decision"], "pass")
        with self.assertRaisesRegex(ValueError, "currently pending"):
            apply_pending_only_override(
                current, {"r1": review("r1", "reject")}, role="round59"
            )

    def test_full_pantograph_coverage_is_hard_gate(self) -> None:
        receipts = {"r1": [SimpleNamespace(backend="pantograph")]}
        with self.assertRaisesRegex(RuntimeError, "coverage is 1/2"):
            require_full_pantograph_coverage({"r1", "r2"}, receipts)
        receipts["r2"] = [SimpleNamespace(backend="local-pantograph-worker")]
        covered = require_full_pantograph_coverage({"r1", "r2"}, receipts)
        self.assertEqual(covered, {"r1", "r2"})

    def test_non_pantograph_receipt_does_not_satisfy_gate(self) -> None:
        receipts = {"r1": [SimpleNamespace(backend="lean-direct")]}
        with self.assertRaisesRegex(RuntimeError, "missing=1"):
            require_full_pantograph_coverage({"r1"}, receipts)


if __name__ == "__main__":
    unittest.main()
