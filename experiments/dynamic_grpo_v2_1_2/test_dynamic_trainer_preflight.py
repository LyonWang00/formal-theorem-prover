from __future__ import annotations

import importlib
import json
import sys
import types

import pytest

from dynamic_cycle_v2 import CycleIdentity, build_cycle_schedule, write_cycle_artifacts


def load_validation_function():
    stub = types.ModuleType("grpo_remote")
    stub.GRPOTrainConfig = object
    sys.modules["grpo_remote"] = stub
    module = importlib.import_module("grpo_adaptive_remote")
    return module.validate_dynamic_cycle_contract


def test_trainer_preflight_binds_schedule_cycle_and_policy(tmp_path) -> None:
    identity = CycleIdentity("cycle-0042", "a" * 64, "b" * 64)
    source = [{"id": f"p{i:03d}", "repeat": 9} for i in range(512)]
    receipts = []
    for i in range(512):
        for attempt in range(8):
            receipts.append(
                {
                    "problem_id": f"p{i:03d}",
                    "attempt_index": attempt,
                    "success": attempt == 0,
                    "rollout_cycle_id": identity.cycle_id,
                    "old_policy_sha256": identity.old_policy_sha256,
                    "selection_manifest_sha256": identity.selection_manifest_sha256,
                    "reward_source": "screen_selection_only",
                }
            )
    schedule, summary = build_cycle_schedule(
        selected_source_rows=source,
        receipts=receipts,
        identity=identity,
    )
    artifact_dir = tmp_path / "schedule"
    write_cycle_artifacts(artifact_dir, schedule, summary)
    validate = load_validation_function()
    manifest = validate(
        manifest_path=artifact_dir / "CYCLE_SCHEDULE_FROZEN.json",
        train_path=artifact_dir / "train_schedule.jsonl",
        expected_cycle_id=identity.cycle_id,
        expected_old_policy_sha256=identity.old_policy_sha256,
    )
    assert manifest["screen_receipts_enter_reward"] is False

    rows = [json.loads(line) for line in (artifact_dir / "train_schedule.jsonl").read_text().splitlines()]
    rows[0]["dynamic_cycle"]["cycle_id"] = "cycle-stale"
    corrupted = tmp_path / "corrupted.jsonl"
    corrupted.write_text("".join(json.dumps(row) + "\n" for row in rows))
    bad_manifest = dict(manifest)
    import hashlib

    bad_manifest["train_schedule_sha256"] = hashlib.sha256(corrupted.read_bytes()).hexdigest()
    bad_manifest_path = tmp_path / "bad-manifest.json"
    bad_manifest_path.write_text(json.dumps(bad_manifest))
    with pytest.raises(ValueError, match="row cycle ID mismatch"):
        validate(
            manifest_path=bad_manifest_path,
            train_path=corrupted,
            expected_cycle_id=identity.cycle_id,
            expected_old_policy_sha256=identity.old_policy_sha256,
        )

