from __future__ import annotations

import json
from pathlib import Path

from scripts.prepare_stage2_sft_phase0 import (
    classify,
    duplicate_audit,
    eos_static_audit,
    overlap,
)


def _row(index: int) -> dict[str, object]:
    return {
        "record_id": f"row_{index}",
        "theorem_group_id": f"group_{index}",
        "lean_statement": f"theorem t{index} : True",
        "prompt": f"### Lean statement\ntheorem t{index} : True\n### Lean proof\n",
        "proof": "by trivial",
        "label_tokens": 2,
        "total_tokens": 20,
        "pantograph_verified": True,
        "truncated": False,
        "zero_label": False,
        "sampling_source": "WB",
    }


def test_four_candidate_classification_contract() -> None:
    rows = [_row(index) for index in range(3)]
    generations = [
        {"statement_id": f"generated_{index}", "prompt": row["prompt"]}
        for index, row in enumerate(rows)
    ]
    success_counts = [3, 1, 0]
    attempts = []
    for index, count in enumerate(success_counts):
        attempts.extend(
            {
                "problem_id": f"generated_{index}",
                "attempt_index": attempt,
                "success": attempt < count,
                "timed_out": False,
            }
            for attempt in range(4)
        )
    manifest, _ = classify(rows, attempts, generations, "frozen-contract")
    assert [row["classification"] for row in manifest] == [
        "stable",
        "frontier",
        "hard",
    ]
    assert manifest[2]["filtered_hard"] is True


def test_duplicate_leakage_and_eos_helpers() -> None:
    first = _row(1)
    second = _row(2)
    assert duplicate_audit([first, second])["max_repeat"] == 1
    assert duplicate_audit([first, first])["duplicate_rows"] == 1
    assert overlap([first], [second])["passed"] is True
    assert overlap([first], [first])["passed"] is False
    assert eos_static_audit([first, second])["static_gate_passed"] is True


def test_phase0_outputs_are_safety_stopped() -> None:
    root = Path("outputs/stage2_sft_incremental_ablation")
    phase0 = root / "phase0"
    classification = [
        json.loads(line)
        for line in (phase0 / "classification_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    replay = [
        json.loads(line)
        for line in (phase0 / "phase1_replay_control_manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    decision = json.loads(
        (phase0 / "next_phase_decision.json").read_text(encoding="utf-8")
    )
    assert len(classification) == 3000
    assert len(replay) == 0
    assert decision["phase1_authorized"] is False
    assert decision["trainer_created"] is False
    assert decision["gpu_training_started"] is False
    assert not (root / "phase1_replay_control").exists()
