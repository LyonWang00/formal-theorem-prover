from __future__ import annotations

import importlib.util
from pathlib import Path

import yaml

from lean_prover.lean_training.ablation.difficulty import assign_static_buckets


_GATE_SPEC = importlib.util.spec_from_file_location(
    "prepare_ablation_gates",
    Path(__file__).resolve().parents[1] / "scripts" / "prepare_ablation_gates.py",
)
assert _GATE_SPEC and _GATE_SPEC.loader
_GATE_MODULE = importlib.util.module_from_spec(_GATE_SPEC)
_GATE_SPEC.loader.exec_module(_GATE_MODULE)
select_anchor_gate = _GATE_MODULE.select_anchor_gate
select_discovery_gate = _GATE_MODULE.select_discovery_gate


def record(**updates):
    value = {
        "proof_tokens": 10,
        "tactic_step_count": 1,
        "single_tactic": True,
        "branch_count": 0,
        "have_count": 0,
        "calc_count": 0,
        "lemma_reference_count": 0,
        "compile_time_ms": 10,
    }
    value.update(updates)
    return value


def test_static_bucket_rules_are_deterministic_and_structural():
    rows = [record() for _ in range(8)]
    rows.extend(
        [
            record(proof_tokens=100, tactic_step_count=5, single_tactic=False),
            record(branch_count=1, tactic_step_count=2, single_tactic=False),
        ]
    )
    first = [dict(row) for row in rows]
    second = [dict(row) for row in rows]
    assert assign_static_buckets(first) == assign_static_buckets(second)
    assert [row["static_bucket"] for row in first] == [
        row["static_bucket"] for row in second
    ]
    assert first[-1]["static_bucket"] == "static_hard"
    assert "branching" in first[-1]["static_hard_reasons"]


def test_anchor_gate_records_shortfall_and_redistributes_without_duplicates():
    counts = {0: 100, 1: 100, 3: 27, 4: 9}
    profile = []
    difficulty = []
    for success_count, count in counts.items():
        for index in range(count):
            record_id = f"record-{success_count}-{index}"
            profile.append({"id": record_id})
            difficulty.append(
                {"record_id": record_id, "m0_success_count_at_4": success_count}
            )
    selected, metadata = select_anchor_gate(profile, difficulty)
    assert len(selected) == 150
    assert len({row["id"] for row in selected}) == 150
    assert metadata["actual"] == {"hard": 47, "frontier": 67, "medium": 27, "easy": 9}
    assert metadata["requested_shortfall"] == {
        "hard": 0,
        "frontier": 0,
        "medium": 3,
        "easy": 11,
    }


def test_discovery_gate_uses_fixed_success_count_quotas_in_generation_order():
    quotas = {0: 50, 1: 40, 2: 30, 3: 20, 4: 10}
    discovery = []
    generations = []
    verifications = []
    for success_count, count in quotas.items():
        for index in range(count):
            statement_id = f"statement-{success_count}-{index}"
            prompt = f"prompt-{statement_id}"
            discovery.append({"statement_id": f"source-{statement_id}", "prompt": prompt})
            for attempt in range(4):
                generations.append({"statement_id": statement_id, "prompt": prompt})
                verifications.append(
                    {"statement_id": statement_id, "verified": attempt < success_count}
                )
    for index in range(350):
        statement_id = f"extra-hard-{index}"
        prompt = f"prompt-{statement_id}"
        discovery.append({"statement_id": f"source-{statement_id}", "prompt": prompt})
        for _ in range(4):
            generations.append({"statement_id": statement_id, "prompt": prompt})
            verifications.append({"statement_id": statement_id, "verified": False})
    selected, metadata = select_discovery_gate(discovery, generations, verifications)
    assert len(selected) == 150
    assert metadata["actual"] == {str(key): value for key, value in quotas.items()}


def test_ablation_yaml_configs_parse():
    root = Path(__file__).resolve().parents[1]
    for name in (
        "anchor_difficulty_audit.laptop.yaml",
        "expert_sft_ablation.laptop.yaml",
    ):
        value = yaml.safe_load((root / "configs" / name).read_text(encoding="utf-8"))
        assert value["runtime"]["profile"] == "laptop"
