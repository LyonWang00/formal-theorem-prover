from __future__ import annotations

import pytest

from lean_prover.lean_training.evaluation.expanded_validation import (
    PRIMARY_SEED,
    REPLICATION_SEED,
    assert_disjoint,
    benchmark_gate,
    metric_summary,
    paired_analysis,
    primary_gate,
    proof_free_prompt,
    restore_full500_rows,
    seed_map,
    stable_full500_ids,
)


def row(index: int, *, proof: str = "by simp") -> dict:
    return {
        "statement_id": f"stmt_{index}",
        "statement": f"theorem t{index} : True := by trivial",
        "lean_statement": f"theorem t{index} : True",
        "prompt": f"### Lean statement\ntheorem t{index} : True\n\n### Lean proof\n",
        "reference_proof": proof,
    }


def attempts(successes: dict[int, set[int]]) -> list[dict]:
    return [
        {
            "problem_id": f"stmt_{problem}",
            "attempt_index": sample,
            "success": sample in solved,
        }
        for problem, solved in successes.items()
        for sample in range(4)
    ]


def test_full500_order_and_cardinality_contract() -> None:
    generations = [
        {"statement_id": f"stmt_{problem}", "sample_index": sample}
        for problem in range(500)
        for sample in range(4)
    ]
    ids = stable_full500_ids(generations)
    assert ids == [f"stmt_{index}" for index in range(500)]
    with pytest.raises(ValueError):
        stable_full500_ids(generations[:-1])


def test_legacy_round1_ids_restore_by_unique_normalized_statement_hash() -> None:
    discovery = [row(index) for index in range(500)]
    generations = [
        {
            "statement_id": f"legacy_{problem}",
            "sample_index": sample,
            "prompt": discovery[problem]["prompt"],
        }
        for problem in range(500)
        for sample in range(4)
    ]
    restored, mapping = restore_full500_rows(discovery, generations)
    assert [item["statement_id"] for item in restored[:2]] == ["legacy_0", "legacy_1"]
    assert restored[0]["verified_v2_statement_id"] == "stmt_0"
    assert len(mapping) == 500


def test_role_and_normalized_hash_overlap_contract() -> None:
    assert assert_disjoint("left", [row(1)], "right", [row(2)])["valid"]
    duplicate = row(99)
    duplicate["lean_statement"] = row(1)["lean_statement"] + " -- comment"
    report = assert_disjoint("left", [row(1)], "right", [duplicate])
    assert not report["valid"]
    assert report["shared_normalized_statement_hashes"]


def test_reference_proof_never_enters_prompt() -> None:
    assert proof_free_prompt(row(1)).endswith("### Lean proof\n")
    leaked = row(2)
    leaked["prompt"] += leaked["reference_proof"]
    with pytest.raises(ValueError):
        proof_free_prompt(leaked)


def test_seed_maps_match_models_and_change_across_global_seed() -> None:
    rows = [row(index) for index in range(10)]
    primary_m0 = seed_map(rows, PRIMARY_SEED, generator_batch_size=2)
    primary_b2 = seed_map(rows, PRIMARY_SEED, generator_batch_size=2)
    replication = seed_map(rows, REPLICATION_SEED, generator_batch_size=2)
    assert primary_m0 == primary_b2
    assert primary_m0 != replication
    assert seed_map(rows, PRIMARY_SEED, generator_batch_size=2) == primary_m0


def test_metrics_paired_bootstrap_and_mcnemar_fixture() -> None:
    m0 = attempts({0: {0}, 1: set(), 2: {2}, 3: set()})
    b2 = attempts({0: {0}, 1: {1}, 2: set(), 3: {3}})
    metric = metric_summary(b2)
    assert metric["pass_at_1"] == pytest.approx(0.25)
    assert metric["pass_at_2"] == pytest.approx(0.5)
    assert metric["pass_at_4"] == pytest.approx(0.75)
    paired = paired_analysis(m0, b2, bootstrap_seed=7, resamples=1000)
    assert paired["both_solved"] == 1
    assert paired["m0_only_solved"] == 1
    assert paired["b2_only_solved"] == 2
    assert paired["neither_solved"] == 0
    assert paired["delta_pass_at_4"] == pytest.approx(0.25)
    assert 0 <= paired["mcnemar_exact_two_sided_p"] <= 1


def test_primary_gate_blocks_replication_without_zero_filled_metrics() -> None:
    blocked = primary_gate(
        strict_m0_solved=10,
        strict_b2_solved=9,
        nonexpert_m0_solved=20,
        nonexpert_b2_solved=20,
        monitor_m0_solved=5,
        monitor_b2_solved=5,
    )
    assert not blocked["run_replication"]
    assert "replication_metric" not in blocked
    passed = primary_gate(
        strict_m0_solved=10,
        strict_b2_solved=10,
        nonexpert_m0_solved=20,
        nonexpert_b2_solved=19,
        monitor_m0_solved=5,
        monitor_b2_solved=4,
    )
    assert passed["run_replication"]


def test_benchmark_gate_requires_both_seeds_and_monitors() -> None:
    gate = benchmark_gate(
        strict_primary_m0=10,
        strict_primary_b2=11,
        strict_replication_m0=10,
        strict_replication_b2=9,
        monitor_primary_m0=5,
        monitor_primary_b2=5,
        monitor_replication_m0=5,
        monitor_replication_b2=5,
    )
    assert not gate["run_benchmark"]
    gate["checks"]["strict_replication_not_lower"] = True
    assert "round2" not in gate
