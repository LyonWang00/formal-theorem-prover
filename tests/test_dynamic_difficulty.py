from __future__ import annotations

from collections import Counter

import pytest

from lean_prover.lean_training.expert_iteration.discovery_scheduler import (
    DiscoveryScheduler,
)
from lean_prover.lean_training.expert_iteration.dynamic_difficulty import (
    DynamicDifficultyError,
    classify_discovery,
    classify_success_rate,
    classify_theorem,
)
from lean_prover.lean_training.expert_iteration.dynamic_difficulty_sampler import (
    sample_by_difficulty,
    sample_profile,
)


def manifest(theorem_id: str = "thm-1", record_id: str = "row-1") -> dict:
    return {
        "id": record_id,
        "theorem_group_id": theorem_id,
        "source": "WB",
        "category": "Algebra",
        "statement": "theorem t : True",
    }


def candidates(
    successes: int,
    *,
    attempts: int = 8,
    theorem_id: str = "thm-1",
    record_id: str = "row-1",
    failure: str = "unsolved_goals",
) -> list[dict]:
    return [
        {
            "candidate_id": f"candidate-{theorem_id}-{rank}",
            "candidate_rank": rank,
            "candidate_seed": 100 + rank,
            "statement_id": theorem_id,
            "source_id": record_id,
            "pantograph_verified": rank <= successes,
            "failure_taxonomy": "verified" if rank <= successes else failure,
            "generated_proof": f"by exact proof_{rank}",
            "generation_length": rank,
            "checkpoint_hash": "checkpoint",
            "generation_config_hash": "contract",
        }
        for rank in range(1, attempts + 1)
    ]


@pytest.mark.parametrize(
    ("successes", "expected"),
    [(8, "too_easy"), (6, "too_easy"), (5, "easy"), (4, "easy"),
     (3, "medium"), (2, "medium"), (1, "hard"), (0, "impossible")],
)
def test_eight_candidate_boundaries(successes: int, expected: str) -> None:
    assert classify_success_rate(successes, 8) == expected


def test_rate_thresholds_generalize_to_other_attempt_counts() -> None:
    assert classify_success_rate(3, 4) == "too_easy"
    assert classify_success_rate(2, 4) == "easy"
    assert classify_success_rate(1, 4) == "medium"
    assert classify_success_rate(1, 5) == "hard"


def test_theorem_metrics_and_prefix_pass_at_k() -> None:
    row = classify_theorem(manifest(), candidates(3))
    assert row["difficulty"] == "medium"
    assert row["success_rate"] == 3 / 8
    assert row["pass_at_1"] == 1
    assert row["pass_at_2"] == 1
    assert row["pass_at_4"] == 1
    assert row["pass_at_8"] == 1
    assert row["proof_statistics"]["successful_proof_count"] == 3
    assert row["domain"] == "algebra"
    assert row["effective_for_ei_default"]


def test_zero_success_subtypes() -> None:
    capability = classify_theorem(manifest(), candidates(0, failure="tactic_error"))
    invalid = classify_theorem(
        manifest(), candidates(0, failure="environment_error")
    )
    assert capability["difficulty"] == "impossible"
    assert capability["zero_success_subtype"] == "capability_hard"
    assert "exploration" in capability["selection_tags"]
    assert invalid["zero_success_subtype"] == "data_environment_hard"
    assert not invalid["effective_for_ei_default"]


def test_classification_join_and_duplicate_gate() -> None:
    records = classify_discovery([manifest()], candidates(4))
    assert len(records) == 1
    with pytest.raises(DynamicDifficultyError, match="duplicate theorem_id"):
        classify_discovery([manifest(), manifest(record_id="row-2")], candidates(4))


def test_candidate_statement_id_is_canonical_and_group_id_is_preserved() -> None:
    records = classify_discovery(
        [manifest(theorem_id="stable-group")],
        candidates(2, theorem_id="runtime-statement"),
    )
    assert records[0]["theorem_id"] == "runtime-statement"
    assert records[0]["theorem_group_id"] == "stable-group"


def synthetic_rows() -> list[dict]:
    rows = []
    for difficulty, count in (("easy", 6), ("medium", 10), ("hard", 10)):
        for index in range(count):
            rows.append(
                {
                    "theorem_id": f"{difficulty}-{index}",
                    "record_id": f"record-{difficulty}-{index}",
                    "difficulty": difficulty,
                    "zero_success_subtype": None,
                    "source": "WB",
                    "effective_for_ei_default": True,
                }
            )
    rows.append(
        {
            "theorem_id": "impossible-capability",
            "record_id": "record-impossible",
            "difficulty": "impossible",
            "zero_success_subtype": "capability_hard",
            "source": "LD-easy",
            "effective_for_ei_default": False,
        }
    )
    return rows


def test_sampler_categories_ratio_and_profiles() -> None:
    rows = synthetic_rows()
    selected = sample_by_difficulty(
        rows,
        categories=["medium", "hard"],
        ratio={"medium": 0.7, "hard": 0.3},
        total=10,
        seed=7,
    )
    assert Counter(row["difficulty"] for row in selected) == {
        "medium": 7,
        "hard": 3,
    }
    assert len({row["theorem_id"] for row in selected}) == 10
    exploration = sample_profile(rows, profile="exploration", seed=9)
    assert {row["difficulty"] for row in exploration} == {"hard", "impossible"}


def test_scheduler_targets_effective_yield_not_fixed_statement_ratio() -> None:
    scheduler = DiscoveryScheduler({"WB": 10, "LD-easy": 4})
    scheduler.ingest(synthetic_rows())
    assert scheduler.remaining_targets() == {"WB": 0, "LD-easy": 4}
    plan = scheduler.plan_next_batch({"WB": 0.5, "LD-easy": 0.25})
    assert plan == {"WB": 0, "LD-easy": 16}
    assert not scheduler.complete
