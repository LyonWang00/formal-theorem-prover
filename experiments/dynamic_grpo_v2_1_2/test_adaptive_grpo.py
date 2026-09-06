from __future__ import annotations

import math
from types import SimpleNamespace

import torch

from adaptive_grpo_core import (
    LaggedWindowScale,
    RewardConfig,
    Route,
    dataset_is_update_aligned,
    history_rollout_priority,
    make_memory_safe_layout,
    problem_weight,
    route_rollout,
    shape_group_rewards,
    trajectory_rank_factors,
)
from adaptive_trainer import AdaptiveTrainerConfig, make_adaptive_grpo_trainer
from build_training_pool import build_pool
from select_rollout_pool import select_rollouts


def test_routing_boundaries() -> None:
    assert route_rollout(initial_successes=5, total_successes=5, total_attempts=8) is Route.MASTERED
    assert route_rollout(initial_successes=1, total_successes=1, total_attempts=8) is Route.ACTIVE_PASS8
    assert route_rollout(initial_successes=0, total_successes=0, total_attempts=8) is Route.NEED_PASS16_TOPUP
    assert route_rollout(initial_successes=0, total_successes=0, total_attempts=16) is Route.NEED_PASS32_TOPUP
    assert route_rollout(initial_successes=0, total_successes=1, total_attempts=16) is Route.ACTIVE_AFTER_TOPUP
    assert route_rollout(initial_successes=0, total_successes=1, total_attempts=32) is Route.ACTIVE_AFTER_TOPUP
    assert route_rollout(initial_successes=0, total_successes=0, total_attempts=32) is Route.HARD_BANK


def test_problem_weights_are_current_round_only_and_bounded() -> None:
    assert problem_weight(successes=1, attempts=8) == 1.20
    assert problem_weight(successes=2, attempts=8) == 1.10
    assert problem_weight(successes=3, attempts=8) == 1.00
    assert problem_weight(successes=4, attempts=8) == 0.90
    assert problem_weight(successes=5, attempts=8) == 0.0
    assert problem_weight(successes=1, attempts=32) <= 1.30


def test_rare_trajectory_gets_larger_factor() -> None:
    factors = trajectory_rank_factors([-0.1, -0.5, -1.0, -2.0], beta=0.2)
    assert factors[0] < factors[-1]
    assert all(0.8 <= value <= 1.0 for value in factors)


def test_two_level_reward_is_group_centered() -> None:
    group = shape_group_rewards(
        [1, 0, 1, 0, 0, 0, 0, 0],
        [-0.1, -0.2, -2.0, -0.3, -0.4, -0.5, -0.6, -0.7],
        RewardConfig(trajectory_rank_beta=0.125),
    )
    assert group.successes == 2
    assert group.problem_weight == 1.10
    assert math.isclose(sum(group.centered_advantages), 0.0, abs_tol=1e-12)
    # The rarer successful trajectory retains more reward than the common one.
    assert group.shaped_rewards[2] > group.shaped_rewards[0]


def test_scale_is_fixed_inside_window_and_committed_afterward() -> None:
    scale = LaggedWindowScale(groups_per_window=2, ema_decay=0.0, initial_scale=0.5)
    assert scale.normalize([1.0]) == [2.0]
    assert not scale.observe_group([-0.5, 0.5])
    assert scale.normalize([1.0]) == [2.0]
    assert scale.observe_group([-1.0, 1.0])
    expected = math.sqrt((0.25 + 0.25 + 1.0 + 1.0) / 4)
    assert math.isclose(scale.scale, expected)


def test_four_gpu_layout_keeps_pass8_generation_peak() -> None:
    layout = make_memory_safe_layout(
        world_size=4,
        per_device_train_batch_size=1,
        num_generations=8,
        problems_per_update=16,
        num_iterations=2,
    )
    assert layout.generation_batch_size == 8
    assert layout.gradient_accumulation_steps == 64
    assert layout.prompt_rows_per_update == 16
    assert layout.num_iterations == 2
    assert dataset_is_update_aligned(2400, layout)
    assert not dataset_is_update_aligned(2376, layout)


def test_history_priority_prefers_likely_effective_groups() -> None:
    boundary = history_rollout_priority(historical_successes=4, historical_attempts=16)
    mastered = history_rollout_priority(historical_successes=15, historical_attempts=16)
    hard = history_rollout_priority(historical_successes=0, historical_attempts=64)
    assert boundary > mastered
    assert boundary > hard


def test_trainer_adapter_uses_returned_old_policy_logps() -> None:
    class Accelerator:
        process_index = 0

        @staticmethod
        def gather(value):
            return value

    class BaseTrainer:
        def __init__(self, *args, **kwargs):
            del args, kwargs
            self.args = SimpleNamespace(
                num_generations=8,
                generation_batch_size=8,
                num_iterations=2,
                scale_rewards="none",
                loss_type="dapo",
                epsilon=0.2,
                epsilon_high=0.2,
            )
            self.accelerator = Accelerator()
            self.state = SimpleNamespace(global_step=0)
            self.model = SimpleNamespace(training=True)
            self.args.output_dir = "."

        @staticmethod
        def is_world_process_zero():
            return False

        def _generate_and_score_completions(self, inputs):
            del inputs
            return {
                "advantages": torch.tensor([0.75, -0.25, 0.75, -0.25, -0.25, -0.25, -0.25, -0.25]),
                "sampling_per_token_logps": torch.tensor(
                    [[-0.1, -0.1], [-0.2, -0.2], [-2.0, -2.0], [-0.3, -0.3],
                     [-0.4, -0.4], [-0.5, -0.5], [-0.6, -0.6], [-0.7, -0.7]]
                ),
                "completion_mask": torch.ones((8, 2)),
            }

    Trainer = make_adaptive_grpo_trainer(
        BaseTrainer,
        AdaptiveTrainerConfig(
            problems_per_update=2,
            initial_advantage_scale=0.5,
            num_iterations=2,
        ),
    )
    trainer = Trainer()
    output = trainer._generate_and_score_completions([])
    assert math.isclose(float(output["advantages"].sum()), 0.0, abs_tol=1e-6)
    assert output["advantages"][2] > output["advantages"][0]
    assert trainer._adaptive_scale.pending_groups == 1


def test_pool_is_one_ticket_per_problem_and_update_aligned() -> None:
    rows = [{"id": f"p{i:02d}", "prompt": "x", "repeat": 4} for i in range(19)]
    priorities = {f"p{i:02d}": float(i) for i in range(19)}
    selected, holdout = build_pool(
        source_rows=rows,
        active_ids=set(priorities),
        priority=priorities,
        id_field="id",
        problems_per_update=16,
        seed=7,
    )
    assert len(selected) == 16
    assert len(holdout) == 3
    assert all(row["repeat"] == 1 for row in selected)
    assert len({row["id"] for row in selected}) == 16


def test_history_selects_rollouts_but_does_not_modify_source_reward_fields() -> None:
    source = [{"id": "new"}, {"id": "active"}, {"id": "mastered"}]
    history = [
        {"problem_id": "active", "last_route": "active_pass8", "rollout_priority": 0.7, "history_visits": 2},
        {"problem_id": "mastered", "last_route": "mastered", "rollout_priority": 0.2, "history_visits": 3},
    ]
    selected, deferred = select_rollouts(
        source_rows=source,
        history_rows=history,
        budget=2,
        id_field="id",
        seed=1,
    )
    assert {row["id"] for row in selected} == {"new", "active"}
    assert any(row["problem_id"] == "mastered" for row in deferred)
