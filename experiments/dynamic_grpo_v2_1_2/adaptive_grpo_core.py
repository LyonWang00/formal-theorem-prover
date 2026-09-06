"""Pure policy-selection and reward-shaping primitives for adaptive GRPO.

This module deliberately has no torch/TRL dependency.  It is used by both the
rollout router and the trainer adapter, and can therefore be unit-tested on a
login node before a GPU job is submitted.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import math
from typing import Iterable, Sequence


class Route(str, Enum):
    ACTIVE_PASS8 = "active_pass8"
    NEED_PASS16_TOPUP = "need_pass16_topup"
    NEED_PASS32_TOPUP = "need_pass32_topup"
    ACTIVE_AFTER_TOPUP = "active_after_topup"
    MASTERED = "mastered"
    HARD_BANK = "hard_bank"


@dataclass(frozen=True)
class RewardConfig:
    """Two-level reward controls.

    Problem weights use only the current old-policy rollout.  Historical rates
    are intentionally excluded from reward computation and are used only by
    the sampling/routing layer.
    """

    num_generations: int = 8
    mastered_successes: int = 5
    trajectory_rank_beta: float = 0.125
    weight_success_1: float = 1.20
    weight_success_2: float = 1.10
    weight_success_3: float = 1.00
    weight_success_4: float = 0.90
    sparse_pass32_weight_cap: float = 1.30
    scale_epsilon: float = 1e-4

    def validate(self) -> None:
        if self.num_generations != 8:
            raise ValueError("The first adaptive contract is calibrated for pass@8 groups.")
        if not 1 <= self.mastered_successes <= self.num_generations:
            raise ValueError("mastered_successes must be inside the rollout group.")
        if not 0.0 <= self.trajectory_rank_beta < 1.0:
            raise ValueError("trajectory_rank_beta must be in [0, 1).")
        if self.scale_epsilon <= 0:
            raise ValueError("scale_epsilon must be positive.")


@dataclass(frozen=True)
class TrainingLayout:
    world_size: int
    per_device_train_batch_size: int
    num_generations: int
    problems_per_update: int
    generation_batch_size: int
    gradient_accumulation_steps: int
    prompt_rows_per_update: int
    num_iterations: int = 1


def make_memory_safe_layout(
    *,
    world_size: int = 4,
    per_device_train_batch_size: int = 1,
    num_generations: int = 8,
    problems_per_update: int = 16,
    num_iterations: int = 1,
) -> TrainingLayout:
    """Build a sequential-rollout layout without increasing generation peak.

    One generation batch contains exactly one problem group.  Several such
    groups are back-propagated sequentially and share one optimizer update.
    """

    local_global_batch = world_size * per_device_train_batch_size
    if min(local_global_batch, num_generations, problems_per_update, num_iterations) <= 0:
        raise ValueError("layout dimensions must be positive")
    if num_generations % local_global_batch:
        raise ValueError(
            "num_generations must be divisible by world_size * per-device batch"
        )
    microsteps_per_problem = num_generations // local_global_batch
    return TrainingLayout(
        world_size=world_size,
        per_device_train_batch_size=per_device_train_batch_size,
        num_generations=num_generations,
        problems_per_update=problems_per_update,
        generation_batch_size=num_generations,
        # Every generated trajectory is reused ``num_iterations`` times.  Count
        # those replayed microsteps so all ``problems_per_update`` groups are
        # collected under one unchanged policy before the optimizer moves.
        gradient_accumulation_steps=(
            microsteps_per_problem * problems_per_update * num_iterations
        ),
        prompt_rows_per_update=problems_per_update,
        num_iterations=num_iterations,
    )


def route_rollout(*, initial_successes: int, total_successes: int, total_attempts: int) -> Route:
    """Route one problem through capped pass@8 -> pass@16 -> pass@32 stages."""

    if not 0 <= initial_successes <= 8:
        raise ValueError("initial_successes must be in [0, 8]")
    if total_attempts not in {8, 16, 32}:
        raise ValueError("total_attempts must be 8, 16, or 32")
    if not initial_successes <= total_successes <= total_attempts:
        raise ValueError("inconsistent success counts")
    if initial_successes >= 5:
        return Route.MASTERED
    if initial_successes >= 1:
        return Route.ACTIVE_PASS8
    if total_attempts == 8:
        return Route.NEED_PASS16_TOPUP
    if total_successes:
        return Route.ACTIVE_AFTER_TOPUP
    if total_attempts == 16:
        return Route.NEED_PASS32_TOPUP
    return Route.HARD_BANK


def problem_weight(
    *, successes: int, attempts: int, config: RewardConfig | None = None
) -> float:
    """Return the current-round problem-level weight.

    For pass@8, 1/8 and 2/8 get a small lift while 4/8 is discounted.  A
    pass@32 discovery receives a bounded interpolation, never a fourfold mass.
    """

    cfg = config or RewardConfig()
    cfg.validate()
    if successes <= 0 or attempts <= 0:
        return 0.0
    if attempts == 8:
        table = {
            1: cfg.weight_success_1,
            2: cfg.weight_success_2,
            3: cfg.weight_success_3,
            4: cfg.weight_success_4,
        }
        return table.get(successes, 0.0)
    # Top-up rollouts are for discovery and admission.  Their training ticket
    # still appears once in the pool; this weight only records their scarcity.
    rate = successes / attempts
    unclipped = 1.0 + 0.30 * max(0.0, 1.0 - rate / 0.125)
    return min(cfg.sparse_pass32_weight_cap, unclipped)


def _average_tie_ranks_desc(values: Sequence[float]) -> list[float]:
    """Zero-based descending ranks with average ranks for exact ties."""

    order = sorted(range(len(values)), key=lambda i: (-values[i], i))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        average_rank = (start + end - 1) / 2.0
        for position in range(start, end):
            ranks[order[position]] = average_rank
        start = end
    return ranks


def trajectory_rank_factors(
    old_policy_mean_logprobs: Sequence[float], beta: float
) -> list[float]:
    """Downweight common trajectories and retain rarer correct trajectories.

    Larger old-policy log probability means a more common trajectory.  Missing
    or non-finite scores are neutral instead of being assigned an artificial
    rarity bonus.
    """

    if not 0.0 <= beta < 1.0:
        raise ValueError("beta must be in [0, 1)")
    size = len(old_policy_mean_logprobs)
    if size == 0:
        return []
    finite_indices = [
        i for i, value in enumerate(old_policy_mean_logprobs) if math.isfinite(value)
    ]
    factors = [1.0] * size
    if not finite_indices:
        return factors
    finite_values = [old_policy_mean_logprobs[i] for i in finite_indices]
    ranks = _average_tie_ranks_desc(finite_values)
    denominator = len(finite_indices)
    for source_index, rank in zip(finite_indices, ranks):
        factors[source_index] = 1.0 - beta * (denominator - rank) / denominator
    return factors


@dataclass(frozen=True)
class ShapedGroup:
    binary_rewards: tuple[float, ...]
    shaped_rewards: tuple[float, ...]
    centered_advantages: tuple[float, ...]
    trajectory_factors: tuple[float, ...]
    successes: int
    problem_weight: float
    effective: bool


def shape_group_rewards(
    binary_rewards: Sequence[float],
    old_policy_mean_logprobs: Sequence[float],
    config: RewardConfig | None = None,
) -> ShapedGroup:
    """Apply problem and trajectory layers, then subtract the group mean."""

    cfg = config or RewardConfig()
    cfg.validate()
    if len(binary_rewards) != cfg.num_generations:
        raise ValueError("reward group length differs from num_generations")
    if len(old_policy_mean_logprobs) != len(binary_rewards):
        raise ValueError("one old-policy score is required for every trajectory")
    successes = sum(float(value) > 0.5 for value in binary_rewards)
    weight = problem_weight(successes=successes, attempts=len(binary_rewards), config=cfg)
    factors = trajectory_rank_factors(
        old_policy_mean_logprobs, cfg.trajectory_rank_beta
    )
    shaped = [
        weight * factor if float(reward) > 0.5 else 0.0
        for reward, factor in zip(binary_rewards, factors)
    ]
    mean = sum(shaped) / len(shaped)
    centered = [value - mean for value in shaped]
    return ShapedGroup(
        binary_rewards=tuple(float(value) for value in binary_rewards),
        shaped_rewards=tuple(shaped),
        centered_advantages=tuple(centered),
        trajectory_factors=tuple(factors),
        successes=successes,
        problem_weight=weight,
        effective=0 < successes < cfg.mastered_successes,
    )


@dataclass
class LaggedWindowScale:
    """Fixed scale within an update; EMA is committed for the next update."""

    groups_per_window: int = 16
    ema_decay: float = 0.90
    initial_scale: float = 0.50
    epsilon: float = 1e-4
    scale: float = 0.50
    pending_square_sum: float = 0.0
    pending_count: int = 0
    pending_groups: int = 0
    completed_windows: int = 0

    def __post_init__(self) -> None:
        if self.groups_per_window <= 0:
            raise ValueError("groups_per_window must be positive")
        if not 0.0 <= self.ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.initial_scale <= 0 or self.epsilon <= 0:
            raise ValueError("scales must be positive")
        if self.completed_windows == 0 and self.pending_count == 0:
            self.scale = self.initial_scale

    def normalize(self, advantages: Sequence[float]) -> list[float]:
        denominator = max(self.scale, self.epsilon)
        return [float(value) / denominator for value in advantages]

    def observe_group(self, centered_advantages: Iterable[float]) -> bool:
        values = [float(value) for value in centered_advantages]
        self.pending_square_sum += sum(value * value for value in values)
        self.pending_count += len(values)
        self.pending_groups += 1
        if self.pending_groups < self.groups_per_window:
            return False
        observed_rms = math.sqrt(
            self.pending_square_sum / max(self.pending_count, 1)
        )
        observed_rms = max(observed_rms, self.epsilon)
        self.scale = (
            self.ema_decay * self.scale + (1.0 - self.ema_decay) * observed_rms
        )
        self.pending_square_sum = 0.0
        self.pending_count = 0
        self.pending_groups = 0
        self.completed_windows += 1
        return True

    def state_dict(self) -> dict[str, float | int]:
        return asdict(self)

    def load_state_dict(self, state: dict[str, float | int]) -> None:
        for key in asdict(self):
            if key in state:
                setattr(self, key, state[key])


def dataset_is_update_aligned(row_count: int, layout: TrainingLayout) -> bool:
    """Rows are problems, not completions; require whole problem windows."""

    return row_count > 0 and row_count % layout.prompt_rows_per_update == 0


def history_rollout_priority(
    *, historical_successes: int, historical_attempts: int, group_size: int = 8
) -> float:
    """Posterior probability that the next rollout lands in the useful 1-4 band.

    A Beta(1, 1) prior avoids giving unseen problems zero priority.  This score
    affects only which problems are rolled out; it never enters current rewards.
    """

    if historical_successes < 0 or historical_attempts < historical_successes:
        raise ValueError("inconsistent historical counts")
    if group_size != 8:
        raise ValueError("the first adaptive routing contract uses pass@8")
    alpha = historical_successes + 1.0
    beta = historical_attempts - historical_successes + 1.0
    log_denominator = math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta)
    probability = 0.0
    for successes in range(1, 5):
        log_combination = (
            math.lgamma(group_size + 1)
            - math.lgamma(successes + 1)
            - math.lgamma(group_size - successes + 1)
        )
        log_beta = (
            math.lgamma(successes + alpha)
            + math.lgamma(group_size - successes + beta)
            - math.lgamma(group_size + alpha + beta)
        )
        probability += math.exp(log_combination + log_beta - log_denominator)
    return probability
