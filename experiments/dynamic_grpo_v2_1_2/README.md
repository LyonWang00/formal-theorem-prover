# Adaptive two-level GRPO implementation

This directory implements the agreed exploration-oriented GRPO design without
Clip-Higher.  It is an isolated follow-up implementation; historical round 1-6
scripts and frozen datasets are not modified.

## Fixed first-run contract

- Route every old-policy problem with pass@8.
- Remove/cool down problems with at least 5/8 successes.
- Keep 1/8 through 4/8 problems active.
- Top up 0/8 problems to pass@32; admit a problem once if any completion works,
  otherwise place it in the hard bank.
- A pass@32 top-up is admission evidence, not four training tickets.
- Shape binary compile rewards at two levels:
  - problem level: weights 1.20, 1.10, 1.00, 0.90 for 1/8 through 4/8;
  - trajectory level: lightly downweight high-old-policy-probability successes
    with rank beta 0.125, preserving more mass for rarer correct trajectories.
- Subtract each problem's group mean.  Do not divide by its group standard
  deviation.  Divide by a lagged EMA of centered RMS across problem windows.
- On 4 GPUs with one sample/GPU, generate only 8 completions at a time and
  accumulate 32 micro-steps: 16 distinct problems per optimizer update.
- Use `loss_type=dapo`, `scale_rewards=none`, and `num_iterations=1`.
- Keep symmetric clipping (`epsilon_high == epsilon`); Clip-Higher is disabled.

The lagged scale is intentional: every problem inside an optimizer window uses
the same scale.  Statistics from that window are committed only for the next
window, avoiding look-ahead while retaining a stable multi-problem scale.

## Dynamic cycle v2 (follow-up; static E2 remains immutable)

The production follow-up does not expand one screened pool with `repeat=8-9`.
It freezes a new 512-problem screen after at most 16 optimizer steps and builds
256 unique prompt slots (16 unique problems per step).  A shortage is filled
with zero-success exploration problems; useful problems are never duplicated
inside a cycle.  Pass@16 and pass@32 discovery are capped at 128 and 64
problems, respectively, so a low effective rate cannot trigger an unbounded
second 512-problem screen.

For the first comparable v2 experiment, screening and top-up receipts select
problem IDs only.  Online training performs a fresh pass@8 rollout and uses
only that current-cycle result as reward.  Historical rates affect selection
only.  Strategy-drift/KL-based scaling is explicitly disabled.  Difficulty
weights are 1.30, 1.10, 0.80, and 0.60 for current training outcomes 1/8
through 4/8; 0/8 and 5/8 or above have zero advantage.  This more strongly
suppresses repeated optimization of the 3/8-4/8 band without adding a second
new mechanism to the comparison.

Every generation and compile receipt is bound to all of:

- `rollout_cycle_id`;
- the merged screen-policy `SHA256SUMS` hash;
- the frozen 512-problem selection hash;
- a reward-source label (`screen_selection_only` or
  `fresh_training_rollout_current_cycle_only`).

Fresh online training groups additionally record the exact
`rollout_policy_step`.  The screen-policy hash is not misused as the policy
identity of all later steps: after each optimizer update, the next online
rollout has a new step identity even though it remains in the same cycle.

Any mismatch, duplicate attempt, non-canonical top-up index, changed schedule,
or cross-cycle receipt fails closed before training.

## Current production architecture (v2.1.2)

The current run extends the dynamic design to 144 optimizer steps.  It freezes
an intermediate checkpoint at step 108, generates all 244 miniF2F-test
problems at pass@32, and only then resumes the exact optimizer, scheduler, and
four-rank RNG state to step 144.  Generation remains on the four-GPU compute
node; compilation runs on the cloud with two serially prewarmed persistent
Pantograph workers and a 30-second attempt timeout.

The distributed process-group timeout is 3600 seconds.  This accommodates
temporary rank skew caused by Lean compilation without changing gradients,
batching, rewards, or optimizer behavior.  Checkpoint creation remains the
only accepted commit boundary, so a failed partial cycle always resumes from
the preceding complete checkpoint.

`project_source_dynamic_grpo_v2_1_2/` is the canonical, immutable source
bundle synchronized to local, compute, and cloud.  Runtime `code/` folders may
contain a platform-specific staged subset, while results, logs, caches,
checkpoints, rollouts, and compile receipts are intentionally outside the
source-consistency contract.

## Files

- `adaptive_grpo_core.py`: routing, two-level rewards, layout, and EMA state.
- `route_rollouts.py`: JSONL router and append-style per-problem history merger.
- `select_rollout_pool.py`: history-prioritized pass@8 selector; mastered and
  hard-bank problems remain cooled down unless explicitly re-enabled.
- `build_training_pool.py`: one-ticket-per-problem pool builder with deterministic
  priority selection, shuffle, alignment holdout, and SHA-256 freeze marker.
- `adaptive_trainer.py`: narrow TRL 1.12 adapter with checkpointed EMA state.
- `grpo_adaptive_remote.py`: launcher around the frozen round-6 training code.
- `test_adaptive_grpo.py`: pure unit tests runnable without GPUs.
- `dynamic_cycle_v2.py`: v2 difficulty/reward primitives, unique-slot schedule,
  zero filler policy, and fail-closed cycle lineage validation.
- `dynamic_cycle_controller.py`: freezes a cycle, cryptographically joins raw
  generations to compile receipts, plans capped top-ups, and freezes 16 steps.
- `smoke_dynamic_cycle_v2.py`: synthetic 512-problem end-to-end smoke including
  pass@8, capped pass@16/pass@32, 256-slot scheduling, and stale-cycle failures.
- `test_dynamic_cycle_v2.py`: focused reward, schedule, and lineage unit tests.

Before a real run, copy the frozen round-6 `grpo_remote.py` beside the launcher,
stage these files with it, and trim the selected active pool to a multiple of 16
prompt tickets.  The launcher refuses an unaligned dataset or a configuration
that silently re-enables group scaling, repeated PPO iterations, or Clip-Higher.
