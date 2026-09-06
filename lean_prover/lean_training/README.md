# Lean proof-model training

This package contains the mature prover-only training path:

- verified dataset preparation and leakage checks;
- supervised fine-tuning (SFT) with LoRA/QLoRA and distributed launch support;
- GRPO with Lean/Pantograph compile rewards;
- rollout generation and miniF2F evaluation;
- persistent Pantograph worker pools and reproducibility receipts.

Planner orchestration, proof-repair workflows, and application-level agents are
deliberately outside this branch.

## Entrypoints

Run the command help before preparing an experiment:

```bash
python -m lean_prover.lean_training.sft --help
python -m lean_prover.lean_training.grpo --help
python -m lean_prover.lean_training.data.cli --help
```

The launch scripts under `scripts/` cover the distributed SFT path, environment
preflight, benchmark generation, Pantograph verification, and result summaries.
Experiment-specific cluster contracts and batch scripts are preserved under
`experiments/`.

## Current GRPO architecture

`experiments/dynamic_grpo_v2_1_2/` is the immutable source snapshot for the
latest dynamic difficulty-aware GRPO design. It uses periodic old-policy
screening, fresh current-cycle reward rollouts, two-level reward shaping,
persistent Pantograph workers, and fail-closed rollout lineage checks. Read its
`README.md` and `ARCHITECTURE.md` before launching or modifying a run.

## Data and verification invariants

- SFT completions must contain a complete `by ...` proof and be marked as
  Pantograph-verified in the audit manifest.
- GRPO inputs are statement-only. Generated trajectories receive compile-based
  rewards during training.
- Training and benchmark splits must be checked for normalized-statement and
  source-identity leakage.
- Lean, Mathlib, source, data, and run contracts should be hashed and frozen for
  any result intended for comparison.
- Benchmark compilation should use prewarmed persistent workers; the current
  evaluation convention is a 30-second per-attempt timeout.

## Reproducibility

The result bundles in `experiments/sft_latest_pass32_50/` and
`experiments/grpo_initial_pass32_62_30/` include configurations, completion
markers, analyses, and scheduler logs. `ARTIFACTS.sha256` in each bundle binds
the public artifacts byte-for-byte.
