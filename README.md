# Formal Theorem Prover — Prover Branch

This is the primary mature **prover** branch. It is intentionally scoped to
Lean proof-model data preparation, SFT, GRPO, Pantograph verification, and
miniF2F evaluation.

The repository's `main` branch remains under active development and contains
additional components such as Planner orchestration. Those components, along
with the earlier proof-repair workflows, are intentionally not included here.

## miniF2F-test results

All evaluations use 244 miniF2F-test problems and 32 generated attempts per
problem. “Prefix” means whether the first *k* stored attempts contain a valid
Lean proof; “unbiased” is the standard pass@k estimator over all 32 attempts.

| Model | pass@1 prefix | pass@4 | pass@8 | pass@16 | pass@32 | Attempt success |
|---|---:|---:|---:|---:|---:|---:|
| Latest SFT, DeepSeek-Prover-V1.5-Base + LoRA r32/a64, lr 2e-4, 2 epochs | 27.46% | 36.48% | 42.21% | 46.31% | **50.00%** | 27.06% |
| Initial GRPO, lr 1e-5, beta 0, 4 iterations | 37.30% | 50.00% | 54.10% | 59.43% | **62.30%** | 37.86% |

The corresponding unbiased pass@1/4/8/16/32 estimates are:

- Latest SFT: 27.06%, 38.57%, 42.88%, 46.70%, 50.00%.
- Initial GRPO: 37.86%, 49.04%, 53.75%, 58.18%, 62.30%.

These are historical runs with their original evaluation contracts. The latest
SFT bundle used four persistent compile workers with a 30-second attempt
timeout. The initial GRPO bundle preserves its original 180-second evaluation
timeout, so secondary timeout and error-rate comparisons should account for
that difference.

## Repository layout

- `lean_prover/lean_training/`: SFT, GRPO, data, rollout, and evaluation code.
- `lean_prover/backends/`: Pantograph verification backend.
- `lean_project/`: pinned Lean/Mathlib project used for verification.
- `scripts/`: environment, distributed-training, and evaluation utilities.
- `experiments/dynamic_grpo_v2_1_2/`: current immutable dynamic GRPO design.
- `experiments/sft_latest_pass32_50/`: latest SFT evaluation, run contracts,
  cluster training logs, and evaluation logs.
- `experiments/grpo_initial_pass32_62_30/`: initial GRPO evaluation, run
  contracts, compressed cluster training logs, generation logs, and compile
  logs.

## Result provenance

The included cluster logs are not synthetic summaries: they are the scheduler
stdout/stderr and batch contracts retained from the corresponding training and
evaluation jobs. Large GRPO scheduler logs are stored as `.gz`; decompress with
`gzip -dk <file>.gz`. Each result bundle includes `ARTIFACTS.sha256` for
integrity checking.

Model checkpoints and private training datasets are not committed to Git. The
frozen checkpoint/data receipts and run contracts record the cluster paths and
hashes needed to trace the published evaluations.

## Base installation

```bash
python -m pip install -e .
```

GPU training has stricter CUDA, Torch, TRL, vLLM, and bitsandbytes constraints
than the base package. For reproducible cluster work, inspect the relevant
experiment contract and runtime-dependency receipt first, then run the
environment preflight scripts before launching any job.
