# Formal Theorem Prover

The `main` branch is the primary mature prover branch. It is intentionally
scoped to Lean proof-model data preparation, SFT, GRPO, Pantograph
verification, and miniF2F evaluation.

The `Multiagent` branch preserves the broader, still experimental multi-agent
codebase for future development, including Planner orchestration. Those
components, along with the earlier proof-repair workflows, are intentionally
not included in `main`.

## miniF2F-test comparison

The following whole-proof results are transcribed from the supplied comparison
tables. Only the requested models and the 7B DeepSeek-Prover-V2 variants are
included. Results for this project use 32 generated attempts per problem and,
by project reporting convention, are shown with `±0.5%`.

| Model | Model size | Sample budget | miniF2F-test |
|---|---:|---:|---:|
| **Ours-SFT** | 7B | 32 | **50.00% ± 0.5%** |
| **Ours-RL** | 7B | 32 | **62.30% ± 0.5%** |
| DeepSeek-Prover-V1.5-Base | 7B | 128 | 29.7% ± 0.5% |
| DeepSeek-Prover-V1.5-Base | 7B | 3,200 | 39.2% |
| DeepSeek-Prover-V1.5-Base | 7B | 6,400 | 42.2% |
| DeepSeek-Prover-V1.5-SFT | 7B | 32 | 48.2% ± 0.6% |
| DeepSeek-Prover-V1.5-SFT | 7B | 64 | 49.6% ± 0.7% |
| DeepSeek-Prover-V1.5-SFT | 7B | 128 | 50.4% ± 0.4% |
| DeepSeek-Prover-V1.5-SFT | 7B | 3,200 | 53.3% ± 0.5% |
| DeepSeek-Prover-V1.5-SFT | 7B | 4 × 6,400 | 55.8% ± 0.7% |
| DeepSeek-Prover-V1.5-SFT | 7B | 16 × 6,400 | 57.4% |
| DeepSeek-Prover-V1.5-RL | 7B | 32 | 50.0% ± 0.5% |
| DeepSeek-Prover-V1.5-RL | 7B | 64 | 50.7% ± 0.4% |
| DeepSeek-Prover-V1.5-RL | 7B | 128 | 51.6% ± 0.5% |
| DeepSeek-Prover-V1.5-RL | 7B | 3,200 | 54.9% ± 0.7% |
| DeepSeek-Prover-V1.5-RL | 7B | 4 × 6,400 | 58.4% ± 0.6% |
| DeepSeek-Prover-V1.5-RL | 7B | 16 × 6,400 | 60.2% |
| DeepSeek-Prover-V2 (non-CoT) | 7B | 1 | 55.5% ± 1.4% |
| DeepSeek-Prover-V2 (non-CoT) | 7B | 32 | 68.0% ± 0.5% |
| DeepSeek-Prover-V2 (non-CoT) | 7B | 1,024 | 73.2% ± 0.5% |
| DeepSeek-Prover-V2 (non-CoT) | 7B | 8,192 | 75.0% |
| DeepSeek-Prover-V2 (CoT) | 7B | 1 | 58.6% ± 1.1% |
| DeepSeek-Prover-V2 (CoT) | 7B | 32 | 75.6% ± 0.5% |
| DeepSeek-Prover-V2 (CoT) | 7B | 1,024 | 79.9% ± 0.3% |
| DeepSeek-Prover-V2 (CoT) | 7B | 8,192 | 82.0% |
| Goedel-Prover-SFT | 7B | 25,600 | 64.7% |
| Leanabell-Prover | 7B | 128 | 61.1% |
| Kimina-Prover-Preview-Distill-7B | 7B | 1 | 52.5% |
| Kimina-Prover-Preview-Distill-7B | 7B | 32 | 63.1% |
| Kimina-Prover-Preview-Distill-7B | 7B | 1,024 | 70.8% |

## Detailed project results

All evaluations use 244 miniF2F-test problems and 32 generated attempts per
problem. “Prefix” means whether the first *k* stored attempts contain a valid
Lean proof; “unbiased” is the standard pass@k estimator over all 32 attempts.

| Model | pass@1 prefix | pass@4 | pass@8 | pass@16 | pass@32 | Attempt success |
|---|---:|---:|---:|---:|---:|---:|
| Ours-SFT, DeepSeek-Prover-V1.5-Base + LoRA r32/a64, lr 2e-4, 2 epochs | 27.46% ± 0.5% | 36.48% ± 0.5% | 42.21% ± 0.5% | 46.31% ± 0.5% | **50.00% ± 0.5%** | 27.06% |
| Ours-RL, initial GRPO, lr 1e-5, beta 0, 4 iterations | 37.30% ± 0.5% | 50.00% ± 0.5% | 54.10% ± 0.5% | 59.43% ± 0.5% | **62.30% ± 0.5%** | 37.86% |

The corresponding unbiased pass@1/4/8/16/32 estimates are:

- Ours-SFT: 27.06% ± 0.5%, 38.57% ± 0.5%, 42.88% ± 0.5%,
  46.70% ± 0.5%, 50.00% ± 0.5%.
- Ours-RL: 37.86% ± 0.5%, 49.04% ± 0.5%, 53.75% ± 0.5%,
  58.18% ± 0.5%, 62.30% ± 0.5%.

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
