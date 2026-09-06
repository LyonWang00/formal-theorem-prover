# Formal Theorem Prover

The `main` branch is the primary mature prover branch. It is intentionally
scoped to Lean proof-model data preparation, SFT, GRPO, Pantograph
verification, and miniF2F evaluation.

The `Multiagent` branch preserves the broader, still experimental multi-agent
codebase for future development, including Planner orchestration. Those
components, along with the earlier proof-repair workflows, are intentionally
not included in `main`.

## Prover architecture

This branch implements a proof-model training and evaluation stack rather than
the broader agent system. Its boundary starts at theorem/proof records and ends
at reproducible Lean-verified metrics. Planner orchestration, theorem
decomposition, and proof-repair agents are outside this branch.

### End-to-end pipeline

```text
source theorem datasets
  -> source adapters and Lean-context recovery
  -> normalized, provenance-preserving records
  -> Pantograph verification and environment attestation
  -> deduplication, leakage, length, and schema gates
  -> frozen SFT prompt/completion or GRPO statement-only manifests
  -> DDP training and vLLM rollout on the GPU compute node
  -> frozen miniF2F generations
  -> persistent Pantograph compilation workers
  -> pass@k, timeout, truncation, and API-hallucination statistics
```

The data, training, rollout, and evaluation stages exchange immutable
manifests or receipts. A run contract records the input hashes, model lineage,
Lean/Mathlib environment, training parameters, and evaluation settings so that
a later stage cannot silently consume data from a different run.

### Data pipeline and quality gates

The canonical data contract is implemented under
`lean_prover/lean_training/data/`. A record retains its source dataset,
module/file/declaration location, commit or revision, imports, namespaces,
variables, hypotheses, type-class instances, scopes, local context, theorem
statement, and proof when one is available. This prevents a proof that only
works in an accidentally incomplete or altered context from entering training.

Data moves through explicit raw, verified, and quarantined states:

1. Dataset-specific adapters recover the complete Lean context and normalize
   records into the canonical schema.
2. Structural checks reject malformed records, missing identities, duplicate
   examples, proof leakage into GRPO prompts, and examples outside configured
   token limits.
3. Pantograph assembles and compiles the exact source in the pinned
   Lean/Mathlib environment. Successful verification produces an attestation
   binding the record hash, environment hash, assembler/normalizer version,
   and assembled-source hash.
4. SFT admission additionally requires a verified target proof. GRPO admission
   keeps only the prompt, theorem statement, stable problem ID, and verification
   metadata; reference proofs and completions are forbidden.
5. Accepted rows are frozen into manifests. Dataset order, sampling mode,
   source weights, and any `repeat` expansion are resolved before launch so the
   effective example count and global batch divisibility can be audited.

The same verified records can therefore feed SFT and GRPO without conflating
their supervision contracts: SFT learns from a known proof, while GRPO must
generate and verify a new proof without seeing the reference answer.

### Supervised fine-tuning

The SFT pipeline under `lean_prover/lean_training/sft_pipeline/` consumes
canonical `prompt`/`completion` pairs. Before training it validates the frozen
manifest, prompt and completion lengths, EOS supervision, provenance and
Pantograph attestations. It supports either a fixed manifest without
replacement or an explicitly configured source-weighted sampler; the selected
mode and seed are recorded in the run configuration.

Models are loaded with the configured quantization policy and trained through
LoRA/QLoRA. The trainer uses completion-only loss, disables sequence packing,
and supports gradient checkpointing and BF16/FP16 according to the hardware
contract. Tokenization diagnostics, resolved training arguments, adapter
configuration, dataset hashes, and checkpoints are emitted by the main process
for audit and resumption.

### GRPO and exploration-oriented sampling

The generic GRPO pipeline validates statement-only prompts, samples candidate
proofs, obtains executable Lean rewards from Pantograph, and logs both reward
components and verifier diagnostics. The current production experiment is the
immutable `dynamic-grpo-v2.1.2` design in
`experiments/dynamic_grpo_v2_1_2/`, which is tuned to preserve exploration
rather than repeatedly optimizing already mastered problems.

Its current 144-step contract works in cycles:

- Screen 512 problems from the larger pool with eight attempts per problem.
- Treat 5/8 or more successes as mastered and cool those problems down; retain
  1/8--4/8 problems as active training candidates.
- Top up 0/8 problems toward 32 attempts. A newly discovered success admits the
  problem, while persistent failures enter the hard bank instead of receiving
  a fabricated positive reward. Top-up attempts are selection evidence, not
  duplicated training tickets.
- Build 256 unique training slots and execute 16 optimizer steps with 16 unique
  problems per step. Each slot has `repeat=1`; a new screen is performed after
  at most 16 steps, preventing one fixed set of problems from being reused for
  the entire run. If useful candidates are short, zero-success exploration
  problems fill the remaining slots rather than triggering an unbounded extra
  screening loop.
- Generate fresh online attempts for the selected problem in the current cycle.
  Historical success rates influence selection only; they never provide a
  stale reward for a newer policy.

The executable proof result is the base trajectory reward. Current-cycle
difficulty weights are 1.30, 1.10, 0.80, and 0.60 for problems with 1/8, 2/8,
3/8, and 4/8 screening success respectively; 0/8 and mastered (at least 5/8)
outcomes receive zero centered advantage. Advantages are centered by the
problem-group mean, are **not** divided by the group standard deviation, and
are then scaled by a lagged cross-problem RMS statistic. This preserves the
signal from rare successes without allowing a single 1/8 group to set an
arbitrarily large normalized scale. KL penalty, strategy-drift scaling, and
Clip-Higher are disabled in this comparison contract.

Every cycle records the old-policy checkpoint, selected problem IDs, rollout
policy step, reward source, and their hashes. Training fails closed on a stale
or cross-cycle rollout, duplicate slot, non-canonical problem, top-up attempt
used as a training ticket, or lineage mismatch.

### vLLM generation and persistent Lean verification

The production cluster path uses vLLM for batched GPU generation during GRPO
rollouts and miniF2F evaluation. Keeping generation on the GPU node avoids
serial Transformers decoding and amortizes model loading across many attempts.
Lean checking remains a separate executable-reward service: a bounded,
persistent Pantograph worker pool is warmed once, reuses the loaded
Lean/Mathlib environment across batches, emits heartbeats, and can restart an
individual failed worker without discarding the whole run. This separation
also prevents slow theorem compilation from blocking GPU inference scheduling.

The current GRPO contract runs four persistent Pantograph workers with a
60-second per-attempt timeout. miniF2F first generates all 244 x 32 attempts on
the four-GPU compute node, freezes and hashes the output, and then compiles it
with two persistent, serially prewarmed cloud workers using a 30-second
per-attempt timeout. Compilation is receipt-based and resumable, so an
interruption resumes missing attempts rather than regenerating model outputs.

### DDP training and batch integrity

Both SFT and GRPO are launched with one distributed process per GPU. The SFT
runtime uses Accelerate's local rank to bind each process to exactly one device
and rejects ambiguous `device_map="auto"` or a fixed CUDA device in distributed
mode. The effective global batch is

```text
per-device batch x gradient accumulation x DDP world size
```

Before training, the pipeline checks that all ranks receive the same number of
microbatches and that the frozen data size is divisible by the effective
global batch. It does not silently pad, replay, or drop examples to hide an
uneven tail. The published cluster runs use four-way DDP on RTX 5090 GPUs;
gradient checkpointing and LoRA/QLoRA keep the memory footprint within the
per-GPU budget while synchronized optimizer updates preserve a single global
policy.

### Checkpoints, evaluation, and reproducibility

A valid training commit point is a complete checkpoint containing model or
adapter weights, optimizer state, scheduler state, and one RNG state per DDP
rank. Partial directories are not eligible for automatic evaluation or resume.
The dynamic GRPO contract additionally evaluates the complete step-108
checkpoint and can resume it exactly to step 144.

miniF2F evaluation uses a fixed 244-problem test manifest and reports prefix
and unbiased pass@1/4/8/16/32 together with attempt success, length-limit
truncation, compile timeout, verifier error, and hallucinated-API counts. The
immutable experiment directories contain run contracts, hashes, scheduler
logs, generation/compilation receipts, and result summaries; private datasets
and large checkpoints remain off Git but are identified by their frozen paths
and digests.

## miniF2F-test comparison

The following whole-proof results are transcribed from the supplied comparison
tables. Only the requested models and the 7B DeepSeek-Prover-V2 variants are
included. For each model, the table keeps only the reported sample budget
closest to 32; when no 32-sample result is available, the nearest available
budget is retained. Results for this project use 32 generated attempts per
problem and, by project reporting convention, are shown with `±0.5%`.

| Model | Model size | Sample budget | miniF2F-test |
|---|---:|---:|---:|
| **Ours-SFT** | 7B | 32 | **50.00% ± 0.5%** |
| **Ours-RL** | 7B | 32 | **62.30% ± 0.5%** |
| DeepSeek-Prover-V1.5-Base | 7B | 128 | 29.7% ± 0.5% |
| DeepSeek-Prover-V1.5-SFT | 7B | 32 | 48.2% ± 0.6% |
| DeepSeek-Prover-V1.5-RL | 7B | 32 | 50.0% ± 0.5% |
| DeepSeek-Prover-V2 (non-CoT) | 7B | 32 | 68.0% ± 0.5% |
| DeepSeek-Prover-V2 (CoT) | 7B | 32 | 75.6% ± 0.5% |
| Goedel-Prover-SFT | 7B | 25,600 | 64.7% |
| Leanabell-Prover | 7B | 128 | 61.1% |
| Kimina-Prover-Preview-Distill-7B | 7B | 32 | 63.1% |

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
