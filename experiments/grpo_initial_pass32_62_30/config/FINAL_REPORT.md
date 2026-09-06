# GRPO first-round full training report (final)

## Outcome

- Full run: Slurm job `47990`, completed normally on 4 x RTX 5090 (32 GB), pure DDP data parallelism.
- Starting policy: exact SFT epoch-2 adapter at `checkpoint-18688`.
- Training: LoRA `r=16`, 8 vLLM samples per problem, `K=4`, binary compile reward only.
- Physical manifest: 5,000 proof-free rows; positive-repeat rows: 1,034; repeat-expanded tickets: 1,040.
- Final counts: 1,040 rollout groups, 8,320 sampled proofs, 4,160 optimizer steps.
- Compile successes: 367/8,320 (4.4111%).
- Final adapter: `/home/scc/bz22001004/runs/grpo-first5000-epoch2-20260829-v1/full/model`.

The 3,966 physical rows with `repeat=0` remain in the immutable first-round manifest but were not sampled. This preserves the selected 5,000-row view while applying the conservative repeat policy exactly.

## Data and source assembly

- Projection output: `data/grpo_train5000_proof_free.jsonl`.
- Projection SHA-256: `a408a35821b43a9b5fa63f3e1a48f2c124c0a8b879f4de14442357496793b885`.
- Projection audit SHA-256: `9baf78e7c54d95566197a58fc488cb45d9c230e5e0a327993d8a673206542fe3`.
- Required training fields are SFT-like (`prompt`, statement, imports/context and identity metadata), but all proof-bearing fields are absent.
- Prompts end at the open theorem assignment `:=` and never inject `by sorry`.
- Invalid non-Lean import lines found during projection: 30, rejected rather than propagated.
- Duplicate import/namespace prefixes after canonical assembly: 0.
- Runtime Import/module failures: 0; duplicate-declaration/prefix failures: 0.

## Reward and rollout behavior

- Reward is exactly 1 only when Pantograph compilation succeeds and the completion contains no `sorry`/`admit`; otherwise it is 0.
- Explicit raw completion `sorry`/`admit` rejections: 3.
- No partial score, tactic bonus, format bonus or shaping reward is used.
- vLLM colocated rollout is used with 8 generations per prompt.
- A vLLM 0.27 disk-reload incompatibility that would overwrite synchronized LoRA weights was guarded; four projection tensors were checked for exact sync parity before rollout.
- NF4/4-bit LoRA merge produced corrupted rollouts in diagnostics, so the accepted run uses BF16 base weights. This is the configuration validated on RTX 5090.

## Pantograph lifecycle

- One worker is prewarmed per DDP rank before the first rollout and remains resident for the whole run.
- Persistent worker PIDs were unchanged across all 8,320 samples:
  - rank 0: `441810`
  - rank 1: `461626`
  - rank 2: `478836`
  - rank 3: `495717`
- No worker restart, fatal pool error, NCCL error, OOM, NaN/Inf or Python traceback occurred in the accepted full run.

## Curves

- Non-zero loss steps: 396/4,160. Zero-loss steps correspond to rollout groups with uniform all-zero rewards.
- Loss mean: 0.008725; range: [-0.871774, 2.294945].
- Overall mean rollout reward: 0.044111.
- First 100 rollout groups: 0.017500 mean reward.
- Last 100 rollout groups: 0.098750 mean reward.
- The final-window reward is 5.64 times the initial-window reward, while remaining a noisy single-epoch training curve.

## Timing

- Trainer runtime: approximately 17,540 seconds (4 h 52 m 20 s).
- Estimated vLLM generation: 7,534.14 s (43.0%).
- Approximate Pantograph compile wall time: 6,885.09 s (39.3%).
- Estimated optimizer update time: 2,747.96 s (15.7%).
- Remaining orchestration/checkpoint overhead: approximately 372.82 s (2.1%).

These components are derived from per-step and per-reward timestamps; compile work runs concurrently across the four ranks, so they are reported as wall-time estimates rather than summed CPU time.

## GPU memory

- GPU 0: mean 21,692 MiB, peak 31,386 MiB, mean utilization 60.9%.
- GPU 1: mean 21,521 MiB, peak 30,896 MiB, mean utilization 57.9%.
- GPU 2: mean 21,254 MiB, peak 30,906 MiB, mean utilization 63.3%.
- GPU 3: mean 21,084 MiB, peak 30,896 MiB, mean utilization 61.8%.

Four RTX 5090 cards were sufficient, although GPU 0 had only about 1.2 GiB peak headroom. The A100 fallback was not needed.

## Post-training compile smoke

- Job `48204`: completed after correcting the standalone smoke wrapper's Lean PATH and rank-suffixed reward-log path.
- Samples: 8 from a fixed P0 boundary prompt.
- Compile successes: 6/8.
- Generation: 8.77 s; Pantograph wall time including warmup: 78.27 s.
- Single-card peak allocated memory: 14,369 MiB.
- Pantograph worker startup: 5.74 s, post-startup warmup: 0.19 s, restarts: 0, fatal errors: 0.
- Result SHA-256: `5fad30c60b78b4fb2c4d7f77339f6e76133d20ef29c316aece7d2c44ba79ba9f`.

The two earlier wrapper failures were non-model test-harness issues and were retained in the logs; the final corrected smoke run passed.

## Verification artifacts

- Final validation: `audit/full-47990-final-validation.json` (`8b5d504a7ee132796b20b6a6169450ffd7fff4503596ac4c05871d3252d413ac`).
- Machine-readable run summary: `audit/full-47990-summary.json` (`0f26b033627f7e0d761c6b88b179b9edb5b7701b8d0bac92e5f08c3d3979ad5f`).
- Post-training smoke: `audit/posttrain-8sample-compile-smoke.json` (`5fad30c60b78b4fb2c4d7f77339f6e76133d20ef29c316aece7d2c44ba79ba9f`).
- Final adapter weights SHA-256: `0fac7c34af3895c0cc37f6469a418d03def3033af9d912ff490d613742749a49`.
- Unit/integration regression tests: 8/8 passed.
