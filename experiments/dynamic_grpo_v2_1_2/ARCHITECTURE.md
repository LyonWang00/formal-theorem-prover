# Dynamic GRPO v2.1.2 architecture

## Source of truth

The immutable bundle `project_source_dynamic_grpo_v2_1_2` is the only
cross-machine source of truth.  Local, compute, and cloud copies must have the
same `PROJECT_VERSION.json`, relative file list, sizes, and SHA-256 hashes in
`SOURCE_MANIFEST.json`.

No historical round directory may overwrite this bundle.  Older scripts are
inputs only when their hashes are explicitly pinned as vendored dependencies.

## Execution split

- Local: source control, contracts, tests, launch and audit tooling.
- Compute: four-GPU rollout, GRPO updates, checkpointing, and miniF2F
  generation.
- Cloud: Lean compilation of frozen generation attempts and final statistics.

Runtime directories are derived views of the canonical bundle.  They may be
trimmed for a job, but any file with the same relative canonical role must
match the bundle hash.

## State boundaries

- Training commits only complete checkpoints with model, optimizer,
  scheduler, and four RNG-state files.
- Screening receipts affect selection only.
- Rewards use fresh current-cycle online rollouts only.
- miniF2F generation is frozen and hash-audited before cloud transfer.
- Cloud compilation is resumable from immutable receipts.

## Allowed endpoint differences

Only generated state may differ: checkpoints, merged weights, rollout output,
compile receipts, statistics, scheduler metadata, logs, caches, temporary
spools, and machine-specific environment files.  Architecture documents,
contracts, source, tests, and reusable launch/audit scripts may not differ.
