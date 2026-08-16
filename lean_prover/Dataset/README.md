# Dataset storage

This directory is the project-level dataset registry next to `Planner`.

## Layout

- `raw_data/leanworkbook_raw.parquet`: immutable Hugging Face source artifact
  for `InternLM/Lean-Workbook` at commit
  `2e066e310b2c6d2c27616927ae131f82901c8f1c`.
- `raw_data/leandojo_raw.jsonl`: immutable 5,366-record LeanDojo-v2/current-
  mathlib candidate input used by the current Pantograph verification build.
- `verified_data/leanworkbook_verified_success_1.jsonl`: the existing frozen
  canonical LeanWorkbook success pool.
- `verified_data/leandojo_verified_success_1.jsonl`: the existing frozen
  canonical LeanDojo success pool.
- `verified_data/leanworkbook_verified_success_2.jsonl`: successful records
  outside the first frozen LeanWorkbook pool.
- `verified_data/leanworkbook_verified_fail_2.jsonl`: failed records outside
  the first frozen LeanWorkbook pool, including Pantograph diagnostics.

Every verified-data row has:

- `pantograph_verified`: exactly `success` or `fail`;
- `source` identifying the dataset of origin;
- `record_hash`, `statement_sha256`, and `proof_sha256`;
- `error_message` when `pantograph_verified == "fail"`.

`source` is the only canonical origin field. The historical misspelling
`sourcce` is rejected by the schema validator and removed during
materialization.

The formatter preserves all existing source fields. It does not rewrite the
older frozen artifacts under `outputs/`.

## Commands

Materialize raw inputs and the two first frozen success pools:

```bash
python -m lean_prover.Dataset.build_verified_datasets \
  --dataset-root lean_prover/Dataset materialize \
  --leanworkbook-raw /path/to/lean-workbook.parquet \
  --leandojo-raw outputs/leandojo_v2_retrace/processed/current_mathlib_candidates.jsonl \
  --leanworkbook-verified outputs/leandojo_v2_dataset_build/final_pool/lean_workbook_canonical_candidates.jsonl \
  --leandojo-verified outputs/leandojo_v2_dataset_build/final_pool/leandojo_v2_final_train_candidates.jsonl
```

Finish missing theorem-level LeanWorkbook verification in the audited Lean
environment (one worker is the memory-safe default):

```bash
python -m lean_prover.Dataset.build_verified_datasets \
  --dataset-root lean_prover/Dataset verify-remaining-leanworkbook \
  --lean-project lean_project \
  --prior-results data/processed/lean_workbook_verified_v2/audit/verification_results.jsonl
```

The operation is restartable. New Pantograph receipts are appended to
`verified_data/leanworkbook_verification_results_2.jsonl`, and output JSONL
files are rebuilt deterministically from the receipts.

Audit every materialized row, recompute all hashes, and enforce the disjoint
13,517-theorem LeanWorkbook partition:

```bash
python -m lean_prover.Dataset.build_verified_datasets \
  --dataset-root lean_prover/Dataset audit
```

## External formal datasets

`verify_external_datasets.py` supports two pinned datasets:

- `AI-MO/NuminaMath-LEAN` revision
  `51fa67f1f647ae1ecd81eef9f19306aa8a7b3a94`: complete-proof verification.
  Empty proofs and `sorry`/`admit` placeholders are failures, even when Lean
  reports only a placeholder warning;
- `AI-MO/Kimina-Prover-Promptset` revision
  `3009c548d90160d0f5e963d72238610c6732f812`: target-proof-free schema gate
  followed by statement-only elaboration. Its canonical `:= by sorry` is an
  elaboration placeholder, so a warning-only result is a statement success.
  A Kimina success never claims that a proof was supplied or checked.

Pinned raw artifacts are `raw_data/numinamath_raw.parquet` and
`raw_data/kimina_raw.parquet`. Deterministic outputs are:

- `verified_data/numinamath_verified_success.jsonl`;
- `verified_data/numinamath_verified_fail.jsonl`;
- `verified_data/numinamath_invalid.jsonl` for rows whose statement is not a
  legal repair target.  These rows preserve the canonical source record and
  add only `invalid_type` (`missing_context`, `invalid_statement`, or
  `other_invalid_data`).

Remaining NuminaMath failures are routed by
`route_numinamath_failures.py` into six mutually exclusive repair duties:
local repair, missing context, task decomposition, timeout/resource retry,
invalid statement, and from-scratch proof.  Statement-only Pantograph checks
must precede the missing-context/invalid split.  Routed resource retries reuse
the original proof under finite audited limits; decomposition rows use the
Planner -> Blueprint -> Prover pilot in
`run_numinamath_decomposition_repair.py`.
- `verified_data/kimina_verified_success.jsonl`;
- `verified_data/kimina_verified_fail.jsonl`.

Run a schema gate or a restartable verification job:

```bash
python -m lean_prover.Dataset.verify_external_datasets kimina \
  --dataset-root lean_prover/Dataset --audit-only

python -m lean_prover.Dataset.verify_external_datasets numinamath \
  --dataset-root lean_prover/Dataset --lean-project lean_project
```
