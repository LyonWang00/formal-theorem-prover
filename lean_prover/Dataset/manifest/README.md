# Final dataset manifests

This directory is reserved for final, Pantograph-verified, cross-source
materializations. Generated JSON/JSONL payloads are intentionally ignored by
Git; this README records their contract.

The manifest layer is model-independent. It preserves enough information to
reconstruct and revalidate Lean source rather than storing only the two visible
training strings.

## SFT manifest contract

Each `sft_manifest` row is an `SFTGeneralData` record and contains at least:

- `schema_version = "sft_manifest"`;
- `record_id`, `source`, and provenance metadata;
- proof-free `lean_statement` and complete verified `proof`;
- `imports`, `context_lines`, and `unknown_preamble_lines`;
- normalized `statement_hash` and `proof_hash`;
- `pantograph_verified = true` and `verification_scope = "full_proof"`;
- environment/attestation fields needed to reproduce verification.

SFT deduplication uses `(statement_hash, proof_hash)`. Different verified
proofs of the same statement are retained.

## GRPO manifest contract

Each `grpo_manifest` row is a model-independent GRPO record and contains at least:

- `schema_version = "grpo_manifest"`;
- `record_id`, `source`, and provenance metadata;
- proof-free main `lean_statement`;
- optional supporting `imports` and `context_lines`;
- normalized `statement_hash`;
- `pantograph_verified = true` and `verification_scope = "statement_only"`.

The main theorem has no proof. Supporting lemmas in `context_lines` may use
`sorry` for statement-only Pantograph validation. GRPO rows never contain
reference proof, completion, or proof-derived metadata. GRPO deduplication uses
`statement_hash` only.

## Training projection

Manifest rows are not passed directly to a model. A model-specific finalizer
projects SFT manifests into `SFTData` under `../final_data/`, a two-column JSONL whose rows are exactly
`{"prompt": ..., "completion": ...}`. The prompt is self-contained Lean source
ending in `:= by sorry`; completion is the verified proof beginning with `by`
(or another valid Lean proof term). Expert-iteration SFT may add only
`sample_weight`. GRPO manifests are projected to proof-free prompts. Schema
versions, Lean source fields, provenance, hashes, Pantograph receipts, length
statistics, and duplicate reports remain in manifests or dataset-level
sidecars; they are never duplicated into trainer rows. Tokenizer-specific EOS
handling also stays in the training preprocessor rather than the canonical
manifest.

The SFT trainer validates the rich sidecar against the minimal projection
before tokenization. By convention `train.jsonl` is paired with
`train.manifest.jsonl`; explicit manifest paths are also supported.
