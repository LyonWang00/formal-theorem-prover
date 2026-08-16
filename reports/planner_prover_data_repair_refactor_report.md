# Planner / Prover / Data / Repair Refactor Report

Date: 2026-08-06

## Outcome

The inference path now has two explicit Planner modes and one API-only Prover:

```text
single raw problem
  -> mandatory API input-kind gate
  -> natural-language Planner mode OR Lean-native Planner mode
  -> Pantograph-gated Blueprint
  -> API-only Prover
  -> independent Prover success/failure result
```

Planner and Prover outcomes are separate data fields. A Planner success means
that the target and every generated subproblem declaration passed the Planner
gates. A Prover success means that every Blueprint subproblem has a locally
Pantograph-verified proof.

## Planner modes

### Natural-language mode

1. Classify the raw problem with the mandatory API gate.
2. Call the natural-language decomposition prompt. It produces a
   `BlueprintPlan` with empty node `lean_statement` fields.
3. Formalize the root target.
4. Pantograph-check the root declaration.
5. Formalize all frozen plan nodes, including complete preambles, semantic
   alignment, and proof-length estimates.
6. Pantograph-check every node declaration.

### Lean-native mode

1. Classify the raw problem with the mandatory API gate.
2. Normalize and Pantograph-check the supplied proof-free Lean target.
3. Call the dedicated `LeanDecompositionAPI`. This API does not use the
   natural-language decomposition prompt and does not translate Lean to prose.
4. Receive a complete existing `Blueprint` whose subproblems are proof-free
   Lean theorem/lemma declarations with full preambles, alignment fields, and
   proof-length estimates.
5. Pantograph-check every generated node declaration.

Both decomposition prompts explicitly request zero-shot step-by-step reasoning
while requiring JSON-only output.

## Prover organization and outcomes

The canonical implementation is now under `lean_prover/Prover/`:

- `service.py`: API proof generation, verification, optional proof repair, and
  attempt telemetry;
- `results.py`: problem-level aggregation and success/failure JSONL routing;
- `tests/`: schema, result routing, repair, and Pantograph tests.

The old `lean_prover/prover.py` contains only a compatibility facade.

The Prover retains high-level attempt states and adds the exact top-level
verification taxonomy used by training:

- `success`;
- `extraction_error`;
- `syntax_error`;
- `elaboration_error`;
- `tactic_error`;
- `unsolved_goals`;
- `timeout`;
- `forbidden_token`;
- `environment_error`;
- `internal_error`.

A secondary detail preserves finer repair signals, including
`unknown_identifier`, `unknown_constant`, `missing_import`, `parser_error`,
`type_mismatch`, `application_type_mismatch`, and `failed_to_synthesize`.

`ProverProblemResult` contains final-node failure counts, all-attempt failure
counts, and fine-detail counts. `JsonlProverResultStore` routes complete records
to independent success and failure files.

## Unified Data schemas

`lean_prover/Data/` now contains strict schemas for:

- one raw natural-language or Lean inference problem before Planner;
- the unchanged existing Blueprint wrapped with Planner outcome metadata;
- node-level and problem-level Prover results;
- SFT data, which requires a Lean statement and verified proof;
- EI data, with explicit discovery/evaluation role and verification state;
- GRPO data, which intentionally contains a theorem statement and no proof;
- evaluation data, with an optional reference proof.

The existing LeanWorkbook, miniF2F, and generic normalization implementations
remain unchanged in `lean_training/data/preparation.py`. The new Data package
provides a stable router over those three functions. `.gitignore` contains a
narrow exception for the source-code `lean_prover/Data/` directory; other data
directories remain ignored.

## Repair modules

The canonical `lean_prover/Repair/` package distinguishes three responsibilities:

1. `PlannerSubproblemRepairer`: repairs failed Planner node formalizations and
   rejects changes to unaffected nodes.
2. `ProverProofRepairer`: calls the API using the failed proof and fine-grained
   compiler feedback. The replacement is never promoted until the Prover runs
   Pantograph again.
3. `DiscoveryAttemptRepairPipeline`: the relocated EI discovery RepairData v2
   pipeline. Its old `lean_training/repair_pipeline` path is a compatibility
   facade, so existing experiment scripts remain valid.

## Verification

- Focused Planner/Prover/Data/Repair/EI suite: **95 passed, 1 conditionally
  skipped**.
- Training-data, EI, backend, proof-search, and Pantograph regressions:
  **122 passed, 1 conditionally skipped**.
- Real Pantograph dual-mode Planner test: **1 passed**. Both natural-language
  and Lean-native routes passed the root and all-node declaration gates.
- Real Pantograph Prover-repair test: **1 passed**. The initial proof failed with
  an unknown identifier; the replacement entered success only after local
  recompilation.

All API behavior in the tests used deterministic fake clients. No external paid
API call and no training process was started.
