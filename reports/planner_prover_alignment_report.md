# Planner / Prover Alignment Report

Date: 2026-08-06

## Result

The Planner-to-Prover path is aligned with the current Blueprint contract. The
runtime Prover is API-only. Natural-language and Lean inputs now pass through a
mandatory API classification gate before following their respective Planner
routes.

## Input routing

### Natural-language input

1. API input-kind classification gate.
2. Natural-language task decomposition. Node Lean statements are empty at this
   stage.
3. Target theorem formalization and Pantograph declaration check.
4. Per-node formalization, including complete preamble and semantic-alignment
   evidence.
5. Per-node Pantograph declaration checks.
6. API-only Prover generation and Pantograph proof verification.

### Lean input

1. API input-kind classification gate.
2. Lean-to-natural-language translation.
3. Target theorem Pantograph declaration check.
4. Natural-language task decomposition.
5. Per-node formalization and Pantograph declaration checks.
6. API-only Prover generation and Pantograph proof verification.

The classifier is a gate, not a local heuristic. Its normalized decision must
match the supplied input, and invalid or proof-bearing Lean declarations are
rejected before decomposition.

## Schema alignment

The Prover request now consumes the current Planner fields directly:

- `problem_id`, `input_hash`, and `problem_hash`;
- node `lean_statement` rather than the obsolete `lean_decl` field;
- complete node preamble: imports, namespaces, section, variables, local
  notation, and open namespaces;
- available definitions;
- dependency declarations, proofs, and their own preambles;
- semantic-alignment evidence;
- estimated proof length and its evidence;
- exact Lean, Mathlib, project, and environment hashes.

The API prompt contains both structured preamble data and a materialized Lean
context. Proof verification uses the same shared preamble renderer as Planner,
which removes the previous prompt/compiler context mismatch.

## API-only Prover

The runtime exports only the OpenAI-compatible API proof generator and the
Blueprint proof orchestrator. The local 7B generator, 7B capability
decomposition, and 7B candidate-evaluation paths are not exported or reachable.

## Stable identity

`input_hash` is a canonical SHA-256 over the normalized raw input and declared
input kind. `problem_hash` is derived from `input_hash` when a raw routed input
is available, so API wording or formalization variance does not change the
deduplication key. Both values are propagated through Planner results,
Blueprints, Prover requests, telemetry, and pipeline results.

For callers that construct a `TheoremProblem` directly, a deterministic fallback
problem hash is computed from the normalized natural statement, Lean target,
imports, and available definitions.

## Prover result collection audit

No new failure taxonomy or result-classification policy was introduced in this
task. Existing behavior is preserved:

- the Prover returns an updated Blueprint;
- each node records `proof_attempts`, `lean_feedback`, `status`,
  `verified_proof`, and metadata;
- telemetry distinguishes `generation_failed`, `generated_unverified`,
  `proved`, and `verification_failed`;
- dependency-blocked nodes record `metadata.skipped_reason`;
- an unverified proof is never promoted to `verified_proof`.

Current limitation: result collection is node-level. It does not separately
generate, verify, or classify a final proof for the Blueprint `ROOT` target, and
it does not yet map raw Pantograph messages into a fine-grained failure taxonomy
such as syntax, elaboration, unknown identifier, tactic error, or unsolved goals.
Those behaviors were intentionally left read-only as requested.

## Verification

### Automated tests

- Planner, Prover, and end-to-end routing tests: **37 passed**.
- Backend and proof-search regression tests: **10 passed, 1 conditionally
  skipped**.

### Real Pantograph checks

Five API-adapter Prover scenarios passed with real Pantograph verification:

- short, one-node proof;
- medium, three-node dependency chain;
- long estimated proof;
- fine-grained, six-node dependency chain;
- structured preamble with local notation.

Both full input routes also passed with deterministic mock API responses and
real Pantograph compilation:

- natural-language input: four Planner API calls, target and node declaration
  gates passed, Prover proof verified;
- Lean input: four Planner API calls, translation and declaration gates passed,
  Prover proof verified.

No external paid API call was made during these deterministic integration tests.

## Remote baseline

The remote `origin/prover` baseline at commit
`dd580ee644bc0fd178b57ddb1c60ac3a6dcb6a4c` was inspected and tested in an
isolated worktree before migration. Its 20 unit tests passed, but its schema used
the obsolete `lean_decl` field and omitted the current preamble, alignment,
length-estimate, and environment contracts. The temporary worktree was removed
after the comparison; the remote reference remains available.
