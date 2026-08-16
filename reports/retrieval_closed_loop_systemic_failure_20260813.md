# Retrieval closed-loop implementation and stop report (2026-08-13)

## Outcome

The offline Mathlib index, Pantograph diagnostic classifier/retriever, and the
Planner retrieval-generate-compile loop were implemented and passed local
regression tests.  A normal end-to-end miniF2F smoke case also passed.

The directed smoke case that deliberately required a formalization repair then
exposed a system-level index coverage defect.  Per the evaluation stop rule,
testing stopped immediately.  The Planner 100-row rerun, the Prover repair-loop
upgrade, and the final Prover 100-row rerun were not started.

## Implemented components

### Offline declaration index

- SQLite cache key: `mathlib_commit/environment_hash`.
- Current cache:
  `5e932f97dd25535344f80f9dd8da3aab83df0fe6_46b005cc84cb6602c278fcfc596e51a03a5b9296bc7fda34f55d86ebfeb2c51a.sqlite3`.
- Current schema: `mathlib_declaration_index_v3`.
- Extracted declaration rows: 223,191.
- Stored fields: fully-qualified name, short name, declaration kind, type
  signature, namespace, module, required import, source path/line, first
  explicit parameter type, and inferred mathematical domain.
- Retrieval stages: exact name, namespace suffix/short-name match, fuzzy match,
  parameter-type scoring, and domain scoring.
- The parser tracks namespace/section scopes, nested comments, equation-style
  bodies, and rejects anonymous-instance fragments that are not declaration
  names.

### Error-driven local retrieval

The classifier supports:

- `unknown_identifier`
- `invalid_field_notation`
- `missing_import`
- `ambiguous_namespace`
- `application_type_mismatch`
- `unknown_constant`
- `parser_or_old_syntax`

It extracts the unavailable name, receiver expression/type when present,
expected/actual type when present, missing module, and current imports, then
retrieves up to ten local candidates.

### Planner formalization repair loop

- Default maximum formalization-repair rounds: five.
- Each round prompts with the exact Lean/Mathlib identity, root and node natural
  language, current failed statements, complete Pantograph/Verify diagnostics,
  dependency propositions, and indexed candidate signatures/imports/source
  locations.
- One model call produces one to three small node-patch candidates.
- Candidates are compiled sequentially with Pantograph.  The first full pass is
  selected; if none pass, the candidate with the most compiling nodes is kept
  so its diagnostics can drive the next round.
- A compiling repair returns to Verify in the next Planner iteration before it
  can reach Prover.
- The repair metadata retains the retrieval payload and each candidate's
  per-node compilation result.
- Planner still gates Prover on all subproblem statements compiling.

## Test evidence before the stop

### Local tests

- Broad related regression: 90 passed.
- Final targeted regression after index parser refinements: 19 passed.
- No diff whitespace errors were reported for the changed files.

### Normal real miniF2F smoke

The first seeded row, `mathd_algebra_419`, completed the genuine Lean-native
pipeline:

- Planner: success
- generated subgoals: 2
- ordinary nodes proved: 2/2
- ROOT attempted after dependencies: yes
- ROOT pass@4: 4/4
- final pipeline: success
- model: `deepseek-v4-flash`
- key source: Windows-side `DEEPSEEK_API_KEY`

This case did not need formalization repair.

A follow-up small run produced three unique problems with Planner success on
all three; one reached full Prover success and two were ordinary proof failures.
The output directory contains a duplicate first row because an outer command
timeout did not terminate its WSL child before a second invocation.  Therefore
that directory is not suitable for aggregate comparison.

## Systemic failure that stopped evaluation

The directed repair smoke started from:

```lean
lemma L1 (n : Nat) : Nat.obsolete_add_zero n = n
```

Pantograph correctly reported an unknown identifier.  The repair model then
generated a semantically correct current statement and listed `Nat.add_zero`
as a Mathlib hint.  The repair boundary rejected it before candidate
compilation:

```text
ValueError: formalization repair invented Mathlib hints not present in the
local index: ['Nat.add_zero']
```

### Confirmed root cause

`Nat.add_zero` is real in the exact local Lean environment, but it is defined
in Lean Core rather than the Mathlib source tree:

```text
/home/lean/.elan/toolchains/leanprover--lean4---v4.29.1/
src/lean/Init/Core.lean:745:
@[simp] protected theorem Nat.add_zero (n : Nat) : n + 0 = n := rfl
```

Consequently:

1. the requested Mathlib-source index cannot contain every declaration that is
   legal in a Mathlib environment;
2. the repair prompt's local candidates omitted `Nat.add_zero`;
3. the strict allow-list rejected a valid Core declaration;
4. querying the stale name `Nat.obsolete_add_zero` returned unrelated fuzzy
   Mathlib candidates rather than the needed Core declaration.

This is an index-coverage/system-contract defect, not a theorem/proof failure.

## Required next correction (not applied after the stop)

The practical correction is to add a companion environment-declaration index
covering the exact pinned Lean toolchain's `Init` and `Std` source declarations,
with an origin field such as `lean_core`, `std`, or `mathlib`.  Core declarations
should record their source module and that they require no additional Mathlib
import.  Retrieval should merge Core/Std and Mathlib candidates before applying
the same type/domain ranking.  The strict allow-list must validate against this
merged environment index, not against Mathlib-only rows.

After that correction, the directed repair smoke must pass before restarting
the small miniF2F gate or either 100-row experiment.

## Follow-up resolution

The requested companion index was subsequently implemented:

- Lean Core/Std cache key: `lean_commit/environment_hash`.
- Core/Std declaration rows: 39,390.
- Mathlib declaration rows after adding origin metadata: 223,191.
- Origins are explicit: `lean_core`, `lean_std`, or `mathlib`.
- Core declarations use an empty required import; Std and Mathlib declarations
  retain their precise source module/import.
- Unified search merges both SQLite indexes before final type/domain ranking.

The generic-context crowding bug was also corrected: candidates retrieved from
the compiler's unavailable name now have priority, while statement/prose terms
only fill unused candidate slots.

The same directed smoke then completed in one repair round:

- indexed candidate: `Nat.add_zero`
- origin: `lean_core`
- source: `Init/Core.lean:745`
- repaired statement: `lemma L1 (n : Nat) : n + 0 = n`
- candidate C1 Pantograph compilation: passed
- dependency Verify: passed
- formal-statement semantic Verify: passed
- final smoke status: success

The post-fix broad related regression completed with 95 passed tests.
