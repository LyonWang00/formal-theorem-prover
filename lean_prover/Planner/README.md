# Planner package

`lean_prover.Planner` accepts either a natural-language theorem or a proof-free
Lean theorem/proposition and turns it into a validated `Blueprint`.

Every raw input first passes through an API classification gate:

- natural language mode: decompose to a natural-language `BlueprintPlan`,
  formalize the target, then formalize nodes;
- Lean mode: validate the supplied target, call the dedicated Lean-native
  decomposition API, and return already formalized Lean subproblem nodes.

Natural-language mode is a two-stage pipeline followed by an atomic statement
gate:

1. **Decomposition** produces a natural-language-only `BlueprintPlan`. Every
   node has a proof-length estimate and an empty `lean_statement`.
2. **Formalization** preserves that frozen graph and fills each node's
   `lean_statement`, semantic-alignment audit, and complete structured preamble
   using the exact local Lean/Mathlib commits.
3. **Pantograph gate** compiles every node statement. The whole problem succeeds
   only when every node succeeds; failures retain explicit node IDs and
diagnostics.

Lean mode does not translate to natural language and does not call the
natural-language decomposition prompt. `LeanDecompositionAPI` returns the
complete existing `Blueprint` schema directly; the target and every generated
subproblem declaration must pass Pantograph.

Proof-body synthesis remains a later layer.

## Main files

### `schemas.py`

Defines the Pydantic data models shared by the Planner pipeline.

- `TheoremProblem`: input theorem problem, including imports and target Lean declaration.
- `RawTheoremInput`: unclassified natural-language or Lean input with stable
  `input_hash`.
- `InputClassification`: mandatory API-gate result (`natural_language` or
  `lean`).
- `BlueprintPlan`: stage-one natural-language graph; `lean_statement` is always empty.
- `Blueprint`: complete stage-two output with environment identity and formalized nodes.
- `BlueprintNode`: one planned sub-theorem/lemma statement.
- `LeanPreamble`: imports, namespaces, open declarations, variables, and other
  proof-free local context required by a node.
- `ProofLengthEstimate`: expected Lean proof lines/tokens used to control
  decomposition granularity.
- `LeanCheckResult`: normalized result returned by Lean/Pantograph checks.
- `ValidationIssue` and `BlueprintValidationResult`: structured validation errors.
- `PlannerResult`: top-level service result shape.

`TheoremProblem`, `BlueprintPlan`, and `Blueprint` carry a stable
`problem_hash`. Routed problems derive it from `input_hash`, so API wording
changes during translation/formalization do not change the deduplication key.

This file is the source of truth for the JSON shape expected from the LLM.

### `prompts.py`

Contains separate system prompts for natural-language decomposition,
Lean-native decomposition, and formalization. Both decomposition prompts
explicitly request zero-shot step-by-step reasoning while requiring the model
to return only the contracted JSON.

Important responsibilities:

- describe the required JSON output format;
- enforce hard constraints such as “do not include proof bodies”;
- build the theorem-specific user prompt from a `TheoremProblem`.

### `client.py`

Provides the LLM client abstraction and configuration helpers.

Important components:

- `PlannerClientConfig`: configuration for API key, base URL, model, temperature, and max tokens.
- `OpenAICompatibleClient`: client for OpenAI-compatible chat completion APIs.
- `create_deepseek_client_from_env()`: convenience constructor for the configured DeepSeek-compatible client.

This module is responsible for making the model call and parsing the model
response as JSON.

### `validator.py`

Performs static, non-Lean validation of a generated `Blueprint`.

Typical checks include:

- node IDs and dependency references are well formed;
- dependency graph references existing nodes;
- root dependencies are valid;
- nodes do not obviously restate the target theorem when that is disallowed;
- schema-level constraints are respected.

These checks are fast and do not start Lean.

### `graph.py`

Contains helper logic for reasoning about blueprint dependency graphs.

This is where graph-level operations belong, such as dependency lookup,
topological ordering, or cycle detection.

### `lean_checker.py`

Defines the generic blueprint-to-Lean checking flow.

`BlueprintLeanChecker` walks through all blueprint nodes, merges problem and
node imports, supplies each structured preamble, and records a
`NodeLeanCheckResult` for every node. It does not stop at the first failure.

This module is backend-agnostic: it does not know whether the backend is
Pantograph, Lake, or another Lean checking mechanism.

### `pantograph_checker.py`

Implements the Pantograph-backed statement checker used by the current real
tests.

`PantographDeclarationCheckingBackend` checks each node declaration in an
accumulated Lean environment:

1. previous accepted declarations are given temporary `:= by sorry` bodies;
2. the current declaration is also given a temporary `:= by sorry` body;
3. Pantograph is asked whether the combined source elaborates.

This lets Planner check not only whether a node statement is legal in isolation,
but also whether it is legal after earlier blueprint nodes have been introduced.

The temporary `sorry` bodies are only used inside the checker; they are not
stored in the blueprint.

### `repair.py`

Compatibility facade for `lean_prover.Repair.planner_subproblem`, which repairs
only failed formalized subproblem nodes and freezes unaffected nodes.

### `service.py`

Provides a higher-level orchestration surface for running Planner components
together.

This is the natural place for application-facing functions that combine:

1. mandatory input-kind classification API call;
2. routing to the dedicated Lean-native decomposition API for Lean input;
3. natural-language decomposition API call for natural input;
4. target formalization after decomposition when the input is natural language;
5. `BlueprintPlan` schema/graph validation;
6. natural-language-to-Lean node formalization API call;
7. frozen-plan and semantic-contract validation;
8. all-node Lean/Pantograph checking;
9. optional formalization repair without changing stage-one decomposition.

### `exception.py`

Contains Planner-specific exception types.

Use this file for errors that should be distinguished from generic Python,
Pydantic, API, or Lean backend exceptions.

### `__init__.py`

Marks `Planner` as a Python package and provides package-level exports.

Internal modules should generally use relative imports, for example:

```python
from .schemas import Blueprint
```

Code outside the package can import through the package path, for example:

```python
from lean_prover.Planner.schemas import Blueprint
```

## Subdirectories

### `tests/`

Contains unit and interface tests for Planner modules.

These tests are intended to be lightweight and mostly avoid real network or
Lean backend calls unless the test name explicitly targets such integration
behavior.

### `examples/`

Contains manually run examples and integration scripts.

`examples/real_test.py` is the current end-to-end script for:

1. constructing a real `TheoremProblem`;
2. calling the configured LLM;
3. validating the returned `Blueprint`;
4. checking each generated Lean declaration with Pantograph;
5. printing a JSON report.

Because it calls an external model API and starts the Lean/Pantograph backend,
it is intentionally not treated as a normal fast unit test.

## Typical flow

```text
RawTheoremInput
    -> API input-kind gate
    -> natural mode: NL BlueprintPlan -> target + node formalization
    -> lean mode: dedicated Lean-native Blueprint decomposition
    -> static frozen-plan validation
    -> Pantograph check for every node
    -> PlannerResult(success only when all nodes pass)
```

## Current design boundary

Planner currently checks Lean declarations at the statement level. A valid node
should look like:

```lean
lemma L1 (n : Nat) : n = n
```

It should not include a proof body such as:

```lean
lemma L1 (n : Nat) : n = n := by
  rfl
```

The checker may temporarily add `:= by sorry` internally so Lean can elaborate
the declaration, but generated blueprints should not contain proof bodies.
