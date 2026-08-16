# API-only Prover

The Prover consumes the current Planner `TheoremProblem` and validated
`Blueprint` objects. It has one proof-generation path:

```text
OpenAI-compatible API -> proof-body extraction -> optional Pantograph check
```

There is no local 7B adapter, 7B capability decomposition, or 7B capability
evaluation layer.

## Planner contract consumed by every node

- `problem_hash` and exact Lean/Mathlib environment;
- `lean_statement` (available internally as the compatibility attribute
  `lean_decl`);
- complete structured `preamble`;
- problem and node imports, including dependency imports;
- available local definitions;
- dependency declarations with their own preambles;
- `semantic_alignment`;
- `estimated_proof_length`;
- proof strategy and Mathlib hints.

The prompt includes both the structured preamble JSON and a materialized Lean
context. Verification uses the same preamble renderer as Planner.

## Result collection and routing

The result collector remains node-oriented:

- generated proof bodies are appended to `node.proof_attempts`;
- verifier feedback is appended to `node.lean_feedback`;
- node status becomes `proving`, `proved`, or `failed`;
- a successful verified body is stored as `node.verified_proof`;
- optional JSONL telemetry records the `problem_hash`, request, prompt, proof,
  status, and diagnostics.

Attempt telemetry retains `generation_failed`, `generated_unverified`,
`proved`, `verification_failed`, `repair_proved`, and
`repair_verification_failed`. In addition, every attempt and node receives the
same top-level compiler taxonomy used by training: `success`,
`extraction_error`, `syntax_error`, `elaboration_error`, `tactic_error`,
`unsolved_goals`, `timeout`, `forbidden_token`, `environment_error`, or
`internal_error`. A secondary detail preserves signals such as
`unknown_identifier` and `type_mismatch`.

`BlueprintProver.prove()` produces a `ProverProblemResult`. The result is routed
to success only when every Blueprint subproblem has a Pantograph-verified proof;
otherwise it is routed to failure. This outcome is explicitly separate from
Planner success/failure. `JsonlProverResultStore` can persist the two streams to
independent JSONL files.

The canonical implementation is contained in this `Prover/` package. The old
top-level `lean_prover.prover` module is only a compatibility facade.
