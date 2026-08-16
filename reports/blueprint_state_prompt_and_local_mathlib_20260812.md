# BluePrintRepair state prompt and local Mathlib mitigation (2026-08-12)

## State prompt change

The prompt now mandates this exact execution order:

1. snapshot the current input;
2. inspect all format, DAG, logical, and semantic conditions;
3. concretely repair every detected defect in the returned Blueprint;
4. compare returned Blueprint data with the snapshot;
5. assign state last, then serialize mandatory state as the first JSON key.

Valid state pairs are only:

- `success` plus an unchanged Blueprint;
- `failed` plus a visibly changed, repaired Blueprint.

Malformed/partial model output has no valid state and is recorded with
`state=null` and `output_error`; it is not mislabeled as a failed repair.
Each audit row additionally stores `input_snapshot` and deterministic
`changed_fields`.

## Real results

- Three-example full pipeline: four valid BluePrintRepair rounds, comprising
  three `success + unchanged` and one `failed + changed`; zero
  `failed + unchanged`.
- Two-example confirmation: two `success + unchanged`; zero state violations.
- Directed defect test: clearing `root_dependencies` caused round one to add
  `nodes[1].depends_on[0]` and `root_dependencies[0]` with state `failed`;
  round two made no change and returned `success`.
- Final focused regression: 69 passed.

## Local Mathlib facts

The active Mathlib commit is
`5e932f97dd25535344f80f9dd8da3aab83df0fe6`. The local checkout contains 7,871
Lean source files plus built olean artifacts. This is sufficient for offline
declaration/module retrieval and compile-driven validation without web search.

## Recommended implementation sequence for version-mismatch failures

1. Build an offline declaration index pinned to the environment hash. Parse
   local Mathlib sources and/or Lean environment exports into records containing
   fully qualified name, kind, type signature, namespace, defining module, and
   source location. Cache it by Mathlib commit. Query exact names first, then
   suffix/fuzzy names. Feed only a small ranked result set to repair prompts.
2. Add a deterministic compatibility diagnostic before LLM repair. Classify
   Pantograph messages such as unknown identifier, invalid field notation,
   missing import, type mismatch, and ambiguous namespace. Extract the bad name
   and receiver type. For invalid field notation, query the index for functions
   whose first explicit argument accepts that receiver type; prefer explicit
   qualified calls such as `Nat.factorization n` over `n.factorization`.
3. Make statement repair tool-assisted and compile-in-the-loop. Supply the exact
   local environment identity, Pantograph error, current declaration, dependencies,
   and retrieved local declarations/modules. Generate a small number of repaired
   statements and Pantograph-check each. Never accept a replacement only because
   the model says it is equivalent; Verify must recheck semantic alignment.
4. Maintain a deterministic compatibility map learned only from compiled local
   evidence. Key entries by old spelling/error fingerprint and Mathlib commit;
   value should include replacement spelling, required import, type constraints,
   and a minimal compile-tested example. Apply exact high-confidence mappings
   before invoking an LLM, and invalidate the cache when the environment hash
   changes.
5. Build an offline retrieval corpus from the pinned checkout: declaration
   signatures, module docstrings, nearby examples/tests, and successful project
   proofs. Use local BM25/SQLite FTS or a small local embedding index. Retrieve by
   mathematical intent plus bad identifier plus expected type. This gives the
   model current-library evidence without exposing all 7,871 files in a prompt.
6. Add a current-environment prompt pack generated from the index: common
   namespaces, canonical fully qualified APIs, field-notation cautions, imports,
   and several Pantograph-verified few-shots for frequently failing domains.
   Regenerate it whenever the Mathlib commit changes.
7. Record and reuse every successful repair as commit-scoped telemetry. Track
   failure category, bad code, retrieved candidates, accepted replacement, and
   compile result. Promote only repeatedly successful patterns into the
   deterministic compatibility map; do not learn from uncompiled model output.

The highest-value first increment is items 1-3: a commit-pinned declaration
index, compiler-error-driven retrieval, and Pantograph-checked statement repair.
It directly addresses nonexistent/renamed APIs while remaining offline and
bounded in latency.
