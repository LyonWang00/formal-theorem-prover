# RootFirst experimental pipeline

This directory is an opt-in experiment. It does not modify or register itself
with the existing `PlannerService`, `BlueprintProver`, or main pipeline.

The initial graph contains only immutable `L0`. `L0` receives pass@4, and each
failed independent attempt enters its own retrieval-guided proof-repair loop.
If `L0` remains unproved, `BlueprintRefinement` adds at most one helper in that
round. The experiment currently compiles at most three repair candidates per
independent base attempt (also at most three repair rounds) and caps graph
growth at two helper nodes; pass@4 remains unchanged.
Program-owned graph assembly validates reciprocal father/children edges,
acyclicity, reachability, and `L0` as the unique sink before Pantograph checks
the new statement. Proved nodes are frozen by content/proof fingerprint; later
rounds may only change their relationship edges. The result records the attempt
count at freeze, flags any post-freeze proof attempt, and reports repair,
proof-token, and tactic statistics per node and in aggregate.

A node is marked `disproved` only when Pantograph accepts a proof of the
program-constructed exact negation of its full proposition. A non-root
disproved node is revised without adding a node. `L0` is never revised because
changing it would no longer solve the input theorem.

Run tests from the repository root:

```text
PYTHONPATH=. .venv/bin/pytest -q lean_prover/Planner/RootFirst/tests
```

Run the fixed miniF2F manifest with an existing compiled Lean/Mathlib project:

```text
PYTHONPATH=. .venv/bin/python -m lean_prover.Planner.RootFirst.run_minif2f \
  --reference-manifest outputs/minif2f100_environment_prover_repair_v2_seed20260811/sample_manifest.jsonl \
  --lean-project /path/to/compiled/lean_project \
  --output-dir lean_prover/Planner/RootFirst/outputs/first10 \
  --count 10
```
