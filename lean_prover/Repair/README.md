# Repair modules

The three repair responsibilities are deliberately separate:

- `planner_subproblem.py`: repairs failed Planner subproblem formalizations and
  freezes unaffected Blueprint nodes;
- `prover_proof.py`: repairs one API-generated proof using its fine-grained
  Pantograph diagnostics; the Prover recompiles the replacement before use;
- `discovery_attempt/`: the relocated attempt-level EI discovery RepairData v2
  pipeline, including minimal and clean proof generation and Pantograph gates.

The old `lean_training/repair_pipeline` path contains compatibility imports so
existing experiment scripts continue to run.
