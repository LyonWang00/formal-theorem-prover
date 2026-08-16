# Unified Data contracts

This package owns the project-level schemas for:

- one raw natural-language or Lean inference input before Planner;
- the existing Planner `Blueprint`, wrapped with an explicit Planner outcome;
- the node-level and problem-level Prover result, with independent Prover
  success/failure routing;
- SFT, EI, GRPO, and evaluation data roles.

GRPO inputs intentionally contain a theorem statement and no proof target. SFT
records require both a Lean statement and verified proof. EI records explicitly
record whether a supplied proof was Pantograph-verified.

`normalization.py` is a stable facade over the existing LeanWorkbook, miniF2F,
and generic normalizers. Their implementation and behavior remain in
`lean_training/data/preparation.py`.
