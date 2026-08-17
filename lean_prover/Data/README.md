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

`normalization.py` is the stable facade over `lean_training/data/adapters/`:

- LeanWorkbook SFT/trajectory reconstruction uses `adapters/lean_workbook.py`;
- NuminaMath full-proof rows use `adapters/numinamath.py` for SFT;
- NuminaMath and Kimina statement-only rows use `adapters/numinamath.py` and
  `adapters/kimina.py` for GRPO;
- verified miniF2F valid/test rows use `adapters/minif2f.py` and are benchmark-only;
- LeanDojo reconstruction uses `adapters/leandojo.py`.

GRPO adapters reject proof-bearing input fields and GRPO training projection
never reads or emits reference-proof hashes or lengths.
