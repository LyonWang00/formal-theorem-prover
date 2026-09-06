# Training data contracts

This package contains the strict data models shared by the SFT, GRPO, and
miniF2F evaluation paths. It intentionally excludes planning and proof-repair
workflow contracts.

- `SFTGeneralData` is the audited manifest form of a verified proof.
- `SFTData` is the minimal prompt/completion projection consumed by SFT.
- `GRPOGeneralData` and `GRPOData` contain statements only; proof completions
  are generated online and scored by Lean/Pantograph.
- `EvaluationData` represents benchmark inputs.
- `compile_errors.py` provides the shared Lean diagnostic taxonomy.
