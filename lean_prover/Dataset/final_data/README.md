# Final model-facing datasets

This directory contains the generated JSONL files consumed directly by
training and evaluation. The payloads are intentionally excluded from Git.

- `sft_data.jsonl`: minimal `{"prompt", "completion"}` SFT rows projected
  from `../manifest/sft_manifest.jsonl`.
- `grpo_data.jsonl`: proof-free GRPO prompts projected from the unified GRPO
  manifest. It must never contain a reference proof or completion.
- `minif2f_data.jsonl`: all 488 Pantograph-verified miniF2F benchmark prompts,
  preserving the original `valid` and `test` split labels.

Rich provenance, hashes, deduplication information, and verification receipts
belong in `../manifest/`; rollout and benchmark outputs belong in
`../experiment_result/`.
