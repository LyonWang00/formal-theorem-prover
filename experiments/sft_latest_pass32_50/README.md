# Latest SFT: miniF2F-test pass@32 = 50.00%

This bundle records the completed SFT run based on
DeepSeek-Prover-V1.5-Base, 75,344 verified training rows, LoRA r32/alpha64,
learning rate 2e-4, and two epochs on four RTX 5090 GPUs.

- `results/FINAL_RESULT.json` is the concise authoritative metric summary.
- `results/analysis.json` contains per-problem and aggregate analysis.
- `config/` contains the run contract, checkpoint freeze receipt, data audit,
  and cluster batch scripts.
- `logs/train-*.{out,err}` are scheduler logs for the original and resumed
  training jobs.
- `logs/eval-*` and `logs/generate-shard*` are evaluation and four-shard
  generation logs; `logs/freeze-*` records checkpoint freezing.

The evaluation generated 7,808 attempts over 244 problems and compiled them
with four persistent workers using a 30-second attempt timeout. It achieved
2,113 successful attempts and 122/244 prefix pass@32 problems.

Run `sha256sum -c ARTIFACTS.sha256` from this directory to verify the bundle.
