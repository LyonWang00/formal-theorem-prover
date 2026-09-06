# Initial GRPO: miniF2F-test pass@32 = 62.30%

This bundle records the first GRPO experiment using learning rate 1e-5, no KL
penalty (`beta=0`), and four policy iterations on four RTX 5090 GPUs.

- `results/analysis.json` is the corrected authoritative analysis.
- `results/REPORT.md` gives the human-readable evaluation report.
- `config/` contains the GRPO configuration, cluster batch scripts, training
  completion summaries, and final training report.
- `logs/full-47990.out.gz` and `.err.gz` preserve the full cluster training
  scheduler logs. They are compressed only to remain comfortably within
  GitHub's per-file limit.
- `logs/full-48225.*` and `logs/generate-shard*` record pass@32 generation;
  `logs/grouped-compile-48392.*` records the corrected grouped compilation.

The evaluation generated 7,808 attempts over 244 problems, with 2,956 compile
successes and 152/244 prefix pass@32 problems. This historical evaluation used
its original 180-second per-attempt compile timeout.

Run `sha256sum -c ARTIFACTS.sha256` from this directory to verify the bundle.
