# BluePrintRepair implementation and first-10 evaluation (2026-08-12)

## Pipeline under test

`miniF2F formal_statement + informal_stmt + header` -> Lean-native Planner
decomposition -> BluePrintRepair (up to three stateless rounds) -> parallel Verify
per node -> Pantograph statement gate / subproblem repair -> Prover pass@4 per node
-> ROOT pass@4 after dependencies -> aggregation.

All ten records used deterministic Lean-first routing with
`deepseek-v4-flash` and the Windows-side `DEEPSEEK_API_KEY`.

## BluePrintRepair behavior

- Output is a complete Blueprint plus `state` (`success` or `failed`).
- `failed` means the input to that round required repair; `success` means no
  Blueprint data changed.
- A subsequent round receives only the previous output and no chat history.
- The loop stops on `success` or after exactly three rounds.
- Nonempty malformed JSON, schema-invalid repair output, and state-contract
  errors are retained and independently rechecked in the next round.
- Deterministic checks cover required JSON fields, node IDs/order, references,
  cycles, ROOT reachability/unique-sink equivalence, problem/environment identity,
  duplicate statements, declaration names, and required imports. The model prompt
  additionally checks mathematical/semantic coherence against natural statements.
- JSONL audit storage records every round, including raw invalid output and the
  exact contract/schema error when applicable.

Across the final ten-example evaluation there were 29 BluePrintRepair rounds:
1 `success`, 28 `failed`, 5 rounds with actual Blueprint changes, and 23 state
contract errors where the model returned `failed` without changing data. The
three-round bound allowed the pipeline to continue when the final Blueprint was
deterministically valid, but the model's state-label compliance is weak.

## Structural defects found during real testing

1. The first implementation stopped immediately if BluePrintRepair itself
   emitted a schema-invalid output. This violated the required self-loop. It was
   changed so the raw output is the next round's sole input and the invalid round
   is recorded as `failed`.
2. Verify could return a corrected preamble with an empty `raw_header`, replacing
   the authoritative dataset header. The Planner now preserves the pre-Verify
   raw header while accepting Verify's corrections to all other preamble fields.

Both fixes have regression coverage. The final focused suite passed 68/68 tests.

## Final first-10 results

The first four rows were rerun after the header fix. The final metrics combine
that post-fix four-row run with rows 5-10, which had already run after the fix.
Every final node retained the exact source header and all ten records verified
Lean-native routing.

| Problem | Planner | Nodes | Proved nodes | End-to-end | Attempts | Avg proof tokens | Avg tactics |
|---|---:|---:|---:|---:|---:|---:|---:|
| mathd_algebra_419 | yes | 2 | 2 | yes | 12 | 7.3333 | 1.3333 |
| imo_1982_p1 | yes | 5 | 1 | no | 16 | 364.5000 | 19.6875 |
| mathd_algebra_332 | yes | 3 | 2 | no | 12 | 14.8333 | 1.0000 |
| mathd_numbertheory_247 | yes | 2 | 1 | no | 8 | 5.5000 | 0.5000 |
| mathd_numbertheory_155 | yes | 3 | 0 | no | 12 | 60.3333 | 5.6667 |
| algebra_amgm_sumasqdivbgeqsuma | yes | 2 | 0 | no | 4 | 53.0000 | 1.0000 |
| amc12b_2003_p9 | yes | 4 | 2 | no | 12 | 11.6667 | 2.0000 |
| amc12a_2003_p23 | no | 3 | 0 | no | 0 | n/a | n/a |
| mathd_numbertheory_211 | yes | 2 | 1 | no | 8 | 146.8750 | 9.5000 |
| mathd_numbertheory_345 | yes | 4 | 3 | no | 16 | 11.7500 | 1.5000 |

Aggregate: Planner 9/10; end-to-end 1/10; 30 generated nodes (3.0/problem);
12/30 nodes passed pass@4 (40.00%); ROOT 1/9 (11.11%); 100 attempts;
85.81 average proof tokens/attempt; 5.43 average tactics/attempt.

The single Planner failure (`amc12a_2003_p23`) was an ordinary exhausted Lean
formalization failure, not a routing/format/field-transfer bug. Generated nodes
used invalid field notation such as `.factorization`, `.perfectSquares`, and
`.divisors`; Verify identified the missing `.card` in the intended divisor-count
statement, but three downstream repairs still did not produce compiling nodes.

Verify made 35 node calls across initial and repair rounds: zero dependency
failures and two formal-statement failures.

## Same-ID comparison with the prior two complete baselines

The ten source IDs match both baseline manifests exactly and in order.

| Metric | Earlier natural/decomposition run | Previous verified Lean run | BluePrintRepair run |
|---|---:|---:|---:|
| Planner success | 6/10 | 8/10 | 9/10 |
| End-to-end success | 1/10 | 2/10 | 1/10 |
| Generated nodes | 14 | 23 | 30 |
| Avg nodes/problem | 1.4 | 2.3 | 3.0 |
| Node pass@4 | 5/14 (35.71%) | 10/23 (43.48%) | 12/30 (40.00%) |
| ROOT pass@4 | 1/6 (16.67%) | 2/8 (25.00%) | 1/9 (11.11%) |
| Attempts | 44 | 88 | 100 |
| Avg proof tokens/attempt | 60.9545 | 83.9886 | 85.8100 |
| Avg tactics/attempt | 5.0000 | 5.0227 | 5.4300 |

The new module improved Planner admission on this small sample, but did not
improve end-to-end theorem success. The result is stochastic and the sample is
too small for a statistical quality claim.
