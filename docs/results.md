# Results

All tables are copied from the source fact sheet with the one-line measurement condition above each table. The source fact sheet is not republished here, so the numbers below can be checked for arithmetic consistency across this repo's tables, but cannot be independently traced back to the source.

## Main table

Conditions: base = `nfB-think-full` (1M / YaRN×4 / fp8 KV / MTP4); thinking on + the sampling values listed in the README stack table + `max_tokens 16384` + `qwen3_coder`; 2 runs per category; score 0–100; wall ratio = candidate ÷ baseline. The aggregation method across the 2 runs (mean vs median), the per-run scores, the seed, the per-category variance, the timeout-scoring rule, and the absolute wall-clock seconds are not recorded in the sources, so the table can only be checked for arithmetic consistency, not for effect size or significance. "—" = variant never started (see What did not work).

| category | base | nfAB-native (262K) | nfAB-yarn2 (512K) | nfAB-kvbf16 (1M) | nfAB-indexer (4096/2) |
|---|---|---|---|---|---|
| c1-kbqa | 88.3 | 93.3 (+5.0) 0.98× | 85.0 (−3.3) 0.91× | 83.3 (−5.0) 0.91× | — |
| c5-extract | 84.2 | 86.0 (+1.8) 1.31× | 84.4 (+0.2) 0.84× | 83.0 (−1.2) 0.89× | — |
| c7-agentic-if | 91.7 | 92.5 (+0.8) 1.02× | 94.2 (+2.5) 0.99× | 90.8 (−0.9) 1.09× | — |
| c8-judgment | 98.3 | 98.3 (+0.0) 1.05× | 93.3 (−5.0) 0.96× | 96.7 (−1.6) 0.95× | — |
| c9-long-coding | 100.0 | 100.0 (+0.0) 0.62× | 98.3 (−1.7) 0.60× | 100.0 (+0.0) 0.98× | — |

## MTP control

Conditions: c7-agentic-if × 3, both arms same explicit-xhigh params; MTP4 arm = `nfB-think16-c7a`, MTP-off arm = `nfAB-mtpoff-c7a`.

| arm | three scores | median | wall clock |
|---|---|---|---|
| MTP4 (baseline) | 88.3, 91.7, 91.7 | 91.7 | — (absolute seconds not recorded) |
| MTP off | 95.0, 93.3, 93.3 | 93.3 | ~30% slower (reported; absolute seconds, per-run values, and aggregation method not recorded) |

## KV pool

Conditions: engine config line `GPU KV cache size`; 1M fp8 baseline = 3.44–3.48M tokens across the 1M bring-up runs; KV bf16 variant = 3.45M→1.86M tokens.

| form | GPU KV cache size (tokens) | quality vs fp8 |
|---|---|---|
| 1M baseline (YaRN×4, fp8 KV, MTP4) | 3.44–3.48M | — (reference) |
| KV bf16 @ 1M | 3.45M → 1.86M | no fp8 score decrease observed across the five listed categories (official recipe does not use fp8 KV, so there is no same-recipe fp8-vs-non-fp8 control; this is a bounded observation, not a losslessness proof) |

## Verdict

Conditions: as recommended in the sources; see the Update note in the README for the outcome.

- Default profile: native 262K, fp8 KV, MTP4, no-async, `qwen3_coder`. The default engine exposes a fast tier and a quality tier; their thinking-effort assignments are not recorded in the sources, so no thinking-tier recommendation is made here.
- On-demand profile: kept (1M ctx, YaRN×4), switched in via the switch script (the 4 min switch time is a reported figure whose start/end points, cache state, repetition count, and switch script are not recorded). The public abstraction for the switch is `switch/stack-mode.sh` in the sibling repo dell-pro-max-gb10-vllm-stack-ab.

## Source discrepancy (unresolved)

Conditions: two statements from the sources describing the same native-vs-YaRN×4 comparison, neither adjusted.

| statement | source wording | compare-table figure |
|---|---|---|
| "static YaRN hurts short text" | "c1 −5, c9 38% slower" | native's c9 wall ratio = 0.62× (candidate ÷ baseline: native used 38% less time; the baseline used about 1.61 times native's time, or about 61% more; native's c9 score was unchanged at 100.0) |

The legacy "c9 38% slower" quote is inconsistent with the 0.62× wall ratio; swapping the comparison sides does not resolve the discrepancy. The quoted wording and measured ratio are preserved, without treating them as equivalent.
