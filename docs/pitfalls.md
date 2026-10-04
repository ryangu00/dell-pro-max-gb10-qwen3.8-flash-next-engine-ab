# Pitfalls

Each pitfall is expanded: **symptom** / **root cause** / **fix** / **how we found it**. Numbers and engine strings are copied unchanged from the fact sheet.

## 1. MTP draft rejects the configured sequence length

- **Symptom:** `QSA sequence length 1000000 exceeds the configured limit 262144` at startup, with MTP enabled. (The startup error prints 1000000 as the requested length; the example fix JSON below uses 1048576 = 262144×4 as the actual configured limit for the 1M variant. The exact configured token limit per variant is the `--max-model-len` value listed in the README variant rows; the main model and the MTP draft are configured to the same value.)
- **Root cause:** This vLLM build appears to construct the MTP draft's model config from `speculative_config.max_model_len`, which defaults to `None`, and to not pass the dict `--hf-overrides` to the draft side (the relevant code path in `mtp.py` was not matched to a pinned vLLM version, so this is an inference from the symptom, not a code-verified root cause). On that reading, the draft inherits the default 262144 limit while the main model is configured for 1M, and the QSA check trips on the mismatch.
- **Fix:** Set `max_model_len` **inside the `--speculative-config` JSON**, matching the main context length (e.g. `--speculative-config '{"method":"mtp","num_speculative_tokens":4,"max_model_len":1048576}'` for the 1M variant). This is why the fixed-flags row in the stack table states draft-side `max_model_len` is mandatory.
- **How we found it:** The startup error message names the QSA sequence-length check and the two limits (1000000 vs 262144), which pointed at a config-split between main model and draft; reading the launch config confirmed the draft had no `max_model_len` of its own.

## 2. Runaway agent loops and 3–5× wall clocks on agentic categories

- **Symptom:** Agentic categories (c7-agentic-if, c9-long-coding) ran 3–5× slower than expected, with runaway agent loops.
- **Root cause:** Working hypothesis, not a confirmed root cause — the exact vLLM version, the matching code path, a controlled reproduction, and a before/after measurement are not recorded here. The suspected mechanism is a vLLM async-scheduling × MTP host/device race in which accepted-token counts appear to leak across requests. Community issues #53912 and #51571 show the same symptom shape; this build auto-enables async scheduling whenever MTP is on. A public minimal reproduction and a before/after result are not included here.
- **Fix:** Pass `--no-async-scheduling`. All A/B variants include this flag (it is in the fixed-flags set in the stack table).
- **How we found it:** Wall clocks on the agentic categories were multiples of the baseline before this flag was added; the pattern matched the two community issues once the MTP × async interaction was suspected.

## 3. KV bf16 halves the token pool for no quality gain

- **Symptom:** The KV bf16 variant (`nfAB-kvbf16`) logged `GPU KV cache size` dropping from 3.45M to 1.86M tokens, and scored Δ ≤ 0 in all five measured categories (c1 −5.0, c5 −1.2, c7a −0.9, c8 −1.6, c9 +0.0).
- **Root cause:** bf16 KV costs double the bytes per token compared with fp8 KV, so the same GPU memory holds roughly half the tokens. On this bank the higher precision produced no observed quality gain (no fp8 score decrease was observed across the five listed categories vs the official non-fp8 recipe, which does not use fp8 KV — so there is no same-recipe fp8-vs-non-fp8 control, and this is a bounded observation, not a losslessness proof).
- **Fix:** Keep fp8 KV. The default profile ships fp8 KV.
- **How we found it:** The per-variant engine config line `GPU KV cache size` was extracted and archived post-boot (per the launch sequence), and the banked scores showed no positive Δ in any category.

## 4. Static YaRN taxes short-context quality

- **Symptom:** The YaRN×4 1M baseline underperforms native 262K on the short-context categories in this bank — native wins c1 +5.0 (88.3 → 93.3) vs the YaRN×4 baseline, and the official card's own "static YaRN hurts short text" check is reported as "c1 −5, c9 38% slower" in one source (see the README source-discrepancy note: the legacy timing quote is inconsistent with the 0.62× wall ratio, and swapping the comparison sides does not resolve the discrepancy).
- **Root cause:** Observation, not an isolated causal root cause. No input-length sweep and no max_model_len-only control were run, so the short-context difference is associated with the YaRN×4 form by the experiment design (which changes RoPE factor and context length together), not separately proven to stem from static YaRN alone.
- **Fix:** Default to native 262K; escalate to the 1M YaRN×4 profile only when a job actually needs the long context (switched in via the switch script; the 4 min switch time is a reported figure whose measurement conditions are not recorded).
- **How we found it:** The `nfAB-native` variant (no rope override, `--max-model-len 262144`) was positive or flat in every measured category vs the 1M YaRN×4 baseline, with c1 the largest gain.

## 5. "Override silently not applied" contaminating an A/B

- **Symptom:** A variant's intended engine override (e.g. a different `max_seq_len`, `kv_cache_dtype`, `indexer_budget`, or `speculative_config=None`) is silently dropped, so the variant runs with the wrong config and its scores pollute the comparison.
- **Root cause:** A specific dropped input, the override precedence that applied, the matching log line, and a minimal reproduction are not recorded here, so the mechanism is stated only at the level observed: in this build, an unsupported or shadowed override (a misformed `--hf-overrides` or an ignored flag) can leave the baseline config in place with no fatal log line.
- **Fix:** The launch helper asserts the vLLM config lines post-boot — `max_seq_len`, `kv_cache_dtype`, `indexer_budget`, `speculative_config=None` — and skips the variant on mismatch. Concrete asserts used: `max_seq_len=262144`, `kv_cache_dtype=(auto|bfloat16)`, `indexer_budget.{0,4}4096`, `speculative_config=None`. These regexes confirm the listed config lines printed at boot; they do not by themselves confirm the YaRN factor, compress_ratio, the target-vs-draft split, or the resolved dtype, nor do they rule out a silent override applied elsewhere, so the assert-then-bank pattern narrows but does not eliminate the contamination risk. Copy this assert-then-bank pattern in any harness.
- **How we found it:** Built into the per-variant launch sequence — after polling readiness, the helper extracts the engine config lines and regex-asserts the intended override actually took effect before banking the eval runs.

## 6. SM12x QSA indexer knob locked

- **Symptom:** The indexer variant (`nfAB-indexer`, `indexer_budget: 4096`, `indexer_compress_ratio: 2`) fails to start with `SM12x QSA integration requires compress_ratio=4`.
- **Root cause:** This vLLM build's kernel raises the error — the SM12x QSA kernel in this build requires `compress_ratio=4`. The community-forum (#260) lever (budget 4096 / ratio 2) is unavailable on this build. Budget 4096 / ratio 4 and other ratios were not tested, and other vLLM builds were not tested, so the lock is scoped to this build, not asserted for the whole GPU generation.
- **Fix:** `compress_ratio` must stay 4 on this vLLM build regardless of `indexer_budget`. Budget 4096 / ratio 2 is dead on this build; the indexer lever is unavailable on this build (other builds not tested).
- **How we found it:** The fatal pattern grep in the launch sequence caught the `SM12x QSA ...` error, the variant never reached readiness, and its column in the main table is "—" (never started). The exact engine text is `SM12x QSA integration requires compress_ratio=4`.
