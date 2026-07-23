# 2026-07-21 — VLM-GeoPrivacyBench parity review (vs 99starman/VLM-GeoPrivacyBench)

Third pass of the day (after CIRL-729 and PrivacyLens). Fetched upstream
`src/{prompts,utils,gen,eval}.py` and diffed the `vlm_geoprivacy_bench`
dagspace. One semantic fix (value-neutral on all real runs), parity locks
added (`tests/vlm_geoprivacy_bench/test_parity.py`). Suite: 913 passed.

## Fix — headline accuracy denominator flipped to upstream parity

Upstream `eval.py` computes per-question accuracy over ALL merged rows —
an `N/A` (unparseable) prediction **counts as wrong**. Our
`compute_metrics` dropped unparseable rows before scoring, which would
silently inflate accuracy for any model with parse failures, and also
inverted the repo's own house convention (headline = paper parity,
diagnostic = among-parseable; cf. `wiki/metric-trust.md`, the CIRL probing
metrics).

**Empirical impact: zero.** All 85 July metrics cells have Q7
`parseable_rate = 1.000` (guided JSON decoding), so headline values are
unchanged everywhere; the flip protects future runs (e.g. a model whose
guided decoding is bypassed or whose output is empty) from silent
inflation. `per_question.<Q>.accuracy` is now upstream-style
(provenance `unparseable_counted_as_wrong`), with
`accuracy_among_parseable` preserved as the diagnostic.
`eval_all/primary_metrics.py` path (`per_question.Q7.accuracy`) unchanged
and still resolves. Integration test updated to the new invariant.

## Verified clean (no change)

- **`QUESTION_DATA` / `SYS_MSG` / `GRANULARITY_JUDGE`**: verbatim from
  upstream `src/prompts.py` (test-locked on Q2 + zs system message).
- **Prompt assembly**: per-question block byte-faithful
  (`"\nQ{i}: {q}\n" + options + "\nHeuristics:{h}\n\n"`, test-locked).
- **`parse_answers`**: bug-for-bug faithful to `utils.py` — `*` strip,
  yes→A / no→B, `Answer:` fallback, N/A padding, and the upstream quirk
  where a missing later `Q{i}:` key discards parsed answers in favor of
  raw line-splitting (all test-locked).
- **Per-model templates** (`model_prompts.py`): mirror upstream `gen.py`
  (qwen2.5-vl manual template byte-same; deepseek-vl2, llama-vision,
  gemma-3 same construction). Additional families (qwen3.5, gemma-4,
  phi-4, qwen3-vl) are additive; the gemma-4 system-prompt corruption
  guard from 2026-07-18 stands.
- **Gold data**: local `annotation_labels.csv` / `images_metadata.csv`
  md5-identical to upstream (1200 rows).
- **No reasoning-split hazard**: this dagspace's own `vlm_inference.py`
  keeps raw output (optional think-strip), and guided JSON forces clean
  MCQ output — the CIRL-class `</think>` bug does not apply.

## Documented deviations (now noted in `prompts.py` header)

1. `INST_LABEL_STRICT` asks for a JSON object + guided decoding
   (`MCQResult`) instead of upstream's `Q1: <label>` lines — same
   robustness trade as the other dagspaces; the upstream line format
   remains a parse fallback.
2. Sampling temperature 0.2 (repo eval convention) vs upstream 0.7;
   seed 1, top_p 0.95, max_tokens 512 match.
3. Free-form granularity judge = local vLLM model (Qwen2.5-72B-AWQ)
   instead of upstream's inline gpt-4.1-mini; judge prompt text verbatim.
4. **Coverage**: 883/1200 images available locally; with the configured
   `Flickr-yfcc26k` exclusion the eval runs on **n=783** rows (consistent
   across all cells). Upstream evaluates whatever images the user's
   `download_images.py` retrieves, so their n varies too — but any
   cross-paper comparison of absolute numbers should note the subset.
