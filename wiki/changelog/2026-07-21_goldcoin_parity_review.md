# 2026-07-21 — GoldCoin-HIPAA parity review (vs HKUST-KnowComp/GoldCoin)

Fourth pass of the day (CIRL-729, PrivacyLens, VLM-GeoPrivacy, now GoldCoin).
Fetched upstream `eval/{build_instruction_*,eval_api,eval_llm,parse_eval_result}.py`
and diffed the `goldcoin_hipaa` dagspace. One approved semantic fix with real
number impact; parity locks in `tests/goldcoin_hipaa/test_parity.py`.
Suite: 922 passed.

## Fix (Matt-approved) — headline denominator flipped to upstream parity

Upstream `parse_eval_result.py` never drops an unparseable response: its
fallback is `gt.remove(truth); random.choice(gt)` — over a binary label set
that is **deterministically the wrong label** — and the row stays in
accuracy and macro-F1. Our `compute_metrics` dropped unparseable rows,
inflating accuracy for weak-format models.

**Quantified (July runs):** 19/266 cells had `parseable_rate < 0.99` — all
**Gemma-4-E2B-it** (applicability, worst pr=0.715: acc 0.399 → 0.285
upstream-style) and **GPT-OSS-20B** (compliance 0.756 → 0.636 on the
canonical instruct sweep). Headline `accuracy` / `macro_f1` / confusion
matrix / per-class counts now use the forced-wrong substitution over ALL
rows (provenance `unparseable_forced_wrong`); the old behavior survives as
`accuracy_among_parseable`.

**Exact retro-conversion (no re-run needed):** the substitution never
produces a correct prediction, so for any pre-flip metrics.json,
`accuracy_new = accuracy_old × parseable_rate`. Affected camera-ready
cells (gemma-4-E2B, gpt-oss) can be re-derived from stored metrics alone.

**⚠ Mixed-semantics hazard:** the running judge-free variance sweep
(array 150351) imports live repo code — cells whose `compute_metrics`
runs after this change use forced-wrong semantics while earlier cells used
drop semantics. Reconcile in the noise-floor notebook via the exact
conversion above (both variants carry `parseable_rate`).

## Verified clean (no change)

- **Alpaca template + instruction texts + few-shot blocks**: verbatim from
  upstream (test-locked).
- **Ground-truth mapping**: Permit/Forbid → "Applicable" for the
  applicability task, matching `parse_eval_result.py`.
- **Test data**: local `test_real_cases_hipaa_{applicability,compliance}.csv`
  md5-identical to upstream.
- **`extract_step_result` / `clean_response`**: verbatim (with
  `_strip_think_blocks` prepended — harmless here since parsing is
  keyword/JSON-based, no `</think>` dependency; no CIRL-class hazard).
- **eval_all wiring**: `primary_metrics` paths (`accuracy` under
  `compute_metrics_{applicability,compliance}`) and the orchestrator W&B
  formatter keys all still resolve.

## Documented deviations (now in `prompts.py` header)

1. JSON-format instruction + guided decoding
   (`ComplianceResult`/`ApplicabilityResult`) appended to the upstream
   prompt; upstream keyword scan kept as fallback **with the 2026-07-14
   negation fixes** ("impermissible" ≠ "permis", word boundaries) — a
   deliberate correctness improvement over upstream's substring scan,
   previously reviewed and reported.
2. Parser emits `"unparseable"` instead of upstream's random-wrong guess;
   the forced-wrong substitution now happens transparently in
   `compute_metrics` with provenance.
3. Sampling: temp 0.2 (repo convention) vs upstream 0.7; max_tokens
   1024/4096 vs upstream 512.
