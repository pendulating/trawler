# 2026-07-21 — PrivacyLens parity review (vs SALT-NLP/PrivacyLens)

Same playbook as the CIRL-729 review: fetched the upstream evaluation code
(`evaluation/probing.py`, `get_final_action.py`, `evaluate_final_action.py`)
and diffed the `privacylens` dagspace against it. Four fixes applied, all
regression-locked in `tests/privacylens/test_judge_response_parsing.py` +
updated canary tests. Suite: 903 passed.

## F1 — CRITICAL: judge-response parsers corrupted guided-JSON responses

`parse_helpfulness_responses` / `parse_leakage_responses` /
`finalize_async._row_yes` applied the upstream *free-text* substring scan
FIRST, treating JSON as a "legacy" fallback — but the live and export judge
stages have always requested guided-JSON (`HelpfulnessResult` /
`YesNoResult`). Scanning a JSON blob with the greedy
Poor→'0'→Unsatisfactory→'1'→… rule mis-scores any response whose free-text
`reasoning` contains a colliding digit or word (`"sent at 10 AM"` → `'0'` →
score 0).

**Quantified on a real canonical cell**
(`multirun/2026-07-16_eval_canonical_sft_gemma4/14-15-39/4`):
- helpfulness: **84/391 (21.5%) mis-scored** — every one a true 3 dragged to
  0/1/2; mean_score 2.345 → 1.859; helpful_rate 0.783 → 0.601 (−18 pp)
- leakage per-secret: 4/1114 false "Yes" flips
- The ordering landed 2026-04-26 (44484d4) — **all PL helpfulness /
  adjusted-leakage numbers computed since then are affected**, including
  keeper-era camera-ready columns.

**Fix:** JSON-first parsing everywhere; upstream substring semantics
preserved verbatim for genuine free-text (provider ignoring
`response_format`, synthesized `Answer: Yes./No.` aggregates).

**Rescue path (no GPU needed):** raw judge responses live on disk
(`output.jsonl` / `helpfulness_judge_text`); re-running finalize +
compute_metrics with the fixed parser recovers correct numbers for
existing runs.

## F2 — HIGH: stale metric names in primary_metrics + W&B formatter

`eval_all/primary_metrics.py` read `leakage.leakage_rate` /
`helpfulness.mean_score` — names `compute_metrics.py` stopped emitting when
it split into `*_among_parseable` / `*_overall_with_default_zero` →
resolved to **None** in every eval_all summary. The privacylens
orchestrator's W&B formatter read the same stale keys with `get(..., 0.0)`
defaults → logged **constant zeros**. (Complementary to, and no overlap
with, the `wandb-parity` branch.) Fixed to the `*_among_parseable` primary
variants (per compute_metrics's own "primary, paper-quoted" doc), with the
overall audit variants logged alongside.

## F3 — protocol: restored upstream's tool pin (approved by Matt)

Upstream `get_final_action.py` ends the agent prompt with
`The [Action] should be {final_action}.` — `T['final_action']` is a **bare
tool name** (`GmailSendEmail`; verified across all 493 rows), part of the
task spec. Our `build_action_prompt` omitted it on a mistaken gold-leak
rationale, letting models pick a different tool (helpfulness Note 2 scores
wrong-tool as Poor 0). Restored; canary tests updated —
`sensitive_info_items` remains the forbidden gold field.

## F4 — protocol: judges no longer see the model's [Thought] (approved)

Upstream slices the judge input from the FIRST `Action:` occurrence
(`action[action.find('Action:') + len('Action:'):].strip()`), so judges
never see the chain of thought. Our `_clean_generated_action` only stripped
a literal *leading* `Action:` — Thought-first generations (which the prompt
explicitly requests) reached both judges with CoT attached, so a model
merely deliberating about the secret could be flagged as leaking. Fixed to
the upstream slice.

## Verified clean (no change)

- QA probing prompts: seed/vignette/trajectory bodies faithful to
  `probing.py` including article-strip, the naive `+ing` gerund
  (`"share"` → `"shareing "` trailing-space quirk, test-locked),
  vignette-only `.lower().strip('.')`, trajectory truncation at
  `### Scratchpad`, and the email/message/post medium collapse. The JSON
  answer-format instruction (vs upstream's `(A)/(B)` + `\bB\b` scan) is a
  documented, deliberate deviation.
- `post_process_action`: byte-identical to upstream (`Observation:` /
  `Final Answer:` cut, first-`}` bound, Llama-3 header strip).
- Leakage/helpfulness judge prompt templates: byte-identical few-shots
  (incl. the `""to""` CSV-quoting quirks and the 😄 emoji).
- Judge gate `'Action:' in action` (case-sensitive): matches upstream; both
  `*_overall_with_default_zero` (upstream denominator: unjudged rows count
  as non-leak / score-0) and `*_among_parseable` variants emitted.
- Adjusted leakage: leakage among helpful (score ≥ 2), matching upstream.

## Comparability note

F3 + F4 change the eval protocol (prompt + judge input): **runs after
2026-07-21 are not comparable with keeper-era PrivacyLens rows.** Since F1
already invalidates keeper-era helpfulness / adjusted-leakage values, the
PL columns need refreshing regardless — leakage/QA via re-finalize
(parse-only), helpfulness/adjusted + the F3/F4 deltas via re-run.
