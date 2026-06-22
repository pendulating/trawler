# 2026-06-10 — Judge structured output: `guided_json` → `response_format`

## Symptom

The 2026-06-09 GRPO redesign smoke run
(`multirun/2026-06-09_grpo_redesign_smoke/15-18-42`) completed all 6 steps,
but **100% of ranking-judge calls failed** (68/68 ranked flow scorings in
`reward_traces.jsonl` show `judge_failed: true`). Every group fell back to
uniform R_ground = 0.5 — zero advantage from the grounding signal for the
entire run. Nothing surfaced in stdout: validation failures in
`_ranking_single` were silent, and the `rground/judge_failed_group_frac`
health metric only goes to W&B (disabled in the smoke config).

## Root cause

All three judge endpoints in `dagspaces/grpo_training/stages/clients.py`
passed the JSON schema via the legacy `guided_json` extra-body param.
**vLLM ≥ 0.19 silently ignores `guided_json`** — the request pydantic model
has no such field and drops it without error. Structured output must go
through `response_format: {"type": "json_schema", "json_schema": {...}}`
(verified live against the klara:8002 Qwen3.6-27B server, vLLM 0.19.1).

Unenforced, the judge free-formed `candidate_id` instead of the schema's
`candidate_index`; the full-coverage validation in `_ranking_single`
rejected every response → `None` → uniform 0.5 fallback. The absolute
(per-flow) and no-flow coverage judges had the same latent bug but survived
because their parsers tolerate freeform JSON — their scores were never
schema-enforced either, which likely contributed to the bimodal/quantized
score distributions seen in the May λ sweep.

## Fix

- `clients.py`: all three judge endpoints (`_judge_single`,
  `_coverage_single`, `_ranking_single`) now send
  `response_format` via the `_json_schema_response_format()` helper.
- `_ranking_single`: parse/coverage failures now raise internally and are
  logged per attempt (previously silent fall-through); parser accepts
  `candidate_id` as an alias for `candidate_index`.
- `online_rground.py` `_call_ranked`: stdout WARNING whenever any group's
  ranking judge fails (uniform-0.5 fallback is otherwise invisible without
  W&B).
- `common/vllm_inference.py` `_sp_to_openai_kwargs`: the server-mode
  `guided_decoding` translation emitted the same dead `guided_*` params —
  now emits `structured_outputs` (verified live). Affects
  `historical_norms` stages when run against an external vLLM server.
- New tests: `tests/grpo_training/test_judge_structured_output.py` (6) pin
  the `response_format` envelope on all three endpoints, the
  `candidate_id` alias, and the partial-coverage → `None` contract.

## Post-fix verification

Live end-to-end call through `JudgeClient.judge_ranking_batch` with a
realistic 8-candidate group (7 flow extractions + 1 unjustified no-flow
declaration): strict distinct ranks 1–8, correct `candidate_index`, and
the bad no-flow candidate ranked last with grounding_score 0.0 — exactly
the anti-collapse signal the ranked design intends.

## Consequences for the smoke run

- The smoke run's optimizer-side checks remain valid: eval_reward logged at
  steps 3/6, KL active (beta=0.01), zero fully-tied groups, prescreen +
  dev split + gated composition + vignette mix all wired and recorded in
  `training_metadata.json`.
- Its R_ground signal was dead, so the prescreen selection (33/75 kept) was
  driven by non-grounding components only, and the no-flow gate failure
  (59.7% tail no-flow vs 6.25% gold) had no grounding pressure against it.
  **The smoke must be rerun after this fix** (no prescreen cache was
  written — `cache_path` was empty — so no stale cache to clear).
- The run predates the λ=1.0 alignment edit (resolved config: 0.5); the
  rerun picks up λ=1.0 from the YAML.
