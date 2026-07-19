# Migration — old→new map, deletions, keeper guarantee, test plan

**Parent:** [README.md](README.md) · **Date:** 2026-07-16 · **Status:** drafted

## The parallel-stack rule (non-negotiable)

The v9-ckpt100 keeper must remain byte-reproducible until the camera-ready is
out the door. Therefore the m-series is **additive code**: new modules, a new
training-config group, new tests. The frozen surfaces —
`training/grpo/online_rground_external.yaml`, the `directional` composition
path in `rewards.py`, `online_rground.py`'s clamp/multiplier machinery, the
v-era prescreen formula versions — are not edited, not refactored, not
"cleaned up in passing." Deletion happens once, after the keeper is obsolete,
as its own commit.

## Old → new map

| v9–v12a mechanism | m-series home | disposition |
|---|---|---|
| `CompositeRewardFunction` 6-weight sum | new `ModularReward` (working name), config-selected | parallel implementation |
| r_uncert / r_complete / r_consist | [`R-VALID`](reward-valid.md) binary gate | merged; confidence + construct-discrimination facets dropped |
| `confidence_fallthrough` knob | — | no counterpart (confidence unscored) |
| R_ground correct pass (listwise) | [`R-GROUND`](reward-ground.md) auxiliary | kept; rubric slimmed to 2 criteria |
| `contrastive_lambda` clamp | [`R-CONTRAST`](reward-contrast.md) auxiliary | reshaped 1−wrong; λ deleted |
| `contrastive_ratio` rows | — | already 0; does not migrate |
| `rground_app_mode/floor/floor_prohibit/hedge_prohibit` | — | subsumed by [`R-OUTCOME`](reward-outcome.md); `diag/direction_consistency` metric only |
| `abstention_score` bypass | [`A-ABSTAIN`](reward-abstain.md) | kept; extended to gold-NO extractions |
| single-vignette rows + 3-term judgment reward | [`T-VIGNETTE`](task-vignettes.md) batteries + deontic distance | rebuilt; keyword-counter term deleted |
| R_context / R_cohere | — | cut (resurrection below) |
| `vignette_ratio` | `task_mix.vignette` | renamed, same meaning |
| prescreen (variance-only) | stratified prescreen | extended; `formula_version=m1` |
| optimizer knobs | `training/grpo/m_series.yaml` | copied verbatim ([optimizer.md](optimizer.md)) |

## New components to build (implementation checklist)

1. **Probe builder** (dataset build step): per-chunk retrieval over teacher
   flows → probe pool → null-answerability filter → stratified K-sample.
   Deterministic; artifacts written next to the prompt set.
2. **Battery builder**: context clustering, composition constraints, seeds.
3. **Answerer client**: reuse `judge_client.py` plumbing; one batched call
   per completion; retry-then-group-neutral fallback.
4. **`ModularReward`**: gate → routing → outcome + auxiliaries with the 2:1
   weight rule; per-module W&B namespaces.
5. **Deontic-distance scorer** in `deontic.py` (axis map + `1 − |Δ|/2`),
   beside `FORCE_TO_GOLD`.
6. **Stratified prescreen** + m1 cache signature.
7. **`training/grpo/m_series.yaml`** + `conf/sweep/grpo_m1_grid.yaml`.

## Test plan (new tests; the old suite keeps passing untouched)

- **Unit — scoring:** deontic-distance table (all 25 force pairs), battery
  mean/rescale, routing table (all 5 rows), weight renormalization per
  module subset, gate criteria (each failing independently).
- **Unit — builders:** probe determinism (same chunk_id → same probes),
  force stratification guarantees, battery composition floors, null-filter
  logic (mocked answerer).
- **Leak canaries:** no probe/battery `prompt_text` contains source
  articulation tokens or any force word (PrivacyLens canary pattern).
- **Integration:** cell config → resolved module set → prescreen signature
  changes iff the module list / task mix / seeds change (extends
  `tests/grpo_training/test_prescreen_cache_key.py` patterns in a new file).
- **Keeper guard:** an explicit test that `online_rground_external.yaml`
  still composes to the frozen values (a checksum-style regression, so a
  "cleanup" that touches the keeper fails CI loudly).

## Resurrecting cut modules

`R-CONTEXT` and `R-COHERE` were cut for lack of isolated evidence, not
disproven. If a reviewer asks: each returns as one additive auxiliary in
`reward_auxiliaries` (the module interface is exactly their shape) and one
add-one-in cell — no redesign required. The same door admits `R-OUTCOME`
variants (answerer swap, K sweep) as m2 cells.

## Deletion list (post-camera-ready, single future commit)

Legacy reward paths in `rewards.py`/`online_rground.py`, the v-era grpo
training yamls except the keeper's (kept for the archive), the
`confidence_fallthrough` knob and its tests, obsolete model yamls already
marked OBSOLETE. Until then they cost nothing but disk and patience.
