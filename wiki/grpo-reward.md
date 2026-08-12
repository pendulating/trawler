# GRPO Composite Reward

**Scope.** ⚠️ **HISTORICAL as of 2026-08-05.** v9-ckpt100 is **deprecated**; the camera-ready GRPO model is the m2 `full` cell ([grpo_redesign/](grpo_redesign/README.md); ruling in [2026-07-31_kto_plan.md](2026-07-31_kto_plan.md) §19). This page documents the *superseded keeper-era recipe* (the v9→v12a lineage) exactly as implemented — config `dagspaces/grpo_training/conf/training/grpo/online_rground_external.yaml`, code `stages/{grpo_training,rewards,online_rground,deontic,prompt_screening}.py`. Verified 1:1 with the code 2026-07-20 (see `changelog/2026-07-20_grpo_methodology_congruency_review.md` for the drift that was fixed). The ground-up modular redesign for new runs lives at `grpo_redesign/`.

GRPO (Shao et al. 2024) samples `G` completions per prompt, scores each with a scalar reward, and updates the policy from within-group mean-centered advantages — no preference pairs, no learned reward model. Trawler's reward is composite: mostly deterministic programmatic checks over the structured CI output, plus one LLM-judge component (`R_ground`) evaluated **listwise** within each group.

## Judge

- **Keeper-era judge (all v9–v12a paper runs): Qwen3.6-27B** (`/share/pierson/matt/zoo/models/Qwen3.6-27B`, TP=2 on klara:8002), declared in `conf/config.yaml` (`judge_model.model_source`, which must equal the served model name). Decoding: temperature 0.0, guided JSON — the judge is deterministic up to server nondeterminism.
- ⚠️ **Launch-time mismatch hazard (since 2026-07-16):** the Gemma stack migration changed `scripts/judge_server.sub`'s default to **Gemma-4-31B-it**. Reproducing keeper-era reward requires `JUDGE_MODEL=/share/pierson/matt/zoo/models/Qwen3.6-27B` when launching the server; otherwise the served name mismatches `judge_model.model_source` and judge calls fail.
- Pre-Qwen3.6 runs (Qwen2.5-72B-Instruct-AWQ) are flagged OBSOLETE in their model yamls.
- Embeddings for norm retrieval: Qwen3-Embedding-8B (server on port 8001). `R_context` uses a separate local `all-MiniLM-L6-v2` — the two are unrelated.

## Optimizer recipe (production values)

From `online_rground_external.yaml`; rationale for each change is in the yaml comments and the field notes it cites.

| Knob | Value | Note |
|---|---|---|
| `num_generations` (G) | 8 | v1's G=2 gave mostly-tied pairwise groups |
| `learning_rate` | 2e-5 | cosine_with_min_lr, `min_lr_rate=0.3`, warmup 0.1 |
| batch | 1×32 grad-accum | = 4 prompts/step at G=8 |
| `num_epochs` | 3 | `save_steps=50` (dense, for kill-at-peak selection) |
| `beta` (KL to SFT ref) | 0.02 | ref = merged SFT (trainer disables the fresh LoRA); stability floor, not an anchor (β·KL ≈ 0.001) |
| `scale_rewards` | `"none"` | Dr. GRPO — no group-std scaling; absolute gaps ARE the signal, so tier magnitudes below are load-bearing |
| `mask_truncated_completions` | `true` | DAPO — truncated outputs fail JSON parsing and would reward the short no-flow path |
| `vllm_importance_sampling_mode` | `token_truncate` | GSPO-style per-token clamp; `sequence_mask` preferentially zeroed long (extraction) completions — the v1–v4 over-abstention cause |
| `num_iterations` (μ) | 1 | v8's μ=2 re-triggered the entropy breakout; reverted in v9 |
| `epsilon_high` | 0.28 | **INERT at μ=1** (PPO ratio ≡ 1, clip never binds — v8 analysis). Kept in config; do not report Clip-Higher as active for v9+ runs |
| `loss_type` | *(not set)* | inherits TRL default — **`"dapo"` in TRL 1.8.0** (token-level global normalization; also covers Dr. GRPO's length-bias fix). Should be pinned before any TRL upgrade |
| `max_completion_length` | 3072 | vLLM colocate, `gpu_memory_utilization=0.45`, sleep mode, `max_model_len=16384` |
| `seed` | 42 | seeds Python/NumPy/torch + GRPOConfig `seed`/`data_seed`; basis of the seed-variance sweep |

Checkpoints from the old recipe (G=2 / lr=1e-6 / beta=0, the `grpo-l*-r*` yamls) are not comparable to new-recipe runs.

## Reward routing: three row types

`CompositeRewardFunction.__call__` (`rewards.py`) routes each completion by its prompt's `task_type`:

1. **CI-extraction row, flows extracted** → 6-component directional composite (below).
2. **CI-extraction row, no-flow declaration** (`has_information_exchange=false`, empty flows) → **bypasses the composite entirely**: `R = no_flow_reward(gold)` = **0.6** (gold agrees no flow) / **0.4** (gold unknown) / **0.1** (gold says a flow exists). The listwise judge still ranks the no-flow candidate (anti-collapse instruction: unjustified no-flow ranks last), but its judged score is discarded.
3. **Judgment-vignette row** (`task_type: norm_judgment`, mixed in at `vignette_ratio: 0.3`) → 3-component judgment reward: `0.5·r_judgment + 0.25·r_judgment_reasoning + 0.25·r_norm_cite`. Never touches the 6-component composite.

Groups are homogeneous (a group = G completions of one prompt), so the different scales across row types never meet inside one advantage computation.

## Components and composition (CI-extraction rows)

Production is `reward_composition: directional` (v9). The six components are partitioned:

$$R \;=\; \underbrace{\frac{\sum_{k \in \text{gate}} w_k R_k}{\sum_{k \in \text{gate}} w_k}}_{\text{gate: well-formedness}} \times \underbrace{\frac{\sum_{k \in \text{content}} w_k R_k}{\sum_{k \in \text{content}} w_k}}_{\text{content: substance}}$$

| # | Component | Raw w | Partition | Normalized weight | What it measures |
|---|---|---|---|---|---|
| 1 | `R_uncert` | 0.10 | gate | 1/3 of gate | Schema validity + `has_information_exchange` present + self-reported confidence (facet 3 is monotone in confidence — **not** calibration; also carries the keeper-repro bug behind `confidence_fallthrough: false`) |
| 2 | `R_complete` | 0.05 | gate | 1/6 of gate | Fraction of non-null CI-tuple + metadata fields |
| 3 | `R_consist` | 0.05 | gate | 1/6 of gate | Boolean invariants (flag↔flows agreement, sender≠recipient, is_new_flow→non-appropriate) |
| 4 | `R_context` | 0.20 | content | **2/7 ≈ 0.29** | Per-flow max cosine (MiniLM) of stated context vs the source's norm contexts |
| 5 | `R_cohere` | 0.10 | gate | 1/3 of gate | Extracted entities appear in the reasoning trace (token overlap) |
| 6 | **`R_ground`** | 0.50 | content | **5/7 ≈ 0.71** | Listwise judge grounding × deontic direction multiplier (below) |

Quote the **normalized** weights in the paper — the raw 0.50 understates R_ground (0.71 of the content factor) and the additive `R = Σ wᵢRᵢ` formula was retired with v8; no v9+ run used it. Rationale for R_ground's dominance: highest inter-completion variance in pilots → primary driver of the advantage. Note `r_cohere` sits in the *gate* under directional (the old page's "discriminative" typing was gated-era).

Legacy compositions still in code, selectable by config: `additive` (Σ wᵢRᵢ) and `gated` (gate{1,2,3} × disc{4,5,6}). `abstention_penalty` is 0.0 in production — directional's gold-aware no-flow path replaced it (a post-hoc subtraction would double-count).

## `R_ground` — ranked (listwise) mode, production

`rground_scoring: ranked` since 2026-06. One listwise judge call per (group, universe); within a reward call, completions are grouped by prompt text.

**Why listwise.** LLM judges are poorly calibrated at absolute scoring: in the May sweep the absolute judge's scores were quantized/bimodal and **60% of groups tied on R_ground** → zero advantage from the dominant component. The listwise judge is forced to discriminate. Consequence for the paper: R_ground is a **within-group comparative reward, not a pure function of (prompt, completion)** — the rank component, the shared retrieval (mean of chunk + all candidates' flow-query embeddings, re-normalized; `rank_top_k=5` norms per group), and forced strict ranking of near-duplicates all make a completion's reward depend on its siblings. This is a deliberate deviation from vanilla GRPO's reward assumption; frame it as such (see the 2026-07-20 congruency review, §1).

Per group of `n` judged candidates, with `rank_weight` w_r = 0.5 and contrastive λ = 1.0:

1. **Correct-universe listwise call** → strict ranks + per-candidate absolute grounding scores g_i ∈ [0,1].
2. **Wrong-universe listwise call** (one random wrong source per group per call, shared by all members) → grounding scores g̃_i only; ranks against a wrong universe are forced-choice noise and are ignored.
3. **Blend + symmetric contrast (v8):** the contrast applies to the grounding term only, leaving the rank term (the anti-tie signal) contrast-free:

$$\text{base}_i \;=\; w_r \cdot \frac{n - \text{rank}_i}{n-1} \;+\; (1-w_r)\cdot \mathrm{clamp}\big(g_i - \lambda\,\tilde g_i,\ 0,\ 1\big)$$

   (The pre-v8 form `clamp(s_i − λ·g̃_i)` subtracted full wrong-grounding from the rank-diluted correct score — asymmetric, clamped ~1/3 of well-grounded extractions to 0, and ate the rank signal.)
4. **Deontic direction multiplier (v9/v10/v12a):** `R_ground,i = clamp(base_i × m_i, 0, 1)`, where m_i comes from `deontic.py`'s cost-sensitive ladder, keyed off the **top-1 retrieved norm's `normative_force`** (obligatory/recommended ⇒ expected "appropriate"; prohibited/discouraged ⇒ "inappropriate"; permitted/unknown ⇒ no expectation). Per flow, then averaged over the candidate's flows:

| Verdict vs governing force | m | Since |
|---|---|---|
| correct direction | **1.0** | v9 |
| hedge ("ambiguous"/missing/unrecognized), norm not prohibitive — or no directional force | **0.7** = (1+floor)/2 | v9 |
| hedge on a prohibited/discouraged-governed flow | **0.5** (`rground_app_hedge_prohibit`) | v12a |
| false-forbid (said inappropriate, norm obligates) | **0.4** (`rground_app_floor`) | v9 |
| false-permit (said appropriate, norm prohibits) | **0.1** (`rground_app_floor_prohibit`) | v10 |

   Floors discount rather than annihilate (deontic-retrieval-noise guard). No-flow candidates carry no appropriateness labels → neutral m (and their score is discarded anyway under directional). `rground_app_weight=0.3` acts only as the on-switch in multiplicative mode; its magnitude is unused. Setting either prohibit knob `null` reproduces v10/v9 exactly; the formula version tag is `v12a_cost_sensitive_hedge`, and changing either knob bumps the prescreen cache signature.

   Rationale for the ladder: the fiction-derived governing norms are ~4:1 appropriate:inappropriate, so a symmetric floor makes permissive-when-unsure EV-optimal (v9 plateau: 53% false-permit on prohibited flows); the v10 floor priced the wrong-commit tail but left hedging the safe optimum (hedge mass frozen ~72%, GoldCoin Forbid recall pinned 0.55 < SFT 0.65); v12a widens the commit-vs-hedge gap 0.3→0.5 exactly where it binds. No-flow economics are untouched.

**Failure semantics** (all deliberate zero-advantage, never noise): parse failure → R_ground = 0.0 without judging; judge-failed group → uniform 0.5 for the whole group, wrong-universe side zeroed (a surviving wrong score would create spurious within-group gradient); degenerate zero embedding → routed to the judge-failed path; singleton group → grounding score alone (rank undefined). Watch `rground/judge_failed_group_frac` on W&B plus the stdout WARNING; per-call health streams under `rground/*` (`commit=False`, merged into TRL's step).

### Absolute mode (`rground_scoring: absolute`, legacy)

Per extracted flow: retrieve k=3 norms (cosine), judge scores 0.4·norm_match + 0.4·governance + 0.2·appropriateness_consistent; average over flows; `R_ground = clamp(correct − λ·wrong, 0, 1)`. No-flow completions get a coverage-judge path (`_score_no_flow_coverage`) mapping dual coverage + gold to [0,1]. Kept for May-sweep reproduction only.

## Contrastive scoring — two independent knobs

- **`contrastive_lambda` (production 1.0)** — per-completion dual evaluation, inherent to every R_ground call: score against the correct universe and one random wrong universe, subtract λ·wrong (grounding-vs-grounding in ranked mode, step 3 above). The wrong source is drawn per *group* per call in ranked mode (consistent within a group).
- **`contrastive_ratio` (production 0.0)** — legacy additive wrong-source dataset rows. The COLM sweep spans both axes; λ=1.0, ρ=0 is the paper-primary cell.

## Training-set construction (`_build_grpo_dataset` + prescreen)

- **No-flow downsampling**: source chunks are ~87% no-flow; capped at 1:1 with flow chunks.
- **Judgment vignettes** (`vignette_ratio: 0.3` pre-screen): generated from `governs_info_flow` norms with directional force; gold = `FORCE_TO_GOLD` (shared single source of truth with the deontic multiplier); scenario deliberately omits `norm_articulation` (answer leak). Optional separate vignette universe via `VIGNETTE_NORM_UNIVERSES_PATH` (set-but-invalid paths fail loud).
- **Variance pre-screening** (`prescreen.*`): 8 samples/prompt from the SFT policy at temperature 1.0; drop groups with reward std < 0.05 or unanimous abstention (`require_flow_variance`). Cache via `GRPO_PRESCREEN_CACHE` keyed on SFT checkpoint + reward-config signature. **This is a static off-policy filter** (variance under the SFT policy, not the moving policy) and it strips vignettes disproportionately and force-asymmetrically — the configured 0.3 is the *pre-screen* mix. Realized counts and the yes:no force mix, pre and post screen, are recorded in `training_metadata.json` (`n_vignettes_{yes,no}_{pre,post}_screen`) — quote those, not the configured ratio.
- **Held-out dev split: DISABLED in production** (`dev_fraction: 0.0`, since 2026-06-15): TRL's in-loop eval forward OOMs against the resident colocate vLLM engine. Promotion gates (`gates.py`, auto-run post-training → `promotion_gates.json` + `gates/*` in W&B) read the train reward curve and `reward_traces.jsonl` instead. Re-enable only with `vllm_mode=server`.
- Per-call vignette verdict health streams under `vignette/*`; offline forensics: `scripts/analyze_grpo_verdict_traces.py`.

## Deviations from vanilla GRPO — checklist for the camera-ready

Own these explicitly; each is standard or evidenced, but none should be described as plain GRPO:

1. **Listwise comparative R_ground** (not a per-sample reward) — see above. Includes bounded manufactured advantage from forced strict ranking of duplicates.
2. **`scale_rewards="none"`** (Dr. GRPO) — no group-std scaling; near-tie judge noise is not amplified to ±1; absolute tier gaps carry the signal.
3. **Token-level DAPO loss** (TRL 1.8.0 default `loss_type="dapo"`) — no per-completion length normalization bias.
4. **`mask_truncated_completions`** (DAPO overlong filtering).
5. **`token_truncate` vLLM importance-sampling correction** (GSPO-style) — clamps per-token instead of zeroing whole long completions.
6. **KL (β=0.02) to the SFT policy** as reference, computed by adapter-disable on the merged SFT model — an exact SFT anchor, magnitude a stability floor.
7. **SFT-policy variance prescreening** of the prompt set (selection bias, documented above).
8. **Not active despite being configured**: Clip-Higher (`epsilon_high=0.28` is inert at μ=1).

Known reward-hacking surfaces (accepted, monitored): keyword-count `r_judgment_reasoning` and token-Jaccard `r_norm_cite` (0.5 of the vignette reward combined); `r_uncert`'s confidence facet rewards high stated confidence.

## Ablation configs (Qwen3.5-9B)

Model yamls under `dagspaces/common/conf/model/qwen3.5-9b/`. OBSOLETE-headed yamls are pre-Qwen3.6-judge artifacts — archival only.

| Config | Variant | Status |
|---|---|---|
| `grpo-cratio-*`, `grpo-v2-lambda*`, `grpo-v3-*`, `grpo-{ctx,cohere,structural}-c10` | old-judge / book-only sweeps | OBSOLETE |
| `sft-and-progonly-grpo` | R_ground zeroed | model variant |
| `sft-and-grounded-grpo` | full reward | model variant |

The COLM λ×ratio sweep (15 cells) lives at `conf/sweep/contrastive_lambda_ratio_*.yaml` / `lambda_axis.yaml` / `ratio_axis.yaml`; the per-component reward ablation uses `confidence_fallthrough=true` (corrected facet-3 chain — changes composite values, keeper-repro requires `false`).

## Running

```bash
# COLM λ × ratio sweep (15 cells)
./scripts/launch_lambda_ratio_sweep.sh

# Single cell (paper-primary settings shown; requires external servers —
# JUDGE_MODEL must be Qwen3.6-27B for keeper-era reproduction, see Judge §)
python -m dagspaces.grpo_training.cli -m pipeline=grpo_only_online_external \
  model=qwen3.5-9b/sft-ci \
  training/grpo=online_rground_external \
  training.grpo.contrastive_lambda=1.0 \
  training.grpo.contrastive_ratio=0.0

# Programmatic-only (R_ground=0, weights redistributed)
python -m dagspaces.grpo_training.cli -m pipeline=grpo_programmatic_only \
  model=qwen3.5-9b/base training.grpo.use_vllm=false
```
