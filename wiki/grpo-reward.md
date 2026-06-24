# GRPO Composite Reward

GRPO (Shao et al. 2024) samples `G` completions per prompt, scores each via `R`, and updates policy from within-group relative rankings — no preference pairs, no separate reward model. CI's structured outputs admit programmatic verification, so most reward components are deterministic.

> **Judge:** Qwen3.6-27B (`/share/pierson/matt/zoo/models/Qwen3.6-27B`, TP=2 on klara). Served by `scripts/judge_server.sub` on port 8002 (or `scripts/launch_auxiliary_servers.sh` on port 8002). Older runs that used Qwen2.5-72B-Instruct-AWQ are flagged OBSOLETE in their model yamls.

> **Optimizer recipe (2026-06):** the May λ×ρ sweep trained with G=2 / lr=1e-6 / no KL anchor and produced a flat reward curve (near-no-op training). The production config `training/grpo/online_rground_external.yaml` now uses G=8, lr=1e-5, 3 epochs, `beta=0.01` (KL to the SFT reference), `scale_rewards="none"`, `mask_truncated_completions=true`. Details + evidence: `wiki/changelog/2026-06-09_grpo_phase1_optimizer_revision.md`. Checkpoints from the old recipe (`grpo-l*-r*` yamls) are not comparable to new-recipe runs.

> **Reward redesign (2026-06, Phases 2–5):** production now also uses `rground_scoring: ranked` (one listwise judge call per group instead of per-flow absolute scores — fixes the 60%-tied-groups problem), `reward_composition: gated` (R = gate × discriminative), SFT-policy variance **pre-screening** of prompts (`prescreen.*`, cache via `GRPO_PRESCREEN_CACHE`), `vignette_ratio: 0.3`, a `dev_fraction: 0.05` held-out reward eval, and **promotion gates** (`scripts/check_grpo_promotion_gates.py` — run before spending eval compute on a cell). Details: `wiki/changelog/2026-06-09_grpo_phase2-5_reward_redesign.md`.

> **v9 two-sided reward (2026-06-23):** after v8's ground-truth eval showed GRPO learned an indiscriminate "engage/permit" bias (GoldCoin compliance Forbid recall 0.70→0.35) because the reward was one-directional, production moves to `reward_composition: directional`: `R = gate · content · direction` for extractions, `no_flow_reward(gold)` for abstentions. The **appropriateness direction** (deontic Raz force → expected appropriateness, `deontic.direction_multiplier`) is promoted from a 0.3 *additive* blend inside R_ground to a *multiplier-with-floor* (`rground_app_mode: multiplicative`, `rground_app_floor: 0.4`), so mis-judging a violation costs a large fraction of the reward instead of a diluted sliver. This drops the post-hoc `abstention_penalty` and the gated `gate×disc`. Rationale, the hedge-equilibrium evidence, and falsifiable predictions: `wiki/grpo_training_field_notes/2026-06-23_v9_plan.md` (and `2026-06-23_v8_results_groundtruth.md`).

## Components

$R = \sum_i w_i R_i$ over six components. Gating signals saturate after SFT (low weight); discriminative signals carry the GRPO learning signal (high weight).

| # | Component | Weight | Type | What it measures |
|---|---|---|---|---|
| 1 | `R_uncert` (task clarity) | 0.10 | gating, programmatic | Schema validity + construct discrimination + extraction confidence |
| 2 | `R_complete` (structural completeness) | 0.05 | gating, programmatic | Presence of all required IFT fields |
| 3 | `R_consist` (internal consistency) | 0.05 | gating, programmatic | Reasoning ↔ extraction non-contradiction |
| 4 | `R_context` (context identification) | 0.20 | discriminative, programmatic | Stated context matches a prominent context in `N_b` |
| 5 | `R_cohere` (reasoning-to-extraction coherence) | 0.10 | discriminative, programmatic | Reasoning trace supports the extracted tuples |
| 6 | **`R_ground` (normative grounding)** | **0.50** | **discriminative, LLM judge** | **Judge evaluates flow against retrieved norms from `N_b`** |

Rationale for `R_ground=0.5`: pilots showed it has the **highest inter-completion variance**, making it the primary driver of GRPO's advantage estimates.

> **Vignette rows are scored separately.** Rows mixed in via `vignette_ratio` (`task_type: norm_judgment`) bypass the 6-component composite entirely (neither additive nor gated) and receive a direct 3-component judgment reward: `0.5·r_judgment + 0.25·r_judgment_reasoning + 0.25·r_norm_cite` (`judgment_reward_weights`). The R = Σ wᵢRᵢ formula applies only to CI-extraction rows — say so in the paper. Also note: pre-screening can strip vignettes disproportionately, so the configured `vignette_ratio: 0.3` is the *pre-screen* mix; the realized training-set mix is recorded in `training_metadata.json` as `n_vignettes_post_screen` (vs `n_vignettes_pre_screen`).

## `R_ground` details

`rground_scoring` selects between two judge protocols. Both produce a scalar per completion — GRPO itself never sees ranks; it centers scalar rewards against the group mean, so its entire learning signal is the *within-group spread* of those scalars.

### Absolute mode (`rground_scoring: absolute`, legacy)

For each extracted flow in a completion:
1. Retrieve `k=3` most similar norms from $\hat{\mathcal{N}}_b$ via semantic similarity (Qwen3-Embedding-8B + cosine).
2. LLM judge (**Qwen3.6-27B**, served on klara:8002 by `scripts/judge_server.sub`) scores three criteria:
   - **Norm awareness**: invoked norms match retrieved norms
   - **Flow governance**: this flow is governed by the retrieved norms
   - **Appropriateness consistency**: appropriateness judgment agrees with the governing norm

### Ranked mode (`rground_scoring: ranked`, production since 2026-06)

One **listwise** judge call per (group, universe): the judge sees all `G` same-prompt completions at once, must produce a strict ranking (distinct ranks, no ties), and assigns each candidate an absolute grounding score. `_rankings_to_scores` converts to a scalar:

$$s_i = w_r \cdot \frac{n - \text{rank}_i}{n - 1} + (1 - w_r) \cdot \text{grounding}_i$$

with `rank_weight` $w_r$ (default 0.5). The contrastive penalty still applies: the wrong-universe pass uses grounding scores only (ranks are meaningless across universes), and $R_{\text{ground}} = \text{clamp}(s_{\text{correct}} - \lambda \cdot \text{grounding}_{\text{wrong}}, 0, 1)$.

**Why listwise instead of absolute (2026-06 finding).** LLM judges are poorly calibrated at absolute scoring but sharp at comparative judgment. In the May λ×ρ sweep the absolute judge's scores were quantized and bimodal (mass at 0.0 and 0.85–1.0), so same-prompt completions usually tied: **60% of groups had identical `R_ground`** → zero advantage → the dominant reward component (weight 0.5) contributed nothing to the gradient for most groups. The listwise judge is *forced* to discriminate between completions an absolute judge would score identically, guaranteeing non-zero `R_ground` spread in every group. The `rank_weight` blend keeps an absolute anchor so a uniformly-bad group isn't rewarded just for winning a weak contest, and unjustified no-flow declarations get ranked last (the anti-collapse signal). Judge-failed groups fall back to uniform 0.5 — deliberately zero advantage, never noise; watch `rground/judge_failed_group_frac` on W&B and the stdout WARNING (see `wiki/changelog/2026-06-10_judge_response_format_fix.md` for the failure mode that motivated the loud logging).

Implementation lives under `dagspaces/grpo_training/runners/grpo_training.py` and the reward modules it imports; judge calls go through `dagspaces/common/judge_client.py` (HTTP) when `use_external_judge=true`.

`R_context` uses a separate, much smaller `all-MiniLM-L6-v2` sentence-transformers embedder (`context_embedding_model` knob in `training/grpo/online_rground_external.yaml`). This is **distinct** from the Qwen3-Embedding-8B used by `R_ground` norm retrieval.

## Per-completion contrastive scoring (two independent knobs)

There are **two** independent contrastive mechanisms — keep them straight:

### `contrastive_lambda` — per-completion penalty weight
Every `R_ground` call always scores the completion against **both** the correct universe $\hat{\mathcal{N}}_b$ and a randomly chosen wrong universe $\hat{\mathcal{N}}_{b'}$:

$$R_{\text{ground}} = \text{clamp}\!\left(\bar{r}_{\text{correct}} - \lambda \cdot \bar{r}_{\text{wrong}},\ 0,\ 1\right)$$

λ is `training.grpo.contrastive_lambda` (default 1.0 in `online_rground_external.yaml`, aligned 2026-06-09 with the paper-primary sweep origin λ=1.0, ρ=0; sweep cells always pin λ explicitly). This dual evaluation is **inherent** to the scoring mechanism and applies to every completion regardless of `contrastive_ratio`.

### `contrastive_ratio` — additive wrong-source row fraction
A **separate, legacy** mechanism (`training.grpo.contrastive_ratio`, default 0.0) that injects additional rows into the GRPO training dataset where the model is told its task involves the *wrong* source. These additive rows reach R_ground with naturally low scores (since the policy will extract correct-source norms while being judged against wrong-source norms). Set ratio=0 to disable; ratio=0.1 means 10% extra wrong-source rows.

Both mechanisms can be active simultaneously. The COLM paper sweeps both, with ratio=0 as the canonical λ axis.

## Ablation configs (Qwen3.5-9B)

Model yamls under `dagspaces/common/conf/model/qwen3.5-9b/`. The OBSOLETE-headed yamls are pre-Qwen3.6-27B-judge artifacts kept only for archival reproducibility — do not use for paper results.

| Config | Variant | Status |
|---|---|---|
| `grpo-cratio-{00,05,10,15,20,50,100}` | `contrastive_ratio` sweep (0.0, 0.05, 0.10, 0.15, 0.20, 0.50, 1.00) | OBSOLETE — old judge |
| `grpo-v2-lambda{05,10}` | `contrastive_lambda` sweep (0.5, 1.0) | OBSOLETE — old judge |
| `grpo-v3-*` | Vignette-mix variants | OBSOLETE — book-only training |
| `grpo-ctx-c10` | Context-only reward ablation | OBSOLETE — old judge |
| `grpo-cohere-c10` | Coherence-only reward ablation | OBSOLETE — old judge |
| `grpo-structural-c10` | Structural-only reward ablation | OBSOLETE — old judge |
| `sft-and-progonly-grpo` | Phase1+Phase2 with `R_ground` zeroed | (model variant, see file) |
| `sft-and-grounded-grpo` | Phase1+Phase2 with full reward | (model variant, see file) |

The current COLM paper sweep (15 cells over λ × ratio with Qwen3.6-27B judge) lives at `dagspaces/grpo_training/conf/sweep/contrastive_lambda_ratio_*.yaml` and produces new model yamls after training completes (named `grpo-lambdaXX-ratioYY.yaml`).

## Running reward ablations

```bash
# Sweep contrastive_lambda × contrastive_ratio (COLM paper sweep — 15 cells)
./scripts/launch_lambda_ratio_sweep.sh

# Single-cell training (manual)
python -m dagspaces.grpo_training.cli -m pipeline=grpo_only_online_external \
  model=qwen3.5-9b/sft-ci \
  training/grpo=online_rground_external \
  training.grpo.contrastive_lambda=1.0 \
  training.grpo.contrastive_ratio=0.0

# Programmatic-only (R_ground=0, weights redistributed)
python -m dagspaces.grpo_training.cli -m pipeline=grpo_programmatic_only \
  model=qwen3.5-9b/base training.grpo.use_vllm=false
```
