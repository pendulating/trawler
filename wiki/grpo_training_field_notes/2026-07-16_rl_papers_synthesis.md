# GRPO field notes — external RL-paper synthesis → v13a plan (μ=2 / Clip-Higher activation)

**Date:** 2026-07-16 · **Author:** paper synthesis + staging · **Status:** synthesis done; v13a staged, pre-launch

Continues [2026-07-03_v12a_plan.md](2026-07-03_v12a_plan.md) (v12a itself is
**still unrun** — its 2026-07-03 launch died with SLURM job 711954, CANCELLED at
0s elapsed, no checkpoint; `multirun/2026-07-03_grpo_v12a_hedge_tier/` is an
empty shell). The last *completed* run in the lineage is the **v11 probe**
([2026-07-01_v11_probe_midrun_forensics.md](2026-07-01_v11_probe_midrun_forensics.md)).

## Sources

Three external GRPO/RL-finetuning papers, read 2026-07-16 (local copies at repo root):

1. **Memory-R1** (ACL 2026, `2026.acl-long.583.md`) — RL-tunes a memory manager
   + answer agent with a pure **outcome reward** (exact-match correctness of the
   *downstream* answer). 152 training examples suffice; GRPO ≥ PPO.
2. **URPO** (AAAI 2026, `16953-AAAI26.LuS-NLP.md`) — unifies policy + reward
   model in one GRPO loop; preference data recast as generative ranking rewarded
   by Kendall's τ; DAPO tricks (no length norm, clip-higher 1.28, β=0).
3. **SEC '25 privacy rewriting** (`3769102.3774433.md`) — GRPO + 6-component
   composite reward for privacy-preserving rewriting; conditional reward
   activation (fires only on detected stylistic outliers); adversarial
   probe-classifier rewards instead of judge opinion.

## TL;DR

Most of the papers' algorithmic recommendations are **already in the keeper
recipe** (`training/grpo=online_rground_external`) — the audit below corrects an
initial mis-profile taken from `default.yaml`, which does NOT carry the keeper's
optimizer knobs. What remains live: (1) **Clip-Higher is configured but inert**
(`epsilon_high: 0.28` at `num_iterations: 1` → PPO ratio ≡ 1, v4 measured
`high_mean ~1e-4`), and the v9 plan's explicit deferral condition — "re-evaluate
μ>1 only if movement stalls" — is now met (prohibited-flow hedge mass frozen at
~72% across v10 + v11-probe; v12a unrun). → **v13a staged**: μ=2, single
variable vs the v11 probe. (2) Our reward is judge-opinion-heavy with no
**outcome-grounded** component; Memory-R1's Table 2 (LLM-judge reward → verbose
metric-gaming; EM outcome reward fixes it) is the external evidence that this
breeds exactly our hedging equilibrium. (3) Two code-level candidates deferred:
GDPO decoupled reward normalization, judge distillation with reasoning traces.

## Scorecard: paper recommendation vs keeper status

| Recommendation (source) | Keeper status (`online_rground_external.yaml`) |
|---|---|
| Remove per-response length normalization (URPO/DAPO) | ✅ already — TRL 0.29.1 default `loss_type="dapo"` (unset in our configs) |
| No group-std advantage scaling (Dr. GRPO) | ✅ already — `scale_rewards: "none"` (since Jun-10 redesign) |
| Exclude truncated completions (DAPO) | ✅ already — `mask_truncated_completions: true` |
| Per-token IS correction for variable length (GSPO) | ✅ already — `vllm_importance_sampling_mode: token_truncate` (v5) |
| KL as stability floor, not anchor | ✅ already — `beta: 0.02` (v7/v8; β·KL ≈ 0.001) |
| Verifiable-anchor data in the mix (URPO 0:1:1 collapse; Memory-R1 EM) | ✅ partially — `vignette_ratio: 0.3`, force→gold verifiable |
| Conditional reward activation (SEC '25 outlier-gated) | ✅ same pattern — v12a `rground_app_hedge_prohibit` tier (unrun) |
| **Clip-Higher / asymmetric clip (DAPO/URPO)** | ⚠️ **configured but inert** — `epsilon_high: 0.28` never binds at μ=1 (v4: `high_mean ~1e-4`; see yaml comment at the `num_iterations` block) |
| **Outcome reward from a frozen downstream consumer (Memory-R1)** | ❌ absent — all 6 components score extraction *quality*, none score *sufficiency for a correct downstream privacy decision* |
| Decoupled per-reward normalization (GDPO, TRL `normalize_then_sum`) | ❌ absent — composite is pre-blended inside `CompositeRewardFunction`; TRL sees one scalar, r_ground (0.5 wt) variance swamps the rest |
| Train/distill the judge with reasoning data (URPO RewardBench ablation) | ❌ absent — judge frozen; ties to the stalled reranker-distillation thread (zero-shot Spearman ~0.2) |
| Co-evolving unified player/referee (URPO core) | ❌ absent — deliberate: keeper reproducibility forbids a moving referee mid-run |

**Profile correction (for the record):** `default.yaml` carries none of the
optimizer knobs, so profiling it suggests TRL defaults (`scale_rewards="group"`,
symmetric clip, β=0). The keeper's training config is
`online_rground_external.yaml`, which overrides all of these. Any audit of "what
the keeper runs" must read that yaml plus the TRL-default fallbacks listed in
`grpo_training.py` (~line 1057).

## Live issue 1 — judge-opinion reward, no outcome anchor (Memory-R1)

Memory-R1 Table 2: rewarding with an LLM-judge score inflated the judge metric
while degrading F1/BLEU — the policy learned verbose judge-pleasing outputs.
Switching to outcome EM (does the *frozen downstream answerer* get the right
answer?) fixed it. Our composite is ~65% judge-opinion (r_ground 0.50 +
r_context 0.20 + r_cohere 0.10 of judged quality) and 0% outcome. The 72% frozen
hedge mass and the 0.55 Forbid-recall ceiling are the same signature: hedging is
the safe optimum under a quality judge that never pays a downstream price.

**Proposal (v14 candidate, code): outcome-grounded R_ground** — reward an
extraction by whether a frozen answerer reaches the correct privacy judgment
**given only the structured extraction** (no source text). Gold = the governing
norm's force (same `FORCE_TO_GOLD` as vignettes). A hedged/contentless
extraction cannot support the correct answer, so this attacks hedge EV at the
incentive root rather than penalizing hedge phrasing (v12a's approach). This is
the only proposal here that *changes the incentive* instead of the penalty.
Memory-R1 shows outcome rewards work at very small data scale (152 examples).

## Live issue 2 — Clip-Higher inert; v9 deferral condition met → **v13a**

Mechanism recap (from the yaml's `num_iterations` comment + v4/v8 traces): at
μ=1 the rollout logp snapshot is taken with no optimizer step before the single
loss pass, so the PPO ratio ≡ 1 and *neither* clip ever binds — the configured
`epsilon_high=0.28` has been dead weight in every run since v4. At μ=2 the
second inner pass departs from ratio 1 and Clip-Higher finally lets
advantage>0 (commit/extraction) tokens take asymmetrically larger up-steps.

History: v8 ran μ=2 on the pre-directional reward → first real held-out
movement (GoldCoin applicability 0.921→0.972) but an entropy breakout ~step 240
and an indiscriminate-permit bias → killed; v9 reverted to μ=1 with the
concentrating (directional) reward and said: *"re-evaluate μ>1 only if movement
stalls."* Movement has stalled — hedge mass ~72% and Forbid recall 0.55 held
across v10 and the v11 probe (3rd consecutive plateau).

**Honest counter-evidence:** the v11 exploration guard (~0.41–0.50) showed a
correct committer *exists* in ~half the groups and still loses — arguing hedge
**EV** binds, not exploration. But v5/v7's headline (+0.72 within-group
advantage present every step, unfollowed, grad_norm ~0.1) and v8's held-out
movement at μ=2 argue **update strength** binds. v12a (EV lever) and v13a
(update-strength lever) are disjoint mechanisms with disjoint trace fingerprints
sharing the v11 probe as control — running both disambiguates the frozen-hedge
cause regardless of which one moves it.

### v13a cell definition (staged: `scripts/run_grpo_v13a_mu2.sh`)

Single variable vs the **v11 probe**: `num_iterations: 1→2`. Everything else
v11-identical, which requires **pinning `rground_app_hedge_prohibit=null`** —
the yaml default is now 0.5 (the v12a value), and riding that default would
silently make v13a a two-variable cell. β=0.02 (μ>1 precondition, v8) and
`epsilon_high=0.28` pinned explicitly as cell-defining companions. Same top100
vignette universe auto-discovery as v11/v12a; own prescreen cache
(`cache/grpo_prescreen_v13a_mu2.json` — reward-identical to v11, but the cache
key schema grew `confidence_fallthrough` on 2026-07-14, so the old file would
miss anyway). `save_steps: 50` (yaml) gives pre-breakout checkpoints.

Falsifiable predictions:
- **Traces** (`scripts/analyze_grpo_verdict_traces.py`, table 3): prohibited-flow
  correct-commit share rises off ~0.10; hedge mass falls below ~0.70. Optimizer:
  `clip_ratio/high_mean` finally **nonzero** (the direct μ=2-activation check).
- **Held-out**: GoldCoin Forbid recall off 0.55 toward SFT 0.65 without the v8
  indiscriminate-permit mirror (watch Permit recall + applicability).
- **Kill criterion** (v8 fingerprint, via `trace_metrics.summarize_log_history`):
  entropy breakout (trend ↑↑ with corr(entropy, logp_diff) → +0.9) or IS
  collapse → kill and keep the best pre-breakout ckpt; if that reproduces v8's
  breakout *despite* the concentrating reward, μ>1 is ruled out for good and the
  update-strength hypothesis dies with it.

Run order vs v12a is the user's call; both need aux servers (embedding :8001,
judge :8002) and a free training GPU (SFT canonical sweeps currently occupy the
queue). Note added 2026-07-16: "arm 2" from the chat synthesis
(`scale_rewards: false`) was **resolved as moot** — already the keeper value.

## Deferred candidates (ranked)

1. **Outcome-grounded R_ground** (above) — biggest expected information, code.
2. **GDPO decoupled normalization** — split the composite into separate TRL
   reward functions + `normalize_then_sum`; saturated components (r_complete,
   r_consist) stop diluting, r_ground stops dominating group variance. Mechanical
   but touches the prescreen cache key. Do after v12a/v13a tell us whether
   normalization is even binding.
3. **Judge distillation with reasoning traces** (URPO: reasoning data improved
   the judge more than more preference data) — revives the reranker thread;
   distill (extraction, judge-reasoning, score) triples, not (extraction, score)
   pairs.
4. **Conditional prompt conditioning** (SEC '25) — when a chunk's universe is
   prohibition-dense, say so in the training rollout prompt. Cheap, but changes
   the task definition; needs care vs eval-prompt parity.
