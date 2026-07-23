# 2026-07-20 — GRPO methodology congruency review (wiki ↔ code ↔ paper)

Full read of `wiki/grpo-reward.md` against the production implementation
(`dagspaces/grpo_training/conf/training/grpo/online_rground_external.yaml`,
`stages/grpo_training.py`, `stages/rewards.py`, `stages/online_rground.py`,
`stages/deontic.py`) and the installed TRL 1.8.0 defaults. Purpose: make the
wiki 1:1 with the code before the camera-ready update, and state precisely
where the method deviates from vanilla GRPO (Shao et al. 2024) so the paper
can own those deviations rather than mis-describe them.

**Verdict:** mechanics are sound; every optimizer-level deviation is
deliberate and field-standard (Dr. GRPO, DAPO, GSPO). One genuine
GRPO-tenet deviation must be framed honestly in the paper (§1). The wiki
page had drifted from the code in five places (§3) — **fixed 2026-07-20**
by rewriting `grpo-reward.md` from the code.

## 1. GRPO-tenet deviation the paper must own

GRPO's advantage estimator assumes the reward is a function of
`(prompt, completion)` alone. In production ranked mode, **R_ground is a
listwise comparative reward** — it depends on the sibling completions in
the group, three ways:

1. **Rank component**: `s_i = w_r·(n−rank_i)/(n−1) + (1−w_r)·g_i`
   (`online_rground.py:85-123`) is defined only relative to the group.
2. **Group-dependent retrieval**: the shared norm set comes from the mean
   of the chunk embedding and *every* candidate's flow-query embeddings
   (`online_rground.py:643-651`), so the evidence a completion is judged
   against depends on what its siblings extracted.
3. **Forced strict ranking of duplicates**: the judge must emit distinct
   ranks, so byte-identical extractions (common at G=8 sampling) receive
   different rewards spaced `rank_weight/(n−1)` ≈ 0.07 apart — manufactured
   advantage noise. Bounded under `scale_rewards="none"`, but it means
   every group carries nonzero gradient even when all completions are
   equally good.

**Camera-ready framing:** present R_ground as a within-group listwise
comparative reward (chosen because absolute judge scores tie in 60% of
groups → zero advantage), not as a standard scalar reward model that GRPO
happens to center. Do **not** print `R = Σ wᵢRᵢ` as the production
composition (see §3.3).

**Connected hypothesis (unverified):** guaranteed rank spread + the
`cosine_with_min_lr` floor (30% of peak, forever) means the policy keeps
taking gradient steps from rank noise after real signal saturates —
consistent with held-out behaviour peaking at early checkpoints (v8
ckpt200, v9 ckpt100) and degrading after.

**Cheap mitigations (open, for redesign runs):** dedupe identical
`candidate_texts` before ranking and share the score; assert
`len(judged) <= num_generations` to catch the duplicate-prompt group-merge
edge case (`online_rground.py:564-570` keys groups by prompt text; two
dataset rows with identical formatted prompts would merge two TRL groups of
8 into one 16-candidate ranking while TRL still centers per 8).

## 2. Second-order findings (code behaviour, all currently benign)

- **`epsilon_high=0.28` is inert at μ=1** (the production setting since
  v9). The config still ships it and `training_metadata.json` records it.
  Do not report Clip-Higher as an active part of the v9+ recipe.
- **`loss_type` is silently inherited from TRL.** Nothing in the repo sets
  it; TRL 1.8.0 defaults to `"dapo"` (token-level global normalization —
  which also covers Dr. GRPO's length-bias fix, so current behaviour is
  fine). But the knob-forwarding loop (`stages/grpo_training.py:1070-1075`)
  pins every *other* objective-shaping knob precisely so TRL defaults can't
  drift. `loss_type` should be pinned the same way before any TRL upgrade.
- **No-flow candidates distort rank granularity, then their R_ground is
  discarded.** Under `reward_composition: directional` a no-flow completion
  is scored by `no_flow_reward(gold)` (`rewards.py:1266-1271`) — but it
  still occupies a rank slot in the listwise call
  (`online_rground.py:594-599`). Wasted judge tokens, and the rank spacing
  `1/(n−1)` for extraction candidates varies with the group's abstention
  mix.
- **Prescreening is a static off-policy filter**: variance measured under
  the SFT policy at step 0; prompts that would become informative as π
  moves are pruned forever. Documented efficiency trade-off; realized mixes
  audited in `training_metadata.json`. State the selection bias in the
  paper (force-asymmetric vignette stripping already documented).
- **Half the vignette reward is lexically gameable**:
  `r_judgment_reasoning` is a keyword count (`rewards.py:692-727`),
  `r_norm_cite` token Jaccard (`rewards.py:730-751`) — together 0.5 of the
  vignette reward. `vignette/*` health tracks verdict drift but nothing
  watches reasoning-keyword inflation.
- **`r_uncert` facet 3 rewards high self-reported confidence, not
  calibration** (`rewards.py:237-277`). Do not call it "calibration" in the
  paper. (Also carries the documented keeper-repro bug behind
  `confidence_fallthrough: false`.)

## 3. Wiki ↔ code inconsistencies — fixed in the 2026-07-20 `grpo-reward.md` rewrite

1. **Judge identity needed precision, not correction.** Keeper-era GRPO
   runs (v9–v12a, all paper results) used **Qwen3.6-27B** — still the
   dagspace default (`conf/config.yaml:32`). But the 2026-07-16 Gemma stack
   migration changed `scripts/judge_server.sub:54` to default to
   **Gemma-4-31B-it**. Launching the judge server today without
   `JUDGE_MODEL=` serves a model whose name mismatches the dagspace's
   `judge_model.model_source` → API calls fail (or, if overridden
   carelessly, silently judge with the wrong model). Reproduction of
   keeper-era reward requires `JUDGE_MODEL=/share/pierson/matt/zoo/models/Qwen3.6-27B`.
2. **`dev_fraction`**: wiki claimed production runs a 0.05 held-out reward
   eval; production is `0.0` (disabled 2026-06-15 — colocate OOM, yaml
   lines 291-304). Promotion gates read train-side signals instead.
3. **Composition formula**: the wiki's Components section headlined
   `R = Σ wᵢRᵢ` and typed `r_cohere` "discriminative" — the additive/gated
   era. Production is `directional`: gate = {r_uncert, r_complete,
   r_consist, **r_cohere**}, content = {r_context, r_ground}, R = gate ×
   content. The old page even instructed "say so in the paper" with the Σ
   formula — following it would have printed a formula no v9+ run used.
4. **Effective weights**: under directional normalization r_ground's
   content-side weight is 0.5/0.7 ≈ **0.71** and r_context's ≈ 0.29 (gate
   side: r_uncert ⅓, r_cohere ⅓, r_complete ⅙, r_consist ⅙ of 0.30). The
   paper should report normalized weights.
5. **Ranked-mode contrast formula was pre-v8**: wiki printed
   `R = clamp(s_correct − λ·g_wrong)`; since v8 the contrast is
   grounding-vs-grounding *inside* the `(1−w_r)` term, rank component
   contrast-free (`online_rground.py:800-838`).

## 4. Housekeeping

- `conf/config.yaml:30` referenced a deleted `CONGRUENCE.md` — repointed to
  this page.
- `wiki/README.md` index updated (this entry + grpo-reward.md description).

## Open action items

- [ ] Pin `loss_type: dapo` explicitly in `online_rground_external.yaml`
      (and the redesign configs) before any TRL upgrade.
- [ ] Camera-ready: describe R_ground as listwise comparative (§1); report
      normalized directional weights (§3.4); tier ladder 1.0/0.7/0.5/0.4/0.1;
      do not claim Clip-Higher active; do not call facet 3 "calibration".
- [ ] Redesign runs: dedupe identical candidates before ranking; group-size
      assert; consider skipping no-flow candidates in the listwise call when
      composition is directional (their score is discarded).
