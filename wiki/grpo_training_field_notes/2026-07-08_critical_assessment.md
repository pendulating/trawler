# GRPO redesign field notes — critical assessment: did it work?

**Date:** 2026-07-08 · **Status:** assessment (no new runs) · **Companion:** the same-day paper sync recorded in `papers/colm26_normative-simulacra/CONGRUENCE.md` (2026-07-08 entry)

A deliberately harsh, external-reviewer-style pass over the v1→v11 evidence,
written after syncing the manuscript to the v10/v11 findings. Verdict up front:
**GRPO did something real, but small — and the two headline claims are not
equally supported.** The GoldCoin gain is probably genuine; the PrivacyLens
"Pareto win" is noise-level by our own data; and attribution of either to
*normative grounding* is unproven until the programmatic-only ablation runs.

## What survives scrutiny

1. **The direction multiplier causally steers held-out verdict behaviour.**
   v8→v9 is the cleanest A/B in the project: one lever (one-directional →
   two-sided reward), pre-registered prediction, held-out flip as predicted
   (Forbid recall 0.35→0.55, precision 0.50→0.65). v10's floor change likewise
   produced the predicted over-permit signature on an *independent* benchmark
   (CIRL accepts +30%, acc 0.915→0.880). The lever works.
2. **The GoldCoin gain replicates.** Three training runs with three reward
   variants (v9/v10/v11) all land at best-arm compliance macro-F1 0.755–0.76 vs
   SFT 0.723 / base 0.713, matched batches. +3pts reproducing across three
   independent runs is unlikely to be seed luck.
3. **Capability control held.** MMLU flat (78.4/78.4/78.5) across the
   progression.
4. **The diagnostic/honesty work is strong.** Judge validation moved the
   metric *against* us (κ=0.47→0.79 swap, leakage ~doubles) and we reported it;
   the verdict-trace forensics (now paper §app:verdict-forensics) give a real
   mechanism for the ceiling.

## What does not survive

1. **PrivacyLens "Pareto win" is inside our own noise band.** Claimed margins
   ~1pp on n=493; the 2026-07-02 re-batch shows 1–2pp run-to-run judge
   variance, and the judge-ablation table shows the *ordering flips* between
   batches (there, GRPO leaks more than SFT: 0.481 vs 0.467). Two batches, two
   orderings = null result. Reframe as "no degradation on the privacy–utility
   frontier" — that claim is bulletproof; the current one isn't.
2. **Nothing attributes the gain to normative grounding.** The load-bearing
   ablation — GRPO with programmatic components only, no R_ground — was never
   run. Until it exists, +3 macro-F1 is attributable to *generic RL on
   structured extraction with a KL anchor*. Worse, λ=0 (no wrong-universe
   penalty) lands within 0.011 of λ=1.0 on every metric: the one component
   designed to force conditioning on the correct universe does nothing
   measurable downstream. Robustness reading vs decorative-mechanism reading —
   a hostile reviewer takes the second.
3. **Full pipeline vs base is roughly a wash.** Base → SFT+GRPO: compliance
   +4.2, CIRL −6.0 (97.2→91.3, reject-skewed gold), VLM −3.8, QA-probe −1.7,
   ConfAIde +0.1 (SFT gained 5.5, GRPO gave it back), PrivacyLens flat. The
   defensible claim is narrow: GRPO partially repairs pathologies SFT itself
   introduced, plus one benchmark family improves.
4. **The training barely trains.** Mean KL to SFT 0.037–0.045; held-out reward
   gain +0.015 over the whole run vs within-group std 0.256; verdict policy
   frozen by epoch 2 on 704 prompts; three consecutive reward revisions failed
   their pre-registered ceiling predictions (Forbid recall 0.55 ×3); hedge mass
   72% immovable (hedge EV, not exploration — committer present in ~half the
   groups and loses).
5. **The cheapest lever rivals the most expensive one.** The SFT pair-format
   factorial: toggling the *appropriateness field* in SFT data moves GoldCoin
   compliance +4.7 — more than the entire RL stage.
6. **No uncertainty quantification where it matters.** The 5-seed sweep
   measured *training-reward* CV (and predates v9); it says nothing about
   held-out benchmark variance across retrains. n=107 compliance / n=20 Forbid:
   +3.2 macro-F1 is a couple of case flips. No bootstrap CIs anywhere.
   Cross-model rows (CR-7B compliance 57.0 < zero-shot Qwen) smell like harness
   confounds.

## Recommendations, ranked by information-per-GPU-hour

1. **Programmatic-only GRPO ablation** (no R_ground; one run + judge-free
   evals). Thesis-critical: if it matches v9, the normative-grounding claim
   dies; if v9 beats it, the central claim finally has support. Either outcome
   is honest and publishable. More important than v12a *for the paper*.
2. **Error bars on the headline.** Two more v9 seeds → ckpt-100 → GoldCoin +
   ConfAIde (judge-free, cheap), plus bootstrap CIs on the n=107 table.
3. **v12a: run it (staged, cheap) but believe the pre-registration** — the plan
   itself predicts Forbid recall stays ≈0.55. Trace-level prediction falsifies
   at ~⅓ run; kill early if the tier table is v11-shaped.
4. **To move the ceiling, the data is the lever**: the ~32h top100 *flows* run
   attacks all three diagnosed constraints at once (704-prompt smallness, ~5:1
   permissive skew, epoch-2 freeze). Only untried intervention aimed at the
   binding constraint rather than the reward's pricing of it.
5. **Reframe PrivacyLens** in the manuscript per (1) above.

## Related

- [README.md](README.md) — the v1→v12a saga this assesses
- [2026-07-03_v12a_plan.md](2026-07-03_v12a_plan.md) — staged next run
- [2026-07-01_v11_probe_midrun_forensics.md](2026-07-01_v11_probe_midrun_forensics.md) — the forensics underpinning §4 above
- `papers/colm26_normative-simulacra/CONGRUENCE.md` (2026-07-08) — the manuscript sync applying the reporting-side fixes
- [[project-grpo-flat-reward]]
