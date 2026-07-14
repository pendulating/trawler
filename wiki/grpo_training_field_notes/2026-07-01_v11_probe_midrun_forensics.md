# GRPO redesign field notes — v11 probe mid-run forensics

**Date:** 2026-07-01 · **Status:** probe COMPLETED 2026-07-02 01:04 (job 488187, 528/528 steps, exit 0); held-out sweeps LAUNCHED (see update banner) · **Run:** `multirun/2026-06-30_grpo_probe_top100_vignettes/11-53-47`

> **Update (2026-07-02) — run completed; full-trace forensics confirm the mid-run
> picture; held-out sweeps launched.** Final-bin numbers (full 545-call trace):
> realised vignette mix **2.15:1**; gold-"yes" acc 0.67→0.74 (climbing, the
> dominant vignette gradient), gold-"no" acc 0.94→0.89 with says-yes drift
> 0.01→**0.07** (the predicted mild over-permit creep); prohibited-flow hedge mass
> 0.70→0.74→0.71 with correct-commit *declining* 0.12→0.08 late; exploration guard
> 0.46→0.41. Promotion gates: `promote: false` on `no_flow_rate` (0.60 vs gold
> 0.27) — but v10 failed the same gate at 0.57, so pattern-normal; the probe
> **passes** the two gates v10 failed (reward trend +0.003 vs −0.002; mean KL
> **0.045** vs v10's 6.3) — the healthiest training mechanics of any arm yet.
> Held-out: GoldCoin 7-arm ckpt sweep (`scripts/run_eval_v11probe_goldcoin.sh`,
> SLURM array 680686) + ConfAIDE 5-arm (`scripts/run_eval_v11probe_confaide.sh`,
> array 680687), both judge-free, launched 2026-07-02 13:45. Model yamls:
> `qwen3.5-9b/v11probe-ckpt{50,100,150,200,350,528}`.

Continues [2026-06-27_v11_plan.md](2026-06-27_v11_plan.md). The plan's "blocked on
aux servers" note went stale within hours: the klara servers came back and the
probe launched 2026-06-30 11:53 via `scripts/run_grpo_probe_top100vig.sh`
(prescreen kept 704/1103 prompts, fresh cache, checkpoints every 50 steps).
This note reports verdict-behaviour forensics mined from the probe's live
`reward_traces.jsonl`, side-by-side with the identical analysis on v10 — run
**before** the held-out sweep, so the predictions below are pre-registered.

Reproduce (tool added this session, with the same-numbered tables):

```bash
python scripts/analyze_grpo_verdict_traces.py \
  multirun/2026-06-30_grpo_probe_top100_vignettes \
  multirun/2026-06-24_grpo_redesign_full_v10
```

## TL;DR

The rebalance **landed** (realised vignette mix 2.08:1 vs v10's 5.2:1) and it
**halts the verdict erosion** v10's skewed mix was causing — but the forensics
also show the vignette task **was never permissive-biased in the first place**
(gold-"no" accuracy 0.94 at the start of training), and the extraction-side
hedge mass on prohibited-governed flows is **frozen at ~72%, byte-for-byte the
v10 fingerprint**. Lever (a) fixes a real but second-order problem. Expect the
held-out GoldCoin Forbid recall to stay ≈0.55; the binding constraint is the
**hedge economics of the extraction reward**, not the judgment-prompt base rate.

## (1) The rebalance landed — and quantifies the screen's force bias

| | candidate pool | realised (completion level) |
|---|---|---|
| v10 (fiction10 vignettes) | 3.07:1 | **5.2:1** (1208:232) |
| v11 probe (top100 vignettes) | 1.72:1 | **2.08:1** (848:408) |

The force-blind variance screen + sampling drift roughly *doubled* the skew in
v10; the probe got off lightly. This drift was previously invisible — the
realised ratio had to be mined from traces. Fixed this session: realised gold
mix is now logged pre/post screen in `training_metadata.json`
(`n_vignettes_{yes,no}_{pre,post}_screen`), in `prescreen_report.json`
(`vignette_{yes,no}_{in,kept}`), and per-call per-class accuracy/drift streams
to W&B under `vignette/*` (`CompositeRewardFunction._push_vignette_health`) —
every completion, not the 8-per-call trace sample.

## (2) The surprise: the vignette task was never permissive-biased

Per-gold-class verdict behaviour over training (thirds of the run):

| | gold="yes" acc (early→late) | gold="no" acc (early→late) | gold-"no" yes-rate |
|---|---|---|---|
| v10 (5.2:1 mix) | 0.56 → 0.64 | 0.84 → **0.77** | 0.12 → **0.20** (eroding) |
| v11 probe (2.08:1 mix) | 0.64 → 0.75 | 0.94 → 0.91 | 0.01 → 0.05 (near-flat) |

Two readings:

- **Within-task, lever (a) works**: v10's skewed mix was actively *teaching
  permissiveness* on the one place it could (gold-"no" yes-rate 0.12→0.20);
  the balanced mix halts that erosion. If ConfAIDE-2b (the judgment-shaped
  held-out metric where GRPO regressed below SFT, 63.2 vs 68.6) moves anywhere,
  it should be here.
- **But the premise of the v11 mechanism is falsified as stated**: the policy
  could already commit "no" on prohibited-governed vignettes at 0.84–0.94
  *before any GRPO*. The "EV-optimal when-unsure verdict is permissive" story
  never manifested as permissive vignette answers. The low-accuracy class is
  gold-"yes" (0.64), so the dominant vignette gradient in the probe teaches
  *"say yes more"* — watch the late checkpoints for over-permit drift
  (`vignette/says_yes_gold_no` creeping 0.01→0.05 already).

## (3) The binding constraint is untouched: extraction hedge mass frozen

Direction-multiplier tier mass on prohibited/discouraged-governed flows
(the behaviour that maps to GoldCoin compliance verdicts):

| | correct-commit ×1.0 | hedge ×0.7 | false-permit ×0.1 |
|---|---|---|---|
| v10 (early→late) | 0.11 → 0.06 | 0.79 → 0.78 | 0.10 → 0.11 |
| v11 probe (early→late) | 0.09 → 0.10 | 0.74 → 0.71 | 0.16 → 0.12 |

~72% hedge, all run, both arms. Structurally expected — the vignette lever
does not enter the extraction reward path — but this is the second consecutive
iteration (after v10's floor) that left it unmoved.

**Exploration guard** (new metric): fraction of traced prohibited-governed
groups containing ≥1 correct commit — probe 0.36→0.50, v10 0.32→0.35. The
within-group gradient toward committing *exists in roughly half the groups*
and still doesn't win. That points at **hedge EV, not exploration starvation**:
under `R_ground = base × direction`, a correct commit's direction edge over a
hedge is only 1.0-vs-0.7, and a hedging completion with better
grounding/rank (rank_weight 0.5) routinely outscores a committing one with
mediocre grounding. The 0.1 false-permit floor (v10) punished the wrong-commit
tail but left hedging the safe optimum.

## Pre-registered predictions for the held-out sweep

- **GoldCoin Forbid recall stays ≈0.55** (the v9/v10 plateau). NB the Forbid
  split is n=20 — 0.55→0.65 is two cases; a "hit" or "miss" here is weak
  evidence either way. Judge the probe on the full compliance macro-F1 (n=107)
  and the metrics below, not Forbid recall alone.
- **ConfAIDE-2b is the metric lever (a) should help** (halted judgment-verdict
  erosion): include it in the sweep alongside GoldCoin.
- **Over-permit watch**: if gold-"no" vignette accuracy keeps sliding late
  (`vignette/acc_gold_no`), later checkpoints may *lower* Forbid recall —
  prefer earlier checkpoints, consistent with the v8/v9/v10 early-peak pattern.

## If Forbid recall is flat: the pivot (v12), ranked

1. **v12a — cost-sensitive *hedge* tier (single reward variable).** Extend
   `deontic.appropriateness_multiplier` with `hedge_prohibit` (~0.5): correct
   1.0 / hedge-on-prohibited 0.5 / false-permit 0.1; hedge elsewhere stays 0.7.
   Nearly doubles the commit-vs-hedge gap (0.3→0.5) exactly where it binds;
   composes with v10's floor. Bump `rground_formula_version` in
   `prompt_screening._reward_signature`. Falsifiable on the traces this note's
   script reads: prohibited-flow correct-commit share off ~0.10.
2. **v12b — lever (b), prohibited-rich extraction prompt upweighting.** Raises
   the *frequency* of directional-gradient events but not the within-group EV
   ordering that keeps hedging safe; run with/after v12a, not instead.
3. **top100 flows run (~32h)** if anything moves directionally: 704 prompts ×
   3 epochs is small and v10's verdict policy froze by epoch 2 — fresh,
   better-balanced extraction prompts attack both the marginal prior and the
   freeze.

## Related

- [2026-06-27_v11_plan.md](2026-06-27_v11_plan.md) — the plan this probes (lever (a))
- [2026-06-24_v10_plan.md](2026-06-24_v10_plan.md) — the cost-sensitive floor whose "honest limit" the frozen hedge mass re-confirms
- `scripts/analyze_grpo_verdict_traces.py` · [grpo-reward.md](../grpo-reward.md) · [[project-grpo-flat-reward]]
