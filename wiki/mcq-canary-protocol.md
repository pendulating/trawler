# MCQ Canary Protocol — per-epoch capability/format probe for SFT checkpoints

**Status: TBD — designed 2026-07-18, deliberately not implemented.** Decision
was to ship the 2026-07-18 SFT guards (best-epoch selection on eval_loss,
grad-spike/init-loss/clip alarms, template preflight) without this and revisit
after the canonical DFT sweep. This page records the protocol so it can be
implemented without re-deriving the methodology constraints.

## The gap it fills

`training/sft/default.yaml` documents the limitation of the held-out eval
split: eval_loss is computed on the *same task in the same target format*, so a
model that over-commits to the `{"reasoning": ...}` output shape scores
**better** on eval_loss while getting **worse** on every out-of-format
benchmark. Both observed instances of this failure mode were invisible until
the downstream benchmark sweep, days later:

- openthinker3-7b: abandoned ReAct for JSON on 493/493 PrivacyLens vignettes
- gpt-oss-20b: 33% empty harmony final-channel outputs

eval_loss cannot catch this by construction. Catching it needs an
**out-of-format probe** — that is what the canary is.

## The protocol

Per saved epoch checkpoint, run a small fixed MCQ probe through the model's
**native serve-time chat template** (not the training template — the point is
to measure the serve-time behavior the benchmarks will see):

- **Probe set:** ~500 questions from the **MMLU validation split**, fixed once
  (fixed seed, fixed question list checked into the repo) and reused for every
  model and every epoch. Binomial noise at n=500 is ≈ ±2pt, so a 5pt relative
  threshold is ~2σ.
- **Two metrics, never pooled** (per the [metric-trust.md](metric-trust.md)
  FAIL-gate convention):
  1. **parseable-rate** — fraction of responses yielding an extractable
     letter. This is the *format canary*: it is the metric that catches the
     openthinker3/gpt-oss failure class, at epoch 1 instead of after the
     benchmark sweep.
  2. **accuracy-among-parseable** — the *capability probe* (true
     forgetting/degradation signal).
- **Baseline:** the base model's own probe numbers, measured once per model
  with the same probe set and serve config.
- **Trigger rule (pre-registered):** flag the checkpoint if
  parseable-rate drops >10pt absolute OR accuracy-among-parseable drops >5%
  relative vs the base model. Thresholds fixed *before* the sweep; changing
  them after seeing results forfeits the pre-registration claim.
- **Action:** loud warning + trace/W&B entry, and the flagged checkpoint is
  ineligible for promotion (to GRPO or to eval_all). Auto-*stop* of training
  is an opt-in knob, not the default — best-epoch selection stays on
  eval_loss, and the canary **vetoes**, it does not **select**.

## Methodology constraints (why the design looks like this)

These came out of the 2026-07-18 review discussion; violating either one turns
a rigor feature into a reviewer objection:

1. **Split-level firewall — never select on what you report.** MMLU appears in
   eval_all as capability-retention evidence. If checkpoint selection consults
   the *reported* MMLU numbers, the "CI training preserves general capability"
   claim becomes circular (the checkpoint was filtered to satisfy it). Hence:
   monitor on the **validation split**, report the **test split**, and keep the
   probe question list disjoint from anything quoted in the paper.
2. **Pre-registration.** The trigger rule must appear in the paper's training
   protocol description ("training checkpoints failing a capability probe
   [thresholds] were not promoted"). A disclosed guard reads as rigor; a
   silent one discovered in code reads as p-hacking.
3. **Format ≠ capability.** A raw accuracy stop conflates the two failure
   modes. Reporting them separately is what makes the canary diagnostic
   rather than just a tripwire — a parseable-rate collapse with stable
   accuracy-among-parseable means format drift (fix the SFT data/template),
   not forgetting (fix LR/epochs).

## Implementation notes (for whoever picks this up)

- **Preferred variant: post-hoc pass, not in-loop.** Per-epoch in-loop evals
  would need a vLLM load of base+adapter inside the training job's GPU
  footprint — awkward alongside the HF trainer. Instead: SFT now retains all
  epoch checkpoints (`save_total_limit: null`), so run the canary as a
  separate stage over the saved checkpoints after training, before anything is
  promoted. Same selection safety, no training-loop changes.
- Natural shape: a `sft_canary` stage in `grpo_training` (or a standalone
  script first), using `dagspaces/common/vllm_inference.py` with the model's
  standard serve config (`model=` yaml → native template, thinking mode,
  reasoning-budget handling — gpt-oss needs its harmony split and max_tokens
  bump exactly as in eval, see
  [canonical-models.md](canonical-models.md)).
- Emit results into the checkpoint dir alongside `sft_traces.jsonl`
  (e.g. `canary.jsonl`: one row per epoch × {parseable_rate, acc_parseable,
  base_deltas, flagged}) and to W&B under `canary/*`.
- LoRA SFT at this scale (r=64, ~2.9k pairs, 3 epochs) is unlikely to cause
  genuine >5% capability loss; expect the parseable-rate leg to be the one
  that fires. That is the intended behavior, not a miscalibration.
- Cost envelope: 500 MCQs × ~13 checkpoints (11 models × ≤3 epochs where
  flagged interesting) is trivial vLLM throughput; the dominant cost is
  engine spin-up per checkpoint. Batch multiple epochs of the same model into
  one engine load with LoRA adapter swapping if it matters.
