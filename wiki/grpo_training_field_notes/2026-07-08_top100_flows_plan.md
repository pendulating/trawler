# GRPO redesign field notes — top100-flows run plan (the data lever)

**Date:** 2026-07-08 · **Status:** planned + scripts staged (`scripts/run_extract_top100_flows.sh` → `scripts/run_grpo_top100_flows.sh`); launch gated on the v12a verdict · **Author:** design from the 2026-07-08 critical assessment

Continues [2026-07-08_critical_assessment.md](2026-07-08_critical_assessment.md)
(recommendation 4) and implements the third-ranked pivot of
[2026-07-01_v11_probe_midrun_forensics.md](2026-07-01_v11_probe_midrun_forensics.md).

## Why

Three reward iterations (v9 symmetric floor → v10 false-permit floor → v11
rebalanced vignettes) pinned GoldCoin Forbid recall at 0.55 (SFT 0.65). The
reward-tweak vein is mined out; the diagnosed **non-reward** constraints are:

1. **Smallness**: 704 post-screen prompts × 3 epochs (fiction10: 2,993 chunks,
   12.9% flow-bearing → ~387 flow chunks).
2. **Permissive skew**: fiction10 governing norms ~4–5:1
   appropriate:inappropriate → hedging/permitting is EV-optimal when unsure.
3. **Epoch-2 freeze**: v10 verdict policy byte-identical from ckpt-350 to
   ckpt-500; repetition of a small pool stops teaching.

The top100 corpus attacks all three: ~5.3× the chunks, a norm distribution
measurably richer in *prohibited* force (the paper's top100 scaling analysis,
`app:corpus-scaling`), and enough fresh prompts to train a full run without
ever repeating data.

## Corpus facts (verified 2026-07-08)

- Chunk cache `zoo/datasets/gutenberg_cache/chunks_top100_fiction_en.parquet`:
  **100/100 books, 15,875 chunks** (manifest: `books_requested: 100,
  books_cached: 100`). The cache is complete.
- The "97 books" in the paper is an **extraction-stage** phenomenon, not a
  corpus failure: books **35** (*The Time Machine*), **215** (*The Call of the
  Wild*), **6133** (*Arsène Lupin*) had *every* chunk judged
  `has_prescriptive_content: false` by the Qwen3.6 reasoning gate (0 errors;
  next-lowest books had 1 and 3 norm-bearing chunks — a continuous tail, not a
  failure class). They have **no normative universe** and must be excluded
  from GRPO training data (empty grounding retrieval otherwise —
  `norm_universe_json` silently defaults to `[]`).
- Fact-check of the adjacent memory: `chunks_top1000_fiction_en` = 1000/1001
  cached; `chunks_top1000_en` = **930/1000** (70 fetch failures). The "97"
  belongs to top100-fiction extraction; the top1000-all shortfall is separate.
- Already built and ready: top100 norm universes
  (`multirun/2026-06-30_build_top100_universe/.../norm_universes.json`, 97
  sources, 21,510 norms) **and** the per-source embeddings (97 `.npy`).

## What's missing = one ~3h job

GRPO consumes `CI_REASONING_PATH` — the flows pipeline's **stage-1 reasoning
output** (`article_text`/`gutenberg_id`/`has_information_exchange`), not the
structured 5-tuples. So only `ci_reasoning` over the top100 chunks blocks the
run (~3h on 4 GPUs; the norms-track reasoning over the same chunks took 2.8h).
New stage-1-only pipeline: `COLM_flows_reasoning_prefetched_qwen36`
(the ~19h `ci_extraction` can be resumed later from the saved reasoning
parquet via `ci_extraction_from_reasoning_fiction`, wasting nothing — do this
before any paper flows-side scaling analysis or a future SFT refresh).

## Design

**Job 1** — `scripts/run_extract_top100_flows.sh`: flows reasoning over the
top100 cache (Qwen3.6-27B, DP2×TP2, ~3h).

**Job 2** — `scripts/run_grpo_top100_flows.sh`:

| element | setting | rationale |
|---|---|---|
| training data | curated top100 `ci_reasoning.parquet` → `/share/pierson/matt/n2s4cir/data/top100flows/` | books 35/215/6133 dropped; seeded (42) chunk cap baked into the artifact so what trains is what's on disk |
| chunk cap | `TARGET_FLOW_CHUNKS=1400` (env-tunable, 0=off) | lands ~2,800 extraction prompts + 30% vignettes ≈ 4,000 pre-screen ≈ 2,500 post-screen (v11 keep-rate) ≈ 625 steps ≈ **~44h** at the v11 probe's 4.2 min/step |
| epochs | `num_epochs=1` | anti-freeze: fresh data at every step; v8–v11 peaks all landed by epoch ~0.5–2 anyway |
| universes | grounding + contrastive + vignettes ALL from the top100 build | keys align (same gutenberg cache); contrastive wrong-universe pool 9 → 96 |
| reward | **v11 config pinned**: `floor_prohibit=0.1`, `hedge_prohibit=null` | best-validated arm (only one passing reward-trend + KL gates, KL 0.045); yaml default is now v12a, so both knobs are pinned explicitly |
| prescreen | fresh cache `cache/grpo_prescreen_top100flows.json` | key hashes the exact prompt set anyway; explicit path per cell discipline |
| SFT base | `qwen3.5-9b/sft-contentless-v6` (unchanged) | data lever only; no SFT re-run |

**Decision gate (reward knob):** if the v12a run's mid-run traces show the
hedge tier working (prohibited-flow correct-commit share off ~0.10 at ⅓-run),
launch job 2 with `HEDGE_PROHIBIT=0.5` (control = the v12a run). Otherwise
launch as-is with the v11 reward (control = the v11 probe).

**Honesty note on controls:** relative to the v11 probe this run changes the
*data regime as a bundle* — extraction prompts, grounding universe, and
contrastive pool move together. It is a regime experiment, not a single-knob
experiment; the reward is what's held fixed.

## Pre-registered predictions

1. **Realized force mix**: the share of prohibited/discouraged-governed
   directional events rises vs fiction10's ~4.8:1 (check
   `training_metadata.json` force-mix fields + forensics table 2).
2. **If frequency/balance was co-binding**: prohibited-flow correct-commit
   share rises off the 0.06–0.12 band and GoldCoin Forbid recall finally moves
   off 0.55 toward SFT 0.65 (judged on compliance macro-F1 n=107 per the n=20
   caveat).
3. **If hedge EV alone binds** (the forensics' preferred reading): hedge mass
   stays ~72% even on fresh balanced data with the v11 reward → decisive
   evidence that the v12a hedge tier (or stronger) is *required*, and the two
   levers should be composed in the next run.
4. **Freeze watch**: with 1 epoch of fresh data, no ckpt-N ≡ ckpt-M verdict
   freeze; if the policy still freezes, the constraint is optimizer-side, not
   data-side.
5. **Out-of-domain guard**: ConfAIDE-2a/2b and CIRL at-or-above the v11 probe
   levels (the vignette mix is unchanged); PrivacyLens frontier within noise
   of the v9 keeper. Over-permit or indiscriminate-forbid drift on these =
   kill.

Mid-run: `scripts/analyze_grpo_verdict_traces.py <run_dir>` (tables 2/3 +
exploration guard), checkpoints every 50 steps, kill-at-peak discipline.

## Paper implication (if a checkpoint from this run is ever reported)

Training-data provenance changes (top100, not fiction10): §source-texts, the
extraction-stats table, and the vignette-source disclosure all need updates —
logged in `papers/colm26_normative-simulacra/CONGRUENCE.md` (2026-07-08 entry,
"NOT changed" list).

## Related

- [2026-07-08_critical_assessment.md](2026-07-08_critical_assessment.md) — recommendation 4 of the ranked list
- [2026-07-03_v12a_plan.md](2026-07-03_v12a_plan.md) — the staged run whose traces gate the reward knob here
- [2026-07-01_v11_probe_midrun_forensics.md](2026-07-01_v11_probe_midrun_forensics.md) — the diagnosis (smallness / skew / freeze) this attacks
- [grpo-reward.md](../grpo-reward.md) · [[project-grpo-flat-reward]]
