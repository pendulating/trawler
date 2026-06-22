# 2026-06-09 — Code review: `historical_norms` + `grpo_training` vs. COLM paper claims

**Status:** ALL items closed 2026-06-09 — F1–F7 and S1–S6, S11 resolved in
the working tree; S12 verified (TRL 0.29.1 logs the gate key); S7–S9
manuscript actions; S10 accepted by design. Full suite green (415 passed).
Reviewed the current working tree
(including the uncommitted Phase 1–5 GRPO redesign) against the paper claims
documented in `wiki/overview.md`, `wiki/normative-simulacra.md`,
`wiki/grpo-reward.md`, and the two 2026-06-09 GRPO changelog entries.

**TLDR:** The core machinery matches the claims well — reward weights, gated
composition, ranked-judge formula, KL anchor, promotion gates, fiction
prompts, corpus, and guided decoding all verified with file:line evidence.
But there are **3 findings that directly contradict paper claims** and
**4 more that can silently corrupt results**.

Remaining work (all code findings fixed):
- Re-run `norm_universe_and_reward_prep` to pick up the F5 dedup before the sweep.
- Manuscript: use the code's field names (S7/S8), scope R = Σ wᵢRᵢ to CI-extraction rows (F7), don't claim character-norm removal (S9), cite the realized post-screen vignette fraction (F6).

---

## Part 1 — Claims-vs-code contradictions (fix before camera-ready)

### F1. CRITICAL — "Paper primary λ=1.0" vs. production config λ=0.5

> **RESOLVED 2026-06-09.** `online_rground_external.yaml` and the smoke
> config now default `contrastive_lambda: 1.0`, matching the paper-primary
> sweep origin declared in `sweep/lambda_axis.yaml` and `sweep/ratio_axis.yaml`.
> Sweep cells always pin λ explicitly, so swept runs are unaffected; legacy
> configs (`default.yaml`, `online_rground_4gpu/5gpu`, `_g4`, `no_thinking`)
> keep 0.5 for archival comparability. `wiki/grpo-reward.md` updated.

- `dagspaces/grpo_training/conf/training/grpo/online_rground_external.yaml:72` sets `contrastive_lambda: 0.5`.
- `dagspaces/grpo_training/conf/sweep/ratio_axis.yaml:20-23` pins λ=1.0 and calls it the "paper primary setting"; `wiki/grpo-reward.md` repeats both values.

Whichever checkpoint the paper reports as primary, the stated λ must match
the config that actually trained it. Anyone reproducing from
`online_rground_external.yaml` defaults gets λ=0.5.

**Fix:** align the production default with the paper's primary cell, or
explicitly document that "primary λ" refers to the swept optimum, not the
config default.

### F2. MAJOR — Ranked-mode judge failure is not "uniform 0.5 for the group"

> **RESOLVED 2026-06-09.** `online_rground.py` now skips the contrastive
> subtraction when the correct-side judge fails — the whole group scores a
> uniform 0.5 (zero advantage) even if the wrong-universe pass succeeded.
> Regression test:
> `tests/grpo_training/test_reward_improvements.py::TestRankedOnlineRGround::test_correct_judge_failure_neutral_even_if_wrong_pass_succeeds`
> (the pre-existing judge-failure test only covered both passes failing).

`dagspaces/grpo_training/stages/online_rground.py:611-641` (independently
confirmed). The changelog claims judge failure → uniform 0.5 → zero
advantage. Actual behavior: when the correct-side ranking call fails,
`correct_scores = [0.5] * n_cand`, but wrong-universe grounding scores are
still subtracted per-candidate:

```python
if rankings is None:
    correct_scores = [0.5] * n_cand          # neutral, as documented
...
raw = correct_scores[pos] - self.contrastive_lambda * wrong_grounding[pos]
scores[i] = max(0.0, min(1.0, raw))          # NOT uniform if wrong judge succeeded
```

If the wrong-side judge succeeded, scores vary within the group — a spurious
gradient driven entirely by the wrong-universe judge, on exactly the turns
where the correct judge timed out.

**Fix:** when `rankings is None`, skip the contrastive subtraction (set the
whole group to 0.5).

### F3. CRITICAL-ish — Chunks can exceed 6000 chars by up to ~1000 (~17%)

> **RESOLVED 2026-06-09 (code); paper data unaffected in practice.** Both
> chunkers (`historical_norms/stages/fetch_gutenberg.py` and
> `common/gutenberg/chunking.py`) now enforce a hard invariant: no chunk
> exceeds `chunk_size`. Oversize content is packed sentence-by-sentence and
> unbreakable sentences are hard-split; duplicate-only (pure overlap-seed)
> chunks can no longer be emitted. The stage's fallback defaults were also
> aligned to 6000/1000 (closes S4). Tests:
> `tests/common/test_gutenberg_chunking.py` (parametrized over both
> implementations).
>
> **Empirical impact on the existing COLM data**
> (`/share/pierson/matt/n2s4cir/data/fiction10/ci_reasoning.parquet`,
> 2,993 chunks): only **8 chunks (0.27%) exceed 6000 chars, max 6736**
> (Les Misérables 0.8%, Anna Karenina 0.2%, Middlemarch 0.3%; mean 5628).
> The already-extracted data is fine to keep — "chunks of up to ~6000
> characters" in the paper is accurate to within 8 outliers. Re-chunking is
> NOT required for the 10-novel corpus; the fix matters for the planned
> 100-novel run.

`dagspaces/historical_norms/stages/fetch_gutenberg.py:82-104` (independently
confirmed); same logic lifted into `dagspaces/common/gutenberg/chunking.py:62-76`.
After flushing a full chunk, the next chunk is seeded with the 1000-char
overlap and the next paragraph is appended **without a size check**, so
chunks reach `overlap + chunk_size ≈ 7000` chars. Reproduction: two
5998-char paragraphs yield chunk sizes `[5998, 7000]`. Long-paragraph novels
(Les Misérables, Anna Karenina) hit this constantly. (The
sentence-fallback path for giant paragraphs has the same unguarded append.)

**Fix:** add a size guard after seeding the overlap, or soften the paper to
"chunks of ~6000 characters (paragraph-aligned, up to chunk_size + overlap)."

---

## Part 2 — Silent-corruption risks (affect results, not just prose)

### F4. MAJOR — Prescreen cache key omits `rank_top_k` and `rank_weight`

> **RESOLVED 2026-06-09.** The signature is factored into
> `_reward_signature()` in `prompt_screening.py` and now covers
> `rank_top_k`, `rank_weight`, `judge_model`, and `max_tokens` in addition
> to the original fields. Existing prescreen caches are invalidated once
> (correct — reward semantics also changed with the F2 fix). Tests:
> `tests/grpo_training/test_prescreen_cache_key.py`.

`dagspaces/grpo_training/stages/prompt_screening.py:119-128` (independently
confirmed). The `reward_signature` covers weights / composition / lambda /
scoring_mode / temperature but **not** `rank_top_k`, `rank_weight`, or the
judge model identity. A sweep cell changing only retrieval depth or rank
blending silently reuses a stale screening result — the 15-cell sweep would
train on a prompt set screened under different reward dynamics.

**Fix:** add `rank_top_k`, `rank_weight` (and ideally judge model name) to
the signature dict.

### F5. MAJOR — No dedup of norms/flows across the 1000-char chunk overlap

> **RESOLVED 2026-06-09 — and the feared mechanism is empirically
> negligible.** Measured on the fiction10 data:
>
> - **Norms** (11,554 valid rows feeding `norm_universe`): 56 exact
>   duplicates (0.5%) — of which only **1 group spans more than one chunk**;
>   47 are same-chunk extractor stutter. Token-Jaccard ≥ 0.7 near-duplicates
>   across adjacent chunks: 19 pairs (~0.2%).
> - **Flows** (1,241 rows): 15 exact duplicate IFT tuples (1.2%) —
>   **0 groups span chunks**; 16 near-dup pairs across adjacent chunks.
>
> Cross-boundary duplication essentially does not occur (reasoning anchors
> norms/flows to specific snippets). Also note the original finding's hope
> that `norm_consolidation` absorbs duplicates was moot: the production
> COLM pipeline (reasoning → extraction → role_abstraction) has **no
> consolidation step**, and `norm_universe.py` says so explicitly.
>
> Fix applied where it matters: `norm_universe.py` now runs
> `dedup_universe_norms()` — exact dedup per book on the normalized
> embedding text (the norm's retrieval identity) — removing the same-chunk
> stutter before universes are built. Re-run `norm_universe_and_reward_prep`
> to regenerate `norm_universes.json` (~56 norms fewer). Paper guidance:
> report post-dedup counts; no extraction re-run needed. Tests:
> `tests/grpo_training/test_norm_universe_dedup.py`.

No `drop_duplicates` or semantic dedup anywhere in the extraction pipeline
(`fetch_gutenberg.py`, `ci_reasoning.py`, `ci_extraction.py`,
`norm_reasoning.py`, `norm_extraction.py`, `norm_role_abstraction.py`). A
norm straddling a chunk boundary is extracted from both chunk N's tail and
chunk N+1's head. Norm consolidation (embedding clustering per
`gutenberg_id`, `norm_consolidation.py:412`) likely absorbs most duplicates
on the norms track, but:

1. the CI-flows track has **no consolidation analog**, and
2. any pre-consolidation count reported in the paper ("we extracted N
   norms/flows") is inflated, and boundary-region norms are over-represented
   in the normative universes.

**Fix:** dedup pass after each extraction stage keyed on
`(gutenberg_id, snippet)` or similar; report post-dedup counts in the paper.

### F6. MAJOR — Post-screen vignette fraction is unaudited

> **RESOLVED 2026-06-09.** `training_metadata.json` now records
> `n_vignettes_pre_screen` and `n_vignettes_post_screen` (final training
> set, post-screen + post-split, same semantics as `n_flow_chunks`), and
> the stage logs the realized vignette fraction vs the configured ratio.
> Paper guidance: state vignette_ratio=0.3 as the pre-screen target and
> cite the realized fraction from metadata.

`dagspaces/grpo_training/stages/grpo_training.py` (~768 mix-in, ~822
prescreen). Vignettes are mixed at the configured 30% **before**
pre-screening, and vignette rewards (binary-ish judgment) plausibly have
degenerate variance under the SFT policy, so screening can strip them
disproportionately. The paper's "30% judgment vignettes" describes the
pre-screen mix only; `training_metadata.json` does not record the
post-screen vignette count.

**Fix:** record `n_vignettes_post_screen` in `training_metadata.json` (or
prescreen CI rows only and add vignettes after); document that 30% is the
pre-screen target.

### F7. MINOR (worth a paper sentence) — Vignette rows bypass the 6-component composite

> **RESOLVED 2026-06-09 (documentation).** Behavior is by design; now
> documented in `wiki/grpo-reward.md` (vignette rows get
> `0.5·r_judgment + 0.25·r_judgment_reasoning + 0.25·r_norm_cite`, never
> the composite). The paper must scope "R = Σ wᵢRᵢ" to CI-extraction rows.

`dagspaces/grpo_training/stages/rewards.py` (~869, ~955): vignette rows set
`partial_components[i] = None` and take their reward directly as
`0.5·judgment + 0.25·reasoning + 0.25·norm_cite` — neither additive nor
gated. If the paper says "R = Σ wᵢRᵢ" for all training rows, that is
inaccurate for 30% of them.

---

## Part 3 — Smaller items

| # | Item | Location | Status |
|---|---|---|---|
| S1 | `contrastive_ratio` Python fallback default was **0.1**, not 0.0 | `grpo_training/stages/grpo_training.py:481` | **RESOLVED 2026-06-09** — fallback changed to 0.0 (production YAMLs all set it explicitly; only direct invocation was at risk). |
| S2 | Zero-embeddings on server failure → R_ground silently 0 for the batch | `grpo_training/stages/clients.py` | **RESOLVED 2026-06-09** — `EmbeddingClient` now counts consecutive fully-failed `encode_batch` calls and raises `RuntimeError` after `fail_after` (default 3); transient blips still degrade gracefully, persistent outages abort instead of training on zeroed R_ground. Counts true server failures (not zero scores), so legitimate all-zero reward batches can't false-positive. Tests: `tests/grpo_training/test_embedding_client_failure.py`. |
| S3 | `reward_trend` promotion gate read *training* reward, not `eval_reward` | `grpo_training/gates.py` | **RESOLVED 2026-06-09** — the trend gate now prefers the held-out `eval_reward` curve when ≥3 points are logged (dev_fraction > 0), falling back to the training-batch reward otherwise; result records `source`. Tests added in `test_reward_improvements.py::TestPromotionGates`. |
| S4 | Stage fallback chunking defaults 2000/200 diverged from paper 6000/1000 | `historical_norms/stages/fetch_gutenberg.py` | **RESOLVED 2026-06-09** with the F3 fix — stage defaults now 6000/1000, matching `common/gutenberg/chunking.py`. |
| S5 | qwen36 pipeline comments claim DP=2×TP=2 but set no engine overrides | `COLM_flows_fiction_prefetched_qwen36.yaml`, `COLM_norms_fiction_prefetched_qwen36.yaml`, `role_abstraction_standalone_qwen36.yaml` | **RESOLVED 2026-06-09** — all five GPU nodes across the three yamls now set `model.engine_kwargs.data_parallel_size: 2` / `tensor_parallel_size: 2`, matching their headers and the non-qwen36 counterpart. |
| S6 | Stale corpus comments listed 5 books incl. **Frankenstein** (not in corpus) | `COLM_norms_fiction.yaml`, `COLM_flows_fiction.yaml` | **RESOLVED 2026-06-09** — both headers now list the actual 10 novels from `data/fiction_sample_for_colm.yaml`. |
| S7 | Terminology drift: code says `information_type`, paper says "attribute" | `historical_norms/ci_schema.py:103` | **MANUSCRIPT ACTION** — prompts also say "Information Type"; the paper should match the implementation. No code change. |
| S8 | Terminology drift: Raz fields are `prescriptive_element/norm_subject/norm_act/condition_of_application`, paper says `(deontic, subject, act, condition)` | `historical_norms/ci_schema.py:172-211` | **MANUSCRIPT ACTION** — semantically equivalent; paper should map or match names. No code change. |
| S9 | `norm_quality_passed` (character-name contamination flag) is advisory only | `historical_norms/stages/norm_extraction.py` | **ACCEPTED (by design)** — still flag-only, but detection upgraded same day to layered spaCy NER, extended to the flows track, and recomputed post-abstraction (see `2026-06-09_ner_quality_checks.md`). The paper must not claim character-specific norms were "removed". |
| S10 | qwen36 norms pipeline omits role abstraction; model pinning depends on caller passing `model=` | `COLM_norms_fiction_prefetched_qwen36.yaml` | **ACCEPTED (known Hydra limitation)** — documented in the yaml itself; forgetting `model=` on the standalone role-abstraction run mixes qwen2.5-72b into a qwen3.6-27b pipeline. Double-check the CLI invocation when running it. |
| S11 | `has_information_exchange` fallback (`len(flows) > 0`) is dead code under guided decoding | `historical_norms/stages/ci_reasoning.py:100` | **RESOLVED 2026-06-09** — `wiki/normative-simulacra.md` rewritten to describe it as a guarded edge case, not a routine silent fallback. |
| S12 | `frac_reward_zero_std` gate skips silently if the TRL version doesn't log that key | `grpo_training/gates.py` | **VERIFIED 2026-06-09** — installed TRL 0.29.1 logs `frac_reward_zero_std` (`trl/trainer/grpo_trainer.py:1946`); the gate is operative on this cluster. Re-verify if TRL is upgraded. |

---

## Part 4 — Claims verified as accurate

### grpo_training

| Claim | Evidence |
|---|---|
| Composite weights [0.10, 0.05, 0.05, 0.20, 0.10, 0.50], sum 1.0; additive Σ wᵢRᵢ | `online_rground_external.yaml:125-132`; `rewards.py:769`; `CompositeRewardFunction.__init__` enforces exactly 6 weights |
| Gated composition `R = gate × disc` with weight-normalized means, div-by-zero guard; production sets `gated` | `rewards.py:763-768`; `online_rground_external.yaml:91` |
| R_ground absolute: top-k=3 cosine (L2-normalized), wrong source guaranteed ≠ correct, `clamp(r_c − λ·r_w, 0, 1)` | `clients.py:499,575`; `online_rground.py:75,268-325,392` |
| Ranked formula exactly as documented: 1-based ranks (`schemas.py:134`, `ge=1`), G=1 guard, wrong side grounding-only | `online_rground.py:79-117`; `rank_top_k=5`, `rank_weight=0.5` in config:83-84 |
| `contrastive_lambda` and `contrastive_ratio` are independent mechanisms (ratio adds rows pre-prescreen) | `grpo_training.py:206-251` |
| R_context uses all-MiniLM-L6-v2 (not Qwen3 embedder); gold-aware no-flow 0.9/0.0/0.4 | config:133; `grpo_training.py:455`; `rewards.py` r_context ~419-422 |
| Prescreen scores with the same `CompositeRewardFunction` instance as training (ranked + gated + contrastive included) | `prompt_screening.py:303` |
| Optimizer recipe: G=8, lr=1e-5, 3 epochs, β=0.01, `scale_rewards="none"`, `mask_truncated_completions=true`, grad-accum 32; forwarding uses `if _val is not None` (no falsy-check bug) | `online_rground_external.yaml:32-55`; `grpo_training.py:882-886` |
| KL anchor = exact SFT checkpoint via merge_and_unload → reload → fresh LoRA | `grpo_training.py:662-718` |
| Dev split seeded `train_test_split(test_size=0.05)`, eval batch pinned to G | `grpo_training.py:836,894` |
| Vignette pre-screen mix formula gives exactly `ratio` post-mix | `grpo_training.py:263` (`ceil(n_ci·r/(1−r))`) |
| Promotion gate thresholds: gain>0, zero-std<0.2, KL<1.0 (skip if β=0), no-flow ±0.15; metadata fields written from post-screen post-split training set | `gates.py:32-37`; `grpo_training.py:1069-1070` |
| Norm-universe JSON keys and training-time lookups both stringify `source_id` consistently | `norm_universe.py`; `grpo_training.py` |

### historical_norms

| Claim | Evidence |
|---|---|
| Production chunking params 6000/1000 (but see F3 for overflow) | `data/fiction_sample_for_colm.yaml:16-17` |
| Guided decoding applied in all four extraction stages | `ci_reasoning.py:76`, `ci_extraction.py:126`, `norm_reasoning.py:79`, `norm_extraction.py:274` |
| CI IFT schema: subject/sender/recipient/information_type/transmission_principle + appropriateness/norms_invoked/confidence | `ci_schema.py:103` ff. |
| Raz schema: 4 components + `normative_force` with the exact 5 literals + `governs_information_flow` + confidence | `ci_schema.py:172-211` |
| All six COLM fiction pipeline YAMLs override reasoning AND extraction prompts to fiction variants; no prescriptive-default leak | `COLM_{norms,flows}_fiction*.yaml` |
| Corpus is exactly the 10 claimed novels (9 Gutenberg IDs + 1984 via URL); religious texts only reachable via clearly separate `*_prescriptive` pipelines | `data/fiction_sample_for_colm.yaml`; `data/religious_texts.yaml` |
| `gutenberg_id` survives all stages; consolidation groups by it | `norm_consolidation.py:412` |
| Two-stage CoT (reasoning → per-snippet extraction, separate guided-decoded calls) implemented as described | stages above |
| Parse failures are non-silent: error columns set, nulls propagate, `extract_json` logs decode + repair failures | `_postprocess` in both extraction stages; `_utils.py` |

---

## Provenance

Produced by two parallel code-review agents (one per dagspace) on 2026-06-09,
with the highest-stakes findings (F1, F2, F3, F4) independently re-verified
by direct code inspection. Two agent-reported concerns were checked and
retracted during review (embedding double-prefix; gate base-rate
bookkeeping) — both paths are consistent and are listed as verified above.
