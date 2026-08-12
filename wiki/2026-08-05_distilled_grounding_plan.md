# Distilled grounding: re-extracting flows with the fine-tuned weights

**Status:** EXECUTED 2026-08-06/07. Results in §0.
**Owner:** Matt. **Target:** COLM 2026 camera-ready, §5.2 (Normative grounding).

---

## 0. Results

Run: `multirun/2026-08-06_policy_flows_fiction10/09-17-02` (4 arms × 2,993
chunks × 10 books). Grounding tables:
`outputs/2026-08-06_distilled_grounding/`. Notebook:
`notebooks/colm-camera-ready/distilled_grounding.py` (asserts it reproduces the
published §5.2 numbers before reporting — it does).

### The stated hypothesis is FALSIFIED. A different, stronger result survives.

Primary cell, `double-heldout` (503 chunks, unseen by GRPO and KTO):

| arm | flows | D | κ | Δ vs null |
|---|---|---|---|---|
| teacher (Gemma-4-31B) | 2,764 | 32.2% | 0.063 | +2.3pt |
| Qwen3.5-9B base | 2,537 | 26.9% | 0.029 | +0.8pt |
| + SFT | 2,502 | 27.2% | 0.054 | +1.7pt |
| + GRPO m2-full | 2,153 | 28.1% | **0.113** | **+3.8pt** |
| + KTO k3-verdict | 2,154 | 28.5% | 0.102 | +3.4pt |

**Every Qwen arm reclassifies below 30.9% — but so does the untuned base.** The
32%→27% drop is a *model-family* effect, not a fine-tuning effect, and D then
*rises* with training (26.9 → 27.2 → 28.1 → 28.5), opposite to the prediction.
**Arm B is what makes this visible**; without it base's 26.9% reads as success.
This vindicates §3's insistence that the control was not optional.

**The chance-corrected metrics reverse the reading.** κ doubles base→SFT
(0.029→0.054) and doubles again with RL (→0.113); Δ-vs-null goes +0.8 → +1.7 →
+3.8. Because the null shuffles each arm's *own* labels within book, it holds
that arm's prior fixed — so a rising Δ is alignment the prior cannot buy. **The
RL arms beat the teacher on its own measure** (κ 0.113 vs 0.063) on chunks they
never trained on.

**Not memorization.** On `grpo-unseen-books` (1,689 chunks, the 6 novels GRPO
never saw — a book-level holdout, the stronger form): m2-full κ 0.111 / Δ +3.4
vs base κ 0.041 / Δ +1.1.

### Norms invoked (§4.2), τ=0.60

| arm | citations | hit | hit (wrong book) | **margin** | **concordance** |
|---|---|---|---|---|---|
| base | 22,708 | 0.706 | 0.426 | 0.281 | 0.149 |
| sft | 22,488 | 0.709 | 0.424 | 0.285 | 0.148 |
| m2-full | 13,045 | 0.783 | 0.465 | **0.318** | **0.195** |
| k3-verdict | 13,055 | 0.781 | 0.463 | 0.318 | 0.194 |
| teacher | 16,682 | 0.765 | 0.444 | 0.320 | 0.148 |

The wrong-book control earns its place: ~0.43–0.47 of citations clear τ against a
*random other novel*, so raw `hit` is mostly generic norm-shaped phrasing and
must not be reported alone.

**Concordance is the standout.** The RL arms cite the norm retrieval would have
picked 19.5% of the time vs 14.8% for base, SFT, *and the teacher* — the only
metric where the RL arms clearly exceed Gemma. And **SFT ≈ base on every
citation metric** (margin 0.285 vs 0.281, concordance 0.148 vs 0.149): the entire
gain comes from the RL stage, not from SFT.

### Three constraints on the claim

1. **GRPO and KTO are indistinguishable** — 91.7% *identical* generated reasoning
   text, despite genuinely different adapters (distinct md5, same shared
   `_merged_sft` backbone). Report them as one result; this design cannot
   separate them.
2. **SFT is contaminated everywhere**, `double-heldout` included (it trained on
   all 2,993). The clean comparison is RL-vs-SFT on held-out chunks.
3. **The flow populations differ by arm.** RL arms extract ~16% fewer flows
   (4.24/chunk vs 5.04) and shift their own inappropriate-rate 0.09→0.12 toward
   the grounded 0.26. Δ controls for the prior; it does not make this a matched
   comparison.

### Honest framing

Fine-tuning did **not** make grounding redundant — the rate at which grounding
changes a label barely moved. What changed is that the policy's ungrounded
judgment became *systematically norm-related*, from near-chance (κ 0.029) to
roughly double the teacher's alignment (κ 0.113), and its norm citations moved
toward the governing norm. That is a claim about **mechanism**, not accuracy;
establishing the grounded label is *more correct* still needs external gold.

---

## 1. The idea

§5.2 currently measures normative grounding as a property of the *corpus*, with
no policy in the loop. The teacher (Gemma-4-31B-it) labels each of the 16,200
fiction-10 flows for appropriateness with no norm in context; the same flow is
then re-labelled by retrieving its nearest governing norm from the book's own
normative universe and reading appropriateness off `normative_force ×
act_polarity`. Grounding reclassifies **30.9%** of flows, 21.1% appropriate →
inappropriate against 9.8% the other way.

That number describes the *teacher*. It says nothing about whether our
fine-tuning moved a model's ungrounded judgment toward the grounded one.

The proposal: re-run the **same two-stage `historical_norms` flow pipeline**
(`ci_reasoning` → `ci_extraction`), with the **same fiction prompts** and the
**same chunks**, substituting each fine-tuned checkpoint for Gemma-4-31B-it as
the extractor. Each arm then emits its own `ci_appropriateness` and
`ci_norms_invoked`. Apply the identical grounding procedure to each arm's flows
and recompute the reclassification rate.

**Hypothesis (Matt, 2026-08-05):** the fine-tuned arms reclassify *less* than
30.9%, because training pushed their ungrounded judgment toward the grounded
one. The grounding effect has been partly **distilled into the weights**.

Holding prompts and chunks fixed while varying only the weights is what makes
the resulting rate comparable to 30.9%. That is the reason to reuse the
teacher's pipeline rather than the policies' own one-stage training
instruction — see §7.1 for the cost of that choice.

---

## 2. Why the pipeline swap is nearly free

Three properties of the existing code make this a config change, not a build:

1. **`ci_extraction` uses guided decoding** against
   `CIExtractionResult.model_json_schema()`
   (`dagspaces/historical_norms/stages/ci_extraction.py:126-129`). Every arm is
   *constrained* to emit the same schema, so there is no parse-rate confound and
   `ci_appropriateness` / `ci_norms_invoked` are guaranteed present. A weaker
   model cannot degrade into unparseable output and silently drop rows.
2. **`run_vllm_inference` already carries LoRA.** Both stages call it
   (`ci_reasoning.py:112`, `ci_extraction.py:178`), and it resolves
   `model.lora_path`, applies `_remap_lora_keys_for_vlm` for the Qwen3.5
   VLM-arch key trap, and issues a `LoRARequest`
   (`dagspaces/common/vllm_inference.py:1553-1573`). The camera-ready model
   yamls are already in the LoRA-eval shape.
3. **The pipeline is model-agnostic.**
   `COLM_flows_fiction_prefetched_gemma4.yaml` takes the model from the CLI and
   the chunks from `FICTION_CHUNKS_PATH`. Only the engine overrides are
   Gemma-shaped and need replacing (§5.1).

The grounding side is equally reusable:
`scripts/build_grounding_disagreement.py` already runs retrieval through
production `NormRetriever` + `deontic.flow_appropriateness` over the
polarity-merged universe. It needs to be *generalized* to accept an arbitrary
flows parquet and universe, not forked (§5.3).

---

## 3. What the arms are, and what each is contaminated by

| arm | model config | weights |
|---|---|---|
| T | `gemma-4-31b/instruct` | teacher — already run, this is the 30.9% |
| B | `qwen3.5-9b/instruct` | stock Qwen3.5-9B, no adapter |
| S | `qwen3.5-9b/sft-canonical` | Qwen3.5-9B + SFT LoRA |
| G | `qwen3.5-9b/m2-full-ckpt450` | m2 `_merged_sft` + GRPO LoRA (camera-ready GRPO) |
| K | `qwen3.5-9b/k3-verdict` | m2 `_merged_sft` + KTO LoRA (camera-ready KTO) |

**Arm B is not optional.** Without stock Qwen3.5-9B, a lower rate for S/G/K is
indistinguishable from "Qwen judges appropriateness differently than Gemma."
Arm B is the same pipeline with no adapter and costs one extra cell.

Optional arm K′ = `qwen3.5-9b/k3-sftctrl`, the KTO-side SFT control. It
separates the KTO objective from "more gradient steps on this data." Cheap;
include it if the budget allows.

### 3.1 Training contamination — measured, not assumed

I pulled the actual training sets rather than trusting the configs.

**GRPO (m2 `full`)** — from
`multirun/2026-07-28_grpo_m2_full/.../checkpoint/reward_traces.jsonl`, keyed on
`(source_id, chunk_id)` because `chunk_id` is per-book, not global:

- 11,808 extract rows over **492 unique chunks** (600 unique prompts total, 492
  extract + ~108 vignette, matching the 0.82/0.18 task mix; 3 epochs × 8
  completions = 24 rows per chunk).
- Those 492 chunks come from **only 4 of the 10 novels**: Les Misérables (375),
  Bleak House (63), Pride and Prejudice (50), Alice (4).
- **Six novels — Monte Cristo, Anna Karenina, Middlemarch, Nineteen Eighty-Four,
  The Age of Innocence, Dorian Gray — were never seen by GRPO. 1,689 chunks.**

That concentration is worth a second look on its own (the reward-variance
prescreen appears to have collapsed onto the longest book), but for this
experiment it is a gift: a book-level GRPO holdout inside fiction-10, for free.

**KTO (K1 dataset, fingerprint `b27a46f8e7f5`)** — from
`outputs/2026-07-31_k1_full/kto_metadata.json`: a stratified chunk-level 80/20
split at seed 42, **2,394 train / 599 held out**, spanning all 10 books.
`heldout_keys` is recorded in the metadata, so the split is reproducible without
re-deriving it.

**SFT (`sft-canonical`)** — trained on the fiction-10 teacher flows. **All 2,993
chunks are in-domain.** This is the load-bearing limitation of the whole design:
*fiction-10 cannot test arm S at all.* SFT was trained to reproduce the
teacher's ungrounded labels on these exact chunks, so D(S) ≈ D(T) is what
memorization predicts, and a deviation measures capacity, not alignment.

Intersecting the two recorded key sets:

| chunk set | n | GRPO saw | KTO saw | SFT saw |
|---|---|---|---|---|
| `fiction10-all` | 2,993 | 492 | 2,394 | all |
| `kto-heldout` | 599 | some | none | all |
| `grpo-unseen-books` | 1,689 (6 novels) | none | ~80% | all |
| `double-heldout` = `kto-heldout` − GRPO | **503** | none | none | all |
| **`top100`** | 15,875 / 100 books | **none** | **none** | **none** |

The 503-chunk doubly-held-out set covers all 10 books (68 Les Mis, 110 Monte
Cristo, 85 Anna Karenina, 79 Middlemarch, 69 Bleak House, 27 1984, 24 Age of
Innocence, 22 P&P, 13 Dorian Gray, 6 Alice).

### 3.2 Scope decision (Matt, 2026-08-05): fiction-10 only

**top100 is deferred.** This round runs fiction-10 and the four arms in §3, and
nothing else. What that costs, stated plainly so it is not rediscovered later:

- **The SFT arm cannot be read as generalization.** SFT trained on all 2,993
  fiction-10 chunks, so its number is training fit. It is still worth having —
  it is the sanity check of §7.2 (SFT trained on the teacher's *ungrounded*
  labels, so D(S) should track D(T); if it does not, the run is broken) and it
  is the base the two RL arms sit on.
- **GRPO and KTO carry the claim, on the 503-chunk `double-heldout` subset.**
  That set is genuinely unseen by both, spans all 10 novels, and is large enough
  to separate the effects the teacher measurement found (a 30.9% rate at n=503
  has a binomial SE of ~2.1pt).
- **Nothing here supports a cross-corpus generalization claim.** The paper text
  must not imply one. If the fiction-10 result is interesting, top100 becomes
  the obvious follow-up and the plan for it is preserved in §10.

So the reporting structure for this round:

- **`double-heldout` (503) — primary for GRPO and KTO.** Clean for both.
  Contaminated for SFT; label it.
- **`fiction10-all` (2,993) — descriptive.** Directly comparable to 30.9%
  because it is the same chunks and the same prompts. Every arm has seen some or
  all of it; report as training fit and say so.
- **`grpo-unseen-books` (1,689, 6 novels) — secondary for GRPO.** A book-level
  rather than chunk-level holdout, which is the stronger form of held-out for a
  claim about transfer across normative universes. KTO saw ~80% of it, so it
  reads for GRPO only.

---

## 4. Metrics

### 4.1 Appropriateness

Per arm × chunk set, with the teacher's coding rule applied identically
(`ambiguous → inappropriate`, per CI conservatism):

- **D — reclassification rate.** `P[grounded ≠ coded(model's own label)]`. The
  headline, directly comparable to 30.9%.
- **The 2×2 flip table**, not just D. The teacher's asymmetry is 21.1% A→I vs
  9.8% I→A. If grounding has been internalized, the **A→I excess should shrink
  specifically** — the model should already be calling those flows
  inappropriate. D can stay flat while the asymmetry collapses, and that would
  still be the effect we are looking for. Report the ratio alongside D.
- **Cohen's κ** between the model's label and the grounded label (teacher:
  0.069 over all flows, 0.053 over the binary intersection).
- **Both class priors** — `P(model = inappropriate)` and `P(grounded =
  inappropriate)` — per arm. Teacher: 14.9% → 26.1%.
- **Prior-matched null.** Shuffle the model's labels within book, recompute D →
  `D_null`. Report `Δ = D_null − D`. **This is the honest quantity.** A policy
  that merely shifted its base rate toward 26% would beat the teacher on raw D
  with no alignment whatsoever; Δ is immune to that. The within-book shuffle
  machinery already exists from the §17 audit (which found retrieval at 7.7 SD
  above the shuffle null).

  > **Superseded 2026-08-08.** As implemented, this null is Monte Carlo
  > estimating something with a closed form: under a within-book permutation
  > `E[D_null] = 1 − Σ_b (n_b/N)·[p_b q_b + (1−p_b)(1−q_b)]`, which the
  > notebook now computes exactly (matched the 200-draw estimate to inside its
  > own MC error on all five arms). And the raw `Δ` is not comparable across
  > arms, whose null rates span 0.28–0.33 — so the notebook reports
  > `κ = 1 − D/E[D_null]`, the same quantity normalized, with a 95% CI from a
  > bootstrap resampling whole novels. `Δ` and the separate global κ column are
  > gone; `kappa` in `per_arm_by_chunk_set.csv` is now the within-book one.

### 4.2 Norms invoked

The user asked for this explicitly and it may be the more interesting result,
since it is about *what the model cites* rather than what it concludes.

For each arm, embed every string in `ci_norms_invoked` and compare against the
book's governing-norm index:

- **Universe-hit rate.** Fraction of cited norms whose top-1 cosine against the
  book's own universe clears τ.
- **Retrieval concordance.** Does the model's cited norm match the norm that
  retrieval would have picked for that flow? Report the rank and cosine of the
  retrieved norm within the cited set.
- **Wrong-book control.** The same measurement against a randomly drawn *other*
  novel's universe, mirroring R-CONTRAST. **If the hit rate rises equally
  against the wrong book, the model learned norm-shaped phrasing, not the
  universe.** Without this control the hit-rate number means nothing.

### 4.3 Extraction-side descriptives (guard against reading D wrong)

D is computed over each arm's *own* flows, so the flow population differs by
arm. Report, per arm × chunk set: flows per chunk, `has_information_exchange`
rate, share of chunks yielding zero flows, and the distribution of top-1
retrieval similarity. A drop in D accompanied by a collapse in flow count or a
jump in retrieval similarity is an extraction shift, not an alignment gain.

Optional, if D moves and we need to rule out extraction shift: align each arm's
flows to the teacher's within the same chunk by cosine of the retrieval query,
and recompute D on matched flows only, reporting the unmatched fraction.

---

## 5. Implementation

Additive only — no edits to the m-series or keeper surfaces.

### 5.1 New pipeline config

`dagspaces/historical_norms/conf/pipeline/COLM_flows_fiction_policy.yaml` —
a copy of `COLM_flows_fiction_prefetched_gemma4.yaml` with the Gemma-shaped
engine overrides replaced:

- `tensor_parallel_size: 1`, `data_parallel_size: 4` (9B on A6000s; the TP=2×DP=2
  layout in the Gemma config exists because the 31B is KV-hungry at ~0.94
  MB/token — irrelevant here).
- Keep `max_model_len: 24576`. The Gemma config's comment is load-bearing:
  706 of 2,993 fiction-10 prompts exceed 16,384 with the ~6k-token system
  prompt + book summary, and clamping truncates mid-JSON, biasing
  `has_information_exchange` against long-summary books. The model yamls say
  16,384; the pipeline must override it.
- Preserve `enable_lora: true`, `max_lora_rank: 64`, `enforce_eager: true` from
  the model yaml — **do not** let the pipeline overrides clobber them.
- Keep `prompt_ci_reasoning: ${prompt_ci_reasoning_fiction}` and
  `prompt_ci_extraction: ${prompt_ci_extraction_fiction}` byte-identical to the
  teacher run. Confirm the `PROMPT PROVENANCE: ci_reasoning_fiction` banner and
  `prompt_name` column, per `tests/historical_norms/test_prompt_wiring.py`.

**Open risk:** `data_parallel_size > 1` combined with `enable_lora` and
`enforce_eager` is an untested combination in this stack. Smoke it at
`runtime.sample_n=8` before the overnight launch; fall back to DP=1 × 4
sequential cells if it breaks.

### 5.2 Launch script + sweep

`scripts/run_policy_flow_extraction.sh`, modelled on
`run_extract_fiction10_flows_gemma4.sh`. The arm sweep is a plain Hydra
multirun over the model group:

```bash
FICTION_CHUNKS_PATH=/share/pierson/matt/UAIR/outputs/2026-03-20_historical_norms/\
23-05-10/COLM_fetch_fiction/outputs/fetch/chunks.parquet \
/share/pierson/matt/UAIR/.venv-vllm025cu129/bin/python \
  -m dagspaces.historical_norms.cli --multirun \
  pipeline=COLM_flows_fiction_policy \
  model=qwen3.5-9b/instruct,qwen3.5-9b/sft-canonical,\
qwen3.5-9b/m2-full-ckpt450,qwen3.5-9b/k3-verdict \
  experiment.name=policy_flows_fiction10
```

Launch with the **absolute venv python** — submitit bakes the monitor `srun`
python from the driver `sys.executable`, and shell activation does not survive
the hop (the m2 grid header records this).

top100 uses the same command with
`FICTION_CHUNKS_PATH=/share/pierson/matt/zoo/datasets/gutenberg_cache/chunks_top100_fiction_en.parquet`
and a **stratified subsample** — 15,875 chunks × 4 arms × 2 stages does not fit
overnight. Sample 20 chunks per book (2,000 chunks) with a recorded seed, and
write the sampled key list to disk so the teacher-side top100 reference (§6, P4)
is computed on exactly the same chunks.

### 5.3 Generalize the grounding builder

Extend `scripts/build_grounding_disagreement.py` with `--flows`, `--universe`,
`--out`, and an `--arm` tag; default every path to the current teacher values so
existing invocations are unchanged. Keep the wiki-§17 faithfulness guard **armed
whenever the inputs are the defaults**, and have `--skip-guard` record its reason
in the output manifest rather than silently disarming.

The flow-side embedding cache will miss — these are new flows — so
`EMBEDDING_SERVER_URL` (Qwen3-Embedding-8B) must be live. Rough budget:
2,993 chunks × ~5 flows × 4 arms ≈ 60k embeds for fiction-10, plus the top100
sample.

### 5.4 Analysis

- `scripts/analyze_norms_invoked.py` — §4.2, including the wrong-book control.
- `notebooks/colm-camera-ready/distilled_grounding.py` — marimo, following the
  conventions of `norm_grounding_disagreement.py` (`TAB_DIR`, `save_table`,
  provenance table in the header cell). Emits the arm × chunk-set table and one
  figure.

### 5.5 Tests

New `tests/historical_norms/test_policy_flow_extraction.py` plus additions:

- the policy pipeline's prompt selections are byte-identical to the gemma4
  pipeline's (extends the existing prompt-wiring guard);
- the `ambiguous → inappropriate` coding in the generalized builder matches the
  teacher measurement exactly;
- the `double-heldout` set is reproducible to 503 from the two recorded key
  lists (`kto_metadata.json['heldout_keys']` and the m2 trace chunk keys);
- the top100 universe carries `act_polarity` (the builder already raises, but a
  test catches it before a night is spent).

---

## 6. Prerequisites

Status as of 2026-08-05.

- **P1 — the two `_merged_sft` backbones are identical. ✅ DONE.**
  `m2-full-ckpt450.yaml` points `model_source` at `cell=full/.../_merged_sft`,
  `k3-verdict.yaml` at `cell=core/.../_merged_sft`. Full byte compare of both
  `model.safetensors` (18,819,722,392 bytes each): **identical**. Arms G and K
  therefore sit on the same base and are directly comparable to each other, and
  the no-adapter behaviour of that base is the SFT policy.
- **P2 — the generalized builder still reproduces the teacher audit. ✅ DONE.**
  Regression run of `scripts/build_grounding_disagreement.py` on the default
  (teacher) inputs after the generalization: 16,200 flows, 15,493 dual-labelled,
  all six §17 guard stats inside tolerance (κ 0.0527 vs 0.0532, agreement 0.7059
  vs 0.7057, margin median 0.0136). The refactor did not move the teacher number.
- **P3 — `import dagspaces.historical_norms.cli` was broken at HEAD. ✅ FIXED.**
  See §6.1 — this blocked the entire dagspace, not just this experiment.
- **P4 — confirm `sft-canonical`'s exact chunk coverage** of fiction-10
  (positives + contentless-v6 negatives). §3.1 assumes all 2,993; if the
  contentless filter dropped a meaningful slice, those chunks are an SFT
  holdout too and the design gets better. Not blocking — it can only improve
  what arm S supports.
- **P5 — smoke test. ✅ DONE 2026-08-06.** `SMOKE=1
  scripts/run_policy_flow_extraction.sh`, arm `m2-full-ckpt450`, 8 chunks, both
  stages. Run dir `multirun/2026-08-05_policy_flows_fiction10/23-48-06/0`.

  | check | result |
  |---|---|
  | DP=4 + LoRA + `enforce_eager` | 4 independent EngineCore replicas, no conflict |
  | 24,576 context override | applied over the model yaml's 16,384 |
  | LoRA actually loads | resolved to `grpo_m2_full/.../checkpoint-450`, VLM key-remap cache hit, on both stages |
  | prompt provenance | `ci_reasoning_fiction` / `ci_extraction_fiction`, in the logs *and* the parquet `prompt_name` |
  | reasoning | 8/8 chunks, 0 parse errors, 35 flows (4.4/chunk vs the teacher's corpus 5.4) |
  | extraction schema | 35 rows, **every required column 100% non-null** — guided decoding behaving as §2 predicted |
  | labels | 31 appropriate / 2 inappropriate / 2 ambiguous; `ci_norms_invoked` in the teacher's JSON-articulation format |

  Wall clock: launch 23:48:06 → reasoning 23:58:27 → extraction 00:01:23. So
  **~10 min of fixed engine-startup overhead per arm** (two weight loads);
  generation on 8 chunks was negligible. That is the only firm number the smoke
  gives — it does not constrain the 2,993-chunk generation time.

  The 4.4 flows/chunk is the first (weak, n=8) evidence against the §7.1
  off-distribution worry: a policy that could not follow the teacher's prompts
  would have shown it here as parse errors or a flow-count collapse.

- **P6 — the embedding server must be up for the ANALYSIS step.** Verified the
  builder consumes a policy parquet end-to-end: the guard auto-disarmed with its
  reason, the universe loaded, and it then failed **loudly** on the cache miss
  ("no embedding ... and no server URL") rather than embedding zeros. That is
  the intended failure mode, but it means `EMBEDDING_SERVER_URL`
  (`klara:8001`, currently down) has to be live before §4 can run. **Not needed
  for the overnight generation** — only for the step after it.

### 6.1 A pre-existing break found en route

`dagspaces/historical_norms/runners/fetch_gutenberg.py` imported
`_collect_outputs` / `_save_stage_outputs` from `..orchestrator`, but those
helpers had moved to `dagspaces.common.orchestrator` and the local re-export was
dropped. Because `historical_norms/orchestrator.py` builds `_STAGE_REGISTRY` at
**module scope** (line 80), the failed import took down `import
dagspaces.historical_norms.cli` entirely — so *every* historical_norms pipeline
was unrunnable at HEAD, not just this one. It surfaced as a misleading
"circular import" message, because Python guesses that cause when a name is
missing from a partially-initialized module.

Every sibling runner in that package had been migrated to
`DataFrameStageRunner`; `fetch_gutenberg` was the one left behind. Fixed by
importing from `dagspaces.common.orchestrator`, matching what the
`grpo_training` runners already do. `grpo_training`, `privacylens`, and
`goldcoin_hipaa` were unaffected.

Worth knowing because it means **no historical_norms extraction has run since
that refactor landed**, and anything that would have caught it (a CLI smoke
test) is missing from the suite.

---

## 7. Threats to the conclusion

### 7.1 The policies are off-distribution under these prompts

S, G, and K were trained on the terse one-stage extraction instruction
(`sft_data_prep._build_ci_instruction`, Appendix E
`prompt:extraction-instruction`) with thinking disabled. The fiction pipeline
prompts them with a ~6k-token system prompt, a book summary, and a two-stage
reasoning-then-extraction split they never saw. This is the price of holding the
measurement instrument fixed at the teacher's, and it is the right trade for
comparability against 30.9% — but it means a *rise* in D for a fine-tuned arm is
ambiguous between "grounding was not internalized" and "the arm is off its
prompt distribution." Arm B is the partial answer: it establishes what the
untuned backbone does under the same off-distribution conditions.

Guided decoding removes the worst version of this risk (no format collapse), and
the m1 post-mortem's 34.4%-vs-2.7% R-VALID gap was a *reward-gate* artifact that
does not apply under constrained decoding. Still: report the reasoning-stage
`has_information_exchange` rate and flow counts per arm (§4.3) so a
prompt-following collapse is visible rather than inferred.

### 7.2 The RL arms were trained toward this exact target

R-DIRECT's gold *is* `flow_appropriateness` over this universe, and the KTO K1
labels derive from the same gold. **On chunks GRPO or KTO trained on, a low D is
close to tautological.** This is why §3.2 puts top100 first and labels fiction-10
descriptive. State it in the paper rather than letting a reviewer find it.

The mirror-image prediction is a useful sanity check: SFT trained on the
teacher's *ungrounded* labels, so **D(S) should track D(T)**. If it does not,
something is wrong with the run before any conclusion is drawn.

### 7.3 D is confounded by the flow population

Each arm extracts its own flows, so D can fall through extraction shift, label
prior shift, or retrieval shift with no alignment gain (§4.3). The prior-matched
null Δ (§4.1) is the primary defence; matched-flow analysis is the fallback.

### 7.4 Pre-registered null

**If D drops on fiction-10 but not on top100, the finding is "training fit, not
distillation," and that is what gets reported.** Writing this down now is what
keeps the post-hoc version of this analysis honest, given §7.2.

---

## 8. Cost and staging

The teacher run was ~1h reasoning + ~1h extraction for 2,993 chunks at TP=2×DP=2
on 4 A6000s for a 31B dense model. A 9B at TP=1×DP=4 should be materially
faster; budget ~45 min per stage per arm as a planning number until P5 measures
it. The sweep is serial (both launchers pin `array_parallelism: 1`), so four
arms is ~6h — one night.

1. **Now:** P5 smoke.
2. **Overnight (~6h):** `scripts/run_policy_flow_extraction.sh` — fiction-10,
   4 arms, both stages.
3. **Next (CPU + embedding server):** the generalized builder per arm, then the
   §4.1 / §4.2 metrics and the notebook. The flow-side embedding cache **will
   miss** — these are new flows — so `EMBEDDING_SERVER_URL` must be live for
   this step. Roughly 2,993 chunks × ~5 flows × 4 arms ≈ 60k embeds.

### 8.1 What exists now

| artifact | state |
|---|---|
| `dagspaces/historical_norms/conf/pipeline/COLM_flows_fiction_policy.yaml` | new — prompt-identical to the teacher pipeline, engine re-shaped for a 9B + LoRA |
| `scripts/run_policy_flow_extraction.sh` | new — serial 4-arm sweep, `SMOKE=1` mode |
| `scripts/build_grounding_disagreement.py` | generalized: `--arm` / `--flows` / `--universe`; guard auto-disarms off-default and records why; teacher path regression-verified |
| `tests/historical_norms/test_policy_flow_extraction.py` | new — prompt parity, generation budget, 24576 override, LoRA survival, base-arm-has-no-adapter, and the 503-chunk contamination bookkeeping |
| `dagspaces/historical_norms/runners/fetch_gutenberg.py` | fixed (§6.1) |

Still to write, after the flows land: `scripts/analyze_norms_invoked.py` (§4.2)
and `notebooks/colm-camera-ready/distilled_grounding.py` (§5.4).

---

## 9. Where it lands in the paper

A new paragraph or short subsection in §5.2, after the corpus-level grounding
result. It does not displace 30.9%; it extends it from a property of the corpus
to a property of the trained weights. One table (arm × chunk set: D, flip
asymmetry, κ, Δ-vs-shuffle) and, if §4.2 holds up, the norms-invoked hit rate
with its wrong-book control.

It also answers a question §5.4 currently leaves open. The benchmark table is
noisy and, by Matt's own assessment, cannot reliably rank these checkpoints on
contextual privacy reasoning. This measurement is internal, row-wise, and does
not route through an LLM judge — so it is a treatment-effect probe the benchmark
suite structurally cannot provide. That is the argument for the subsection, and
it is worth making explicitly given §5.4's stated distrust of the benchmarks.

---

## 10. Deferred: the top100 arm

Kept because it is the natural follow-up if fiction-10 shows anything.

top100 is the only chunk set held out from **every** arm, SFT included — 100
different novels with their own normative universe, and `sft-canonical` never
touched it (the separate `multirun/2026-07-15_sft_canonical_top100_gemma4` is a
*different* SFT model, not the camera-ready one). It is what would license a
generalization claim; fiction-10 alone cannot.

To run it:

- Prerequisite: confirm
  `multirun/2026-07-17_universe_top100_gemma4/.../norm_universe/` carries
  `act_polarity`. Without it, `flow_appropriateness` falls back to `performing`
  and ~19% of grounded labels invert (measured 2026-07-25). If absent, run
  `scripts/apply_act_polarity.py` as was done for fiction-10. The builder raises
  rather than writing a silently-inverted table.
- Prerequisite: compute the **teacher's** top100 reclassification rate. 30.9% is
  a fiction-10 number and is not the right comparator here. The teacher's top100
  flows (`outputs/2026-07-13_top100_flows_gemma4/`) already exist, so this is one
  builder run, no GPU beyond the embedding server.
- Sample stratified — 15,875 chunks × 4 arms × 2 stages does not fit a night.
  20 chunks per book (2,000) with a recorded seed, and write the sampled key
  list to disk so the teacher reference is computed on exactly the same chunks.
- Same pipeline, `FICTION_CHUNKS_PATH=/share/pierson/matt/zoo/datasets/gutenberg_cache/chunks_top100_fiction_en.parquet`.

## 11. Related

- `notebooks/colm-camera-ready/norm_grounding_disagreement.py` — the teacher-side
  measurement this extends
- `scripts/build_grounding_disagreement.py` — the builder to generalize
- `wiki/2026-07-31_kto_plan.md` §17 — the construct-mismatch audit (κ 0.053, 7.7
  SD above a within-book shuffle null) that this analysis is downstream of
- `wiki/normative-simulacra.md`, `wiki/grpo-reward.md`
- `scripts/kto_heldout_probe.py` — the multi-arm single-engine LoRA pattern, if
  the DP+LoRA route in §5.1 fails and a bespoke driver is needed after all
