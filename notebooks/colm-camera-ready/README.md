# Camera-ready analysis notebooks

Every figure, table, and reported number in *Reinforcing privacy reasoning in LLMs via normative
simulacra* (COLM 2026) is produced by a notebook in this directory. These are
[marimo](https://marimo.io) notebooks stored as plain Python — run them directly
(`python <notebook>.py`) or open them with `marimo edit`.

> **Read this first: two delivery modes, and both can bite.**
>
> Some notebooks write straight into `papers/colm26_normative-simulacra/`. Others write only into
> this directory's `figures/` and `tables/`, and their output reached the paper by **manual copy** —
> re-running one of those does *not* update the paper.
>
> The direct writers have the opposite hazard. Verified 2026-08-12: several paper tables carry
> **hand edits made after generation** — `source_texts.tex` was switched to `\scriptsize`, and
> `corpus_scaling.tex`'s caption gained a clause about the 2026-07-12 prompt-wiring fix. Re-running
> `gen_corpus_tables.py` silently reverts all of it. The *numbers* regenerate correctly (fiction10
> 2,993 / 16,200 / 10,034 and top100 15,875 / 90,091 / 53,492, matching `04_results.tex`); it is the
> typography and prose that get clobbered.
>
> **So: always `git -C papers/colm26_normative-simulacra diff` after running a direct writer**, and
> re-apply anything you did not mean to lose.

Use the canonical venv:

```bash
/share/pierson/matt/UAIR/.venv-vllm025cu129/bin/python notebooks/colm-camera-ready/<notebook>.py
```

## Producers

| Notebook | Delivery | Paper artifacts |
|---|---|---|
| `benchmark_results.py` | manual | `tables/benchmark_results{,_grpo,_kto}.tex` — the main results table plus the RL-stage rows |
| `gen_corpus_tables.py` | **direct** (`:30`) | `tables/{corpus_scaling,source_texts,top100_corpus}.tex` |
| `grpo_kto_training_diagnostics.py` | **direct** (`:78`, `:145`) | `figures/fig_grpo_arms_{health,discrimination}.pdf`, `figures/fig_kto_{arms_dynamics,heldout_threshold}.pdf`, `tables/t1_grpo_cell_summary.tex`, `tables/t2_kto_arm_summary.tex` |
| `norm_flow_embedding_space.py` | **direct** (`:145-146`) | `figures/fig_paired_displacement_*.pdf`, `figures/fig_retrieval_concentration.pdf`, `tables/hub_norms.tex` |
| `privacylens_judge_human_agreement.py` | **direct** (`:102`) | `tables/judge_human_agreement.tex` |
| `sft_training_diagnostics.py` | **direct** (`:88-90`) | `figures/fig_sft_training_diagnostics.pdf`, `tables/sft_training_summary.tex` |
| `corpus_descriptives_two_corpora.py` | manual (`:102-103`) | 11× `figures/corpus_*.pdf`; also emits `per_book_yield.csv` |
| `stage_weight_deltas.py` | manual (`:77-78`) | `figures/fig_stage_weight_deltas.pdf` |
| `distilled_grounding.py` | manual (`:79-80`) | `tables/distilled_grounding.tex` |
| `norm_grounding_disagreement.py` | numbers only (`:130`) | §5.2's reclassification rates — no figure or table |
| `norm_distribution_top100_vs_fiction10.py` | — | **Figures superseded** by the `corpus_*` set. Kept because `tables/norm_distribution/book_meta.csv` is a required input to `gen_corpus_tables.py`. |

`tables/canon.tex` in the paper is hand-authored and has no producer here.

## Upstream data producers

These live in `scripts/` and must run before the notebooks that consume them:

| Script | Feeds |
|---|---|
| `scripts/embed_camera_ready_norms_flows.py` | `norm_flow_embedding_space.py` (`:69`) |
| `scripts/build_grounding_disagreement.py` | `norm_grounding_disagreement.py` (`:88`), `distilled_grounding.py` (`:85`) |
| `scripts/rescore_cirl_scorable.py` | `benchmark_results.py` (`:109`, `:1044`, `:1229`) |

Two notebooks need a running server: `norm_flow_embedding_space.py` expects the embedding server
(`scripts/embedding_server.sub`, see `:71`), and `privacylens_judge_human_agreement.py` expects
the judge server (`scripts/judge_server.sub`, see `:33`).

## Dependency graph

```
scripts/embed_camera_ready_norms_flows.py ──> norm_flow_embedding_space.py ──direct──> fig_paired_displacement_*,
                                                                                       fig_retrieval_concentration.pdf,
                                                                                       tables/hub_norms.tex

scripts/build_grounding_disagreement.py ──┬─> norm_grounding_disagreement.py           (§5.2 numbers only)
                                          └─> distilled_grounding.py ──manual──> tables/distilled_grounding.tex

scripts/rescore_cirl_scorable.py ─────────> benchmark_results.py ──manual──> benchmark_results{,_grpo,_kto}.tex

corpus_descriptives_two_corpora.py ──manual──> 11x figures/corpus_*.pdf
                                   ──────────> per_book_yield.csv ─┐
norm_distribution_top100_vs_fiction10.py ────> book_meta.csv ──────┴─> gen_corpus_tables.py ──direct──>
                                                                       corpus_scaling.tex,
                                                                       source_texts.tex,
                                                                       top100_corpus.tex

grpo_kto_training_diagnostics.py ──direct──> fig_grpo_arms_*, fig_kto_*,
                                             t1_grpo_cell_summary.tex, t2_kto_arm_summary.tex
sft_training_diagnostics.py ───────direct──> fig_sft_training_diagnostics.pdf, sft_training_summary.tex
privacylens_judge_human_agreement.py direct──> tables/judge_human_agreement.tex
stage_weight_deltas.py ────────────manual──> figures/fig_stage_weight_deltas.pdf
```

## One coupling worth knowing

`norm_grounding_disagreement.py` produces the §5.2 grounding numbers, and
`distilled_grounding.py:97` **hardcodes** them as `PAPER = {"D": 0.309, "a2i": 0.211, "i2a": 0.098, ...}`
for its comparison against the per-arm re-run. If the disagreement analysis is ever recomputed,
that dict must be updated by hand — nothing checks it.

## Regenerable outputs

`figures/`, `tables/`, `cache/`, `__marimo__/`, and `nb_*.log` are gitignored. Only the notebooks
themselves and this README are tracked.
