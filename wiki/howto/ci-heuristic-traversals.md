# Run the CI decision-heuristic traversal experiments (E1–E7)

`dagspaces/ci_heuristic` audits whether LLMs can work *through* Nissenbaum's
9-step CI decision heuristic. Design: `planning/ci-heuristic-llm-experiments.md`
in the umbrella repo; decisions D1–D6 in `planning/decisions.md`. Part 1
companion: [geoprivacy-hypotheticals.md](geoprivacy-hypotheticals.md).

## Mechanics in one paragraph

`load_cases` pulls the corpus tiers (A = expert gold, B = 540 synthetic
single-departure vignettes, C = the Part 1 imaging practices) →
`traverse` runs the scaffolding ladder (L0 zero-shot verdict … L5
deliberative) as batched rounds with per-step guided decoding →
`score_traversal` applies the no-judge scorers (extraction F1, prima facie,
misapplication probes a–f, entailment consistency, factor coverage). The
contextualization judge runs separately on the judge server. Everything is
greedy-decoded; ladder comparisons must not confound with sampling noise.

⚠ **Server mode for real runs**: the chain calls inference once per step
(9–20+ rounds at L5). In-process vLLM reloads the engine per round. Point
`model.server_url` at a long-lived vLLM OpenAI server (judge_server.sub
pattern) and the rounds share one loaded model.

## Smoke tests (first thing on the cluster)

```bash
# chain smoke: 2 cases, small model, no SLURM
python -m dagspaces.ci_heuristic.cli runtime.debug=true runtime.sample_n=2 \
  hydra/launcher=null ladder=l3 cases=pilot model=qwen3-8b/instruct

# Part 1 smoke (companion dagspace)
python -m dagspaces.vlm_geoprivacy_aug.cli runtime.debug=true \
  runtime.sample_n=5 hydra/launcher=null model=qwen3-vl-8b/instruct

# PLURALS optional-path spike (needs a served model; see script header)
SPIKE_BASE_URL=http://<node>:<port>/v1 python prompt_dev/plurals_spike.py
```

## E1 — scaffolding ladder (core)

Ladder × cases × model sweep. `cases=held_out` excludes contaminated Tier A.

```bash
for L in l0 l1 l2 l3 l4 l5; do
  python -m dagspaces.ci_heuristic.cli ladder=$L cases=held_out \
    model=qwen3-8b/instruct
done
# model axis (D1): qwen3-8b/{instruct,sft-ci,grpo-ci}, qwen3.5-9b/*,
# llama3.3-70b/instruct, gemma-3-12b/it, phi-4/base — sweep with -m model=...
```

Hypothesis checks read `outputs/.../score_traversal/metrics.json`:
extraction saturates early (tier_a mean F1 by ladder), steps 7–8 need the
ladder (coverage.mean_factor_recall), L0 anchors on generic privacy
(compare L0 decisions vs. traversal-derived decisions on the same cases).
The **failure-distribution profile** (probes.\*.rate per model) is the
point-(1) figure; compare against the human misapplication patterns
documented in Kumar et al. §2/§4 and Skeba & Baumer.

## E2 — extraction micro-benchmark

E1's Tier B rows already carry per-parameter hits; the probe variant:

```bash
python -m dagspaces.ci_heuristic.cli pipeline=tp_probe_eval cases=held_out \
  model=qwen3-8b/instruct
```

Hard-case breakdowns: slice per_case.parquet by `multi_tp`,
`sender_is_subject`; non-human-actor treatment comes from Tier C traversals'
`s3.nonhuman_roles` (is the robot a sender or an instrument?).

## E3 — prima facie under perturbation

Full Tier B sweep = E1 at the chosen ladder level with `cases=held_out`;
read `prima_facie.*` (sensitivity by parameter, attribution, incompleteness
recognition, presumption asymmetry). The incompleteness trap = the 36
`incomplete_norms` vignettes (probe f); the entrenchment trap = probe b.

## E4 — deliberative structures

```bash
for S in ensemble chain debate; do
  python -m dagspaces.ci_heuristic.cli ladder=l5 ladder.s7_structure=$S \
    cases=held_out model=qwen3-8b/instruct
done
# McDonald–Forte ablation arm:
python -m dagspaces.ci_heuristic.cli ladder=l5 ladder.include_marginalized=false \
  cases=held_out model=qwen3-8b/instruct
```

McDonald–Forte test: compare `s7` merged factors WITH vs. WITHOUT
marginalized personas — are disparate-impact harms present only when those
voices are in the panel? (`raised_by` provenance on merged factors tells you
who surfaced what.) Also validate the s5 simulated population against
published factorial-vignette norms (Apthorpe et al. smart-home, Martin &
Nissenbaum location) before trusting completeness verdicts.

## E5 — convergent validity (the bridge to Part 1)

```bash
# Part 1 shift metrics, both arms
python -m dagspaces.vlm_geoprivacy_aug.cli model=<m>                       # bridged
python -m dagspaces.vlm_geoprivacy_aug.cli model=<m> hypotheticals=capture_devices_raw
# Part 2 traversals on the same practices (same variant ids)
python -m dagspaces.ci_heuristic.cli ladder=<best-from-E1> cases=tier_c model=<m>
```

Join on variant id (`tier_c_<variant>` ↔ `hyp_id`). Across models:
condemnation strength (s9 decision + s7 harm counts) vs. toward-abstention
shift. Within model: fast Q7 vs. deliberate verdict — dissociations are the
finding (shift-without-derivation = vibes; derivation-without-shift =
knowledge not deployed).

## E6 / E7

E6: rubric = Kumar Table 1 goal/aim column (mirrored in
`heuristic_text.py` `goal` fields); stratified sample from E1 outputs;
D4 has the rater budget (needs sign-off). E7: paraphrase/renaming variants
of Tier B cases → `scoring/consistency.flip_rate` between runs;
cross-cultural step-5 probe = s5 elicitation with region-conditioned
personas vs. the default population.

## Outputs map

| Artifact | Where |
|---|---|
| per-step traversal rows (incl. `s5:elicit:*`, `s7:member:*`) | `outputs/traverse/dataset.parquet` |
| per-case probe/consistency/extraction flags | `outputs/score_traversal/per_case.parquet` |
| aggregate metrics + provenance | `outputs/score_traversal/metrics.json` |
| TP-probe conditions | `outputs/tp_probe/dataset.parquet` |

Metric-trust rules apply (n_real/n_defaulted on every rate); probe rates
are lexicon-based **lower bounds** — say so wherever they're quoted.
