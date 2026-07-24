# `R-CONTRAST` — wrong-book contrast (auxiliary)

**Parent:** [README.md](README.md) · **Date:** 2026-07-16 · **Status:** drafted
· **Kind:** ablatable additive auxiliary (weight 0.25 in `full` by the 2:1
rule). Removal = deletion + renormalization; the `−contrast` cell.

## One-liner

Score the completion's flows against a **wrong book's** universe (retrieval +
absolute judge grounding, no ranking) and reward the complement:
`r_contrast = 1 − grounding_wrong`. High reward = the extraction is
*distinctively of its book*, not generic privacy boilerplate that would
ground equally well anywhere.

## From clamp to module (the structural change vs v9)

The v9 lineage computed contrast *inside* R_ground:
`R_ground = clamp(s_correct − λ·grounding_wrong, 0, 1)`, λ = 1.0 in
production. Two things were wrong with that shape for a modular grid:

1. **Not removable** — setting λ = 0 changes R_ground's formula and score
   distribution rather than deleting a component; an honest `−contrast` cell
   was impossible.
2. **Coupled failure modes** — the v8 "contrastive-clamp asymmetry" bug
   (rank-diluted correct side minus full-grounding wrong side zeroed ~1/3 of
   well-grounded extractions) existed *because* two differently-scaled
   quantities were subtracted inside one formula. Additive decoupling makes
   that class of bug unrepresentable.

**Bridging note for the paper:** the published λ-sweep results (λ × ratio
grid) characterize the *clamp* operationalization; m-series contrast scores
are not numerically comparable to them. The λ knob itself no longer exists —
the module's influence is its weight (0.25 in `full`), set by the 2:1 rule
like every other auxiliary. `contrastive_ratio` (legacy additive wrong-source
*rows*) was already 0 and does not migrate.

## Protocol

- **Wrong-universe sampling:** one other book's universe, uniform over the
  training books, **seeded by `chunk_id`** — fixed per prompt across the
  whole run, so all G completions in every epoch face the same wrong book
  and the term enters the prescreen cache signature deterministically. (The
  v9 lineage resampled per call; determinism costs nothing and buys
  reproducibility.)
- **Scoring:** retrieve k = 3 norms per flow *from the wrong universe* (flow
  text as query, same retriever); the judge assigns each completion an
  absolute grounding score against those norms — **no ranking** (ranks are
  meaningless across universes; this was already the v9 convention for the
  wrong pass). The m1 run pins k = 3 (`rank_top_k: 3` in `m_series.yaml`);
  the keeper listwise path used the code default k = 5 (2026-07-24).
- `r_contrast = 1 − grounding_wrong ∈ [0,1]`.
- **Failure fallback:** judge-failed group → uniform 0.5, same convention as
  every judged module; `reward/contrast/judge_failed_group_frac`.

## What the term can and cannot claim

- **Rewards** book-specificity: flows articulated with the book's own roles,
  information types, and transmission principles ground poorly in a random
  other book.
- **Cannot punish honest universality:** some norms genuinely recur across
  books (confidentiality of medical disclosures, say); a correct extraction
  of a universal flow will partially match wrong universes and lose some
  contrast reward. That is a *bias toward distinctive framing*, accepted and
  bounded by the 0.25 weight — and it is exactly the property the `−contrast`
  cell measures the value of.
- **Hard negatives are an m2 option, not m1:** sampling the *most similar*
  other book (by universe embedding centroid) would sharpen the signal but
  adds a similarity computation and a new failure mode (near-duplicate
  universes); m1 ships uniform sampling.

## Cost

One absolute judge call per group per step (~the size of the `R-GROUND`
call). `full` therefore runs two judge calls + G answerer calls per group;
`core` runs answerer calls only.

## Diagnostics (`reward/contrast/*`)

`grounding_wrong_mean` (the raw wrong-book score — rising means genericness
drifting up), `judge_failed_group_frac`, and the pairing
(`chunk_id → wrong_book`) recorded in `training_metadata.json`.

## `−contrast` prediction

Pre-registered in [ablation-protocol.md](ablation-protocol.md): benchmarks ≈
`full`; the *trace* diagnostic moves instead — `grounding_wrong_mean` rises
without the penalty (book-specificity erodes). If benchmarks also drop,
book-specificity is doing outward-facing work (the paper's normative-
simulacra claim strengthens); if nothing moves at all, contrast is judge
cost with no effect and `core+ground` becomes the leaner canonical stack.
