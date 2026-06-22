# Cultural name/location bank provenance

These resources back the PrivacyLens **cultural name-perturbation** eval variant
(`dagspaces/privacylens/perturb/`). The experiment swaps person names and locations in the
PrivacyLens vignettes to non-Western cultural alternatives via a deterministic,
gender-preserving, record-seeded substitution, then runs the unchanged eval pipeline. The
goal is to **measure** whether models' contextual-integrity reasoning (leakage / helpfulness)
shifts as a function of the cultural origin of names — i.e., to surface model bias.

Entity detection uses a RoBERTa-large NER model (`Jean-Baptiste/roberta-large-ner-english`)
run through the project's `transformers` stack on GPU when available — not spaCy, whose
transformer models pin `transformers<4.54` and conflict with the vLLM/training stack.

## Files

- `name_banks.json` — per-culture pools of gendered first names, surnames, and
  culture-congruent locations. The `western` culture is `null`: a sentinel meaning **identity
  passthrough** (the control variant runs the same code with a no-op replacement map, so it
  shares the treatment code path and has no code-path confound).
- `first_name_gender.json` — a given-name → gender map (`m`/`f`) for common English/Western
  first names. Used only to read off the inferred gender of an *original* name so the chosen
  replacement preserves it (keeping story pronouns coherent). Curated from public-domain US
  SSA baby-name frequency lists. Intentionally not exhaustive; unknown names resolve to `u`
  and draw from the combined male+female pool.

## Culture groups

`east_asian`, `south_asian`, `arabic_me` (Arabic / Middle Eastern), `african` (Sub-Saharan
African), `african_american`. Name lists are compiled from publicly documented
high-frequency given names and surnames for each group, chosen for recognizability; they are
representative samples, not census-weighted distributions.

## Ethical note on the `african_american` set

This variant operationalizes the audit methodology of **Bertrand & Mullainathan (2004),
"Are Emily and Greg More Employable than Lakisha and Jamal? A Field Experiment on Labor
Market Discrimination"** (*American Economic Review* 94(4): 991–1013), which used
distinctively-Black vs. distinctively-White first names to measure discrimination. The names
here are drawn from / faithful to that and subsequent name-audit literature.

The purpose is to **detect and quantify model bias, not to endorse, essentialize, or
stereotype**. Distinctively-named groups are a measurement instrument for disparate model
behavior. Results should be reported with this framing, and the set should not be used to
make claims about individuals.

## Reproducibility

Replacement selection is seeded by `record_id | culture | identity-key`, so a given record
maps to the same alternates on every run. The substitution is checkpoint-independent.
NER model: `Jean-Baptiste/roberta-large-ner-english` (downloaded from the HuggingFace hub on
first use; uses GPU automatically when available).
