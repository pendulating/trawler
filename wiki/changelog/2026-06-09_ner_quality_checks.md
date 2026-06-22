# 2026-06-09 — NER-based name QA for norm + flow extraction

**Status:** in working tree. Replaces the manually curated character
blocklist as the *primary* QA detector and fixes two structural gaps the
empirical check exposed.

## Why

`norm_extraction.py` carried a hand-list of ~180 character/place names for
the 10-novel corpus. Three problems:

1. **Doesn't scale** — on the planned 100-novel run, QA would silently
   degrade to the title-regex for the 90 unlisted books.
2. **Flows track had NO quality check at all**, even though the CI
   extraction prompt itself forbids character names (`ci_schema.py:188`).
3. **Flags were stale post-abstraction** — `norm_quality_passed` was
   computed before role abstraction and never recomputed: on fiction10,
   61% of abstracted rows carried failures for names abstraction had
   already removed, while real residual leaks went unflagged.

## What changed

**New `dagspaces/historical_norms/name_detection.py`** — layered
`PersonNameDetector`:

1. *blocklist* — `cfg.norm_quality.character_blocklist` + the built-in
   10-novel list (kept for aliases/places NER can't see: "big brother",
   "monte cristo", "pemberley").
2. *titled* — `Mr./Mrs./Lady/... + ProperNoun` regex.
3. *person_entity* — spaCy `en_core_web_sm` PERSON NER (already in the
   venv; no new dependency). Corpus-agnostic — this is the layer that
   scales. Toggle: `norm_quality.use_ner` (default true). Model-load
   failure degrades to layers 1–2 with one loud warning, never a crash.

Precision filter on PERSON entities (measured, not guessed): drop
lowercase entities and single-token entities in sentence-initial position
— the small model tags "Law enforcement…", "Citizens of…", "Children
who…" as PERSON otherwise. Accepted trade-off: a single-token name
appearing ONLY sentence-initially is missed; multi-token names are always
kept.

**Integration:**

- `norm_extraction.py` — `_validate_norm_quality` now uses the detector
  (same `norm_quality_flags`/`norm_quality_passed` columns; new flag kind
  `person_entity_in_<field>:<name>`).
- `norm_role_abstraction.py` — new `revalidate_norm_quality()` recomputes
  quality on the ABSTRACTED fields; extraction-time values preserved as
  `pre_abstraction_norm_quality_*`. `norm_quality_passed` now measures
  abstraction success (and feeds the `data_quality/*` W&B metric with a
  truthful rate).
- `ci_extraction.py` — new `_validate_flow_quality()` adds
  `flow_quality_flags`/`flow_quality_passed` over all six flow fields;
  surfaced as `data_quality/flow_quality_passed_rate`.

Rows are still flagged, never dropped (S9 policy unchanged).

## Measured on fiction10 (800-row sample of abstracted norms)

- Old blocklist verdicts post-abstraction: 61% "failed" — almost all stale
  pre-abstraction flags.
- NER-only, after precision filter: **15/800 flagged (1.9%)**, of which
  ~12 are TRUE residual leaks the old system missed on abstracted text
  (M. Cavalcanti, Hawdon, Grisha, Jo, M. Charcellay, Martin Verga,
  Lady Catherine, Mihailov…), ~3 mild false positives (Easter,
  Savoyard, "An inn-keeper").

## Tests

`tests/historical_norms/test_name_detection.py` (16 tests: NER layer with
the real model, precision filter, pattern layers, norm-validation and
flow-validation integration, post-abstraction revalidation). Full suite:
442 passed.

## Follow-ups

- Re-run role abstraction (or just `revalidate_norm_quality` over the
  existing parquet) before building paper-final universes if you want the
  ~1.5% residual-leak rows flagged in the data the paper reports.
- The 100-novel run needs no per-book curation — only optional blocklist
  entries for known aliases (e.g. epithets like "the Count").
