# Gold traversal annotation guide (Tier A)

How to turn a published CI analysis into a per-step gold file the extraction
and coverage scorers can consume. Schema: `gold_schema.json` (mirrors the
traversal state object in `planning/ci-heuristic-llm-experiments.md` §4, plus
gold-only fields). One JSON file per case in `tier_a/`.

## Ground rules

1. **Annotate what the source argues, not what you believe.** Gold = the
   published expert traversal. If the source is wrong by your lights, it is
   still gold; note disagreements in `annotator_notes`.
2. **Step content must come from the step where the source does that work.**
   If a paper mixes evaluation into its flow description, put the flow content
   in s1 and the evaluative content in s7/s8 — and record the mixing in
   `source_deviations` (sources themselves misapply the heuristic; that is
   data, not noise).
3. **Contamination flag is mandatory.** `contaminated: true` for anything in
   Nissenbaum (2010) or other pre-cutoff canonical texts. Contaminated cases
   are few-shot/rubric material only — never held-out test items.
4. **Granularity**: one flow entry per distinct (sender, recipient, subject,
   info-type, TP-set) tuple. When the source treats "disclose / show /
   transmit" as different flows (as Kumar et al. do), so do we.
5. **TP surface forms**: record the source's own label (`"need"`,
   `"voluntary"`) plus a normalized form from the working vocabulary below;
   scorers match on aliases.
   - Working TP vocabulary (extend as needed, log additions here):
     confidentiality, secrecy, need, voluntary, notice, consent, exchange,
     mandatory/compulsion, entitlement, dessert, reciprocity, anonymity,
     aggregation, ephemerality, temporality, mutuality, desire,
     purpose (added for martens2021_contact_tracing — empirical CI studies
     often fold purpose-of-use into the TP parameter. NOTE: this records
     SOURCE practice for extraction scoring only; our normative reference
     (Kumar et al., Nissenbaum 2019) situates purpose with context (s2), and
     probe (c) still flags purpose-language in MODEL s4 outputs as a
     misapplication. Gold files using this entry must log the source's
     loose TP reading in source_deviations — martens2021 does.)
6. **Factor checklists (s7/s8)**: enumerate each moral/political factor the
   source raises as `{factor, kind, affected_parties, direction}` — these are
   the recall targets for coverage scoring. `kind` from: autonomy, freedom,
   power, justice, equality, fairness, democracy, discrimination,
   information_asymmetry, coercion, trust, other(label).
7. **Step 9**: `decision` ∈ {continue, modify, reject}; every condition the
   source imposes goes in `conditions[]`, verbatim-ish.
8. **Incompleteness**: if the source concludes norms are incomplete/contested
   rather than violated, s6.violation = "incomplete_norms" — do not coerce to
   yes/no.

## Workflow per case

1. Fill `meta` (id, source citation, practice one-liner, contaminated?).
2. Read the source once fully; then annotate steps in order, quoting or
   tightly paraphrasing; page/section refs in `evidence`.
3. Steps the source skips: `"absent"` (most published analyses skip some —
   record, don't invent). Partial coverage is expected; the `steps_present`
   list drives which scorers run on the case.
4. Second annotator (or second pass ≥1 week later for solo annotation) on the
   s7 factor checklist and s9 decision — the fields coverage/consistency
   scoring depends on. Log disagreements in `annotator_notes`.

## Sources queue (Tier A survey)

Confirmed:
- kumar2024_fitbit — Kumar, Zimmer & Vitak 2024 (CSCW), Fitbit PFI →
  health care. Complete 9-step walk-through incl. TP table. NOT contaminated
  as a *traversal* (post-2010) but recent enough to be in training data —
  mark `contaminated: true` conservatively.
- nissenbaum2010_p148 / nissenbaum2010_p181 — the book's illustrations.
  Contaminated by definition.

Candidates to assess (target 8–15; prefer post-2023 or obscure venues to
reduce contamination):
- PrivaCI symposium proceedings (2018–2025) case analyses
- Vitak/Zimmer COVID-19 contact-tracing CI analyses
- Smart-home CI analyses (Apthorpe et al. — factorial, may lack full traversal)
- Skeba & Baumer facial recognition (explicitly partial — useful as a
  misapplication exemplar for probe (d), not as full gold)
- King DTC genetic testing
- Bowser et al. citizen science (engages steps 7–8 — rare)
