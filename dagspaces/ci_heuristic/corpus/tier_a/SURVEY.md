# Tier A candidate survey — published CI analyses

Machine-assisted literature survey (2026-07-06) for gold-traversal candidates.
Ranked by usefulness; annotate top-down per ANNOTATION_GUIDE.md. Counts:
8 full traversals (substantive steps 7-9), 8 solid partials (1-6), 4
exemplars/marginal.

## Tier 1 — full/near-full traversals (annotate first)

| # | Case | Practice | Steps | Contamination | Notes |
|---|------|----------|-------|---------------|-------|
| 1 | Kumar, Zimmer & Vitak 2024 (CSCW, 10.1145/3653710) | Fitbit PFI → healthcare | all 9 | moderate | ANNOTATED: `kumar2024_fitbit.json` |
| 2 | Zimmer 2018 (SM+S, 10.1177/2056305118768300) | OkCupid data release / research ethics | 1-6 + 7-8, verdict | high | canonical "already public" rebuttal |
| 3 | Bloch & Bashir 2017/18 (AHFE, 10.1007/978-3-319-60483-1_59) | cardiac implants + remote monitoring | self-labeled 9-step | **low** (obscure venue) | paywalled (Springer chapter) — needs MF library access |
| 4 | Sar & Al-Saggaf 2014 (Ethics Inf Technol, 10.1007/s10676-013-9329-y) | SNS third-party tracking | heuristic-titled, 1-8 + verdict | moderate | "decision heuristic" in title |
| 5 | Vitak & Zimmer 2020 (SM+S, 10.1177/2056305120948250) | COVID contact tracing | 1-7 + 9 | high | short-form |
| 6 | Zeide & Nissenbaum 2018 (TRE, 10.1177/1477878518815340) | MOOCs / virtual education | 2,5,7,8 strong | mod-high | best step-8 exemplar; Nissenbaum-authored |
| 7 | de Groot 2024 (MHCP, 10.1007/s11019-024-10211-0) | genomic data across contexts | 1-9 multi-case | **low-mod** (post-2023, non-US) | ANNOTATED: `degroot2024_genomic_forensic.json` (forensic strand) |
| 8 | Wernick et al. 2025 (FAccT, arXiv:2506.00218) | algorithmic travel surveillance (FI) | 1-8, s6=incomplete_norms | **low** (2025, non-US) | ANNOTATED: `wernick2025_travel_surveillance.json` |

## Tier 2 — solid partials (steps 1-6)

9. Bowser et al. 2017 (CSCW) — citizen science — ANNOTATED: `bowser2017_citizen_science.json` (s2-s5,s7,s8; honest partial)
10. Sanfilippo et al. 2020 (JASIST) — disaster apps; strong s1/s4-s6 flow gold
11. Apthorpe et al. 2018 (IMWUT) — smart home IoT; parameter gold; HIGH contamination (canonical); also the step-5 population-validation target for Phase 3
12. Huang & Bashir 2015 (ASIS&T) — DTC genetic testing policies; low-mod contamination
13. King 2019 (CSCW) — DTC genetics users; mod-high
14. Kumar et al. 2020 (M&C) — children's password sharing; best step-2 exemplar
15. Martens et al. 2021 (Tech in Society) — contact tracing/triage (BE); non-US comparator to #5
16. Tran et al. 2025 (AIES, arXiv:2508.06760) — LLM chatbot data sharing; low contamination, modern practice

## Tier 3 — special purpose

17. Skeba & Baumer 2020 (CSCW) — facial recognition; **deliberate partial/misapplication exemplar** for probe (d)
18. Zimmer 2010 — Facebook T3; informal pre-heuristic pair to #2
19. Mutimukwe et al. 2023 — online proctoring scoping review; marginal
20. Brehm et al. 2023 — VR classrooms (PrivaCI'23); conceptual-only

Extra leads: PrivaCI 2023 short case studies (Napoli & Chiasson remote healthcare;
Marmorato et al. older adults, CA) — very low contamination, limited depth.

**Annotation order**: 3, 7, 8 (low-contamination fulls) → 2, 4, 6 → 5 →
partials as extraction-only gold (steps_present limited) → 17 as
misapplication exemplar. Target: 10-15 total incl. Kumar.
