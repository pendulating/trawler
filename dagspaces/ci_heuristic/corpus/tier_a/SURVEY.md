# Tier A candidate survey — published CI analyses

Machine-assisted literature survey (2026-07-06) for gold-traversal candidates.
Ranked by usefulness; annotate top-down per ANNOTATION_GUIDE.md. Counts:
8 full traversals (substantive steps 7-9), 8 solid partials (1-6), 4
exemplars/marginal.

## Tier 1 — full/near-full traversals (annotate first)

| # | Case | Practice | Steps | Contamination | Notes |
|---|------|----------|-------|---------------|-------|
| 1 | Kumar, Zimmer & Vitak 2024 (CSCW, 10.1145/3653710) | Fitbit PFI → healthcare | all 9 | moderate | ANNOTATED: `kumar2024_fitbit.json` |
| 2 | Zimmer 2018 (SM+S, 10.1177/2056305118768300) | OkCupid data release / research ethics | all 9, s9=reject | high | ANNOTATED: `zimmer2018_okcupid.json` (via Wayback) |
| 3 | Bloch & Bashir 2017/18 (AHFE, 10.1007/978-3-319-60483-1_59) | cardiac implants + remote monitoring | self-labeled 9-step | **low** (obscure venue) | paywalled (Springer chapter) — needs MF library access |
| 4 | Sar & Al-Saggaf 2014 (Ethics Inf Technol, 10.1007/s10676-013-9329-y) | SNS third-party tracking | all 9, s9=reject w/ conditions | moderate | ANNOTATED: `sar2014_sns_tracking.json` (via CSU repository) |
| 5 | Vitak & Zimmer 2020 (SM+S, 10.1177/2056305120948250) | COVID contact tracing | all 9 (compressed), s6=no, s9=modify | high | ANNOTATED: `vitak2020_covid_surveillance.json` (via PMC) |
| 6 | Zeide & Nissenbaum 2018 (TRE, 10.1177/1477878518815340) | MOOCs / virtual education | s1-s8, s9 absent (directional) | mod-high | ANNOTATED: `zeide2018_moocs.json` (via author site) |
| 7 | de Groot 2024 (MHCP, 10.1007/s11019-024-10211-0) | genomic data across contexts | 1-9 multi-case | **low-mod** (post-2023, non-US) | ANNOTATED: `degroot2024_genomic_forensic.json` (forensic strand) |
| 8 | Wernick et al. 2025 (FAccT, arXiv:2506.00218) | algorithmic travel surveillance (FI) | 1-8, s6=incomplete_norms | **low** (2025, non-US) | ANNOTATED: `wernick2025_travel_surveillance.json` |

## Tier 2 — solid partials (steps 1-6)

9. Bowser et al. 2017 (CSCW) — citizen science — ANNOTATED: `bowser2017_citizen_science.json` (s2-s5,s7,s8; honest partial)
10. Sanfilippo et al. 2020 (JASIST) — disaster apps; strong s1/s4-s6 flow gold
11. Apthorpe et al. 2018 (IMWUT) — smart home IoT; parameter gold; HIGH contamination (canonical); also the step-5 population-validation target for Phase 3
12. Huang & Bashir 2015 (ASIS&T) — DTC genetic testing policies; low-mod contamination
13. King 2019 (CSCW) — DTC genetics users; mod-high
14. Kumar et al. 2020 (M&C) — children's password sharing — ANNOTATED: `kumar2020_passwords.json` (s2-s5,s7,s8; family-context primary, cross-context s5 variation)
15. Martens et al. 2021 (Tech in Society) — contact tracing/triage (BE); non-US comparator to #5
16. Tran et al. 2025 (AIES, arXiv:2508.06760) — LLM chatbot data sharing; low contamination, modern practice

## Tier 3 — special purpose

17. Skeba & Baumer 2020 (CSCW) — facial recognition — ANNOTATED: `skeba2020_facial_recognition.json` (s1,s3-s6; 11 source_deviations = probe-(d) reference material; via Wayback ACM PDF)
18. Zimmer 2010 — Facebook T3; informal pre-heuristic pair to #2
19. Mutimukwe et al. 2023 — online proctoring scoping review; marginal
20. Brehm et al. 2023 — VR classrooms (PrivaCI'23); conceptual-only

Extra leads: PrivaCI 2023 short case studies (Napoli & Chiasson remote healthcare;
Marmorato et al. older adults, CA) — very low contamination, limited depth.

**Corpus status (2026-07-07)**: 10 gold files — kumar2024 (exemplar), degroot2024 + wernick2025 (uncontaminated, held-out), zimmer2018, vitak2020, sar2014 (full 9-step), zeide2018 (s1-s8, step-8-rich), bowser2017 + kumar2020 (honest partials), skeba2020 (misapplication exemplar). Outcome coverage: reject x3, modify x2, no x1, incomplete_norms x1, absent x3.

**Remaining candidates** (annotate opportunistically): #3 Bloch & Bashir (paywalled — needs MF library access), #10 Sanfilippo 2020, #12 Huang & Bashir 2015, #13 King 2019, #15 Martens 2021 (PMC OA), #16 Tran 2025 (arXiv). Target of 10-15 incl. Kumar: MET at 10.
