# Trawler Wiki

Engineering reference for the COLM 2026 paper *"Reinforcing privacy reasoning in LLMs via normative simulacra from fiction"* (Franchi, Choksi, Triedman, Nissenbaum — Cornell Tech).

Manuscript: `papers/colm26_normative-simulacra/`. Experiment runbook: `EXPERIMENTS.md`.

## Index

**Understanding the system**
- [overview.md](overview.md) — paper goals, end-to-end pipeline, three phases
- [architecture.md](architecture.md) — dagspace pattern, shared configs, StageRunner protocol
- [dagspaces.md](dagspaces.md) — the six active dagspaces
- [metric-trust.md](metric-trust.md) — what to quote in the paper per benchmark, format-adherence FAIL gate, `metric_provenance` schema

**Method deep-dives**
- [2026-07-31_kto_plan.md](2026-07-31_kto_plan.md) — **⚠️ ACTIVE (2026-07-31): KTO pivot (k-series) pre-registration** — after the m2 wave-A clean negative (GRPO exploration failure: no within-group disagreement on the violation flow → no gradient), offline prospect-theoretic tuning on norm-grounded binary desirability labels; label-corrected counterfactual pairs from the audited m2 reward stack, held-out chunk split, mandatory corrected-SFT control
- [grpo_redesign/](grpo_redesign/README.md) — modular GRPO redesign (m-series). **CLOSED as a negative 2026-07-31**: the m1 wave was invalidated by four stack defects; the repaired m2 wave-A ran clean and still held every cell at the blanket floor (post-mortems: `notebooks/normative-simulacra/grpo_m{1,2}_wave_analysis_*.py`). The v9-ckpt100 keeper stays the paper's model; this tree is the ablation/limitations record
- [grpo-reward.md](grpo-reward.md) — composite reward, directional composition, listwise R_ground, deontic tier ladder (the v9–v12a lineage; keeper-era reference — **rewritten 2026-07-20 to be 1:1 with the code**, incl. a deviations-from-vanilla-GRPO checklist for the camera-ready)
- [grpo_training_field_notes/](grpo_training_field_notes/README.md) — dated scratch notes from training-run analysis (reward traces, sweeps, gold-label behavior)
- [normative-simulacra.md](normative-simulacra.md) — IFT + Raz norm extraction from fiction
- [thinking-modes.md](thinking-modes.md) — `<think>` token handling across SFT, GRPO, and eval
- [mcq-canary-protocol.md](mcq-canary-protocol.md) — **TBD (designed 2026-07-18, not implemented)**: per-epoch MMLU-val canary for SFT checkpoints — parseable-rate (format drift) + accuracy-among-parseable (forgetting), split-level firewall vs reported MMLU, pre-registered veto thresholds

**Benchmarks** — per-benchmark research references (see [benchmarks/README.md](benchmarks/README.md))
- [benchmarks/confaide.md](benchmarks/confaide.md) — ConfAIde (ICLR 2024)
- [benchmarks/privacylens.md](benchmarks/privacylens.md) — PrivacyLens (NeurIPS 2024)
- [benchmarks/contextreasoner.md](benchmarks/contextreasoner.md) — ContextReasoner (HKUST, PPO baseline)

**Infrastructure**
- [canonical-models.md](canonical-models.md) — **the canonical task-LLM set** (13 models, est. 2026-07-13): what to evaluate on and why; the gpt-oss **harmony** requirement; vLLM version blockers
- [models.md](models.md) — model config conventions, zoo layout
- [slurm-and-env.md](slurm-and-env.md) — launchers, `server.env`, GPU sanitization

**Integrations**
- [integrations/batch-judging.md](integrations/batch-judging.md) — OpenAI Batch API judging for PrivacyLens + CIRL-Vignettes, 1:1 upstream prompt parity, offline export, finalize flow
- [integrations/openai-batch-api.md](integrations/openai-batch-api.md) — upstream OpenAI Batch API reference (mirror)
- [integrations/wandb-parity.md](integrations/wandb-parity.md) — local metrics.json ↔ W&B parity contract: mechanical mirror, bidirectional linkage, derived sweep groups, verify/push/pull sync tool

**Changelog (run-affecting changes)**
- [changelog/2026-07-20_grpo_methodology_congruency_review.md](changelog/2026-07-20_grpo_methodology_congruency_review.md) — GRPO wiki↔code↔paper congruency review: listwise R_ground is a deliberate GRPO-tenet deviation the camera-ready must own; epsilon_high inert at μ=1; `loss_type` silently inherited (TRL 1.8.0 `dapo`); judge-server default (Gemma-4-31B-it) mismatches the dagspace's keeper judge (Qwen3.6-27B); 5 wiki drifts fixed by rewriting grpo-reward.md
- [changelog/2026-07-13_canonical_models_and_harmony.md](changelog/2026-07-13_canonical_models_and_harmony.md) — **⚠️ IN PROGRESS, has a "Resume here"**: canonical 13-model set established; gpt-oss was returning its hidden `analysis` channel as its answer (all prior gpt-oss numbers suspect) — fixed; vLLM 0.19.1→0.25 upgrade blocked on a CUDA-13/driver question that needs a GPU node; the overnight top100 chain died with its SLURM session (reasoning done, extraction never submitted)
- [changelog/2026-05-12_privacylens_action_prompt_react.md](changelog/2026-05-12_privacylens_action_prompt_react.md) — PrivacyLens action-inference prompt rewritten to upstream SALT-NLP ReAct (May 12+ runs not byte-comparable to Mar/Apr)
- [changelog/2026-06-09_code_review_norms_grpo.md](changelog/2026-06-09_code_review_norms_grpo.md) — code review of historical_norms + grpo_training vs. paper claims; ALL code findings fixed same day (λ primary→1.0, ranked judge-failure neutrality, chunker size invariant, prescreen cache key, universe dedup, vignette accounting, embedding-outage abort, eval-reward trend gate). Before sweeping: re-run `norm_universe_and_reward_prep`; manuscript actions listed at the end
- [changelog/2026-06-09_wandb_logging_rationalization.md](changelog/2026-06-09_wandb_logging_rationalization.md) — W&B logging overhaul for training + extraction: `rground/*` per-step reward health, `prescreen/*` + `gates/*` surfaced, `grpo_runtime` config = full training_metadata, `data_quality/*` per-stage QA scalars, norm-universe artifact, readable run groups, trace size cap
- [changelog/2026-06-09_ner_quality_checks.md](changelog/2026-06-09_ner_quality_checks.md) — manual character blocklist replaced by layered spaCy-NER QA (scales past the 10 hand-listed novels); flows track gains its first quality check; norm quality recomputed post-abstraction (61% of old flags were stale; ~1.5% true residual leaks now caught)

**How-to (bootstrap extensions)**
- [howto/add-model.md](howto/add-model.md) — download + yaml + wire into pipelines
- [howto/add-stage.md](howto/add-stage.md) — new stage in an existing dagspace
- [howto/add-dagspace.md](howto/add-dagspace.md) — scaffold a new pipeline / benchmark
- [howto/add-eval-benchmark.md](howto/add-eval-benchmark.md) — build a new CI eval from scratch
- [howto/build-gutenberg-corpus.md](howto/build-gutenberg-corpus.md) — top-K / top-author selection + durable disk cache
- [howto/colm-100-novel-run.md](howto/colm-100-novel-run.md) — scale COLM norm + flow extraction from 10 to 100 novels (3-pipeline plan, ~78h wall)
- [howto/geoprivacy-hypotheticals.md](howto/geoprivacy-hypotheticals.md) — run/extend the augmented geoprivacy benchmark (capture-context shifts, CVPR 2027)
- [howto/ci-heuristic-traversals.md](howto/ci-heuristic-traversals.md) — run the CI decision-heuristic praxis experiments E1–E7 (scaffolding ladder, deliberative structures, probes; CVPR 2027)
- [howto/run-experiments.md](howto/run-experiments.md) — running training, eval, ablations

**Reviews**
- [jul19_refactoring.md](jul19_refactoring.md) — 2026-07-19 redundancy/complexity review (orchestrator copy-paste, silent excepts, JSON-extraction sprawl, god modules) with a prioritized roadmap
- [jul19_orchestrator_unification_plan.md](jul19_orchestrator_unification_plan.md) — test-first plan to unify the 7 eval orchestrators (Finding 1)

## Related top-level docs

| File | Purpose |
|---|---|
| `CLAUDE.md` | Authoritative project instructions for coding assistants |
| `EXPERIMENTS.md` | Full COLM execution runbook (commands, ablation matrix) |
| `README.md` | High-level project description |
| `server.env.example` | Cluster/site config template |
