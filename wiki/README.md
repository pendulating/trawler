# Trawler Wiki

Engineering reference for the COLM 2026 paper *"Reinforcing privacy reasoning in LLMs via normative simulacra from fiction"* (Franchi, Choksi, Triedman, Nissenbaum — Cornell Tech).

Manuscript: `papers/colm26_normative-simulacra/`. Experiment runbook: `EXPERIMENTS.md`.

## Index

**Understanding the system**
- [overview.md](overview.md) — paper goals, end-to-end pipeline, three phases
- [architecture.md](architecture.md) — dagspace pattern, shared configs, StageRunner protocol
- [dagspaces.md](dagspaces.md) — the six active dagspaces

**Method deep-dives**
- [grpo-reward.md](grpo-reward.md) — composite reward components and contrastive scoring
- [normative-simulacra.md](normative-simulacra.md) — IFT + Raz norm extraction from fiction
- [thinking-modes.md](thinking-modes.md) — `<think>` token handling across SFT, GRPO, and eval

**Benchmarks** — per-benchmark research references (see [benchmarks/README.md](benchmarks/README.md))
- [benchmarks/confaide.md](benchmarks/confaide.md) — ConfAIde (ICLR 2024)
- [benchmarks/privacylens.md](benchmarks/privacylens.md) — PrivacyLens (NeurIPS 2024)
- [benchmarks/contextreasoner.md](benchmarks/contextreasoner.md) — ContextReasoner (HKUST, PPO baseline)

**Infrastructure**
- [models.md](models.md) — model config conventions, zoo layout
- [slurm-and-env.md](slurm-and-env.md) — launchers, `server.env`, GPU sanitization

**How-to (bootstrap extensions)**
- [howto/add-model.md](howto/add-model.md) — download + yaml + wire into pipelines
- [howto/add-stage.md](howto/add-stage.md) — new stage in an existing dagspace
- [howto/add-dagspace.md](howto/add-dagspace.md) — scaffold a new pipeline / benchmark
- [howto/add-eval-benchmark.md](howto/add-eval-benchmark.md) — build a new CI eval from scratch
- [howto/run-experiments.md](howto/run-experiments.md) — running training, eval, ablations

## Related top-level docs

| File | Purpose |
|---|---|
| `CLAUDE.md` | Authoritative project instructions for coding assistants |
| `EXPERIMENTS.md` | Full COLM execution runbook (commands, ablation matrix) |
| `README.md` | High-level project description |
| `server.env.example` | Cluster/site config template |
