# Dagspaces

One directory per pipeline. All invoked with `python -m dagspaces.{name}.cli pipeline=<yaml-name> model=<family>/<variant> [hydra/launcher=null] [runtime.debug=true runtime.sample_n=N]`.

---

## `historical_norms` — Norm extraction from fiction

**Purpose**: Extract CI information flow tuples + Raz norms from 10 Gutenberg novels.

**Entrypoint**: `python -m dagspaces.historical_norms.cli`

**Primary pipelines**:
| Pipeline | Purpose |
|---|---|
| `COLM_fetch_fiction` | Download + chunk novels (6000-char / 1000 overlap) + enrich with Wikipedia plot summaries. Outputs `chunks.parquet`. |
| `COLM_norms_fiction_prefetched` | DAG: reasoning → extraction → role_abstraction (needs `FICTION_CHUNKS_PATH`) |
| `COLM_norms_fiction` | fetch + extract in one run |
| `COLM_flows_fiction_prefetched` | Parallel track: reasoning → flow extraction (IFTs) |

**Outputs**: `structured_norms.parquet`, `abstracted_norms.parquet`, `structured_flows.parquet`.

**Default extractor**: `qwen2.5-72b/awq` (2-GPU, guided decoding for schema validity).

**Extras**: `ci_schema.py`, `schema_builders.py` — JSON Schemas for guided decoding; `logging_filters.py` — per-book logging.

---

## `grpo_training` — SFT + GRPO

**Purpose**: Train the CI-reasoning policy.

**Entrypoint**: `python -m dagspaces.grpo_training.cli`

**Pipelines**:
| Pipeline | DAG |
|---|---|
| `sft_only` | sft_data_prep → sft_training |
| `norm_universe_and_reward_prep` | norm_universe → reward_prep (one-time) |
| `full_training` | norm_universe + sft_data_prep → sft_training + reward_prep → grpo_training |
| `grpo_only` | grpo_training (reuses existing SFT checkpoint + cache) |
| `grpo_programmatic_only` | GRPO with `R_ground` zeroed; weights redistributed to 5 programmatic components |
| `grpo_only_online` / `grpo_only_online_external` | Online `R_ground` via embedding + judge HTTP servers |
| `grpo_with_norms` | Norm-aware variant |

**Stages** (see `runners/__init__.py`): `norm_universe`, `sft_data_prep`, `sft_training`, `reward_prep`, `grpo_training`.

**Training overrides**: `training/sft=<default|sft_27b|gpt_oss|no_negatives>`, `training/grpo=<default|colocate_1gpu_9b|online_rground_4gpu|...>`.

**Inputs (via `server.env`)**: `CI_REASONING_PATH`, `CI_EXTRACTION_PATH`, `ABSTRACTED_NORMS_PATH`, plus (for `grpo_only`) `SFT_CHECKPOINT_PATH`, `REWARD_CACHE_PATH`, `NORM_UNIVERSES_PATH`, `SFT_PAIRS_PATH`.

**Outputs**: `norm_universes.json` + `embeddings/*.npy`, `reward_cache.parquet`, `sft/checkpoint/`, `grpo/checkpoint/`.

---

## `goldcoin_hipaa` — Healthcare CI benchmark

**Purpose**: Evaluate CI applicability + compliance on 214 HIPAA / non-HIPAA court cases.

**Entrypoint**: `python -m dagspaces.goldcoin_hipaa.cli pipeline=full_eval model=<model>`

**Two branches**: applicability (214 cases) + compliance (107 HIPAA-relevant).

**Metrics**: accuracy, macro-F1, per-class P/R, confusion matrix.

**W&B project**: `goldcoin-hipaa`.

---

## `privacylens` — Agent action / QA leakage benchmark

**Purpose**: 493 seeds, QA probing across Subject/Vector/Target axes + agent-action leakage judgment.

**Entrypoint**: `python -m dagspaces.privacylens.cli pipeline=privacylens_clean model=<model>`

**Metrics**: per-axis QA accuracy, leakage rate.

**W&B project**: `privacylens-eval`.

---

## `vlm_geoprivacy_bench` — Visual geolocation CI

**Purpose**: MCQ over 783 images with expert-defined permissible disclosure granularity. Requires a VLM.

**Entrypoint**: `python -m dagspaces.vlm_geoprivacy_bench.cli pipeline=mcq_eval model=qwen3-vl-8b/instruct`

**W&B project**: `vlm-geoprivacy-bench`.

---

## `confaide` — ConfAIde tiers 1–2

**Purpose**: Probe disclosure/withholding under contextual constraints. Tiers 1–2 include human annotations for correlation analysis.

**Entrypoint**: `python -m dagspaces.confaide.cli pipeline=<pipeline> model=<model>`

---

## `cirl_vignettes` — CI-RL structured vignettes

**Purpose**: 729 vignettes with explicit flows + norms; tests norm-conditioned appropriateness judgment. Uses adapted ToolEmu code (`toolemu/`).

**Entrypoint**: `python -m dagspaces.cirl_vignettes.cli pipeline=<pipeline> model=<model>`

---

## `eval_all` — cross-benchmark convenience

Top-level aggregator for sweeping a model across all benchmarks.
