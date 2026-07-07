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

**Gotcha — prompt defaults**: always use the fiction-specific pipeline variants when processing Gutenberg text (`COLM_norms_fiction_prefetched`, `COLM_flows_fiction_prefetched`, `ci_extraction_from_reasoning_fiction`). The base `ci_extraction.yaml` defaults to **prescriptive** prompts (written for religious texts) and will produce misaligned output on fiction.

**Building larger / different corpora**: the legacy `COLM_fetch_fiction` pipeline is hardcoded to a 10-book list and re-downloads from gutenberg.org on every run. For top-K-by-popularity, top-K-authors × N, or arbitrary ID lists with a durable on-disk cache, use `dagspaces.common.gutenberg` instead — see [howto/build-gutenberg-corpus.md](howto/build-gutenberg-corpus.md). It produces a `chunks.parquet` with a superset of the legacy schema, so `COLM_norms_fiction_prefetched` consumes it via `FICTION_CHUNKS_PATH` unchanged.

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

### Judges (privacylens, cirl_vignettes)

Both `privacylens` and `cirl_vignettes` use `dagspaces/common/judge_client.py` (`JudgeClient`) to score outputs via an OpenAI-compatible chat endpoint. The same client now works against:

- **vLLM** (default) — launch via `sbatch scripts/judge_server.sub`, `export JUDGE_SERVER_URL=http://host:port`
- **OpenAI** — `judge.base_url=https://api.openai.com/v1 judge.model_name=gpt-4o judge.api_key_env=OPENAI_API_KEY`
- **Anthropic** — `judge.base_url=https://api.anthropic.com/v1/ judge.model_name=claude-3-5-sonnet-20241022 judge.api_key_env=ANTHROPIC_API_KEY`
- **Google Gemini** — `judge.base_url=https://generativelanguage.googleapis.com/v1beta/openai/ judge.model_name=gemini-2.0-flash judge.api_key_env=GOOGLE_API_KEY`
- **OpenRouter / Together / Groq / Fireworks / DeepInfra** — set `base_url` to their OpenAI-compatible endpoint; `api_key_env` is required.

The provider is auto-detected from the hostname; override with `judge.provider=<name>` if needed. `chat_template_kwargs` (vLLM-specific) is only sent to vLLM endpoints. If a commercial provider rejects the structured-output `response_format`, the client retries once without it.

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

## `vlm_geoprivacy_aug` — Augmented visual geolocation CI (CVPR 2027)

**Purpose**: Fork of `vlm_geoprivacy_bench` that "inpaints" hypothetical capture contexts (smart glasses, dashcam, delivery robot, ...) into the prompt and measures paired per-image shifts in normative judgments vs. an un-framed baseline. For *"Tracing Norms Around the Imaging of Public Space"*. The original benchmark dagspace stays untouched.

**Entrypoint**: `python -m dagspaces.vlm_geoprivacy_aug.cli pipeline=mcq_eval_hypotheticals model=qwen3-vl-8b/instruct`

**Pipelines**: `mcq_eval_hypotheticals` (default), `freeform_eval_hypotheticals` (Q7-only shifts via granularity judge). Both pin `temperature=0.0` — sampling noise in paired comparisons masquerades as normative shift.

**Hypothetical sets** (`hypotheticals=` group): `capture_devices` (default; photo-taker bridge folded in), `capture_devices_raw` (no-bridge ablation), `none` (baseline-only control).

**Key metrics** (`compute_hypothetical_metrics`, full provenance): per-variant Q7 abstention rate + label distributions; paired vs. baseline: `flip_rate`, `mean_ordinal_shift`, `toward_abstention_rate`, `delta_abstention_rate`. Original human labels score the **baseline only** — they were annotated without device context.

**W&B project**: `vlm-geoprivacy-aug`.

See [howto/geoprivacy-hypotheticals.md](howto/geoprivacy-hypotheticals.md) for running arms and adding variants/dimensions.

---

## `ci_heuristic` — CI decision-heuristic traversal (CVPR 2027)

**Purpose**: Can LLMs work *through* Nissenbaum's 9-step CI decision heuristic — praxis, not just endpoint verdicts? Runs cases through a scaffolding ladder (L0 zero-shot verdict → L4 stepwise chain + guiding questions + worked exemplar; L5 deliberative pending) and persists per-step artifacts for scoring. Part 2 companion to `vlm_geoprivacy_aug`; E5 joins them on capture-device ids.

**Entrypoint**: `python -m dagspaces.ci_heuristic.cli ladder=l3 cases=pilot model=qwen3-8b/instruct`

**Config groups**: `ladder={l0..l4}`, `cases={pilot, held_out, tier_c}`. Greedy decoding pinned (scaffolding effects must not confound with sampling noise). Corpus + gold schema + annotation guide: `dagspaces/ci_heuristic/corpus/`.

**Chain mechanics**: 9 sequential *batched* rounds; each case's step-k prompt embeds its own accumulated state; per-step guided decoding (`schemas.py`); steps 1–5 carry a descriptive firewall ("do not evaluate yet"); parse failures degrade rather than abort. ⚠ In-process vLLM reloads the engine per round — use server mode (`model.server_url`) for real runs.

**Gold corpus**: Tier A published expert traversals (`corpus/tier_a/`, see `SURVEY.md` for the annotation queue), Tier B synthetic single-departure vignettes (pending), Tier C the Part 1 imaging practices (`corpus/tier_c/practices.yaml`).

**W&B project**: `ci-heuristic`.

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

Top-level aggregator for sweeping a model across all benchmarks. Runs each benchmark as a child subprocess; each benchmark then submits its own GPU SLURM job for its inference stage.

**Entrypoint**: `python -m dagspaces.eval_all.cli -m pipeline=all_benchmarks model=qwen3.5-9b/base`

**Default = async-judge mode**: `all_benchmarks.yaml` ships with `judge.mode=async` and `judge_sidecar.enabled=true`. Privacylens uses its `privacylens_async` export pipeline (no in-line judge call); a small CPU sidecar launched by the eval_all monitor forwards each judge request to `scripts/judge_server.sub` while other benchmarks proceed; after every benchmark dispatches, the monitor drains the sidecar and runs `privacylens_async_finalize` for parse + metrics. Requires a running judge endpoint reachable at `${JUDGE_BASE_URL,http://klara:8002}`.

**Live mode (legacy)**: Use `pipeline=all_benchmarks_live` for the strictly-live path that blocks each benchmark on its inline judge. Pick this when the cluster judge_server isn't up or for live-vs-async parity comparisons.

### Shared-server mode (avoid reloading the model per benchmark)

By default each benchmark loads vLLM fresh in its own SLURM GPU job — the model is loaded N times for N benchmarks. Enable `server_mode` to host the model in a single long-lived SLURM job and have every benchmark route inference over HTTP.

```bash
python -m dagspaces.eval_all.cli -m pipeline=all_benchmarks \
  model=qwen3.5-9b/base \
  server_mode.enabled=true
```

What happens:
1. `eval_all` launches one SLURM GPU job via submitit running `python -m vllm.entrypoints.openai.api_server --model ... --reasoning-parser <auto-detected>`.
2. That job writes its `host:port` to `vllm_server_logs/address.txt` under the run's output directory.
3. `eval_all` polls the address file and `/health`, then exports `VLLM_SERVER_URL=http://host:port/v1` into each child benchmark's environment.
4. `run_vllm_inference` detects the env var and routes every inference call through the OpenAI-compatible client (thread-pooled, ~32 concurrent requests) instead of instantiating `LLM()`.
5. On exit (normal, exception, or SIGTERM), the server job is cancelled.

**All five benchmarks work unchanged** — the server branch lives inside `run_vllm_inference`. Config knobs (all under `server_mode` in `dagspaces/eval_all/conf/config.yaml`):

| Key | Default | Purpose |
|---|---|---|
| `enabled` | `false` | Master switch |
| `launcher` | `slurm_gpu_1x` | SLURM launcher for the server job |
| `port` | `8000` | Server port |
| `served_model_name` | `""` | Falls back to `model.model_source` |
| `tensor_parallel_size` | `null` | Falls back to `model.engine_kwargs.tensor_parallel_size` |
| `max_model_len` | `null` | Falls back to `model.engine_kwargs.max_model_len` |
| `gpu_memory_utilization` | `0.90` | `--gpu-memory-utilization` |
| `reasoning_parser` | `"auto"` | `"auto"` detects family; `"none"` disables; else explicit name |
| `startup_timeout_s` | `900` | Wait for `/health` to return 200 |
| `extra_args` | `[]` | Passed verbatim to `api_server` |

**Caveats**:
- LoRA checkpoints: `run_vllm_inference`'s server branch doesn't yet attach LoRA adapters dynamically. Use in-process mode (`server_mode.enabled=false`) for LoRA evals.
- Multi-GPU models: set `server_mode.launcher=slurm_gpu_2x` (or larger) to match the model's `tensor_parallel_size`.
- Server client concurrency can be tuned via `VLLM_SERVER_CLIENT_CONCURRENCY` env var (default 32).
