# Batch judging via OpenAI Batch API

How Trawler evaluates CI benchmarks when the judge is a commercial LLM
(OpenAI / compatible) without burning per-minute rate limits and without
requiring an OpenAI API key on the machine that runs inference.

**When to use:** You want to judge PrivacyLens or CIRL-Vignettes trajectory
evals with `gpt-5.2` / `gpt-4o` / etc. at 50% cost, asynchronous (24h),
hand-off-able (submit from a different account than the one that ran the
task LLM), and **1:1 parity** with the upstream SALT-NLP/PrivacyLens
evaluation harness.

**When not to use:**
- `grpo_training`'s online reward judge — incompatible with 24h turnaround.
- `vlm_geoprivacy_bench`'s granularity judge — uses `run_vllm_inference`
  directly instead of `JudgeClient`; would need its own port.
- Quick local smoke tests — live vLLM judging is still supported and faster
  for ≤100 examples.

Underlying OpenAI API docs are mirrored at
[`integrations/openai-batch-api.md`](openai-batch-api.md).

## Architecture

Batch judging splits the eval into two phases separated by an
out-of-band batch submission to OpenAI:

```
┌────────────────────┐    ┌───────────────┐    ┌─────────────────┐
│  Phase 1: Export   │ →  │  OpenAI Batch │ →  │ Phase 2: Finalize│
│  (local GPUs)      │    │  API (24h)    │    │  (CPU only)      │
├────────────────────┤    ├───────────────┤    ├─────────────────┤
│ load_dataset       │    │ batch_api     │    │ batch_api merge │
│ qa_probe_inference │    │   submit      │    │ parse_responses │
│ action_inference   │    │   status      │    │ compute_metrics │
│ *_judge_export     │    │   fetch       │    │                 │
│  → requests.jsonl  │    │  output.jsonl │    │  → results.parq │
│  → items.parquet   │    │               │    │  → metrics.json │
│  → manifest.json   │    │               │    │                 │
└────────────────────┘    └───────────────┘    └─────────────────┘
```

**Phase 1 (export)** runs the task LLM on local GPUs exactly as the live
pipeline does, then substitutes each live judge stage with a sibling
`*_judge_batch_export` stage that writes an OpenAI Batch API JSONL file
instead of calling the judge. No network calls are made during judge
export — the machine running Phase 1 does **not** need an
`OPENAI_API_KEY`. This lets you hand the emitted JSONLs to a
collaborator whose OpenAI account pays for the batch.

**Phase 2 (finalize)** runs on any machine with `OPENAI_API_KEY` set (not
necessarily the same as Phase 1). You submit each `requests.jsonl`,
poll for completion, download `output.jsonl`, and run a dagspace-local
finalize script that merges the downloaded responses back into the
pending parquet, reapplies the live-mode parsers, and runs
`compute_metrics` — producing a `results.parquet` with the same schema
as a live-judge run.

## 1:1 parity with upstream SALT-NLP/PrivacyLens

The judge prompts are **byte-identical** to the upstream PrivacyLens
evaluation harness
([`evaluation/evaluate_final_action.py`](https://github.com/SALT-NLP/PrivacyLens/blob/main/evaluation/evaluate_final_action.py)),
verified by round-tripping through upstream's own `prepare_prompt_to_*`
functions and diffing the output:

| Aspect | Trawler (this doc) | Upstream |
|---|---|---|
| Leakage judge fan-out | One call per `(row, sensitive_info_item)` pair | Same |
| Helpfulness judge | One call per row | Same |
| Message turns | Single `user` turn (no system message) | Same |
| Leakage prompt header | `Judge whether the following action leaks...` + literal `${...}` schema demo + 2 positive few-shots | Byte-identical |
| Helpfulness prompt header | `Your task is to evaluate [Agent]'s helpfulness...` + rubric + 3 evaluation notes + literal `${...}` schema demo | Byte-identical (zero-shot) |
| Action pre-processing | Strip leading `Action:` substring (`s[s.find("Action:")+len("Action:"):]`) | Same |
| Row-skip rule | Skip rows whose `generated_action` is not a string containing `Action:` | Same |
| Row-level aggregation | Leakage: OR across per-secret answers. Helpfulness: one score per row. | Same |

**Difference on purpose:** upstream parses `Answer: Yes/No` /
`Answer: Poor/Unsatisfactory/Good/Excellent` via substring matching on
the model's free-form output. Trawler layers **JSON guided decoding** on
top of the same prompts via OpenAI `response_format: json_schema` →
`YesNoResult` / `HelpfulnessResult`. The upstream prompts' trailing
`Reasoning: Let's think step by step in order to` priming still seeds
the chain-of-thought, which gets absorbed into the JSON `reasoning`
field instead of continuing as raw text. The parsers prefer the JSON
path and fall back to upstream's substring semantics so old parquets
(from before the refactor) and any provider that ignores
`response_format` still decode correctly.

### Upstream quirks we inherit verbatim

Preserved for byte-parity — do not "fix":

- **`""to""` / `""body""` / `""recipient_id""` double-double-quotes** in
  the leakage few-shot JSON blobs. An upstream source artifact; we keep
  it so any judge prompted identically won't see surface drift.
- **Both leakage few-shots are positive (`Answer: Yes.`)**. There is no
  negative (`No`) exemplar in upstream. Running an A/B against a
  balanced few-shot block is a follow-up experiment, not a parity fix.
- **`it leak the sensitive information`** (sic) typo in the second
  few-shot's reasoning.
- **Greedy `Poor → Unsatisfactory → Good → Excellent` helpfulness
  parser order.** A `'0'` digit anywhere in the answer tail wins over
  `'Excellent'` or `'3'`. Upstream behavior; kept as the JSON-fallback.

### Files that implement this

| Path | Role |
|---|---|
| `dagspaces/privacylens/prompts.py` | `build_leakage_judge_prompt_per_secret`, `build_helpfulness_judge_prompt`, `_clean_generated_action`, `_extract_t_dict`, `_list_sensitive_items`. Single source of truth for prompt text. |
| `dagspaces/privacylens/stages/llm_inference.py` | `_build_leakage_fanout_items`, `_get_batch_export_client`, `export_leakage_judge_batch`, `export_helpfulness_judge_batch`, plus the live `run_*_judge_inference` counterparts that call the same prompt builders. |
| `dagspaces/privacylens/stages/parse_responses.py` | `parse_leakage_responses` / `parse_helpfulness_responses` — JSON-first, upstream-substring fallback. |
| `dagspaces/cirl_vignettes/stages/judge_leakage.py` | Fan-out per secret + batch export helpers, mirrors the privacylens structure. |
| `dagspaces/cirl_vignettes/stages/judge_helpfulness.py` | One-call-per-row batch export. |
| `dagspaces/common/judge_client.py` | `JudgeClient` with `offline=True` and `export_batch_jsonl()` method. |
| `dagspaces/common/batch_api.py` | `submit` / `status` / `fetch` / `merge` CLI and library. |
| `scripts/prepare_judge_batches.py` | Standalone utility — emit judge batches from a pre-existing stage parquet. |
| `scripts/privacylens_batch_finalize.py` | Aggregate PrivacyLens leakage + helpfulness outputs and run compute_metrics. |
| `scripts/cirl_trajectory_batch_finalize.py` | Same for CIRL-Vignettes trajectory eval. |
| `scripts/test_batch_export.py` | No-network smoke test (8 checks). |

## Pipelines and config knobs

### Per-dagspace pipeline variants

| Dagspace | Live pipeline | Batch pipeline |
|---|---|---|
| `privacylens` | `privacylens_clean` | `privacylens_clean_batch` |
| `cirl_vignettes` (trajectory eval) | `cirl_trajectory_eval` | `cirl_trajectory_batch` |

Each batch pipeline **stops at the export step**. It writes
`pending.parquet` + `items.parquet` + `requests.jsonl` + `manifest.json`
and terminates. Downstream `compute_metrics` / `compute_trajectory_metrics`
run via the finalize script, not via the pipeline DAG, so you don't have
to set `HYDRA_RUN_DIR` and re-invoke the pipeline after downloading
`output.jsonl`.

### Cross-benchmark variant (eval_all)

`dagspaces/eval_all/conf/pipeline/all_benchmarks_batch_export.yaml` runs
every eval end-to-end, but for the two judged benchmarks it substitutes
the batch pipeline and appends `judge.mode=batch_export` via
`extra_args`. Non-judged benchmarks (goldcoin, confaide,
cirl_vignettes probing, vlm_geoprivacy) run their regular pipelines
unchanged — they have no judge stage so there's nothing to export.

```bash
python -m dagspaces.eval_all.cli -m \
    pipeline=all_benchmarks_batch_export \
    model=qwen3.5-9b/base
```

Override the judge model for the whole variant at the CLI:

```bash
python -m dagspaces.eval_all.cli -m \
    pipeline=all_benchmarks_batch_export model=qwen3.5-9b/base \
    '++judge_batch_extras=[judge.mode=batch_export,judge.batch.target_model=gpt-4o]'
```

### Config knobs

In each judged dagspace's `conf/config.yaml` the `judge:` block exposes:

```yaml
judge:
  mode: live                       # live | batch_export
  # ----- live-mode fields (ignored when mode=batch_export) -----
  base_url: ${judge_server_url}
  model_name: default
  provider: null
  api_key: null
  api_key_env: null
  max_workers: 8
  temperature: 0.0                 # shared with batch-export (body.temperature)
  max_tokens: 1024                 # shared with batch-export (body.max_tokens)
  # ----- batch-export fields (only consulted when mode=batch_export) -----
  batch:
    target_model: gpt-5.2          # model name written into each JSONL body.model
    target_endpoint: /v1/chat/completions
    output_jsonl: null             # finalize override; default = output.jsonl next to pending.parquet
```

`judge.mode=batch_export` is the **only** knob you need to flip. Every
live-mode field (`base_url`, `provider`, `api_key*`) is **ignored** in
batch-export mode — the export stages construct the `JudgeClient` with
`provider="openai"` + `offline=True` hardcoded and read only
`judge.batch.target_model`, `judge.batch.target_endpoint`,
`judge.temperature`, `judge.max_tokens`.

## Output layout

Every batch-export judge stage writes a sidecar directory with four files:

```
<output_root>/<stage>/
  requests.jsonl    # Batch API input — submit this
  items.parquet     # custom_id ↔ (row_idx, sub_idx, secret) mapping
  pending.parquet   # original df with empty *_judge_text column
  manifest.json     # count, model, provider, fanout, skipped_row_count
```

After Phase 2:

```
<output_root>/<stage>/
  requests.jsonl
  items.parquet
  pending.parquet
  manifest.json
  output.jsonl      # downloaded from batch_api fetch
  results.parquet   # merged + parsed, same schema as live-mode output
```

### `items.parquet` schema

| Column | Type | Notes |
|---|---|---|
| `judge_custom_id` | string | e.g. `privacylens:leakage_judge:12:3` |
| `row_idx` | int64 | Index into `pending.parquet` |
| `sub_idx` | int64 | 0..N-1 within each `row_idx` (leakage fan-out) |
| `secret` | string | The sensitive item judged in this request (leakage only) |

For helpfulness, `items.parquet` has `judge_custom_id` + `row_idx`
only — one row per request, no `sub_idx` / `secret`.

### `custom_id` format

| Stage | Format |
|---|---|
| `privacylens:leakage_judge:{row}:{sub}` | PrivacyLens leakage, per-secret fan-out |
| `privacylens:helpfulness_judge:{row}` | PrivacyLens helpfulness, per-row |
| `cirl_vignettes:judge_leakage:{row}:{sub}` | CIRL-Vignettes leakage, per-secret fan-out |
| `cirl_vignettes:judge_helpfulness:{row}` | CIRL-Vignettes helpfulness, per-row |

## End-to-end flow (PrivacyLens)

```bash
# 1. Export phase — task LLM runs on GPU, judge stages write JSONL.
#    No OPENAI_API_KEY required on this host.
python -m dagspaces.privacylens.cli pipeline=privacylens_clean_batch \
    judge.mode=batch_export \
    judge.batch.target_model=gpt-5.2

# 2. Submit each judge JSONL from any host with OPENAI_API_KEY set.
python -m dagspaces.common.batch_api submit \
    <run_dir>/privacylens_eval/outputs/leakage_judge_batch/requests.jsonl
python -m dagspaces.common.batch_api submit \
    <run_dir>/privacylens_eval/outputs/helpfulness_judge_batch/requests.jsonl
# submit back-writes batch_id into the adjacent manifest.json

# 3. Poll and fetch once status == completed.
python -m dagspaces.common.batch_api status <batch_id>
python -m dagspaces.common.batch_api fetch <batch_id> \
    -o <run_dir>/privacylens_eval/outputs/leakage_judge_batch/output.jsonl

# 4. Finalize: OR-aggregate leakage, reapply parsers, compute_metrics.
python scripts/privacylens_batch_finalize.py \
    --run-dir <run_dir>/privacylens_eval
```

`_finalize_leakage` reads `items.parquet` + `output.jsonl`, parses each
per-secret response with the JSON-first / upstream-substring-fallback
extractor, groups by `row_idx`, and ORs the per-secret answers into a
row-level `leak_judge_text`. `_finalize_helpfulness` maps each response
back to its `row_idx` via `items.parquet` (skipped rows get
`'{"score": 0, "reasoning": "no valid action"}'`), then calls
`parse_helpfulness_responses`. Both reapply the live-mode parsers so
the final `results.parquet` is schema-identical to a live run.

## Standalone: emit batches from an existing stage parquet

If you already have a `<stage>/results.parquet` on disk from a prior run
and just want the judge JSONLs without re-running the full pipeline:

```bash
# Default: privacylens.agent_action_inference → leakage + helpfulness
python scripts/prepare_judge_batches.py \
    --input outputs/.../agent_action_inference/results.parquet \
    --stage privacylens.agent_action_inference

# Only one judge
python scripts/prepare_judge_batches.py \
    --input outputs/.../agent_action_inference/results.parquet \
    --stage privacylens.agent_action_inference \
    --judges helpfulness

# Override the target model
python scripts/prepare_judge_batches.py \
    --input outputs/.../trajectory_inference/dataset.parquet \
    --stage cirl_vignettes.trajectory_inference \
    --target-model gpt-4o
```

Registry of supported stages lives in
`scripts/prepare_judge_batches.py::_build_registry()`. Adding a new
stage is ~10 lines — point it at the dagspace's existing
`export_*_judge_batch` functions.

## `JudgeClient` offline mode

```python
from dagspaces.common.judge_client import JudgeClient

client = JudgeClient(
    base_url="https://api.openai.com/v1",  # placeholder, never contacted
    model_name="gpt-5.2",                  # must be explicit (no "default")
    provider="openai",
    offline=True,                          # skips API-key check + SDK client
)

manifest = client.export_batch_jsonl(
    items=items,
    build_messages_fn=build_messages,
    output_path="requests.jsonl",
    custom_id_fn=lambda item, idx: f"my_benchmark:my_judge:{idx}",
    json_schema=MySchema.model_json_schema(),
    schema_name="MySchema",
    endpoint_url="/v1/chat/completions",
)
```

When `offline=True`:

- The commercial-provider API-key requirement is skipped.
- `self._client` (OpenAI SDK) is never constructed.
- `health_check()` and `_call_single()` raise `RuntimeError` if called.
- `export_batch_jsonl()` works normally and emits a JSONL suitable for
  later submission from any host that *does* have credentials.

The `provider="vllm"` guard in `export_batch_jsonl()` still fires — the
Batch API is OpenAI-only, so pointing offline export at a vLLM endpoint
is rejected with a clear error message.

## Smoke test

```bash
python scripts/test_batch_export.py
```

Runs 8 no-network checks (takes ~10s; must be run from the repo root):

1. `vLLM` provider rejected by `export_batch_jsonl`.
2. Basic OpenAI export produces 3 unique requests with embedded schema
   and no `self._client` built.
3. Duplicate `custom_id` raises.
4. Offline mode blocks `health_check` / `_call_single` with `RuntimeError`.
5. `model_name="default"` rejected with explicit error.
6. PrivacyLens fixture: 2 rows × 2 sensitive items = **4 fanned-out
   leakage requests** with the expected `row:sub` custom IDs + 2
   helpfulness requests; both assert `response_format.json_schema.name`
   and verify real `User Instruction:` / `Past Action Trajectory:`
   content lands in the prompt body (no dict dump).
7. CIRL-Vignettes fixture: fan-out with a skipped row (no valid `Action:`).
8. Round-trip merge: `merge_batch_output` joins a mocked `output.jsonl`
   back into a pending parquet with matched/missing bookkeeping.

The test deliberately `os.environ.pop("OPENAI_API_KEY", None)` at the
top so it proves the export path doesn't need credentials.

## Verifying a real run against upstream

```python
import json, difflib, pandas as pd

# 1. Grab upstream's prompt function directly (skipping its imports).
src = open("/tmp/upstream_evaluate_final_action.py").read()  # from the repo
idx = src.find("def prepare_prompt_to_identify_leakage")
code = src[idx:src.find("def main(")]
ns = {}; exec(code, ns)

# 2. Load our first batch request.
first = json.loads(open(
    "<run_dir>/outputs/leakage_judge_batch/requests.jsonl"
).readline())

# 3. Reconstruct the (user_name, secret, action) for that row.
items = pd.read_parquet("<run_dir>/outputs/leakage_judge_batch/items.parquet")
pending = pd.read_parquet("<run_dir>/outputs/leakage_judge_batch/pending.parquet")
row_idx0 = int(items.iloc[0]["row_idx"])
secret0 = str(items.iloc[0]["secret"])
row0 = pending.iloc[row_idx0]
action0 = str(row0["generated_action"])
if "Action:" in action0:
    action0 = action0[action0.find("Action:")+len("Action:"):].strip()

# 4. Byte-diff.
upstream_prompt = ns["prepare_prompt_to_identify_leakage"](
    row0["T"]["user_name"], secret0, action0,
)
ours_prompt = first["body"]["messages"][0]["content"]
assert upstream_prompt == ours_prompt, list(difflib.unified_diff(
    upstream_prompt.splitlines(), ours_prompt.splitlines(), lineterm="",
))[:10]
```

## Known limitations

- **OpenAI-only.** The JSONL shape (`/v1/chat/completions` endpoint,
  `response_format: json_schema`) is the OpenAI Batch API protocol.
  Anthropic and Gemini have their own batch endpoints with different
  shapes; `export_batch_jsonl` warns when called with a non-OpenAI
  provider but still emits OpenAI-shaped lines.
- **Per-batch limits:** 50,000 requests and 200 MB per input file
  (upstream OpenAI limits). `export_batch_jsonl` warns if either is
  exceeded. Split `requests.jsonl` by hand if you hit the cap.
- **Leakage fan-out multiplies request count.** A 493-row PrivacyLens
  eval fans out to ~1487 leakage requests (mean 3 secrets/row on the
  HF dataset, max observed: 16). Helpfulness stays at 1:1. Plan batch
  size accordingly.
- **Unique `custom_id`s required.** The utility checks uniqueness at
  export time and raises on collision; custom `custom_id_fn` callers
  are responsible for their own uniqueness scheme.

## See also

- [`integrations/openai-batch-api.md`](openai-batch-api.md) — upstream
  OpenAI docs mirror.
- [`benchmarks/privacylens.md`](../benchmarks/privacylens.md) — what the
  judge stages are actually evaluating.
- [`dagspaces.md`](../dagspaces.md) — where privacylens and
  cirl_vignettes fit in the overall Trawler layout.
