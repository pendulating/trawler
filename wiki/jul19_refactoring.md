# Refactoring Review — 2026-07-19

A review of the Trawler codebase (`dagspaces/`, ~58k lines of Python) for
redundancy and unnecessary complexity, judged against the
[Jane Street House Style](https://opensource.janestreet.com/standards/)
(small, explicit, non-clever modules; no speculative abstraction; make illegal
states unrepresentable; prefer duplication over the *wrong* abstraction, but
eliminate duplication once the right abstraction is clear) and scientific
computing norms (reproducibility, fail-loud, minimal magic, testable units).

This is a **living document** — findings are appended as the codebase is
crawled. Each finding is tagged with a severity and an effort estimate.

Severity: 🔴 high (large duplication / correctness risk) · 🟡 medium · 🟢 low
Effort: S (<1 day) · M (1–3 days) · L (>3 days)

---

## Summary of top findings

| # | Finding | Severity | Effort |
|---|---------|----------|--------|
| 1 | 9 eval dagspaces each re-implemented the same run loop (~95% copy-paste; ~3000 duplicated lines) — **DONE 2026-07-19**, see plan | ✅ | M |
| 2 | The SLURM/NFS result-waiting block (~120 lines) duplicated verbatim across the eval orchestrators — **DONE 2026-07-19** (folded into Finding 1 as `await_slurm_result`) | ✅ | S |
| 3 | Runner boilerplate: every runner is "read parquet → call fn → write parquet → StageResult" | 🟡 | M |
| 4 | Deprecated dagspaces (`.uair`, `.rule_tuples`) still in tree, ~12k lines — but `common/` still imports one symbol from `.uair` | 🟡 | S |
| 5 | Pervasive bare `except Exception: pass` swallows errors silently (anti-pattern for science code) | 🔴 | M |

---

## Finding 1 — Orchestrator copy-paste across dagspaces ✅ DONE (2026-07-19)

> **Status: implemented.** All **nine** eval dagspaces now run on one shared
> loop in `dagspaces/common/orchestrator.py` (`OrchestratorHooks` +
> `run_experiment` + `await_slurm_result`); each `orchestrator.py` is a thin
> stub whose only dagspace-specific code is `_log_eval_metrics`. ~3000 lines
> deleted; full suite green; parity test covers all nine. Commits:
> `7d79921` (generic loop), `6c34574` (mmlu), `5aedadb` (remaining 8).
>
> **Correction vs. the original review:** there were **nine** copies, not seven
> — `vlm_geoprivacy_aug` and `ci_heuristic` were additional copies the initial
> survey missed; both migrated too.
>
> **Plan:** [jul19_orchestrator_unification_plan.md](jul19_orchestrator_unification_plan.md).
>
> **Phase 2 declined (2026-07-19):** `historical_norms` and `grpo_training`
> are deliberately *not* consolidated. They differ in purpose (norm extraction
> vs. policy training) and are a distinct "training loop" shape — WANDB_GROUP
> propagation, per-stage GPU sanitization, bespoke table logging, and NFS
> result-wait algorithms that differ from the eval loop *and from each other*.
> Unifying them would be the wrong abstraction and would change waiting
> behavior on the frozen training pipeline for a 2-file payoff. Left as-is by
> design.

**Evidence.** Nine dagspaces ship their own `orchestrator.py` with a `run_experiment`
loop. Seven of them (the eval benchmarks) are 441–501 lines explicitly
documented as copies of one another; `historical_norms` and `grpo_training`
carry their own near-identical copies of the same loop (with thin
`_CONF_DIR`-binding wrappers):

```
goldcoin_hipaa/orchestrator.py   485  "Copied from vlm_geoprivacy_bench/..."
vlm_geoprivacy_bench/orchestrator.py 501  (source)
privacylens/orchestrator.py      464
confaide/orchestrator.py         445  "Copied from goldcoin_hipaa/..."
cirl_vignettes/orchestrator.py   479  "Copied from goldcoin_hipaa/..."
mmlu/orchestrator.py             445  "Copied from simpleqa_verified/..."
simpleqa_verified/orchestrator.py 441  "Copied from goldcoin_hipaa/..."
```

Pairwise diffs show only **116–156 differing lines** out of ~450–500. The
differences reduce to three things:

1. The dagspace name string (passed to `pipeline_run_id`, `build_run_config`,
   the output subdir name).
2. The `_log_eval_metrics()` function — per-benchmark metric formatting
   (confusion matrix for goldcoin, Pearson-r for confaide, leak-rate for
   privacylens, etc.).
3. The local `wandb_logger` import.

Everything else — `run_experiment()`, `_serialize_context_data()`,
`execute_stage_job()`, `_get_wandb_logger()`, `_resolve_hydra_output_dir()` —
is byte-identical or trivially renamed. `historical_norms/orchestrator.py`
even re-wraps the shared `_inject_prompt_from_file` / `_load_launcher_config`
just to bind `_CONF_DIR` — exactly the parameter a hooks dataclass should carry.

**Why it matters (Jane Street).** This is the canonical "wrong abstraction was
never extracted, so the code was forked" failure. A bug fix to the SLURM
waiting logic (which has already happened multiple times — see the dated
comments about NFS result-pickle races) must be applied to 9 files by hand, and
they have already drifted (goldcoin imports `clone_config`, confaide does not).
Scientific-computing-wise, drift between "identical" eval harnesses is a
silent reproducibility hazard: two benchmarks can report numbers produced by
subtly different orchestration code.

**Proposed fix.** Extract a single generic orchestrator into
`dagspaces/common/orchestrator.py` (which already holds the shared helpers):

```python
# dagspaces/common/run_experiment.py  (new)
@dataclass(frozen=True)
class DagspaceHooks:
    dagspace_name: str                 # "goldcoin_hipaa"
    wandb_dagspace: str                # "goldcoin"
    output_subdir: str                 # subdir under hydra output dir
    log_eval_metrics: Callable[[Logger, dict, str], None]  # the ONLY per-bench code
    make_wandb_logger: Callable[..., Any]

def run_experiment(cfg: DictConfig, hooks: DagspaceHooks) -> None: ...
```

Each dagspace's `orchestrator.py` shrinks to ~40 lines: define its
`_log_eval_metrics`, build a `DagspaceHooks`, call `run_experiment`. The
`execute_stage_job` / serialization / SLURM-wait logic lives once.

This is the highest-leverage change in the repo: it deletes ~2000 lines and
makes the eval harnesses provably identical.

---

## Finding 2 — SLURM/NFS result-waiting block duplicated verbatim ✅ DONE (2026-07-19)

> **Status: implemented as part of Finding 1 (PR #4).** The block was extracted
> verbatim into `await_slurm_result(job, cfg, node_key) -> StageResult`
> (`dagspaces/common/orchestrator.py`), called once by the generic
> `run_experiment`. All nine eval dagspaces now use it; none retain a local
> copy (`grep result_pickle dagspaces/*/orchestrator.py` → only `common`,
> `grpo_training`, `historical_norms`). Covered by
> `tests/common/test_run_experiment.py::TestAwaitSlurmResult` (happy path,
> tuple unpack ok/error, pickle recovery, squeue fallback, all-fail re-raise).
>
> **Remaining (out of scope):** `grpo_training` and `historical_norms` each
> still carry their *own* result-wait block, but those are **different
> algorithms** — not verbatim copies of the eval block nor of each other — in
> the two dagspace whose consolidation was declined (Phase 2). So the
> "duplicated verbatim" problem this finding names is resolved; unifying the
> training waits would be a behavior change on the frozen training pipeline and
> was deliberately not pursued.

---

## Finding 3 — Runner boilerplate 🟡 (M)

**Evidence.** Every stage runner follows the same shape (see
`goldcoin_hipaa/runners/eval_stages.py`, and the near-identical
`confaide`, `mmlu`, `simpleqa_verified` copies — 30 diff lines out of 138):

```python
class LLMInferenceRunner(StageRunner):
    stage_name = "llm_inference"
    def run(self, context):
        from ..stages.llm_inference import run_llm_inference
        df = pd.read_parquet(context.inputs["dataset"])
        result_df = run_llm_inference(df, context.cfg)
        out_path = context.output_paths["dataset"]
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        result_df.to_parquet(out_path, index=False)
        return StageResult(outputs={"dataset": out_path}, metadata={"rows": len(result_df)})
```

The read-parquet / write-parquet / makedirs / `StageResult` ceremony is
repeated in essentially every runner across every dagspace.

**Why it matters.** Boilerplate invites inconsistency (some runners add sanity
checks, some don't; some write `metrics.json`, some don't) and makes the
"what does this stage actually do" signal (one function call) hard to see.

**Proposed fix.** Provide a small declarative base in
`dagspaces/common/runners/base.py`:

```python
class DataFrameStage(StageRunner):
    """read input parquet -> self.transform(df, cfg) -> write output parquet."""
    input_key = "dataset"
    output_key = "dataset"
    def transform(self, df: pd.DataFrame, cfg) -> tuple[pd.DataFrame, dict]: ...
    def run(self, context):  # provided once, correctly
        ...
```

Runners then declare `transform` only. Stages that need bespoke I/O (metrics
JSON, multiple outputs) keep the explicit form — do **not** force-fit them
(Jane Street: prefer a little duplication to a leaky abstraction).

---

## Finding 4 — Deprecated dagspaces still in the tree 🟡 (S)

**Evidence.** `dagspaces/.uair/` and `dagspaces/.rule_tuples/` are
dot-prefixed and documented (CLAUDE.md, wiki) as deprecated / not used for
COLM. `.uair` alone is ~12k lines (`topic.py` 1587, `classify.py` 1406,
`verification_core.py` 857, plus 10+ stage files). They inflate the codebase
by ~20% and show up in every `find`/`wc`/grep, and in the test-import surface.

**Why it matters.** Dead code is a tax on every future reader and on tooling.
The dot-prefix hides them from `python -m dagspaces.<TAB>` but not from
reviewers or from `pytest` collection if anything imports them.

**Caveat — one live coupling.** `common/stage_utils.py:maybe_silence_vllm_logs`
does `from dagspaces.uair.logging_filters import PatternModuloFilter`. So
`common/` (shared, active) depends on a *deprecated* dagspace — itself a smell
(the dependency arrow points the wrong way). Before deleting `.uair`, move
`PatternModuloFilter` (a ~30-line logging filter) into `common/` and update the
import. The import is wrapped in `except Exception: pass`, so today a deletion
would silently disable vLLM log-throttling rather than crash — another instance
of Finding 5 masking a real dependency.

**Proposed fix.** (1) Relocate `PatternModuloFilter` → `common/logging_filters.py`.
(2) Confirm nothing else in the active COLM path imports `.uair`/`.rule_tuples`
(`grep -rn "dagspaces.uair\|rule_tuples" dagspaces tests scripts`). (3) Move the
rest to a git tag / branch (`archive/uair`) and delete from `main`, or under a
clearly-named `archive/` dir excluded from packaging and test discovery if an
older result needs it for reproducibility.

---

## Finding 5 — Pervasive silent `except Exception: pass` 🔴 (M)

**Evidence.** The shared `common/orchestrator.py` alone contains dozens of
bare `except Exception: pass` / `except Exception: return None` blocks, e.g.
in `build_run_config` (5 separate try/excepts), `_resolve_pipeline_name`,
`_resolve_eval_task`, `_resolve_checkpoint_name`, `_inject_prompt_from_file`,
`_clean_df_for_parquet`, `_log_gpu_environment`. Example:

```python
try:
    grpo_cfg = OmegaConf.select(cfg, "training.grpo")
    if grpo_cfg is not None:
        run_config["grpo"] = {...}
except Exception:
    pass
```

**Why it matters.** For scientific code this is the most dangerous pattern in
the repo: a typo'd config key, a schema change, or a serialization bug is
silently swallowed, and the run "succeeds" while logging the wrong/empty
metadata to W&B. The 2026-06-09 code-review changelog already documents real
bugs of exactly this class (λ defaulting wrong, stale quality flags). Silent
catches hide the next one.

**Proposed fix.**
- Distinguish *expected-absent* (use `OmegaConf.select(..., default=None)` and
  an `if`, no try/except) from *genuinely-can't-fail* (delete the guard).
- Where a catch must stay, log at `warning` with the exception, never bare
  `pass`.
- Add a lint rule (ruff `BLE001` / `S110`) to CI so new bare-excepts are
  flagged.

---

## Finding 6 — `CompositeRewardFunction`: nested, not composed 🟡 (L, in flight)

**Context (from `wiki/grpo_redesign/README.md`).** The team has *already
diagnosed this* and is building a parallel modular stack. Recording it here so
the refactoring doc is complete and so the keeper-freeze constraint is
respected.

**Evidence (`dagspaces/grpo_training/stages/rewards.py`, 1220 lines).**
`CompositeRewardFunction` exhibits exactly the anti-patterns the redesign doc
names ("mechanisms are *nested*, not composed"):

- A **hard-coded 6-weight vector** guarded by `if len(weights) != 6: raise`,
  with component identity recovered from positional index tuples:
  `gate_idx = (0, 1, 2, 4)` / `content_idx = (3, 5)`.
- **Three composition modes** (`additive` / `gated` / `directional`) selected
  by a string flag, each a different formula over the same index tuples.
- **Cross-component coupling**: the appropriateness *direction* is a multiplier
  folded into `r_ground` *upstream* (`app_mode="multiplicative"`), and the
  contrastive penalty is a clamp *inside* `r_ground` — so "remove direction"
  is not a local edit.
- **Sentinel-based control flow** in `__call__`: judgment-vignette rows push
  `None` into `partial_components` and "bypass Phases 3–4"; no-flow rows route
  through a separate `no_flow_reward` / abstention-penalty path; `task_type`
  string dispatch splits the loop body.
- A pile of orthogonal behavior flags on one constructor:
  `no_flow_scoring`, `judgment_weights`, `composition`, `abstention_penalty`,
  `confidence_fallthrough`, `online_rground`.

**Why it matters.** As the redesign doc states, an honest take-one-out
ablation — the main outstanding camera-ready experiment — is "nearly
impossible" here because removing any mechanism changes another's formula.
This is the opposite of the Jane Street ideal where "removal is the identity
operation."

**Proposed fix (already designed).** The `grpo_redesign/` stack: one module =
one sentence; additive modules delete + renormalize; task modules go to mix 0;
verifiable core with judged auxiliaries only. **Constraint:** the v9-ckpt100
keeper path (`reward_composition: directional` + `online_rground_external.yaml`)
must stay byte-frozen for existing paper results — the redesign is a parallel
stack, not an in-place edit. Post-camera-ready, the legacy `rewards.py`
composition machinery is deleted (see `grpo_redesign/migration.md`).

**Action for this review:** none beyond tracking — do not "clean up"
`rewards.py` in place; it is deliberately frozen. Ensure new work lands in the
modular stack and that a keeper-freeze regression guard exists (migration.md
test plan).

---

## Finding 7 — `vllm_inference.py` is a 2098-line god module 🟡 (M)

**Evidence.** `dagspaces/common/vllm_inference.py` (2098 lines, the 2nd-largest
file) bundles at least nine unrelated responsibilities:

1. LoRA adapter key-remapping for VLMs (`_remap_lora_keys_for_vlm`)
2. Reasoning/`<think>` splitting + harmony-model channel splitting
   (`_split_reasoning`, `_split_harmony`, `_is_harmony_model`,
   `_detect_reasoning_parser`)
3. GPU detection + type sniffing via `nvidia-smi` (`detect_num_gpus`,
   `detect_gpu_type`, `_run_nvidia_smi`)
4. NCCL / PCIe / runtime env-var construction (`get_pcie_nccl_env_vars`,
   `get_vllm_runtime_env_vars`, `apply_gpu_aware_settings`)
5. Engine-kwarg building + version filtering (`_build_engine_kwargs`,
   `filter_vllm_engine_kwargs`)
6. OpenAI-compatible **server** client (`_run_server_inference`)
7. Streaming-shard write + resume (`_resolve_streaming_dir`, shard recovery)
8. A **subprocess entrypoint** `main()` (227 lines) for data-parallel workers,
   plus `_run_data_parallel`
9. A native **transformers** fallback (`_run_transformers_text_inference`)
   …and the actual `run_vllm_inference` (itself ~500 lines).

`run_vllm_inference` is a single ~500-line function mixing server routing,
transformers fallback, env setup, LoRA, tokenizer loading, preprocessing,
prompt-length validation/clamping, DP dispatch, and result assembly.

**Why it matters.** A module this size with this many axes is untestable in
units and hard to reason about; the reasoning-parser/harmony logic in
particular is correctness-critical (the 2026-07-13 changelog documents gpt-oss
returning its hidden `analysis` channel as its answer — a reasoning-split bug).
That logic deserves its own module with focused tests.

**Proposed fix.** Split along the responsibility seams (each is already a
coherent function cluster):

```
common/inference/
  reasoning.py     # _split_reasoning, harmony, parser detection  (+ unit tests)
  gpu_env.py       # detect_num_gpus/type, NCCL/PCIe env, apply_gpu_aware_settings
  engine.py        # _build_engine_kwargs, filter_vllm_engine_kwargs, LoRA remap
  server_client.py # _run_server_inference
  dp_worker.py     # main(), _run_data_parallel
  transformers_fb.py # _run_transformers_text_inference
  core.py          # run_vllm_inference (orchestrates the above)
```

Re-export `run_vllm_inference` from `common/vllm_inference.py` for
backward-compat. The reasoning-split functions are the highest-value extraction
because they are pure and unit-testable.


---

## Finding 8 — JSON-from-LLM extraction re-implemented ≥3 ways 🔴 (S–M)

**Evidence.** "Extract a JSON object from LLM text that may be wrapped in
prose / `<think>` blocks" is a common, error-prone task with **at least three
divergent implementations** in the active tree:

| Location | Strategy |
|---|---|
| `common/stage_utils.py:extract_last_json` | regex `findall(r"\{[\s\S]*\}")`, try each **from the last backwards** |
| `grpo_training/stages/rewards.py:_parse_completion` | `re.search(r"\{[\s\S]*\}")` — **first / greedy** match, then schema-normalise |
| `historical_norms/stages/_utils.py:extract_json` | **outermost** `{`…`}` via `find`/`rfind`, + `json_repair` fallback |

Plus narrower cousins: `privacylens/prompts.py:_parse_json_like_payload`,
`privacylens/stages/parse_responses.py:_extract_yes_no_json`,
`goldcoin_hipaa/stages/parse_responses.py:_try_json_classification`,
`mmlu/stages/parse_responses.py:_try_json_answer`,
`historical_norms/stages/ci_extraction.py:_parse_reasoning_json`, and
`toolemu/utils/tool.py:get_first_json_object_str`.

**Why it matters.** "Last", "first/greedy", and "outermost" return *different
objects* for the same model output (e.g. a completion that echoes the schema
example before emitting its answer). When two benchmarks parse the same model
generation with different extractors, parse-rate and accuracy numbers are not
comparable — a quiet reproducibility hazard, and exactly the kind of near-dup
the Jane Street style forbids once the right abstraction is known.

**Proposed fix.** Consolidate on one canonical, well-tested extractor in
`common/stage_utils.py` (keep the `json_repair` fallback from the
historical_norms version; make last-vs-first a named parameter). Have
`_parse_completion` and the per-benchmark parsers *delegate* to it for the
extraction step and keep only their schema-specific normalisation. Add a
property test pinning behavior on adversarial inputs (echoed schema, multiple
objects, trailing prose).

---

## Finding 9 — Dead docs / scratch dirs at repo root 🟢 (S)

**Evidence.** `old/` holds 7 superseded `PRIVACYLENS_*.md` design docs;
`debug/`, `prompt_dev/`, and a 218 MB `notebooks/` live at the top level
alongside ~10 stray `*.launch.log` / `*.html` inspector artifacts
(`sft.html`, `inspection.html`, `privacylens_audit_n100_seed777.html`, …).

**Why it matters.** Low severity, but root clutter obscures the real layout
and large binary-ish artifacts bloat clones.

**Proposed fix.** Move `old/` → `wiki/archive/` (or delete if wiki supersedes
it); move scratch HTML/logs to `outputs/` or `.gitignore` them; confirm
`notebooks/` is meant to be checked in (218 MB) or LFS/ignore it.

---

## Finding 10 — The duplicated orchestration is untested 🟡 (M)

**Evidence.** `tests/` has strong coverage of reward components, prompts,
parse-health, batch export, etc., but **no test exercises the SLURM/NFS
result-waiting block** (Finding 2) or the generic run loop (Finding 1). The
eval dagspaces (`goldcoin_hipaa`, `vlm_geoprivacy_bench`, `mmlu`,
`simpleqa_verified`) have no dedicated test directory — only
`tests/integration/test_compute_metrics_all_benchmarks.py` touches their
metrics stage.

**Why it matters.** The most-duplicated, most-bug-prone code (the part with
all the dated NFS war-story comments) is the least tested. Extracting it
(Findings 1–2) is the natural moment to add a focused test with a fake
`job` object simulating the result-pickle race.

**Proposed fix.** As part of Finding 1, add `tests/common/test_run_experiment.py`
covering: local execution path, SLURM path with a stubbed executor, the
result-pickle-recovery fallback, and the `(outcome, payload)` unpacking. This
both de-risks the refactor and locks the behavior the 9 copies have drifted
around.

---

## Things that are *fine* (do not refactor)

Recorded so the review is balanced and nobody "fixes" these by mistake:

- **`cli.py` files** (17–21 lines, near-identical). Tiny, stable entry points;
  abstracting them buys nothing. Leave as-is.
- **`wandb_logger.py` shims** per dagspace. Genuine thin wrappers supplying
  per-dagspace defaults over a shared base — the right pattern.
- **`common/eval_sanity.py`**. Clear classes (`SanityReport`,
  `compute_parse_health/format_health/judge_health`), focused functions,
  well-tested. A model for what the orchestrator extraction should look like.
- **Per-benchmark `prompts.py` and `stages/llm_inference.py` preprocess/
  postprocess**. Genuinely benchmark-specific logic; duplication here would be
  the *wrong* abstraction. (Exception: the "bump `max_tokens` when stripping
  `<think>`" snippet in each `llm_inference.py` is copy-pasted and could become
  one helper — minor.)
- **The `grpo_redesign/` modular stack.** This *is* the refactor of Finding 6;
  support it, don't duplicate it.

---

## Prioritized roadmap

Ordered by leverage (lines deleted / correctness risk per unit effort):

1. ~~**Finding 2 → 1: extract the generic orchestrator + SLURM-wait helper**~~
   ✅ **DONE (2026-07-19, PRs #4/#6).** Nine eval dagspaces on one shared loop;
   `await_slurm_result` extracted; golden-params + run-loop tests added.
   Phase 2 (training orchestrators) declined — see Finding 1 status.
2. **Finding 8: consolidate JSON extraction** (🔴, S–M). Small, high
   correctness value for cross-benchmark comparability.
3. **Finding 5: retire silent `except Exception: pass`** (🔴, M). Add ruff
   `BLE001`/`S110` to CI; audit the ~168 swallowing catches in active code,
   starting with `build_run_config` and the W&B-metadata path (where silent
   drops corrupt experiment records).
4. **Finding 4: archive `.uair` / `.rule_tuples`** (🟡, S). Quick ~20%
   codebase reduction; verify no COLM-path imports first.
5. **Finding 7: split `vllm_inference.py`** (🟡, M). Start with the pure,
   testable `reasoning.py` extraction (harmony + `<think>` splitting) since
   that logic is correctness-critical and currently under-tested.
6. **Finding 3: declarative `DataFrameStage` base** (🟡, M). Do *after*
   Finding 1 settles, so the runner contract is stable.
7. **Finding 9: root cleanup** (🟢, S). Any time.
8. **Finding 6: tracked only** — lands via `grpo_redesign/`; keep keeper frozen.

### Suggested sequencing note
~~Findings 1, 2, 3, 10 are one coherent program ("make the eval harness a single
tested thing")~~ — **Findings 1, 2, 10 are done**; Finding 3 (`DataFrameStage`)
remains and is now safe to attempt since the runner↔orchestrator contract is
stable. Findings 5 and
8 are independent and can proceed in parallel. Finding 7 is independent but
touches hot inference code — do it behind the existing harmony/reasoning tests
and add new ones first.
