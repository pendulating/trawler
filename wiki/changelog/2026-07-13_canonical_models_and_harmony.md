# 2026-07-13 — Canonical model set, gpt-oss harmony fix, vLLM 0.25 upgrade

**Status: DONE and verified on GPU.** Remaining: apply the `pyproject.toml` change and
cut `.venv` over once the top100 runs drain.

Session context: overhauling the task-LLM set away from previous-generation
checkpoints, ahead of replicating the SFT results on the newly regenerated fiction10
flows. Reference page: [../canonical-models.md](../canonical-models.md).

---

## Remaining work

### 1. Drop RAPIDS from `pyproject.toml` (decided, not yet applied)

vLLM 0.25 pulls `nvidia-cutlass-dsl` → `cuda-python>=12.8`, which cannot co-resolve with
`cudf-cu12` / `cuml-cu12` / `dask-cuda` / `raft-dask-cu12`. Those are imported in
**exactly one file** — `dagspaces/.uair/stages/topic.py`, in the deprecated `.uair`
dagspace — behind lazy optional-GPU imports with CPU fallbacks. No active pipeline
touches them.

Lines to remove: `cudf-cu12`, `cuml-cu12`, `dask-cuda`, `dask-cudf-cu12`,
`libcudf-cu12`, `libcuml-cu12`, `pylibcudf-cu12`, `raft-dask-cu12`,
`rapids-dask-dependency`, `rapids-logger`. Also drop `torchcodec` (audio/video only;
its ≥0.14 requirement has no CUDA-12 build and vLLM runs text fine without it).

### 2. Cut `.venv` over to vLLM 0.25.0+cu129 — after top100 drains

Stages import vLLM fresh from the venv when SLURM starts them, so an in-place upgrade
can break an in-flight run and split a corpus across two engine versions. The validated
environment is `.venv-vllm025cu129/`.

---

## ✅ vLLM 0.25.0 works here — take the `+cu129` wheel

**Verified end-to-end on klara, 2026-07-13.** The install recipe and the two dependency
landmines are in [../canonical-models.md](../canonical-models.md#engine-requirements).

The trap that nearly cost us the upgrade: **PyPI's default vLLM wheel is a CUDA 13
build** (`nvidia-cutlass-dsl[cu13]`, torch 2.11 cu13), and klara's driver is
**570.124.06, capped at CUDA 12.8**. Reading PyPI metadata alone makes 0.25 look
impossible — `import torch` dies with `undefined symbol: ncclDevCommDestroy`, and the
newest CUDA-12 release on PyPI looks like 0.21.0 (the cu13 switch landed in 0.22.0).

**That conclusion was wrong.** vLLM also ships a **`+cu129`** wheel on its GitHub
releases, and CUDA minor-version compatibility runs it happily on a 12.8 driver.
Verified: `torch.cuda.is_available()` True, GPU matmul OK, and —

- **`Gemma4UnifiedForConditionalGeneration` is registered** (`False` on 0.19.1). The
  12B **loads and generates**: *"No, this is not an appropriate information flow because
  it violates patient confidentiality and HIPAA regulations…"*. The one hard blocker in
  the canonical set is cleared.

> **Lesson:** do not conclude a package is unavailable from PyPI metadata alone — check
> the project's own release assets / wheel index for CUDA-variant builds.

## ✅ gpt-oss harmony fix verified on real model output

The unit tests used synthetic fixtures. Driving the real weights through the actual
production path (`run_vllm_inference`, `.bench/smoke_gptoss_harmony.py`):

```
[harmony_smoke] harmony model detected — skip_special_tokens=False

row 0  nurse -> employer, HIV diagnosis
  generated_text      : '1'                                  <- clean, parseable
  generated_reasoning : "We need to decide rating 1-5..."    <- 201 chars, separated
row 1  doctor -> treating specialist
  generated_text      : '5'
  generated_reasoning : 587 chars, separated

PASS: final channel isolated; no analysis leakage; reasoning captured.
```

The answers are also correct CI judgements. Before the fix, `generated_text` was the
entire chain-of-thought.

---

## ⚠️ The overnight chain died — and the resubmission exposed a worse bug

`scripts/run_overnight_gemma4_chain.sh` was launched inside SLURM session job **845966**,
which **timed out at 09:52** and took the driver with it. The already-submitted reasoning
job survived; nothing after it was ever submitted.

| stage | state |
|---|---|
| fiction10_flows | ✅ DONE (validated — see the marimo notebook) |
| top100_norms · reasoning | ✅ DONE — `reasoning.parquet`, 15,875 / 15,875 rows |
| top100_norms · extraction | ▶️ **resubmitted 2026-07-13**, running |
| top100_flows · both stages | ▶️ **resubmitted 2026-07-13**, queued |

Resumed via a new extraction-only pipeline,
**`COLM_norms_extraction_from_reasoning_gemma4`** — the orchestrator has **no node-level
resume**, so re-running the full pipeline would have redone ~12h of reasoning. Drivers are
now themselves `sbatch` jobs (`scripts/run_resume_top100_norms_extraction_gemma4.sh`,
`scripts/run_top100_flows_gemma4_sbatch.sh`) so they cannot die with a session.

### 🔥 The `srun` trap — submitit silently ran a 31B model on a CPU node

Submitting the drivers via `sbatch` surfaced a **silent, catastrophic** failure mode.
Full write-up in [../slurm-and-env.md](../slurm-and-env.md#running-a-driver-under-sbatch-the-srun-trap).

submitit's SLURM detection is literally:

```python
def affinity(cls) -> int:
    return -1 if shutil.which("srun") is None else 2
```

The SLURM clients on this cluster are ssh-forwarding shims in `~/.local/bin` —
`sacct`/`sbatch`/`scancel`/`sinfo`/`squeue` — and **there is no `srun` shim**. A login
shell also carries `/usr/local/slurm/current/bin` (a real `srun`), so interactive runs
work; an `sbatch` job gets a minimal PATH, loses it, and `AutoExecutor` **falls back to
its LOCAL executor with no error**, running the GPU stage as a subprocess on the driver's
CPU node. The only symptom was `CUDA_VISIBLE_DEVICES=''` buried in a stage log.

Fixes applied:
- Driver scripts set `PATH="$HOME/.local/bin:$PATH:/usr/local/slurm/current/bin"`
  (shims first so `sbatch`/`squeue` resolve to the working ones; the native dir last so
  `srun` merely *exists* — `slurm_use_srun=False`, so it is never executed) plus an
  explicit preflight that refuses to start without it.
- **`_create_submitit_executor()` now raises** if `srun` is absent rather than degrading
  to CPU. Do not remove that check.

Tell-tale for diagnosing it after the fact: a real submitit SLURM job leaves a
`*_submission.sh` in `.slurm_jobs/<node>/` and its id appears in `sacct`. A local-fallback
job leaves neither — the "job id" is a PID.

### And one self-inflicted bug the tests could not catch

The harmony hook was written as `_is_harmony_model(model_source)`, but the name bound in
`run_vllm_inference`'s scope is **`_model_source`** — the bare one exists only inside the
nested `if lora_path:` branch. That is an `UnboundLocalError` **for every model**, from a
gpt-oss-only feature, and it took down both top100 runs on first submit.

Neither the unit suite (never executes `run_vllm_inference` — needs a GPU) nor pyflakes
(`model_source` *is* assigned in the function, just conditionally, so it is a legal local)
catches it. Locked down by
`tests/common/test_harmony_parsing.py::TestRunVllmInferenceScope`.

---

## DONE — canonical model set

Thirteen models, all configs present, all weights in the zoo. Full rationale in
[../canonical-models.md](../canonical-models.md).

New this session:

- **`dagspaces/common/conf/model/gemma-4-12b/{base,instruct}.yaml`** — weights were
  already in the zoo (23 GB) but there was **no config at all**, so the model was
  unusable from any pipeline. ⚠️ Cannot actually be served yet — see “Resume here”.
- **`dagspaces/common/conf/model/harc-llama3.1-8b/instruct.yaml`** + downloaded
  `HARC-Llama-3.1-8B-Instruct` (15 GB, `LlamaForCausalLM`).

**Why HARC matters:** it is the HARC safety LoRA merged into
`meta-llama/Llama-3.1-8B-Instruct` — same base weights, tokenizer, and chat template as
`llama3.1-8b/instruct`, differing *only* in safety alignment. That makes it the one
clean instrument for “does refusal-style safety tuning help or hurt CI reasoning?”, with
no scale/family/pretraining confound. **Score its refusals separately** — a refusal is an
*unparseable* answer, not a wrong one, and pooling the two lets a refusal-happy model
look safe by being silent.

---

## DONE — gpt-oss was reading its own scratchpad as its answer

**Every gpt-oss number produced before today is suspect.** gpt-oss speaks
[harmony](https://github.com/openai/harmony), a channel protocol — `analysis` is hidden
CoT, `commentary` is tool traffic, only `final` is the answer. Two compounding bugs meant
the `analysis` channel was returned as `content`:

1. `_detect_reasoning_parser` returned `"gptoss"`; vLLM registers it as
   **`openai_gptoss`** → `KeyError` → swallowed by `except Exception: pass` → fell
   through to the `<think>` regex, which harmony never emits.

   Renaming it does **not** fix this: vLLM's `openai_gptoss` parser raises
   `NotImplementedError` on non-streaming input (*"gpt-oss has a special branch for
   parsing reasoning in non-streaming mode"*). vLLM only handles harmony in its
   **OpenAI-server** path; this repo calls `LLM.generate()` **offline**, everywhere.
   **There is no vLLM parser to use.**

2. `skip_special_tokens=True` (vLLM's default) **deletes the delimiters**, collapsing
   output to the unsplittable `"analysisHidden CoT…assistantfinalAnswer: 3"`. Once the
   markers are gone, no parser can recover the final channel.

**Fixed** in `dagspaces/common/vllm_inference.py`:

- `_is_harmony_model()` + **`_split_harmony()`** — a direct channel parser.
  `_detect_reasoning_parser()` now deliberately returns `None` for gpt-oss.
- `run_vllm_inference()` **forces `skip_special_tokens=False`** for harmony models, in
  code rather than in the model YAML — a stage overriding `sampling_params` would
  otherwise silently drop it.
- A truncated generation that never reaches `final` yields `content == ""` **on
  purpose**. An empty answer is caught by the format-adherence gate; hidden CoT graded
  as an answer is not.

**The interaction that bites (a test caught this).** Removing gpt-oss from
`_detect_reasoning_parser` also silently disabled the **reasoning-budget trigger** —
gpt-oss's config has a bare `chat_template_kwargs: {}`, so the `enable_thinking` trigger
cannot fire for it, making the parser check its *only* trigger. Short-answer stages
default to `max_tokens` 16–64; gpt-oss always reasons first, so without the bump it
never reaches `final`. Combined with the honest-truncation fix, that would have produced
**empty answers on every short-answer benchmark** — quieter and worse than the original
bug. `model_needs_reasoning_budget()` now checks `_is_harmony_model()` separately. **Do
not "simplify" that back into a single parser lookup.**

Tests: `tests/common/test_harmony_parsing.py` (12 new) and the two pre-existing
`tests/common/test_reasoning_budget.py` assertions that caught the regression.
Full suite: **630 passed**.

---

## DONE — QA gates (from earlier the same day)

- **`flow_quality_passed` deleted.** It enforced a role requirement neither CI prompt
  ever stated — the 2026-06-09 changelog justified it by citing `ci_schema.py:188`,
  which is inside **`RazNormTuple`** (the *norms* schema). It flagged 37.6% of fiction10
  flows for doing exactly what the prompt asked. See
  [2026-06-09_ner_quality_checks.md](2026-06-09_ner_quality_checks.md) (retraction filed).
- **`norm_quality_passed` blocklist fixed.** `may` (May Welland) and `will` (Will
  Ladislaw) were matched case-insensitively across all ten books, so every modal verb
  tripped the gate. `name_detection.AMBIGUOUS_NAMES` now matches those case-sensitively:
  **440 → 67 flagged, 4.39% → 0.67%**, verified on the real fiction10 parquet.

Neither gate ever dropped a row — no corpus was corrupted.

---

## Next, once the above clears

Replicate the paper's **SFT results on the regenerated fiction10 flows**
(`outputs/2026-07-12_fiction10_flows_gemma4/23-14-17`, 16,200 CI tuples — the first
flows corpus produced with the *fiction* prompts actually applied). That was the
original goal; the model-set overhaul is the prerequisite.
