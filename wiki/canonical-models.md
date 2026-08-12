# Canonical Model Set

**Established 2026-07-13.** This is the task-LLM set for all COLM evaluation going
forward. It replaces the accreted set of previous-generation checkpoints
(`qwen3-8b`, `qwen2.5-*`, `gemma-3-12b`, `openthinker-7b`, `llama3.3-70b`, …) that
the repo had drifted into using.

Config conventions and the YAML schema live in [models.md](models.md); this page
says **which** models and **why**. If you are adding one, read
[howto/add-model.md](howto/add-model.md).

---

## The set

Thirteen models. Every entry has a config under
`dagspaces/common/conf/model/` and weights in `/share/pierson/matt/zoo/models/`.
Verified present 2026-07-13.

### Scale ladders — for scaling curves within a fixed family

The point of a ladder is that everything except parameter count is held constant
(pretraining data, tokenizer, chat template, post-training recipe). A CI result
that holds across a ladder is a result about *capability*; one that appears only
at the top is a result about *that checkpoint*.

| config | zoo dir | arch | ctx | disk |
|---|---|---|---|---|
| `gemma-4-e2b/instruct` | `Gemma-4-E2B-it` | `Gemma4ForConditionalGeneration` | 131k | 9.6 GB |
| `gemma-4-e4b/instruct` | `Gemma-4-E4B-it` | `Gemma4ForConditionalGeneration` | 131k | 15 GB |
| `gemma-4-12b/instruct` | `gemma-4-12B-it` | `Gemma4Unified…` ⚠️ | 262k | 23 GB |
| `qwen3.5-2b/instruct` | `Qwen3.5-2B` | `Qwen3_5ForConditionalGeneration` | 262k | 4.3 GB |
| `qwen3.5-4b/instruct` | `Qwen3.5-4B` | `Qwen3_5ForConditionalGeneration` | 262k | 8.8 GB |
| `qwen3.5-9b/instruct` | `Qwen3.5-9B` | `Qwen3_5ForConditionalGeneration` | 262k | 19 GB |

**Qwen3.5-9B is the training subject** — the SFT and GRPO work
([grpo_redesign/](grpo_redesign/README.md)) is done on this model. The camera-ready
models are the **m2 `full` GRPO cell** and the **k3 `verdict` KTO arm**
(both 2026-08-05; see [2026-07-31_kto_plan.md](2026-07-31_kto_plan.md) §19).
`qwen3.5-9b/v9-ckpt100` is **deprecated** and no longer the keeper. The 2B and 4B rungs exist so "does normative-simulacra
training help?" can be asked at three scales rather than asserted at one.

**Gemma-4 is the gold-label family**, not a training target: `gemma-4-31b/instruct`
generates the norm and flow corpora. E2B/E4B/12B are here as *evaluation* subjects,
and they give a second, independent ladder — if a CI finding replicates across both
Qwen3.5 and Gemma-4, it is not a family artifact.

### Reasoning baselines

| config | zoo dir | base | why it is here |
|---|---|---|---|
| `openthinker3-7b/instruct` | `OpenThinker3-7B` | Qwen2.5-7B | Strong open reasoning model. Tests whether *general* reasoning ability alone buys CI competence, or whether the normative grounding is doing distinct work. |

### CI-specific baselines — the ones we must beat

| config | zoo dir | base | why it is here |
|---|---|---|---|
| `cirl/base` | `CIRL` | Qwen2.5 | Prior work's CI-tuned model. Direct method comparison. |
| `context-reasoner/ppo` | `context-reasoner-ppo_open_thinker_acc_reward` | Qwen2.5 | ContextReasoner (HKUST) PPO baseline — see [benchmarks/contextreasoner.md](benchmarks/contextreasoner.md). The closest published analogue to our GRPO approach; the honest comparison is against this, not against a base model. |

### The safety-alignment matched pair

| config | zoo dir | arch | disk |
|---|---|---|---|
| `llama3.1-8b/instruct` | `Llama-3.1-8B-Instruct` | `LlamaForCausalLM` | 30 GB |
| `harc-llama3.1-8b/instruct` | `HARC-Llama-3.1-8B-Instruct` | `LlamaForCausalLM` | 15 GB |

This is the most experimentally valuable pair in the set. HARC (Microsoft,
[arXiv:2607.00572](https://arxiv.org/abs/2607.00572) — *"Coupling Harmfulness and
Refusal Directions for Robust Safety Alignment"*) is the **HARC safety LoRA merged
into `meta-llama/Llama-3.1-8B-Instruct`**. Same base weights, same tokenizer, same
chat template, differing *only* in safety alignment.

So any CI delta between the two is attributable to the alignment and nothing else —
no scale, family, or pretraining confound. It is the one clean instrument we have for
**"does refusal-style safety tuning help or hurt contextual-integrity reasoning?"** —
a question the paper is otherwise only able to gesture at.

> ⚠️ **Score its refusals, don't pool them.** A model that *declines* to answer a
> privacy vignette has not reasoned correctly about it. A refusal is an
> **unparseable** answer, not a wrong one, and the two must never be averaged
> together — pooling them would let a refusal-happy model look "safe" by being
> silent. The format-adherence FAIL gate in [metric-trust.md](metric-trust.md)
> matters more here than anywhere else in the set. Expect HARC's parseable-rate to
> be the number to watch.

### General instruction-following

| config | zoo dir | arch | note |
|---|---|---|---|
| `phi-4/instruct` | `Phi-4` | `Phi3ForCausalLM` | Non-thinking. Sets `force_answer_format: true`. 16k ctx — the shortest in the set; watch for prompt clamping on long vignettes. |

### Frontier open-weights

| config | zoo dir | arch | note |
|---|---|---|---|
| `gpt-oss-20b/instruct` | `GPT-OSS-20B` | `GptOssForCausalLM` | **Speaks harmony — read the next section before using it.** |

---

## GPT-OSS and the harmony response format

**gpt-oss does not emit `<think>` blocks. It speaks
[harmony](https://github.com/openai/harmony), and getting this wrong silently
corrupts every number the model produces.**

Harmony is a *channel protocol*, not a reasoning wrapper. The assistant emits one or
more segments:

```
<|channel|>analysis<|message|>hidden chain-of-thought…<|end|>
<|start|>assistant<|channel|>commentary to=browser.search<|message|>tool call<|call|>
<|start|>assistant<|channel|>final<|message|>Answer: 3<|return|>
```

Three channels: **`analysis`** is hidden CoT, **`commentary`** is tool traffic, and
only **`final`** is the answer. A consumer that does not split on the channel markers
is reading the model's scratchpad as its response.

### What was broken (fixed 2026-07-13)

Two compounding bugs meant **every gpt-oss run before this date graded the model's
hidden reasoning as its answer**:

1. **`_detect_reasoning_parser` returned `"gptoss"`.** vLLM registers that parser as
   **`openai_gptoss`**, so the lookup raised `KeyError`, the caller's
   `except Exception: pass` swallowed it, and execution fell through to the `<think>`
   regex — which harmony never emits. Result: the entire `analysis` channel was
   returned as `content`, and `reasoning` came back empty.

   Fixing the *name* is not sufficient. vLLM's `openai_gptoss` parser raises
   `NotImplementedError` on non-streaming input — *"gpt-oss has a special branch for
   parsing reasoning in non-streaming mode. This method shouldn't be used."* vLLM only
   handles harmony transparently in its **OpenAI-server** path; this repo calls
   `LLM.generate()` **offline**, everywhere. **There is no vLLM parser to use.**

2. **`skip_special_tokens=True`** (vLLM's `SamplingParams` default) *deletes* the
   delimiters during detokenization, collapsing the output to the unsplittable
   `"analysisHidden CoT…assistantfinalAnswer: 3"`. Once the markers are gone, no
   downstream parser can recover the final channel — not even a correct one.

### What is required now

Both handled in `dagspaces/common/vllm_inference.py`; **no per-stage config needed**:

- `_is_harmony_model()` routes gpt-oss to **`_split_harmony()`**, a direct channel
  parser, instead of a vLLM reasoning parser. `_detect_reasoning_parser()` now
  deliberately returns `None` for gpt-oss.
- `run_vllm_inference()` **forces `skip_special_tokens=False`** for harmony models
  after `sp_dicts` are assembled — deliberately in code rather than in the model YAML,
  because a stage that overrides `sampling_params` would otherwise silently drop it.
- A **truncated** generation that never reaches `final` yields `content == ""`, on
  purpose. An empty answer is caught by the format-adherence gate; a plausible-looking
  analysis blob graded as the answer is not.

Guarded by `tests/common/test_harmony_parsing.py`.

### The interaction that bites

Removing gpt-oss from `_detect_reasoning_parser` also disabled the **reasoning-budget
trigger** — and gpt-oss's config carries a bare `chat_template_kwargs: {}`, so the
`enable_thinking` trigger cannot fire for it either. `model_needs_reasoning_budget()`
therefore checks `_is_harmony_model()` **separately**. Do not "simplify" that back into
a single parser lookup.

Why it matters: short-answer stages (ConfAIde ratings, CIRL A/B, MMLU letters) default
to `max_tokens` of 16–64. gpt-oss always reasons first, so without the bump it spends
the whole budget inside `analysis` and never reaches `final` — which, now that
truncation is reported honestly, means **empty answers on every short-answer
benchmark**. The old bug at least returned garbage; the fix makes under-budgeting
return nothing. Both tests in `tests/common/test_reasoning_budget.py` exist to catch
exactly this.

**Sampling:** keep `temperature: 1.0` (set in the model YAML). OpenAI recommends it,
and low temperatures drive gpt-oss into repetition loops that exhaust `max_tokens`
inside the `analysis` channel — observed as 100% `finish_reason=length` in the
2026-05-27 `eval_all` sweep.

---

## Engine requirements

### `gemma-4-12b` needs vLLM ≥ 0.25 — use the **cu129** wheel

The 12B checkpoint's architecture is **`Gemma4UnifiedForConditionalGeneration`**, a
different class from the rest of the family (`Gemma4ForConditionalGeneration` for
31B / E4B / E2B) and **absent from vLLM 0.19.1's registry** — the engine refuses to load
it outright. Verified 2026-07-13 on both engines:

```python
from vllm.model_executor.models.registry import ModelRegistry
"Gemma4UnifiedForConditionalGeneration" in ModelRegistry.get_supported_archs()
#   vLLM 0.19.1      -> False   (engine refuses the model)
#   vLLM 0.25.0+cu129 -> True   (loads; "Resolved architecture: Gemma4Unified…")
```

### The vLLM 0.19.1 → 0.25.0 upgrade: take the `+cu129` wheel, not the PyPI default

**The trap: PyPI's default vLLM wheel is a CUDA 13 build, and this cluster cannot run
it.** klara's driver is **570.124.06, which caps at CUDA 12.8**; CUDA 13 needs ≥580.
Reading only PyPI metadata makes 0.25 look impossible:

```
vllm 0.25.0 (PyPI default) -> nvidia-cutlass-dsl[cu13], torch==2.11.0 (cu13 build)
  => import torch: undefined symbol: ncclDevCommDestroy
```

That conclusion is **wrong**. vLLM also publishes a **`+cu129`** wheel on its GitHub
releases, and CUDA *minor-version compatibility* means a cu129 build runs fine on a
12.8 driver. Verified working on klara:

```bash
uv pip install \
  "vllm @ https://github.com/vllm-project/vllm/releases/download/v0.25.0/vllm-0.25.0%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
  "torch==2.11.0" "transformers>=5.5.3" \
  --extra-index-url https://download.pytorch.org/whl/cu129 \
  --index-strategy unsafe-best-match
uv pip uninstall torchcodec     # see below
```

Result (`.venv-vllm025cu129`, 2026-07-13): `vllm 0.25.0+cu129`, `torch 2.11.0+cu129`,
`transformers 5.13.1` — `torch.cuda.is_available()` True, GPU matmul OK, and
gemma-4-12b **loads and generates**.

For the record, if you ever *do* need a CUDA-12 default-PyPI wheel: the cu13 switch
landed in **0.22.0**, so 0.19.x–0.21.0 are the CUDA-12 releases there. Prefer the cu129
wheel — it gets you the current release instead.

**Two dependency landmines:**

1. **`torchcodec` must be uninstalled.** vLLM declares `torchcodec>=0.14`, but only a
   **cu13** build exists at that version (`libnvrtc.so.13: cannot open shared object
   file`), and the cu129 index only carries 0.11.1, which is too old for torch 2.11.
   torchcodec is audio/video-only — **vLLM imports and runs text models fine without
   it.** Remove it.

2. **RAPIDS blocks resolution and must be dropped from `pyproject.toml`.** vLLM 0.25
   pulls `nvidia-cutlass-dsl` → `cuda-python>=12.8`, which cannot co-resolve with the
   `cudf-cu12` / `cuml-cu12` / `dask-cuda` / `raft-dask-cu12` pins. RAPIDS is imported
   in **exactly one file** — `dagspaces/.uair/stages/topic.py`, in the deprecated
   `.uair` dagspace — behind lazy optional-GPU imports with CPU fallbacks. **No active
   pipeline touches it.**

**What else the upgrade buys:** 0.25.0 declares `transformers>=5.5.3` with no upper
bound, so the manual two-step install dance in [models.md](models.md) (vLLM 0.19.0
falsely declaring `transformers<5`) can be deleted. Plus Qwen3.5 LoRA fixes that may
obsolete `_remap_lora_keys_for_vlm()` — retest before removing it.

**Sequencing.** Do not upgrade the canonical venv (`.venv-vllm025cu129`) in place while a gold-label chain is running:
stages import vLLM fresh from the venv when SLURM starts them, so an in-place upgrade
can break an in-flight run *and* split a corpus across two engine versions. Validate in
a parallel venv, then cut over when the queue drains.

### Architecture support matrix (vLLM 0.19.1, verified 2026-07-13)

| arch | models | 0.19.1 | 0.25.0+cu129 |
|---|---|---|---|
| `Gemma4ForConditionalGeneration` | gemma-4 E2B/E4B/31B | ✅ | ✅ |
| `Gemma4UnifiedForConditionalGeneration` | **gemma-4-12B** | ❌ | ✅ |
| `Qwen3_5ForConditionalGeneration` | qwen3.5 2B/4B/9B | ✅ | ✅ |
| `Qwen2ForCausalLM` | OpenThinker3, CIRL, context-reasoner | ✅ | ✅ |
| `LlamaForCausalLM` | llama3.1-8b, HARC | ✅ | ✅ |
| `Phi3ForCausalLM` | phi-4 | ✅ | ✅ |
| `GptOssForCausalLM` | gpt-oss-20b | ✅ (but see harmony, above) | ✅ |

Note that Qwen3.5 is a **`…ForConditionalGeneration`** (VLM-shaped) architecture, not a
plain CausalLM — which is why its LoRA adapters need key remapping
(`_remap_lora_keys_for_vlm()` in `vllm_inference.py`).

---

## What this set replaces

Retired from active use. The configs remain on disk (older runs reference them) but new
experiments should not add them:

| retired | superseded by |
|---|---|
| `qwen3-8b/*` | `qwen3.5-9b/*` |
| `qwen2.5-7b`, `qwen2.5-72b/*` | `qwen3.5-*`; 72B judge role only |
| `gemma-3-12b/*` | `gemma-4-12b/*` |
| `openthinker-7b` | `openthinker3-7b` |
| `llama3.3-70b` | out of scope — no matched control |
| `qwen3.6-27b`, `qwen3.5-27b` | not in the canonical ladders |

`qwen2.5-72b/judge` and `gemma-4-31b/instruct` are **not task models** and are exempt:
they are infrastructure (the reward judge and the gold-label generator respectively).

---

## Sizing on klara (8× A6000, 48 GB, PCIe)

Everything in the set except gpt-oss-20b fits on **one** A6000 at bf16 and should run
`tensor_parallel_size: 1`. Do not reach for TP to buy KV headroom — on this box the
PCIe collectives are the bottleneck, and TP doubles them. See
[slurm-and-env.md](slurm-and-env.md) for the measured numbers and the
`NCCL_SHM_DISABLE=0` requirement.

- **gpt-oss-20b** dequantizes (MXFP4 → bf16) to ~40 GB. Inference fits one GPU; SFT
  needs 2 (`training/sft=gpt_oss launcher=slurm_train_2x`).
- **gemma-4 KV cost** (measured from `config.json`): E2B 0.04 MB/token, E4B 0.09,
  12B 0.39, 31B 0.98. The 12B is ~2.5× leaner than the 31B and tolerates a larger
  `max_num_seqs`.
- **head_dim=256 across the gemma-4 family** forces `TRITON_ATTN`; FLASH_ATTN and
  FLASHINFER both reject it (*"head_size not supported"*). Not a tunable.

---

## Adding to the set

The bar is a **question the model answers that nothing else in the set can**. Six
general instruction-followers is not a canonical set, it is a leaderboard. Each entry
above earns its place: a rung on a ladder, a published baseline to beat, or a matched
control that isolates one variable.

Mechanics in [howto/add-model.md](howto/add-model.md). Before you trust a first run,
confirm the model's arch is in `ModelRegistry.get_supported_archs()` and that its
reasoning format is actually handled — the gpt-oss bug above lived for months, and it
did not look like a bug. It looked like results.
