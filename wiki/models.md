# Model Configs

## Layout

Shared across all dagspaces: `dagspaces/common/conf/model/<family>/<variant>.yaml`. Resolved via Hydra searchpath (`pkg://dagspaces.common.conf`). Override with `model=<family>/<variant>` on the command line.

Dagspace-local overrides (e.g. `dagspaces/vlm_geoprivacy_bench/conf/model/`) take precedence over shared when present.

## Model zoo

Weights live at `/share/pierson/matt/zoo/models/<ModelName>/`. Existing entries:

```
Qwen3.5-{0.8,2,4,9,27}B, Qwen3.5-9B-SFT-CI, Qwen3.5-9B-SFT-from-2026-03-21
Qwen2.5-7B-Instruct, Qwen2.5-72B-Instruct, Qwen2.5-72B-Instruct-AWQ
Qwen2.5-VL-{3,7}B-Instruct(-AWQ), Qwen3-VL-8B-Instruct
Qwen3-30B-A3B-Instruct-2507, Qwen3-32B-AWQ
Llama-3.1-8B, Llama-3.1-8B-Instruct, Llama-3.2-11B-Vision, Llama-3.3-70B-Instruct
Phi-4, Phi-4-multimodal-instruct
Gemma-3-12B-IT, Gemma-4-E4B, Gemma-4-31B-it, Gemma-4-26B-A4B
GPT-OSS-20B, GPT-OSS-20B-SFT-merged
OpenThinker-7B, OpenThinker3-7B
context-reasoner-ppo_open_thinker_acc_reward
intfloat_e5_base  (embeddings)
```

## YAML format

All model yamls use the `@package _global_` directive so the `model:` key is merged into the top level.

Minimal base model:

```yaml
# @package _global_
model:
  model_source: /share/pierson/matt/zoo/models/Qwen3.5-9B
  model_family: qwen3.5
  chat_template_kwargs:
    enable_thinking: false
  engine_kwargs:
    max_model_len: 8192
    max_num_seqs: 32
    tensor_parallel_size: 1
    trust_remote_code: true
  batch_size: 0
  concurrency: 1
```

SFT/GRPO checkpoint variant (vLLM native LoRA):

```yaml
# @package _global_
model:
  model_source: /share/pierson/matt/zoo/models/Qwen3.5-9B
  model_family: qwen3.5
  lora_path: /share/pierson/matt/UAIR/multirun/.../sft_only/outputs/sft/checkpoint
  chat_template_kwargs:
    enable_thinking: false
  engine_kwargs:
    max_model_len: 8192
    max_num_seqs: 16
    tensor_parallel_size: 1
    trust_remote_code: true
    enforce_eager: true
    enable_lora: true
    max_lora_rank: 64
  batch_size: 0
  concurrency: 1
```

## Key fields

| Field | Purpose |
|---|---|
| `model_source` | Absolute path to HF directory |
| `model_family` | Used for W&B `family:<x>` auto-tag; also affects template handling |
| `chat_template_kwargs.enable_thinking` | If false, vLLM auto-strips `<think>` blocks from output |
| `engine_kwargs.tensor_parallel_size` | Must match launcher GPU count |
| `engine_kwargs.enable_lora` + `max_lora_rank` + `lora_path` | LoRA adapter serving |
| `engine_kwargs.enforce_eager` | Disable CUDA graphs (often required with LoRA on Qwen3.5) |
| `batch_size: 0` | Unbounded / vLLM-managed batching |

## Dependency version notes

### Gemma-4 requires `transformers>=5.5.0` (overrides vLLM's declared pin)

Gemma-4 models (`gemma4` architecture, `Gemma4ForConditionalGeneration`) require **transformers ≥ 5.5.0**, but **vLLM 0.19.0's wheel metadata declares `transformers<5,>=4.56.0`** — a stale pin that does not match vLLM's own release notes (which explicitly state `transformers>=5.5.0` is required for Gemma-4). The `gemma4` code paths inside vLLM 0.19.0 depend on classes only present in transformers 5.5+.

Pins in `pyproject.toml`:
```
vllm>=0.19.0
transformers>=5.5.0
```

Install order (uv will otherwise downgrade transformers to satisfy the stale vLLM pin):

```bash
# 1. install vllm (will pull transformers down to 4.57.x)
uv pip install --upgrade "vllm>=0.19.0"
# 2. force transformers back up (transitively bumps huggingface-hub, numpy)
uv pip install --upgrade "transformers>=5.5.0,<6"
```

Verify after install:
```python
from transformers import Gemma4ForConditionalGeneration  # must import
from vllm.model_executor.models.registry import ModelRegistry
assert "Gemma4ForConditionalGeneration" in ModelRegistry.get_supported_archs()
```

Expected to be fixed upstream in vLLM 0.19.1 — drop the manual override then.

## Special-case handling

**Qwen3.5 LoRA + vLLM**: the shared helper at `dagspaces/common/vllm_inference.py` contains `_remap_lora_keys_for_vlm()` which remaps adapter keys from CausalLM → VLM prefix when needed. This workaround was added for vLLM 0.17/0.18. **vLLM 0.19.0 includes [PR #36976 "Fix Qwen35 LoRA"](https://github.com/vllm-project/vllm/pull/36976) which may obsolete this patch** — smoke-test a Qwen3.5 LoRA load without the remap cache (`_vlm_remapped/`) to confirm before removing.

**Qwen3.5 + TRL GRPO**: vLLM is disabled (`training.grpo.use_vllm=false`) — unresolved compat issues between Qwen3.5 + vLLM + TRL. vLLM 0.19.0 includes several Qwen3.5 fixes; retest with `use_vllm=true` after upgrade.

**QLoRA models (≥12B)**: single GPU only (quantized params can't DDP-sync) — use `training/sft=sft_27b`.

**GPT-OSS-20B**: dequantized (Mxfp4 → bf16) at ~40GB, requires 2 GPUs via DDP — use `training/sft=gpt_oss launcher=slurm_train_2x`.
