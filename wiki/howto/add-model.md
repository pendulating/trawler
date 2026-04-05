# How to add a new model

## 1. Download to the zoo

```bash
# Standard HF download
huggingface-cli download <org>/<ModelName> \
  --local-dir /share/pierson/matt/zoo/models/<ModelName> \
  --local-dir-use-symlinks False

# For gated models, login first:  huggingface-cli login
```

By convention, `<ModelName>` matches the HF repo leaf (e.g. `Qwen3.5-9B`, `Phi-4`, `Gemma-3-12B-IT`).

## 2. Create the model yaml

Path: `dagspaces/common/conf/model/<family>/<variant>.yaml`.

Directory naming: `<family>` = model family (e.g. `qwen3.5-9b`, `gemma-3-12b`). Variants: `base`, `instruct`, `it`, `sft-ci`, `grpo-*`, `awq`, etc.

Template:

```yaml
# @package _global_
model:
  model_source: /share/pierson/matt/zoo/models/<ModelName>
  model_family: <family_tag>           # used for W&B tagging
  chat_template_kwargs:
    enable_thinking: false             # true only if model natively has <think> and you want to keep it
  engine_kwargs:
    max_model_len: 8192
    max_num_seqs: 32
    tensor_parallel_size: 1            # match launcher GPU count
    trust_remote_code: true
  batch_size: 0
  concurrency: 1
```

**Tensor parallelism**: for 13B–30B dense models try TP=2 first; for 70B+ use TP=4 or AWQ variant on TP=2.

**For LoRA checkpoints** — add `lora_path`, set `enable_lora: true`, `max_lora_rank: 64` (or your rank), `enforce_eager: true`.

**For AWQ quantized** — usually no extra kwargs needed; vLLM auto-detects.

## 3. Smoke-test with an eval

```bash
python -m dagspaces.goldcoin_hipaa.cli pipeline=full_eval \
  model=<family>/<variant> \
  runtime.debug=true runtime.sample_n=5 hydra/launcher=null
```

This loads the model via vLLM on your login node and runs on 5 cases.

## 4. Wire into training pipelines (if applicable)

- For SFT as base: `python -m dagspaces.grpo_training.cli -m pipeline=sft_only model=<family>/<variant>`
- If the model is ≥12B and needs QLoRA: append `training/sft=sft_27b`
- If dequantized large (like GPT-OSS-20B): `training/sft=gpt_oss pipeline.graph.nodes.sft_training.launcher=slurm_train_2x`

## 5. Register a post-training variant

After SFT produces a checkpoint, create a sibling yaml (e.g. `<family>/<variant>-sft-ci.yaml`) that adds `lora_path` pointing to `multirun/.../sft_only/outputs/sft/checkpoint`. Evaluate with that config.

## Gotchas

- **Qwen3.5 + vLLM 0.17 + LoRA**: requires the key-remap patch (see memory `feedback_lora_key_remap.md`).
- **Qwen3.5 + TRL GRPO**: set `training.grpo.use_vllm=false`.
- **VLMs**: add `dagspaces/vlm_geoprivacy_bench/conf/model/<family>/<variant>.yaml` for dagspace-local overrides if image handling differs.
- **Chat template**: verify `<model_source>/tokenizer_config.json` has a `chat_template` or vLLM falls back to a generic one and your prompts may misalign.
