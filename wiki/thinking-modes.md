# Thinking Modes: SFT ↔ GRPO ↔ Eval

How reasoning/`<think>` tokens are handled across the three phases, why the current approach works, and how to keep it consistent.

## TL;DR

- **Thinking-capable models** (Qwen3+, Gemma-4, DeepSeek-R1, etc.) can emit `<think>...</think>` blocks before their structured answer.
- This project controls thinking at **three independent layers**, which together form the "mode contract":
  1. **Chat template** (via `enable_thinking` kwarg) — controls what generation prompt the model sees.
  2. **Reasoning parser** (vLLM native or regex fallback) — splits reasoning from content after generation.
  3. **Reward / parsing code** — reads `content`, not `reasoning`.
- Today those layers are set via a handful of independent knobs (`chat_template_kwargs.enable_thinking`, `enable_thinking_grpo`, and SFT template choice). A mismatch between them is silent and can wreck training/eval. Use `model.thinking_mode` as the single source of truth.

## The three knobs, what they do

### 1. `model.thinking_mode` (new, preferred) — single source of truth

One field per model yaml:
```yaml
model:
  thinking_mode: off   # "on" or "off"
```

Replaces (but stays backward-compatible with) `model.chat_template_kwargs.enable_thinking`. When `thinking_mode` is set, every phase derives its thinking flag from this single field.

### 2. `chat_template_kwargs.enable_thinking` (legacy, still read)

Passed verbatim into `tokenizer.apply_chat_template(..., enable_thinking=...)`. For Qwen3+ templates this injects either:
- `<think>\n\n</think>\n\n` (closed, model skips reasoning) when `False`
- `<think>\n` (open, model fills reasoning) when `True`

### 3. `enable_thinking_grpo` (training-time, in GRPO config)

Controls the same chat-template flag at GRPO-generation time. When unset, derived from `model.thinking_mode`.

## Current behavior by phase

| Phase | Mechanism | What the model learns / does |
|---|---|---|
| **SFT** (`grpo_training/stages/sft_training.py:38`) | Qwen template hard-codes `<think>\n\n</think>\n` as a prefix inside the `{% generation %}` block | Model is trained to emit empty-think sentinel, then JSON |
| **GRPO** (`grpo_training.py:166`, `rewards.py:743`) | `enable_thinking_grpo` flag on chat template | `True` → model reasons freely, then JSON; `False` → model emits JSON directly |
| **Eval** (`common/vllm_inference.py`) | `chat_template_kwargs.enable_thinking` + vLLM reasoning parser (+ regex fallback) | `False` → model skips reasoning; content parsed directly |

## The asymmetry trap

**Default in this codebase (when `thinking_mode` is unset):**
- SFT: no-think (hard-coded template prefix)
- GRPO default: **think-then-answer** (`enable_thinking_grpo: true` in `default.yaml`)
- Eval: **no-think** (most Qwen3.5 yamls set `enable_thinking: false`)

If you run with defaults, you get **three different distributions**:
1. SFT teaches `<think></think>` sentinel + JSON
2. GRPO re-trains the policy to fill `<think>...</think>` with reasoning before JSON
3. Eval deploys under no-think

This is actually defensible (many reasoning-model papers do this), but it's load-bearing:

- **GRPO with thinking enabled can blow the token budget**: `online_rground_4gpu.yaml` disables thinking because "the model fills entire `max_completion_length` with `<think>` blocks, leaving no tokens for JSON output → all rewards = 0 → no training signal." This is a real failure mode — worth being the default, not an override.
- **Eval distribution shift**: you deploy in a mode GRPO never trained. JSON format is stable (template stamps the sentinel), but JSON *quality* extrapolates from GRPO's think-mode distribution.

**Recommendation**: match GRPO to eval. If you deploy with `thinking_mode: off`, train GRPO with thinking off. This eliminates the distribution shift and saves the token budget.

## What's standard in the literature

- **DeepSeek-R1** (Guo et al. 2025): reasoning during RL *and* eval; answer extracted after `</think>`.
- **Qwen3/Qwen3.5**: dual-mode — trained in both thinking and non-thinking, switched at inference via `enable_thinking`. Empty `<think>\n\n</think>\n` is the canonical no-think sentinel.
- **Kimi k1.5**: long-CoT RL; reward on extracted answer span, not reasoning.
- **OpenAI o-series**: reasoning produced but hidden from API.

**Stripping reasoning before parsing is the standard**, not an artifact — the parser is either a substring-after-`</think>` or a structured-output schema. Your `_fallback_strip_reasoning` is semantically identical.

## The reasoning parser (vLLM 0.19+)

vLLM ships per-family reasoning parsers that split `reasoning_content` and `content` as separate fields. Built-in parsers in 0.19.0:

```
deepseek_r1, deepseek_v3, qwen3, gemma4, gptoss, granite,
hunyuan_a13b, kimi_k2, minimax_m2, mistral, nemotron_v3, olmo3, seedoss
```

`dagspaces/common/vllm_inference.py` auto-detects the right parser from the model path and uses it for reasoning extraction, falling back to regex only when no parser matches (or when output is truncated mid-think with no closing tag). This future-proofs against new architectures: Gemma-4 has its own parser that understands Gemma-4's reasoning format, not Qwen's.

## Reasoning trace preservation

Reasoning is **kept**, not discarded. Inference outputs include:

| Column | Content |
|---|---|
| `generated_text` | The answer (post-reasoning, structured) |
| `generated_reasoning` | The raw reasoning trace (may be empty) |

This preserves research value — `R_cohere` measures reasoning-to-extraction coherence, Appendix C evaluation traces rely on it, and qualitative review requires it.

## Model-family notes

- **Qwen3.5-9B**: native `enable_thinking` kwarg. `False` emits closed sentinel; `True` emits open `<think>\n`.
- **Gemma-4**: tokenizer has no chat template (it's in the multimodal *processor*). vLLM's `gemma4_reasoning_parser` handles parsing. **Do not** reuse the Qwen `<think>` regex for Gemma-4 — it will silently fail.
- **DeepSeek-R1 / distills**: use the `deepseek_r1` parser. Regex handles most cases.
- **Phi-4, Llama 3.x, GPT-OSS-20B**: non-thinking by default; no think tokens expected. `thinking_mode` is a no-op for these families.

## Migration guide

For existing model yamls:

```yaml
# Before (still works):
model:
  chat_template_kwargs:
    enable_thinking: false

# After (preferred):
model:
  thinking_mode: off
  chat_template_kwargs:
    enable_thinking: false   # keep for backwards compat; will be removed in a future pass
```

For new model yamls: set `thinking_mode` only. The runtime derives `chat_template_kwargs.enable_thinking` from it automatically.

## Programmatic API

```python
from dagspaces.common.stage_utils import resolve_thinking_mode
think_on = resolve_thinking_mode(cfg.model)  # → bool
```

Reads in priority order:
1. `cfg.model.thinking_mode` — "on"/"off"/True/False
2. `cfg.model.chat_template_kwargs.enable_thinking` — bool
3. Default: `True` (matches `apply_chat_template` default)
