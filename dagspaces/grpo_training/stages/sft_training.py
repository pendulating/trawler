"""SFT training stage: LoRA fine-tuning via TRL's SFTTrainer.

Takes SFT pairs (in chat messages format) and fine-tunes a base model
using parameter-efficient LoRA adaptation.

Multi-GPU support:
  - Uses `accelerate launch` for DDP data-parallel training across GPUs.
    The runner spawns this as a subprocess so each GPU gets its own process.
  - When LOCAL_RANK is set (i.e. inside accelerate), the Trainer handles
    device placement automatically.
"""

import json
import os
import time
from typing import Any

import pandas as pd
from omegaconf import OmegaConf

# TRL-compatible chat templates with {% generation %} blocks for loss masking.
# Each template marks assistant content so SFTTrainer only computes loss on
# completion tokens. The model family is detected from the tokenizer's native
# template or config architecture.
#
# These are FALLBACKS: since 2026-07-18 the primary path is TRL 1.8.0's own
# registry of matched native training templates (trl.get_training_chat_template
# — byte-matches the tokenizer's native template and returns the official
# render with generation markers). Manual templates are used only when the
# registry has no match (gemma-4, phi-4, phi-4-mm as of TRL 1.8.0).
#
# INVARIANT (added 2026-07-18): the end-of-turn token MUST sit INSIDE the
# {% generation %} block. Before this, every manual template left it outside,
# so with assistant_only_loss the stop token was label-masked (-100) and the
# model was never trained to emit it. TRL's own training templates include it
# (is_chat_template_stop_token_trained warns exactly about this). Guarded by
# tests/grpo_training/test_sft_chat_templates.py.

def _qwen_sft_template(thinking_enabled: bool) -> str:
    """Build the Qwen SFT chat template.

    When ``thinking_enabled=False`` (no-think training), the assistant
    generation block is prefixed with the empty-think sentinel
    ``<think>\\n\\n</think>\\n\\n`` — the official Qwen3/Qwen3.5 no-think
    form (verified against the Qwen3.5 chat_template.jinja 2026-07-18: the
    serve-time prompt with ``enable_thinking=false`` ends with
    ``</think>\\n\\n``, i.e. a blank line before the content). Until
    2026-07-18 this template emitted a single trailing newline, leaving
    training one ``\\n`` off from the serve-time render.

    When ``thinking_enabled=True``, no prefix is injected; the model is
    free to emit its own ``<think>...</think>`` block before the content.
    Note: unless the SFT dataset contains reasoning traces, this won't
    actively *train* reasoning — it merely leaves the pretrained
    reasoning distribution untouched.
    """
    prefix = "<think>\n\n</think>\n\n" if not thinking_enabled else ""
    # Tags use `{%` NOT `{%-` (same rationale as the gemma-4 template below):
    # `{%-` strips preceding whitespace, which silently ate the "\n" after
    # every `<|im_end|>` and rendered glued turns (`U<|im_end|><|im_start|>
    # assistant`) instead of the native `U<|im_end|>\n<|im_start|>assistant`.
    # Ground-truthed against the 2026-07-15 sweep traces: every prior qwen SFT
    # run trained with the glued form. The string is concatenated with no
    # incidental whitespace between tags, so non-stripping tags emit exactly
    # these literals and nothing more.
    return (
        "{% for message in messages %}"
        "{% if message.role == 'system' %}"
        "<|im_start|>system\n{{ message.content | trim }}<|im_end|>\n"
        "{% elif message.role == 'user' %}"
        "<|im_start|>user\n{{ message.content | trim }}<|im_end|>\n"
        "{% elif message.role == 'assistant' %}"
        "<|im_start|>assistant\n"
        "{% generation %}" + prefix + "{{ message.content | trim }}<|im_end|>\n{% endgeneration %}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|im_start|>assistant\n"
        "{% endif %}"
    )


_SFT_TEMPLATES = {
    # Qwen family: <|im_start|>role\ncontent<|im_end|>
    # Preserved as a no-think template for backwards compat; use
    # ``_qwen_sft_template(thinking_enabled=...)`` to build either variant.
    "qwen": _qwen_sft_template(thinking_enabled=False),
    # Phi-4: <|im_start|>role<|im_sep|>content<|im_end|>
    "phi-4": (
        "{%- for message in messages %}"
        "{%- if message.role == 'system' %}"
        "<|im_start|>system<|im_sep|>{{ message.content | trim }}<|im_end|>"
        "{%- elif message.role == 'user' %}"
        "<|im_start|>user<|im_sep|>{{ message.content | trim }}<|im_end|>"
        "{%- elif message.role == 'assistant' %}"
        "<|im_start|>assistant<|im_sep|>"
        "{% generation %}{{ message.content | trim }}<|im_end|>{% endgeneration %}"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "<|im_start|>assistant<|im_sep|>"
        "{%- endif %}"
    ),
    # Phi-4-multimodal: <|role|>content<|end|>
    "phi-4-mm": (
        "{%- for message in messages %}"
        "{%- if message.role == 'system' %}"
        "<|system|>{{ message.content | trim }}<|end|>"
        "{%- elif message.role == 'user' %}"
        "<|user|>{{ message.content | trim }}<|end|>"
        "{%- elif message.role == 'assistant' %}"
        "<|assistant|>"
        "{% generation %}{{ message.content | trim }}<|end|>{% endgeneration %}"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "<|assistant|>"
        "{%- endif %}"
    ),
    # Gemma-4: <|turn>role\ncontent<turn|>\n  — NOT interchangeable with Gemma-3.
    # Verified byte-for-byte against the native chat_template.jinja of
    # gemma-4-{12B,E2B,E4B,31B}-it for our data shape:
    #   <bos><|turn>user\nU<turn|>\n<|turn>model\nA<turn|>\n
    # Differences from Gemma-3 that matter: distinct turn delimiters, a real
    # `system` turn (Gemma-3 folds system into the first user turn), and role
    # name "model" for the assistant.
    #
    # NOTE on the generation prompt: gemma-4-12b/31b natively append an empty
    # thinking channel (`<|turn>model\n<|channel>thought\n<channel|>`) while
    # E2B/E4B append only `<|turn>model\n`. Training never uses the
    # add_generation_prompt branch (SFTTrainer renders complete conversations),
    # and the full-conversation render is identical across all four, so this
    # template matches training exactly and leaves the channel priming to the
    # native template at serve time.
    # Tags use `{%` NOT `{%-` deliberately. `{%-` strips preceding whitespace,
    # which silently ate the "\n" after every `<turn|>` and produced
    # `...U<turn|><|turn>model...` instead of the native `...U<turn|>\n<|turn>
    # model...`. The template is one concatenated string with no incidental
    # whitespace between tags, so non-stripping tags emit exactly these literals
    # and nothing more. Verified byte-for-byte against all four checkpoints.
    "gemma-4": (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if message.role == 'system' %}"
        "<|turn>system\n{{ message.content | trim }}<turn|>\n"
        "{% elif message.role == 'user' %}"
        "<|turn>user\n{{ message.content | trim }}<turn|>\n"
        "{% elif message.role == 'assistant' %}"
        "<|turn>model\n"
        "{% generation %}{{ message.content | trim }}<turn|>\n{% endgeneration %}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<|turn>model\n"
        "{% endif %}"
    ),
    # Gemma-3: <start_of_turn>role\ncontent<end_of_turn>
    # System message is prepended to first user message.
    # Fixed 2026-07-18: (1) non-stripping `{%` tags — `{%-` ate the "\n" after
    # every <end_of_turn> and after "user"; (2) namespace() for the system
    # fold — a plain `{% set %}` inside the loop does not persist across
    # iterations, and the old `loop.first` check could never fire for a
    # [system, user, ...] conversation anyway (the user turn is index 1).
    "gemma": (
        "{{ bos_token }}"
        "{% set ns = namespace(system_message='') %}"
        "{% for message in messages %}"
        "{% if message.role == 'system' %}"
        "{% set ns.system_message = message.content | trim + '\n\n' %}"
        "{% elif message.role == 'user' %}"
        "<start_of_turn>user\n"
        "{% if ns.system_message %}{{ ns.system_message }}{% set ns.system_message = '' %}{% endif %}"
        "{{ message.content | trim }}<end_of_turn>\n"
        "{% elif message.role == 'assistant' %}"
        "<start_of_turn>model\n"
        "{% generation %}{{ message.content | trim }}<end_of_turn>\n{% endgeneration %}"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "<start_of_turn>model\n"
        "{% endif %}"
    ),
    # Llama-3: <|start_header_id|>role<|end_header_id|>\n\ncontent<|eot_id|>
    "llama": (
        "{{ bos_token }}"
        "{%- for message in messages %}"
        "{%- if message.role == 'system' %}"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        "{{ message.content | trim }}<|eot_id|>"
        "{%- elif message.role == 'user' %}"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        "{{ message.content | trim }}<|eot_id|>"
        "{%- elif message.role == 'assistant' %}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "{% generation %}{{ message.content | trim }}<|eot_id|>{% endgeneration %}"
        "{%- endif %}"
        "{%- endfor %}"
        "{%- if add_generation_prompt %}"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        "{%- endif %}"
    ),
    # GPT-OSS-20B deliberately has NO manual template. The pre-2026-07-18 one
    # corrupted harmony on three axes (verified against the official
    # openai/gpt-oss-20b template + harmony spec + OpenAI fine-tuning
    # cookbook): (1) it omitted the always-on harmony system preamble
    # ("You are ChatGPT... Reasoning: ... # Valid channels: ...") and put the
    # stage instructions in the `system` role, where harmony post-training
    # only ever saw that fixed meta block — instructions belong in a
    # `developer` turn under "# Instructions"; (2) it terminated the final
    # assistant turn with <|end|> where training must use <|return|> (the
    # stop token harmony inference stacks key on); (3) it never rendered an
    # analysis->final channel transition. Plausible root cause of the 33%
    # empty-final-channel regression in the 2026-07-15 sweep. gpt-oss SFT
    # must go through TRL's registry training template (official harmony
    # render + generation markers); the stage raises if that match fails
    # rather than silently reintroducing the broken format.
}


def _detect_template_family(tokenizer, model_path: str) -> str:
    """Detect which SFT chat template to use based on model family.

    Gemma-3 vs Gemma-4 MUST be distinguished. They share no control tokens:
    Gemma-3 turns are ``<start_of_turn>``/``<end_of_turn>``, Gemma-4 turns are
    ``<|turn>``/``<turn|>`` with an optional ``<|channel>thought`` block, and
    Gemma-4 has a real ``system`` turn where Gemma-3 folds the system message
    into the first user turn.

    Until 2026-07-18 this function matched the bare substring "gemma" and
    handed every Gemma-4 model the Gemma-3 template. `<start_of_turn>` is not in
    the Gemma-4 vocabulary at all, so each turn delimiter was tokenized as SEVEN
    arbitrary sub-word pieces ('<','start','_','of','_','turn','>'). All three
    gemma-4 cells of the 2026-07-15 canonical SFT sweep trained that way, which
    is what produced their anomalous optimisation (gemma-4-12b: initial loss
    3.12 vs ~0.9 elsewhere, median grad-norm 6.15 vs 0.54, a 644 spike, and 31
    of 54 logged steps clipped at max_grad_norm=1.0). Those adapters were also
    served under the model's NATIVE template at eval time, so train and serve
    formats disagreed.
    """
    path_lower = model_path.lower()
    if "gpt-oss" in path_lower:
        return "gpt-oss"
    if "phi-4-multimodal" in path_lower:
        return "phi-4-mm"
    if "phi-4" in path_lower or "phi-3" in path_lower:
        return "phi-4"
    if "gemma" in path_lower:
        # Prefer the tokenizer over the path: a renamed or symlinked checkpoint
        # directory must not decide which control tokens we emit.
        vocab = tokenizer.get_vocab() if hasattr(tokenizer, "get_vocab") else {}
        if "<|turn>" in vocab:
            return "gemma-4"
        if "<start_of_turn>" in vocab:
            return "gemma"
        # Fall back to the path only when the vocab is uninformative.
        return "gemma-4" if ("gemma-4" in path_lower or "gemma4" in path_lower) else "gemma"
    if "llama" in path_lower:
        return "llama"
    # Check tokenizer for Qwen-style markers
    native = tokenizer.chat_template or ""
    if "<|im_start|>" in native:
        return "qwen"
    if "<|turn>" in native:
        return "gemma-4"
    if "<start_of_turn>" in native:
        return "gemma"
    if "<|start_header_id|>" in native:
        return "llama"
    # Default to Qwen (ChatML) as fallback
    return "qwen"


# Control tokens each manual template family emits as turn delimiters. Every one
# of these MUST tokenize to a single id in the target model's vocab; if it
# splits into sub-word pieces, the template does not belong to this model and
# training silently degrades (the 2026-07-15 gemma-4 cells trained with Gemma-3
# delimiters that split into SEVEN pieces each — initial loss 3.12 vs ~0.9,
# grad-norm spikes to 644, 31/54 steps clipped). Checked by
# _assert_template_tokens_atomic() before training starts.
_FAMILY_CONTROL_TOKENS = {
    "qwen": ["<|im_start|>", "<|im_end|>"],
    "phi-4": ["<|im_start|>", "<|im_sep|>", "<|im_end|>"],
    "phi-4-mm": ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
    "gemma-4": ["<|turn>", "<turn|>"],
    "gemma": ["<start_of_turn>", "<end_of_turn>"],
    "llama": ["<|start_header_id|>", "<|end_header_id|>", "<|eot_id|>"],
    "gpt-oss": ["<|start|>", "<|channel|>", "<|message|>", "<|end|>"],
}

# Early-warning thresholds (loud console + trace warnings, not hard failures).
# Calibrated on the 2026-07-15 canonical sweep post-mortem: 8 healthy cells
# (init loss 0.85-1.38, grad-norm max <=1.64, <=4% steps clipped) vs 3
# template-corrupted gemma-4 cells (init 1.35/1.82/3.12, grad max
# 16.5/17.7/644, clipped 11%/65%/100%). Composite rule: any of the three
# firing means suspect. NOTE final loss and completion/total mask ratio are
# provably USELESS detectors — all corrupted cells converged to normal final
# loss (0.32-0.44) and dead-center-normal mask ratios; do not add them.
_WARN_INITIAL_LOSS = 1.6      # catches 12b (3.12) + e2b (1.82); e4b (1.35)
                              # hides in-band — the grad signals catch it
_WARN_CLIP_FRACTION = 0.10    # healthy <=4%, corrupted >=11%
_WARN_GRAD_SPIKE_MULT = 3.0   # healthy max <=1.64x clip; corrupted >=16.5x.
                              # Fires immediately mid-run, not at train end.


def _assert_template_tokens_atomic(tokenizer, family: str) -> None:
    """Hard-fail before training if the template's control tokens are not
    single tokens in this model's vocabulary (i.e. wrong-family template)."""
    bad = []
    for tok in _FAMILY_CONTROL_TOKENS.get(family, []):
        try:
            ids = tokenizer(tok, add_special_tokens=False)["input_ids"]
        except Exception:
            ids = []
        if len(ids) != 1:
            bad.append((tok, len(ids)))
    if bad:
        detail = ", ".join(f"{t!r} -> {n} pieces" for t, n in bad)
        raise ValueError(
            f"Chat-template/model mismatch: family={family!r} control tokens "
            f"do not tokenize atomically in this model's vocab ({detail}). "
            "Training with a wrong-family template silently corrupts the run "
            "(see the 2026-07-15 gemma-4 canonical-sweep incident). Fix the "
            "template family detection before training."
        )


def _append_traces(path: str, entries: list[dict[str, Any]]) -> None:
    """Append trace entries to a JSONL file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _log_init_trace(
    trace_path: str,
    model,
    peft_config,
    base_model: str,
    family: str,
    dataset_size: int,
    sft_cfg: dict,
) -> None:
    """Log model/LoRA architecture summary at training start."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())

    # Collect which modules PEFT actually targeted
    targeted = set()
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A"):
            # Extract the short module name (e.g. "q_proj" from "model.layers.0.self_attn.q_proj")
            targeted.add(name.rsplit(".", 1)[-1] if "." in name else name)

    _append_traces(trace_path, [{
        "type": "init",
        "base_model": base_model,
        "template_family": family,
        "dataset_size": dataset_size,
        "trainable_params": trainable,
        "total_params": total,
        "trainable_pct": round(100 * trainable / total, 2) if total else 0,
        "lora_rank": peft_config.r,
        "lora_alpha": peft_config.lora_alpha,
        "lora_target_modules_config": (
            peft_config.target_modules
            if isinstance(peft_config.target_modules, str)
            else list(peft_config.target_modules)
        ),
        "lora_target_modules_resolved": sorted(targeted),
        "loss_type": sft_cfg.get("loss_type") or "trl-default",
        "num_epochs": sft_cfg.get("num_epochs", 3),
        "per_device_batch_size": sft_cfg.get("per_device_batch_size", 4),
        "gradient_accumulation_steps": sft_cfg.get("gradient_accumulation_steps", 4),
        "learning_rate": sft_cfg.get("learning_rate", 2e-5),
        "max_seq_length": sft_cfg.get("max_seq_length", 8192),
    }])


def _log_tokenization_samples(
    trace_path: str,
    trainer,
    tokenizer,
    n_samples: int = 3,
) -> None:
    """Log a few tokenized examples showing label masking."""
    dataloader = trainer.get_train_dataloader()
    batch = next(iter(dataloader))
    for i in range(min(n_samples, batch["input_ids"].size(0))):
        input_ids = batch["input_ids"][i]
        labels = batch["labels"][i]

        # Find boundaries: prompt (masked) vs completion (trained)
        train_mask = labels != -100
        n_train = train_mask.sum().item()
        n_masked = (~train_mask).sum().item()
        total = input_ids.size(0)

        # Decode prompt and completion portions separately
        prompt_ids = input_ids[~train_mask].tolist()
        completion_ids = input_ids[train_mask].tolist()

        _append_traces(trace_path, [{
            "type": "tokenization_sample",
            "idx": i,
            "total_tokens": total,
            "prompt_tokens": n_masked,
            "completion_tokens": n_train,
            "prompt_text": tokenizer.decode(prompt_ids, skip_special_tokens=False),
            "completion_text": tokenizer.decode(completion_ids, skip_special_tokens=False),
        }])


class _SFTTraceCallback:
    """Transformers TrainerCallback that logs periodic step traces to JSONL."""

    def __init__(self, trace_path: str, trace_every: int, tokenizer):
        from transformers import TrainerCallback
        self._base = TrainerCallback
        self.trace_path = trace_path
        self.trace_every = trace_every
        self.tokenizer = tokenizer
        self._start_time = time.time()
        # Overfitting / degradation telemetry
        self._first_loss = None
        self._last_train_loss = None
        self._prev_eval_loss = None
        self._grad_steps = 0
        self._clipped_steps = 0
        self._grad_spike_warned = False

    def on_log(self, args, state, control, logs=None, **kwargs):
        # Rank-0 only: under accelerate DDP/FSDP this callback fires on every
        # rank; without the guard the 2-GPU gpt-oss cell interleaves duplicate
        # trace entries and prints every warning twice.
        if logs is None or not state.is_world_process_zero:
            return
        loss = logs.get("loss")
        grad_norm = logs.get("grad_norm")
        if loss is not None:
            self._last_train_loss = loss
            # Early warning: an anomalously high FIRST loss is the signature of
            # a template/model mismatch (2026-07-15 gemma-4 cells: 3.12 vs ~0.9
            # healthy) or broken label masking. Warn loudly at step 1, not at
            # the post-benchmark autopsy.
            if self._first_loss is None:
                self._first_loss = loss
                if loss > _WARN_INITIAL_LOSS:
                    msg = (f"[sft_training] WARNING initial loss {loss:.2f} > "
                           f"{_WARN_INITIAL_LOSS} — possible chat-template/model "
                           "mismatch or corrupted label masking (healthy cells "
                           "start near ~0.9). Inspect tokenization_sample "
                           "entries in sft_traces.jsonl before trusting this run.")
                    print(msg, flush=True)
                    _append_traces(self.trace_path, [{
                        "type": "warning", "kind": "high_initial_loss",
                        "global_step": state.global_step, "loss": loss,
                        "threshold": _WARN_INITIAL_LOSS,
                    }])
        if grad_norm is not None:
            self._grad_steps += 1
            if args.max_grad_norm and grad_norm >= args.max_grad_norm:
                self._clipped_steps += 1
            # Immediate alarm: a grad-norm spike is the single most
            # discriminating template-corruption signal (healthy runs never
            # exceeded 1.64x the clip threshold; all three corrupted 2026-07-15
            # cells hit >=16.5x). Fires mid-run so a doomed cell can be killed
            # in minutes instead of after 6-14 GPU-hours.
            if (not self._grad_spike_warned and args.max_grad_norm
                    and grad_norm >= _WARN_GRAD_SPIKE_MULT * args.max_grad_norm):
                self._grad_spike_warned = True
                print(f"[sft_training] WARNING grad_norm {grad_norm:.1f} >= "
                      f"{_WARN_GRAD_SPIKE_MULT}x max_grad_norm "
                      f"({args.max_grad_norm}) at step {state.global_step} — "
                      "signature of chat-template/model mismatch. Inspect "
                      "tokenization samples; consider killing this run.",
                      flush=True)
                _append_traces(self.trace_path, [{
                    "type": "warning", "kind": "grad_norm_spike",
                    "global_step": state.global_step,
                    "grad_norm": grad_norm,
                    "threshold": _WARN_GRAD_SPIKE_MULT * args.max_grad_norm,
                }])
        if state.global_step % self.trace_every != 0:
            return
        entry = {
            "type": "step",
            "global_step": state.global_step,
            "epoch": round(state.epoch, 3) if state.epoch else 0,
            "wall_seconds": round(time.time() - self._start_time, 1),
            "loss": loss,
            "learning_rate": logs.get("learning_rate"),
            "grad_norm": grad_norm,
        }
        _append_traces(self.trace_path, [entry])

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Log eval loss, the train/eval gap, and a rising-eval warning."""
        if not metrics or "eval_loss" not in metrics or not state.is_world_process_zero:
            return
        eval_loss = metrics["eval_loss"]
        gap = (eval_loss - self._last_train_loss
               if self._last_train_loss is not None else None)
        delta = (eval_loss - self._prev_eval_loss
                 if self._prev_eval_loss is not None else None)
        _append_traces(self.trace_path, [{
            "type": "eval",
            "global_step": state.global_step,
            "epoch": round(state.epoch, 3) if state.epoch else 0,
            "eval_loss": eval_loss,
            "train_loss_recent": self._last_train_loss,
            "overfit_gap": round(gap, 4) if gap is not None else None,
            "eval_loss_delta": round(delta, 4) if delta is not None else None,
        }])
        if delta is not None and delta > 0:
            print(f"[sft_training] WARNING eval_loss rose "
                  f"{self._prev_eval_loss:.4f} -> {eval_loss:.4f} at epoch "
                  f"{state.epoch:.1f} — overfitting past the best epoch. "
                  "Best-epoch selection will discard these weights "
                  "(select_best_epoch).", flush=True)
        self._prev_eval_loss = eval_loss

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        clip_frac = (self._clipped_steps / self._grad_steps
                     if self._grad_steps else None)
        if clip_frac is not None and clip_frac > _WARN_CLIP_FRACTION:
            print(f"[sft_training] WARNING {self._clipped_steps}/"
                  f"{self._grad_steps} logged steps "
                  f"({clip_frac:.0%}) hit max_grad_norm — the 2026-07-15 "
                  "template-corrupted cells clipped 11-100% of steps vs <=4% "
                  "healthy. Treat this run as suspect until the tokenization "
                  "samples are checked.",
                  flush=True)
        _append_traces(self.trace_path, [{
            "type": "final",
            "global_step": state.global_step,
            "total_wall_seconds": round(time.time() - self._start_time, 1),
            "best_metric": state.best_metric,
            "best_model_checkpoint": state.best_model_checkpoint,
            "initial_loss": self._first_loss,
            "final_eval_loss": self._prev_eval_loss,
            "grad_clip_fraction": round(clip_frac, 3) if clip_frac is not None else None,
            "total_flos": state.total_flos,
        }])


def _make_trace_callback(trace_path: str, trace_every: int, tokenizer):
    """Build a TrainerCallback for SFT trace logging."""
    from transformers import TrainerCallback

    class SFTTraceCallback(TrainerCallback):
        def __init__(self):
            self._inner = _SFTTraceCallback(trace_path, trace_every, tokenizer)

        def on_log(self, args, state, control, logs=None, **kwargs):
            self._inner.on_log(args, state, control, logs=logs, **kwargs)

        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            self._inner.on_evaluate(args, state, control, metrics=metrics, **kwargs)

        def on_train_end(self, args, state, control, **kwargs):
            self._inner.on_train_end(args, state, control, **kwargs)

    return SFTTraceCallback()


def run_sft_training_stage(
    dataset_path: str,
    base_model: str,
    output_dir: str,
    cfg: Any,
) -> None:
    """Run SFT training with TRL.

    Args:
        dataset_path: Path to sft_pairs.parquet with 'messages' column.
        base_model: HuggingFace model ID or local path for the base model.
        output_dir: Directory to save the LoRA checkpoint.
        cfg: Hydra config with training.sft section.
    """
    import torch
    from datasets import Dataset
    from peft import LoraConfig, TaskType
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl import SFTConfig, SFTTrainer

    sft_cfg = OmegaConf.to_container(
        OmegaConf.select(cfg, "training.sft"), resolve=True
    )

    # Load training data
    df = pd.read_parquet(dataset_path)
    print(f"[sft_training] Loaded {len(df)} training pairs from {dataset_path}")

    # Book-level filter (safety net for reused sft_pairs.parquet)
    book_id = OmegaConf.select(cfg, "runtime.book_id", default=None)
    if book_id is not None and "source_id" in df.columns:
        book_id = str(book_id)
        pre = len(df)
        df = df[df["source_id"].astype(str) == book_id].reset_index(drop=True)
        print(f"[sft_training] Filtered to book_id={book_id}: {len(df)}/{pre} pairs")

    # Parse messages from JSON strings back to lists
    def parse_messages(row):
        msgs = row["messages"]
        if isinstance(msgs, str):
            return {"messages": json.loads(msgs)}
        return {"messages": msgs}

    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(parse_messages)

    # Optional: subsample training data for quick iteration
    sample_fraction = sft_cfg.get("sample_fraction")
    if sample_fraction is not None and 0.0 < sample_fraction < 1.0:
        n_keep = max(1, int(len(dataset) * sample_fraction))
        dataset = dataset.shuffle(seed=42).select(range(n_keep))
        print(f"[sft_training] Sampled {n_keep}/{len(df)} examples ({sample_fraction:.0%})")

    # ── Held-out evaluation split ────────────────────────────────────────────
    # Until 2026-07-18 SFT ran with NO validation set at all: `eval keys: NONE`
    # in every trainer_state.json of the canonical sweep. Three epochs with no
    # held-out signal means overfitting and divergence are invisible until a
    # downstream benchmark trips days later.
    #
    # split="grouped" (default) holds out WHOLE source novels. A random split
    # leaks: chunks of one novel share characters, prose style and norms, so
    # train and eval halves of the same book are near-duplicates and eval loss
    # reads optimistically low. Grouped measures the thing the project actually
    # claims — transfer to an unseen normative universe.
    #
    # CAVEAT worth stating plainly: this eval loss is computed on held-out
    # examples of the SAME ci_extraction task in the SAME `{"reasoning": ...}`
    # target format. It therefore does NOT detect format over-commitment — a
    # model that learns to emit that JSON shape for every input scores BETTER
    # here while getting worse on ReAct/MCQ benchmarks. Catching that needs an
    # out-of-format probe, not this metric.
    eval_dataset = None
    eval_split = str(sft_cfg.get("eval_split", "grouped")).lower()
    eval_fraction = float(sft_cfg.get("eval_fraction", 0.05) or 0.0)
    eval_seed = int(sft_cfg.get("eval_seed", 42))

    if eval_split == "none" or eval_fraction <= 0.0:
        print("[sft_training] No held-out eval split (eval_split=none) — "
              "training loss will be the only signal.")
    elif eval_split == "grouped" and "source_id" in df.columns:
        counts = df["source_id"].astype(str).value_counts()
        target = eval_fraction * len(df)
        if len(counts) < 2:
            print(f"[sft_training] WARNING only {len(counts)} source_id present "
                  "— cannot hold out a novel; falling back to random split.")
            eval_split = "random"
        else:
            # Prefer the SINGLE novel closest to the target size: holding out one
            # book is interpretable ("generalises to an unseen novel") and keeps
            # the other normative universes in training. Fall back to greedily
            # accumulating the smallest novels only if no single book is big
            # enough to reach the target.
            best = min(counts.index, key=lambda s: abs(counts[s] - target))
            held = [best]
            if counts[best] < target * 0.5:
                held, acc = [], 0
                for s in counts.sort_values().index:
                    held.append(s)
                    acc += counts[s]
                    if acc >= target:
                        break
            held_set = {str(s) for s in held}
            src = [str(s) for s in dataset["source_id"]]
            tr_idx = [i for i, s in enumerate(src) if s not in held_set]
            ev_idx = [i for i, s in enumerate(src) if s in held_set]
            eval_dataset = dataset.select(ev_idx)
            dataset = dataset.select(tr_idx)
            print(f"[sft_training] Held-out split (grouped by source_id): "
                  f"novels {sorted(held_set)} -> {len(eval_dataset)} eval / "
                  f"{len(dataset)} train ({len(eval_dataset)/(len(eval_dataset)+len(dataset)):.1%})")

    if eval_dataset is None and eval_split == "random" and eval_fraction > 0.0:
        split = dataset.train_test_split(test_size=eval_fraction, seed=eval_seed)
        dataset, eval_dataset = split["train"], split["test"]
        print(f"[sft_training] Held-out split (random, seed={eval_seed}): "
              f"{len(eval_dataset)} eval / {len(dataset)} train. NOTE: chunks of "
              "the same novel appear on both sides — eval loss is optimistic.")

    print(f"[sft_training] Base model: {base_model}")
    print(f"[sft_training] Output dir: {output_dir}")

    # Load weights from the node-local /scratch registry mirror when synced
    # (basename-preserving, so the family/template detection below is
    # unaffected). Config-level records keep the canonical zoo path.
    from dagspaces.common.model_registry import resolve_model_source
    base_model = resolve_model_source(base_model, stage_name="sft_training")

    # Configure LoRA
    lora_cfg = sft_cfg.get("lora", {})
    peft_config = LoraConfig(
        r=lora_cfg.get("rank", 64),
        lora_alpha=lora_cfg.get("alpha", 128),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        target_modules=lora_cfg.get("target_modules", "all-linear"),
        task_type=TaskType.CAUSAL_LM,
    )

    # Device placement: accelerate sets LOCAL_RANK for DDP
    is_distributed = os.environ.get("LOCAL_RANK") is not None
    n_gpus = torch.cuda.device_count()

    # Prefer flash_attention_2 for full attention layers; fall back to sdpa
    try:
        import flash_attn  # noqa: F401
        _attn_impl = "flash_attention_2"
    except ImportError:
        _attn_impl = "sdpa"
    print(f"[sft_training] Using attention implementation: {_attn_impl}")

    # QLoRA + flash_attention_2 can cause illegal memory access on some
    # architectures (e.g. Qwen3.5) due to position ID handling.  Fall back
    # to eager attention when 4-bit quantization is enabled.
    quant_cfg_check = sft_cfg.get("quantization")
    if quant_cfg_check and quant_cfg_check.get("load_in_4bit"):
        _attn_impl = "eager"
        print(f"[sft_training] QLoRA detected — forcing attn_implementation=eager")

    # FlashAttention capability check (rewritten 2026-07-19 — the previous
    # blanket head_dim>=256 -> sdpa downgrade was stale). Per the flash-attn
    # README and probe-verified on klara's A6000s (sm86, FA 2.8.3): head
    # dims up to 256 are supported incl. backward on consumer Ampere since
    # 2.5.5 (given attention_dropout=0), and attn-logit softcap (gemma-4's
    # 30.0) since 2.6.0 — fwd+bwd at head_dim=256/bf16 passes with softcap 0
    # and 30. Downgrade to sdpa only for genuinely unsupported combos; on any
    # probe failure fall back to sdpa (never wrong, only slower — keeping
    # flash on an unknown config is what crashed cells).
    #
    # The effective head dim is the MAX over per-layer-type dims: gemma-4
    # unified models report head_dim=256 (sliding layers) but their
    # full-attention layers use global_head_dim=512, which FA2 cannot run —
    # that (not an old flash-attn build) is what crashed e2b/e4b on
    # 2026-07-15 and 12b on 2026-07-18. Every gemma-4 variant in the zoo
    # carries global_head_dim=512, so the whole family runs sdpa (torch
    # picks a per-call backend that handles 512).
    if _attn_impl == "flash_attention_2":
        try:
            import flash_attn as _fa
            from packaging.version import Version as _V
            from transformers import AutoConfig as _AttnCfg
            _hd_cfg = _AttnCfg.from_pretrained(base_model, trust_remote_code=True)
            _tc = getattr(_hd_cfg, "text_config", _hd_cfg)
            _hd_candidates = [
                getattr(_tc, name, None)
                for name in ("head_dim", "global_head_dim")
            ]
            _hd_candidates = [d for d in _hd_candidates if d]
            _head_dim = max(_hd_candidates) if _hd_candidates else None
            _softcap = getattr(_tc, "attn_logit_softcapping", None)
            _attn_dropout = getattr(_tc, "attention_dropout", 0.0) or 0.0
            _fa_ver = _V(_fa.__version__)
            _reasons = []
            if _head_dim is not None and _head_dim > 256:
                _reasons.append(
                    f"max head_dim={_head_dim} > 256 (FA2 hard limit; "
                    f"candidates={_hd_candidates})")
            elif _head_dim == 256:
                if _fa_ver < _V("2.5.5"):
                    _reasons.append(
                        f"head_dim=256 backward on non-A100/H100 needs "
                        f"flash-attn >= 2.5.5 (have {_fa_ver})")
                if _attn_dropout > 0:
                    _reasons.append(
                        f"head_dim=256 backward on consumer GPUs requires "
                        f"attention_dropout=0 (have {_attn_dropout})")
            if _softcap and _fa_ver < _V("2.6.0"):
                _reasons.append(
                    f"attn_logit_softcapping={_softcap} needs flash-attn "
                    f">= 2.6.0 (have {_fa_ver})")
            if _reasons:
                _attn_impl = "sdpa"
                print(f"[sft_training] Downgrading to sdpa: {'; '.join(_reasons)}")
            elif _head_dim == 256 or _softcap:
                print(f"[sft_training] Keeping flash_attention_2 "
                      f"(head_dim={_head_dim}, softcap={_softcap}, "
                      f"attention_dropout={_attn_dropout}; flash-attn "
                      f"{_fa_ver} supports this combo)")
        except Exception as _e:  # noqa: BLE001
            _attn_impl = "sdpa"
            print(f"[sft_training] FA capability probe failed ({_e}); "
                  "falling back to sdpa")

    model_kwargs = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "attn_implementation": _attn_impl,
        "low_cpu_mem_usage": True,
    }

    # Handle pre-quantized models (e.g. Mxfp4) that don't support training:
    # dequantize on load to bf16, then LoRA attaches normally.  Mxfp4 can't
    # be stacked with BnB QLoRA, so we skip BnB for these models.
    # GPT-OSS-20B dequantized to bf16 + LoRA fits on a 48GB A6000 (~43GB).
    _needs_dequantize = False
    try:
        from transformers import AutoConfig as _AC
        _model_cfg = _AC.from_pretrained(base_model, trust_remote_code=True)
        _qcfg = getattr(_model_cfg, "quantization_config", None)
        if _qcfg is not None:
            _qtype = _qcfg.get("quant_method", "") if isinstance(_qcfg, dict) else getattr(_qcfg, "quant_method", "")
            if _qtype and _qtype.lower() not in ("bitsandbytes", "gptq", "awq"):
                _needs_dequantize = True
    except Exception:
        pass

    if _needs_dequantize:
        from transformers import Mxfp4Config
        model_kwargs["quantization_config"] = Mxfp4Config(dequantize=True)
        # GPT-OSS requires eager attention for training
        model_kwargs["attn_implementation"] = "eager"
        print(f"[sft_training] Model ships pre-quantized — dequantizing to bf16 for training")

    # Optional 4-bit quantization (QLoRA) for large models
    quant_cfg = sft_cfg.get("quantization")
    print(f"[sft_training] DEBUG quantization config: {quant_cfg}, needs_dequantize={_needs_dequantize}")
    if quant_cfg and quant_cfg.get("load_in_4bit") and not _needs_dequantize:
        from transformers import BitsAndBytesConfig
        compute_dtype = getattr(torch, quant_cfg.get("bnb_4bit_compute_dtype", "bfloat16"))
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type=quant_cfg.get("bnb_4bit_quant_type", "nf4"),
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=quant_cfg.get("bnb_4bit_use_double_quant", True),
        )
        # DDP: pin to local GPU. Single-GPU: use "auto" with max_memory so
        # transformers offloads to CPU during loading — the bf16→4bit
        # conversion temporarily needs more VRAM than the final model.
        if is_distributed:
            local_rank = int(os.environ.get("LOCAL_RANK", 0))
            model_kwargs["device_map"] = {"": local_rank}
        else:
            model_kwargs["device_map"] = "auto"
            model_kwargs["max_memory"] = {0: "40GiB", "cpu": "120GiB"}
        print(f"[sft_training] QLoRA: 4-bit quantization enabled (nf4, compute={compute_dtype}, device_map={model_kwargs['device_map']})")

    if is_distributed:
        print(f"[sft_training] DDP mode (LOCAL_RANK={os.environ['LOCAL_RANK']}, {n_gpus} GPUs)")
    else:
        print(f"[sft_training] Single-GPU mode ({n_gpus} GPUs visible)")

    # Load model
    # Phi-4-multimodal-instruct workaround: the model's __init__ internally
    # calls get_peft_model() for vision LoRA, which requires
    # prepare_inputs_for_generation on the inner model.  Since transformers
    # 4.50 removed GenerationMixin from PreTrainedModel, Phi4MMModel no
    # longer has this method.  Patch it in before loading.
    try:
        from transformers import AutoConfig as _ACfg
        _pre_cfg = _ACfg.from_pretrained(base_model, trust_remote_code=True)
        if getattr(_pre_cfg, "model_type", "") == "phi4mm":
            from transformers.dynamic_module_utils import get_class_from_dynamic_module
            _cls_ref = _pre_cfg.auto_map.get("AutoModelForCausalLM", "")
            if _cls_ref:
                # Load the CausalLM class (triggers dynamic module caching),
                # then grab Phi4MMModel from the same module.
                _causal_cls = get_class_from_dynamic_module(
                    _cls_ref, base_model, trust_remote_code=True)
                import sys
                _mod = sys.modules[_causal_cls.__module__]
                _inner_cls = getattr(_mod, "Phi4MMModel", None)
                if _inner_cls is not None and not hasattr(_inner_cls, "prepare_inputs_for_generation"):
                    _inner_cls.prepare_inputs_for_generation = lambda self, *a, **kw: {}
                    print("[sft_training] Patched Phi4MMModel.prepare_inputs_for_generation for PEFT compat")
    except Exception as _e:
        print(f"[sft_training] Phi4MM patch skipped: {_e}")

    print(f"[sft_training] Loading model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(base_model, **model_kwargs)

    # Fix _no_split_modules for FSDP: PEFT's fsdp_auto_wrap_policy uses this
    # to find the transformer layer class. For newer architectures (e.g. Qwen3.5)
    # the inherited value from the parent class may reference a non-existent class.
    # Patch it to match the config if specified.
    fsdp_layer_cls = sft_cfg.get("fsdp", {}).get("transformer_layer_cls_to_wrap")
    if fsdp_layer_cls:
        model._no_split_modules = [fsdp_layer_cls]
        print(f"[sft_training] Patched _no_split_modules -> {model._no_split_modules}")

    # Phi-4-multimodal text-only SFT: the model ships with built-in LoRA
    # adapters (speech, vision) wrapped around every linear layer via PEFT's
    # base_layer mechanism.  PEFT's get_peft_model() cannot add new LoRA on
    # top of existing PEFT-wrapped modules — gradients don't flow (grad_norm=0).
    #
    # Fix: merge the built-in LoRAs into the base weights and unwrap the
    # base_layer structure, giving PEFT clean nn.Linear modules to work with.
    # Also remove unused modality encoders for text-only SFT.
    _is_phi4mm = "phi-4-multimodal" in base_model.lower()
    if _is_phi4mm:
        import gc as _gc

        _ete = model.model.embed_tokens_extend
        # Remove both image and audio encoders (text-only)
        if hasattr(_ete, "audio_embed"):
            del _ete.audio_embed
            print("[sft_training] Phi-4-MM: deleted audio_embed")
        if hasattr(_ete, "image_embed"):
            del _ete.image_embed
            print("[sft_training] Phi-4-MM: deleted image_embed")

        # Merge built-in LoRA adapters into base weights and unwrap.
        # The model's linear layers are PEFT LoraLayer instances with
        # base_layer + lora_A/lora_B.  We need to:
        # 1. Compute merged weight: W = base_layer.weight + sum(B @ A * scaling)
        # 2. Replace the LoraLayer with a plain nn.Linear using the merged weight.
        _n_unwrapped = 0
        for _layer in model.model.layers:
            for _parent_name in ("self_attn", "mlp"):
                _parent = getattr(_layer, _parent_name, None)
                if _parent is None:
                    continue
                for _proj_name in list(vars(_parent).keys()):
                    _proj = getattr(_parent, _proj_name, None)
                    if _proj is None or not hasattr(_proj, "base_layer"):
                        continue
                    # Merge: start with base weight
                    _base = _proj.base_layer
                    _merged_w = _base.weight.data.clone()
                    _has_bias = _base.bias is not None
                    _merged_b = _base.bias.data.clone() if _has_bias else None
                    # Add each adapter's contribution: B @ A * scaling
                    if hasattr(_proj, "lora_A") and hasattr(_proj, "lora_B"):
                        _scaling = getattr(_proj, "scaling", {})
                        for _adapter_name in list(getattr(_proj.lora_A, "_modules", {}).keys()):
                            _a = _proj.lora_A[_adapter_name].weight.data
                            _b = _proj.lora_B[_adapter_name].weight.data
                            _s = _scaling.get(_adapter_name, 1.0)
                            _merged_w += (_b @ _a) * _s
                    # Create clean linear layer
                    _new = torch.nn.Linear(
                        _merged_w.shape[1], _merged_w.shape[0],
                        bias=_has_bias, dtype=_merged_w.dtype, device=_merged_w.device,
                    )
                    _new.weight.data.copy_(_merged_w)
                    if _has_bias:
                        _new.bias.data.copy_(_merged_b)
                    setattr(_parent, _proj_name, _new)
                    _n_unwrapped += 1

        print(f"[sft_training] Phi-4-MM: merged + unwrapped {_n_unwrapped} LoRA layers to plain nn.Linear")

        # Patch embed_tokens_extend to skip the training assert for text-only
        _orig_ete_forward = _ete.forward
        def _patched_ete_forward(*args, **kwargs):
            was_training = _ete.training
            _ete.training = False
            try:
                return _orig_ete_forward(*args, **kwargs)
            finally:
                _ete.training = was_training
        _ete.forward = _patched_ete_forward
        _gc.collect()
        torch.cuda.empty_cache()
        print("[sft_training] Phi-4-MM: text-only cleanup complete")

    # Configure SFT training
    gc_kwargs = sft_cfg.get("gradient_checkpointing_kwargs", {"use_reentrant": False})
    # FSDP for models that need sharding across GPUs (dequantized large models).
    # DDP replicates the full model on each GPU; FSDP shards it.
    fsdp_cfg = sft_cfg.get("fsdp")
    fsdp_kwargs = {}
    if fsdp_cfg:
        fsdp_kwargs["fsdp"] = fsdp_cfg.get("strategy", "full_shard auto_wrap")
        fsdp_config = {
            "auto_wrap_policy": fsdp_cfg.get("auto_wrap_policy", "TRANSFORMER_BASED_WRAP"),
            "backward_prefetch": fsdp_cfg.get("backward_prefetch", "BACKWARD_PRE"),
            "forward_prefetch": fsdp_cfg.get("forward_prefetch", False),
            "cpu_ram_efficient_loading": fsdp_cfg.get("cpu_ram_efficient_loading", True),
        }
        if fsdp_cfg.get("transformer_layer_cls_to_wrap"):
            fsdp_config["transformer_layer_cls_to_wrap"] = fsdp_cfg["transformer_layer_cls_to_wrap"]
            # PEFT's fsdp_auto_wrap_policy reads this env var to find the layer
            # class, overriding whatever SFTConfig sets. Set it explicitly so
            # models not in PEFT's auto-detection list (e.g. Qwen3.5) work.
            os.environ["FSDP_TRANSFORMER_CLS_TO_WRAP"] = fsdp_cfg["transformer_layer_cls_to_wrap"]
        fsdp_kwargs["fsdp_config"] = fsdp_config
        print(f"[sft_training] FSDP enabled: {fsdp_kwargs['fsdp']}")

    # Loss type: "dft" (Dynamic Fine-Tuning, arXiv:2508.05629) rescales each
    # token's NLL by its detached probability; TRL applies it via an internal
    # compute_loss_func, so it composes with assistant_only_loss masking and
    # gradient accumulation. When unset, TRL's own default applies (chunked_nll
    # on the non-Liger path in 1.8.0) — i.e. the stock SFT used before
    # 2026-07-18. NOTE: with dft, eval_loss is DFT-weighted and NOT comparable
    # to nll-scale eval curves from earlier runs.
    loss_type = sft_cfg.get("loss_type")
    loss_kwargs = {"loss_type": str(loss_type)} if loss_type else {}
    if loss_type:
        print(f"[sft_training] Loss type: {loss_type}"
              + (" — eval_loss will be DFT-weighted (not comparable to prior "
                 "nll-scale eval losses)" if str(loss_type) == "dft" else ""))

    training_args = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=sft_cfg.get("num_epochs", 3),
        per_device_train_batch_size=sft_cfg.get("per_device_batch_size", 4),
        gradient_accumulation_steps=sft_cfg.get("gradient_accumulation_steps", 4),
        learning_rate=sft_cfg.get("learning_rate", 2e-5),
        warmup_ratio=sft_cfg.get("warmup_ratio", 0.1),
        weight_decay=sft_cfg.get("weight_decay", 0.01),
        max_length=sft_cfg.get("max_seq_length", 8192),
        gradient_checkpointing=sft_cfg.get("gradient_checkpointing", True),
        gradient_checkpointing_kwargs=gc_kwargs,
        max_grad_norm=sft_cfg.get("max_grad_norm", 1.0),
        bf16=sft_cfg.get("bf16", True),
        logging_steps=sft_cfg.get("logging_steps", 10),
        save_strategy=sft_cfg.get("save_strategy", "epoch"),
        # Evaluate only when a held-out set exists; "no" otherwise, since HF
        # errors if an eval strategy is set without an eval_dataset.
        eval_strategy=(sft_cfg.get("eval_strategy", "epoch")
                       if eval_dataset is not None else "no"),
        eval_steps=sft_cfg.get("eval_steps"),
        per_device_eval_batch_size=sft_cfg.get(
            "per_device_eval_batch_size",
            sft_cfg.get("per_device_batch_size", 4)),
        # Best-epoch selection (added 2026-07-18): with a held-out set, reload
        # the checkpoint with the lowest eval_loss at train end, so
        # trainer.save_model() persists the BEST epoch rather than the last.
        # Under loss_type=dft the metric is DFT-weighted — fine for WITHIN-run
        # ranking, do not compare across loss types. save_total_limit keeps all
        # epoch checkpoints by default (best is always protected by HF).
        load_best_model_at_end=(
            eval_dataset is not None
            and bool(sft_cfg.get("select_best_epoch", True))),
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_total_limit=sft_cfg.get("save_total_limit"),
        report_to="wandb" if OmegaConf.select(cfg, "wandb.enabled") else "none",
        remove_unused_columns=False,
        dataloader_pin_memory=True,
        dataloader_num_workers=sft_cfg.get("dataloader_num_workers", 4),
        ddp_find_unused_parameters=False,
        # Enable loss masking via {% generation %} blocks in the chat template.
        # Without this, SFTTrainer trains on all tokens (prompt + completion).
        assistant_only_loss=True,
        **loss_kwargs,
        **fsdp_kwargs,
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # SFTTrainer needs {% generation %} blocks in the chat template so it
    # knows which tokens are assistant completions (and should contribute to
    # loss). Without this, all labels become -100 and training produces zero
    # loss / NaN gradients.
    #
    # Prefer the model's native template if it already includes generation
    # blocks (e.g. Qwen3+, SmolLM3). Fall back to our manual templates for
    # architectures that don't ship them yet.
    native_template = tokenizer.chat_template or ""
    has_native_generation = "{% generation %}" in native_template

    # Resolve thinking mode from cfg.model (single source of truth). For SFT
    # this controls whether the Qwen manual template injects the empty-think
    # sentinel prefix. For the native-template path, we simply log the mode
    # so downstream GRPO/eval can verify alignment.
    from dagspaces.common.stage_utils import resolve_thinking_mode
    model_cfg = getattr(cfg, "model", None) or {}
    _thinking_enabled_sft = resolve_thinking_mode(model_cfg, default=False)
    print(f"[sft_training] Thinking mode: "
          f"{'on' if _thinking_enabled_sft else 'off'} "
          f"(from cfg.model.thinking_mode or chat_template_kwargs.enable_thinking)")

    family = _detect_template_family(tokenizer, base_model)
    if has_native_generation:
        print(f"[sft_training] Chat template: native (family={family}, has {{% generation %}} blocks)")
    else:
        # ── Template-selection order (reworked 2026-07-18) ──────────────────
        # 1. TRL registry: trl.get_training_chat_template byte-matches the
        #    tokenizer's NATIVE template against TRL's known set and returns
        #    the official render + generation markers. This is the preferred
        #    path — it reproduces the exact serve-time bytes (llama's date
        #    preamble, qwen3.5's think handling, gpt-oss's harmony
        #    system/developer scaffold) and trains the stop token.
        # 2. Manual _SFT_TEMPLATES fallback for families TRL doesn't cover
        #    (gemma-4, phi-4 — both byte-verified against the official repos
        #    2026-07-18).
        # 3. gpt-oss with no registry match is a hard error — the old manual
        #    harmony template corrupted training (see _SFT_TEMPLATES comment).
        trl_template = None
        try:
            from trl import get_training_chat_template
            trl_template = get_training_chat_template(tokenizer)
        except Exception as _e:  # ValueError: template not in TRL's registry
            print(f"[sft_training] TRL registry: no training-template match "
                  f"({type(_e).__name__}: {_e}) — falling back to manual template")

        if trl_template:
            tokenizer.chat_template = trl_template
            print(f"[sft_training] Chat template: TRL registry training "
                  f"template (family={family}) — official native render with "
                  "{% generation %} markers")
        elif family == "gpt-oss":
            raise ValueError(
                "gpt-oss SFT requires TRL's registry harmony training template, "
                "but get_training_chat_template found no match for this "
                "tokenizer's chat template. Refusing to fall back: the "
                "pre-2026-07-18 manual harmony template corrupted training "
                "(wrong role scaffold, <|end|> instead of <|return|>). "
                "Check that the checkpoint ships the official "
                "openai/gpt-oss-20b chat_template.jinja."
            )
        else:
            if family == "qwen":
                sft_template = _qwen_sft_template(thinking_enabled=_thinking_enabled_sft)
            else:
                sft_template = _SFT_TEMPLATES[family]
            # Preflight: the chosen family's control tokens must exist
            # atomically in THIS model's vocab (catches wrong-family templates
            # at job start — the 2026-07-15 gemma-4 incident class).
            _assert_template_tokens_atomic(tokenizer, family)
            tokenizer.chat_template = sft_template
            print(f"[sft_training] Chat template: manual override "
                  f"(family={family}, thinking={'on' if _thinking_enabled_sft else 'off'})")

    # Trace logging: log diagnostics ~10 times during training.
    trace_log_path = os.path.join(output_dir, "sft_traces.jsonl")
    total_steps = int(
        (len(dataset) * sft_cfg.get("num_epochs", 3))
        / max(sft_cfg.get("per_device_batch_size", 4) * sft_cfg.get("gradient_accumulation_steps", 4), 1)
    )
    trace_every = max(total_steps // 10, 1)

    print(f"[sft_training] Starting SFT with LoRA (r={peft_config.r}, alpha={peft_config.lora_alpha})")

    # Gemma 3 requires token_type_ids during training (used for text vs image
    # token masking in the bidirectional attention). For text-only SFT, these
    # are all zeros.  Wrap the default collator to inject them.
    # Applied to gemma-4 as well: the 2026-07-15 canonical sweep ran gemma-4
    # through this collator (it was classified as family="gemma" then) and
    # trained without error, so keeping it preserves known-good behaviour while
    # the 2026-07-18 fix changes only the chat template.
    data_collator = None
    if family in ("gemma", "gemma-4"):
        import torch

        class _GemmaCollator:
            """Injects zero-valued token_type_ids for Gemma 3 text-only training."""
            def __init__(self, inner):
                self.inner = inner
            def __call__(self, features):
                batch = self.inner(features)
                if "token_type_ids" not in batch and "input_ids" in batch:
                    batch["token_type_ids"] = torch.zeros_like(batch["input_ids"])
                return batch

        print("[sft_training] Gemma 3: injecting token_type_ids via custom data collator")

    if family == "phi-4-mm":
        import torch

        class _Phi4MMCollator:
            """Injects input_mode=0 (LANGUAGE) for Phi-4-multimodal text-only SFT."""
            def __init__(self, inner):
                self.inner = inner
            def __call__(self, features):
                batch = self.inner(features)
                if "input_mode" not in batch:
                    bs = batch["input_ids"].shape[0]
                    batch["input_mode"] = torch.zeros(bs, dtype=torch.long, device=batch["input_ids"].device)
                return batch

        print("[sft_training] Phi-4-MM: text-only SFT setup")

    callbacks = [_make_trace_callback(trace_log_path, trace_every, tokenizer)]
    # Optional early stopping: stop after N consecutive evals without
    # eval_loss improvement. Requires the held-out set. With epoch-level eval
    # and 3 epochs, patience=1 saves up to a third of the compute when epoch 2
    # already overfits. Off by default (null) — best-epoch selection alone
    # already discards overfit epochs without changing the training trajectory.
    early_stop_patience = sft_cfg.get("early_stopping_patience")
    if early_stop_patience and eval_dataset is not None:
        from transformers import EarlyStoppingCallback
        callbacks.append(EarlyStoppingCallback(
            early_stopping_patience=int(early_stop_patience)))
        print(f"[sft_training] Early stopping: patience={early_stop_patience} "
              "eval rounds on eval_loss")

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
        callbacks=callbacks,
    )

    # Wrap trainer's collator after init (SFTTrainer builds its own internally)
    if family in ("gemma", "gemma-4"):
        trainer.data_collator = _GemmaCollator(trainer.data_collator)
    if family == "phi-4-mm":
        trainer.data_collator = _Phi4MMCollator(trainer.data_collator)

    # Log init trace (LoRA resolved modules, param counts, config).
    # Rank-0 only under DDP/FSDP — same rationale as the trace callback.
    _is_main_rank = int(os.environ.get("RANK", 0) or 0) == 0
    if _is_main_rank:
        _log_init_trace(trace_log_path, model, peft_config, base_model, family, len(dataset), sft_cfg)
        print(f"[sft_training] Traces -> {trace_log_path} (every {trace_every} steps)")

    # Validate tokenization and label masking on first batch + log samples
    sample = next(iter(trainer.get_train_dataloader()))
    labels = sample["labels"][0]
    n_train = (labels != -100).sum().item()
    n_masked = (labels == -100).sum().item()
    print(f"[sft_training] Label check: {n_train} train tokens, {n_masked} masked tokens")
    if n_train == 0:
        raise ValueError(
            "All labels are -100 — chat template masking is broken. "
            "Ensure the tokenizer chat_template includes {% generation %} blocks."
        )
    if n_masked == 0:
        raise ValueError(
            "No labels are masked — loss is computed on the entire sequence "
            "(prompt + completion). Ensure assistant_only_loss=True is set in "
            "SFTConfig and the chat template includes {% generation %} blocks."
        )
    if _is_main_rank:
        _log_tokenization_samples(trace_log_path, trainer, tokenizer, n_samples=3)

    trainer.train()

    if training_args.load_best_model_at_end:
        best_ckpt = trainer.state.best_model_checkpoint
        best_metric = trainer.state.best_metric
        print(f"[sft_training] Best-epoch selection: saving weights from "
              f"{best_ckpt or 'final state'} (eval_loss="
              f"{best_metric if best_metric is not None else 'n/a'})")

    # Save final checkpoint.
    # Restore the original chat template before saving — the {% generation %}
    # blocks are only needed during training and must NOT persist to checkpoints.
    original_template = AutoTokenizer.from_pretrained(
        base_model, trust_remote_code=True
    ).chat_template
    if original_template:
        tokenizer.chat_template = original_template

    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[sft_training] Saved LoRA checkpoint to {output_dir}")
