"""Per-model-family prompt formatters for VLM inference.

Each builder takes (model_source, sys_msg, user_msg_parts) and returns a
prompt string formatted for that model family's expected chat template.
"""

from __future__ import annotations

from typing import Callable, Dict, List


def build_qwen2_5_vl_prompt(model_source: str, sys_msg: str, usr_msgs: List[str]) -> str:
    """Qwen2.5-VL and Qwen3-VL chat template (manual)."""
    return (
        f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{''.join(usr_msgs)}<|im_end|>\n"
        "<|im_start|>assistant\n"
    )


def build_qwen3_5_prompt(model_source: str, sys_msg: str, usr_msgs: List[str]) -> str:
    """Qwen3.5 VLM chat template with thinking disabled."""
    return (
        f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
        f"<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{''.join(usr_msgs)}<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )


def build_llama_vision_prompt(model_source: str, sys_msg: str, usr_msgs: List[str]) -> str:
    """Llama 3.2 Vision — uses AutoTokenizer.apply_chat_template."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_source, use_fast=True)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_msg}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "".join(usr_msgs)}]},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_gemma3_prompt(model_source: str, sys_msg: str, usr_msgs: List[str]) -> str:
    """Gemma-3 — uses AutoProcessor.apply_chat_template."""
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_source, use_fast=True)
    messages = [
        {"role": "system", "content": [{"type": "text", "text": sys_msg}]},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "".join(usr_msgs)}]},
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_internvl2_5_prompt(model_source: str, sys_msg: str, usr_msgs: List[str]) -> str:
    """InternVL2.5 — uses AutoTokenizer with <image> prefix."""
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)
    messages = [
        {"role": "system", "content": sys_msg},
        {"role": "user", "content": f'<image>\n{"".join(usr_msgs)}'},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def build_deepseek_vl2_prompt(model_source: str, sys_msg: str, usr_msgs: List[str]) -> str:
    """DeepSeek-VL2 manual template."""
    return (
        f"<|System|>:{sys_msg}\n\n<|User|>: <image>\n"
        + "".join([f"<|User|>:{msg}" for msg in usr_msgs])
        + "\n\n<|Assistant|>:"
    )


def build_phi4mm_prompt(model_source: str, sys_msg: str, usr_msgs: List[str]) -> str:
    """Phi-4-multimodal-instruct — uses <|image_1|> token."""
    return (
        f"<|system|>{sys_msg}<|end|>"
        f"<|user|><|image_1|>\n{''.join(usr_msgs)}<|end|>"
        "<|assistant|>"
    )


# Registry: model_family -> prompt builder
PROMPT_BUILDERS: Dict[str, Callable[[str, str, List[str]], str]] = {
    "qwen2.5-vl": build_qwen2_5_vl_prompt,
    "qwen3-vl": build_qwen2_5_vl_prompt,  # Same chat template
    "qwen3.5": build_qwen3_5_prompt,
    "llama-vision": build_llama_vision_prompt,
    "gemma-3": build_gemma3_prompt,
    "internvl2.5": build_internvl2_5_prompt,
    "deepseek-vl2": build_deepseek_vl2_prompt,
    "phi-4": build_phi4mm_prompt,
}


def get_prompt_builder(model_family: str) -> Callable[[str, str, List[str]], str]:
    """Look up the prompt builder for a given model family."""
    builder = PROMPT_BUILDERS.get(model_family)
    if builder is None:
        raise ValueError(
            f"Unknown model_family '{model_family}'. "
            f"Available: {list(PROMPT_BUILDERS.keys())}"
        )
    return builder
