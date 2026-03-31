# Vendored from CI-RL (github.com/EricGLan/CI-RL) - Apache 2.0 License
# Minimal subset: only what tools/ and prompts/ actually need at runtime.
from .tool import (
    ArgException,
    ArgParameter,
    ArgReturn,
    create_str,
    insert_indent,
    load_dict,
    validate_inputs,
)
from .my_typing import *


def get_num_tokens(prompt: str, encoding: str = "cl100k_base") -> int:
    """Token counter (inlined to avoid rouge_score dependency in misc.py)."""
    try:
        import tiktoken
        enc = tiktoken.get_encoding(encoding)
        return len(enc.encode(prompt))
    except ImportError:
        return len(prompt.split())


def make_colorful(role: str, text: str) -> str:
    """Stub — only used in __main__ blocks we never run."""
    return text
