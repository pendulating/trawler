"""SFT chat-template regression tests.

Guards the 2026-07-18 bug: `_detect_template_family` matched the bare substring
"gemma" and handed every Gemma-4 model the **Gemma-3** template. The two share
no control tokens — `<start_of_turn>` is not in the Gemma-4 vocabulary at all
and tokenizes into seven arbitrary sub-word pieces — so all three gemma-4 cells
of the 2026-07-15 canonical SFT sweep trained with garbage turn delimiters
(gemma-4-12b: initial loss 3.12 vs ~0.9 elsewhere, median grad-norm 6.15 vs
0.54, 31/54 steps clipped) and were then served under the native template.

These tests are deliberately weight-free: they use a stub tokenizer so they run
in the normal suite without model files on disk.
"""

import pytest

from dagspaces.grpo_training.stages.sft_training import (
    _SFT_TEMPLATES,
    _detect_template_family,
    _qwen_sft_template,
)


class _StubTokenizer:
    """Minimal stand-in exposing only what _detect_template_family reads."""

    def __init__(self, vocab_tokens=(), chat_template=""):
        self._vocab = {t: i for i, t in enumerate(vocab_tokens)}
        self.chat_template = chat_template

    def get_vocab(self):
        return self._vocab


GEMMA4_TOKENS = ("<|turn>", "<turn|>", "<|channel>", "<channel|>")
GEMMA3_TOKENS = ("<start_of_turn>", "<end_of_turn>")


@pytest.mark.parametrize(
    "path",
    [
        "/zoo/models/gemma-4-12B-it",
        "/zoo/models/Gemma-4-E2B-it",
        "/zoo/models/Gemma-4-E4B-it",
        "/zoo/models/Gemma-4-31B-it",
    ],
)
def test_gemma4_never_gets_the_gemma3_template(path):
    """The core regression: gemma-4 must resolve to its own family."""
    tok = _StubTokenizer(vocab_tokens=GEMMA4_TOKENS)
    assert _detect_template_family(tok, path) == "gemma-4"


def test_gemma3_still_resolves_to_gemma():
    """The fix must not push Gemma-3 onto the Gemma-4 template."""
    tok = _StubTokenizer(vocab_tokens=GEMMA3_TOKENS)
    assert _detect_template_family(tok, "/zoo/models/gemma-3-12b-it") == "gemma"


def test_vocab_wins_over_path_for_renamed_checkpoints():
    """A renamed/symlinked directory must not decide the control tokens."""
    g4 = _StubTokenizer(vocab_tokens=GEMMA4_TOKENS)
    # Path says gemma-3, vocab says gemma-4 -> trust the vocab.
    assert _detect_template_family(g4, "/zoo/models/gemma-3-mystery") == "gemma-4"
    g3 = _StubTokenizer(vocab_tokens=GEMMA3_TOKENS)
    assert _detect_template_family(g3, "/zoo/models/gemma-4-mislabelled") == "gemma"


def test_gemma4_template_uses_only_gemma4_delimiters():
    """The gemma-4 template must not contain any Gemma-3 control token."""
    tpl = _SFT_TEMPLATES["gemma-4"]
    for forbidden in GEMMA3_TOKENS:
        assert forbidden not in tpl, f"gemma-4 template leaks {forbidden}"
    for required in ("<|turn>", "<turn|>"):
        assert required in tpl


def test_gemma4_template_does_not_strip_turn_newlines():
    """Turn delimiters must be followed by a literal newline.

    The first attempt at this template used `{%-` tags, whose whitespace
    stripping silently ate the "\\n" after every `<turn|>` and produced
    `...U<turn|><|turn>model...` instead of the native
    `...U<turn|>\\n<|turn>model...`. Rendering equality against the real
    checkpoints is verified out-of-band; here we just pin the property that
    made it fail, so a future edit reintroducing `{%-` is caught.
    """
    tpl = _SFT_TEMPLATES["gemma-4"]
    assert "<turn|>\n" in tpl, "turn-end must be followed by a newline"
    assert "{%-" not in tpl, (
        "whitespace-stripping tags eat the newline after <turn|>; "
        "use non-stripping {% %} in this template"
    )


def test_every_template_has_generation_blocks():
    """assistant_only_loss=True requires {% generation %} in every template."""
    for family, tpl in _SFT_TEMPLATES.items():
        assert "{% generation %}" in tpl, f"{family} lacks a generation block"
        assert "{% endgeneration %}" in tpl, f"{family} lacks endgeneration"


# End-of-turn terminator for each manual family. Must appear INSIDE the
# {% generation %} block (i.e. before {% endgeneration %} within the assistant
# branch): until 2026-07-18 every manual template left it outside, so
# assistant_only_loss masked it to -100 and the model was never trained to
# emit its own stop token. TRL's registry training templates include it;
# is_chat_template_stop_token_trained exists to warn about exactly this.
_FAMILY_TERMINATORS = {
    "qwen": "<|im_end|>",
    "phi-4": "<|im_end|>",
    "phi-4-mm": "<|end|>",
    "gemma-4": "<turn|>",
    "gemma": "<end_of_turn>",
    "llama": "<|eot_id|>",
}


def test_terminator_inside_generation_block():
    """The stop token must be part of the loss span in every manual template."""
    for family, tpl in _SFT_TEMPLATES.items():
        terminator = _FAMILY_TERMINATORS[family]
        gen_start = tpl.index("{% generation %}")
        gen_end = tpl.index("{% endgeneration %}")
        span = tpl[gen_start:gen_end]
        assert terminator in span, (
            f"{family}: terminator {terminator!r} is outside the "
            "{% generation %} block — the model would never be trained to "
            "emit its stop token"
        )


# Expected renders, byte-verified against the official templates 2026-07-18
# (gemma-4: google/gemma-4-*-it repos; phi-4: microsoft/phi-4; llama: Meta's
# prompt-format doc modulo the date preamble, which is masked prompt-side;
# qwen: Qwen3.5 chat_template.jinja incl. the two-newline empty-think
# sentinel). Rendering uses plain jinja2 with the generation markers stripped —
# faithful here because these concatenated-string templates contain no
# incidental whitespace, so env flags (trim_blocks etc.) have nothing to act
# on; whitespace behavior is decided entirely by `{%` vs `{%-` tags.
_MSGS = [
    {"role": "system", "content": "S"},
    {"role": "user", "content": "U"},
    {"role": "assistant", "content": "A"},
]

_EXPECTED_RENDERS = {
    "qwen": (
        "<|im_start|>system\nS<|im_end|>\n<|im_start|>user\nU<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nA<|im_end|>\n"
    ),
    "phi-4": (
        "<|im_start|>system<|im_sep|>S<|im_end|><|im_start|>user<|im_sep|>U"
        "<|im_end|><|im_start|>assistant<|im_sep|>A<|im_end|>"
    ),
    "phi-4-mm": "<|system|>S<|end|><|user|>U<|end|><|assistant|>A<|end|>",
    "gemma-4": (
        "<bos><|turn>system\nS<turn|>\n<|turn>user\nU<turn|>\n"
        "<|turn>model\nA<turn|>\n"
    ),
    "gemma": (
        "<bos><start_of_turn>user\nS\n\nU<end_of_turn>\n"
        "<start_of_turn>model\nA<end_of_turn>\n"
    ),
    "llama": (
        "<bos><|start_header_id|>system<|end_header_id|>\n\nS<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\nU<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\nA<|eot_id|>"
    ),
}


def _render(tpl, messages=_MSGS):
    from jinja2 import Template
    stripped = tpl.replace("{% generation %}", "").replace("{% endgeneration %}", "")
    return Template(stripped).render(
        messages=messages, bos_token="<bos>", add_generation_prompt=False)


@pytest.mark.parametrize("family", sorted(_EXPECTED_RENDERS))
def test_manual_template_renders_expected_bytes(family):
    """Every manual template must render the verified native byte form.

    Guards the 2026-07-18 whitespace fixes: `{%-` tags ate the turn-separator
    newline after every `<|im_end|>` (all pre-2026-07-18 qwen SFT runs trained
    with glued turns), and the old Qwen sentinel had one trailing newline where
    the official Qwen3/3.5 no-think form has two.
    """
    assert _render(_SFT_TEMPLATES[family]) == _EXPECTED_RENDERS[family]


def test_qwen_template_no_system_and_thinking_variants():
    """The actual SFT data shape (user+assistant, no system turn)."""
    no_sys = _MSGS[1:]
    assert _render(_SFT_TEMPLATES["qwen"], no_sys) == (
        "<|im_start|>user\nU<|im_end|>\n"
        "<|im_start|>assistant\n<think>\n\n</think>\n\nA<|im_end|>\n"
    )
    # thinking-on variant: no sentinel injected
    assert _render(_qwen_sft_template(thinking_enabled=True)) == (
        _EXPECTED_RENDERS["qwen"].replace("<think>\n\n</think>\n\n", "")
    )


def test_gemma3_system_fold_survives_the_loop():
    """namespace() fix: a [system, user] conversation must fold the system
    message into the first user turn (plain `{% set %}` in a jinja loop does
    not persist across iterations, and the old `loop.first` check never fired
    for [system, user, ...] anyway)."""
    render = _render(_SFT_TEMPLATES["gemma"], _MSGS[:2])
    assert "S\n\nU" in render


def test_no_whitespace_stripping_tags_in_newline_templates():
    """Templates whose native form has inter-turn newlines must not use
    `{%-` (it strips the preceding literal newline)."""
    for family in ("qwen", "gemma", "gemma-4"):
        assert "{%-" not in _SFT_TEMPLATES[family], (
            f"{family}: whitespace-stripping tags eat inter-turn newlines"
        )


def test_gpt_oss_has_no_manual_template():
    """gpt-oss must go through TRL's registry harmony training template.

    The pre-2026-07-18 manual template corrupted harmony (instructions in the
    `system` role instead of `developer`, missing the mandatory system meta
    preamble, `<|end|>` instead of `<|return|>` on the final turn) — the
    plausible root cause of the 33% empty-final-channel regression. The stage
    raises rather than falling back for this family.
    """
    assert "gpt-oss" not in _SFT_TEMPLATES
