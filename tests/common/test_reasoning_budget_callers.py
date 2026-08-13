"""Guard: every stage must size its token budget with the SHARED predicate.

The defect this prevents
------------------------
``model_needs_reasoning_budget`` has TWO triggers:

  1. ``chat_template_kwargs.enable_thinking`` is explicitly ``False``;
  2. the model reasons STRUCTURALLY — it ships a vLLM reasoning parser
     (qwen3, deepseek-r1) or speaks harmony (gpt-oss).

Five stages hand-rolled an inline copy that had only trigger 1. gpt-oss and
openthinker3 carry a bare ``chat_template_kwargs: {}``, so those stages kept
the SMALL token budget for exactly the models that need the large one.

Measured, not theorised. On the 2026-07-17 canonical instruct run, GoldCoin
compliance (max_tokens 1024 instead of 4096):

    gpt-oss-20b      24/107 rows finish_reason=length, 17 rows EMPTY
    openthinker3-7b  12/107 rows finish_reason=length

Both models are rows in the camera-ready benchmark table.

``strips_think_blocks`` answers a DIFFERENT question — trigger 1 only — and is
the right predicate for deciding whether to strip ``<think>`` blocks. Using it
to size a budget reintroduces the defect; using the budget predicate to decide
stripping strips output that was never wrapped.
"""

from __future__ import annotations

import inspect
import re

import pytest
from omegaconf import OmegaConf

from dagspaces.common.reasoning import (
    model_needs_reasoning_budget,
    strips_think_blocks,
)

# Models whose configs make the two predicates DISAGREE. These are the models
# the hand-rolled copies got wrong.
STRUCTURAL_REASONERS = ["gpt-oss-20b/instruct", "openthinker3-7b/instruct"]


def _model_cfg(name: str):
    return OmegaConf.load(f"dagspaces/common/conf/model/{name}.yaml").model


@pytest.mark.parametrize("name", STRUCTURAL_REASONERS)
def test_structural_reasoners_need_a_budget_but_do_not_strip(name):
    """The exact configuration the hand-rolled check mishandled."""
    cfg = _model_cfg(name)
    assert model_needs_reasoning_budget(cfg) is True, (
        f"{name} reasons structurally and must get the large token budget"
    )
    assert strips_think_blocks(cfg) is False, (
        f"{name} does not set enable_thinking=false, so nothing is stripped. "
        f"This is the disagreement that makes the two predicates distinct."
    )


def test_an_explicit_enable_thinking_false_triggers_both():
    cfg = OmegaConf.create({"chat_template_kwargs": {"enable_thinking": False}})
    assert strips_think_blocks(cfg) is True
    assert model_needs_reasoning_budget(cfg) is True


def test_a_plain_model_triggers_neither():
    cfg = OmegaConf.create({
        "model_source": "/models/Llama-3.1-8B-Instruct",
        "model_family": "llama3.1",
        "chat_template_kwargs": {},
    })
    assert strips_think_blocks(cfg) is False
    assert model_needs_reasoning_budget(cfg) is False


def test_strips_think_blocks_reports_and_degrades(capsys, monkeypatch):
    """A bad config must not crash a stage, and must not go unmentioned."""
    from dagspaces.common import reasoning as R

    monkeypatch.setattr(R, "_STRIP_PREDICATE_WARNED", False)

    class _Exploding:
        @property
        def enable_thinking(self):
            raise RuntimeError("unresolvable interpolation")

    class _Cfg:
        chat_template_kwargs = _Exploding()

    assert R.strips_think_blocks(_Cfg()) is False   # must NOT raise
    err = capsys.readouterr().err
    assert "enable_thinking" in err and "RuntimeError" in err


# --------------------------------------------------------------------------
# No stage may hand-roll the check again
# --------------------------------------------------------------------------

BUDGET_STAGES = [
    "dagspaces/goldcoin_hipaa/stages/llm_inference.py",
    "dagspaces/simpleqa_verified/stages/llm_inference.py",
    "dagspaces/privacylens/stages/llm_inference.py",
    "dagspaces/confaide/stages/llm_inference.py",
    "dagspaces/cirl/stages/llm_inference.py",
    "dagspaces/mmlu/stages/llm_inference.py",
]

STRIPPING_MODULES = [
    "dagspaces/vlm_geoprivacy_bench/vlm_inference.py",
    "dagspaces/vlm_geoprivacy_aug/vlm_inference.py",
]

# The inline pattern that was copy-pasted into five modules.
_HAND_ROLLED = re.compile(
    r"=\s*not\s+bool\(\s*ctk[\.\[]", re.MULTILINE
)


@pytest.mark.parametrize("path", BUDGET_STAGES + STRIPPING_MODULES,
                         ids=[p.split("/")[1] + ":" + p.split("/")[-1]
                              for p in BUDGET_STAGES + STRIPPING_MODULES])
def test_no_module_hand_rolls_the_thinking_check(path):
    src = open(path).read()
    assert not _HAND_ROLLED.search(src), (
        f"{path} reads chat_template_kwargs.enable_thinking inline again. Use "
        f"model_needs_reasoning_budget() to size a token budget, or "
        f"strips_think_blocks() to decide stripping. The inline copy has only "
        f"one of the two triggers — see this module's docstring."
    )


@pytest.mark.parametrize("path", BUDGET_STAGES,
                         ids=[p.split("/")[1] for p in BUDGET_STAGES])
def test_budget_stages_use_the_budget_predicate(path):
    src = open(path).read()
    assert "model_needs_reasoning_budget" in src, (
        f"{path} sizes a token budget, so it must call "
        f"model_needs_reasoning_budget()"
    )


def test_privacylens_uses_one_rule_for_both_of_its_stages():
    """It used the hand-rolled check for the QA probe and the shared helper
    for the action stage — two rules in one file, so the QA probe
    under-budgeted gpt-oss while the action stage did not."""
    src = open("dagspaces/privacylens/stages/llm_inference.py").read()
    assert src.count("model_needs_reasoning_budget(") >= 2, (
        "both the QA probe and the action stage must size their budgets the "
        "same way"
    )


def test_the_two_predicates_are_documented_as_distinct():
    """A reader must be told not to swap them."""
    strip_doc = inspect.getdoc(strips_think_blocks) or ""
    budget_doc = inspect.getdoc(model_needs_reasoning_budget) or ""
    assert "model_needs_reasoning_budget" in strip_doc
    assert "budget" in strip_doc.lower()
    assert budget_doc.strip(), "the budget predicate must keep its rationale"
