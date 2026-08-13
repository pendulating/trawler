"""Guard: vlm_geoprivacy_aug must SHARE benchmark code, not copy it.

Why this test exists
--------------------
``vlm_geoprivacy_aug`` began as a file-level copy of ``vlm_geoprivacy_bench``.
The copy then fell behind, and nobody saw it, because a copy gives no signal.
A review on 2026-08-12 found three benchmark corrections that never reached
this dagspace:

* 2026-07-14 (21838ab) — the Q7 judge label parser. The copy took the first
  character in {A, B, C, D}, which is the ``a`` of "answer" in any verbose
  completion, so the row got the abstention class. A completion with no such
  character got the default ``D``, a real class.
* 2026-07-18 — ``build_gemma4_prompt``. Without it, and without a "gemma-4"
  registry key, get_prompt_builder raised "Unknown model_family 'gemma-4'":
  gemma-4 could not run on this dagspace at all.
* 2026-07-21 — the accuracy denominator, from the parity review.

Each test below fails if somebody replaces a shared module with a copy again.
"""

from __future__ import annotations

import importlib

import pytest

# (aug module, bench module, names that must be the SAME object)
SHARED_MODULES = [
    (
        "dagspaces.vlm_geoprivacy_aug.model_prompts",
        "dagspaces.vlm_geoprivacy_bench.model_prompts",
        ["get_prompt_builder", "build_gemma4_prompt", "PROMPT_BUILDERS"],
    ),
    (
        "dagspaces.vlm_geoprivacy_aug.stages.load_dataset",
        "dagspaces.vlm_geoprivacy_bench.stages.load_dataset",
        ["load_dataset"],
    ),
    (
        "dagspaces.vlm_geoprivacy_aug.stages.parse_responses",
        "dagspaces.vlm_geoprivacy_bench.stages.parse_responses",
        ["parse_mcq_responses", "parse_freeform_responses"],
    ),
    (
        "dagspaces.vlm_geoprivacy_aug.stages.granularity_judge",
        "dagspaces.vlm_geoprivacy_bench.stages.granularity_judge",
        ["run_granularity_judge"],
    ),
    (
        "dagspaces.vlm_geoprivacy_aug.stages.compute_metrics",
        "dagspaces.vlm_geoprivacy_bench.stages.compute_metrics",
        ["compute_metrics", "metrics_to_dataframe"],
    ),
    (
        "dagspaces.vlm_geoprivacy_aug.prompts",
        "dagspaces.vlm_geoprivacy_bench.prompts",
        ["QUESTION_DATA", "NUM_QUESTIONS", "SYS_MSG", "GRANULARITY_JUDGE",
         "REFUSAL_PHRASES", "parse_answers"],
    ),
]


@pytest.mark.parametrize(
    "aug_mod,bench_mod,names",
    SHARED_MODULES,
    ids=[m[0].split(".")[-1] for m in SHARED_MODULES],
)
def test_aug_reexports_bench_objects(aug_mod, bench_mod, names):
    """Each shared name must be the SAME object in both dagspaces.

    An equal-but-separate object means somebody copied the code again. The
    copy will fall behind, exactly as it did before 2026-08-12.
    """
    aug = importlib.import_module(aug_mod)
    bench = importlib.import_module(bench_mod)
    for name in names:
        assert hasattr(aug, name), f"{aug_mod} lost the re-export of {name!r}"
        assert getattr(aug, name) is getattr(bench, name), (
            f"{aug_mod}.{name} is not {bench_mod}.{name}. "
            f"Do not copy benchmark code into vlm_geoprivacy_aug — import it. "
            f"A copy falls behind; see this module's docstring."
        )


def test_gemma4_builder_is_reachable_from_aug():
    """gemma-4 must resolve to its OWN builder.

    Until 2026-08-12 this dagspace had no "gemma-4" key at all, and
    get_prompt_builder RAISES on an unknown family rather than substituting
    another builder, so gemma-4 simply could not run here.

    The second assertion guards the separate hazard that motivated a distinct
    builder: the gemma-3 one passes a list-valued system message, which
    Gemma4Processor renders as a Python repr instead of text — corrupt, and
    with no error. Nothing routes gemma-4 there today; the assertion keeps it
    that way.
    """
    from dagspaces.vlm_geoprivacy_aug.model_prompts import (
        build_gemma3_prompt,
        build_gemma4_prompt,
        get_prompt_builder,
    )

    assert get_prompt_builder("gemma-4") is build_gemma4_prompt
    assert get_prompt_builder("gemma-4") is not build_gemma3_prompt


def test_baseline_prompt_matches_the_benchmark_exactly():
    """With no hypothetical, aug must return the benchmark's prompt verbatim.

    This is the contract that makes a hypothetical run comparable with a
    benchmark run: the baseline arm must be the benchmark.
    """
    from dagspaces.vlm_geoprivacy_aug.prompts import (
        prepare_question_prompt as aug_prepare,
    )
    from dagspaces.vlm_geoprivacy_bench.prompts import (
        prepare_question_prompt as bench_prepare,
    )

    for is_free_form in (False, True):
        for include_heuristics in (False, True):
            for enforce_format in (False, True):
                kwargs = dict(
                    mode="zs",
                    is_free_form=is_free_form,
                    include_heuristics=include_heuristics,
                    enforce_format=enforce_format,
                )
                assert aug_prepare(**kwargs) == bench_prepare(**kwargs), (
                    f"aug prompt differs from the benchmark at {kwargs}"
                )


def test_baseline_variant_changes_nothing():
    """The baseline variant must behave the same as ``hypothetical=None``."""
    from dagspaces.vlm_geoprivacy_aug.hypotheticals import BASELINE_VARIANT
    from dagspaces.vlm_geoprivacy_aug.prompts import prepare_question_prompt

    kwargs = dict(mode="zs", is_free_form=False, include_heuristics=True)
    assert (
        prepare_question_prompt(hypothetical=BASELINE_VARIANT, **kwargs)
        == prepare_question_prompt(hypothetical=None, **kwargs)
    )


def test_non_baseline_variant_injects_the_frame():
    """A real variant must put its frame into the prompt, at its position."""
    from dagspaces.vlm_geoprivacy_aug.hypotheticals import HypotheticalVariant
    from dagspaces.vlm_geoprivacy_aug.prompts import prepare_question_prompt

    kwargs = dict(mode="zs", is_free_form=False, include_heuristics=True)
    base_sys, base_usr = prepare_question_prompt(**kwargs)

    user_variant = HypotheticalVariant(
        id="v_user", dimension="device", frame="FRAME-TEXT",
        position="user_prefix",
    )
    sys_p, usr_p = prepare_question_prompt(hypothetical=user_variant, **kwargs)
    assert sys_p == base_sys
    assert len(usr_p) == len(base_usr) + 1
    assert "FRAME-TEXT" in usr_p[0], "the frame must lead the user message"
    assert usr_p[1:] == base_usr, "the question blocks must not change"

    sys_variant = HypotheticalVariant(
        id="v_sys", dimension="device", frame="FRAME-TEXT",
        position="system_suffix",
    )
    sys_p, usr_p = prepare_question_prompt(hypothetical=sys_variant, **kwargs)
    assert sys_p == f"{base_sys} FRAME-TEXT"
    assert usr_p == base_usr, "a system-suffix frame must not touch the user turn"
