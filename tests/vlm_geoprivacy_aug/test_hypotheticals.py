"""Tests for hypothetical capture-context variants: loading/validation,
dataset expansion, and prompt injection.

The core invariant: a ``None`` hypothetical (or the baseline variant) must
leave the prompt byte-identical to the un-augmented benchmark, so baseline
runs stay comparable with vlm_geoprivacy_bench results.
"""

from __future__ import annotations

import pandas as pd
import pytest

from dagspaces.vlm_geoprivacy_aug.hypotheticals import (
    BASELINE_ID,
    FRAME_TEMPLATE,
    HypotheticalVariant,
    load_variants,
    render_user_frame,
)
from dagspaces.vlm_geoprivacy_aug.prompts import (
    INST_FREE_FORM,
    NUM_QUESTIONS,
    prepare_question_prompt,
)
from dagspaces.vlm_geoprivacy_aug.stages.inpaint_hypotheticals import (
    HYP_COLUMNS,
    expand_with_hypotheticals,
)

GLASSES = {
    "id": "smart_glasses",
    "dimension": "capture_device",
    "frame": "This photo was captured by smart glasses.",
    "ci_params": {"sender": "wearer"},
}

GLASSES_BRIDGED = {
    **GLASSES,
    "bridge": "The photo-taker is the passerby wearing the glasses.",
}


class TestLoadVariants:
    def test_baseline_auto_inserted_first(self):
        variants = load_variants([GLASSES])
        assert [v.id for v in variants] == [BASELINE_ID, "smart_glasses"]
        assert variants[0].is_baseline and variants[0].frame == ""

    def test_empty_config_yields_baseline_only(self):
        assert [v.id for v in load_variants(None)] == [BASELINE_ID]
        assert [v.id for v in load_variants([])] == [BASELINE_ID]

    def test_explicit_baseline_not_duplicated(self):
        variants = load_variants([{"id": BASELINE_ID, "dimension": "control"}, GLASSES])
        assert [v.id for v in variants] == [BASELINE_ID, "smart_glasses"]

    def test_duplicate_id_raises(self):
        with pytest.raises(ValueError, match="Duplicate"):
            load_variants([GLASSES, GLASSES])

    def test_dot_in_id_raises(self):
        with pytest.raises(ValueError, match="must not contain"):
            load_variants([{**GLASSES, "id": "glasses.v2"}])

    def test_missing_frame_raises(self):
        with pytest.raises(ValueError, match="need a frame"):
            load_variants([{"id": "x", "dimension": "capture_device"}])

    def test_missing_dimension_raises(self):
        with pytest.raises(ValueError, match="need a dimension"):
            load_variants([{"id": "x", "frame": "f"}])

    def test_baseline_with_frame_raises(self):
        with pytest.raises(ValueError, match="baseline"):
            load_variants([{"id": BASELINE_ID, "dimension": "control", "frame": "f"}])

    def test_bad_position_raises(self):
        with pytest.raises(ValueError, match="position"):
            load_variants([{**GLASSES, "position": "sideways"}])


class TestBridges:
    """The photo-taker bridge maps the benchmark's 'photo-taker' onto the CI
    sender, keeping sender and capture device as separate parameters. It is
    folded into the frame by default and dropped in the ablation arm."""

    def test_bridge_folded_into_frame_by_default(self):
        variant = load_variants([GLASSES_BRIDGED])[1]
        assert variant.frame == f"{GLASSES['frame']} {GLASSES_BRIDGED['bridge']}"

    def test_bridge_dropped_when_disabled(self):
        variant = load_variants([GLASSES_BRIDGED], include_bridges=False)[1]
        assert variant.frame == GLASSES["frame"]
        assert GLASSES_BRIDGED["bridge"] not in variant.frame

    def test_variant_without_bridge_unaffected_by_flag(self):
        with_flag = load_variants([GLASSES], include_bridges=True)[1]
        without_flag = load_variants([GLASSES], include_bridges=False)[1]
        assert with_flag == without_flag

    def test_baseline_with_bridge_raises(self):
        with pytest.raises(ValueError, match="bridge"):
            load_variants([{"id": BASELINE_ID, "dimension": "control", "bridge": "b"}])


class TestExpandWithHypotheticals:
    def _df(self):
        return pd.DataFrame(
            {"numeric_id": ["1", "2", "3"], "image_path": ["a.jpg", "b.jpg", "c.jpg"]}
        )

    def test_cross_product_shape_and_columns(self):
        variants = load_variants([GLASSES])
        out = expand_with_hypotheticals(self._df(), variants)
        assert len(out) == 3 * 2
        for col in HYP_COLUMNS:
            assert col in out.columns
        # Original columns preserved, baseline block first.
        assert out["hyp_id"].tolist() == [BASELINE_ID] * 3 + ["smart_glasses"] * 3
        assert set(out["numeric_id"]) == {"1", "2", "3"}

    def test_baseline_rows_have_empty_frame(self):
        out = expand_with_hypotheticals(self._df(), load_variants([GLASSES]))
        base = out[out["hyp_id"] == BASELINE_ID]
        assert (base["hyp_frame"] == "").all()

    def test_column_clash_raises(self):
        df = self._df()
        df["hyp_id"] = "oops"
        with pytest.raises(ValueError, match="hypothetical columns"):
            expand_with_hypotheticals(df, load_variants([GLASSES]))

    def test_empty_variants_raises(self):
        with pytest.raises(ValueError):
            expand_with_hypotheticals(self._df(), [])


class TestPromptInjection:
    def test_none_and_baseline_leave_prompt_unchanged(self):
        plain = prepare_question_prompt("zs", is_free_form=False, include_heuristics=True)
        with_none = prepare_question_prompt(
            "zs", is_free_form=False, include_heuristics=True, hypothetical=None
        )
        baseline = load_variants([])[0]
        with_baseline = prepare_question_prompt(
            "zs", is_free_form=False, include_heuristics=True, hypothetical=baseline
        )
        assert plain == with_none == with_baseline

    def test_user_prefix_frame_prepended_mcq(self):
        variant = load_variants([GLASSES])[1]
        sys_plain, usr_plain = prepare_question_prompt(
            "zs", is_free_form=False, include_heuristics=True
        )
        sys_msg, usr_prompts = prepare_question_prompt(
            "zs", is_free_form=False, include_heuristics=True, hypothetical=variant
        )
        # Frame block first; system message and all question parts untouched.
        assert usr_prompts[0] == FRAME_TEMPLATE.format(frame=variant.frame)
        assert variant.frame in usr_prompts[0]
        assert sys_msg == sys_plain
        assert usr_prompts[1:] == usr_plain
        # Question parts still present and intact (7 questions + instruction).
        assert len(usr_prompts) == NUM_QUESTIONS + 2

    def test_user_prefix_frame_precedes_freeform_instruction(self):
        variant = load_variants([GLASSES])[1]
        _, usr_prompts = prepare_question_prompt(
            "zs", is_free_form=True, include_heuristics=False, hypothetical=variant
        )
        assert usr_prompts == [FRAME_TEMPLATE.format(frame=variant.frame), INST_FREE_FORM]

    def test_system_suffix_appends_to_system_message(self):
        variant = HypotheticalVariant(
            id="glasses_sys",
            dimension="capture_device",
            frame="Captured by smart glasses.",
            position="system_suffix",
        )
        sys_plain, usr_plain = prepare_question_prompt(
            "zs", is_free_form=False, include_heuristics=True
        )
        sys_msg, usr_prompts = prepare_question_prompt(
            "zs", is_free_form=False, include_heuristics=True, hypothetical=variant
        )
        assert sys_msg == f"{sys_plain} {variant.frame}"
        assert usr_prompts == usr_plain

    def test_render_user_frame_empty_for_baseline_and_system_variants(self):
        baseline = load_variants([])[0]
        sys_variant = HypotheticalVariant(
            id="x", dimension="d", frame="f", position="system_suffix"
        )
        assert render_user_frame(baseline) == ""
        assert render_user_frame(sys_variant) == ""
