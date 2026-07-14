"""Layered person-name QA detection (2026-06-09, NER upgrade).

The manual per-book blocklist only covered the 10-novel corpus; the spaCy
PERSON-NER layer is what keeps QA coverage on novels nobody hand-listed.
These tests exercise the real en_core_web_sm model (installed in the venv).
"""

import pandas as pd
import pytest

from dagspaces.historical_norms.name_detection import PersonNameDetector


@pytest.fixture(scope="module")
def ner_detector():
    det = PersonNameDetector(blocklist={"big brother", "monte cristo"},
                             use_ner=True)
    # Skip NER-dependent tests cleanly if the model cannot load here.
    from dagspaces.historical_norms.name_detection import _get_nlp
    if _get_nlp() is None:
        pytest.skip("spaCy en_core_web_sm unavailable")
    return det


class TestNERLayer:
    def test_unlisted_character_name_flagged(self, ner_detector):
        # A name from a novel nobody hand-listed — the scaling case.
        found = ner_detector.detect(
            "Emma Woodhouse ought to conceal her matchmaking schemes."
        )
        assert any("Emma" in n for n in found["person_entity"])

    def test_role_abstracted_norm_clean(self, ner_detector):
        found = ner_detector.detect(
            "A guest ought to keep confidences shared by the host."
        )
        assert found["person_entity"] == []
        assert found["titled"] == []
        assert found["blocklist"] == []

    def test_field_flags_format(self, ner_detector):
        # Name mid-sentence: single-token names in sentence-initial position
        # are deliberately dropped by the precision filter (see
        # _plausible_person) — multi-token or mid-sentence names are kept.
        flags = ner_detector.field_flags(
            "articulation", "It falls to Elizabeth to guard her sister's reputation."
        )
        assert any(f.startswith("person_entity_in_articulation:") for f in flags)


class TestPatternLayers:
    def test_titled_name_without_ner(self):
        det = PersonNameDetector(use_ner=False)
        found = det.detect("Mrs. Reynolds should not gossip about her employer.")
        assert found["titled"] == ["Mrs. Reynolds"]

    def test_blocklist_word_boundary(self):
        det = PersonNameDetector(blocklist={"pearl"}, use_ner=False)
        # Substring inside another phrase must not fire ...
        assert det.detect("a pearl of great price")["blocklist"] == ["pearl"]
        # ... but an embedded-substring word must not match.
        det2 = PersonNameDetector(blocklist={"ada"}, use_ner=False)
        assert det2.detect("an adage about trust")["blocklist"] == []

    def test_blocklist_alias_ner_would_miss(self):
        det = PersonNameDetector(blocklist={"big brother"}, use_ner=False)
        flags = det.field_flags("subject", "a loyal servant of Big Brother")
        assert "named_char_in_subject:big brother" in flags

    def test_empty_and_none_text(self):
        det = PersonNameDetector(use_ner=False)
        assert det.detect("") == {"blocklist": [], "titled": [],
                                  "person_entity": []}
        assert det.field_flags("act", None) == []


class TestPersonEntityPrecisionFilter:
    def test_sentence_initial_common_nouns_not_flagged(self, ner_detector):
        # en_core_web_sm tags these as PERSON without the filter.
        for text in (
            "Law enforcement officers must handcuff the ruffians.",
            "Citizens of a totalitarian state must guard their thoughts.",
            "Children who are customers must leave the shop after eating.",
        ):
            assert ner_detector.detect(text)["person_entity"] == [], text

    def test_multi_token_name_kept_even_sentence_initial(self, ner_detector):
        found = ner_detector.detect(
            "Martin Verga requires the nuns to wear drugget chemises."
        )
        assert any("Martin Verga" in n for n in found["person_entity"])

    def test_lowercase_entity_dropped(self):
        from dagspaces.historical_norms.name_detection import _plausible_person

        class _Ent:
            def __init__(self, text, start):
                self.text, self.start_char = text, start

        assert _plausible_person(_Ent("inn", 17), "The women of the inn") is False


class TestRevalidationAfterAbstraction:
    def test_stale_flags_replaced_and_preserved(self):
        from omegaconf import OmegaConf
        from dagspaces.historical_norms.stages.norm_role_abstraction import (
            revalidate_norm_quality,
        )
        cfg = OmegaConf.create(
            {"norm_quality": {"use_ner": False,
                              "character_blocklist": ["valjean"]}}
        )
        df = pd.DataFrame([
            {   # abstraction cleaned the name → stale fail must flip to pass
                "raz_norm_subject": "a mayor",
                "raz_norm_act": "ought to protect the vulnerable",
                "raz_condition_of_application": "",
                "raz_norm_articulation": "A mayor ought to protect the vulnerable.",
                "norm_quality_flags": "named_char_in_subject:valjean",
                "norm_quality_passed": False,
            },
            {   # abstraction leaked the name → pass must flip to fail
                "raz_norm_subject": "valjean",
                "raz_norm_act": "must conceal his past",
                "raz_condition_of_application": "",
                "raz_norm_articulation": "Valjean must conceal his past.",
                "norm_quality_flags": None,
                "norm_quality_passed": True,
            },
        ])
        out = revalidate_norm_quality(df, cfg)
        assert out["norm_quality_passed"].tolist() == [True, False]
        assert out["pre_abstraction_norm_quality_passed"].tolist() == [False, True]
        assert "named_char_in_subject:valjean" in out["norm_quality_flags"][1]


class TestNormValidationIntegration:
    def test_validate_norm_quality_uses_detector(self):
        from dagspaces.historical_norms.stages.norm_extraction import (
            _validate_norm_quality,
        )
        det = PersonNameDetector(blocklist={"valjean"}, use_ner=False)
        row = {
            "raz_norm_subject": "valjean",
            "raz_norm_act": "must conceal his past",
            "raz_condition_of_application": "",
            "raz_norm_articulation": "Mr. Madeleine ought to hide his identity.",
        }
        out = _validate_norm_quality(dict(row), det)
        assert out["norm_quality_passed"] is False
        assert "named_char_in_subject:valjean" in out["norm_quality_flags"]
        assert "titled_name_in_articulation" in out["norm_quality_flags"]

    def test_clean_norm_passes(self):
        from dagspaces.historical_norms.stages.norm_extraction import (
            _validate_norm_quality,
        )
        det = PersonNameDetector(use_ner=False)
        row = {
            "raz_norm_subject": "a mayor",
            "raz_norm_act": "ought to protect the vulnerable",
            "raz_condition_of_application": "when they hold public office",
            "raz_norm_articulation": "A mayor ought to protect the vulnerable.",
        }
        out = _validate_norm_quality(dict(row), det)
        assert out["norm_quality_passed"] is True
        assert out["norm_quality_flags"] is None


class TestAmbiguousBlocklistNames:
    """Blocklist entries that are also ordinary English words.

    The built-in blocklist carries "will" (Will Ladislaw) and "may" (May
    Welland). It is global and was matched case-insensitively, so the modal
    verbs in ordinary norm text tripped the gate in every book — 85% of
    fiction10's `norm_quality_passed == False` was modal verbs, inflating
    apparent character leakage from 0.32% to 4.39%.

    The invariant: ambiguous entries match the *capitalized* form only.
    """

    def test_modal_verbs_do_not_trip_the_gate(self):
        det = PersonNameDetector(blocklist={"may", "will"}, use_ner=False)
        for text in (
            "a servant may not disclose a confidence",
            "the heir will inherit the estate",
            "a guest may decline an invitation and will not be censured",
        ):
            assert det.detect(text)["blocklist"] == [], text

    def test_capitalized_character_still_flagged(self):
        det = PersonNameDetector(blocklist={"may", "will"}, use_ner=False)
        assert det.detect("May refuses to break her engagement")["blocklist"] == ["may"]
        assert det.detect("Will Ladislaw accepts the inheritance")["blocklist"] == ["will"]

    def test_unambiguous_names_stay_case_insensitive(self):
        """Only AMBIGUOUS_NAMES changes behaviour — everything else is untouched."""
        det = PersonNameDetector(blocklist={"valjean", "big brother"}, use_ner=False)
        assert det.detect("valjean conceals his identity")["blocklist"] == ["valjean"]
        assert det.detect("Valjean conceals his identity")["blocklist"] == ["valjean"]
        assert det.detect("big brother is watching")["blocklist"] == ["big brother"]

    def test_ambiguous_set_is_lowercase(self):
        """Membership is tested against lowercased blocklist entries."""
        from dagspaces.historical_norms.name_detection import AMBIGUOUS_NAMES

        assert all(n == n.lower() for n in AMBIGUOUS_NAMES)
        assert {"may", "will"} <= AMBIGUOUS_NAMES


class TestFlowQualityCheckRemoved:
    """The flows track deliberately has NO character-name QA gate (2026-07-13).

    `_validate_flow_quality` enforced a role requirement neither CI prompt ever
    stated (`norm_extraction_fiction` demands roles five times; the CI prompts
    never do), flagging 37.6% of fiction10 flows for doing what they were asked.
    Nissenbaum's sender/recipient are actors in a context — naming them is the
    point. This test exists so the check is not reintroduced by reflex.
    """

    def test_no_flow_quality_validator(self):
        from dagspaces.historical_norms.stages import ci_extraction

        assert not hasattr(ci_extraction, "_validate_flow_quality")
        assert not hasattr(ci_extraction, "_FLOW_QA_FIELDS")

    def test_no_flow_quality_metric(self):
        from dagspaces.historical_norms.stage_metrics import (
            compute_stage_quality_metrics,
        )

        df = pd.DataFrame([{
            "ci_sender": "Elizabeth",
            "ci_recipient": "Jane",
            "ci_appropriateness": "appropriate",
            "flow_quality_passed": False,  # stale column from an older parquet
        }])
        m = compute_stage_quality_metrics("ci_extraction", df)
        assert not any("flow_quality" in k for k in m)
