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


class TestFlowValidationIntegration:
    def test_flow_quality_columns(self):
        from omegaconf import OmegaConf
        from dagspaces.historical_norms.stages.ci_extraction import (
            _validate_flow_quality,
        )
        cfg = OmegaConf.create(
            {"norm_quality": {"use_ner": False,
                              "character_blocklist": ["javert"]}}
        )
        df = pd.DataFrame([
            {  # flagged: blocklisted name in sender
                "ci_subject": "an ex-convict",
                "ci_sender": "javert",
                "ci_recipient": "a magistrate",
                "ci_information_type": "criminal history",
                "ci_transmission_principle": "official report",
                "ci_context": "law enforcement",
            },
            {  # clean role-abstracted flow
                "ci_subject": "a patient",
                "ci_sender": "a physician",
                "ci_recipient": "a colleague",
                "ci_information_type": "diagnosis",
                "ci_transmission_principle": "professional consultation",
                "ci_context": "medical care",
            },
            {  # parse-error row: no flow fields → null quality columns
                "ci_subject": None, "ci_sender": None, "ci_recipient": None,
                "ci_information_type": None,
                "ci_transmission_principle": None, "ci_context": None,
            },
        ])
        out = _validate_flow_quality(df, cfg)
        assert out["flow_quality_passed"].tolist() == [False, True, None]
        assert "named_char_in_sender:javert" in out["flow_quality_flags"][0]
        assert out["flow_quality_flags"][1] is None
