"""Static leak-check on the capture-device frames: frames must state capture
FACTS (device, manner, notice), never verdict language that telegraphs the
'expected' Q7 answer. Complements — does not replace — the live-model
prompt-dev pass (TODOS Part 1)."""

from __future__ import annotations

import re

import pytest
import yaml

FRAMES_PATH = "dagspaces/vlm_geoprivacy_aug/conf/hypotheticals/capture_devices.yaml"

# Verdict / normative-conclusion vocabulary that would leak the answer.
LEAK_PATTERNS = [
    r"\bprivacy\b", r"\bprivate\b", r"\babstain\w*", r"\bviolat\w+",
    r"\binappropriate\b", r"\bappropriate\b", r"\bharm\w*", r"\brisk\w*",
    r"\bconsent\w*", r"\bintrusive\b", r"\bcreepy\b", r"\bsurveillance\b",
    r"\bshould\b", r"\bwrong\b", r"\bsensitive\b", r"\bdisclos\w+",
]
LEAK_RE = re.compile("|".join(LEAK_PATTERNS), re.IGNORECASE)


@pytest.fixture(scope="module")
def variants():
    doc = yaml.safe_load(open(FRAMES_PATH))
    return doc["hypotheticals"]["variants"]


def test_frames_state_facts_not_verdicts(variants):
    offenders = {}
    for v in variants:
        text = f"{v.get('frame', '')} {v.get('bridge', '')}"
        hits = sorted(set(m.group(0).lower() for m in LEAK_RE.finditer(text)))
        if hits:
            offenders[v["id"]] = hits
    assert not offenders, f"verdict-language leakage in frames: {offenders}"


def test_frames_do_not_mention_the_questions_or_answers(variants):
    for v in variants:
        text = f"{v.get('frame', '')} {v.get('bridge', '')}".lower()
        assert "q7" not in text and "granularity" not in text and "location disclosure" not in text
