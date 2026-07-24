"""Tests for the frozen-answerer client (`R-OUTCOME` core, m-series).

Mocks the HTTP layer (no live server): a ``FakeSession`` records posted bodies
and returns scripted completions, so every test asserts against the exact call
shape the client builds and the exact parse/normalize/retry behavior it applies.

Covers the frozen contract in wiki/grpo_redesign/reward-outcome.md and
migration.md item 3: verbatim system prompt, structured-fields-only
serialization (reasoning never leaks), clean/channel-wrapped/truncated/prose
parsing, wrong-length padding, the normalization table, retry-then-failed
semantics, ``em`` with ``cannot_determine`` = 0, and empty-flow serialization.
"""

from __future__ import annotations

import json
import re
from types import SimpleNamespace

import pytest

from dagspaces.grpo_training.stages.answerer_client import (
    ANSWERER_SYSTEM,
    AnswererClient,
    make_answerer_from_cfg,
)


# ---------------------------------------------------------------------------
# HTTP mock
# ---------------------------------------------------------------------------
class FakeResponse:
    def __init__(self, content: str = "", *, status_ok: bool = True):
        self._content = content
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP 500")

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


class FakeSession:
    """Scripted session. ``script`` is a list of items consumed per POST:
    a str → that content; an Exception → raised; a FakeResponse → returned.
    Falls back to the last item once exhausted (so 'always error' is easy).
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls: list[dict] = []
        self._i = 0

    def post(self, url, json=None, timeout=None):  # noqa: A002 (mirror requests API)
        self.calls.append({"url": url, "json": json, "timeout": timeout})
        item = self.script[min(self._i, len(self.script) - 1)]
        self._i += 1
        if isinstance(item, Exception):
            raise item
        if isinstance(item, FakeResponse):
            return item
        return FakeResponse(str(item))


def make_client(script, **kw) -> tuple[AnswererClient, FakeSession]:
    sess = FakeSession(script)
    client = AnswererClient(
        base_url=kw.pop("base_url", "http://localhost:8000"),
        model=kw.pop("model", "gemma-4-31b"),
        session=sess,
        **kw,
    )
    return client, sess


def posted_user(sess: FakeSession, idx: int = 0) -> str:
    msgs = sess.calls[idx]["json"]["messages"]
    return next(m["content"] for m in msgs if m["role"] == "user")


def posted_system(sess: FakeSession, idx: int = 0) -> str:
    msgs = sess.calls[idx]["json"]["messages"]
    return next(m["content"] for m in msgs if m["role"] == "system")


# ---------------------------------------------------------------------------
# Call-shape snapshot
# ---------------------------------------------------------------------------
def test_system_prompt_verbatim():
    # Pinned against wiki/grpo_redesign/reward-outcome.md ("What the answerer
    # sees") and the calibration harness — a byte change alters the reward.
    assert ANSWERER_SYSTEM == (
        "You answer questions using ONLY the structured information-flow "
        "extraction provided. If the extraction does not determine an answer, "
        'reply "cannot_determine".'
    )


def test_call_shape_and_structured_fields_only():
    flow = {
        "subject": "Alice",
        "sender": "Dr. Bob",
        "recipient": "the insurer",
        "information_type": "HIV status",
        "transmission_principle": "with consent",
        # context IS whitelisted (decision 2026-07-24): flows and norms are
        # context-relative, so the answerer needs the extraction's context.
        "context": "a hospital ward at midnight",
        "appropriateness": "inappropriate",
        # Free-text / smuggling channels that must NEVER be serialized:
        "reasoning": "SECRETLEAK the chunk says he should not share",
        "chunk_text": "FULLCHUNKTEXT ...",
        "norms_invoked": ["confidentiality"],
    }
    client, sess = make_client(['{"answers": ["no"]}'])
    client.answer_probes([flow], ["Should this be shared?"])

    assert posted_system(sess) == ANSWERER_SYSTEM
    user = posted_user(sess)

    # Structured fields present.
    for val in ("Alice", "Dr. Bob", "the insurer", "HIV status",
                "with consent", "inappropriate", "hospital ward"):
        assert val in user
    # Reasoning / chunk / norms never serialized.
    for leaked in ("SECRETLEAK", "reasoning",
                   "FULLCHUNKTEXT", "chunk_text", "confidentiality"):
        assert leaked not in user, f"{leaked!r} leaked into answerer input"

    # Parse the EXTRACTION JSON and assert the flow dict is whitelist-exact.
    m = re.search(r"EXTRACTION: (\{.*\})", user)
    assert m, user
    extraction = json.loads(m.group(1))
    assert set(extraction["flows"][0].keys()) == {
        "subject", "sender", "recipient", "information_type",
        "transmission_principle", "context", "appropriateness",
    }

    # Q-line + reply-line shape.
    assert "Q1: Should this be shared?" in user
    assert 'Reply as JSON: {"answers":' in user


def test_multiple_probes_numbered():
    client, sess = make_client(['{"answers": ["yes", "no", "cannot_determine"]}'])
    client.answer_probes([], ["Pa?", "Pb?", "Pc?"])
    user = posted_user(sess)
    assert "Q1: Pa?" in user
    assert "Q2: Pb?" in user
    assert "Q3: Pc?" in user


def test_empty_flows_serialize_as_empty_list():
    client, sess = make_client(['{"answers": ["yes"]}'])
    client.answer_probes([], ["Q?"])
    assert 'EXTRACTION: {"flows": []}' in posted_user(sess)


def test_appropriateness_omitted_when_absent():
    flow = {"subject": "A", "sender": "B", "recipient": "C",
            "information_type": "D", "transmission_principle": "E"}
    client, sess = make_client(['{"answers": ["yes"]}'])
    client.answer_probes([flow], ["Q?"])
    extraction = json.loads(re.search(r"EXTRACTION: (\{.*\})", posted_user(sess)).group(1))
    assert "appropriateness" not in extraction["flows"][0]


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------
def test_parse_clean_json():
    client, _ = make_client(['{"answers": ["yes", "no"]}'])
    out = client.answer_probes([], ["Q1?", "Q2?"])
    assert out["answers"] == ["yes", "no"]
    assert out["failed"] is False
    assert out["em_input_ready"] is True


def test_parse_channel_wrapped_json():
    # Gemma wraps the JSON in channel markers; extract_json_from_text slices
    # the outermost {...} span and parses it.
    raw = '<|channel|>final<|message|>{"answers": ["no", "yes"]}<|end|>'
    client, _ = make_client([raw])
    out = client.answer_probes([], ["Q1?", "Q2?"])
    assert out["answers"] == ["no", "yes"]
    assert out["failed"] is False


def test_parse_truncated_repairable():
    # Malformed-but-braced JSON (trailing comma) → json_repair rescues it.
    raw = '{"answers": ["yes", "no",]}'
    client, _ = make_client([raw])
    out = client.answer_probes([], ["Q1?", "Q2?"])
    assert out["answers"] == ["yes", "no"]
    assert out["failed"] is False


def test_parse_prose_channel_wrapped_answer():
    # No JSON envelope at all → prose fallback recovers a channel-wrapped 'no'.
    raw = "<|channel|>final<|message|> The answer is no."
    client, _ = make_client([raw])
    out = client.answer_probes([], ["Q1?"])
    assert out["answers"] == ["no"]
    assert out["failed"] is False


def test_parse_prose_cannot_determine():
    raw = "I cannot_determine this from the extraction."
    client, _ = make_client([raw])
    out = client.answer_probes([], ["Q1?"])
    assert out["answers"] == ["cannot_determine"]
    assert out["failed"] is False


def test_singular_answer_key_accepted():
    client, _ = make_client(['{"answer": "yes"}'])
    out = client.answer_probes([], ["Q1?"])
    assert out["answers"] == ["yes"]
    assert out["failed"] is False


# ---------------------------------------------------------------------------
# Wrong-length padding
# ---------------------------------------------------------------------------
def test_wrong_length_padded_short():
    client, _ = make_client(['{"answers": ["yes"]}'])
    out = client.answer_probes([], ["Q1?", "Q2?", "Q3?"])
    assert out["answers"] == ["yes", "cannot_determine", "cannot_determine"]
    assert out["em_input_ready"] is True


def test_wrong_length_truncated_long():
    client, _ = make_client(['{"answers": ["yes", "no", "yes", "no"]}'])
    out = client.answer_probes([], ["Q1?", "Q2?"])
    assert out["answers"] == ["yes", "no"]


# ---------------------------------------------------------------------------
# Normalization table
# ---------------------------------------------------------------------------
@pytest.mark.parametrize("raw_token,expected", [
    ("yes", "yes"),
    ("Yes", "yes"),
    ("YES", "yes"),
    ("yes.", "yes"),
    ("no", "no"),
    ("NO", "no"),
    ("no,", "no"),
    ("cannot_determine", "cannot_determine"),
    ("cannot determine", "cannot_determine"),
    ("Cannot", "cannot_determine"),
    ("maybe", "cannot_determine"),
    ("", "cannot_determine"),
    ("42", "cannot_determine"),
])
def test_normalization_table(raw_token, expected):
    client, _ = make_client([json.dumps({"answers": [raw_token]})])
    out = client.answer_probes([], ["Q1?"])
    assert out["answers"] == [expected]


# ---------------------------------------------------------------------------
# Retry-then-failed semantics
# ---------------------------------------------------------------------------
def test_transport_failure_retries_then_failed():
    client, sess = make_client([RuntimeError("conn refused")], max_retries=1)
    out = client.answer_probes([], ["Q1?", "Q2?"])
    assert out["failed"] is True
    assert out["em_input_ready"] is False
    assert out["answers"] == ["cannot_determine", "cannot_determine"]
    assert len(sess.calls) == 2  # one try + one retry
    assert "transport_error" in out["raw"]


def test_transport_recovers_on_retry():
    client, sess = make_client(
        [RuntimeError("blip"), '{"answers": ["yes"]}'], max_retries=1
    )
    out = client.answer_probes([], ["Q1?"])
    assert out["failed"] is False
    assert out["answers"] == ["yes"]
    assert len(sess.calls) == 2


def test_unparseable_retries_then_failed():
    # Empty completions carry no JSON and no prose signal → parse failure.
    client, sess = make_client(["", ""], max_retries=1)
    out = client.answer_probes([], ["Q1?"])
    assert out["failed"] is True
    assert len(sess.calls) == 2


def test_http_status_error_is_transport_failure():
    client, sess = make_client(
        [FakeResponse(status_ok=False), FakeResponse(status_ok=False)],
        max_retries=1,
    )
    out = client.answer_probes([], ["Q1?"])
    assert out["failed"] is True
    assert len(sess.calls) == 2


def test_max_retries_zero_single_attempt():
    client, sess = make_client([RuntimeError("x")], max_retries=0)
    out = client.answer_probes([], ["Q1?"])
    assert out["failed"] is True
    assert len(sess.calls) == 1


# ---------------------------------------------------------------------------
# em()
# ---------------------------------------------------------------------------
def test_em_all_correct():
    assert AnswererClient.em(["yes", "no"], ["yes", "no"]) == 1.0


def test_em_cannot_determine_scores_zero():
    # cannot_determine is priced at the floor — the module's tooth.
    score = AnswererClient.em(
        ["yes", "no", "cannot_determine"], ["yes", "no", "no"]
    )
    assert score == pytest.approx(2 / 3)


def test_em_cannot_determine_zero_even_against_cd_gold():
    # A cannot_determine answer never earns credit, even if gold were cd.
    assert AnswererClient.em(["cannot_determine"], ["cannot_determine"]) == 0.0


def test_em_wrong_answers():
    assert AnswererClient.em(["no", "yes"], ["yes", "no"]) == 0.0


def test_em_denominator_is_golds():
    # Missing answers score 0 against the K gold probes.
    assert AnswererClient.em(["yes"], ["yes", "no", "yes"]) == pytest.approx(1 / 3)


def test_em_empty_golds():
    assert AnswererClient.em([], []) == 0.0


# ---------------------------------------------------------------------------
# Config wiring
# ---------------------------------------------------------------------------
def test_make_answerer_from_cfg_env_url(monkeypatch):
    monkeypatch.setenv("MY_ANSWERER_URL", "http://klara:9100")
    cfg = {
        "training": {"grpo": {"answerer": {
            "base_url_env": "MY_ANSWERER_URL",
            "model": "gemma-4-31b",
            "timeout_s": 30.0,
            "max_retries": 2,
            "temperature": 0.0,
            "max_tokens": 64,
        }}}
    }
    client = make_answerer_from_cfg(cfg)
    assert client.base_url == "http://klara:9100"
    assert client.model == "gemma-4-31b"
    assert client.max_retries == 2
    assert client.max_tokens == 64
    assert client._endpoint == "http://klara:9100/v1/chat/completions"


def test_make_answerer_from_cfg_default_env_name(monkeypatch):
    monkeypatch.setenv("VLLM_SERVER_URL", "http://host:8000")
    client = make_answerer_from_cfg({"training": {"grpo": {"answerer": {}}}})
    assert client.base_url == "http://host:8000"


def test_make_answerer_from_cfg_falls_back_to_default(monkeypatch):
    monkeypatch.delenv("VLLM_SERVER_URL", raising=False)
    client = make_answerer_from_cfg({})
    assert client.base_url == "http://localhost:8000"


def test_make_answerer_from_cfg_attr_style():
    # Attr-style config node (e.g. SimpleNamespace / OmegaConf-like).
    ns = SimpleNamespace(training=SimpleNamespace(grpo=SimpleNamespace(
        answerer=SimpleNamespace(base_url="http://x:1", model="m"))))
    client = make_answerer_from_cfg(ns)
    assert client.base_url == "http://x:1"
    assert client.model == "m"


def test_endpoint_normalizes_trailing_v1():
    client = AnswererClient("http://host:8000/v1", "m")
    assert client._endpoint == "http://host:8000/v1/chat/completions"
