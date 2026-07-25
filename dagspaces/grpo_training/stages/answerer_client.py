"""Frozen-answerer HTTP client for the m-series `R-OUTCOME` reward (core).

The outcome reward (wiki/grpo_redesign/reward-outcome.md) evaluates a
completion's *structured extraction* by asking a **frozen answerer**
(gemma-4-31b per the revised D1) to answer K probe questions **from the
extraction alone** — no source text, no norms. The reward is mean exact-match
against force-derived gold answers; ``cannot_determine`` scores 0 (the module's
tooth). This file is the online client the reward function calls: one batched
HTTP request per completion, retry-then-group-neutral fallback.

Additive m-series code (the parallel-stack rule, wiki/grpo_redesign/
migration.md item 3): it *reuses* the request/retry/inflight plumbing of the
keeper's online R_ground clients and imports shared helpers — it edits nothing
under the frozen surfaces.

Reused plumbing
---------------
- Request/retry shape: mirrors ``dagspaces.grpo_training.stages.clients``'s
  ``JudgeClient._judge_single`` — a ``requests.Session`` POST to
  ``/v1/chat/completions`` with ``temperature``/``max_tokens`` and
  ``chat_template_kwargs.enable_thinking=False`` (suppress the reasoning trace;
  the answer is short structured JSON).
- Parsing: ``dagspaces.common.json_extraction.extract_json_from_text`` with
  ``repair=True``, plus the channel-wrapped / prose fallback lifted from the
  offline calibration harness (``scripts/run_probe_calibration.py::_parse_vote``
  — the 2026-07-24 lesson that gemma wraps answers in ``<|channel|>`` markers).

The batched-across-completions inflight fan-out (ThreadPoolExecutor) lives in
the caller (``ModularReward``); one ``AnswererClient`` call is one completion.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from typing import Any

import requests

from dagspaces.common.json_extraction import extract_json_from_text

__all__ = ["AnswererClient", "make_answerer_from_cfg", "ANSWERER_SYSTEM"]


# System prompt — VERBATIM from reward-outcome.md ("What the answerer sees",
# the ``system:`` block) and the calibration harness's ``ANSWERER_SYSTEM``.
# Do not reflow: a byte change here silently changes the reward definition.
ANSWERER_SYSTEM = (
    "You answer questions using ONLY the structured information-flow extraction "
    "provided. If the extraction does not determine an answer, reply "
    '"cannot_determine".'
)

# The reply-format line appended to the user turn. Mirrors the doc's call shape.
_REPLY_LINE = (
    'Reply as JSON: {"answers": ["yes"|"no"|"cannot_determine", ...]}'
)

# Structured-fields-only projection. NEVER the free-text reasoning trace and
# never the source chunk — those are the content-smuggling channel the reward
# is designed to close (reward-outcome.md, "What the answerer sees"). Only these
# whitelisted keys are ever serialized to the answerer; every other key on the
# completion's flow dict (reasoning / chunk_text / norms_invoked / …) is
# dropped. ``context`` is whitelisted by decision 2026-07-24: flows and norms
# are context-relative, so the answerer cannot judge a flow without the
# extraction's context field (it is a bounded structured field, not a
# free-text smuggling channel). Per-flow ``appropriateness`` is included only
# when present.
_STRUCTURED_FIELDS: tuple[str, ...] = (
    "subject",
    "sender",
    "recipient",
    "information_type",
    "transmission_principle",
    "context",
    "appropriateness",
)

_VALID_ANSWERS = ("yes", "no", "cannot_determine")

# Sane default when neither the env var nor cfg supplies a URL (server mode).
_DEFAULT_BASE_URL = "http://localhost:8000"


def _normalize_answer(value: Any) -> str:
    """Normalize a raw answer token to one of ``yes`` / ``no`` /
    ``cannot_determine`` (anything else → ``cannot_determine``).

    Mirrors ``run_probe_calibration._parse_vote``'s normalization ladder.
    """
    v = str(value).strip().lower()
    if v in _VALID_ANSWERS:
        return v
    if v.startswith("cannot"):
        return "cannot_determine"
    if v.startswith("yes"):
        return "yes"
    if v.startswith("no"):
        return "no"
    return "cannot_determine"


def _prose_answer(text: str) -> str | None:
    """Prose / channel-wrapped fallback for a single answer slot.

    Order (most specific first): standalone ``cannot_determine`` token, then a
    word-boundary yes/no. Returns ``None`` when the text carries no usable
    signal at all (empty completion, or an ambiguous both-yes-and-no) — the
    caller treats that as a parse failure. Closes the documented masquerade
    (calibration ``_parse_vote``) where a genuine yes/no is channel-wrapped
    and defeats JSON extraction.
    """
    low = str(text or "").lower()
    if "cannot_determine" in low or "cannot determine" in low:
        return "cannot_determine"
    has_yes = bool(re.search(r"\byes\b", low))
    has_no = bool(re.search(r"\bno\b", low))
    if has_yes and not has_no:
        return "yes"
    if has_no and not has_yes:
        return "no"
    return None


def _fit_length(answers: list[str], k: int) -> list[str]:
    """Pad wrong-length answer lists with ``cannot_determine``; truncate longer."""
    if len(answers) < k:
        answers = answers + ["cannot_determine"] * (k - len(answers))
    return answers[:k]


def _parse_answers(raw: str, k: int) -> list[str] | None:
    """Parse an answerer completion into K normalized answers.

    Primary path: ``extract_json_from_text(repair=True)`` → ``{"answers": [...]}``
    (or singular ``{"answer": ...}``). Fallback: whole-completion prose scan,
    applied to every slot (all we can recover when the JSON envelope is absent).

    Returns a length-``k`` list of normalized answers, or ``None`` on a genuine
    parse failure (no JSON envelope AND no prose signal) so the caller can retry.
    """
    obj, _err = extract_json_from_text(raw or "", repair=True)
    raw_answers: list[Any] | None = None
    if isinstance(obj, dict):
        arr = obj.get("answers")
        if isinstance(arr, list):
            raw_answers = list(arr)
        elif "answer" in obj:
            raw_answers = [obj.get("answer")]

    if raw_answers is not None:
        return _fit_length([_normalize_answer(a) for a in raw_answers], k)

    # JSON envelope absent → prose / channel-wrapped fallback per slot. With no
    # per-slot structure to key on, the single recovered token fills every slot.
    prose = _prose_answer(raw)
    if prose is None:
        return None
    return [prose] * k


def _project_flow(flow: Any) -> dict[str, Any]:
    """Project a completion flow dict down to the structured-fields whitelist.

    Every non-whitelisted key (reasoning / context / chunk_text / …) is dropped,
    so free-text can never be serialized to the answerer. Keys present with a
    ``None`` value are omitted; empty strings are preserved (they still carry
    tuple structure without leaking prose).
    """
    if not isinstance(flow, Mapping):
        return {}
    out: dict[str, Any] = {}
    for key in _STRUCTURED_FIELDS:
        if key in flow:
            val = flow[key]
            if val is not None:
                out[key] = val
    return out


class AnswererClient:
    """Frozen-answerer client over an OpenAI-compatible chat endpoint (vLLM
    server mode).

    One :meth:`answer_probes` call is one batched HTTP request per completion:
    the completion's structured extraction plus its K probe questions in, a
    length-K answer list out. Transport or parse failure is retried once; a
    still-failing call returns ``failed=True`` and the caller applies the
    group-neutral 0.5 fallback (reward-outcome.md, "Failure handling") — pricing
    the failure is not this client's job.
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        *,
        timeout_s: float = 60.0,
        max_retries: int = 1,
        temperature: float = 0.0,
        max_tokens: int = 128,
        session: requests.Session | None = None,
    ):
        self.base_url = str(base_url).rstrip("/")
        self.model = model
        self.timeout_s = float(timeout_s)
        self.max_retries = int(max_retries)
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self._session = session or requests.Session()
        # vLLM only exposes the OpenAI-compat surface under /v1; accept a bare
        # host:port or a URL that already ends in /v1 (mirrors judge_client).
        root = self.base_url[: -len("/v1")] if self.base_url.endswith("/v1") else self.base_url
        self._endpoint = f"{root}/v1/chat/completions"

    # ------------------------------------------------------------------
    # Call-shape construction
    # ------------------------------------------------------------------
    def build_user(self, extraction_flows: list[dict], probes: list[str]) -> str:
        """Build the user turn: ``EXTRACTION: {...}`` + ``Qi:`` lines + reply line.

        The extraction serializes **structured fields only** (``_project_flow``);
        empty ``extraction_flows`` serializes as ``{"flows": []}``.
        """
        import json

        structured = [_project_flow(f) for f in (extraction_flows or [])]
        extraction_json = json.dumps({"flows": structured}, ensure_ascii=False)
        lines = [f"EXTRACTION: {extraction_json}"]
        for i, probe in enumerate(probes, start=1):
            lines.append(f"Q{i}: {probe}")
        lines.append(_REPLY_LINE)
        return "\n".join(lines)

    def build_messages(
        self, extraction_flows: list[dict], probes: list[str]
    ) -> list[dict[str, str]]:
        """Full chat messages: verbatim system prompt + built user turn."""
        return [
            {"role": "system", "content": ANSWERER_SYSTEM},
            {"role": "user", "content": self.build_user(extraction_flows, probes)},
        ]

    def _request_body(
        self, messages: list[dict[str, str]]
    ) -> dict[str, Any]:
        return {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            # Suppress <think> — the answer is short structured JSON, not a
            # reasoning chain (same directive the online JudgeClient sends).
            "chat_template_kwargs": {"enable_thinking": False},
        }

    def _post(self, body: dict[str, Any]) -> str:
        """One transport round-trip. Returns the assistant content string;
        raises on any transport / HTTP error (caller handles the retry)."""
        resp = self._session.post(self._endpoint, json=body, timeout=self.timeout_s)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"] or ""

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def answer_probes(
        self, extraction_flows: list[dict], probes: list[str]
    ) -> dict[str, Any]:
        """Answer K probes from one completion's extraction. One batched call.

        Returns ``{"answers", "em_input_ready", "failed", "raw"}``:
        - ``answers``: length-``len(probes)`` list of normalized answers
          (``cannot_determine`` on every slot when ``failed``).
        - ``em_input_ready``: the answers are aligned to the probes and safe to
          pass to :meth:`em`.
        - ``failed``: transport/parse failure persisted through the one retry;
          the caller applies the group-neutral 0.5 fallback.
        - ``raw``: the last raw completion (or transport-error marker) for tracing.
        """
        k = len(probes)
        body = self._request_body(self.build_messages(extraction_flows, probes))

        last_raw = ""
        # max_retries=1 → one retry → two attempts total.
        for _attempt in range(self.max_retries + 1):
            try:
                raw = self._post(body)
            except Exception as exc:  # transport failure → retry
                last_raw = f"[transport_error] {exc}"
                continue
            last_raw = raw
            answers = _parse_answers(raw, k)
            if answers is not None:
                return {
                    "answers": answers,
                    "em_input_ready": len(answers) == k,
                    "failed": False,
                    "raw": raw,
                }
            # parse failure → retry

        return {
            "answers": ["cannot_determine"] * k,
            "em_input_ready": False,
            "failed": True,
            "raw": last_raw,
        }

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------
    @staticmethod
    def em(answers: list[str], golds: list[str]) -> float:
        """Mean exact-match over the K gold probes.

        ``cannot_determine`` scores 0 unconditionally — the module's tooth: an
        extraction too hedged or too empty to determine an answer is priced
        identically to a wrong one (reward-outcome.md, "Scoring"). The
        denominator is the number of gold probes; missing/extra answers score 0.
        """
        if not golds:
            return 0.0
        matches = 0
        for ans, gold in zip(answers, golds):
            if ans == "cannot_determine":
                continue
            if ans == gold:
                matches += 1
        return matches / len(golds)

    @staticmethod
    def em_macro(answers: list[str], golds: list[str]) -> float:
        """Class-balanced EM: mean of per-gold-class EM over the classes present.

        The production scoring rule since 2026-07-25. Micro-averaged EM
        (:meth:`em`) is dominated by the corpus's force skew: the realized
        training probe set is **88.2% gold-yes**, so a blanket-"yes" answer
        scores micro-EM 0.882 — the anti-gaming claim in reward-outcome.md
        ("blanket labels ⇒ EV ≤ base rate") is technically true but the base
        rate is nearly the maximum, which is no defense at all.

        Averaging per class first prices a blanket answer at **0.5** on any row
        carrying both gold classes, restoring the intended asymmetry: acing one
        class while zeroing the other earns the midpoint, not the skew. Rows
        with a single class present are unchanged (macro == micro), so this
        never penalizes a row for the corpus's composition — it only removes
        the free lunch where discrimination is actually measurable.

        ``cannot_determine`` still scores 0 within its class (the tooth is
        untouched).
        """
        if not golds:
            return 0.0
        by_class: dict[str, list[float]] = {}
        for ans, gold in zip(answers, golds):
            by_class.setdefault(gold, []).append(1.0 if ans == gold else 0.0)
        if not by_class:
            return 0.0
        return sum(sum(v) / len(v) for v in by_class.values()) / len(by_class)


# ----------------------------------------------------------------------
# Config wiring (thin)
# ----------------------------------------------------------------------
def _cfg_get(node: Any, key: str, default: Any = None) -> Any:
    """Read ``key`` from a dict-like or attr-like config node."""
    if node is None:
        return default
    if isinstance(node, Mapping):
        val = node.get(key, default)
    elif hasattr(node, "get") and not isinstance(node, (str, bytes)):
        try:
            val = node.get(key, default)
        except Exception:
            val = getattr(node, key, default)
    else:
        val = getattr(node, key, default)
    return default if val is None else val


def make_answerer_from_cfg(cfg: Any) -> AnswererClient:
    """Build an :class:`AnswererClient` from the m-series config.

    Reads ``training.grpo.answerer.{base_url_env, model, timeout_s, max_retries,
    temperature, max_tokens}``. The base URL is resolved from the env var whose
    *name* is ``base_url_env`` (default ``VLLM_SERVER_URL``); if that env var is
    unset, falls back to an explicit ``base_url`` key, then to a sane local
    default. Deliberately thin — the config schema is the authority.
    """
    node = cfg
    for part in ("training", "grpo", "answerer"):
        node = _cfg_get(node, part, None)
        if node is None:
            break

    base_url_env = _cfg_get(node, "base_url_env", "VLLM_SERVER_URL")
    base_url = (
        os.environ.get(str(base_url_env), "")
        or str(_cfg_get(node, "base_url", "") or "")
        or _DEFAULT_BASE_URL
    )

    return AnswererClient(
        base_url=base_url,
        model=str(_cfg_get(node, "model", "default")),
        timeout_s=float(_cfg_get(node, "timeout_s", 60.0)),
        max_retries=int(_cfg_get(node, "max_retries", 1)),
        temperature=float(_cfg_get(node, "temperature", 0.0)),
        max_tokens=int(_cfg_get(node, "max_tokens", 128)),
    )
