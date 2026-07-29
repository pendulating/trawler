"""`ModularReward` — the m-series modular reward stack (redesign item 4).

The reward in three sentences (wiki/grpo_redesign/README.md): an extraction
scores zero unless it parses and is schema-complete (``R-VALID``); a valid one
is scored by whether a frozen answerer, reading only the extraction, correctly
answers probe questions whose gold comes from the book's governing norms
(``R-OUTCOME``), plus optional deletable auxiliaries (judge grounding, wrong-book
contrast); a vignette row is an 8-item deontic battery scored by deontic distance
+ citation. Rows where probes cannot run (no-flow declarations, and every
completion on a gold-NO chunk) are scored by a fixed abstention table
(``A-ABSTAIN``) with no server calls.

This is additive m-series code (the parallel-stack rule,
wiki/grpo_redesign/migration.md): it is selected only when
``training.grpo.reward_composition == "modular"``; the keeper's ``directional``
path in :mod:`rewards` is never touched. It *imports* the four frozen wave-1
contracts and never reimplements them:

* :mod:`probes` — ``sample_probes`` / ``apply_null_filter`` (probe attachment).
* :mod:`deontic_distance` — ``parse_battery_completion`` / ``score_battery``.
* :mod:`answerer_client` — ``AnswererClient`` / ``make_answerer_from_cfg``.
* :mod:`deontic` — ``FORCE_TO_GOLD`` and the ``candidate_appropriateness_*``
  diagnostic (``diag/direction_consistency`` only — never a reward term).

The listwise judge auxiliaries (``R-GROUND`` / ``R-CONTRAST``) are supplied to
:class:`ModularReward` as *injected callables* so the reward core stays
judge-free and unit-testable; the production adapters in
:func:`make_modular_reward_from_cfg` reuse the keeper's ``OnlineRGround`` client
without editing it.
"""

from __future__ import annotations

import json as _json
import math
import os as _os
from collections.abc import Mapping, Sequence
from typing import Any, Callable

from dagspaces.common.json_extraction import extract_json_from_text
from dagspaces.common.vllm_inference import _strip_think_blocks

from .answerer_client import AnswererClient
from .deontic import FORCE_TO_GOLD, candidate_appropriateness_consistency
from .deontic_distance import parse_battery_completion, score_battery
from .probes import apply_null_filter, sample_probes

__all__ = [
    "ModularReward",
    "is_modular_composition",
    "compute_module_weights",
    "valid_gate",
    "GateResult",
    "attach_probes",
    "make_modular_reward_from_cfg",
    "build_modular_dataset",
]

# The five core CI-tuple fields every flow must carry (R-VALID criterion 4).
CORE_FIELDS: tuple[str, ...] = (
    "sender",
    "recipient",
    "subject",
    "information_type",
    "transmission_principle",
)

# Fields the R-VALID length cap applies to (criterion 5, anti-content-stuffing).
# Exactly the answerer-visible structured fields (answerer_client._STRUCTURED_FIELDS,
# which now includes ``context`` per the 2026-07-24 whitelist decision) — a cap on
# any field the answerer reads closes the smuggling channel.
CAPPED_FIELDS: tuple[str, ...] = (
    "sender",
    "recipient",
    "subject",
    "information_type",
    "transmission_principle",
    "context",
    "appropriateness",
    # v2 (2026-07-25): norms_invoked became answerer-visible, so it joins the
    # anti-content-stuffing cap. `appropriateness` stays capped although it is
    # no longer sent — capping a field the answerer cannot see costs nothing
    # and keeps the gate's contract stable if the whitelist changes again.
    "norms_invoked",
)

# Per-flow field token cap (reward-valid.md: "order-of-64 tokens/field").
FIELD_CAP_TOKENS = 64

# The answerer group-failure fallback and the judge group-failure fallback share
# one constant: deliberate zero-advantage neutrality, never noise
# (reward-outcome.md "Failure handling"; reward-ground.md "Failure fallback").
GROUP_NEUTRAL = 0.5

# Floor for a VALID, gold-YES, flow-bearing (normal-path) completion's composite
# score. reward-abstain.md pins the ordering invariant
# invalid(0) < wrong-abstention(0.1) < unverifiable(0.4) < correct-abstention(0.6):
# a schema-valid extraction must never score at or below the invalid gate (0),
# nor lose to a wrong abstention (0.1), else abstaining dominates engaging on
# gold-YES chunks — exactly the over-abstention the table exists to prevent.
# EM alone can reach 0.0 for a valid-but-useless extraction (all probes wrong),
# which in the core cell (weight 1.0) ties the invalid floor and is beaten by
# 0.1. This floor binds ONLY in that pathological case; it sits strictly in
# (wrong-abstention 0.1, unverifiable 0.4) so engaging beats wrong-abstaining
# while a useless gold-YES extraction still ranks below a neutral gold-NO one.
# The EM gradient is preserved wherever the composite already exceeds it.
# (Design choice 2026-07-24 — the docs assert the invariant but not its
# enforcement; value/mechanism are a knob, not load-bearing at exactness.)
VALID_PATH_FLOOR = 0.15


def _is_embedding_abort(exc: BaseException) -> bool:
    """True if exc is EmbeddingClient's fail-loud corpus-wide abort (clients.py).

    Matched by its stable contract signature rather than a custom type, so no
    shared-plumbing edit is needed. Distinguishes a persistent-outage abort
    (must crash) from an ordinary transient scorer error (group-neutral).
    """
    return isinstance(exc, RuntimeError) and "aborting instead of training" in str(exc)

_MODULAR = "modular"
_AUX_NAMES: tuple[str, ...] = ("ground", "contrast")


# ---------------------------------------------------------------------------
# Dispatch + weights (pure, testable)
# ---------------------------------------------------------------------------
def is_modular_composition(composition: Any) -> bool:
    """Whether a ``reward_composition`` value selects the modular stack.

    The keeper's ``directional`` (and legacy ``additive`` / ``gated``) values
    return ``False`` and must keep selecting the frozen ``CompositeRewardFunction``
    path — the config-dispatch invariant asserted by the keeper guard tests.
    """
    return str(composition).strip().lower() == _MODULAR


def compute_module_weights(
    auxiliaries: Sequence[str], reward_core: bool = True
) -> dict[str, float]:
    """The fixed 2:1 outcome:auxiliary weight rule (ablation-protocol.md).

    The outcome core weighs **2× each active auxiliary**; auxiliaries are equal
    among themselves; weights are normalized to sum 1. ``reward_core=False``
    (the ``-outcome`` cell) drops the outcome term and splits weight equally
    among the active auxiliaries. Returns a dict over the *active* module names
    only (``outcome`` / ``ground`` / ``contrast``).

    Worked table (matches ablation-protocol.md):
      full      → outcome 0.50, ground 0.25, contrast 0.25
      -ground   → outcome 0.67,             contrast 0.33
      -contrast → outcome 0.67, ground 0.33
      core      → outcome 1.00
      -outcome  →             ground 0.50, contrast 0.50
    """
    active_aux = [a for a in _AUX_NAMES if a in set(auxiliaries)]
    parts: dict[str, float] = {}
    if reward_core:
        parts["outcome"] = 2.0
    for a in active_aux:
        parts[a] = 1.0
    total = sum(parts.values())
    if total <= 0.0:
        # Degenerate cell (no core, no auxiliaries) — nothing to score on the
        # normal path. Caller/config validation should prevent this; return an
        # empty map so the normal-path score collapses to 0.0 rather than div0.
        return {}
    return {k: v / total for k, v in parts.items()}


# ---------------------------------------------------------------------------
# R-VALID gate (pure, testable)
# ---------------------------------------------------------------------------
class GateResult:
    """Outcome of :func:`valid_gate`: pass flag, parsed object, failure reason.

    ``reason`` is ``None`` on pass, else one of ``"parse"`` / ``"schema"`` /
    ``"consistency"`` / ``"core_fields"`` / ``"field_cap"`` — the first criterion
    that failed (criteria are checked in the reward-valid.md order).
    ``parsed`` is the JSON object when it parsed (even if a later criterion
    failed), else ``None``. ``no_flow`` is ``True`` for a schema-valid no-flow
    declaration (``has_information_exchange == False`` and zero flows).
    """

    __slots__ = ("passed", "parsed", "reason", "no_flow", "flows")

    def __init__(
        self,
        passed: bool,
        parsed: dict | None,
        reason: str | None,
        no_flow: bool,
        flows: list,
    ):
        self.passed = passed
        self.parsed = parsed
        self.reason = reason
        self.no_flow = no_flow
        self.flows = flows

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (
            f"GateResult(passed={self.passed}, reason={self.reason!r}, "
            f"no_flow={self.no_flow}, n_flows={len(self.flows)})"
        )


def _field_token_len(value: Any) -> int:
    """Whitespace-token length of a field value (lists joined)."""
    if value is None:
        return 0
    if isinstance(value, (list, tuple)):
        return sum(len(str(v).split()) for v in value)
    return len(str(value).split())


def valid_gate(text: str, field_cap_tokens: int = FIELD_CAP_TOKENS) -> GateResult:
    """The R-VALID binary gate (reward-valid.md, five criteria in order).

    1. **Parses** — a single JSON object (``<think>`` stripped first). Uses
       ``extract_json_from_text`` with ``repair=False`` deliberately: a truncated
       or broken completion must *fail* the gate (it is the short no-flow path
       ``mask_truncated_completions`` exists to reject), never be salvaged.
    2. **Schema** — top-level ``reasoning``, ``has_information_exchange`` (bool),
       ``flows`` (list) all present.
    3. **Consistency** — ``has_information_exchange == (len(flows) > 0)``.
    4. **Core fields** — every flow has non-empty ``sender`` / ``recipient`` /
       ``subject`` / ``information_type`` / ``transmission_principle``.
    5. **Field caps** — no capped field exceeds ``field_cap_tokens`` tokens.

    Pass → route per A-ABSTAIN / the scored path. Fail → R = 0, strictly beneath
    every abstention-table entry (the *invalid < wrong-but-valid* ordering).
    """
    stripped = _strip_think_blocks(text or "")
    obj, _err = extract_json_from_text(stripped, repair=False)

    # Criterion 1: parses to a JSON object.
    if not isinstance(obj, dict):
        return GateResult(False, None, "parse", False, [])

    # Criterion 2: schema.
    has_ex = obj.get("has_information_exchange")
    flows = obj.get("flows")
    if (
        "reasoning" not in obj
        or not isinstance(has_ex, bool)
        or not isinstance(flows, list)
    ):
        return GateResult(False, obj, "schema", False, [])

    no_flow = has_ex is False and len(flows) == 0

    # Criterion 3: internal consistency.
    if has_ex != (len(flows) > 0):
        return GateResult(False, obj, "consistency", no_flow, flows)

    # Criterion 4: core fields present and non-empty on every flow.
    for f in flows:
        if not isinstance(f, dict):
            return GateResult(False, obj, "core_fields", no_flow, flows)
        for key in CORE_FIELDS:
            v = f.get(key)
            if v is None or str(v).strip() == "":
                return GateResult(False, obj, "core_fields", no_flow, flows)

    # Criterion 5: field caps.
    for f in flows:
        for key in CAPPED_FIELDS:
            if key in f and _field_token_len(f.get(key)) > field_cap_tokens:
                return GateResult(False, obj, "field_cap", no_flow, flows)

    return GateResult(True, obj, None, no_flow, flows)


# ---------------------------------------------------------------------------
# Dataset-build hook: probe attachment (pure, testable)
# ---------------------------------------------------------------------------
def attach_probes(
    pool: list[dict],
    chunk_id: str,
    *,
    k_max: int = 4,
    null_correct_frac: Mapping[str, float] | None = None,
    p_null: float = 0.8,
) -> list[dict]:
    """Attach the row's pre-sampled probes at dataset build (deterministic).

    Applies the null-answerability filter (:func:`probes.apply_null_filter`) —
    dropping probes a bare empty extraction already answers — then the
    force-stratified, ``chunk_id``-seeded K-sample (:func:`probes.sample_probes`).
    Both are frozen contracts; this is the single call site the reward's dataset
    build uses so the two steps always compose in the right order. Same
    ``chunk_id`` ⇒ identical probe ids (the determinism the prescreen cache and
    the group-shared test rely on).
    """
    filtered = (
        apply_null_filter(pool, null_correct_frac, p_null=p_null)
        if null_correct_frac is not None
        else list(pool)
    )
    return sample_probes(filtered, chunk_id, k_max=k_max)


# ---------------------------------------------------------------------------
# The reward
# ---------------------------------------------------------------------------
# An auxiliary scorer maps a group's normal-path (completions, prompts,
# metadata) to a per-completion score in [0, 1]; on its own failure it returns
# uniform GROUP_NEUTRAL (never raises into the gradient).
AuxScorer = Callable[..., Sequence[float]]

#: (flow, source_id) -> (gold appropriateness | None, retrieval margin).
#: None gold => the flow is unscored (permitted/unknown force, or no norm
#: retrieved). Injected so ModularReward stays free of embedding plumbing;
#: the production adapter lives in aux_scorers.make_direct_gold_fn.
DirectGoldFn = Callable[[dict, str], "tuple[str | None, float]"]


def match_flows(sims, tau: float) -> list[tuple[int, int, float]]:
    """Greedy one-to-one matching over a (teacher, policy) similarity matrix.

    Highest-similarity pair first, each side used at most once, pairs below
    ``tau`` never match. Returns ``(teacher_idx, policy_idx, sim)`` triples.
    Pure and dependency-light so the chunk-denominator scorer is unit-testable
    without an embedding server.
    """
    import numpy as _np

    s = _np.asarray(sims, dtype=_np.float32)
    if s.ndim != 2 or 0 in s.shape:
        return []
    # NaN guard (audit 2026-07-28): argsort places NaN last ascending, so the
    # descending order would put NaN FIRST, and `nan < tau` is False — a NaN
    # pair would be matched and poison diag/match_sim. Map NaN below any tau.
    s = _np.nan_to_num(s, nan=-1.0)
    order = _np.argsort(s, axis=None)[::-1]
    used_t: set[int] = set()
    used_p: set[int] = set()
    out: list[tuple[int, int, float]] = []
    for flat in order:
        t, p = divmod(int(flat), s.shape[1])
        sim = float(s[t, p])
        if sim < tau:
            break  # descending order: nothing later can pass
        if t in used_t or p in used_p:
            continue
        used_t.add(t)
        used_p.add(p)
        out.append((t, p, sim))
    return out

#: How the policy's own appropriateness label is normalised before comparison.
#: Anything that is not a decisive commitment is a hedge and scores 0 — the
#: tooth inherited from R-OUTCOME: hedging must never beat committing.
_DECISIVE_LABELS = ("appropriate", "inappropriate")


class ModularReward:
    """The modular reward callable (TRL reward-function shape).

    Callable as ``reward_fn(prompts=..., completions=..., **kwargs) -> list[float]``,
    the same shape TRL calls :class:`~.rewards.CompositeRewardFunction` with, and
    with the same row-metadata plumbing: TRL does not forward dataset columns, so
    per-prompt metadata (task type, gold label, pre-sampled probes, battery gold)
    is looked up from :attr:`prompt_metadata` keyed by the (chat-templated) prompt
    string that the dataset carries.

    Per completion:
      * **T-VIGNETTE** rows → ``parse_battery_completion`` + ``score_battery``
        → ``r_vig`` (= the re-anchored battery mean; cite is diagnostic-only
        since 2026-07-28).
      * **T-EXTRACT** rows → R-VALID gate (fail ⇒ 0.0) → A-ABSTAIN routing
        table (no-flow / gold-NO / unknown rows scored by the fixed table, no
        server calls) → survivors (gold-YES valid extractions) scored by
        ``w·[R-OUTCOME] + Σ w·[auxiliary]`` under the 2:1 weight rule.
    """

    __name__ = "modular_ci_reward"

    def __init__(
        self,
        *,
        auxiliaries: Sequence[str] = (),
        reward_core: bool = True,
        core_mode: str = "direct",
        direct_gold_fn: "DirectGoldFn | None" = None,
        direct_chunk_gold=None,
        # tau calibrated 2026-07-28 (outputs/2026-07-28_tau_calibration):
        # signal/null separation plateaus at 0.55-0.575 (J=0.70); 0.55 is the
        # signal-preserving end (87.5% true-match retention, 17.5% null) —
        # a rejected true match becomes a recall miss that punishes the
        # completion directly, a false accept is diluted by greedy 1:1.
        direct_match_threshold: float = 0.55,
        answerer: AnswererClient | None = None,
        ground_scorer: AuxScorer | None = None,
        contrast_scorer: AuxScorer | None = None,
        abstain: Mapping[str, float] | None = None,
        prompt_metadata: dict[str, dict[str, Any]] | None = None,
        field_cap_tokens: int = FIELD_CAP_TOKENS,
        trace_log_path: str = "",
        trace_every_n_calls: int = 1,
        trace_max_bytes: int = 200_000_000,
    ):
        self.auxiliaries = [a for a in _AUX_NAMES if a in set(auxiliaries)]
        self.reward_core = bool(reward_core)
        # "direct"  — R-DIRECT: the norm classifies the flow; score the policy's
        #             own appropriateness label against it. No model in the loop.
        # "outcome" — the legacy frozen-answerer core. Retained only so the
        #             negative result stays reproducible (reward-direct-spec.md);
        #             not used by any proposed keeper config.
        self.core_mode = str(core_mode)
        self.weights = compute_module_weights(self.auxiliaries, self.reward_core)
        self._direct_gold_fn = direct_gold_fn
        # Chunk-denominator R-DIRECT (R2, 2026-07-28): when a DirectChunkGold
        # index is attached (dataset build), scored completions are matched
        # against the CHUNK's teacher flows and macro-averaged over the
        # chunk's gold classes — the denominator is fixed and identical for
        # all G siblings, so the policy no longer curates its own exam.
        # Without an index (tests, missing chunks) the per-flow path applies.
        self._direct_chunk_gold = direct_chunk_gold
        self.direct_match_threshold = float(direct_match_threshold)
        self.answerer = answerer
        self._ground_scorer = ground_scorer
        self._contrast_scorer = contrast_scorer
        _ab = dict(abstain or {})
        self.abstain = {
            "wrong": float(_ab.get("wrong", 0.1)),
            "correct": float(_ab.get("correct", 0.6)),
            "unknown": float(_ab.get("unknown", 0.4)),
        }
        self.prompt_metadata = prompt_metadata or {}
        self.field_cap_tokens = int(field_cap_tokens)
        # Populated after every __call__ (tests + offline reads; W&B mirror).
        self.last_metrics: dict[str, float] = {}
        # Set by grpo_training.py for trace continuity; unused by scoring.
        self.enable_thinking_grpo: bool | None = None
        # ---- reward traces (parity with the keeper's reward_traces.jsonl) ----
        # Until 2026-07-25 the modular path wrote NO traces while
        # grpo_training.py printed "Reward traces → …" for both branches, so
        # the m1 wave produced none despite claiming to. Traces are the
        # forensics substrate the ablation protocol reports from, and the only
        # way to recompute alternative scorings (e.g. micro- vs macro-EM) after
        # the fact. Failures here are swallowed: tracing must never perturb a run.
        self._trace_path = str(trace_log_path or "")
        self._trace_every = max(1, int(trace_every_n_calls))
        self._trace_max_bytes = int(trace_max_bytes)
        self._call_count = 0
        self._trace_writes = 0
        # Per-call scratch: completion index -> {probe_ids, golds, answers}.
        self._probe_io: dict[int, dict[str, Any]] = {}
        # Per-call scratch for the chunk-denominator direct core: completion
        # index -> {direct_flows, direct_missed, direct_spurious} (R5: traces
        # must carry per-flow gold/pred so discrimination is recomputable from
        # disk — the m1 W&B crash made core's tail unrecoverable).
        self._direct_io: dict[int, dict[str, Any]] = {}
        # Persistent gate-failure counter: the per-call accumulator resets
        # each __call__, which made the "1-in-8" text sampling actually
        # sample the first failure of every call (audit 2026-07-28).
        self._gatefail_seen = 0
        # Per-call scratch for battery rows: completion index ->
        # {model_forces, vig_result} (R5: the re-anchored battery scale is
        # wave 2's biggest behavioural change — whether hedging falls must be
        # answerable from disk, per row, not only from W&B aggregates).
        self._vignette_io: dict[int, dict[str, Any]] = {}

    def set_direct_chunk_gold(self, chunk_gold) -> None:
        """Attach the chunk-gold index (called by the dataset-build hook)."""
        self._direct_chunk_gold = chunk_gold

    # ---- reward traces ---------------------------------------------------
    def _should_trace(self) -> bool:
        if not self._trace_path:
            return False
        return (self._call_count == 0) or (self._call_count % self._trace_every == 0)

    _trace_warned = False

    def _log_trace(self, entries: list[dict[str, Any]]) -> None:
        """Append trace rows as JSONL. Never raises — tracing is observational."""
        if not self._trace_path or not entries:
            return
        try:
            _os.makedirs(_os.path.dirname(self._trace_path), exist_ok=True)
            with open(self._trace_path, "a", encoding="utf-8") as f:
                for entry in entries:
                    f.write(_json.dumps(entry, ensure_ascii=False, default=str) + "\n")
            self._maybe_truncate_trace()
        except Exception as exc:
            # Never perturbs scoring — but never silently either: the m1
            # lesson is that unlogged data is unrecoverable (warn once).
            if not ModularReward._trace_warned:
                ModularReward._trace_warned = True
                print(f"[modular_reward] WARNING trace write failed "
                      f"({exc!r}); further failures will be silent")

    def _maybe_truncate_trace(self) -> None:
        """Keep the trace file bounded (mirrors the keeper's policy): when it
        exceeds ``_trace_max_bytes``, keep the newest half on whole-line
        boundaries. Checked every 100 writes so the cost stays negligible."""
        self._trace_writes += 1
        if self._trace_writes % 100:
            return
        try:
            if _os.path.getsize(self._trace_path) <= self._trace_max_bytes:
                return
            keep = self._trace_max_bytes // 2
            with open(self._trace_path, "rb") as f:
                f.seek(-keep, _os.SEEK_END)
                tail = f.read()
            tail = tail[tail.index(b"\n") + 1 :]
            with open(self._trace_path, "wb") as f:
                f.write(tail)
            print(f"[modular_reward] reward_traces.jsonl exceeded "
                  f"{self._trace_max_bytes} bytes — truncated to newest half")
        except Exception:
            pass

    # ---- helpers ---------------------------------------------------------
    @staticmethod
    def _extract_text(completion: Any) -> str:
        """Plain assistant text with ``<think>`` blocks stripped (TRL may hand a
        string or a ``[{"role","content"}]`` conversational list)."""
        if isinstance(completion, str):
            text = completion
        elif isinstance(completion, list):
            text = ""
            for msg in completion:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    text = msg.get("content", "")
                    break
            if not text:
                text = " ".join(
                    msg.get("content", "")
                    for msg in completion
                    if isinstance(msg, dict)
                )
        else:
            text = str(completion)
        return _strip_think_blocks(text)

    @classmethod
    def _extract_raw_text(cls, completion: Any) -> str:
        """Assistant text WITHOUT think-stripping (trace forensics only).

        A runaway ``<think>`` block strips to ``""`` — sampling the stripped
        text would log an empty string for exactly the gate-failure mode the
        sample exists to diagnose (audit 2026-07-28).
        """
        if isinstance(completion, str):
            return completion
        if isinstance(completion, list):
            for msg in completion:
                if isinstance(msg, dict) and msg.get("role") == "assistant":
                    return msg.get("content", "")
            return " ".join(
                msg.get("content", "")
                for msg in completion
                if isinstance(msg, dict)
            )
        return str(completion)

    @staticmethod
    def _prompt_key(prompt: Any) -> str:
        """Metadata key for a prompt (mirrors the keeper's convention)."""
        if isinstance(prompt, list):
            return " ".join(
                m.get("content", "")
                for m in prompt
                if isinstance(m, dict) and m.get("role") == "user"
            )
        return str(prompt) if prompt is not None else ""

    def _abstain_score(self, gold_has_exchange: Any, no_flow: bool) -> float:
        """The A-ABSTAIN four-entry routing table (reward-abstain.md).

        Only the *decidable* branches return a table value; a gold-YES extraction
        returns ``None`` (the normal scored path). Every other cell — no-flow on
        any gold, and any extraction on gold-NO / unknown-gold — is a constant.
        """
        if no_flow:
            if gold_has_exchange is True:
                return self.abstain["wrong"]  # 0.1 wrong abstention
            if gold_has_exchange is False:
                return self.abstain["correct"]  # 0.6 correct abstention
            return self.abstain["unknown"]  # 0.4 unknown gold
        # Extraction (flows present).
        if gold_has_exchange is True:
            return None  # normal path
        # gold-NO extraction (unverifiable, neutral) or unknown gold → 0.4.
        return self.abstain["unknown"]

    def _score_aux(
        self,
        kind: str,
        texts: list[str],
        prompts: list[str],
        metas: list[dict],
    ) -> tuple[list[float], bool]:
        """Run one injected auxiliary scorer over a group; group-neutral on failure."""
        scorer = self._ground_scorer if kind == "ground" else self._contrast_scorer
        n = len(texts)
        if scorer is None:
            raise RuntimeError(
                f"auxiliary {kind!r} is active but no {kind}_scorer was injected "
                "(make_modular_reward_from_cfg wires the production adapter)"
            )
        try:
            scores = scorer(completions=texts, prompts=prompts, metadata_list=metas)
        except Exception as exc:  # deliberate neutrality, never noise
            if _is_embedding_abort(exc):
                # The embedding client's deliberate fail-loud abort (raised
                # after N consecutive outages, clients.py) means grounding
                # retrieval is broken corpus-wide — training through it would
                # silently corrupt every aux score. Propagate: a persistent
                # embedding outage must crash the run, not neutralize the
                # auxiliary (which would silently delete ground/contrast for
                # the rest of training). Restores the keeper's fail-loud
                # contract on the modular path (2026-07-24).
                raise
            print(f"[modular_reward] WARNING {kind} scorer failed for group: {exc}")
            return [GROUP_NEUTRAL] * n, True
        if scores is None or len(list(scores)) != n:
            print(f"[modular_reward] WARNING {kind} scorer returned wrong length")
            return [GROUP_NEUTRAL] * n, True
        return [float(s) for s in scores], False

    # ---- the callable ----------------------------------------------------
    def __call__(
        self,
        *,
        prompts: Sequence[Any] | None = None,
        completions: Sequence[Any],
        **kwargs: Any,
    ) -> list[float]:
        n = len(completions)
        # Clear per-call scratch at ENTRY: an abort raised mid-call (embedding
        # fail-loud) must not leak stale io rows into a later call's traces.
        self._probe_io = {}
        self._direct_io = {}
        self._vignette_io = {}
        texts = [self._extract_text(c) for c in completions]
        keys = [self._prompt_key(prompts[i]) if prompts else "" for i in range(n)]
        metas = [self.prompt_metadata.get(k, {}) for k in keys]

        scores: list[float] = [0.0] * n

        # Accounting for W&B (built into last_metrics at the end).
        acc = _MetricAccumulator()

        # Per-completion classification; collect the normal-path groups.
        # normal_groups: prompt_key -> list of global indices (gold-YES, valid,
        # flow-bearing extractions — the only rows the outcome/aux path scores).
        normal_groups: dict[str, list[int]] = {}
        gate_pass_flows: dict[int, list] = {}  # i -> parsed flows for the answerer
        route: dict[int, dict[str, Any]] = {}  # i -> trace detail (observational)

        for i in range(n):
            meta = metas[i]
            task_type = str(meta.get("task_type", "extract"))

            if task_type == "vignette":
                scores[i] = self._score_vignette(texts[i], meta, acc, i)
                route[i] = {"task_type": "vignette", "route": "vignette"}
                continue

            # T-EXTRACT.
            acc.n_extract += 1
            gate = valid_gate(texts[i], self.field_cap_tokens)
            if not gate.passed:
                acc.gate_fail += 1
                scores[i] = 0.0
                route[i] = {"task_type": "extract", "route": "gate_fail",
                            "gate_reason": gate.reason}
                # R5: sample failing text into the trace (1-in-8 failures
                # run-wide, capped). Sample the RAW completion, not the
                # think-stripped text — a runaway <think> block strips to ""
                # and is exactly the failure mode this exists to diagnose.
                self._gatefail_seen += 1
                if self._gatefail_seen % 8 == 1:
                    raw = self._extract_raw_text(completions[i])
                    route[i]["completion_text_sample"] = str(raw)[:600]
                continue

            gold = meta.get("gold_has_exchange")
            acc.observe_abstain(gold, gate.no_flow)

            table = self._abstain_score(gold, gate.no_flow)
            if table is not None:
                scores[i] = table
                route[i] = {"task_type": "extract", "route": "abstain_table",
                            "gold_has_exchange": gold, "no_flow": gate.no_flow}
                continue

            # Normal path (gold-YES, valid, flow-bearing).
            normal_groups.setdefault(keys[i], []).append(i)
            gate_pass_flows[i] = gate.flows
            route[i] = {"task_type": "extract", "route": "scored",
                        "gold_has_exchange": gold, "n_flows": len(gate.flows)}

        # ---- score the normal-path groups -------------------------------
        outcome_term: dict[int, float] = {}
        aux_term: dict[str, dict[int, float]] = {"ground": {}, "contrast": {}}

        for key, group in normal_groups.items():
            g_texts = [texts[i] for i in group]
            g_prompts = [keys[i] for i in group]
            g_metas = [metas[i] for i in group]

            # Verifiable core.
            if self.reward_core and self.core_mode == "direct":
                for i in group:
                    if self._direct_chunk_gold is not None:
                        val = self._score_direct_chunk(
                            gate_pass_flows.get(i, []), metas[i], acc, i
                        )
                    else:
                        val = self._score_direct(
                            gate_pass_flows.get(i, []), metas[i], acc
                        )
                    outcome_term[i] = GROUP_NEUTRAL if val is None else val
            elif self.reward_core and self.answerer is not None:
                self._score_outcome_group(
                    group, gate_pass_flows, metas, outcome_term, acc
                )
            elif self.reward_core:  # core active but nothing wired
                for i in group:
                    outcome_term[i] = GROUP_NEUTRAL

            # Auxiliaries.
            for kind in self.auxiliaries:
                vals, failed = self._score_aux(kind, g_texts, g_prompts, g_metas)
                if failed:
                    acc.aux_failed[kind] += 1
                acc.aux_groups[kind] += 1
                for pos, i in enumerate(group):
                    aux_term[kind][i] = vals[pos]

            # Combine + diagnostic direction consistency for this group.
            for i in group:
                r = 0.0
                w = self.weights
                if "outcome" in w:
                    r += w["outcome"] * outcome_term.get(i, GROUP_NEUTRAL)
                for kind in self.auxiliaries:
                    if kind in w:
                        r += w[kind] * aux_term[kind].get(i, GROUP_NEUTRAL)
                # Ordering-invariant floor: a valid gold-YES extraction never
                # falls to/below the invalid gate or a wrong abstention.
                scores[i] = max(r, VALID_PATH_FLOOR)
                acc.observe_direction(gate_pass_flows[i], metas[i])

            # group_spread: within-group std of the outcome term (advantage carrier).
            if self.reward_core:
                acc.observe_group_spread(
                    [outcome_term.get(i, GROUP_NEUTRAL) for i in group]
                )

        self.last_metrics = acc.build(self)
        _push_metrics(self.last_metrics)

        # ---- reward traces (observational; never perturbs scoring) ---------
        if self._should_trace():
            rows = []
            for i in range(n):
                meta = metas[i]
                det = route.get(i, {})
                row = {
                    "call": self._call_count,
                    "reward_composition": "modular",
                    "auxiliaries": list(self.auxiliaries),
                    "reward_core": self.reward_core,
                    "weights": self.weights,
                    "prompt_key": keys[i],
                    "chunk_id": meta.get("chunk_id"),
                    "source_id": meta.get("source_id"),
                    "score": scores[i],
                    **det,
                }
                if det.get("route") == "scored":
                    row["outcome_term"] = outcome_term.get(i)
                    row["aux_terms"] = {
                        k: aux_term[k].get(i) for k in self.auxiliaries
                    }
                    row.update(self._probe_io.get(i, {}))
                    row.update(self._direct_io.get(i, {}))
                elif det.get("route") == "vignette":
                    row["battery_id"] = meta.get("battery_id")
                    row["gold_forces"] = [
                        g.get("gold_force") for g in (meta.get("gold_items") or [])
                    ]
                    row.update(self._vignette_io.get(i, {}))
                rows.append(row)
            self._log_trace(rows)
        self._call_count += 1
        self._probe_io = {}
        self._direct_io = {}
        self._vignette_io = {}
        return scores

    # ---- component scorers ----------------------------------------------
    def _score_vignette(
        self, text: str, meta: dict, acc: "_MetricAccumulator", idx: int = -1
    ) -> float:
        """Score a T-VIGNETTE row on the re-anchored battery scale."""
        gold_items = meta.get("gold_items") or []
        k = len(gold_items)
        if k == 0:
            return 0.0
        parsed = parse_battery_completion(text, k)
        result = score_battery(parsed, gold_items)
        acc.observe_vignette(result)
        if idx >= 0:
            # R5: per-row battery forensics for the trace — answered forces
            # (None = unparsed slot) + the full result dict, so hedge drift
            # under the re-anchored scale is recomputable from disk.
            self._vignette_io[idx] = {
                "model_forces": [
                    (p.get("force") if isinstance(p, dict) else None)
                    for p in parsed
                ],
                "vig_result": {kk: round(float(v), 4)
                               for kk, v in result.items()},
            }
        return float(result["r_vig"])

    def _score_direct_chunk(
        self, flows: list, meta: dict, acc: "_MetricAccumulator", idx: int
    ) -> float | None:
        """R-DIRECT with the chunk-fixed denominator (R2, 2026-07-28).

        The chunk's teacher flows carry the golds (k=3-majority over the
        restricted index, precomputed at build). The completion's flows are
        embedding-matched one-to-one against them (greedy, cosine >=
        ``direct_match_threshold``); each matched teacher flow scores the
        policy's ``appropriateness`` label (hedge/wrong 0.0, right 1.0), each
        UNMATCHED teacher flow scores 0.0 to its class — omitting a governed
        flow now costs recall, closing the curate-your-own-exam hole. The
        return is the macro-average over the classes present in the CHUNK's
        golds, identical for all G siblings. Unmatched policy flows are not
        scored (counted as ``spurious``). Falls back to the per-flow path for
        chunks absent from the index; embedding failure returns None (the
        caller applies GROUP_NEUTRAL — never a penalty for infra faults).
        """
        entry = (
            self._direct_chunk_gold.get(
                str(meta.get("source_id", "")), str(meta.get("chunk_id", ""))
            )
            if self._direct_chunk_gold is not None
            else None
        )
        if entry is None:
            return self._score_direct(flows, meta, acc)

        golds: list[str] = list(entry["golds"])
        p_flows = [f for f in (flows if isinstance(flows, list) else [])
                   if isinstance(f, dict)]
        acc.direct_completions += 1
        matches: list[tuple[int, int, float]] = []
        if p_flows:
            try:
                p_emb = self._direct_chunk_gold.embed_flows(p_flows)
            except Exception as exc:
                # A dead embedding server must ABORT training, exactly as the
                # aux path does — scoring through it silently corrupts the
                # core term (audit 2026-07-28, gates C1).
                if _is_embedding_abort(exc):
                    raise
                acc.direct_embed_failed += 1
                return None
            # EmbeddingClient degrades to ZERO vectors on transient failures
            # (it raises only after 3 consecutive) — a zero row would match
            # nothing and score a correct completion 0.0. Infra faults are
            # group-neutral, never a penalty (audit 2026-07-28, R2-M2).
            import numpy as _np
            if not _np.asarray(p_emb).any(axis=1).all():
                acc.direct_embed_failed += 1
                return None
            sims = entry["emb"] @ p_emb.T
            matches = match_flows(sims, self.direct_match_threshold)

        matched_t = {t: (p, sim) for t, p, sim in matches}
        by_class: dict[str, list[float]] = {}
        trace_flows: list[dict[str, Any]] = []
        for t, gold in enumerate(golds):
            if t in matched_t:
                p, sim = matched_t[t]
                label = str(p_flows[p].get("appropriateness") or "").strip().lower()
                if label not in _DECISIVE_LABELS:
                    acc.direct_hedge += 1
                    hit = 0.0
                else:
                    hit = 1.0 if label == gold else 0.0
                    if hit == 0.0:
                        acc.direct_antithesis += 1
                # Label-semantics accumulator: MATCHED flows only, so
                # agreement/balanced_accuracy/youden_j keep m1-comparable
                # meaning ("given a correspondence, is the label right").
                acc.direct_by_class.setdefault(gold, []).append(hit)
                acc.direct_scored += 1
                acc.direct_match_sims.append(sim)
                trace_flows.append({"gold": gold, "pred": label or None,
                                    "sim": round(sim, 4)})
            else:
                hit = 0.0
                acc.direct_missed_by_class[gold] = (
                    acc.direct_missed_by_class.get(gold, 0) + 1
                )
                acc.direct_missed += 1
                trace_flows.append({"gold": gold, "pred": None, "sim": None})
            # The REWARD prices recall: misses stay in the completion's macro.
            by_class.setdefault(gold, []).append(hit)
        n_spurious = len(p_flows) - len(matches)
        acc.direct_spurious += n_spurious
        acc.direct_policy_flows += len(p_flows)
        self._direct_io[idx] = {
            "direct_flows": trace_flows,
            "direct_spurious": n_spurious,
        }
        if not by_class:
            return None
        return sum(sum(v) / len(v) for v in by_class.values()) / len(by_class)

    def _score_direct(
        self, flows: list, meta: dict, acc: "_MetricAccumulator"
    ) -> float | None:
        """R-DIRECT: the norm classifies the flow; score the policy's own label.

        Per flow: retrieve the governing norm (k=1) and derive the flow's
        appropriateness from force x act_polarity; compare to the policy's own
        ``appropriateness`` label. Exact match 1.0; opposite 0.0; hedge
        (``ambiguous`` / missing / unrecognised) 0.0 — hedging must never beat
        committing. ``permitted``/unknown force and unretrievable flows are
        UNSCORED (they neither earn nor lose).

        Returns the macro-average over the gold classes present, so a blanket
        label scores 0.5 on any completion carrying both classes instead of
        riding the corpus skew. Returns None when nothing was scorable — the
        caller then applies GROUP_NEUTRAL for this term only.
        """
        if self._direct_gold_fn is None:
            raise RuntimeError(
                "core_mode='direct' but no direct_gold_fn was injected "
                "(make_modular_reward_from_cfg wires the production adapter)"
            )
        source_id = str(meta.get("source_id", ""))
        by_class: dict[str, list[float]] = {}
        n_unscored = 0
        for flow in flows if isinstance(flows, list) else []:
            if not isinstance(flow, dict):
                continue
            gold, margin = self._direct_gold_fn(flow, source_id)
            if gold is None:
                n_unscored += 1
                continue
            label = str(flow.get("appropriateness") or "").strip().lower()
            if label not in _DECISIVE_LABELS:
                acc.direct_hedge += 1
                hit = 0.0
            else:
                hit = 1.0 if label == gold else 0.0
                if hit == 0.0:
                    acc.direct_antithesis += 1
            by_class.setdefault(gold, []).append(hit)
            acc.direct_by_class.setdefault(gold, []).append(hit)
            acc.direct_margins.append(float(margin))
            acc.direct_scored += 1
        acc.direct_unscored += n_unscored
        if not by_class:
            return None
        return sum(sum(v) / len(v) for v in by_class.values()) / len(by_class)

    def _score_outcome_group(
        self,
        group: list[int],
        gate_pass_flows: dict[int, list],
        metas: list[dict],
        outcome_term: dict[int, float],
        acc: "_MetricAccumulator",
    ) -> None:
        """R-OUTCOME over one group; group-neutral 0.5 if any answerer call fails.

        Per completion, the frozen answerer answers the row's pre-sampled probes
        from the *structured extraction alone*; the term is mean EM
        (``cannot_determine`` scores 0). If any completion's answerer reply fails
        after one retry, the whole group gets uniform 0.5 for this term — zero
        advantage (reward-outcome.md "Failure handling").
        """
        per_completion_em: dict[int, float] = {}
        group_failed = False

        for i in group:
            probes = metas[i].get("probes") or []
            probe_texts = [p.get("prompt_text", "") for p in probes]
            golds = [p.get("gold") for p in probes]
            if not probe_texts:
                # Empty-pool chunks are excluded at build; defensive neutral.
                per_completion_em[i] = GROUP_NEUTRAL
                continue
            flows = gate_pass_flows.get(i, [])
            # extraction_token_len drift diagnostic (reward-outcome.md): total
            # answerer-visible field tokens per extraction — a content-stuffing
            # canary (the field caps are the actual defense).
            acc.extraction_lens.append(
                sum(_field_token_len(v) for f in flows for v in dict(f).values())
                if isinstance(flows, list) else 0
            )
            result = self.answerer.answer_probes(flows, probe_texts)
            if result.get("failed"):
                group_failed = True
                continue
            answers = result.get("answers", [])
            # Class-balanced EM (2026-07-25): the training probe set is 88.2%
            # gold-yes, so micro-EM hands a blanket-"yes" extraction 0.882.
            # Macro prices it at 0.5 wherever both classes are present.
            per_completion_em[i] = AnswererClient.em_macro(answers, golds)
            acc.observe_probes(answers, golds)
            # Retain answers+golds so alternative scorings (micro-EM, per-class,
            # re-derived gold) stay recoverable from traces after the fact.
            self._probe_io[i] = {
                "probe_ids": [p.get("probe_id") for p in probes],
                "golds": list(golds),
                "answers": list(answers),
            }

        if group_failed:
            acc.answerer_failed += len(group)
            for i in group:
                outcome_term[i] = GROUP_NEUTRAL
        else:
            for i in group:
                outcome_term[i] = per_completion_em.get(i, GROUP_NEUTRAL)
        acc.n_outcome += len(group)


# ---------------------------------------------------------------------------
# W&B metric accumulation
# ---------------------------------------------------------------------------
class _MetricAccumulator:
    """Rolls per-call module statistics into the per-namespace W&B dict.

    Namespaces mirror the module boundaries (README.md "per-module namespaces"):
    ``reward/outcome/*``, ``vignette/*``, ``abstain/*``, ``reward/valid/*``,
    ``diag/*`` — the same ``commit=False`` streaming discipline the keeper's
    ``CompositeRewardFunction`` uses so values merge into TRL's step commit.
    """

    def __init__(self) -> None:
        self.n_extract = 0
        self.gate_fail = 0
        # abstain
        self.no_flow = 0
        self.wrong_abstention = 0  # no-flow on gold-YES
        self.n_gold_yes = 0
        self.goldno_extraction = 0
        self.n_gold_no = 0
        # outcome
        self.n_outcome = 0
        self.answerer_failed = 0
        self.em_by_force: dict[str, list[float]] = {"yes": [], "no": []}
        self.cannot_determine = 0
        self.n_probe_slots = 0
        self.group_spreads: list[float] = []
        self.extraction_lens: list[int] = []  # answerer-visible field token count
        # direct core
        self.direct_scored = 0
        self.direct_unscored = 0
        self.direct_hedge = 0
        self.direct_antithesis = 0
        self.direct_by_class: dict[str, list[float]] = {}
        self.direct_margins: list[float] = []
        # chunk-denominator direct core (R2)
        self.direct_missed = 0     # teacher flows no policy flow matched
        self.direct_missed_by_class: dict[str, int] = {}
        self.direct_spurious = 0   # policy flows matching no teacher flow
        self.direct_policy_flows = 0
        self.direct_completions = 0
        self.direct_embed_failed = 0  # non-abort embed faults (group-neutral)
        self.direct_match_sims: list[float] = []
        # vignette
        self.vig_antithesis: list[float] = []
        self.vig_hedge: list[float] = []
        self.vig_battery: list[float] = []
        self.vig_cite: list[float] = []
        self.vig_parsed: list[float] = []
        # aux
        self.aux_failed: dict[str, int] = {"ground": 0, "contrast": 0}
        self.aux_groups: dict[str, int] = {"ground": 0, "contrast": 0}
        # diagnostic
        self.direction: list[float] = []

    def observe_abstain(self, gold: Any, no_flow: bool) -> None:
        if gold is True:
            self.n_gold_yes += 1
        elif gold is False:
            self.n_gold_no += 1
        if no_flow:
            self.no_flow += 1
            if gold is True:
                self.wrong_abstention += 1
        elif gold is False:
            self.goldno_extraction += 1

    def observe_probes(self, answers: list[str], golds: list[str]) -> None:
        for ans, gold in zip(answers, golds):
            self.n_probe_slots += 1
            if ans == "cannot_determine":
                self.cannot_determine += 1
                em = 0.0
            else:
                em = 1.0 if ans == gold else 0.0
            if gold in ("yes", "no"):
                self.em_by_force[gold].append(em)

    def observe_group_spread(self, terms: list[float]) -> None:
        if len(terms) >= 2:
            self.group_spreads.append(_pstdev(terms))
        else:
            self.group_spreads.append(0.0)

    def observe_vignette(self, result: Mapping[str, float]) -> None:
        self.vig_antithesis.append(float(result.get("antithesis_frac", 0.0)))
        self.vig_hedge.append(float(result.get("hedge_frac", 0.0)))
        self.vig_battery.append(float(result.get("battery", 0.0)))
        self.vig_cite.append(float(result.get("cite", 0.0)))
        self.vig_parsed.append(float(result.get("parsed_frac", 0.0)))

    def observe_direction(self, flows: list, meta: dict) -> None:
        """diag/direction_consistency: agreement of the completion's flow
        appropriateness labels with the direction its probes imply (diagnostic
        only — subsumed as a reward term by R-OUTCOME)."""
        probes = meta.get("probes") or []
        if not probes:
            return
        n_no = sum(1 for p in probes if p.get("gold") == "no")
        n_yes = sum(1 for p in probes if p.get("gold") == "yes")
        if n_no == 0 and n_yes == 0:
            return
        force = "prohibited" if n_no > n_yes else "obligatory"
        self.direction.append(candidate_appropriateness_consistency(flows, force))

    def build(self, reward: "ModularReward") -> dict[str, float]:
        out: dict[str, float] = {}

        # reward/valid/*
        if self.n_extract:
            out["reward/valid/gate_frac"] = 1.0 - self.gate_fail / self.n_extract

        # abstain/*
        if self.n_extract:
            out["abstain/no_flow_frac"] = self.no_flow / self.n_extract
        if self.n_gold_yes:
            out["abstain/wrong_abstention_frac"] = (
                self.wrong_abstention / self.n_gold_yes
            )
        if self.n_gold_no:
            out["abstain/goldno_extraction_frac"] = (
                self.goldno_extraction / self.n_gold_no
            )

        # reward/outcome/*
        if self.n_outcome:
            all_em = self.em_by_force["yes"] + self.em_by_force["no"]
            if all_em:
                out["reward/outcome/em_mean"] = sum(all_em) / len(all_em)
            if self.em_by_force["yes"]:
                out["reward/outcome/em_mean_by_force/yes"] = sum(
                    self.em_by_force["yes"]
                ) / len(self.em_by_force["yes"])
            if self.em_by_force["no"]:
                out["reward/outcome/em_mean_by_force/no"] = sum(
                    self.em_by_force["no"]
                ) / len(self.em_by_force["no"])
            if self.n_probe_slots:
                out["reward/outcome/cannot_determine_frac"] = (
                    self.cannot_determine / self.n_probe_slots
                )
            out["reward/outcome/answerer_failed_frac"] = (
                self.answerer_failed / self.n_outcome
            )
            if self.group_spreads:
                out["reward/outcome/group_spread"] = sum(self.group_spreads) / len(
                    self.group_spreads
                )
            if self.extraction_lens:
                out["reward/outcome/extraction_token_len"] = sum(
                    self.extraction_lens
                ) / len(self.extraction_lens)

        # reward/direct/*
        if self.direct_scored:
            tot = self.direct_scored
            allv = [v for vs in self.direct_by_class.values() for v in vs]
            out["reward/direct/agreement_mean"] = sum(allv) / len(allv)
            recalls = {}
            for cls, vs in self.direct_by_class.items():
                recalls[cls] = sum(vs) / len(vs)
                out[f"reward/direct/agreement_by_class/{cls}"] = recalls[cls]
            # Discrimination, not accuracy. The two per-class recalls are
            # conditioned on disjoint gold subsets, so they do NOT sum to 1 and
            # are easy to misread as complementary: ANY non-discriminating
            # policy (blanket label, coin flip) sums to exactly 1.0, while a
            # perfect one sums to 2.0. Balanced accuracy = mean recall, so 0.5
            # is the blanket floor — and it is exactly what the macro-EM reward
            # computes, making it the number the kill criterion should be read
            # against. agreement_mean is micro-averaged over a ~72/28 split and
            # therefore mostly tracks the majority class (2026-07-26).
            if recalls:
                out["reward/direct/balanced_accuracy"] = sum(recalls.values()) / len(recalls)
            if len(recalls) == 2:
                # Youden's J = sensitivity + specificity - 1. 0 = no
                # discrimination, 1 = perfect. Defined for two classes only.
                out["reward/direct/youden_j"] = sum(recalls.values()) - 1.0
            out["reward/direct/hedge_frac"] = self.direct_hedge / tot
            out["reward/direct/antithesis_frac"] = self.direct_antithesis / tot
            if self.direct_unscored:
                # Per-flow (m1) path only. The chunk path never increments it
                # — emitting a constant 0.0 there would masquerade as live
                # confirmation of the unscored-flow fix (audit 2026-07-28).
                out["reward/direct/unscored_flow_frac"] = (
                    self.direct_unscored / (tot + self.direct_unscored)
                )
            if self.direct_margins:
                out["diag/retrieval_margin"] = (
                    sum(self.direct_margins) / len(self.direct_margins)
                )
        # Chunk-denominator diagnostics — OUTSIDE the direct_scored guard
        # (final review M-1): direct_scored counts MATCHED flows only, so it
        # is 0 exactly when the embedding server degrades or every teacher
        # flow is missed — the two states miss_frac/spurious_flow_frac/
        # embed_failed_frac exist to surface. A key that stops logging
        # flatlines at its last healthy value on the W&B panel.
        if True:
            # (semantics audited 2026-07-28):
            # by_class above is MATCHED flows only, so agreement/
            # balanced_accuracy/youden_j keep label-only meaning. Recall
            # (which the REWARD prices — misses are 0.0 to their class) is
            # emitted separately below.
            n_teacher = self.direct_scored + self.direct_missed
            if self.direct_completions:
                out["reward/direct/miss_frac"] = (
                    self.direct_missed / n_teacher if n_teacher else 0.0
                )
                out["reward/direct/spurious_flow_frac"] = (
                    self.direct_spurious / self.direct_policy_flows
                    if self.direct_policy_flows else 0.0
                )
                out["reward/direct/embed_failed_frac"] = (
                    self.direct_embed_failed / self.direct_completions
                )
                recalls_full = {}
                for cls in set(self.direct_by_class) | set(self.direct_missed_by_class):
                    hits = sum(self.direct_by_class.get(cls, []))
                    denom = (len(self.direct_by_class.get(cls, []))
                             + self.direct_missed_by_class.get(cls, 0))
                    if denom:
                        recalls_full[cls] = hits / denom
                        out[f"reward/direct/recall_by_class/{cls}"] = recalls_full[cls]
                if recalls_full:
                    out["reward/direct/balanced_recall"] = (
                        sum(recalls_full.values()) / len(recalls_full)
                    )
                if self.direct_match_sims:
                    # >= tau by construction — read WITH miss_frac, never
                    # alone (drift raises miss_frac, not lowers match_sim).
                    out["diag/match_sim"] = (
                        sum(self.direct_match_sims) / len(self.direct_match_sims)
                    )

        # vignette/*
        if self.vig_antithesis:
            out["vignette/antithesis_frac"] = sum(self.vig_antithesis) / len(
                self.vig_antithesis
            )
            out["vignette/hedge_frac"] = sum(self.vig_hedge) / len(self.vig_hedge)
            # The term R3 re-anchored, watchable LIVE (m1's "pinned at 0.60"
            # had to be mined from traces post-hoc); parsed_frac separates
            # genuine hedging from unparsed batteries — the battery prompt is
            # a task the SFT policy never saw, so R1-class parse failures
            # would otherwise masquerade as hedge_frac (audit 2026-07-28).
            out["vignette/battery_mean"] = sum(self.vig_battery) / len(self.vig_battery)
            out["vignette/cite_mean"] = sum(self.vig_cite) / len(self.vig_cite)
            out["vignette/parsed_frac"] = sum(self.vig_parsed) / len(self.vig_parsed)

        # reward/<aux>/*
        for kind in _AUX_NAMES:
            if self.aux_groups[kind]:
                out[f"reward/{kind}/judge_failed_group_frac"] = (
                    self.aux_failed[kind] / self.aux_groups[kind]
                )

        # diag/*
        if self.direction:
            out["diag/direction_consistency"] = sum(self.direction) / len(
                self.direction
            )

        return {k: round(float(v), 6) for k, v in out.items()}


def _pstdev(values: Sequence[float]) -> float:
    """Population standard deviation (advantage-spread proxy)."""
    vals = [float(v) for v in values]
    m = sum(vals) / len(vals)
    return math.sqrt(sum((v - m) ** 2 for v in vals) / len(vals))


def _push_metrics(metrics: Mapping[str, float]) -> None:
    """Stream metrics to W&B (``commit=False``) if a run is live; else no-op."""
    if not metrics:
        return
    try:
        import wandb

        if wandb.run is not None:
            wandb.log(dict(metrics), commit=False)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Production factory (config → reward; reuses the keeper's judge client)
# ---------------------------------------------------------------------------
def make_modular_reward_from_cfg(
    cfg: Any,
    grpo_cfg: Any,
    norm_universes: dict[str, list] | None = None,
    *,
    answerer: AnswererClient | None = None,
    direct_gold_fn: "DirectGoldFn | None" = None,
    ground_scorer: AuxScorer | None = None,
    contrast_scorer: AuxScorer | None = None,
    trace_log_path: str = "",
    trace_every_n_calls: int = 1,
) -> ModularReward:
    """Build a :class:`ModularReward` from the m-series config keys.

    Consumes ``training.grpo.{reward_auxiliaries, reward_core, abstain,
    answerer, probes}`` (m_series.yaml). The frozen answerer is built via
    :func:`answerer_client.make_answerer_from_cfg` unless one is injected. The
    ground/contrast auxiliaries, when active, are wired to the keeper's
    ``OnlineRGround`` listwise judge through thin adapters
    (:func:`_make_aux_scorers`) — importing that client, never editing it. Tests
    inject mock scorers and bypass this factory.
    """
    auxiliaries = list(grpo_cfg.get("reward_auxiliaries", []) or [])
    reward_core = bool(grpo_cfg.get("reward_core", True))
    abstain = dict(grpo_cfg.get("abstain", {}) or {})
    core_mode = str(grpo_cfg.get("core_mode", "direct") or "direct")

    # The frozen answerer belongs to the RETIRED outcome core only. Building
    # it in direct mode (as the m1 wave did) wires a dead client to a server
    # role the run never uses — misleading in configs and metadata.
    if answerer is None and reward_core and core_mode == "outcome":
        from .answerer_client import make_answerer_from_cfg

        answerer = make_answerer_from_cfg(cfg)

    active_aux = [a for a in _AUX_NAMES if a in set(auxiliaries)]
    if active_aux and (ground_scorer is None and contrast_scorer is None):
        ground_scorer, contrast_scorer = _make_aux_scorers(
            cfg, grpo_cfg, norm_universes or {}, active_aux
        )
    if direct_gold_fn is None and reward_core and core_mode == "direct":
        from .aux_scorers import make_direct_gold_fn

        direct_gold_fn = make_direct_gold_fn(cfg, grpo_cfg, norm_universes or {})

    return ModularReward(
        auxiliaries=auxiliaries,
        reward_core=reward_core,
        core_mode=core_mode,
        direct_gold_fn=direct_gold_fn,
        direct_match_threshold=float(
            0.55 if grpo_cfg.get("direct_match_threshold") is None
            else grpo_cfg.get("direct_match_threshold")
        ),
        answerer=answerer,
        ground_scorer=ground_scorer,
        contrast_scorer=contrast_scorer,
        abstain=abstain,
        trace_log_path=trace_log_path,
        trace_every_n_calls=trace_every_n_calls,
    )


# ---------------------------------------------------------------------------
# Dataset-build hook: the m-series prompt set (extract + battery rows)
# ---------------------------------------------------------------------------
def _gold_class(gold_has_exchange: Any) -> str:
    """Prescreen stratum gold label ("yes" / "no" / "none")."""
    if gold_has_exchange is True:
        return "yes"
    if gold_has_exchange is False:
        return "no"
    return "none"


def _format_prompt(tokenizer, user_prompt: str, enable_thinking: bool) -> str:
    """Chat-template a single user turn (matches the keeper's build convention)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user_prompt}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=enable_thinking,
    )


def build_modular_dataset(
    cfg: Any,
    grpo_cfg: Any,
    chunks_df: Any,
    norm_universes: dict[str, list],
    reward_fn: "ModularReward",
    tokenizer: Any,
    ci_prompt_template: str,
    *,
    output_dir: str,
    seed: int,
    embed_fn: Callable[[list[str]], Any] | None = None,
    enable_thinking: bool = False,
) -> tuple[Any, dict[str, Any]]:
    """Build the m-series prompt set: T-EXTRACT (probe-bearing) + T-VIGNETTE rows.

    The dataset-build hook (migration.md checklist items 1–2, 6). Composes:

      * **T-EXTRACT** rows — one per fiction chunk with a non-empty probe pool
        (flow-bearing chunks with empty probe pools are excluded; gold-NO
        chunks are KEPT as probe-less A-ABSTAIN rows; the ``pools_path`` parquet is
        the source of truth). Each row carries its ``chunk_id``-sampled probes
        (:func:`attach_probes`) as reward metadata.
      * **T-VIGNETTE** rows — deontic batteries built per book from clustered
        contexts (:func:`batteries.cluster_contexts` + ``build_batteries`` using
        the injected small embedder).

    Rows are stratum-tagged ``(task_type, gold_class, force_class)`` and screened
    by :func:`prescreen_m1.stratified_prescreen`; the realized mix + the
    ``m1_cache_signature`` are written to ``training_metadata.json`` (principle 6).
    Sets ``reward_fn.prompt_metadata`` in place and returns ``(dataset, report)``.

    NOTE (item-7 execution surface): the stratified prescreen ranks *within
    strata by SFT-group reward variance*; that variance requires sampling the
    merged policy under this exact reward (the keeper does this in
    ``prompt_screening.prescreen_dataset``). Until that policy-sampling pass is
    wired for the modular reward, rows carry a constant variance and the screen
    degenerates to mix-preserving allocation of ``target_n`` — the composition is
    still correct and reported; only the *within-stratum ranking* is inert.
    """
    import json as _json
    import os as _os

    import pandas as pd
    from datasets import Dataset

    from . import batteries as _batteries
    from .prescreen_m1 import m1_cache_signature, stratified_prescreen

    probes_cfg = grpo_cfg.get("probes", {}) or {}
    pools_path = str(probes_cfg.get("pools_path", ""))
    k_max = int(probes_cfg.get("k_max", 4))
    if not pools_path or not _os.path.exists(pools_path):
        raise FileNotFoundError(
            f"[modular_reward] probe pools parquet not found: {pools_path!r} "
            "(training.grpo.probes.pools_path). Build it via the null-answerability "
            "calibration pass (data.md job list) before an m-series run."
        )
    pools_df = pd.read_parquet(pools_path)
    # Group probes by chunk into pools keyed by (gutenberg_id, chunk_id).
    pool_by_chunk: dict[tuple[str, str], list[dict]] = {}
    for rec in pools_df.to_dict("records"):
        key = (str(rec.get("gutenberg_id", "")), str(rec.get("chunk_id", "")))
        pool_by_chunk.setdefault(key, []).append(
            {
                "probe_id": rec.get("probe_id"),
                "gold": rec.get("gold"),
                "prompt_text": rec.get("prompt_text"),
                "norm_index": rec.get("norm_index"),
            }
        )

    rows: list[dict[str, Any]] = []
    meta: dict[str, dict[str, Any]] = {}
    n_empty_pool = 0
    n_src_goldno = 0
    n_src_probeable = 0  # gold-YES/unknown chunks (should carry probes)
    n_probe_rows = 0     # extract rows that actually got probes attached

    # --- T-EXTRACT rows -----------------------------------------------------
    for rec in chunks_df.to_dict("records"):
        chunk_text = rec.get("chunk_text", "")
        if not chunk_text or (isinstance(chunk_text, float) and pd.isna(chunk_text)):
            continue
        source_id = str(rec.get("source_id", ""))
        chunk_id = str(rec.get("chunk_id", rec.get("chunk_index", "")))
        gold = rec.get("has_information_exchange")
        if gold is None and rec.get("ci_flow_count") is not None:
            gold = int(rec["ci_flow_count"]) > 0
        gold = bool(gold) if gold is not None else None
        if gold is False:
            n_src_goldno += 1
        else:
            n_src_probeable += 1

        pool = pool_by_chunk.get((source_id, chunk_id), [])
        sampled = attach_probes(pool, chunk_id, k_max=k_max)
        if sampled:
            n_probe_rows += 1
        if not sampled and gold is not False:
            # Flow-bearing (or unknown-gold) chunk whose probe pool is empty:
            # nothing R-OUTCOME can verify — excluded per reward-outcome.md.
            # Gold-NO chunks are deliberately NOT excluded here: they never
            # bear probes by design and are exactly the rows A-ABSTAIN exists
            # for (scored entirely by the four-entry table, zero server
            # calls). Conflating the two removed every gold-NO row from the
            # 2026-07-24 wave-1 prompt set and collapsed the two-sided
            # abstention signal.
            n_empty_pool += 1
            continue

        user_prompt = ci_prompt_template.replace("{{chunk_text}}", str(chunk_text)).strip()
        formatted = _format_prompt(tokenizer, user_prompt, enable_thinking)
        rows.append(
            {
                "prompt": formatted,
                "task_type": "extract",
                "gold_has_exchange": gold,
                "gold_class": _gold_class(gold),
                "force_class": "na",
                "reward_std": 0.0,
            }
        )
        meta[formatted] = {
            "task_type": "extract",
            "gold_has_exchange": gold,
            "probes": sampled,
            "chunk_id": chunk_id,
            "source_id": source_id,
            # The R-GROUND / R-CONTRAST judge prompts render {{chunk_text}} as
            # the source passage; without it the judge grounds flows-vs-norms
            # blind to the passage (2026-07-24 review F1). Carry it here.
            "chunk_text": str(chunk_text),
        }

    # --- T-VIGNETTE rows ----------------------------------------------------
    battery_cfg = grpo_cfg.get("battery", {}) or {}
    n_batteries = 0
    battery_compositions: list[dict[str, Any]] = []
    if embed_fn is not None:
        for source_id, norms in norm_universes.items():
            book_norms = list(norms)
            contexts = [str(nrm.get("context") or "") for nrm in book_norms]
            cluster_ids = _batteries.cluster_contexts(contexts, embed_fn)
            built = _batteries.build_batteries(
                book_norms,
                str(source_id),
                cluster_ids,
                k=int(battery_cfg.get("k", 8)),
                min_k=int(battery_cfg.get("min_k", 4)),
                minority_floor=int(battery_cfg.get("minority_floor", 1)),
                minority_target=int(battery_cfg.get("minority_target", 2)),
            )
            for bat in built:
                formatted = _format_prompt(tokenizer, bat["prompt_text"], enable_thinking)
                n_no = bat["composition"]["n_gold_no"]
                n_yes = bat["composition"]["n_gold_yes"]
                rows.append(
                    {
                        "prompt": formatted,
                        "task_type": "vignette",
                        "gold_has_exchange": None,
                        "gold_class": "na",
                        "force_class": "no_major" if n_no >= n_yes else "yes_major",
                        "reward_std": 0.0,
                    }
                )
                meta[formatted] = {
                    "task_type": "vignette",
                    "gold_items": [
                        {"gold_force": it["gold_force"], "articulation": it["articulation"]}
                        for it in bat["items"]
                    ],
                    "battery_id": bat["battery_id"],
                    "source_id": str(source_id),
                }
                n_batteries += 1
                battery_compositions.append(
                    {"battery_id": bat["battery_id"], **bat["composition"]}
                )

    # --- Build-time invariant (2026-07-24 lesson): the two-sided A-ABSTAIN
    # signal requires gold-NO rows in the prompt set. If the source corpus has
    # gold-NO chunks but none survived the build, the exclusion logic has
    # regressed — fail loudly instead of training a one-sided reward.
    n_goldno_rows = sum(
        1 for r in rows if r["task_type"] == "extract" and r["gold_class"] == "no"
    )
    if n_src_goldno > 0 and n_goldno_rows == 0:
        raise ValueError(
            f"[modular_reward] source corpus has {n_src_goldno} gold-NO chunks "
            "but the built dataset has ZERO gold-no extract rows — the "
            "empty-pool exclusion is eating A-ABSTAIN's rows (see "
            "reward-abstain.md; gold-NO chunks bear no probes by design)."
        )

    # Symmetric guard (2026-07-24 review): the outcome core needs probe-bearing
    # gold-YES rows. If the source has probeable (gold-YES/unknown) chunks but
    # NONE got probes attached, the (source_id, chunk_id) ↔ (gutenberg_id,
    # chunk_id) join silently missed (namespace or dtype drift) and every
    # flow-bearing chunk fell into n_empty_pool — leaving an all-abstain
    # dataset with no error. Fail loudly rather than train a coreless run.
    if n_src_probeable > 0 and n_probe_rows == 0:
        raise ValueError(
            f"[modular_reward] source has {n_src_probeable} probeable "
            f"(gold-YES/unknown) chunks but ZERO got probes attached "
            f"(n_empty_pool={n_empty_pool}). The probe-pool join "
            "((source_id, chunk_id) vs the parquet's (gutenberg_id, chunk_id)) "
            "silently missed — check chunk_id dtype and the gutenberg_id→"
            "source_id rename. R-OUTCOME would train on nothing."
        )

    # --- Battery composition invariants (2026-07-28, R3) --------------------
    # Both-sides-present is enforced at build in build_batteries; assert it
    # here too so a regression in the builder can never silently ship
    # one-sided batteries again (44% of m1 batteries were one-sided and the
    # term carried ~1% of advantage mass on 30% of rows).
    task_mix = dict(grpo_cfg.get("task_mix", {}) or {})
    _one_sided = [
        c["battery_id"] for c in battery_compositions
        if c.get("n_gold_no", 0) == 0 or c.get("n_gold_yes", 0) == 0
    ]
    if _one_sided:
        raise ValueError(
            f"[modular_reward] {len(_one_sided)} one-sided batteries built "
            f"(e.g. {_one_sided[:3]}) — the both-sides invariant regressed "
            "(batteries.build_batteries minority-floor guard)."
        )
    if float(task_mix.get("vignette", 0.0) or 0.0) > 0.0 and n_batteries == 0:
        # NO embed_fn escape hatch (audit 2026-07-28): embed_fn is None
        # exactly when the clustering embedder failed to load — the most
        # likely cause of zero batteries. A cell that silently trains
        # all-extract while its config says vignette > 0 is an unlabelled
        # duplicate of the -vignette cell.
        raise ValueError(
            "[modular_reward] task_mix.vignette > 0 but ZERO batteries were "
            "built"
            + ("" if embed_fn is not None else
               " (context embedding model failed to load — check the "
               "sentence-transformers/HF-cache availability on this node)")
            + ". Lower the floor deliberately or set task_mix.vignette=0; "
            "training would silently run without the battery task otherwise."
        )

    # --- Stratified prescreen (mix-preserving; realized mix reported) -------
    target_n = int((grpo_cfg.get("prescreen", {}) or {}).get("target_n", len(rows)))
    rows_df = pd.DataFrame(rows)
    if len(rows_df) == 0:
        raise ValueError("[modular_reward] modular dataset built zero rows")
    selected_df, report = stratified_prescreen(
        rows_df, target_n=target_n, seed=seed, task_mix=task_mix
    )
    selected_prompts = set(selected_df["prompt"].tolist())
    reward_fn.prompt_metadata = {k: v for k, v in meta.items() if k in selected_prompts}

    # --- Chunk-gold index for the chunk-denominator direct core (R2) --------
    # Built AFTER prescreen so only surviving chunks are embedded. Gold-NO
    # rows are routed by A-ABSTAIN and never reach the scored path, so only
    # gold-YES/unknown chunks enter the index. Gated on a wired gold_fn so
    # unit builds (mock rewards, no embedding server) skip it; production
    # failures here must raise — a half-built index would silently score a
    # fraction of chunks on the wrong denominator.
    if (
        reward_fn.reward_core
        and getattr(reward_fn, "core_mode", "") == "direct"
        and reward_fn._direct_gold_fn is not None
        and bool(grpo_cfg.get("direct_chunk_denominator", True))
    ):
        from .aux_scorers import make_direct_chunk_gold

        chunk_keys = {
            (str(v["source_id"]), str(v["chunk_id"]))
            for v in reward_fn.prompt_metadata.values()
            if v.get("task_type") == "extract"
            and v.get("gold_has_exchange") is not False
        }
        if chunk_keys:
            _gk = grpo_cfg.get("direct_gold_k", 1)
            _gk = 1 if _gk is None else int(_gk)
            chunk_gold = make_direct_chunk_gold(
                cfg, grpo_cfg, norm_universes or {}, chunk_keys, gold_k=_gk,
            )
            # Fail-loud coverage guard (audit 2026-07-28, R2-M6): a dtype
            # drift in a rebuilt extraction parquet would empty the join and
            # silently revert every row to the m1 per-flow path — the exact
            # defect wave 2 exists to fix. Mirrors the probe-pool join guard.
            if len(chunk_gold.index) < 0.5 * len(chunk_keys):
                raise ValueError(
                    f"[modular_reward] chunk-gold index covers only "
                    f"{len(chunk_gold.index)}/{len(chunk_keys)} chunks — the "
                    "(source_id, chunk_id) join against extraction_path "
                    "missed (dtype drift?). Refusing to train mostly on the "
                    "per-flow fallback."
                )
            reward_fn.set_direct_chunk_gold(chunk_gold)
            print(f"[modular_reward] chunk-denominator R-DIRECT active "
                  f"(tau={reward_fn.direct_match_threshold}, gold_k={_gk})")

    dataset = Dataset.from_pandas(
        selected_df[["prompt", "task_type", "gold_has_exchange"]].reset_index(drop=True)
    )

    signature = m1_cache_signature(
        module_list=list(reward_fn.auxiliaries)
        + (["outcome"] if reward_fn.reward_core else []),
        task_mix=task_mix,
        seed=int(seed),
        data_fingerprint=_os.path.basename(pools_path),
    )
    metadata = {
        "reward_composition": "modular",
        "reward_auxiliaries": list(reward_fn.auxiliaries),
        "reward_core": reward_fn.reward_core,
        "task_mix": task_mix,
        "m1_cache_signature": signature,
        "n_extract_excluded_empty_pool": n_empty_pool,
        "n_src_goldno_chunks": n_src_goldno,
        "n_goldno_rows_prescreen_pool": n_goldno_rows,
        "n_src_probeable_chunks": n_src_probeable,
        "n_probe_bearing_rows": n_probe_rows,
        "n_batteries_built": n_batteries,
        # Per-battery polarity counts (task-vignettes.md principle-6 accounting).
        "battery_compositions": battery_compositions,
        "prescreen_report": report,
    }
    try:
        _os.makedirs(output_dir, exist_ok=True)
        with open(_os.path.join(output_dir, "training_metadata.json"), "w") as f:
            _json.dump(metadata, f, indent=2, default=str)
    except Exception as exc:  # pragma: no cover - io best-effort
        print(f"[modular_reward] WARNING could not write training_metadata.json: {exc}")

    print(
        f"[modular_reward] Built modular dataset: {len(dataset)} rows "
        f"(excluded {n_empty_pool} flow-bearing empty-pool chunks; kept "
        f"{n_goldno_rows}/{n_src_goldno} gold-NO chunks as probe-less "
        f"A-ABSTAIN rows; {n_batteries} batteries); "
        f"realized task mix {report.get('realized_task_mix')}"
    )

    # --- Realized-vs-configured mix invariant (audit 2026-07-28) ------------
    # The prescreen is a mix-preserving allocator that CANNOT fail: a stratum
    # shortfall (e.g. too few batteries) silently backfills the other strata.
    # That is how a pre-registered vignette share of 0.3 would have trained at
    # 0.143 with nothing but a stdout line. The cell keys ARE the protocol —
    # a realized mix off by more than task_mix_tolerance aborts the build.
    _tol = grpo_cfg.get("task_mix_tolerance", 0.03)
    _tol = 0.03 if _tol is None else float(_tol)
    _realized = dict(report.get("realized_task_mix") or {})
    for _task, _want in task_mix.items():
        _got = float(_realized.get(_task, 0.0))
        if abs(_got - float(_want)) > _tol:
            raise ValueError(
                f"[modular_reward] realized task mix for {_task!r} is "
                f"{_got:.3f} vs configured {float(_want):.3f} "
                f"(tolerance {_tol}). The stratum pool cannot fill the "
                "configured share — fix task_mix (or the pool) explicitly "
                "rather than training a silently different protocol."
            )
    return dataset, metadata


def _make_aux_scorers(
    cfg: Any,
    grpo_cfg: Any,
    norm_universes: dict[str, list],
    active_aux: list[str],
) -> tuple[AuxScorer | None, AuxScorer | None]:
    """Wire the production ground/contrast adapters (reward-ground.md /
    reward-contrast.md), reusing the keeper's frozen judge + retrieval plumbing.

    Delegates to :func:`aux_scorers.make_aux_scorers`, which builds the shared
    gemma-4-31b judge (same server as the answerer) plus the keeper's
    ``EmbeddingClient`` / ``NormRetriever`` and returns the two injected
    callables. Each is ``None`` when its auxiliary is inactive. Tests bypass this
    entirely by passing ``ground_scorer`` / ``contrast_scorer`` to
    :func:`make_modular_reward_from_cfg`.
    """
    from .aux_scorers import make_aux_scorers

    return make_aux_scorers(cfg, grpo_cfg, norm_universes, active_aux)
