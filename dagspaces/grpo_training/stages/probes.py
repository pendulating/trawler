"""Outcome-probe builder for the m-series GRPO redesign (`R-OUTCOME` core).

Builds, at dataset-build time, the per-prompt probe set the frozen answerer is
tested against (wiki/grpo_redesign/reward-outcome.md, "Probe generation"). A
probe is a single yes/no question templated from a governing norm's *fields*
(never its articulation), whose gold answer comes from the norm's deontic force
(``FORCE_TO_GOLD`` in :mod:`deontic` — the single source of truth shared with
the vignette task). Probes are a property of the *prompt*, fixed before any
generation, so every completion in a group faces the identical test.

This is additive m-series code (the parallel-stack rule,
wiki/grpo_redesign/migration.md): it imports from the keeper surfaces but edits
none of them. In particular ``FORCE_TO_GOLD`` is *imported* from
:mod:`dagspaces.grpo_training.stages.deontic` (declared single source of truth),
and :func:`flow_to_query` is a deliberate *reimplementation* (not an import) of
``online_rground._flow_to_query`` to keep this module import-light; a parity
unit test pins the two together.
"""

from __future__ import annotations

import hashlib
import random
import re
from collections.abc import Mapping, Sequence
from typing import Callable

# Single source of truth for force → gold (imported, never re-declared).
from .deontic import FORCE_TO_GOLD

# The decisive Raz forces that carry a directional gold answer. Exactly the
# keys of FORCE_TO_GOLD ("permitted"/unknown are skipped — no direction).
DECISIVE_FORCES: set[str] = {
    "obligatory",
    "recommended",
    "prohibited",
    "discouraged",
}

# Fixed yes/no question appended to every probe scenario. Phrased so the gold
# maps straight through FORCE_TO_GOLD: a prohibiting/discouraging norm ⇒ "no",
# an obligating/recommending norm ⇒ "yes".
_QUESTION_SUFFIX = "Should this information be shared? Answer yes or no."

# Word stems for the Raz force vocabulary (+ adjective/adverb/noun variants).
# A prompt token that *starts with* any of these is a force-word leak:
#   oblig      → obligatory, obligation, obliged, obligate…
#   recommend  → recommended, recommendation, recommends…
#   prohibit   → prohibited, prohibition, prohibitive…
#   discourag  → discouraged, discouragement, discourages…
#   permit     → permitted, permitting, permits…
#   permiss    → permission, permissible, permissive…
_FORCE_STEMS: tuple[str, ...] = (
    "oblig",
    "recommend",
    "prohibit",
    "discourag",
    "permit",
    "permiss",
)

# Minimum contiguous run of articulation tokens that counts as a leak
# (PrivacyLens canary pattern: a verbatim span of the withheld source text).
_LEAK_NGRAM = 6

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokens(text: str) -> list[str]:
    """Lowercased alphanumeric word tokens."""
    return _WORD_RE.findall(str(text).lower())


def _s(norm: dict, key: str) -> str:
    """A norm string field, stripped and lowercased (empty if missing/None)."""
    val = norm.get(key)
    return str(val).strip().lower() if val else ""


def _force_str(norm: dict) -> str:
    """The norm's normative force, lowercased (cleaned or raz_-prefixed)."""
    val = norm.get("normative_force") or norm.get("raz_normative_force")
    return str(val).strip().lower() if val else ""


def _articulation(norm: dict) -> str:
    """The withheld norm articulation, from whichever field carries it."""
    for key in (
        "norm_articulation",
        "raz_norm_articulation",
        "articulation",
        "canonical_norm_articulation",
    ):
        val = norm.get(key)
        if val:
            return str(val)
    return ""


def flow_to_query(flow: dict) -> str:
    """Build a retrieval query from a single flow's CI-tuple fields.

    Verbatim mirror of
    :func:`dagspaces.grpo_training.stages.online_rground._flow_to_query`
    (reimplemented, not imported, to keep the m-series import-light — a parity
    unit test asserts equality on fixtures). Joins the non-empty CI fields in a
    fixed order, then appends the ``norms_invoked`` list; falls back to
    ``"information flow"`` when nothing is present.
    """
    parts: list[str] = []
    for key in (
        "sender",
        "recipient",
        "information_type",
        "context",
        "transmission_principle",
        "subject",
    ):
        val = flow.get(key, "")
        if val:
            parts.append(str(val))
    invoked = flow.get("norms_invoked", [])
    if isinstance(invoked, list):
        parts.extend(str(n) for n in invoked)
    return " ".join(parts) if parts else "information flow"


def norm_dedupe_key(norm: dict) -> tuple:
    """Identity for union-dedupe across a chunk's reference flows.

    Two norms that share subject, act, condition, context and force are the
    same governing rule for probe purposes. Each component is stripped and
    lowercased so surface variation does not defeat the dedupe.
    """
    return (
        _s(norm, "norm_subject"),
        _s(norm, "norm_act"),
        _s(norm, "condition_of_application"),
        _s(norm, "context"),
        _force_str(norm),
    )


def probe_id(gutenberg_id: str, norm: dict) -> str:
    """Deterministic short id for a probe: sha1(gutenberg_id + dedupe key)."""
    payload = f"{gutenberg_id}|{norm_dedupe_key(norm)}"
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:16]


def template_probe(norm: dict) -> str:
    """Template a probe scenario + fixed yes/no question from a norm's fields.

    Mirrors the shared vignette builder ``_generate_vignettes``
    (:mod:`dagspaces.grpo_training.stages.grpo_training`, line 28): the scenario
    is assembled from ``context``, ``norm_subject`` and
    ``condition_of_application`` / ``norm_act``, then a fixed yes/no question is
    appended.

    ANTI-LEAK (inherited from the vignette builder): the ``norm_articulation``
    is never included, and the template itself introduces no force word. If a
    *field* the template draws on happens to contain the articulation or a force
    word, that is caught downstream by :func:`probe_leaks` (build_probe_pool
    skips such norms) — this function does not scrub, it only refrains from
    adding leaks of its own.
    """
    subject = norm.get("norm_subject") or "a person"
    act = norm.get("norm_act") or "share this information"
    condition = norm.get("condition_of_application") or ""
    context = norm.get("context") or ""

    parts: list[str] = []
    if context:
        parts.append(f"In a setting involving {context},")
    parts.append(str(subject))
    if condition:
        parts.append(f"is in a situation where {condition}.")
    else:
        parts.append(f"is considering whether to {act}.")
    scenario = " ".join(parts)
    return f"{scenario} {_QUESTION_SUFFIX}"


def probe_leaks(prompt_text: str, norm: dict) -> bool:
    """True if the prompt leaks the answer.

    Two leak channels (PrivacyLens canary pattern):
      1. any force word (a token starting with a Raz force stem), or
      2. a contiguous run of >= ``_LEAK_NGRAM`` tokens from the withheld
         ``norm_articulation`` appearing verbatim in the prompt.
    """
    prompt_toks = _tokens(prompt_text)

    # Channel 1: force word.
    for tok in prompt_toks:
        if any(tok.startswith(stem) for stem in _FORCE_STEMS):
            return True

    # Channel 2: verbatim articulation span.
    art_toks = _tokens(_articulation(norm))
    if len(art_toks) >= _LEAK_NGRAM:
        prompt_windows = {
            tuple(prompt_toks[i : i + _LEAK_NGRAM])
            for i in range(len(prompt_toks) - _LEAK_NGRAM + 1)
        }
        for i in range(len(art_toks) - _LEAK_NGRAM + 1):
            if tuple(art_toks[i : i + _LEAK_NGRAM]) in prompt_windows:
                return True

    return False


def build_probe_pool(
    reference_flows: list[dict],
    book_norms: list[dict],
    retrieve_top_k: Callable[[str, int], Sequence[int]],
    k: int = 3,
) -> list[dict]:
    """Build a chunk's candidate probe pool from its reference flows.

    For each reference flow, retrieve the top-``k`` norms from the chunk's own
    book universe (``retrieve_top_k(flow_to_query(flow), k)`` → indices into
    ``book_norms``); union the retrieved norms over all flows, deduped by
    :func:`norm_dedupe_key`; keep only norms with ``governs_info_flow is True``
    and a force in :data:`DECISIVE_FORCES`; drop any whose templated probe leaks
    (:func:`probe_leaks`). The gold answer comes from ``FORCE_TO_GOLD``.

    Order is deterministic — by first retrieval appearance across the flows.

    Returns a list of candidate probe dicts, each::

        {"probe_id", "norm_index", "norm", "gold", "prompt_text"}

    where ``norm_index`` indexes into ``book_norms``. (The count of
    leak-skipped norms is available via :func:`build_probe_pool_with_stats`.)
    """
    pool, _ = build_probe_pool_with_stats(
        reference_flows, book_norms, retrieve_top_k, k=k
    )
    return pool


def build_probe_pool_with_stats(
    reference_flows: list[dict],
    book_norms: list[dict],
    retrieve_top_k: Callable[[str, int], Sequence[int]],
    k: int = 3,
) -> tuple[list[dict], dict]:
    """As :func:`build_probe_pool`, also returning build stats.

    Stats dict: ``{"n_leak_skipped": int}`` — the number of decisive,
    flow-governing norms excluded because their templated probe leaked (the
    drop count the reward-outcome spec asks to report).
    """
    # Union over flows, deduped, in first-appearance order.
    seen_keys: set[tuple] = set()
    ordered_indices: list[int] = []
    for flow in reference_flows:
        query = flow_to_query(flow)
        for idx in retrieve_top_k(query, k):
            if idx < 0 or idx >= len(book_norms):
                continue
            norm = book_norms[idx]
            key = norm_dedupe_key(norm)
            if key in seen_keys:
                continue
            seen_keys.add(key)
            ordered_indices.append(idx)

    pool: list[dict] = []
    n_leak_skipped = 0
    for idx in ordered_indices:
        norm = book_norms[idx]
        if norm.get("governs_info_flow") is not True:
            continue
        force = _force_str(norm)
        if force not in DECISIVE_FORCES:
            continue
        gold = FORCE_TO_GOLD.get(force)
        if gold is None:  # defensive: DECISIVE_FORCES already guarantees this
            continue
        prompt_text = template_probe(norm)
        if probe_leaks(prompt_text, norm):
            n_leak_skipped += 1
            continue
        pool.append(
            {
                "probe_id": probe_id(str(norm.get("gutenberg_id", "")), norm),
                "norm_index": idx,
                "norm": norm,
                "gold": gold,
                "prompt_text": prompt_text,
            }
        )

    return pool, {"n_leak_skipped": n_leak_skipped}


def sample_probes(
    pool: list[dict], chunk_id: str, k_max: int = 4
) -> list[dict]:
    """Force-stratified deterministic sample of K probes (D2).

    ``K = min(k_max, len(pool))``. The RNG is seeded from ``chunk_id`` so the
    same chunk always yields the same sample (and enters the prescreen cache
    signature). Stratification guarantee: if the pool contains a gold-**no**
    probe, the sample contains at least one; if the pool contains *both* gold
    classes (and K >= 2), the sample contains both. Remaining slots are filled
    uniformly at random. The gold-no reservation is first because it is the
    Forbid-recall carrier the whole redesign turns on.
    """
    if not pool:
        return []
    K = min(k_max, len(pool))
    seed = int(hashlib.sha1(chunk_id.encode("utf-8")).hexdigest(), 16) % (2**32)
    rng = random.Random(seed)

    yes = [p for p in pool if p["gold"] == "yes"]
    no = [p for p in pool if p["gold"] == "no"]

    chosen: list[dict] = []
    chosen_ids: set[str] = set()

    def _take_from(bucket: list[dict]) -> None:
        cands = [p for p in bucket if p["probe_id"] not in chosen_ids]
        if not cands or len(chosen) >= K:
            return
        p = rng.choice(cands)
        chosen.append(p)
        chosen_ids.add(p["probe_id"])

    # Reserve a gold-no first (Forbid recall), then a gold-yes if room.
    if no:
        _take_from(no)
    if yes:
        _take_from(yes)

    # Fill remaining slots uniformly.
    while len(chosen) < K:
        remaining = [p for p in pool if p["probe_id"] not in chosen_ids]
        if not remaining:
            break
        p = rng.choice(remaining)
        chosen.append(p)
        chosen_ids.add(p["probe_id"])

    # Deterministic output order: pool (first-appearance) order.
    order = {p["probe_id"]: i for i, p in enumerate(pool)}
    chosen.sort(key=lambda p: order[p["probe_id"]])
    return chosen


def apply_null_filter(
    pool: list[dict],
    null_correct_frac: Mapping[str, float],
    p_null: float = 0.8,
) -> list[dict]:
    """Drop probes answerable without the extraction (null-answerability filter).

    ``null_correct_frac`` maps ``probe_id`` → fraction of empty-extraction votes
    that matched gold. A probe with ``frac >= p_null`` carries no signal about
    the extraction (world knowledge answers it) and is dropped. Probes *missing*
    from the mapping are KEPT — uncalibrated is treated as extraction-dependent
    (the calibration pass is expected to cover the whole pool; a gap must not
    silently remove a probe).
    """
    kept: list[dict] = []
    for p in pool:
        frac = null_correct_frac.get(p["probe_id"])
        if frac is not None and frac >= p_null:
            continue
        kept.append(p)
    return kept
