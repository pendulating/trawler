"""Battery builder for the m-series `T-VIGNETTE` task (migration.md item 2).

A battery is a K-item test drawn from one book and one *context cluster*, mixed
in gold polarity (wiki/grpo_redesign/task-vignettes.md, "Battery construction"):

  1. **Group** decisive-force, ``governs_info_flow`` norms by (book, context
     cluster). Contexts are free-text; :func:`cluster_contexts` clusters them
     with a small sentence embedder (injected) + a similarity threshold so
     "marriage negotiations" and "arranging a marriage" pool.
  2. **Compose** batteries of up to ``k`` items from one cluster: both
     polarities whenever the cluster has them, ``>= minority_floor`` minority
     items (target ``minority_target``). Clusters with ``< min_k`` eligible
     norms are skipped. Composition is deterministic (RNG seeded by
     ``(gutenberg_id, cluster_id)``).
  3. **Scenario per item** via the shared probe templating (:mod:`probes`):
     same anti-leak contract (articulation withheld, force words never in the
     scenario text). The battery *instruction* asks for a 5-way deontic force,
     and the five option words MAY appear there — never in a scenario. A norm
     whose scenario would leak (:func:`probes.probe_leaks`) is skipped + counted.

Additive m-series code (parallel-stack rule): imports the keeper surfaces
(:mod:`probes` templating/leak helpers, :data:`probes.DECISIVE_FORCES`,
``FORCE_TO_GOLD``) but edits none of them. A later ``ModularReward`` agent scores
these batteries with :mod:`deontic_distance`.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Sequence

import numpy as np

from . import probes
from .deontic import FORCE_TO_GOLD, canonical_force, flow_appropriateness

# The output-schema instruction. The five force option words live HERE (allowed
# in the instruction, never in a scenario) — the anti-leak contract binds the
# per-item scenario text, checked by probes.probe_leaks.
_FORCE_OPTIONS = "obligatory, recommended, permitted, discouraged, prohibited"

_INSTRUCTION_HEAD = (
    "You are assessing a set of information-sharing scenarios that all arise in "
    "a single narrative context. For each scenario, decide the deontic force "
    "that governs the act it describes, choosing exactly one of: "
    f"{_FORCE_OPTIONS}. Give one or two sentences of reasoning and state the "
    "governing norm you believe applies."
)

_INSTRUCTION_TAIL = (
    'Respond with JSON only, in the form '
    '{"items": [{"id": 1, "force": "...", "reasoning": "...", '
    '"governing_norm": "..."}, ...]}, one entry per scenario.'
)


def _scenario_text(norm: dict) -> str:
    """The per-item scenario a battery asks the policy to assign a force to.

    Self-contained since 2026-07-25 (previously derived from
    ``probes.template_probe`` by stripping its yes/no suffix). The shared
    template preferred ``condition_of_application`` over ``norm_act``
    (``if condition: "is in a situation where {condition}"``), and on this
    corpus 99.1% of eligible norms carry both — so **99.1% of battery scenarios
    dropped the act entirely** and asked the policy to assign a five-way deontic
    force to a scenario containing no action:

        "In a setting involving professional conduct, an author of non-fiction
         is in a situation where when presenting accounts of events as true."

    Now the act is ALWAYS rendered and the condition is retained as a trailing
    qualifier, comma-joined verbatim with no connector. The bare comma composes
    correctly for clause conditions ("when …", "unless …") and prepositional
    ones ("during formal court proceedings") alike; a connector heuristic was
    tried and produced "in circumstances where *during* formal court
    proceedings".

    The condition is KEPT here (unlike the probe path, where it is dropped):
    a vignette scenario is the *object of judgment* rather than something that
    must correspond to an extraction, so there is no reason to strip content,
    and the condition is often what makes the force determinate.

    ANTI-LEAK: ``norm_articulation`` is never rendered and the template adds no
    force word of its own; field-borne leaks are caught downstream by
    :func:`probes.probe_leaks`, which ``build_batteries`` applies per scenario.
    Known residual risk (see reward-outcome-v2-proposal.md): acts extracted from
    directional norms are sometimes phrased directionally ("ensure …",
    "avoid …"), which no force-word filter catches.
    """
    context = str(norm.get("context") or "").strip()
    subject = str(norm.get("norm_subject") or "a person").strip()
    act = str(norm.get("norm_act") or "share this information").strip()
    condition = str(norm.get("condition_of_application") or "").strip()

    parts: list[str] = []
    if context:
        parts.append(f"In a setting involving {context},")
    parts.append(subject)
    parts.append(f"is considering whether to {act.rstrip('.')}")
    scenario = " ".join(parts).rstrip(".")

    if condition:
        cond = condition.rstrip(".")
        cond = cond[0].lower() + cond[1:]
        scenario = f"{scenario}, {cond}"
    return scenario + "."


def _norm_force(norm: dict) -> str:
    """The norm's normative force, lowercased (cleaned or raz_-prefixed)."""
    val = norm.get("normative_force") or norm.get("raz_normative_force")
    return str(val).strip().lower() if val else ""


def _context(norm: dict) -> str:
    """The norm's context string (empty if missing)."""
    val = norm.get("context")
    return str(val).strip() if val else ""


def _is_eligible(norm: dict) -> bool:
    """Battery eligibility: governs_info_flow ∧ gradient force ∧ non-empty context.

    Uses the FULL five-point gradient — ``permitted`` included (decision
    2026-07-25). Excluding it made one of the five answers the policy must
    choose from *never correct*, so the optimal policy learned "never say
    permitted" — a property of our battery construction, not of the books'
    norms. Worse, ``permitted`` is axis 0, the exact centre of the scale, so
    the exclusion removed every case where the centre is the right answer while
    leaving it fully available as a wrong one.

    Safe from a hedging exploit: a blanket-"permitted" policy scores 0.184 on
    the current pool, and permitted golds are ~2.8% of available items, so this
    cannot open a hedge loophole. What the policy now learns about permitted is
    a *frequency from the corpus* rather than an absolute we imposed.
    """
    if norm.get("governs_info_flow") is not True:
        return False
    if canonical_force(_norm_force(norm)) is None:
        return False
    if not _context(norm):
        return False
    return True


def _polarity(norm: dict) -> str | None:
    """"yes"/"no" side of the gradient for battery composition balancing.

    Derived from :func:`deontic.flow_appropriateness` so ``permitted`` counts
    as the appropriate ("yes") side, consistent with the standing gradient.
    Act polarity is deliberately NOT applied here: the battery asks for the
    NORM's force, not the flow's appropriateness, so a "refrain from…" act is
    still an obligation to refrain and belongs on the side its force names.
    """
    app = flow_appropriateness(_norm_force(norm), None)
    if app is None:
        return None
    return "yes" if app == "appropriate" else "no"


def _cosine_matrix(embeddings: np.ndarray) -> np.ndarray:
    """Row-wise cosine-similarity matrix (zero-norm rows → zero similarity)."""
    emb = np.asarray(embeddings, dtype=float)
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = emb / norms
    return unit @ unit.T


def cluster_contexts(
    contexts: list[str],
    embed_fn: Callable[[list[str]], np.ndarray],
    threshold: float = 0.8,
) -> list[int]:
    """Greedy single-linkage agglomerative clustering of context strings.

    ``embed_fn`` maps the context list to an ``(n, d)`` embedding array (injected
    so tests can stub it; production uses an all-MiniLM-class sentence embedder
    per task-vignettes.md). Processing contexts in order, each context joins the
    cluster of its most-similar already-assigned context when that cosine
    similarity ``>= threshold``, else opens a new cluster. Deterministic given
    the inputs. Returns a cluster-id list aligned with ``contexts`` (ids assigned
    0-indexed in order of first appearance).
    """
    n = len(contexts)
    if n == 0:
        return []
    sim = _cosine_matrix(embed_fn(contexts))

    assignments: list[int] = []
    next_id = 0
    for i in range(n):
        best_cluster = -1
        best_sim = float("-inf")
        for j in range(i):
            s = float(sim[i, j])
            if s > best_sim:
                best_sim = s
                best_cluster = assignments[j]
        if best_cluster >= 0 and best_sim >= threshold:
            assignments.append(best_cluster)
        else:
            assignments.append(next_id)
            next_id += 1
    return assignments


def _seeded_rng(gutenberg_id: str, cluster_id: int) -> random.Random:
    """Deterministic RNG seeded by (gutenberg_id, cluster_id)."""
    payload = f"{gutenberg_id}|{cluster_id}".encode("utf-8")
    seed = int(hashlib.sha1(payload).hexdigest(), 16) % (2**32)
    return random.Random(seed)


def _select_battery_indices(
    yes_idx: list[int],
    no_idx: list[int],
    rng: random.Random,
    k: int,
    minority_floor: int,
    minority_target: int,
) -> list[int]:
    """Pick up to ``k`` norm indices with both polarities + a minority floor.

    ``yes_idx`` / ``no_idx`` are the still-available norm indices of each gold
    polarity. The smaller pool is the *minority* (ties → gold-no, the
    Forbid-recall carrier). Reserves ``minority_target`` minority items (bounded
    by availability, never below ``minority_floor`` when the pool has them),
    fills the rest from the majority, and back-fills any leftover slots with
    extra minority. Selection is via the seeded ``rng`` (deterministic).
    Returns the chosen indices sorted ascending (stable battery ordering).
    """
    # Minority = smaller pool; tie or single-polarity handled below.
    if len(no_idx) <= len(yes_idx):
        minority, majority = no_idx, yes_idx
    else:
        minority, majority = yes_idx, no_idx

    battery_size = min(k, len(yes_idx) + len(no_idx))

    if minority and majority:
        n_minority = min(minority_target, len(minority))
        n_minority = max(n_minority, min(minority_floor, len(minority)))
        n_minority = min(n_minority, battery_size)
        n_majority = min(battery_size - n_minority, len(majority))
        # Back-fill leftover slots with extra minority when available.
        leftover = battery_size - n_minority - n_majority
        n_minority += min(leftover, len(minority) - n_minority)
    else:
        # Single-polarity cluster (or nothing to mix): take from whichever pool.
        pool = majority if majority else minority
        n_minority = 0
        n_majority = min(battery_size, len(pool))
        minority, majority = [], pool

    chosen: list[int] = []
    if n_minority:
        chosen += rng.sample(minority, n_minority)
    if n_majority:
        chosen += rng.sample(majority, n_majority)
    return sorted(chosen)


def build_batteries(
    book_norms: list[dict],
    gutenberg_id: str,
    cluster_ids: list[int],
    *,
    k: int = 8,
    min_k: int = 4,
    minority_floor: int = 1,
    minority_target: int = 2,
) -> list[dict]:
    """Build deontic batteries for one book from its clustered norms.

    ``cluster_ids`` is aligned with ``book_norms`` (``cluster_ids[i]`` is the
    context cluster of ``book_norms[i]``, e.g. from :func:`cluster_contexts`).
    Eligible norms (``governs_info_flow`` ∧ decisive force ∧ non-empty context)
    are grouped by cluster; a norm whose templated scenario would leak
    (:func:`probes.probe_leaks`) is skipped and counted. Clusters with fewer than
    ``min_k`` non-leaking eligible norms are skipped. A cluster yields as many
    non-overlapping batteries of up to ``k`` items as its pool supports (down to
    the ``min_k`` remainder), each mixing both polarities when present with a
    ``minority_floor`` (target ``minority_target``) minority share.

    Each battery dict::

        {"battery_id", "gutenberg_id", "cluster_id",
         "items": [{"norm_index", "gold_force", "scenario_text", "articulation"}],
         "prompt_text",
         "composition": {"n", "n_gold_no", "n_gold_yes"},
         "n_leak_skipped": int}

    where ``n_leak_skipped`` is the cluster's total leak drops, attributed to its
    first battery (0 on subsequent batteries from the same cluster).
    """
    # Group eligible norm indices by cluster, in book order.
    clusters: dict[int, list[int]] = {}
    leak_by_cluster: dict[int, int] = {}
    for idx, norm in enumerate(book_norms):
        if not _is_eligible(norm):
            continue
        cid = cluster_ids[idx]
        if probes.probe_leaks(_scenario_text(norm), norm):
            leak_by_cluster[cid] = leak_by_cluster.get(cid, 0) + 1
            continue
        clusters.setdefault(cid, []).append(idx)

    batteries: list[dict] = []
    for cid in sorted(clusters):
        clean = clusters[cid]
        if len(clean) < min_k:
            continue
        rng = _seeded_rng(str(gutenberg_id), cid)

        # Composition polarity follows the gradient (permitted -> appropriate
        # -> the "yes" side), while the item's GOLD stays the 5-way force. The
        # two are different questions: polarity balances the battery, the force
        # is what the policy must answer.
        yes_idx = [i for i in clean if _polarity(book_norms[i]) == "yes"]
        no_idx = [i for i in clean if _polarity(book_norms[i]) == "no"]

        seq = 0
        while len(yes_idx) + len(no_idx) >= min_k:
            chosen = _select_battery_indices(
                yes_idx, no_idx, rng, k, minority_floor, minority_target
            )
            if not chosen:
                break

            chosen_set = set(chosen)
            yes_idx = [i for i in yes_idx if i not in chosen_set]
            no_idx = [i for i in no_idx if i not in chosen_set]

            items: list[dict] = []
            n_gold_no = 0
            n_gold_yes = 0
            for norm_index in chosen:
                norm = book_norms[norm_index]
                force = _norm_force(norm)
                gold = FORCE_TO_GOLD.get(force)
                if gold == "no":
                    n_gold_no += 1
                else:
                    n_gold_yes += 1
                items.append(
                    {
                        "norm_index": norm_index,
                        "gold_force": force,
                        "scenario_text": _scenario_text(norm),
                        "articulation": probes._articulation(norm),
                    }
                )

            prompt_text = _render_prompt(items)
            battery_id = _battery_id(str(gutenberg_id), cid, seq, chosen)
            batteries.append(
                {
                    "battery_id": battery_id,
                    "gutenberg_id": str(gutenberg_id),
                    "cluster_id": cid,
                    "items": items,
                    "prompt_text": prompt_text,
                    "composition": {
                        "n": len(items),
                        "n_gold_no": n_gold_no,
                        "n_gold_yes": n_gold_yes,
                    },
                    "n_leak_skipped": leak_by_cluster.get(cid, 0) if seq == 0 else 0,
                }
            )
            seq += 1

    return batteries


def _render_prompt(items: Sequence[dict]) -> str:
    """Assemble the battery prompt: instruction + numbered scenarios + schema."""
    lines = [_INSTRUCTION_HEAD, ""]
    for i, item in enumerate(items, start=1):
        lines.append(f"Scenario {i}: {item['scenario_text']}")
    lines.append("")
    lines.append(_INSTRUCTION_TAIL)
    return "\n".join(lines)


def _battery_id(gutenberg_id: str, cluster_id: int, seq: int, indices: Sequence[int]) -> str:
    """Deterministic battery id: sha1(book | cluster | seq | member indices)."""
    payload = f"{gutenberg_id}|{cluster_id}|{seq}|{','.join(map(str, indices))}"
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    return f"{gutenberg_id}-c{cluster_id}-{digest}"
