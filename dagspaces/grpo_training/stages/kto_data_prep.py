"""k-series KTO dataset build (wiki/2026-07-31_kto_plan.md §4–§5, K0-locked).

Turns policy samples into `(prompt, completion, label)` rows for TRL's
``KTOTrainer``, through the audited m2 labeler (valid gate → chunk-gold
match → per-flow gold) and the supervision-depth ladder edits
(:mod:`kto_edits`). One build emits every arm's data:

  * ``label=False`` rows (undesirable): D1′ undesirables — completions that
    mislabel a matched violation flow, PLUS false alarms (K0 §12 amendment:
    16.1% of completions cry wolf; leaving them unlabeled while upweighting
    violation-catching invites the seesaw) — shared by all arms
    (``depth="shared"``).
  * ``label=True`` rows (desirable):
      - R-MINE naturals (``recipe="mine"``, shared);
      - R-VERDICT / R-CITATION / R-SCRUTINIZE edited counterfactuals
        (``recipe="edit"``, one row per depth per source pair);
      - R-ABSTAIN synthesized no-flow completions on gold-NO chunks
        (``recipe="abstain"``, shared; K0: the policy abstains on only 9%
        of gold-NO samples, so desirable abstains are synthesized in the
        SFT no-flow format — sampled hallucinated extractions supply the
        undesirable side).

Split discipline (§5): a stratified chunk-level 80/20 train/held-out split
by (book × mixed-gold × gold-YES/NO); held-out chunks contribute ZERO rows
— they exist only for the per-checkpoint probe. Split membership is written
into the metadata and asserted at build.

Composition invariants (launch-checks discipline): every edited desirable
round-trips through the production ``valid_gate``; every expected stratum
is non-empty; the held-out set has no rows; realized class counts land in
the metadata with the recommended TRL weights.

Additive k-series code (parallel-stack rule): imports m-series surfaces
(`valid_gate`, `match_flows`, `DirectChunkGold`, `sft_data_prep` formats),
edits none.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
from collections import Counter, defaultdict
from typing import Any, Callable

import numpy as np
import pandas as pd

from .kto_edits import (
    Correction,
    apply_citation_edit,
    apply_scrutinize_edit,
    apply_verdict_edit,
    rationale_is_valid,
    render_rationale,
    serialize_completion,
)
from .modular_reward import match_flows, valid_gate
from .sft_data_prep import _NO_EXCHANGE_REASONING

DEPTHS: dict[str, Callable] = {
    "verdict": apply_verdict_edit,
    "citation": apply_citation_edit,
    "scrutinize": apply_scrutinize_edit,
}


# ---------------------------------------------------------------------------
# Labeling (pure)
# ---------------------------------------------------------------------------
def d1_prime_label(per_flow: list[tuple[str, bool]]) -> str:
    """D1′ verdict over MATCHED flows: ``(gold, correct)`` pairs.

    D1 (plan §3) + the K0 §12 amendment: false alarms (mislabeling an
    appropriate flow as inappropriate, on all-appropriate evidence) are
    UNDESIRABLE, not unlabeled.
    """
    if not per_flow:
        return "excluded"
    viol = [c for g, c in per_flow if g == "inappropriate"]
    if viol and not all(viol):
        return "undesirable"
    n_ok = sum(c for _, c in per_flow)
    if viol and all(viol) and n_ok >= len(per_flow) / 2:
        return "desirable"
    if not viol:
        return "desirable" if n_ok == len(per_flow) else "undesirable"
    return "neither"


def label_completion(
    text: str,
    entry: dict | None,
    embed_flows_fn: Callable,
    tau: float,
    min_edit_sim: float,
) -> dict[str, Any]:
    """Gate → match → D1′ label + corrections for one completion.

    ``entry`` is the chunk-gold index entry (``golds``/``emb``/``norms``);
    ``embed_flows_fn`` embeds a flow list (injected — tests use fakes).
    Statuses: ``gate_fail`` / ``no_entry`` / ``embed_fail`` / ``scored``.
    """
    g = valid_gate(text)
    if not g.passed:
        return {"status": "gate_fail"}
    if entry is None or not g.flows:
        return {"status": "no_entry"}
    p_emb = np.asarray(embed_flows_fn(g.flows))
    if p_emb.ndim != 2 or not p_emb.any(axis=1).all():
        return {"status": "embed_fail"}
    matches = match_flows(entry["emb"] @ p_emb.T, tau)
    norms = entry.get("norms") or [{} for _ in entry["golds"]]
    per_flow: list[tuple[str, bool]] = []
    corrections: list[Correction] = []
    for t, p, sim in matches:
        gold = entry["golds"][t]
        label = str(g.flows[p].get("appropriateness") or "").strip().lower()
        correct = label == gold
        per_flow.append((gold, correct))
        if not correct and sim >= min_edit_sim:
            corrections.append(Correction(
                flow_index=p, gold=gold, norm=norms[t], match_sim=float(sim)))
    return {
        "status": "scored",
        "label": d1_prime_label(per_flow),
        "parsed": g.parsed,
        "corrections": corrections,
        "per_flow": per_flow,
        "n_flows": len(g.flows),
        "n_matched": len(matches),
    }


# ---------------------------------------------------------------------------
# Split (pure)
# ---------------------------------------------------------------------------
def stratified_split(
    chunk_info: dict[tuple[str, str], dict[str, Any]],
    heldout_frac: float,
    seed: int,
) -> dict[tuple[str, str], str]:
    """Chunk-level train/heldout split, stratified by
    (book × mixed-gold × gold-YES/NO), deterministic under ``seed``.

    Every stratum with >= 2 chunks contributes at least one held-out chunk
    (ceil), so no stratum is unmeasurable at probe time; singleton strata
    stay in train (holding out the only exemplar would starve training and
    measure nothing meaningful).
    """
    strata: dict[tuple, list] = defaultdict(list)
    for key, info in chunk_info.items():
        strata[(info["book"], bool(info.get("mixed")),
                bool(info.get("gold_yes")))].append(key)
    out: dict[tuple[str, str], str] = {}
    for stratum, keys in sorted(strata.items()):
        keys = sorted(keys)
        rng = random.Random(f"{seed}|{stratum}")
        rng.shuffle(keys)
        n_held = (0 if len(keys) < 2
                  else max(1, int(round(len(keys) * heldout_frac))))
        for i, k in enumerate(keys):
            out[k] = "heldout" if i < n_held else "train"
    return out


# ---------------------------------------------------------------------------
# R-ABSTAIN synthesis (pure)
# ---------------------------------------------------------------------------
def synth_abstain_completion() -> str:
    """A desirable no-flow declaration in the exact SFT no-flow format
    (same reasoning constant + shape ``sft_data_prep`` trains on)."""
    return json.dumps({
        "reasoning": _NO_EXCHANGE_REASONING,
        "has_information_exchange": False,
        "flows": [],
    }, ensure_ascii=False)


# ---------------------------------------------------------------------------
# The build
# ---------------------------------------------------------------------------
def build_kto_dataset(
    samples: pd.DataFrame,
    chunk_gold,
    chunk_info: dict[tuple[str, str], dict[str, Any]],
    prompts: dict[tuple[str, str], str],
    *,
    tau: float = 0.55,
    min_edit_sim: float = 0.55,
    heldout_frac: float = 0.2,
    max_pairs_per_chunk: int = 4,
    abstain_desirables_per_chunk: int = 2,
    target_weight_ratio: float = 1.15,
    seed: int = 42,
    rationale_fn: "Callable[[Correction, dict], str] | None" = None,
    rationale_batch_fn:
        "Callable[[list[tuple[Correction, dict]]], list[str | None]] | None" = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build the k-series dataset from policy ``samples``.

    ``samples`` columns: ``k0``/``k1`` (chunk key), ``sample``, ``text``.
    ``chunk_gold``: the ``DirectChunkGold`` index built with
    ``keep_norm_info=True`` (gold-YES chunks only). ``chunk_info`` covers
    gold-YES AND gold-NO chunks (``book``/``mixed``/``gold_yes``).
    ``prompts`` maps chunk key -> the chat-templated rollout prompt.
    ``rationale_fn(correction, parsed) -> str`` supplies K-SCRUTINIZE
    rationales one at a time; ``rationale_batch_fn(items) -> texts`` is the
    scale path (all pending corrections in one call — the production driver
    fans it out over a worker pool against the judge server; ~13k serial
    calls would take ~7h at full-corpus scale). Invalid or missing output
    falls back to the deterministic template, counted in the metadata.

    Returns ``(rows_df, metadata)``. Raises on any composition invariant
    violation (launch-checks discipline).
    """
    rng = random.Random(seed)
    split = stratified_split(chunk_info, heldout_frac, seed)

    rows: list[dict[str, Any]] = []
    stats = Counter()
    rationale_fallbacks = 0
    seen_texts: set[str] = set()
    pending_edits: list[tuple[tuple[str, str], dict, list[Correction]]] = []

    def emit(key, completion, label, recipe, depth, **extra):
        rows.append({
            "prompt": prompts[key],
            "completion": completion,
            "label": bool(label),
            "recipe": recipe,
            "depth": depth,
            "book": key[0],
            "chunk_key": f"{key[0]}|{key[1]}",
            "split": "train",
            **extra,
        })

    by_chunk = {tuple(k): g for k, g in samples.groupby(["k0", "k1"])}
    for key, group in sorted(by_chunk.items()):
        info = chunk_info.get(key)
        if info is None:
            stats["skipped_unknown_chunk"] += len(group)
            continue
        if split[key] == "heldout":
            stats["heldout_chunks_skipped"] += 1
            continue

        if not info.get("gold_yes"):
            # ---- R-ABSTAIN: gold-NO chunk ------------------------------
            n_und = 0
            for text in group.sort_values("sample")["text"]:
                g = valid_gate(text)
                if not g.passed:
                    stats["gate_fail"] += 1
                    continue
                if g.no_flow:
                    emit(key, text, True, "abstain", "shared")
                    stats["abstain_desirable_sampled"] += 1
                elif n_und < max_pairs_per_chunk:
                    emit(key, text, False, "abstain", "shared")
                    stats["abstain_undesirable"] += 1
                    n_und += 1
                else:
                    # No silent caps: beyond-cap drops are counted, not hidden.
                    stats["capped_abstain_undesirable"] += 1
            for _ in range(abstain_desirables_per_chunk):
                emit(key, synth_abstain_completion(), True,
                     "abstain", "shared", synthesized=True)
                stats["abstain_desirable_synth"] += 1
            continue

        # ---- gold-YES chunk -------------------------------------------
        entry = chunk_gold.get(*key)
        n_des = n_und = 0
        cands = []
        for text in group.sort_values("sample")["text"]:
            r = label_completion(text, entry, chunk_gold.embed_flows,
                                 tau, min_edit_sim)
            stats[r["status"]] += 1
            if r["status"] == "scored":
                cands.append((text, r))

        rng.shuffle(cands)
        for text, r in cands:
            if text in seen_texts:
                stats["deduped"] += 1
                continue
            if r["label"] == "desirable":
                if n_des < max_pairs_per_chunk:
                    seen_texts.add(text)
                    emit(key, text, True, "mine", "shared")
                    stats["mine_desirable"] += 1
                    n_des += 1
                else:
                    stats["capped_desirable"] += 1
            elif r["label"] == "undesirable":
                if n_und < max_pairs_per_chunk:
                    seen_texts.add(text)
                    emit(key, text, False, "und", "shared",
                         n_corrections=len(r["corrections"]))
                    stats["undesirable"] += 1
                    n_und += 1
                    if r["corrections"]:
                        pending_edits.append((key, r["parsed"], r["corrections"]))
                else:
                    stats["capped_undesirable"] += 1
            else:
                # D1' non-labels ("neither" / "excluded") — counted so the
                # metadata accounts for every scored completion.
                stats[f"d1_{r['label']}"] += 1

    # ---- ladder desirables (deferred so rationales can be batched) -------
    if rationale_batch_fn is None and rationale_fn is not None:
        def rationale_batch_fn(items):  # serial fallback path
            return [rationale_fn(c, parsed) for c, parsed in items]

    rationales_by_pending: dict[int, list[str] | None] = {}
    if rationale_batch_fn is not None and pending_edits:
        flat: list[tuple[Correction, dict]] = []
        spans = []
        for key, parsed, corrs in pending_edits:
            ordered = sorted(corrs, key=lambda c: c.flow_index)
            spans.append((len(flat), len(ordered)))
            flat.extend((c, parsed) for c in ordered)
        texts = rationale_batch_fn(flat)
        if len(texts) != len(flat):
            raise ValueError("[kto_data_prep] rationale_batch_fn returned "
                             f"{len(texts)} texts for {len(flat)} corrections")
        for i, (lo, n) in enumerate(spans):
            out = []
            for j in range(n):
                c = flat[lo + j][0]
                t = texts[lo + j]
                if not (t and rationale_is_valid(t, c)):
                    t = render_rationale(c.flow_index, c)
                    rationale_fallbacks += 1
                out.append(t)
            rationales_by_pending[i] = out

    for i, (key, parsed, corrs) in enumerate(pending_edits):
        rationales = rationales_by_pending.get(i)
        for depth, fn in DEPTHS.items():
            edited = serialize_completion(
                fn(parsed, corrs) if depth != "scrutinize"
                else fn(parsed, corrs, rationales=rationales))
            gate = valid_gate(edited)
            if not gate.passed:
                raise ValueError(
                    f"[kto_data_prep] edited completion failed the gate "
                    f"(depth={depth}, chunk={key}) — edit surgery regressed")
            emit(key, edited, True, "edit", depth,
                 n_edits=len(corrs),
                 min_match_sim=min(c.match_sim for c in corrs))
            stats[f"edit_{depth}"] += 1

    rows_df = pd.DataFrame(rows)

    # ---- composition invariants ------------------------------------------
    if not len(rows_df):
        raise ValueError("[kto_data_prep] built zero rows")
    held_rows = (rows_df["chunk_key"].map(
        lambda ck: split.get(tuple(ck.split("|", 1)))) == "heldout").sum()
    if held_rows:
        raise ValueError(f"[kto_data_prep] {held_rows} rows leaked from "
                         "held-out chunks")
    for required in ("mine_desirable", "undesirable", "edit_verdict",
                     "edit_citation", "edit_scrutinize",
                     "abstain_desirable_synth"):
        if not stats.get(required):
            raise ValueError(f"[kto_data_prep] required stream {required!r} "
                             "is EMPTY — composition regressed")

    # ---- per-arm class weights (TRL band, plan §6) -----------------------
    weights = {}
    for depth in DEPTHS:
        arm = rows_df[(rows_df["depth"] == "shared")
                      | (rows_df["depth"] == depth)]
        n_d = int(arm["label"].sum())
        n_u = int((~arm["label"]).sum())
        weights[depth] = {
            "n_desirable": n_d, "n_undesirable": n_u,
            "desirable_weight": round(target_weight_ratio * n_u / n_d, 3),
            "undesirable_weight": 1.0,
        }

    n_train = sum(1 for v in split.values() if v == "train")
    n_held = sum(1 for v in split.values() if v == "heldout")
    metadata = {
        "recipe_stats": dict(stats),
        "rationale_fallbacks": rationale_fallbacks,
        "split": {"train_chunks": n_train, "heldout_chunks": n_held,
                  "heldout_frac_realized": round(n_held / max(1, n_train + n_held), 3),
                  "seed": seed},
        "heldout_keys": sorted(f"{k[0]}|{k[1]}" for k, v in split.items()
                               if v == "heldout"),
        "arm_class_weights": weights,
        "knobs": {"tau": tau, "min_edit_sim": min_edit_sim,
                  "max_pairs_per_chunk": max_pairs_per_chunk,
                  "abstain_desirables_per_chunk": abstain_desirables_per_chunk,
                  "target_weight_ratio": target_weight_ratio},
        "fingerprint": hashlib.sha1(
            pd.util.hash_pandas_object(
                rows_df[["chunk_key", "label", "recipe", "depth"]]
            ).values.tobytes()).hexdigest()[:12],
    }
    print(f"[kto_data_prep] {len(rows_df)} rows | "
          f"train/heldout chunks {n_train}/{n_held} | "
          f"stats {dict(stats)} | fallbacks {rationale_fallbacks}")
    return rows_df, metadata
