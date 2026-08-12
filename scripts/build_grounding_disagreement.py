#!/usr/bin/env python3
"""Per-flow table for the normative-grounding disagreement analysis.

Two appropriateness labelings over the SAME 16,200 camera-ready Gemma-4 fiction
flows, so they can be compared row-wise:

``teacher``   the teacher's own ``ci_appropriateness``, emitted during CI
              extraction with no norm in context. **This is what SFT trains
              on** (``sft_data_prep.py:233``), so it is the SFT treatment's
              label.

``grounded``  the flow re-classified through the novel's own normative
              universe: embed the six CI fields, retrieve the nearest
              ``governs_info_flow`` norm from the *same book*, and read the
              flow's appropriateness off that norm's
              ``normative_force x act_polarity``
              (:func:`deontic.flow_appropriateness`). This is the normative
              grounding treatment's label, and it is exactly what R-DIRECT
              consumes as gold (``aux_scorers.make_direct_chunk_gold``).

Everything runs through production code — ``NormRetriever`` with the
``governs_info_flow`` filter over the production per-book embedding index, and
``majority_gold`` / ``flow_appropriateness`` for the label — so the output is
the same object the reward saw, not a reimplementation of it.

**No GPU and no embedding server are needed.** The flow-side queries are
byte-identical to the ones ``make_direct_chunk_gold`` builds (verified over all
16,200 rows), and their embeddings are already cached by
``scripts/norm_neighbors.py`` under the ``rground`` space. If the cache ever
misses, pass ``--embed-server-url`` and it re-embeds.

FAITHFULNESS GUARD: the script refuses to write unless it reproduces the wiki
§17 audit (kappa 0.0532, raw agreement 0.7057, class priors 11.0% / 25.8%,
median retrieval margin 0.0136 over 15,493 dual-labelled flows). Those numbers
were measured 2026-08-03 through the live embedding server; if the offline path
disagrees, it is measuring something else and its output means nothing.

POLICY ARMS (added 2026-08-05, wiki/2026-08-05_distilled_grounding_plan.md):
``--flows`` / ``--universe`` / ``--arm`` point the same machinery at a
FINE-TUNED policy's flows instead of the teacher's. The ``teacher`` column then
holds *that policy's* own ungrounded ``ci_appropriateness`` — the column names
are deliberately unchanged so the existing notebook reads either table. The
faithfulness guard auto-disarms when the inputs are not the teacher defaults
(it is a check that we reproduced the teacher audit, and a policy's flows are a
different object by construction); the reason is recorded in the manifest.

Usage:
    # teacher (unchanged; guard armed)
    python scripts/build_grounding_disagreement.py \
        --out outputs/2026-08-03_grounding_disagreement

    # a policy arm (guard auto-disarmed, needs a live embedding server)
    python scripts/build_grounding_disagreement.py \
        --arm m2-full \
        --flows outputs/<run>/COLM_flows_fiction_policy/outputs/ci_extraction/ci_flows.parquet \
        --embed-server-url http://klara:8001 \
        --out outputs/2026-08-05_distilled_grounding/m2-full
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dagspaces.common.stage_utils import ensure_dotenv  # noqa: E402
from dagspaces.grpo_training.stages.clients import NormRetriever  # noqa: E402
from dagspaces.grpo_training.stages.deontic import (  # noqa: E402
    canonical_force,
    flow_appropriateness,
)
from dagspaces.grpo_training.stages.norm_universe import EMBED_INSTRUCTION  # noqa: E402
from dagspaces.grpo_training.stages.online_rground import (  # noqa: E402
    _flatten_flow,
    _flow_to_query,
)
from scripts.norm_neighbors import DEFAULT_CACHE_DIR, embed_cached  # noqa: E402

# --- canonical camera-ready inputs (same artifacts as the other CR notebooks)
FLOWS = ROOT / (
    "outputs/2026-07-12_fiction10_flows_gemma4/23-14-17/"
    "COLM_flows_fiction_gemma4/outputs/ci_extraction/ci_flows.parquet"
)
NORMS = ROOT / (
    "outputs/2026-07-12_fiction10_norms_gemma4/18-36-28/"
    "COLM_norms_fiction_gemma4/outputs/extraction/structured_norms.parquet"
)
# The polarity-merged universe is the one production reads (server.env
# NORM_UNIVERSES_PATH). Without act_polarity, 19% of grounded labels invert.
UNIVERSE = ROOT / "outputs/2026-07-25_universe_fiction10_polarity"

EMB_MODEL = "/share/pierson/matt/zoo/models/Qwen3-Embedding-8B"

#: Exactly ``make_direct_chunk_gold``'s ``field_cols`` — must stay in lockstep
#: with ``DirectChunkGold.MATCH_FIELDS``. Note this excludes ``norms_invoked``:
#: teacher-side gold queries are built from these six fields only.
FIELD_COLS = {
    "ci_sender": "sender",
    "ci_recipient": "recipient",
    "ci_subject": "subject",
    "ci_information_type": "information_type",
    "ci_transmission_principle": "transmission_principle",
    "ci_context": "context",
}

# --- wiki §17 audit, 2026-08-03, measured through the live embedding server --
AUDIT = {
    "n": 15493,
    "kappa": 0.0532,
    "agreement": 0.7057,
    "teacher_inappr": 0.110,
    "grounded_inappr": 0.258,
    "margin_median": 0.0136,
}
TOL = {"kappa": 0.004, "agreement": 0.004, "teacher_inappr": 0.004,
       "grounded_inappr": 0.004, "margin_median": 0.002, "n": 40}


def teacher_queries(flows: pd.DataFrame) -> list[str]:
    """Flow retrieval queries, byte-identical to ``make_direct_chunk_gold``."""
    out = []
    for _, row in flows.iterrows():
        # pd.notna guard: float('nan') is truthy, so an unguarded NaN cell
        # would put a literal "nan" token into the query one-sidedly.
        flat = {v: (row.get(c) if pd.notna(row.get(c)) else None)
                for c, v in FIELD_COLS.items()}
        out.append(_flow_to_query(_flatten_flow(flat)))
    return out


def cohen_kappa(a: pd.Series, b: pd.Series) -> tuple[float, float, float]:
    """(kappa, observed agreement, chance agreement) for two label series."""
    labels = sorted(set(a) | set(b))
    po = float((a.to_numpy() == b.to_numpy()).mean())
    pe = float(sum((a == lab).mean() * (b == lab).mean() for lab in labels))
    kappa = (po - pe) / (1.0 - pe) if pe < 1.0 else 0.0
    return kappa, po, pe


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/2026-08-03_grounding_disagreement")
    ap.add_argument("--top-k", type=int, default=3,
                    help="neighbours retained per flow (production gold uses "
                         "top-1; k>1 is kept for the teacher-rejudge stage)")
    ap.add_argument("--embed-server-url", default="",
                    help="only needed on a cache miss")
    ap.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR))
    ap.add_argument("--skip-guard", action="store_true",
                    help="write even if the wiki §17 audit is not reproduced "
                         "(for deliberate input changes; say so downstream)")
    ap.add_argument("--arm", default="teacher",
                    help="label for this extractor, recorded in the output and "
                         "the manifest (e.g. teacher, base, sft, m2-full, "
                         "k3-verdict)")
    ap.add_argument("--flows", default="",
                    help="flows parquet; defaults to the teacher's camera-ready "
                         "fiction10 flows")
    ap.add_argument("--universe", default="",
                    help="norm-universe dir (must carry act_polarity); defaults "
                         "to the polarity-merged fiction10 universe")
    args = ap.parse_args()

    ensure_dotenv()
    out_dir = ROOT / args.out if not Path(args.out).is_absolute() else Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- inputs ---------------------------------------------------------
    flows_path = Path(args.flows) if args.flows else FLOWS
    if not flows_path.is_absolute():
        flows_path = ROOT / flows_path
    universe = Path(args.universe) if args.universe else UNIVERSE
    if not universe.is_absolute():
        universe = ROOT / universe

    # The guard checks that we reproduced the teacher-side §17 audit. A policy
    # arm's flows are a different object by construction, so the guard cannot
    # apply — disarm it explicitly and record why, rather than letting a caller
    # pass --skip-guard and lose the distinction between "deliberately
    # different inputs" and "the teacher rebuild broke".
    is_teacher_inputs = (flows_path == FLOWS and universe == UNIVERSE)
    guard_armed = is_teacher_inputs and not args.skip_guard
    if not is_teacher_inputs:
        print(f"[guard] DISARMED — non-default inputs (arm={args.arm!r}). "
              "The §17 audit describes the teacher's flows only.")

    flows = pd.read_parquet(flows_path)
    universes = json.loads((universe / "norm_universes.json").read_text())
    print(f"[build] arm {args.arm}  flows {len(flows)}  "
          f"universe books {len(universes)}")
    print(f"[build] flows path: {flows_path}")
    print(f"[build] universe  : {universe}")

    missing = [c for c in (*FIELD_COLS, "ci_appropriateness",
                           "gutenberg_id", "chunk_id") if c not in flows.columns]
    if missing:
        raise RuntimeError(
            f"[build] {flows_path} is missing required columns {missing}. "
            "A policy arm's ci_flows.parquet must carry the same schema as the "
            "teacher's — ci_extraction's guided decoding guarantees it, so a "
            "miss here means the wrong parquet."
        )

    n_gov = sum(1 for ns in universes.values()
                for n in ns if n.get("governs_info_flow") is True)
    n_pol = sum(1 for ns in universes.values()
                for n in ns if n.get("governs_info_flow") is True
                and n.get("act_polarity") is not None)
    print(f"[build] governing norms {n_gov}; with act_polarity {n_pol}")
    if n_pol == 0:
        raise RuntimeError(
            f"[build] {universe} carries no act_polarity — "
            "flow_appropriateness would silently fall back to 'performing' "
            "and 19% of grounded labels would invert. Point --universe at the "
            "polarity-merged universe (scripts/apply_act_polarity.py)."
        )

    # ---- flow-side embeddings (production teacher queries) --------------
    queries = teacher_queries(flows)
    flow_emb = embed_cached(
        queries, EMBED_INSTRUCTION,
        server_url=args.embed_server_url, model_name=EMB_MODEL,
        cache_dir=Path(args.cache_dir), label="flows(teacher-query)",
    )
    flow_emb = np.asarray(flow_emb, dtype=np.float32)
    flow_emb /= np.maximum(np.linalg.norm(flow_emb, axis=1, keepdims=True), 1e-9)

    # ---- production retriever, restricted index -------------------------
    retriever = NormRetriever(
        norm_universes=universes,
        embeddings_dir=str(universe / "embeddings"),
        embedding_client=None,          # pre-computed .npy only; no server
        top_k=max(2, args.top_k),
        norm_filter=lambda n: n.get("governs_info_flow") is True,
    )

    # ---- retrieve + label ------------------------------------------------
    sid = flows["gutenberg_id"].astype(str).to_numpy()
    rows = []
    for i in range(len(flows)):
        raw, sims = retriever.retrieve(
            flow_emb[i], sid[i], return_scores=True, top_k=max(2, args.top_k)
        )
        got = json.loads(raw) if isinstance(raw, str) else (raw or [])
        top = got[0] if got and isinstance(got[0], dict) else None
        # Production drops a flow from the denominator when the top hit is not
        # a governing norm; with the restricted index that can only happen when
        # the book has none at all.
        if top is None or top.get("governs_info_flow") is not True:
            rows.append({"grounded": None, "top_sim": np.nan, "margin": np.nan,
                         "top_force": None, "top_polarity": None,
                         "top_articulation": None, "n_candidates": len(got),
                         "nbr_articulation": [], "nbr_force": [],
                         "nbr_polarity": [], "nbr_sim": [], "nbr_implies": []})
            continue
        force = str(top.get("normative_force") or "")
        # The runner-up neighbours are retained so the viewer can show what
        # else was near — the median top1-top2 margin is 0.0136, so "which norm
        # ranked first" is often a near-tie and inspecting it matters.
        nbrs = [nb for nb in got if isinstance(nb, dict)]
        rows.append({
            "grounded": flow_appropriateness(force, top.get("act_polarity")),
            "top_sim": float(sims[0]) if sims else np.nan,
            "margin": float(sims[0] - sims[1]) if len(sims) > 1 else np.nan,
            "top_force": canonical_force(force),
            "top_polarity": top.get("act_polarity"),
            "top_articulation": top.get("norm_articulation"),
            "n_candidates": len(got),
            "nbr_articulation": [str(nb.get("norm_articulation") or "")
                                 for nb in nbrs],
            "nbr_force": [canonical_force(str(nb.get("normative_force") or ""))
                          or "" for nb in nbrs],
            "nbr_polarity": [str(nb.get("act_polarity") or "") for nb in nbrs],
            "nbr_sim": [float(s) for s in sims[:len(nbrs)]],
            "nbr_implies": [
                flow_appropriateness(str(nb.get("normative_force") or ""),
                                     nb.get("act_polarity")) or ""
                for nb in nbrs
            ],
        })

    ret = pd.DataFrame(rows)
    out = pd.DataFrame({
        "gutenberg_id": flows["gutenberg_id"].astype(str).to_numpy(),
        "chunk_id": flows["chunk_id"].astype(str).to_numpy(),
        "ci_flow_index": flows.get("ci_flow_index"),
        "arm": args.arm,
        "book_title": (flows["book_title"].to_numpy()
                       if "book_title" in flows.columns else None),
        "teacher": flows["ci_appropriateness"].astype("string")
                        .str.strip().str.lower().to_numpy(),
        "ci_sender": flows["ci_sender"].to_numpy(),
        "ci_recipient": flows["ci_recipient"].to_numpy(),
        "ci_information_type": flows["ci_information_type"].to_numpy(),
        "ci_transmission_principle": flows["ci_transmission_principle"].to_numpy(),
        "ci_context": flows["ci_context"].to_numpy(),
        "ci_subject": flows["ci_subject"].to_numpy(),
        "ci_confidence_qual": flows.get("ci_confidence_qual"),
        # Qualitative inspection payload (scripts/../norm_grounding_disagreement
        # viewer). `ci_norms_invoked` is the norm the TEACHER cited on its own,
        # with no universe in context — the natural comparison against whatever
        # retrieval surfaced.
        "ci_flow_snippet": flows.get("ci_flow_snippet"),
        # For a policy arm this is the norm THAT POLICY cited unprompted — the
        # object §4.2 of the plan analyses against the book's actual universe.
        "ci_norms_invoked": (
            flows["ci_norms_invoked"].astype("string")
            if "ci_norms_invoked" in flows.columns else None
        ),
        "flow_query": queries,
    })
    out = pd.concat([out.reset_index(drop=True), ret.reset_index(drop=True)], axis=1)

    # `teacher` carries an "ambiguous" third class the deontic gradient cannot
    # express. Two codings are emitted and the notebook reports both:
    #
    # `teacher_ci`  — CI coding, THE PRIMARY ANALYSIS. Under contextual
    #   integrity a flow is appropriate only insofar as it conforms to the
    #   contextual informational norms in force; a flow whose conformity cannot
    #   be established is not thereby sanctioned. So "ambiguous" resolves to
    #   `inappropriate`, and every scored flow enters the comparison.
    # `dual`        — binary intersection, i.e. drop the ambiguous rows. Kept
    #   as the sensitivity variant, and because it is the subset the wiki §17
    #   audit measured, so the faithfulness guard must run on it.
    out["scored"] = out["grounded"].isin(["appropriate", "inappropriate"])
    out["teacher_ci"] = np.where(
        out["teacher"] == "ambiguous", "inappropriate", out["teacher"]
    )
    out["dual"] = (
        out["teacher"].isin(["appropriate", "inappropriate"]) & out["scored"]
    )
    out["agree_ci"] = np.where(
        out["scored"], out["teacher_ci"] == out["grounded"], None
    )
    out["agree"] = np.where(out["dual"], out["teacher"] == out["grounded"], None)

    # ---- faithfulness guard ---------------------------------------------
    # Runs on the binary intersection: that is the subset §17 measured, so it
    # is the only subset whose numbers are comparable to the published audit.
    d = out[out["dual"]]
    kappa, po, pe = cohen_kappa(d["teacher"], d["grounded"])
    got_stats = {
        "n": int(len(d)),
        "kappa": kappa,
        "agreement": po,
        "teacher_inappr": float((d["teacher"] == "inappropriate").mean()),
        "grounded_inappr": float((d["grounded"] == "inappropriate").mean()),
        "margin_median": float(d["margin"].median()),
    }
    failures = []
    if is_teacher_inputs:
        print("\n[guard] offline rebuild vs wiki §17 (live-server audit, 2026-08-03)")
        print(f"  {'stat':<18} {'audit':>10} {'rebuilt':>10} {'delta':>10}  ok")
        for k, want in AUDIT.items():
            have = got_stats[k]
            delta = have - want
            ok = abs(delta) <= TOL[k]
            if not ok:
                failures.append(f"{k}: audit {want}, rebuilt {have:.4f}")
            print(f"  {k:<18} {want:>10.4f} {have:>10.4f} {delta:>+10.4f}  "
                  f"{'OK' if ok else 'FAIL'}")
        print(f"  chance agreement   {'':>10} {pe:>10.4f}")
    else:
        print(f"\n[stats] arm={args.arm} (guard disarmed)")
        for k, have in got_stats.items():
            print(f"  {k:<18} {have:>10.4f}")
        print(f"  chance agreement   {pe:>10.4f}")

    if failures and not args.skip_guard:
        raise RuntimeError(
            "[guard] the offline rebuild does not reproduce the wiki §17 audit:\n  "
            + "\n  ".join(failures)
            + "\nRefusing to write — this table would not be the object the "
              "reward saw. Re-run with --skip-guard only if an input changed "
              "deliberately."
        )
    if failures:
        print("[guard] OVERRIDDEN via --skip-guard; downstream must say so.")

    # ---- write -----------------------------------------------------------
    path = out_dir / "flow_grounding_labels.parquet"
    out.to_parquet(path, index=False)
    meta = {
        "built": "scripts/build_grounding_disagreement.py",
        "arm": args.arm,
        "flows": str(flows_path),
        "universe": str(universe),
        "embedding_model": EMB_MODEL,
        "instruction": EMBED_INSTRUCTION,
        "top_k": args.top_k,
        "n_flows": int(len(out)),
        "n_scored": int(out["scored"].sum()),
        "n_dual_labelled": int(len(d)),
        "n_ambiguous_recoded": int((out["teacher"] == "ambiguous").sum()),
        "governing_norms": n_gov,
        "governing_norms_with_polarity": n_pol,
        "guard": {
            "armed": guard_armed,
            "disarm_reason": (
                None if is_teacher_inputs
                else "non-default inputs: the wiki §17 audit describes the "
                     "teacher's flows, so it cannot validate this arm"
            ),
            "audit": AUDIT if is_teacher_inputs else None,
            "rebuilt": got_stats,
            "passed": not failures,
            "overridden": bool(failures),
        },
    }
    (out_dir / "build_metadata.json").write_text(
        json.dumps(meta, indent=2) + "\n"
    )
    print(f"\n[build] wrote {path} ({len(out)} rows, {len(d)} dual-labelled)")
    print(f"[build] wrote {out_dir / 'build_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
