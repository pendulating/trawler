import marimo

__generated_with = "0.23.3"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Verify: Mar-30 `qwen3.5-9b/base` vs Apr-20 `qwen3.5-9b/instruct`

    Both runs load **the same weights** (`/share/pierson/matt/zoo/models/Qwen3.5-9B`)
    — the Mar 30 run was labeled `base` but its `model_source` points at the
    instruct checkpoint, because the `qwen3.5-9b/base.yaml` config was later
    repointed at the true `Qwen3.5-9B-Base` weights. So this notebook is a
    sanity check: with identical weights + identical seed (777) + identical
    sampling (temp 0.2), how close are the PrivacyLens action completions?

    | Slot | Path | Hydra config |
    |---|---|---|
    | A ("old base") | `multirun/2026-03-30_eval_all/22-41-52/2` | `pipeline=privacylens_clean model=qwen3.5-9b/base` |
    | B ("new instruct") | `multirun/2026-04-20_eval_all/18-15-21/4` | `pipeline=privacylens_clean model=qwen3.5-9b/instruct` |

    **Phases:**

    0. Pre-flight config diff
    1. Schema + row alignment
    2. Action text similarity (exact / edit / embedding)
    3. Structural agreement (Action: parse, tool name)
    4. Downstream agreement on existing judges (QA, leakage, helpfulness)
    5. Eyeball worst divergences

    **Pass thresholds:** >95% exact action match; >0.98 mean embedding cosine
    on non-exact pairs; Cohen's κ >0.85 on QA + leakage binary; helpfulness
    mean absolute difference <0.15.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    RUN_A = Path("/share/pierson/matt/UAIR/multirun/2026-03-30_eval_all/22-41-52/2")
    RUN_B = Path("/share/pierson/matt/UAIR/multirun/2026-04-20_eval_all/18-15-21/4")
    LABEL_A = "Mar30/base"
    LABEL_B = "Apr20/instruct"

    PL_A = RUN_A / "privacylens" / "privacylens_eval" / "outputs"
    PL_B = RUN_B / "privacylens" / "privacylens_eval" / "outputs"
    HYDRA_A = RUN_A / "privacylens" / ".hydra"
    HYDRA_B = RUN_B / "privacylens" / ".hydra"

    REPORT_DIR = Path(__file__).resolve().parent / "tables" / "verify_qwen35_base_vs_instruct_2026_05_27"
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    return (
        HYDRA_A,
        HYDRA_B,
        LABEL_A,
        LABEL_B,
        PL_A,
        PL_B,
        REPORT_DIR,
        RUN_A,
        RUN_B,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Phase 0 — Pre-flight config diff")
    return


@app.cell
def _(HYDRA_A, HYDRA_B, LABEL_A, LABEL_B):
    from omegaconf import OmegaConf as _OmegaConf
    import pandas as _pd

    cfg_a = _OmegaConf.to_container(_OmegaConf.load(HYDRA_A / "config.yaml"), resolve=False)
    cfg_b = _OmegaConf.to_container(_OmegaConf.load(HYDRA_B / "config.yaml"), resolve=False)

    def _flatten(d, prefix=""):
        out = {}
        for k, v in d.items():
            key = f"{prefix}{k}"
            if isinstance(v, dict):
                out.update(_flatten(v, key + "."))
            else:
                out[key] = v
        return out

    flat_a = _flatten(cfg_a)
    flat_b = _flatten(cfg_b)

    interesting_prefixes = (
        "sampling_params.",
        "model.",
        "runtime.",
        "data.",
        "judge_server_url",
    )
    diffs = []
    for k in sorted(set(flat_a) | set(flat_b)):
        if not any(k.startswith(p) for p in interesting_prefixes):
            continue
        va = flat_a.get(k, "<missing>")
        vb = flat_b.get(k, "<missing>")
        if va != vb:
            diffs.append((k, va, vb))

    diff_df = _pd.DataFrame(diffs, columns=["key", LABEL_A, LABEL_B])
    diff_df
    return (diff_df,)


@app.cell(hide_code=True)
def _(diff_df, mo):
    sampling_drift = diff_df[diff_df["key"].str.startswith("sampling_params.")]
    chat_drift = diff_df[diff_df["key"].str.contains("chat_template|thinking|engine_kwargs")]
    if len(sampling_drift) or len(chat_drift):
        msg = "**Drift detected** in sampling / chat-template settings — completions are not expected to be byte-identical even with the same weights:"
    else:
        msg = "**No sampling / chat-template drift.** Completions should be near-byte-identical if vLLM kernels were deterministic."
    mo.md(msg)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(
        r"""
        ### Phase 0b — Judge endpoint diff (resolved from multirun.yaml)

        `judge_server_url` is stored in `.hydra/config.yaml` as the literal
        `${oc.env:JUDGE_SERVER_URL,""}` — the env value never materializes into
        the saved YAML, so a config-yaml diff misses any judge-endpoint drift.
        The actual URL lives in each multirun's `submitit` setup commands. We
        extract it here.
        """
    )
    return


@app.cell
def _(RUN_A, RUN_B):
    import re as _re
    from pathlib import Path as _Path

    JUDGE_SERVER_LOG_DIR = _Path("/share/pierson/matt/UAIR/.slurm_jobs/judge-server")

    def _extract_judge_url(run_dir):
        """Walk up to the multirun root and grep the setup commands."""
        for d in (run_dir, *run_dir.parents):
            mr = d / "multirun.yaml"
            if mr.exists():
                txt = mr.read_text()
                m = _re.search(r"JUDGE_SERVER_URL=([^\s'\"]+)", txt)
                return (str(mr), m.group(1) if m else "<not set>")
        return ("<no multirun.yaml found>", "<unknown>")

    def _eval_first_judge_call_ts(run_dir):
        """Best estimate of when the run hit its judge endpoint.

        Apr 20+ orchestrators log httpx POSTs into privacylens_eval.log; older
        ones don't. Fall back to the Hydra config mtime (≈ run start), and
        finally to the run dir mtime (≈ run finish).
        """
        import datetime as _dt
        log = run_dir / "privacylens" / "privacylens_eval.log"
        if log.exists():
            with open(log) as f:
                for line in f:
                    if "HTTP Request: POST http" in line and "klara" in line:
                        m = _re.match(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})", line)
                        if m:
                            return _dt.datetime.strptime(m.group(1), "%Y-%m-%d %H:%M:%S")
        cfg = run_dir / "privacylens" / ".hydra" / "config.yaml"
        if cfg.exists():
            return _dt.datetime.fromtimestamp(cfg.stat().st_mtime)
        return _dt.datetime.fromtimestamp(run_dir.stat().st_mtime)

    def _resolve_judge_model(url, run_dir):
        """Pick the judge-server SLURM log whose Launching timestamp is the
        latest one ≤ the run's first judge POST. Each log's header has:

            Model: <path>
            Port:  <int>
            [<date>] Launching judge server...
        """
        if not url or "<" in url:
            return "<unknown>"
        m = _re.search(r":(\d+)$", url.rstrip("/"))
        port = m.group(1) if m else None
        if port is None or not JUDGE_SERVER_LOG_DIR.exists():
            return "<unknown>"
        ref_ts = _eval_first_judge_call_ts(run_dir)
        import datetime as _dt

        def _parse_log_launch(head):
            # Header line: "[Mon Apr 20 05:58:11 PM EDT 2026] Launching judge server..."
            # Anchored to start-of-line to avoid ANSI-escape `[` chars elsewhere.
            m = _re.search(
                r"^\[((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[^\]]+)\] Launching judge server",
                head,
                _re.MULTILINE,
            )
            if not m:
                return None
            parts = " ".join(m.group(1).split()).split()
            # ['Mon', 'Apr', '20', '05:58:11', 'PM', 'EDT', '2026']  (12h)
            # or ['Mon', 'Apr', '20', '17:58:11', 'EDT', '2026']     (24h)
            try:
                if len(parts) >= 7 and parts[4] in ("AM", "PM"):
                    s = " ".join(parts[:5] + parts[6:])  # drop tz
                    return _dt.datetime.strptime(s, "%a %b %d %I:%M:%S %p %Y")
                if len(parts) >= 6:
                    s = " ".join(parts[:4] + parts[5:])  # drop tz
                    return _dt.datetime.strptime(s, "%a %b %d %H:%M:%S %Y")
            except ValueError:
                return None
            return None

        best = (None, None, _dt.datetime.min, None)  # (path, model, launch_ts, port)
        for log in JUDGE_SERVER_LOG_DIR.glob("*.out"):
            try:
                head = log.read_text(errors="ignore")[:4000]
            except OSError:
                continue
            mm = _re.search(r"^Model:\s*(\S+)", head, _re.MULTILINE)
            mp = _re.search(r"^Port:\s*(\d+)", head, _re.MULTILINE)
            launch_ts = _parse_log_launch(head)
            if not (mm and mp and launch_ts and mp.group(1) == port):
                continue
            if launch_ts <= ref_ts and launch_ts > best[2]:
                best = (log, mm.group(1), launch_ts, port)
        if best[1]:
            return f"{best[1]}  (launched {best[2].isoformat()}, slurm log: {best[0].name})"
        return "<no judge-server log started before eval>"

    src_a, judge_a = _extract_judge_url(RUN_A)
    src_b, judge_b = _extract_judge_url(RUN_B)
    model_a = _resolve_judge_model(judge_a, RUN_A)
    model_b = _resolve_judge_model(judge_b, RUN_B)
    print(f"A judge_server_url: {judge_a}   (source: {src_a})")
    print(f"A judge_model:      {model_a}")
    print(f"B judge_server_url: {judge_b}   (source: {src_b})")
    print(f"B judge_model:      {model_b}")
    judge_drift = judge_a != judge_b
    if judge_drift:
        print()
        print("** JUDGE ENDPOINT DRIFT — downstream leakage/helpfulness")
        print("** disagreements are dominated by the judge change, not")
        print("** by task-model drift. Use QA-probe agreement (Phase 4)")
        print("** as the cleanest task-only signal.")
    return (judge_drift,)


@app.cell
def _(LABEL_A, LABEL_B, PL_A, PL_B, pd):
    # Judge-stage duration is a strong fingerprint for "is it the same judge?".
    import json as _json

    def _durations(pl_dir):
        mf = pl_dir.parent / "pipeline_manifest.json"
        if not mf.exists():
            return {}
        return {k: v.get("duration_s") for k, v in _json.load(open(mf))["nodes"].items()}

    da = _durations(PL_A)
    db = _durations(PL_B)
    stages = [
        "qa_probe_inference",
        "agent_action_inference",
        "leakage_judge_inference",
        "helpfulness_judge_inference",
    ]
    dur = pd.DataFrame(
        {
            "stage": stages,
            f"{LABEL_A} (s)": [da.get(s) for s in stages],
            f"{LABEL_B} (s)": [db.get(s) for s in stages],
            "ratio B/A": [
                (db.get(s) / da.get(s)) if da.get(s) and db.get(s) else None
                for s in stages
            ],
        }
    )
    dur
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Phase 1 — Schema + row alignment")
    return


@app.cell
def _(LABEL_A, LABEL_B, PL_A, PL_B):
    import pandas as pd

    aai_a = pd.read_parquet(PL_A / "agent_action_inference" / "results.parquet")
    aai_b = pd.read_parquet(PL_B / "agent_action_inference" / "results.parquet")

    ids_a = set(aai_a["record_id"].astype(str))
    ids_b = set(aai_b["record_id"].astype(str))
    only_a = ids_a - ids_b
    only_b = ids_b - ids_a
    shared = ids_a & ids_b

    align = pd.DataFrame({
        "metric": [
            "rows",
            "unique record_ids",
            "only in A",
            "only in B",
            "shared",
        ],
        LABEL_A: [len(aai_a), len(ids_a), len(only_a), "—", "—"],
        LABEL_B: [len(aai_b), len(ids_b), "—", len(only_b), "—"],
        "common": ["—", "—", "—", "—", len(shared)],
    })
    align
    return aai_a, aai_b, only_a, only_b, pd, shared


@app.cell
def _(aai_a, aai_b, pd, shared):
    if not shared:
        raise RuntimeError("No shared record_ids — comparison is impossible.")

    paired = (
        aai_a[["record_id", "generated_action"]]
        .rename(columns={"generated_action": "action_a"})
        .merge(
            aai_b[["record_id", "generated_action"]].rename(
                columns={"generated_action": "action_b"}
            ),
            on="record_id",
            how="inner",
        )
    )
    paired["action_a"] = paired["action_a"].fillna("").astype(str)
    paired["action_b"] = paired["action_b"].fillna("").astype(str)
    paired = paired.reset_index(drop=True)
    print(f"paired rows: {len(paired)}")
    return (paired,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Phase 2 — Action text similarity")
    return


@app.cell
def _(paired):
    from difflib import SequenceMatcher

    def _norm_edit_ratio(a: str, b: str) -> float:
        # SequenceMatcher.ratio() returns similarity in [0, 1]; 1.0 == identical.
        # We report 1 - ratio() as a normalized distance.
        if not a and not b:
            return 0.0
        return 1.0 - SequenceMatcher(a=a, b=b, autojunk=False).ratio()

    paired_scored = paired.assign(
        exact=lambda d: d["action_a"] == d["action_b"],
        len_a=lambda d: d["action_a"].str.len(),
        len_b=lambda d: d["action_b"].str.len(),
        len_delta=lambda d: (d["action_b"].str.len() - d["action_a"].str.len()).abs(),
        norm_edit_dist=lambda d: [
            _norm_edit_ratio(a, b) for a, b in zip(d["action_a"], d["action_b"])
        ],
    )

    exact_rate = paired_scored["exact"].mean()
    nonexact = paired_scored[~paired_scored["exact"]]
    print(f"exact match: {exact_rate:.1%}  ({paired_scored['exact'].sum()} / {len(paired_scored)})")
    if len(nonexact):
        print(
            "non-exact normalized edit distance: "
            f"mean={nonexact['norm_edit_dist'].mean():.3f}  "
            f"median={nonexact['norm_edit_dist'].median():.3f}  "
            f"p90={nonexact['norm_edit_dist'].quantile(0.9):.3f}  "
            f"max={nonexact['norm_edit_dist'].max():.3f}"
        )
        print(
            "length delta on non-exact: "
            f"mean={nonexact['len_delta'].mean():.0f}  "
            f"median={nonexact['len_delta'].median():.0f}  "
            f"max={nonexact['len_delta'].max()}"
        )
    return exact_rate, nonexact, paired_scored


@app.cell
def _(nonexact, paired_scored, pd):
    # Embedding similarity on non-exact pairs. Skipped if there are none.
    cos = None
    if len(nonexact):
        from sentence_transformers import SentenceTransformer
        import torch
        import numpy as np

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device=device)
        emb_a = model.encode(
            nonexact["action_a"].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        emb_b = model.encode(
            nonexact["action_b"].tolist(),
            convert_to_numpy=True,
            normalize_embeddings=True,
            batch_size=64,
            show_progress_bar=False,
        )
        cos = (emb_a * emb_b).sum(axis=1)
        print(
            f"non-exact embedding cosine: mean={cos.mean():.4f}  "
            f"median={float(np.median(cos)):.4f}  min={cos.min():.4f}"
        )
        # Attach back to paired_scored
        paired_scored.loc[~paired_scored["exact"], "embed_cos"] = cos
    else:
        print("All actions are byte-identical; skipping embedding step.")
    return (paired_scored,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Phase 3 — Structural agreement")
    return


@app.cell
def _(paired_scored, pd):
    import re as _re

    _ACTION_RE = _re.compile(r"^\s*Action:\s*(\S+)", _re.MULTILINE)

    def _tool(s: str) -> str:
        m = _ACTION_RE.search(s or "")
        return m.group(1) if m else ""

    struct = paired_scored.assign(
        has_action_a=lambda d: d["action_a"].str.contains(r"\bAction:\b", regex=True, na=False),
        has_action_b=lambda d: d["action_b"].str.contains(r"\bAction:\b", regex=True, na=False),
        tool_a=lambda d: d["action_a"].map(_tool),
        tool_b=lambda d: d["action_b"].map(_tool),
    )
    struct["tool_match"] = (struct["tool_a"] == struct["tool_b"]) & (struct["tool_a"] != "")
    struct["both_have_action"] = struct["has_action_a"] & struct["has_action_b"]
    struct["neither_has_action"] = ~struct["has_action_a"] & ~struct["has_action_b"]
    struct["xor_action"] = struct["has_action_a"] ^ struct["has_action_b"]

    summary = pd.DataFrame({
        "metric": [
            "has Action: (A)",
            "has Action: (B)",
            "both have Action:",
            "neither has Action:",
            "xor (one missing)",
            "same tool (when both present)",
        ],
        "value": [
            struct["has_action_a"].mean(),
            struct["has_action_b"].mean(),
            struct["both_have_action"].mean(),
            struct["neither_has_action"].mean(),
            struct["xor_action"].mean(),
            struct.loc[struct["both_have_action"], "tool_match"].mean()
            if struct["both_have_action"].any()
            else float("nan"),
        ],
    })
    summary["value"] = summary["value"].map(lambda v: f"{v:.1%}" if pd.notna(v) else "—")
    summary
    return (struct,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Phase 4 — Downstream agreement on existing judges")
    return


@app.cell
def _(PL_A, PL_B, pd):
    qa_a = pd.read_parquet(PL_A / "qa_probe_inference" / "results.parquet")
    qa_b = pd.read_parquet(PL_B / "qa_probe_inference" / "results.parquet")
    leak_a = pd.read_parquet(PL_A / "leakage_judge_inference" / "results.parquet")
    leak_b = pd.read_parquet(PL_B / "leakage_judge_inference" / "results.parquet")
    help_a = pd.read_parquet(PL_A / "helpfulness_judge_inference" / "results.parquet")
    help_b = pd.read_parquet(PL_B / "helpfulness_judge_inference" / "results.parquet")
    return help_a, help_b, leak_a, leak_b, qa_a, qa_b


@app.cell
def _(leak_a, leak_b, pd):
    from sklearn.metrics import cohen_kappa_score

    lk = (
        leak_a[["record_id", "leak_flag"]]
        .rename(columns={"leak_flag": "leak_a"})
        .merge(
            leak_b[["record_id", "leak_flag"]].rename(columns={"leak_flag": "leak_b"}),
            on="record_id",
            how="inner",
        )
        .dropna(subset=["leak_a", "leak_b"])
    )
    lk["leak_a"] = lk["leak_a"].astype(int)
    lk["leak_b"] = lk["leak_b"].astype(int)
    agree_leak = (lk["leak_a"] == lk["leak_b"]).mean()
    kappa_leak = cohen_kappa_score(lk["leak_a"], lk["leak_b"])
    leak_xtab = pd.crosstab(lk["leak_a"], lk["leak_b"], margins=True)
    print(f"leakage: n={len(lk)}  agree={agree_leak:.1%}  Cohen's κ={kappa_leak:.3f}")
    leak_xtab
    return cohen_kappa_score, leak_xtab


@app.cell
def _(cohen_kappa_score, help_a, help_b, pd):
    from scipy.stats import spearmanr

    hh = (
        help_a[["record_id", "helpfulness_score", "helpfulness_binary"]]
        .rename(columns={"helpfulness_score": "score_a", "helpfulness_binary": "bin_a"})
        .merge(
            help_b[["record_id", "helpfulness_score", "helpfulness_binary"]].rename(
                columns={"helpfulness_score": "score_b", "helpfulness_binary": "bin_b"}
            ),
            on="record_id",
            how="inner",
        )
        .dropna(subset=["score_a", "score_b"])
    )
    hh["score_a"] = hh["score_a"].astype(int)
    hh["score_b"] = hh["score_b"].astype(int)
    hh["bin_a"] = hh["bin_a"].astype(int)
    hh["bin_b"] = hh["bin_b"].astype(int)

    mad = (hh["score_a"] - hh["score_b"]).abs().mean()
    exact_score = (hh["score_a"] == hh["score_b"]).mean()
    rho = spearmanr(hh["score_a"], hh["score_b"]).statistic
    kappa_bin = cohen_kappa_score(hh["bin_a"], hh["bin_b"])
    help_xtab = pd.crosstab(hh["score_a"], hh["score_b"], margins=True)
    print(
        f"helpfulness: n={len(hh)}  exact={exact_score:.1%}  MAD={mad:.3f}  "
        f"Spearman ρ={rho:.3f}  binary κ={kappa_bin:.3f}"
    )
    help_xtab
    return (hh,)


@app.cell
def _(cohen_kappa_score, pd, qa_a, qa_b):
    # qa_probe has 3 axes per record; join on (record_id, _qa_axis).
    qa = (
        qa_a[["record_id", "_qa_axis", "generated_text"]]
        .rename(columns={"generated_text": "qa_a"})
        .merge(
            qa_b[["record_id", "_qa_axis", "generated_text"]].rename(
                columns={"generated_text": "qa_b"}
            ),
            on=["record_id", "_qa_axis"],
            how="inner",
        )
    )
    qa["qa_a"] = qa["qa_a"].fillna("").astype(str)
    qa["qa_b"] = qa["qa_b"].fillna("").astype(str)
    qa["exact"] = qa["qa_a"] == qa["qa_b"]

    print(f"qa_probe rows joined: {len(qa)}")
    print(f"qa_probe exact text match: {qa['exact'].mean():.1%}")

    # Parsed answers if a Yes/No or A/B/C/D pattern is present.
    import re as _re
    _YN = _re.compile(r"\b(yes|no)\b", _re.IGNORECASE)

    def _parse_yn(s):
        m = _YN.search(s)
        return m.group(1).lower() if m else ""

    qa["yn_a"] = qa["qa_a"].map(_parse_yn)
    qa["yn_b"] = qa["qa_b"].map(_parse_yn)
    valid = qa[(qa["yn_a"] != "") & (qa["yn_b"] != "")]
    if len(valid):
        kappa_qa = cohen_kappa_score(valid["yn_a"], valid["yn_b"])
        agree_qa = (valid["yn_a"] == valid["yn_b"]).mean()
        print(
            f"qa_probe Yes/No (n={len(valid)}/{len(qa)}): "
            f"agree={agree_qa:.1%}  Cohen's κ={kappa_qa:.3f}"
        )
        by_axis = (
            valid.groupby("_qa_axis")
            .apply(
                lambda g: pd.Series({
                    "n": len(g),
                    "agree": (g["yn_a"] == g["yn_b"]).mean(),
                    "kappa": cohen_kappa_score(g["yn_a"], g["yn_b"]),
                })
            )
            .reset_index()
        )
    else:
        by_axis = pd.DataFrame()
    by_axis
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("## Phase 5 — Eyeball worst divergences")
    return


@app.cell
def _(LABEL_A, LABEL_B, REPORT_DIR, hh, paired_scored, pd):
    # Top-20 by edit distance.
    top_edit = (
        paired_scored.sort_values("norm_edit_dist", ascending=False)
        .head(20)
        [["record_id", "norm_edit_dist", "len_a", "len_b", "action_a", "action_b"]]
        .reset_index(drop=True)
    )

    # Top-20 by helpfulness score delta.
    hh_delta = hh.assign(score_delta=(hh["score_a"] - hh["score_b"]).abs())
    top_help = (
        hh_delta.sort_values("score_delta", ascending=False)
        .head(20)
        [["record_id", "score_a", "score_b", "bin_a", "bin_b", "score_delta"]]
        .reset_index(drop=True)
    )

    # Write the eyeball report as markdown.
    md_path = REPORT_DIR / "comparison_report.md"
    paired_path = REPORT_DIR / "paired_actions.parquet"
    paired_scored.to_parquet(paired_path, index=False)

    def _fmt(s, n=300):
        s = (s or "").strip().replace("\n", " ⏎ ")
        return s[:n] + ("…" if len(s) > n else "")

    lines = [
        f"# {LABEL_A} vs {LABEL_B} — privacylens action comparison",
        "",
        "## Top-20 worst action edit distances",
        "",
        "| record_id | dist | len A | len B | action A | action B |",
        "|---|---:|---:|---:|---|---|",
    ]
    for _, r in top_edit.iterrows():
        lines.append(
            f"| {r['record_id']} | {r['norm_edit_dist']:.3f} | {r['len_a']} | {r['len_b']} | "
            f"{_fmt(r['action_a'])} | {_fmt(r['action_b'])} |"
        )
    lines += [
        "",
        "## Top-20 worst helpfulness score deltas",
        "",
        "| record_id | score A | score B | bin A | bin B | Δ |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for _, r in top_help.iterrows():
        lines.append(
            f"| {r['record_id']} | {int(r['score_a'])} | {int(r['score_b'])} | "
            f"{int(r['bin_a'])} | {int(r['bin_b'])} | {int(r['score_delta'])} |"
        )
    md_path.write_text("\n".join(lines))
    print(f"wrote {md_path}")
    print(f"wrote {paired_path}")
    top_edit[["record_id", "norm_edit_dist", "len_a", "len_b"]]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Verdict — read as two separate axes

    The original question "are these the same model?" splits into two when the
    runs use different external judges. Read these independently:

    **Task-model similarity** (no judge involved):
    - Phase 2 action exact-match + non-exact embedding cosine
    - Phase 3 structural agreement (same tool invoked)
    - Phase 4 **QA probe** Yes/No κ (the QA probe is run by the task model on
      the local vLLM, not by the external judge — so this is the cleanest
      head-to-head signal that the task weights match)

    **Judge-mediated similarity** (dominated by which judge model served the
    request):
    - Phase 4 leakage κ + helpfulness MAD
    - Phase 0b judge-endpoint drift and Phase 4 duration ratio give the
      "is it even the same judge?" answer first

    If Phase 0b shows a judge-endpoint change AND Phase 4 duration ratio on
    `*_judge_inference` is ≫1, then leakage/helpfulness κ are not measuring
    task-model agreement — they are measuring the new vs old judge's verdict
    on the same (or near-same) actions. To get a clean head-to-head on the
    judge-mediated benchmarks under those conditions, you need to rejudge
    both action parquets with a single fixed judge (e.g. via
    `pipeline=privacylens_clean_batch judge.mode=batch_export` on each
    `agent_action_inference/results.parquet`).

    The `tables/verify_qwen35_base_vs_instruct_2026_05_27/` sidecar holds
    `comparison_report.md` (eyeball table) and `paired_actions.parquet`
    (every per-row stat) for ad-hoc digging.
    """)
    return


if __name__ == "__main__":
    app.run()
