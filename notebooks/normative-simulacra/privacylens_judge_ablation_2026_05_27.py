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
    # PrivacyLens — judge ablation across multiruns

    **Question:** how much of the reported PrivacyLens leakage / helpfulness
    spread is attributable to **which LLM judged the actions**, vs. real
    task-model differences? The seed notebook
    (`verify_qwen35_base_vs_instruct_2026_05_27.py`) showed for a single pair
    that swapping `Qwen3-32B-AWQ → Gemma-4-31B-it` flips leakage from 27% to
    65% on byte-identical task weights and a near-identical action parquet.
    This notebook scales that comparison across **five sweeps** with four
    distinct judges (Qwen3-32B-AWQ, Gemma-4-31B-it, Qwen3.6-27B ×2, gpt-5.2).

    | Sweep | Multirun | Judge | Launched |
    |---|---|---|---|
    | Mar30 | `2026-03-30_eval_all/22-41-52` | **Qwen3-32B-AWQ** | 2026-03-30 21:18 |
    | Apr20 | `2026-04-20_eval_all/18-15-21` | **Gemma-4-31B-it** | 2026-04-20 17:58 |
    | Apr24 | `2026-04-24_eval_all/10-13-47` | **Qwen3.6-27B** | 2026-04-24 09:59 |
    | May27 | `2026-03-30_eval_all/22-41-52` (rejudged) | **gpt-5.2** | 2026-05-27 (offline OpenAI Batch) |
    | May30 | `2026-05-30_eval_all/21-35-36` | **Qwen3.6-27B** | 2026-05-30 21:35 |

    **May27 is a pseudo-sweep**: it reuses the Mar 30 agent-action parquets but
    swaps in gpt-5.2 as the leakage + helpfulness judge via offline OpenAI
    Batch (see `scripts/privacylens_finalize_openai_batches_2026_05_27.py`).
    Only the 3 Qwen3.5-9B variants (base/sft-ci/grpo-v3-vr05-lambda05) were
    rejudged, so this sweep only contributes rows for those 3 task models.

    **May30 is the full COLM results sweep** — the canonical async-judged run
    (`eval_all` → `privacylens_async_finalize`, judged by Qwen3.6-27B served
    as a sidecar on klara:8002). It covers **21 task models** (the full COLM
    line-up; only `cirl/base` failed). Crucially it uses the **post-ReAct**
    action prompt (mean action ≈ 7× longer than the Mar30/Apr20/Apr24 era —
    e.g. 120 → 898 chars on Qwen3.5-9B-Base), so its actions are **not**
    byte-identical to the older sweeps. Per the analysis decision for this
    revision we **treat that prompt-drift contribution as negligible relative
    to the judge effect and lump all runs together** — May30 flows through
    every phase as a peer sweep. Phase G still measures the action drift so it
    stays visible rather than silent; read the cross-judge Δ rows that mix
    May30 with a pre-ReAct sweep as judge-swap **plus** residual prompt-drift.

    **Two Qwen3.6-27B sweeps.** Apr24 and May30 share the same judge family
    but differ in action prompt era, so the Apr24 → May30 pair is effectively
    a same-judge / prompt-drift probe rather than a judge swap.

    **May 12 sweeps still excluded** — they are superseded by May30, which uses
    the same post-ReAct prompt era but adds the full task-model line-up and the
    canonical Qwen3.6-27B judge. The original exclusion rationale (the action
    prompt was rewritten between Apr 24 and May 12 — commit `44484d4` +
    follow-ups — to be byte-identical to upstream SALT-NLP/PrivacyLens, roughly
    doubling mean action length) is documented in `wiki/changelog/privacylens-
    action-prompt-react-rewrite-2026-04-26.md`.

    **Design:**

    1. **Phase A** — inventory all five multiruns.
    2. **Phase B** — group runs by `(model_source, lora_path)`. Keep **every**
       run with metrics on disk (so the full May30 COLM line-up is included),
       and tag each group with `n_sweeps` / `n_judges` so the cross-judge
       phases can restrict to multi-coverage models where that matters.
    3. **Phase C** — resolve judge identity per sweep from
       `JUDGE_SERVER_URL` + `.slurm_jobs/judge-server/*.out` headers.
    4. **Phase D** — long-form metrics table: (task model, judge) → metrics.
    5. **Phase E** — pairwise judge shifts at fixed task model.
    6. **Phase F** — cross-judge σ at fixed task vs cross-model σ at fixed
       judge.
    7. **Phase G** — task-output sanity per cross-sweep pair (action exact
       match, QA-probe κ).
    8. **Phase H** — save artifacts + verdict.
    """)
    return


@app.cell
def _():
    from pathlib import Path

    # Each sweep loads its per-sub-run metrics from `<sub_run>/.../outputs/<metrics_subdir>/metrics.parquet`.
    # The default ("compute_metrics") matches the original eval_all output layout. The May27
    # pseudo-sweep reuses the Mar30 sub-runs but loads "compute_metrics_gpt52/" — those metrics
    # were computed by scripts/privacylens_finalize_openai_batches_2026_05_27.py from offline
    # OpenAI Batch outputs (gpt-5.2 rejudging the Mar30 agent actions).
    SWEEPS = [
        {
            "label": "Mar30",
            "root": Path("/share/pierson/matt/UAIR/multirun/2026-03-30_eval_all/22-41-52"),
            "metrics_subdir": "compute_metrics",
        },
        # Apr16 EXCLUDED (2026-05-28): the Apr16 Gemma-4-31B-it sweep used a
        # judge_server endpoint that subsequent investigation showed produced
        # near-identical numbers to Apr20 (same judge, same server config; see
        # the Mar30↔May27 vs Apr16↔Apr20 comparison in earlier runs of this
        # notebook). Dropping it removes a redundant Gemma row from every
        # downstream plot without changing the cross-judge story. Re-enable by
        # uncommenting if you want a same-judge noise-floor check.
        # {
        #     "label": "Apr16",
        #     "root": Path("/share/pierson/matt/UAIR/multirun/2026-04-16_eval_all/15-14-32"),
        #     "metrics_subdir": "compute_metrics",
        # },
        {
            "label": "Apr20",
            "root": Path("/share/pierson/matt/UAIR/multirun/2026-04-20_eval_all/18-15-21"),
            "metrics_subdir": "compute_metrics",
        },
        {
            "label": "Apr24",
            "root": Path("/share/pierson/matt/UAIR/multirun/2026-04-24_eval_all/10-13-47"),
            "metrics_subdir": "compute_metrics",
        },
        {
            "label": "May27",
            "root": Path("/share/pierson/matt/UAIR/multirun/2026-03-30_eval_all/22-41-52"),
            "metrics_subdir": "compute_metrics_gpt52",
            # Hard-coded judge identity: the Phase C slurm-log scan can't find
            # this judge (it ran offline via OpenAI Batch, no judge_server.sub
            # logs). The actual model name returned in the batch responses
            # was `gpt-5.2-2025-12-11`; we use the family name here for
            # cross-row consistency with the other sweeps' short labels.
            "judge_override": {
                "url": "openai-batch://gpt-5.2",
                "judge_model": "gpt-5.2",
                "judge_slurm_log": "openai_batches/may27",
                "judge_launched": "2026-05-27T12:21:00",
            },
        },
        {
            "label": "May30",
            "root": Path("/share/pierson/matt/UAIR/multirun/2026-05-30_eval_all/21-35-36"),
            "metrics_subdir": "compute_metrics",
            # Full COLM results sweep: eval_all async flow with Qwen3.6-27B
            # judged as a sidecar (served on klara:8002). The async finalize
            # path issues no per-record judge POSTs from the eval process
            # (only a GET /v1/models probe), so the Phase C POST-timestamp +
            # slurm-log scan can't resolve the judge. The canonical name is in
            # each sub-run's leakage_judge_batch/manifest.json
            # (model=/share/pierson/matt/zoo/models/Qwen3.6-27B); hard-code it
            # here for cross-row consistency, matching Apr24's resolved label.
            "judge_override": {
                "url": "http://klara.tech.cornell.edu:8002",
                "judge_model": "Qwen3.6-27B",
                "judge_slurm_log": "sidecar (async)",
                "judge_launched": "2026-05-30T21:35:00",
            },
        },
        # May 12 sweeps EXCLUDED: the action-inference prompt was rewritten
        # between Apr 24 and May 12 (commit 44484d4 + a follow-up) to be
        # byte-identical to upstream SALT-NLP/PrivacyLens. The new prompt
        # invokes ReAct (`Thought:` → `Action:`), doubling mean action length
        # (587 → 1188 chars) on identical weights and sampling. That breaks
        # the "same task output, different judge" precondition: Phase G shows
        # action exact-match collapses to 0 for every Apr*↔May12 pair, so
        # judge-attribution becomes contaminated by task-output drift.
        # Re-enable by uncommenting; the codepaths still handle them.
        # {"label": "May12a", "root": Path("/share/pierson/matt/UAIR/multirun/2026-05-12_eval_all/04-05-05")},
        # {"label": "May12b", "root": Path("/share/pierson/matt/UAIR/multirun/2026-05-12_eval_all/10-55-35")},
    ]
    JUDGE_SERVER_LOG_DIR = Path("/share/pierson/matt/UAIR/.slurm_jobs/judge-server")
    REPORT_DIR = (
        Path(__file__).resolve().parent
        / "tables"
        / "privacylens_judge_ablation_2026_05_27"
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    METRIC_COLS = [
        "qa_accuracy",
        "qa_S_accuracy",
        "qa_T_accuracy",
        "qa_V_accuracy",
        "leakage_rate",
        "helpfulness_mean_score",
        "helpfulness_rate",
        "adjusted_leakage_rate",
    ]
    return JUDGE_SERVER_LOG_DIR, METRIC_COLS, REPORT_DIR, SWEEPS


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase A — Inventory
    """)
    return


@app.cell
def _(SWEEPS):
    import pandas as pd
    from omegaconf import OmegaConf

    _rows = []
    for _sweep in SWEEPS:
        _root = _sweep["root"]
        _metrics_subdir = _sweep.get("metrics_subdir", "compute_metrics")
        for _idx in sorted(p for p in _root.iterdir() if p.is_dir() and p.name.isdigit()):
            _cfg_p = _idx / "privacylens" / ".hydra" / "config.yaml"
            _ov_p = _idx / "privacylens" / ".hydra" / "overrides.yaml"
            _metrics_p = (
                _idx / "privacylens" / "privacylens_eval" / "outputs"
                / _metrics_subdir / "metrics.parquet"
            )
            _aai_p = (
                _idx / "privacylens" / "privacylens_eval" / "outputs"
                / "agent_action_inference" / "results.parquet"
            )
            _qa_p = (
                _idx / "privacylens" / "privacylens_eval" / "outputs"
                / "qa_probe_inference" / "results.parquet"
            )
            if not _cfg_p.exists():
                continue
            _cfg = OmegaConf.load(_cfg_p)
            _ov = OmegaConf.load(_ov_p) if _ov_p.exists() else []
            _label = next(
                (str(o).split("=", 1)[1] for o in _ov if str(o).startswith("model=")),
                "?",
            )
            _rows.append({
                "sweep": _sweep["label"],
                "idx": int(_idx.name),
                "model_label": _label,
                "model_source": str(OmegaConf.select(_cfg, "model.model_source") or ""),
                "lora_path": str(OmegaConf.select(_cfg, "model.lora_path") or ""),
                "max_model_len": OmegaConf.select(_cfg, "model.engine_kwargs.max_model_len"),
                "metrics_path": str(_metrics_p) if _metrics_p.exists() else "",
                "aai_path": str(_aai_p) if _aai_p.exists() else "",
                "qa_path": str(_qa_p) if _qa_p.exists() else "",
                "run_dir": str(_idx),
            })
    inv = pd.DataFrame(_rows)
    print(f"{len(inv)} sub-runs total")
    inv[["sweep", "idx", "model_label", "model_source", "lora_path", "metrics_path"]]
    return (inv,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase B — Group task-equivalent runs across sweeps
    """)
    return


@app.cell
def _(inv):
    import re as _re
    inv2 = inv.assign(
        pair_key=inv["model_source"].fillna("") + " || " + inv["lora_path"].fillna("")
    )
    # Keep EVERY row with metrics on disk. Previously this filtered to groups
    # spanning ≥2 sweeps, but that dropped the 11 May30-only models (gpt-oss,
    # openthinker, phi-4, several sft-ci variants) — half the full COLM
    # line-up. Per the "include all results / lump all runs together" decision
    # we retain them all; the cross-judge phases (E/F) restrict to
    # multi-coverage models themselves where the comparison requires it.
    _has = inv2[inv2["metrics_path"].astype(bool)]
    _g = _has.groupby("pair_key").agg(
        n_sweeps=("sweep", "nunique"),
        n_rows=("sweep", "size"),
    )
    # Deduplicate to one row per (sweep, pair_key). E.g. Apr16 has two sub-runs
    # with the same model_source loaded under different model_label aliases
    # ("instruct" + "no-finetuning"); collapse to first observation.
    paired = (
        _has.drop_duplicates(subset=["sweep", "pair_key"], keep="first")
        .copy()
    )
    paired["n_sweeps"] = paired["pair_key"].map(_g["n_sweeps"])
    paired["model_short"] = paired["model_source"].map(lambda s: s.rsplit("/", 1)[-1])
    # Disambiguate weights+LoRA combos that share model_source: e.g.
    # Qwen3.5-9B (no LoRA) vs Qwen3.5-9B + sft-ci LoRA → "Qwen3.5-9B+sft".
    # The stage tag alone ("sft"/"grpo") is NOT unique: Mar30 ran three GRPO
    # hyperparameter variants whose paths all yield tag "grpo", and different
    # SFT checkpoints share tag "sft". Append the training-run id (the HH-MM-SS
    # dir) so each distinct checkpoint gets its own label — e.g.
    # "_merged_sft+grpo@23-25-20" (the canonical v3-vr05-lambda05 model) vs
    # "@23-20-49"/"@23-13-02". Same run id across sweeps still lumps together.
    def _display_name(row):
        if not row["lora_path"]:
            return row["model_short"]
        _lora = row["lora_path"]
        tag = _lora.rsplit("/", 2)[-2] if "/" in _lora else "lora"
        _m = _re.search(r"/(\d{2}-\d{2}-\d{2})/", _lora)
        if _m and tag in ("grpo", "sft"):
            return f"{row['model_short']}+{tag}@{_m.group(1)}"
        return f"{row['model_short']}+{tag}"
    paired["display_name"] = paired.apply(_display_name, axis=1)
    _multi = (_g["n_sweeps"] >= 2).sum()
    print(
        f"{paired['pair_key'].nunique()} task-equivalent groups total "
        f"({_multi} present in ≥2 sweeps; the rest are single-sweep, mostly "
        f"the May30-only COLM models)"
    )
    (
        paired.pivot_table(
            index=["model_short", "lora_path"],
            columns="sweep",
            values="idx",
            aggfunc="first",
        )
        .reset_index()
    )
    return (paired,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase C — Resolve judge identity per sweep
    """)
    return


@app.cell
def _(JUDGE_SERVER_LOG_DIR, SWEEPS):
    import re as _re
    import datetime as _dt

    def _extract_judge_url(run_dir):
        # Old format: multirun.yaml setup commands had `export JUDGE_SERVER_URL=...`
        for d in (run_dir, *run_dir.parents):
            mr = d / "multirun.yaml"
            if mr.exists():
                m = _re.search(r"JUDGE_SERVER_URL=([^\s'\"]+)", mr.read_text())
                if m:
                    return m.group(1)
                break
        # Newer format: the env var was resolved at runtime; recover by
        # parsing the first GET/POST to klara from the privacylens_eval.log.
        log = run_dir / "privacylens" / "privacylens_eval.log"
        if log.exists():
            with open(log) as f:
                for line in f:
                    if "klara" in line and "HTTP Request" in line:
                        m = _re.search(r"(http://[^\s/]+(?::\d+)?)", line)
                        if m:
                            return m.group(1)
        return "<not set>"

    def _first_post_ts(run_dir):
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

    def _parse_launch(head):
        m = _re.search(
            r"^\[((?:Mon|Tue|Wed|Thu|Fri|Sat|Sun)[^\]]+)\] Launching judge server",
            head,
            _re.MULTILINE,
        )
        if not m:
            return None
        parts = " ".join(m.group(1).split()).split()
        try:
            if len(parts) >= 7 and parts[4] in ("AM", "PM"):
                return _dt.datetime.strptime(
                    " ".join(parts[:5] + parts[6:]), "%a %b %d %I:%M:%S %p %Y"
                )
            if len(parts) >= 6:
                return _dt.datetime.strptime(
                    " ".join(parts[:4] + parts[5:]), "%a %b %d %H:%M:%S %Y"
                )
        except ValueError:
            return None
        return None

    def _resolve_judge(url, run_dir):
        if not url or "<" in url:
            return ("?", "?", None)
        m = _re.search(r":(\d+)$", url.rstrip("/"))
        port = m.group(1) if m else None
        if port is None or not JUDGE_SERVER_LOG_DIR.exists():
            return ("?", "?", None)
        ref = _first_post_ts(run_dir)
        best = (None, None, _dt.datetime.min, None)
        for log in JUDGE_SERVER_LOG_DIR.glob("*.out"):
            try:
                head = log.read_text(errors="ignore")[:4000]
            except OSError:
                continue
            mm = _re.search(r"^Model:\s*(\S+)", head, _re.MULTILINE)
            mp = _re.search(r"^Port:\s*(\d+)", head, _re.MULTILINE)
            ts = _parse_launch(head)
            if not (mm and mp and ts and mp.group(1) == port):
                continue
            if ts <= ref and ts > best[2]:
                best = (log, mm.group(1), ts, port)
        if best[1]:
            return (best[1].rsplit("/", 1)[-1], best[0].name, best[2])
        return ("?", "?", None)

    judge_info = {}
    for _sweep in SWEEPS:
        # An offline OpenAI Batch sweep won't have a `JUDGE_SERVER_URL` env var
        # or matching slurm-log entry; honour the explicit override instead of
        # falling through to "?".
        _override = _sweep.get("judge_override")
        if _override:
            judge_info[_sweep["label"]] = dict(_override)
            print(
                f"{_sweep['label']:8s}  {_override['url']}  →  "
                f"{_override['judge_model']}  (override)"
            )
            continue
        _rep = next(
            (p for p in _sweep["root"].iterdir() if p.is_dir() and p.name.isdigit()),
            None,
        )
        _url = _extract_judge_url(_rep) if _rep else "<missing>"
        _model, _log, _launched = _resolve_judge(_url, _rep) if _rep else ("?", "?", None)
        judge_info[_sweep["label"]] = {
            "url": _url,
            "judge_model": _model,
            "judge_slurm_log": _log,
            "judge_launched": _launched.isoformat() if _launched else "?",
        }
        print(f"{_sweep['label']:8s}  {_url}  →  {_model}  (slurm {_log})")
    return (judge_info,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase D — Long-form metrics across (task model × judge)
    """)
    return


@app.cell
def _(METRIC_COLS, judge_info, paired):
    import pandas as _pd

    # The compute_metrics schema gained `*_among_parseable` and
    # `*_overall_with_default_zero` variants between Mar 30 and May 12. Map
    # each canonical name to a fallback chain so older parquets still work.
    _METRIC_ALIASES = {
        "leakage_rate": [
            "leakage_rate_overall_with_default_zero",
            "leakage_rate_among_parseable",
            "leakage_rate",
        ],
        "helpfulness_mean_score": [
            "helpfulness_mean_score_overall_with_default_zero",
            "helpfulness_mean_score_among_parseable",
            "helpfulness_mean_score",
        ],
        "helpfulness_rate": [
            "helpfulness_rate_overall_with_default_zero",
            "helpfulness_rate_among_parseable",
            "helpfulness_rate",
        ],
    }

    def _resolve(row, canonical):
        for c in _METRIC_ALIASES.get(canonical, [canonical]):
            if c in row.index:
                return float(row[c])
        return float("nan")

    _rows = []
    for _, _r in paired.iterrows():
        _m = _pd.read_parquet(_r["metrics_path"]).iloc[0]
        _entry = {
            "model_source": _r["model_source"],
            "model_short": _r["model_short"],
            "display_name": _r["display_name"],
            "lora_path": _r["lora_path"] or "—",
            "sweep": _r["sweep"],
            "judge": judge_info[_r["sweep"]]["judge_model"],
            "model_label": _r["model_label"],
        }
        for _c in METRIC_COLS:
            _entry[_c] = _resolve(_m, _c)
        _rows.append(_entry)
    long_df = _pd.DataFrame(_rows)
    print(f"long-form table: {len(long_df)} rows = "
          f"{long_df['model_source'].nunique()} task models × "
          f"≤{long_df['sweep'].nunique()} sweeps")
    long_df
    return (long_df,)


@app.cell
def _(METRIC_COLS, long_df):
    # Wide pivot: one row per (model, metric), columns = sweep (judge).
    import pandas as _pd

    _melt = long_df.melt(
        id_vars=["model_short", "sweep", "judge"],
        value_vars=METRIC_COLS,
        var_name="metric",
        value_name="value",
    )
    _melt["sweep_judge"] = _melt["sweep"] + " (" + _melt["judge"].str.split("/").str[-1] + ")"
    wide = (
        _melt.pivot_table(
            index=["model_short", "metric"],
            columns="sweep_judge",
            values="value",
            aggfunc="first",
        )
        .reset_index()
    )
    wide
    return (wide,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot D1 — Per-metric grouped bars: each task model under each judge

    One panel per headline metric. Bars grouped by task model, coloured
    by judge. If bars at the same model differ visibly across colours,
    that's judge variance.
    """)
    return


@app.cell
def _(REPORT_DIR, judge_info, long_df):
    import matplotlib.pyplot as _plt
    import numpy as _np

    _focus = [
        ("leakage_rate", "Leakage rate"),
        ("adjusted_leakage_rate", "Adjusted leakage rate"),
        ("helpfulness_mean_score", "Helpfulness mean (0-3)"),
        ("helpfulness_rate", "Helpful rate"),
    ]
    _models = sorted(long_df["display_name"].unique())
    _sweeps = sorted(long_df["sweep"].unique())
    _colors = {
        "Mar30": "#cc6677",
        "Apr16": "#ee99aa",
        "Apr20": "#4477aa",
        "Apr24": "#228833",
        "May27": "#ddaa33",
        "May30": "#117733",  # darker green — same Qwen3.6-27B judge as Apr24
        "May12a": "#aa3377",
        "May12b": "#66ccee",
    }

    _fig, _axes = _plt.subplots(2, 2, figsize=(17, 9), constrained_layout=True)
    _w = 0.8 / max(len(_sweeps), 1)
    for _ax, (_metric, _title) in zip(_axes.flat, _focus):
        for _i, _sweep in enumerate(_sweeps):
            # groupby().mean() (not set_index) so any display_name with >1 row
            # in a sweep collapses to a scalar instead of a Series.
            _sub = long_df[long_df["sweep"] == _sweep].groupby("display_name")[_metric].mean()
            _vals = [float(_sub.get(_m, _np.nan)) for _m in _models]
            _x = _np.arange(len(_models)) + (_i - (len(_sweeps) - 1) / 2) * _w
            _judge_label = judge_info[_sweep]["judge_model"].rsplit("/", 1)[-1]
            _ax.bar(_x, _vals, width=_w * 0.95,
                    color=_colors.get(_sweep, "#888"),
                    label=f"{_sweep} ({_judge_label})")
        _ax.set_title(_title)
        _ax.set_xticks(_np.arange(len(_models)))
        _ax.set_xticklabels(_models, rotation=30, ha="right", fontsize=8)
        _ax.grid(axis="y", alpha=0.3)
    _axes[0, 0].legend(loc="upper left", fontsize=8, framealpha=0.9)
    _fig.suptitle("PrivacyLens metric by task model, grouped by judge", fontsize=12)
    _fig.savefig(REPORT_DIR / "plot_D1_metric_by_model_judge.png", dpi=140)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase E — Pairwise judge-attributable shift at fixed task model

    For each pair of sweeps (judges), at each shared task model, compute
    `metric_B - metric_A`. Then aggregate across task models per pair.
    """)
    return


@app.cell
def _(METRIC_COLS, SWEEPS, judge_info, long_df):
    from itertools import combinations as _combos
    import statistics as _st
    import pandas as _pd

    _rows = []
    for _a, _b in _combos([s["label"] for s in SWEEPS], 2):
        _aj = judge_info[_a]["judge_model"].rsplit("/", 1)[-1]
        _bj = judge_info[_b]["judge_model"].rsplit("/", 1)[-1]
        _shared = (
            long_df[long_df["sweep"].isin([_a, _b])]
            .groupby("model_source")["sweep"]
            .nunique()
        )
        _models = _shared[_shared == 2].index.tolist()
        if not _models:
            continue
        for _metric in METRIC_COLS:
            _gaps = []
            for _ms in _models:
                _va = float(
                    long_df[(long_df["sweep"] == _a) & (long_df["model_source"] == _ms)][_metric].iloc[0]
                )
                _vb = float(
                    long_df[(long_df["sweep"] == _b) & (long_df["model_source"] == _ms)][_metric].iloc[0]
                )
                _gaps.append(_vb - _va)
            _rows.append({
                "judge_pair": f"{_aj} → {_bj}",
                "sweep_pair": f"{_a} → {_b}",
                "metric": _metric,
                "n_models": len(_gaps),
                "mean Δ (B−A)": _st.fmean(_gaps),
                "std Δ": _st.pstdev(_gaps) if len(_gaps) > 1 else 0.0,
                "min Δ": min(_gaps),
                "max Δ": max(_gaps),
            })
    pair_shifts = _pd.DataFrame(_rows)
    pair_shifts
    return (pair_shifts,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot E1 — Forest plot of judge-pair shifts

    For each metric and each judge pair, point = mean Δ (B−A) across
    shared task models, horizontal bar = ±1 std across task models. A
    large mean far from 0 = the judge swap moves the metric consistently.
    A large std = the swap affects different task models differently.
    """)
    return


@app.cell
def _(REPORT_DIR, pair_shifts):
    import matplotlib.pyplot as _plt
    import numpy as _np

    _focus_metrics = [
        "leakage_rate",
        "adjusted_leakage_rate",
        "helpfulness_mean_score",
        "helpfulness_rate",
        "qa_accuracy",
    ]
    _ps = pair_shifts[pair_shifts["metric"].isin(_focus_metrics)].copy()
    # Key rows by sweep_pair, not judge_pair: with two Qwen3.6-27B sweeps
    # (Apr24, May30) several distinct sweep pairs collapse to the same
    # judge_pair label (e.g. Mar30→Apr24 and Mar30→May30 are both
    # "Qwen3-32B-AWQ → Qwen3.6-27B"); keying on judge_pair would silently drop
    # one via .iloc[0]. Order by sweep_pair for a stable y-axis; annotate the
    # judge swap in the tick label.
    _pairs = sorted(_ps["sweep_pair"].unique())
    _label_for = {
        _sp: f"{_sp}\n({_ps[_ps['sweep_pair'] == _sp]['judge_pair'].iloc[0]})"
        for _sp in _pairs
    }
    _colors = ["#cc6677", "#4477aa", "#228833", "#ddaa33", "#117733",
               "#aa3377", "#66ccee", "#882255", "#999933", "#332288"]

    _fig, _axes = _plt.subplots(1, len(_focus_metrics), figsize=(17, 4.6), sharey=True)
    for _ax, _m in zip(_axes, _focus_metrics):
        _sub = _ps[_ps["metric"] == _m]
        _y = _np.arange(len(_pairs))
        for _i, _sp in enumerate(_pairs):
            _row = _sub[_sub["sweep_pair"] == _sp]
            if _row.empty:
                continue
            _r = _row.iloc[0]
            _ax.errorbar(
                _r["mean Δ (B−A)"], _i, xerr=_r["std Δ"],
                fmt="o", color=_colors[_i % len(_colors)],
                ecolor=_colors[_i % len(_colors)], capsize=4, markersize=8,
            )
        _ax.axvline(0, color="grey", linestyle="--", linewidth=1, alpha=0.7)
        _ax.set_yticks(_y)
        _ax.set_yticklabels([_label_for[_sp] for _sp in _pairs], fontsize=7)
        _ax.set_title(_m, fontsize=10)
        _ax.grid(axis="x", alpha=0.3)
    _axes[0].set_ylabel("sweep pair (A → B)")
    _fig.suptitle("Judge swap: mean Δ (B−A) ± σ across task models", fontsize=12)
    _fig.tight_layout()
    _fig.savefig(REPORT_DIR / "plot_E1_pairwise_shifts.png", dpi=140)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot E2 — Judge variance per Qwen3.5-9B finetune (paper figure)

    Stratifies by the **task model** (Qwen3.5-9B-Instruct base vs the same
    weights with an SFT-CI LoRA vs with the GRPO LoRA) and shows the spread
    of each PrivacyLens metric **across the judges** that scored the same
    actions. Within-box spread is judge-attributable variance.

    Strata are now selected by the precise `(model_source, lora_path)` key,
    not the lossy `display_name`. This matters because Mar30 ran **three**
    GRPO hyperparameter variants whose paths all collapse to `_merged_sft+grpo`;
    the grpo box is restricted to the canonical paper model
    **v3-vr05-lambda05** (training run `23-25-20`) so the box is one task
    model across judges, not several models.

    Coverage notes (n annotated below each x-tick):

    - **Qwen3.5-9B base** (Instruct, no LoRA): 5 obs / **4 distinct judges** —
      Mar30 Qwen3-32B-AWQ, Apr20 Gemma-4-31B-it, Apr24 Qwen3.6-27B,
      May27 gpt-5.2, May30 Qwen3.6-27B (Apr24 & May30 share the judge).
    - **+SFT-CI LoRA**: 3 obs / 3 judges — Mar30 Qwen3-32B-AWQ, May27 gpt-5.2,
      May30 Qwen3.6-27B. (Mar30/May27 use the `19-29-34` SFT checkpoint;
      May30 uses the newer `pair_ablation` SFT-CI checkpoint — different
      weights, lumped here as "+SFT-CI" per the negligible-drift decision.)
    - **+GRPO LoRA** (v3-vr05-lambda05): 3 obs / 3 judges — Mar30 Qwen3-32B-AWQ,
      May27 gpt-5.2, May30 Qwen3.6-27B.

    **Caveat (prompt-era drift):** the May30 points use the post-ReAct action
    prompt while the other sweeps use the pre-ReAct prompt (May27 reuses Mar30
    actions). Per this revision's decision we treat that contribution as
    negligible relative to the judge effect, so each May30 marker carries a
    small residual prompt-drift component on top of the judge swap.
    """)
    return


@app.cell
def _(REPORT_DIR, long_df):
    import matplotlib.pyplot as _plt
    import matplotlib.patches as _mpatches
    import numpy as _np

    # Stratify by Qwen3.5-9B finetune using the precise (model_source,
    # lora_path) key, NOT display_name. The training pipeline merges the SFT
    # LoRA into the base weights before GRPO, so the GRPO row's model_source
    # ends in `_merged_sft`. display_name is lossy: Mar30 ran three GRPO
    # hyperparameter variants whose lora paths all collapse to
    # "_merged_sft+grpo", which would pollute the grpo box with three distinct
    # task models. Match the canonical paper GRPO model (v3-vr05-lambda05 =
    # training run 23-25-20) explicitly so each box is one task model.
    def _stratum(_row):
        _src = (_row["model_source"] or "")
        _lora = (_row["lora_path"] or "")
        if _lora == "—":
            _lora = ""
        if _src.endswith("Qwen3.5-9B") and not _lora:
            return "base"
        if _src.endswith("Qwen3.5-9B") and "sft_only" in _lora:
            return "sft"
        if "_merged_sft" in _src and "23-25-20" in _lora:
            return "grpo"
        return None

    _df = long_df.copy()
    _df["stratum"] = _df.apply(_stratum, axis=1)
    _df = _df[_df["stratum"].notna()].copy()

    # Consistent judge color palette; reused across panels.
    _JUDGE_COLORS = {
        "Qwen3-32B-AWQ":  "#cc6677",   # red
        "Gemma-4-31B-it": "#4477aa",   # blue
        "Qwen3.6-27B":    "#228833",   # green
        "gpt-5.2":        "#ddaa33",   # gold
    }
    _STRATA = ["base", "sft", "grpo"]
    _STRATUM_LABELS = {
        "base": "base\n(Instruct)",
        "sft":  "+SFT-CI",
        "grpo": "+GRPO",
    }

    # Title + units folded into one line so we don't fight the subplot
    # title's y-position. (boxplot() in matplotlib also re-asserts xticks,
    # so we set tick labels AFTER all plotting calls below.)
    _METRICS = [
        ("leakage_rate",           "Leakage rate (↓ better)"),
        ("adjusted_leakage_rate",  "Adjusted leakage rate"),
        ("helpfulness_mean_score", "Helpfulness mean (0–3)"),
        ("helpfulness_rate",       "Helpful rate (score ≥ 2)"),
    ]

    _plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "xtick.labelsize": 9,
        "ytick.labelsize": 8,
        "legend.fontsize": 8.5,
    })

    _fig, _axes = _plt.subplots(1, len(_METRICS), figsize=(13.5, 3.6))
    _xs = _np.arange(len(_STRATA))
    _rng = _np.random.default_rng(0)

    for _ax, (_metric, _title) in zip(_axes, _METRICS):
        _ax.set_title(_title, pad=8, fontweight="semibold")
        _ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        _ax.set_axisbelow(True)

        for _i, _stratum_name in enumerate(_STRATA):
            _rows = _df[_df["stratum"] == _stratum_name]
            _values = _rows[_metric].astype(float).tolist()
            _judges = _rows["judge"].tolist()
            _n = len(_values)

            if _n == 0:
                _ax.annotate("no data", (_i, 0), ha="center", va="center",
                             fontsize=8, color="#999")
                continue

            # Box-plot semantics depend on N:
            #   N ≥ 3 → real box (Q1/median/Q3 with min/max whiskers)
            #   N = 2 → vertical span from min to max with a median tick
            #   N = 1 → no spread element, just the marker
            if _n >= 3:
                _bp = _ax.boxplot(
                    [_values], positions=[_i], widths=0.45,
                    patch_artist=True, showfliers=False, zorder=2,
                )
                for _box in _bp["boxes"]:
                    _box.set(facecolor="#f0f0f0", edgecolor="#444", linewidth=1.0)
                for _whisker in _bp["whiskers"]:
                    _whisker.set(color="#444", linewidth=1.0)
                for _cap in _bp["caps"]:
                    _cap.set(color="#444", linewidth=1.0)
                for _median in _bp["medians"]:
                    _median.set(color="#222", linewidth=1.4)
            elif _n == 2:
                _ax.vlines(_i, min(_values), max(_values),
                           color="#888", linewidth=1.2, zorder=2)
                _mid = sum(_values) / 2
                _ax.hlines(_mid, _i - 0.18, _i + 0.18,
                           color="#444", linewidth=1.0, linestyles=":", zorder=2)

            # Always overlay the per-judge markers, lightly jittered if any
            # two values coincide so they don't fully overlap.
            _jitter = _rng.uniform(-0.08, 0.08, size=_n)
            for _v, _judge, _dx in zip(_values, _judges, _jitter):
                _ax.scatter(
                    _i + _dx, _v,
                    s=58,
                    color=_JUDGE_COLORS.get(_judge, "#888"),
                    edgecolor="white", linewidth=0.8,
                    zorder=3,
                )

        _ax.set_xlim(-0.6, len(_STRATA) - 0.4)
        # Subtle headroom so jittered markers near the top don't clip.
        _ymin, _ymax = _ax.get_ylim()
        _pad = max((_ymax - _ymin) * 0.06, 0.005)
        _ax.set_ylim(_ymin - _pad, _ymax + _pad)

        # boxplot() resets x-ticks, so apply tick labels AFTER all
        # plotting calls. Two-line labels: stratum name, then n=N
        # (computed per-metric so missing-data strata show n=0).
        _tick_labels = []
        for _s in _STRATA:
            _n_s = int(_df[_df["stratum"] == _s][_metric].notna().sum())
            _tick_labels.append(f"{_STRATUM_LABELS[_s]}\nn={_n_s}")
        _ax.set_xticks(_xs)
        _ax.set_xticklabels(_tick_labels)

    # Figure-level legend (one entry per judge, ordered by chronology of use).
    _legend_handles = [
        _mpatches.Patch(facecolor=_JUDGE_COLORS[_j], edgecolor="white", label=_j)
        for _j in ["Qwen3-32B-AWQ", "Gemma-4-31B-it", "Qwen3.6-27B", "gpt-5.2"]
    ]
    _fig.legend(
        handles=_legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=4,
        frameon=False,
        title="Judge LLM",
        title_fontsize=9,
    )
    _fig.suptitle(
        "PrivacyLens judge variance on Qwen3.5-9B (same task model, different judges)",
        y=1.14, fontsize=12, fontweight="semibold",
    )
    _fig.text(
        0.5, -0.04,
        "Box = Q1/median/Q3 ± min/max (when n≥3); vertical span = min–max (when n=2). "
        "Markers = per-judge metric value on the same task weights "
        "(May30 uses the post-ReAct action prompt; drift treated as negligible).",
        ha="center", fontsize=8, color="#555",
    )
    _fig.tight_layout(rect=(0, 0, 1, 1))
    _fig.savefig(
        REPORT_DIR / "plot_E2_judge_variance_qwen35_9b.png",
        dpi=200, bbox_inches="tight",
    )
    _fig.savefig(
        REPORT_DIR / "plot_E2_judge_variance_qwen35_9b.pdf",
        bbox_inches="tight",
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot E2 — same data as tables (latest run per judge)

    The Plot E2 box-plot values, transposed into four pastable markdown
    tables — one per headline metric, **rows = task model, columns = judge**.

    Deduped to the **latest run of each judge**: where a judge scored a given
    task model in more than one sweep (only the Qwen3.6-27B base row, Apr24 vs
    May30), the most recent sweep is kept — so the base stratum is **n=4
    distinct judges**, not 5. Cells with no run for that (model, judge) show
    "—". The rendered tables and the raw markdown (printed below, also saved to
    `e2_metric_tables.md`) are identical.
    """)
    return


@app.cell
def _(REPORT_DIR, long_df, mo):
    import pandas as _pd

    # Same stratum predicates as the Plot E2 cell (precise model_source/lora,
    # not the lossy display_name), so the GRPO row is the canonical paper model
    # v3-vr05-lambda05 (run 23-25-20) and not a mix of Mar30's GRPO variants.
    def _stratum(_row):
        _src = (_row["model_source"] or "")
        _lora = (_row["lora_path"] or "")
        if _lora == "—":
            _lora = ""
        if _src.endswith("Qwen3.5-9B") and not _lora:
            return "base"
        if _src.endswith("Qwen3.5-9B") and "sft_only" in _lora:
            return "sft"
        if "_merged_sft" in _src and "23-25-20" in _lora:
            return "grpo"
        return None

    _e2 = long_df.copy()
    _e2["stratum"] = _e2.apply(_stratum, axis=1)
    _e2 = _e2[_e2["stratum"].notna()].copy()

    # Keep only the latest run per (stratum, judge). Recency = sweep launch
    # date; sorting ascending and keeping "last" retains the newest sweep.
    _RECENCY = {
        "Mar30": "2026-03-30", "Apr20": "2026-04-20", "Apr24": "2026-04-24",
        "May27": "2026-05-27", "May30": "2026-05-30",
    }
    _e2["_recency"] = _e2["sweep"].map(_RECENCY)
    _e2 = (
        _e2.sort_values("_recency")
        .drop_duplicates(subset=["stratum", "judge"], keep="last")
    )

    _STRATA = ["base", "sft", "grpo"]
    _STRATUM_LABEL = {
        "base": "base (Instruct)",
        "sft": "+SFT-CI",
        "grpo": "+GRPO (v3-vr05-λ05)",
    }
    # Column order follows chronology of first use; only judges actually
    # present are shown.
    _JUDGE_ORDER = ["Qwen3-32B-AWQ", "Gemma-4-31B-it", "Qwen3.6-27B", "gpt-5.2"]
    _judges = [_j for _j in _JUDGE_ORDER if _j in set(_e2["judge"])]
    _METRICS = [
        ("leakage_rate", "Leakage rate (↓ better)"),
        ("adjusted_leakage_rate", "Adjusted leakage rate"),
        ("helpfulness_mean_score", "Helpfulness mean (0–3)"),
        ("helpfulness_rate", "Helpful rate (score ≥ 2)"),
    ]

    def _table_md(_metric, _title):
        _piv = _e2.pivot_table(
            index="stratum", columns="judge", values=_metric, aggfunc="first"
        )
        _lines = [f"#### {_title}", ""]
        _lines.append("| Task model | " + " | ".join(_judges) + " |")
        _lines.append("|---|" + "---|" * len(_judges))
        for _s in _STRATA:
            if _s not in _piv.index:
                continue
            _cells = []
            for _j in _judges:
                _v = _piv.loc[_s, _j] if _j in _piv.columns else float("nan")
                _cells.append("—" if _pd.isna(_v) else f"{_v:.3f}")
            _lines.append(f"| {_STRATUM_LABEL[_s]} | " + " | ".join(_cells) + " |")
        return "\n".join(_lines)

    _md = "\n\n".join(_table_md(_m, _t) for _m, _t in _METRICS)
    (REPORT_DIR / "e2_metric_tables.md").write_text(_md + "\n")
    print(_md)  # raw markdown, copy-pastable from the cell output
    mo.md(_md)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase F — Judge variance vs model variance

    Per metric:
    - **cross-model σ per judge**: spread across paired task models when
      you fix the judge
    - **cross-judge σ per task model**: spread across judges at a fixed
      task model, averaged across task models

    If `cross-judge σ ≳ cross-model σ`, the leaderboard ranking is fragile
    to judge choice.
    """)
    return


@app.cell
def _(METRIC_COLS, long_df):
    import statistics as _st
    import pandas as _pd

    # cross-model σ per (judge, metric)
    _cm = (
        long_df.groupby(["sweep", "judge"])[METRIC_COLS]
        .agg(lambda s: _st.pstdev(s) if len(s) > 1 else 0.0)
        .reset_index()
        .melt(id_vars=["sweep", "judge"], var_name="metric", value_name="cross_model_sigma")
    )

    # cross-judge σ per (model, metric). Two corrections now that every run is
    # lumped in:
    #   1. Collapse to one value per DISTINCT judge first, so the two
    #      Qwen3.6-27B sweeps (Apr24 & May30) count once — their gap is
    #      prompt-drift, not a judge swap, and shouldn't inflate judge σ.
    #   2. Restrict to models seen under ≥2 distinct judges, so the 11
    #      single-judge May30-only COLM models don't contribute σ=0 and
    #      dilute the mean toward zero (which would understate judge variance).
    _per_judge = (
        long_df.groupby(["model_source", "model_short", "judge"])[METRIC_COLS]
        .mean()
        .reset_index()
    )
    _n_judges = _per_judge.groupby("model_source")["judge"].transform("nunique")
    _multi_judge = _per_judge[_n_judges >= 2]
    _cj = (
        _multi_judge.groupby(["model_source", "model_short"])[METRIC_COLS]
        .agg(lambda s: _st.pstdev(s) if len(s) > 1 else 0.0)
        .reset_index()
        .melt(
            id_vars=["model_source", "model_short"],
            var_name="metric",
            value_name="cross_judge_sigma",
        )
    )
    cj_mean = _cj.groupby("metric")["cross_judge_sigma"].mean().reset_index()
    print(
        f"cross-judge σ averaged over "
        f"{_multi_judge['model_source'].nunique()} models with ≥2 distinct judges"
    )
    cm_mean = _cm.groupby("metric")["cross_model_sigma"].mean().reset_index()

    variance_compare = cm_mean.merge(cj_mean, on="metric")
    variance_compare["ratio (judge_σ / model_σ)"] = (
        variance_compare["cross_judge_sigma"] / variance_compare["cross_model_sigma"]
    )
    print("All values are population σ. Higher 'ratio' = ranking more fragile to judge choice.")
    variance_compare
    return (variance_compare,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot F1 — Cross-judge σ vs cross-model σ per metric

    Per metric, two paired bars: the average spread *across task models
    at a fixed judge* (red) and the average spread *across judges at a
    fixed task model* (blue). If blue ≈ red, you can't separate the two
    sources of variance from a leaderboard alone.
    """)
    return


@app.cell
def _(REPORT_DIR, variance_compare):
    import matplotlib.pyplot as _plt
    import numpy as _np

    _vc = variance_compare.copy().sort_values("cross_model_sigma", ascending=False)
    _metrics = _vc["metric"].tolist()
    _x = _np.arange(len(_metrics))
    _w = 0.36

    _fig, _ax = _plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    _ax.bar(_x - _w / 2, _vc["cross_model_sigma"], _w,
            label="cross-model σ (fix judge, vary task)", color="#cc6677")
    _ax.bar(_x + _w / 2, _vc["cross_judge_sigma"], _w,
            label="cross-judge σ (fix task, vary judge)", color="#4477aa")
    for _i, (_cm, _cj) in enumerate(zip(_vc["cross_model_sigma"], _vc["cross_judge_sigma"])):
        _ratio = _cj / _cm if _cm > 0 else float("inf")
        _ax.text(_i, max(_cm, _cj) * 1.03, f"{_ratio:.2f}",
                 ha="center", va="bottom", fontsize=8, color="#333")
    _ax.set_xticks(_x)
    _ax.set_xticklabels(_metrics, rotation=25, ha="right", fontsize=8)
    _ax.set_ylabel("σ (population)")
    _ax.set_title("Judge variance vs model variance per PrivacyLens metric "
                  "(label = judge_σ / model_σ)")
    _ax.legend(fontsize=9, framealpha=0.9)
    _ax.grid(axis="y", alpha=0.3)
    _fig.savefig(REPORT_DIR / "plot_F1_variance_compare.png", dpi=140)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot F2 — Leaderboard slopegraph: ranking under each judge

    Each line is one task model. y-axis is the rank that judge assigns it
    (1 = strictest / lowest leakage; ties broken by raw value). If lines
    cross, the leaderboard ordering changes when you swap judges.

    Coverage is ragged across the five sweeps, so we keep any task model
    present in **≥2 sweeps** (instead of requiring all five) and render
    missing sweeps as line breaks. Ranks are computed within each sweep over
    the models that sweep actually scored.
    """)
    return


@app.cell
def _(REPORT_DIR, judge_info, long_df):
    import matplotlib.pyplot as _plt
    import numpy as _np

    _metric = "leakage_rate"
    _SWEEP_ORDER = ["Mar30", "Apr20", "Apr24", "May27", "May30"]
    _df = long_df.pivot_table(
        index="display_name", columns="sweep", values=_metric, aggfunc="first"
    )
    _df = _df[[_c for _c in _SWEEP_ORDER if _c in _df.columns]]
    # Keep models present in ≥2 sweeps (ragged coverage allowed — five sweeps
    # rarely all cover the same model). Missing sweeps render as line breaks.
    _df = _df[_df.notna().sum(axis=1) >= 2]
    _ranks = _df.rank(method="min", ascending=True)  # lower leakage = better rank
    _sweeps = list(_df.columns)

    def _last_valid(_vals):
        for _j in range(len(_vals) - 1, -1, -1):
            if not _np.isnan(_vals[_j]):
                return _j, _vals[_j]
        return len(_vals) - 1, _vals[-1]

    _fig, _axes = _plt.subplots(1, 2, figsize=(13, 4.5))
    # Left: rank slopegraph
    _x = _np.arange(len(_sweeps))
    _cmap = _plt.cm.tab20
    for _i, (_m, _row) in enumerate(_ranks.iterrows()):
        _v = _row.values.astype(float)
        _axes[0].plot(_x, _v, marker="o", color=_cmap(_i % 20), label=_m, linewidth=2)
        _lj, _lv = _last_valid(_v)
        _axes[0].annotate(_m, (_x[_lj], _lv),
                          xytext=(5, 0), textcoords="offset points",
                          fontsize=7, va="center")
    _axes[0].set_xticks(_x)
    _axes[0].set_xticklabels(
        [f"{s}\n({judge_info[s]['judge_model'].rsplit('/', 1)[-1]})" for s in _sweeps],
        fontsize=8,
    )
    _axes[0].set_ylabel(f"rank by {_metric} (1=lowest)")
    _axes[0].invert_yaxis()
    _axes[0].set_title(f"Leaderboard ranking by {_metric}")
    _axes[0].grid(axis="y", alpha=0.3)

    # Right: raw value slopegraph (same models, same colours)
    for _i, (_m, _row) in enumerate(_df.iterrows()):
        _v = _row.values.astype(float)
        _axes[1].plot(_x, _v, marker="o", color=_cmap(_i % 20),
                      label=_m, linewidth=2)
        _lj, _lv = _last_valid(_v)
        _axes[1].annotate(_m, (_x[_lj], _lv),
                          xytext=(5, 0), textcoords="offset points",
                          fontsize=7, va="center")
    _axes[1].set_xticks(_x)
    _axes[1].set_xticklabels(
        [f"{s}\n({judge_info[s]['judge_model'].rsplit('/', 1)[-1]})" for s in _sweeps],
        fontsize=8,
    )
    _axes[1].set_ylabel(_metric)
    _axes[1].set_title(f"Raw {_metric} per (task model, judge)")
    _axes[1].grid(axis="y", alpha=0.3)
    _fig.suptitle("Same task models, five sweeps (four judges) — does ranking move?", fontsize=11)
    _fig.tight_layout()
    _fig.savefig(REPORT_DIR / "plot_F2_leaderboard_slopegraph.png", dpi=140)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Phase G — Task-output sanity per cross-sweep pair

    Phase E attributes Δ to the judge. That only holds if the underlying
    action outputs are nearly the same. For each pair of sweeps and each
    shared task model: action exact-match rate + QA-probe Yes/No κ
    (judge-free signal).
    """)
    return


@app.cell
def _(SWEEPS, paired):
    from itertools import combinations as _combos
    from sklearn.metrics import cohen_kappa_score
    import re as _re
    import pandas as _pd

    _YN = _re.compile(r"\b(yes|no)\b", _re.IGNORECASE)

    def _yn(s):
        m = _YN.search(s or "")
        return m.group(1).lower() if m else ""

    _rows = []
    _labels = [s["label"] for s in SWEEPS]
    for _a, _b in _combos(_labels, 2):
        _ga = paired[paired["sweep"] == _a].set_index("model_source")
        _gb = paired[paired["sweep"] == _b].set_index("model_source")
        for _ms in _ga.index.intersection(_gb.index):
            _ra = _ga.loc[_ms] if not isinstance(_ga.loc[_ms], _pd.DataFrame) else _ga.loc[_ms].iloc[0]
            _rb = _gb.loc[_ms] if not isinstance(_gb.loc[_ms], _pd.DataFrame) else _gb.loc[_ms].iloc[0]
            _exact = float("nan")
            _kappa = float("nan")
            _agree = float("nan")
            if _ra["aai_path"] and _rb["aai_path"]:
                try:
                    _aa = _pd.read_parquet(_ra["aai_path"])[["record_id", "generated_action"]]
                    _ab = _pd.read_parquet(_rb["aai_path"])[["record_id", "generated_action"]]
                    _mm = _aa.rename(columns={"generated_action": "a"}).merge(
                        _ab.rename(columns={"generated_action": "b"}),
                        on="record_id",
                        how="inner",
                    )
                    _mm["a"] = _mm["a"].fillna("").astype(str)
                    _mm["b"] = _mm["b"].fillna("").astype(str)
                    _exact = (_mm["a"] == _mm["b"]).mean()
                except Exception:
                    pass
            if _ra["qa_path"] and _rb["qa_path"]:
                try:
                    _qaa = _pd.read_parquet(_ra["qa_path"])[["record_id", "_qa_axis", "generated_text"]]
                    _qab = _pd.read_parquet(_rb["qa_path"])[["record_id", "_qa_axis", "generated_text"]]
                    _qa = _qaa.rename(columns={"generated_text": "a"}).merge(
                        _qab.rename(columns={"generated_text": "b"}),
                        on=["record_id", "_qa_axis"],
                        how="inner",
                    )
                    _qa["yn_a"] = _qa["a"].fillna("").map(_yn)
                    _qa["yn_b"] = _qa["b"].fillna("").map(_yn)
                    _vv = _qa[(_qa["yn_a"] != "") & (_qa["yn_b"] != "")]
                    if len(_vv):
                        _kappa = cohen_kappa_score(_vv["yn_a"], _vv["yn_b"])
                        _agree = (_vv["yn_a"] == _vv["yn_b"]).mean()
                except Exception:
                    pass
            _rows.append({
                "sweep_pair": f"{_a} → {_b}",
                "model_short": _ms.rsplit("/", 1)[-1],
                "action exact": _exact,
                "QA Yes/No agree": _agree,
                "QA κ": _kappa,
            })
    sanity = _pd.DataFrame(_rows).sort_values(["sweep_pair", "model_short"]).reset_index(drop=True)
    sanity
    return (sanity,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Plot G1 — Task-output sanity heatmap

    Per (sweep pair × task model): action exact match and QA-probe κ.
    Cells near 1.0 mean the two runs produced near-identical task outputs
    → any metric Δ in Phase E is judge-attributable. Cells far from 1.0
    mean task-model drift contaminates the judge-attribution.
    """)
    return


@app.cell
def _(REPORT_DIR, sanity):
    import matplotlib.pyplot as _plt
    import numpy as _np

    _mat_exact = sanity.pivot(index="sweep_pair", columns="model_short", values="action exact")
    _mat_kappa = sanity.pivot(index="sweep_pair", columns="model_short", values="QA κ")

    _fig, _axes = _plt.subplots(2, 1, figsize=(11, 5.5), constrained_layout=True)
    for _ax, _mat, _title, _vmin in zip(
        _axes,
        [_mat_exact, _mat_kappa],
        ["Action exact match (1.0 = byte-identical actions)",
         "QA-probe Yes/No Cohen's κ (judge-free task signal)"],
        [0.0, 0.0],
    ):
        _im = _ax.imshow(_mat.values, aspect="auto", vmin=_vmin, vmax=1.0, cmap="RdYlGn")
        _ax.set_xticks(_np.arange(_mat.shape[1]))
        _ax.set_xticklabels(_mat.columns, rotation=30, ha="right", fontsize=8)
        _ax.set_yticks(_np.arange(_mat.shape[0]))
        _ax.set_yticklabels(_mat.index, fontsize=9)
        _ax.set_title(_title, fontsize=10)
        for _i in range(_mat.shape[0]):
            for _j in range(_mat.shape[1]):
                _v = _mat.values[_i, _j]
                if _np.isnan(_v):
                    _txt = "—"
                    _color = "#666"
                else:
                    _txt = f"{_v:.2f}"
                    _color = "white" if _v < 0.5 else "black"
                _ax.text(_j, _i, _txt, ha="center", va="center",
                         fontsize=8, color=_color)
        _plt.colorbar(_im, ax=_ax, shrink=0.85)
    _fig.suptitle("Task-output sanity: precondition for judge attribution", fontsize=11)
    _fig.savefig(REPORT_DIR / "plot_G1_sanity_heatmap.png", dpi=140)
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase H — Save artifacts
    """)
    return


@app.cell
def _(
    REPORT_DIR,
    judge_info,
    long_df,
    pair_shifts,
    paired,
    sanity,
    variance_compare,
    wide,
):
    import json as _json

    long_df.to_parquet(REPORT_DIR / "metrics_long.parquet", index=False)
    wide.to_parquet(REPORT_DIR / "metrics_wide.parquet", index=False)
    pair_shifts.to_parquet(REPORT_DIR / "pairwise_shifts.parquet", index=False)
    variance_compare.to_parquet(REPORT_DIR / "judge_vs_model_variance.parquet", index=False)
    sanity.to_parquet(REPORT_DIR / "task_output_sanity.parquet", index=False)
    paired.to_parquet(REPORT_DIR / "paired_inventory.parquet", index=False)
    (REPORT_DIR / "judge_info.json").write_text(_json.dumps(judge_info, indent=2))
    print(f"artifacts written to: {REPORT_DIR}")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read the result

    - **Phase D / wide**: per-(model, metric) values under each judge.
    - **Phase E**: average judge-attributable shift per pair of judges,
      per metric, across shared task models — read these as
      "if I rejudge from A to B, leakage moves by X ± Y".
    - **Phase F**: the headline judge-vs-model variance comparison.
      Ratio ≳ 1 = leaderboard fragile.
    - **Phase G**: task-output sanity — the precondition that lets you
      attribute shift to the judge. If κ collapses for a pair, exclude
      it from the conclusion or investigate task-model drift first.

    Limitations & next steps:

    - **Coverage is uneven.** The Qwen3.5-9B base/SFT-CI/GRPO strata (Plot E2)
      now have 3–4 distinct judges each; the broader leaderboard (Phase D /
      Plot D1) lumps in the full May30 COLM line-up, but **11 of the 21
      May30 task models are single-judge** (gpt-oss, openthinker, phi-4, and
      several SFT-CI variants only ran under May30's Qwen3.6-27B). They appear
      in the results tables but contribute nothing to the cross-judge phases
      (E/F restrict to multi-judge models). Conclusions about the **shape** of
      judge variance are well-supported; conclusions about the **identity** of
      the "right" judge are not.
    - **Prompt-era drift is lumped in, not removed.** May30 uses the post-ReAct
      action prompt (actions ≈ 7× longer than the Mar30/Apr20/Apr24/May27 era),
      so every cross-judge Δ that mixes May30 with a pre-ReAct sweep carries a
      residual prompt-drift term. This revision treats that as negligible by
      design; Plot G1 still quantifies the action drift so it stays visible.
      The Apr24 → May30 pair is the cleanest same-judge (Qwen3.6-27B) probe of
      that drift. To isolate a pure judge swap on byte-identical post-ReAct
      actions, rejudge the May30 actions with another judge via
      `scripts/prepare_judge_batches.py`.
    - `metrics.parquet` aggregates can hide row-level disagreement. For a
      per-record judge-flip analysis (which actions flip from leak=No to
      leak=Yes when you change judges), point a follow-up notebook at
      `leakage_judge_batch/results.parquet` directly.
    """)
    return


if __name__ == "__main__":
    app.run()
