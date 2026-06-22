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
    # COLM benchmark-results table — Qwen3.6-judged (latest available)

    Regenerates the paper's main results table
    (`papers/colm26_normative-simulacra/tables/benchmark_results.tex`) from the
    **filesystem `eval_all` multiruns**, restricted to **Qwen3.6-27B-judged**
    runs and taking, **per cell**, the most recent run that has it. Adds an
    **MMLU** column so every benchmark the pipeline runs is visible.

    ### Design decisions (per the request)

    1. **Mirror `benchmark_results.tex` + add MMLU.** Same five benchmarks and
       metrics, plus an MMLU overall-accuracy column.
    2. **Judged (LLM-judged) metrics require Qwen3.6-27B; judge-free metrics
       take the latest available run regardless of judge.** Judge-free columns
       (GoldCoin F1, PrivacyLens QA Acc, ConfAIde Tier-2 *r*, VLM Q7, MMLU) are
       deterministic scores against gold labels / human ratings, so the judge is
       irrelevant — they use the most recent run that has them, even if that run
       used a different judge. Judged columns (PrivacyLens Lk / Adj-Lk / Helpful
       / Help, CIRL completeness) draw only from Qwen3.6-judged runs. Per cell we
       take the most recent qualifying run.
    3. **Exact paper rows.** Nine models × {Zero-shot, SFT, GRPO}; GRPO only for
       Qwen3.5-9B. "Zero-shot" = the instruct/it/ppo variant (the model config
       with no SFT/GRPO adapter).
    4. **Per-cell latest-run selection.** Each (model, condition, metric) cell
       independently takes the newest qualifying run, so one row may draw
       different columns from different multiruns. This maximises filled cells
       and recency.

    ### Where each metric lives (single source of truth below)

    | Paper col | Benchmark dir | metrics.json subdir | dotted key | judged? |
    |---|---|---|---|---|
    | Appl. / Comp. | `goldcoin/goldcoin_hipaa` | `compute_metrics_{applicability,compliance}` | `macro_f1` | no |
    | QA Acc | `privacylens/privacylens_eval` | `compute_metrics` | `qa_probing.accuracy` | no |
    | Lk ↓ (raw) | ″ | ″ | `leakage.leakage_rate_overall_with_default_zero` | **yes** |
    | Adj Lk ↓ | ″ | ″ | `adjusted_leakage.adjusted_leakage_rate` | **yes** |
    | Helpful | ″ | ″ | `helpfulness.helpful_rate_overall_with_default_zero` | **yes** |
    | Help | ″ | ″ | `helpfulness.mean_score_overall_with_default_zero` | **yes** |
    | r | `confaide/confaide` | `compute_metrics_tier2b` | `pearson_r` | no |
    | Comp. (CIRL) | `cirl_trajectory/cirl_vignettes` | `compute_trajectory_metrics` | `complete` | **yes** |
    | Q7 | `vlm_geoprivacy/vlm_geoprivacy_bench` | `compute_metrics` | `per_question.Q7.accuracy` | no |
    | MMLU | `mmlu/mmlu` | `compute_metrics` | `overall_accuracy` | no |

    **Caveats.**
    - **CIRL completeness is judged and only `cirl_trajectory` produced it.** The
      May `eval_all` runs switched CIRL to `cirl_vignettes` probing-accuracy
      (a different, judge-free metric). The latest Qwen3.6-judged
      `cirl_trajectory` run is **Apr24** (`2026-04-24_eval_all/10-13-47`), which
      only covers Qwen3.5-{2B,4B,9B}. Other models' CIRL-Comp cells are blank
      ("—") — there is no Qwen3.6-judged completeness for them.
    - **ConfAIde `r`** uses Tier-2b Pearson — this reproduces the paper's `r`
      column exactly (Tier-2a is a different, much higher number). Switch via
      `CONFAIDE_R_SUBDIR`.
    - **Help** is the mean helpfulness score on the native 0–3 scale (the paper
      caption says 0–4, but the values are 0–3); all other columns are
      percentages (×100).
    """)
    return


@app.cell
def _():
    from pathlib import Path
    import re
    import datetime as dt

    MULTIRUN_GLOB_ROOT = Path("/share/pierson/matt/UAIR/multirun")
    REPORT_DIR = (
        Path(__file__).resolve().parent
        / "tables"
        / "colm_benchmark_results_qwen36_2026_06"
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # Which Tier-2 sub-metric backs the ConfAIde "r" column. The paper's "r" is
    # Tier-2b Pearson (verified: Tier-2b reproduces the paper exactly — Gemma
    # 61.3, ContextReasoner 60.2, CIRL 47.0 — whereas Tier-2a is far off, e.g.
    # Qwen3.5-4B 79.4 vs the paper's 53.7).
    CONFAIDE_R_SUBDIR = "compute_metrics_tier2b"

    # ── Paper rows: (display model, [conditions]) in paper order ──────────────
    ROW_ORDER = [
        ("Qwen3.5-9B", ["Zero-shot", "SFT", "GRPO"]),
        ("Qwen3.5-2B", ["Zero-shot", "SFT"]),
        ("Qwen3.5-4B", ["Zero-shot", "SFT"]),
        ("Gemma-3-12B", ["Zero-shot", "SFT"]),
        ("Phi-4", ["Zero-shot", "SFT"]),
        ("GPT-OSS-20B", ["Zero-shot", "SFT"]),
        ("OpenThinker3-7B", ["Zero-shot", "SFT"]),
        ("ContextReasoner-7B", ["Zero-shot"]),
        ("CIRL-7B", ["Zero-shot"]),
    ]

    # ── model= override string  →  (display model, condition) ─────────────────
    # "Zero-shot" maps to the instruct/it/ppo/base variant with no SFT/GRPO
    # adapter (verified against the judge-ablation: Qwen3.5-9B zero-shot = the
    # instruct weights, not the pretrained -Base checkpoint).
    OVERRIDE_TO_ROW = {
        "qwen3.5-9b/instruct": ("Qwen3.5-9B", "Zero-shot"),
        "qwen3.5-9b/sft-ci": ("Qwen3.5-9B", "SFT"),
        "qwen3.5-9b/grpo-v3-vr05-lambda05": ("Qwen3.5-9B", "GRPO"),
        "qwen3.5-2b/instruct": ("Qwen3.5-2B", "Zero-shot"),
        "qwen3.5-2b/sft-ci": ("Qwen3.5-2B", "SFT"),
        "qwen3.5-4b/instruct": ("Qwen3.5-4B", "Zero-shot"),
        "qwen3.5-4b/sft-ci": ("Qwen3.5-4B", "SFT"),
        "gemma-3-12b/it": ("Gemma-3-12B", "Zero-shot"),
        "gemma-3-12b/it-sft-ci": ("Gemma-3-12B", "SFT"),
        "phi-4/instruct": ("Phi-4", "Zero-shot"),
        "phi-4/sft-ci": ("Phi-4", "SFT"),
        "gpt-oss-20b/instruct": ("GPT-OSS-20B", "Zero-shot"),
        "gpt-oss-20b/sft-ci": ("GPT-OSS-20B", "SFT"),
        "openthinker3-7b/instruct": ("OpenThinker3-7B", "Zero-shot"),
        "openthinker3-7b/sft-ci": ("OpenThinker3-7B", "SFT"),
        "context-reasoner/ppo": ("ContextReasoner-7B", "Zero-shot"),
        "cirl/base": ("CIRL-7B", "Zero-shot"),
    }

    # ── Column registry: paper column → where to read it ──────────────────────
    # Fields: group, col, bench_dir, inner, subdir, key, judged, lower_is_better,
    #         scale ("pct" = ×100, "raw" = as-is)
    COLUMNS = [
        ("GoldCoin", "Appl.", "goldcoin", "goldcoin_hipaa",
         "compute_metrics_applicability", "macro_f1", False, False, "pct"),
        ("GoldCoin", "Comp.", "goldcoin", "goldcoin_hipaa",
         "compute_metrics_compliance", "macro_f1", False, False, "pct"),
        ("PrivacyLens", "QA Acc", "privacylens", "privacylens_eval",
         "compute_metrics", "qa_probing.accuracy", False, False, "pct"),
        ("PrivacyLens", "Lk↓", "privacylens", "privacylens_eval",
         "compute_metrics", "leakage.leakage_rate_overall_with_default_zero",
         True, True, "pct"),
        ("PrivacyLens", "Adj Lk↓", "privacylens", "privacylens_eval",
         "compute_metrics", "adjusted_leakage.adjusted_leakage_rate",
         True, True, "pct"),
        ("PrivacyLens", "Helpful", "privacylens", "privacylens_eval",
         "compute_metrics", "helpfulness.helpful_rate_overall_with_default_zero",
         True, False, "pct"),
        ("PrivacyLens", "Help", "privacylens", "privacylens_eval",
         "compute_metrics", "helpfulness.mean_score_overall_with_default_zero",
         True, False, "raw"),
        ("ConfAIde", "r", "confaide", "confaide",
         CONFAIDE_R_SUBDIR, "pearson_r", False, False, "pct"),
        ("CIRL", "Comp.", "cirl_trajectory", "cirl_vignettes",
         "compute_trajectory_metrics", "complete", True, False, "pct"),
        ("VLM", "Q7", "vlm_geoprivacy", "vlm_geoprivacy_bench",
         "compute_metrics", "per_question.Q7.accuracy", False, False, "pct"),
        ("MMLU", "Acc", "mmlu", "mmlu",
         "compute_metrics", "overall_accuracy", False, False, "pct"),
    ]
    return (
        COLUMNS,
        MULTIRUN_GLOB_ROOT,
        OVERRIDE_TO_ROW,
        REPORT_DIR,
        ROW_ORDER,
        dt,
        re,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase A — Scan eval_all multiruns

    Walk every `multirun/*_eval_all/<HH-MM-SS>/<idx>/` sub-run. For each, read
    the `model=` override (→ row identity) and the multirun's judge, then record
    one row per (sub-run × benchmark column) with the metric value if present.
    """)
    return


@app.cell
def _(COLUMNS, MULTIRUN_GLOB_ROOT, OVERRIDE_TO_ROW, dt, re):
    import json as _json

    _MR_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_eval_all/(\d{2}-\d{2}-\d{2})$")

    def _parse_mr_dt(mr_dir):
        m = _MR_RE.search(str(mr_dir))
        if not m:
            return None
        return dt.datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H-%M-%S")

    def _multirun_judge(mr_dir):
        """Resolve the judge for a multirun. Qwen3.6 when the async sidecar
        default (JUDGE_MODEL=Qwen3.6-27B) is set, or the one pre-sidecar served
        Qwen3.6 run (Apr24); everything else is treated as a non-Qwen3.6 judge
        (only matters for judged columns)."""
        y = mr_dir / "multirun.yaml"
        txt = y.read_text(errors="ignore") if y.exists() else ""
        if re.search(r"JUDGE_MODEL[,=]\s*Qwen3\.6-27B", txt):
            return "Qwen3.6-27B"
        if "2026-04-24_eval_all/10-13-47" in str(mr_dir):
            return "Qwen3.6-27B"  # served Qwen3.6 (see judge-ablation notebook)
        return "other"

    def _override_model(sub_dir):
        # The model override lives in any benchmark's .hydra/overrides.yaml.
        for ov in sub_dir.glob("*/.hydra/overrides.yaml"):
            for line in ov.read_text(errors="ignore").splitlines():
                line = line.strip().lstrip("- ").strip()
                if line.startswith("model="):
                    return line.split("=", 1)[1]
        return None

    def _dotted(d, path):
        cur = d
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                return None
            cur = cur[part]
        return cur

    rows = []
    for _mr in sorted(MULTIRUN_GLOB_ROOT.glob("*_eval_all/*")):
        if not _mr.is_dir():
            continue
        _mrdt = _parse_mr_dt(_mr)
        if _mrdt is None:
            continue
        _judge = _multirun_judge(_mr)
        for _sub in sorted(p for p in _mr.iterdir() if p.is_dir() and p.name.isdigit()):
            _ov = _override_model(_sub)
            if _ov is None or _ov not in OVERRIDE_TO_ROW:
                continue
            _model, _cond = OVERRIDE_TO_ROW[_ov]
            for (_grp, _col, _bd, _inner, _subdir, _key, _judged, _lo, _scale) in COLUMNS:
                _mp = _sub / _bd / _inner / "outputs" / _subdir / "metrics.json"
                if not _mp.exists():
                    continue
                try:
                    _val = _dotted(_json.loads(_mp.read_text()), _key)
                except (ValueError, OSError):
                    _val = None
                if _val is None:
                    continue
                rows.append({
                    "model": _model, "condition": _cond,
                    "group": _grp, "col": _col, "col_id": f"{_grp}::{_col}",
                    "value": float(_val), "judged": _judged,
                    "judge": _judge, "mr_dt": _mrdt,
                    "multirun": str(_mr.relative_to(MULTIRUN_GLOB_ROOT)),
                    "override": _ov,
                })

    import pandas as _pd
    scan = _pd.DataFrame(rows)
    print(f"{len(scan)} (sub-run × column) observations across "
          f"{scan['multirun'].nunique()} multiruns")
    print(f"models seen: {sorted(scan['model'].unique())}")
    scan
    return (scan,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase B — Per-cell latest-run selection

    For every (model, condition, column): keep candidates with a value;
    **judged** columns require `judge == Qwen3.6-27B`; **judge-free** columns
    accept any judge. Then take the most recent multirun.
    """)
    return


@app.cell
def _(scan):
    import pandas as _pd

    # Judged (LLM-judged) cells require Qwen3.6-27B. Judge-free cells (GoldCoin
    # F1, QA Acc, ConfAIde Tier-2 r, VLM Q7, MMLU) are deterministic scores that
    # don't depend on the judge, so they take the latest available run
    # regardless of which judge that run happened to use.
    _elig = scan[(~scan["judged"]) | (scan["judge"] == "Qwen3.6-27B")].copy()
    # Latest run per (model, condition, col).
    picked = (
        _elig.sort_values("mr_dt")
        .drop_duplicates(subset=["model", "condition", "col_id"], keep="last")
        .copy()
    )
    print(f"{len(picked)} cells filled "
          f"(judged → Qwen3.6-27B only; judge-free → latest available)")
    picked[["model", "condition", "group", "col", "value", "judge",
            "multirun"]].sort_values(["model", "condition", "col"])
    return (picked,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase C — Assemble the paper table + markdown

    Lay the cells out in the paper's row/column order, format (percentages ×100;
    Help on the 0–3 scale), bold the best per column, render, and save to
    `benchmark_results_qwen36.md` plus a provenance parquet.
    """)
    return


@app.cell
def _(COLUMNS, REPORT_DIR, ROW_ORDER, picked):
    import pandas as _pd
    import math as _math

    # Unique column id = "group::col" (GoldCoin "Comp." and CIRL "Comp." would
    # otherwise collide). specs: (col_id, group, col, lower_better, scale).
    _col_specs = [(f"{c[0]}::{c[1]}", c[0], c[1], c[7], c[8]) for c in COLUMNS]

    # value lookup: (model, condition, col_id) -> value
    _lut = {
        (r["model"], r["condition"], r["col_id"]): r["value"]
        for _, r in picked.iterrows()
    }
    _prov = {
        (r["model"], r["condition"], r["col_id"]): (r["judge"], r["multirun"])
        for _, r in picked.iterrows()
    }

    def _scaled(cid, scale):
        out = {}
        for (mdl, conds) in ROW_ORDER:
            for cond in conds:
                v = _lut.get((mdl, cond, cid))
                if v is None:
                    continue
                out[(mdl, cond)] = v * 100.0 if scale == "pct" else v
        return out

    # Best-per-column (over all filled cells) for bolding.
    _best = {}
    for (cid, grp, col, lo, scale) in _col_specs:
        vals = _scaled(cid, scale)
        if not vals:
            continue
        _best[cid] = (min if lo else max)(vals.values())

    def _fmt(cid, lo, scale, mdl, cond):
        v = _lut.get((mdl, cond, cid))
        if v is None:
            return "—"
        sv = v * 100.0 if scale == "pct" else v
        spec = "{:.1f}" if scale == "pct" else "{:.2f}"
        s = spec.format(sv)
        if cid in _best and abs(sv - _best[cid]) < 1e-9:
            s = f"**{s}**"
        return s

    # Markdown column labels carry the benchmark group as a prefix.
    _hdr_cols = [f"{grp} {col}" for (cid, grp, col, lo, scale) in _col_specs]

    _lines = []
    _lines.append("| Model | Cond. | " + " | ".join(_hdr_cols) + " |")
    _lines.append("|" + "---|" * (2 + len(_col_specs)))
    _dropped = []
    for (mdl, conds) in ROW_ORDER:
        _shown = 0
        for cond in conds:
            _cells = [_fmt(cid, lo, scale, mdl, cond)
                      for (cid, grp, col, lo, scale) in _col_specs]
            # Drop (model, condition) rows with no Qwen3.6-judged data at all —
            # they would otherwise be an all-"—" row sourced from nothing.
            if all(c == "—" for c in _cells):
                _dropped.append(f"{mdl} / {cond}")
                continue
            _mcell = mdl if _shown == 0 else ""
            _lines.append(f"| {_mcell} | {cond} | " + " | ".join(_cells) + " |")
            _shown += 1
    table_md = "\n".join(_lines)
    if _dropped:
        print("Dropped (no Qwen3.6-judged data for any column): " + ", ".join(_dropped))

    _legend = (
        "\n\n*Percentages (×100) except **Help** (mean helpfulness, 0–3). "
        "↓ = lower is better. Best per column in **bold**. "
        "GoldCoin Appl./Comp. = applicability/compliance macro-F1. "
        "PrivacyLens: QA Acc, Lk = leakage rate, Adj Lk = adjusted leakage, "
        "Helpful = helpful rate, Help = mean helpfulness. ConfAIde r = Tier-2b "
        "Pearson. CIRL Comp. = trajectory completeness. "
        "VLM Q7 = location-granularity accuracy. "
        "Judged columns (PrivacyLens Lk/Adj Lk/Helpful/Help, CIRL Comp.) use "
        "Qwen3.6-27B-judged runs only; judge-free columns (GoldCoin, QA Acc, "
        "ConfAIde r, VLM Q7, MMLU) use the latest available run regardless of "
        "judge. Most recent per cell.*"
    )
    _dropped_note = (
        f"\n\n*Rows omitted (no Qwen3.6-27B-judged run for any column): "
        f"{', '.join(_dropped)}.*" if _dropped else ""
    )
    full_md = (
        "## COLM benchmark results — Qwen3.6-judged (latest available)\n\n"
        + table_md + _legend + _dropped_note + "\n"
    )

    (REPORT_DIR / "benchmark_results_qwen36.md").write_text(full_md)

    # Provenance: judge + source multirun per filled cell.
    _prov_rows = []
    for (mdl, conds) in ROW_ORDER:
        for cond in conds:
            for (cid, grp, col, lo, scale) in _col_specs:
                if (mdl, cond, cid) in _prov:
                    j, mr = _prov[(mdl, cond, cid)]
                    _prov_rows.append({
                        "model": mdl, "condition": cond,
                        "column": f"{grp} {col}",
                        "judge": j, "multirun": mr,
                        "value": _lut[(mdl, cond, cid)],
                    })
    provenance = _pd.DataFrame(_prov_rows)
    provenance.to_parquet(REPORT_DIR / "benchmark_results_provenance.parquet", index=False)
    print(f"saved table + provenance to {REPORT_DIR}")
    print(full_md)
    return full_md, provenance


@app.cell
def _(full_md, mo):
    mo.md(full_md)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Provenance — which run & judge backs each cell

    Use this to audit recency / judge per cell, and to spot cells that fell back
    to an older run or are blank because no Qwen3.6-judged run exists.
    """)
    return


@app.cell
def _(provenance):
    provenance.sort_values(["model", "condition", "column"])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read the result

    - The **markdown table** above (also at
      `tables/colm_benchmark_results_qwen36_2026_06/benchmark_results_qwen36.md`)
      is the regenerated `benchmark_results.tex` with a Qwen3.6 judge + MMLU.
    - **Blank cells**: for judged columns, no Qwen3.6-judged run exists for that
      model+condition (e.g. CIRL completeness only exists for Qwen3.5-{2B,4B,9B}
      via the Apr24 trajectory run; GRPO only ran on PrivacyLens for
      Qwen3.5-9B). For judge-free columns, no run of any kind has it.
    - The **provenance table** shows the source multirun + judge per cell.
      Judge-free cells may legitimately come from a non-Qwen3.6 run (latest
      available); judged cells are always Qwen3.6-27B. Check it before quoting,
      since per-cell selection can mix multiruns within a row.
    - To switch the ConfAIde *r* sub-metric (Tier-2a ↔ Tier-2b) edit
      `CONFAIDE_R_SUBDIR` in the config cell. To add the canonical GRPO model for
      other benchmarks, run a Qwen3.6-judged `all_benchmarks` eval on the GRPO
      checkpoint and re-run this notebook.
    """)
    return


if __name__ == "__main__":
    app.run()
