import marimo

__generated_with = "0.23.3"
app = marimo.App()


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # SFT Pair-Format Ablation — Eval Results

    Eval-side companion to `sft_pair_ablation.py` (which analyzes SFT *training* loss).

    **Source sweep**: `multirun/2026-04-29_eval_all/12-15-37/` — 32 jobs:
    - 2 backbones: `Qwen3.5-9B-Base`, `Qwen3.5-9B-Instruct`
    - 16 toggle combinations of four IFT-format flags during SFT data prep:
      - `ctx`  → `flow_context`
      - `appr` → `flow_appropriateness`
      - `norms` → `flow_norms_meta`
      - `conf` → `flow_confidence`

    Each row in the analysis is one (backbone, variant) checkpoint evaluated on five CI benchmarks: GoldCoin-HIPAA, PrivacyLens, ConfAIde, CIRL Vignettes, VLM-GeoPrivacy.

    **Identifier**: runs are matched by `config.model.lora_path` (the `checkpoint_name` heuristic collapses all 32 to `Qwen3.5-9B-{Base,Instruct}+sft_only` so we cannot use it).

    **LoRA path → backbone**:
    - `2026-04-28_sft_pair_ablation/20-11-11/...` → `base`
    - `2026-04-28_sft_pair_ablation/18-34-09/...` → `instruct`

    **PrivacyLens caveat — read before citing**: every PrivacyLens job in this sweep failed the `agent_action_format` sanity check (format adherence ~2–3% vs. 0.9 threshold), so the orchestrator never wrote metrics to W&B. We rescue them from disk in §1b and surface `agent_action_format_rate` (PL Format) so a reader can see the size of the failure. The QA-probing metric is sanity-independent and trustworthy; **leakage / helpful / adjusted-leak are computed over only ~16 of 493 prompts and should be treated as unreliable**.

    **CIRL** uses the `cirl_vignettes_eval` pipeline → `cirl_accuracy` (not the trajectory-pipeline `complete` metric).
    """)
    return


@app.cell
def _():
    import subprocess, sys, re, json
    from pathlib import Path

    # Resolve the wicked dir so we can import its local wandb_cache module
    if "__vsc_ipynb_file__" in globals():
        NB_DIR = Path(globals()["__vsc_ipynb_file__"]).parent
    elif (Path.cwd() / "wandb_cache.py").exists():
        NB_DIR = Path.cwd()
    else:
        _repo = subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip()
        NB_DIR = Path(_repo) / "notebooks" / "wicked"
    if str(NB_DIR) not in sys.path:
        sys.path.insert(0, str(NB_DIR))

    import importlib, wandb_cache; importlib.reload(wandb_cache)
    from wandb_cache import load_runs_raw, cache_info
    import pandas as pd
    import numpy as np
    from IPython.display import display, Latex

    pd.set_option("display.max_columns", 40)
    pd.set_option("display.float_format", "{:.3f}".format)

    # Use the COLM26 cache directory — single source of truth for eval-all runs.
    _REPO = Path(subprocess.check_output(["git", "rev-parse", "--show-toplevel"], text=True).strip())
    CACHE_DIR = _REPO / "notebooks" / "COLM26" / "wandb_cache"

    DS_PREFIX = {
        "goldcoin_hipaa": "gc", "privacylens": "pl",
        "vlm_geoprivacy_bench": "vlm", "confaide": "ca",
        "cirl_vignettes": "cirl",
    }

    # Backbone discrimination by sweep timestamp
    BACKBONE_BY_SWEEP_TS = {
        "20-11-11": "base",
        "18-34-09": "instruct",
    }
    ABLATION_SWEEP_DIR = "2026-04-28_sft_pair_ablation"
    VARIANT_RE = re.compile(r"ctx-(True|False)_appr-(True|False)_norms-(True|False)_conf-(True|False)")
    TOGGLES = ["ctx", "appr", "norms", "conf"]

    # Display constants used throughout the notebook
    BACKBONES = ["base", "instruct"]
    BACKBONE_TITLE = {"base": "Qwen3.5-9B-Base", "instruct": "Qwen3.5-9B-Instruct"}
    out_dir = NB_DIR / "tables"
    out_dir.mkdir(exist_ok=True)

    info = cache_info(CACHE_DIR)
    print(f"Using cached W&B data from {info.get('fetched_at', 'unknown')} ({info.get('total_runs', '?')} runs)")
    print(f"Cache dir: {CACHE_DIR}")
    return (
        ABLATION_SWEEP_DIR,
        BACKBONE_BY_SWEEP_TS,
        BACKBONE_TITLE,
        CACHE_DIR,
        DS_PREFIX,
        NB_DIR,
        Path,
        TOGGLES,
        VARIANT_RE,
        display,
        json,
        load_runs_raw,
        np,
        out_dir,
        pd,
        re,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. Load and filter to the SFT pair-ablation runs

    We hit the raw enriched dicts (not the flat DataFrame loader) so we can read `config.model.lora_path` to identify each variant.
    """)
    return


@app.cell
def _(
    ABLATION_SWEEP_DIR,
    BACKBONE_BY_SWEEP_TS,
    CACHE_DIR,
    VARIANT_RE,
    display,
    load_runs_raw,
    pd,
):
    # Pull all finetuned runs from after the SFT-eval kickoff. We keep a wide
    # window in case some runs straggle across day boundaries.
    raw = load_runs_raw(cache_dir=CACHE_DIR, tags=['finetuned'], after='2026-04-29', before='2026-05-02', has_metrics=True)
    print(f'Pulled {len(raw)} finetuned runs in date window')

    def parse_variant(run):
        """Extract (backbone, variant_str, ctx, appr, norms, conf) from a run, or None."""
        cfg = run.get('config') or {}
        model_cfg = cfg.get('model') or {}
        lora_path = str(model_cfg.get('lora_path') or '')
        if ABLATION_SWEEP_DIR not in lora_path:
            return None
        backbone = None
        for ts, label in BACKBONE_BY_SWEEP_TS.items():
            if f'/{ts}/' in lora_path:
                backbone = label
                break
        if backbone is None:
            return None  # backbone: identify which of the two parent timestamps the path goes through
        m = VARIANT_RE.search(lora_path)
        if not m:
            return None
        ctx, appr, norms, conf = (s == 'True' for s in m.groups())
        variant = m.group(0)
        return (backbone, variant, ctx, appr, norms, conf)
    PRECOMPUTED = ['gc_applicability_f1', 'gc_compliance_f1', 'gc_applicable_f1', 'gc_not_applicable_f1', 'gc_permit_f1', 'gc_forbid_f1', 'pl_qa_accuracy', 'pl_leakage_rate', 'pl_adjusted_leakage_rate', 'pl_helpful_rate', 'vlm_q7_accuracy', 'ca_pearson_r', 'ca_accuracy', 'ca_leak_rate', 'cirl_accuracy', 'cirl_integrity', 'cirl_utility', 'cirl_complete']
    EVAL_PREFIXES = ('compute_metrics/eval/', 'compute_trajectory_metrics/eval/')
    rows = []
    for r in raw:
        parsed = parse_variant(r)
        if parsed is None:
            continue
        backbone, variant, ctx, appr, norms, conf = parsed
    # Build flat rows: one per (run, dagspace) with metrics + parsed metadata
        row = {'run_id': r['run_id'], 'run_name': r['run_name'], 'created_at': r['created_at'], 'dagspace': r.get('dagspace'), 'backbone': backbone, 'variant': variant, 'ctx': ctx, 'appr': appr, 'norms': norms, 'conf': conf}
        for k in PRECOMPUTED:
            if k in r:
                row[k] = r[k]
        for k, _v in (r.get('summary') or {}).items():
            if not isinstance(_v, (int, float)):
                continue
            for prefix in EVAL_PREFIXES:
                if k.startswith(prefix):
                    row[f'eval/{k[len(prefix):]}'] = _v
                    break
        rows.append(row)
    df_raw = pd.DataFrame(rows)
    df_raw['created_at'] = pd.to_datetime(df_raw['created_at'], utc=True)
    print(f'Matched {len(df_raw)} (run, dagspace) rows from the SFT pair-ablation sweep')
    print(f'  backbones: {sorted(df_raw['backbone'].unique())}')
    print(f'  variants:  {df_raw['variant'].nunique()} unique')
    print(f'  dagspaces: {sorted(df_raw['dagspace'].dropna().unique())}')
    display(df_raw.head())
    return df_raw, parse_variant


@app.cell
def _(CACHE_DIR, df_raw, load_runs_raw, parse_variant):
    # Sanity: count per (backbone, variant). With 5 dagspaces × 32 ablations
    # we expect 160 rows max, but PrivacyLens metrics are dropped (see below) so
    # 4 dagspaces × 32 = 128 is the realistic floor.
    coverage = (df_raw.groupby(["backbone", "variant"])["dagspace"]
                .nunique().rename("n_dagspaces").reset_index())
    print(f"(backbone, variant) cells with at least one dagspace: {len(coverage)} (expect 32)")
    print(f"  median dagspaces per cell: {int(coverage['n_dagspaces'].median())} (4 if PrivacyLens fails sanity)")

    # PrivacyLens diagnostic: pull *all* finalize runs (no has_metrics filter) and
    # check which ones reported a sanity failure.
    _pl_finalize = [r for r in load_runs_raw(
                        cache_dir=CACHE_DIR,
                        after="2026-04-29", before="2026-05-02",
                        dagspace="privacylens",
                        tags=["phase:finalize"])
                    if parse_variant(r) is not None]
    _pl_failed = [r for r in _pl_finalize
                  if (r.get("summary") or {}).get("orchestrator/status") == "failed"]
    if _pl_finalize:
        print(f"\nPrivacyLens finalize runs in scope: {len(_pl_finalize)} | "
              f"sanity-failed: {len(_pl_failed)}")
        if _pl_failed:
            sample_err = (_pl_failed[0].get("summary") or {}).get("orchestrator/error", "")
            print(f"  sample error: {sample_err[:200].replace(chr(10), ' ')}...")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1b. PrivacyLens — rescue metrics from disk

    W&B never received metrics for these 32 PrivacyLens runs because the orchestrator's `agent_action_format` sanity stage halted before `compute_metrics` could log to the run summary. But `compute_metrics` itself ran and wrote `metrics.json` to disk under each multirun job's output dir, so we read those files directly and merge them in with a `(unreliable)` caveat.

    The headline trust metric is **`agent_action_format_rate`** — fraction of agent-action responses parseable by the leakage/helpfulness judges. Sanity threshold is 0.9; this sweep is at ~2–3% across the board.
    """)
    return


@app.cell
def _(Path, display, json, pd, re):
    MULTIRUN_ROOT = Path('/share/pierson/matt/UAIR/multirun/2026-04-29_eval_all/12-15-37')
    _OVERRIDE_RE = re.compile('model=qwen3\\.5-9b/(base|instruct)/(ctx-(True|False)_appr-(True|False)_norms-(True|False)_conf-(True|False))')

    def _read_overrides(job_dir):
        f = job_dir / '.hydra' / 'overrides.yaml'
        if not f.exists():
            return None
        text = f.read_text()
        m = _OVERRIDE_RE.search(text)
        if not m:
            return None
        backbone, variant, ctx, appr, norms, conf = m.groups()
        return {'backbone': backbone, 'variant': variant, 'ctx': ctx == 'True', 'appr': appr == 'True', 'norms': norms == 'True', 'conf': conf == 'True'}

    def _load_pl_metrics(job_dir):
        f = job_dir / 'privacylens' / 'privacylens_eval' / 'outputs' / 'compute_metrics' / 'metrics.json'
        if not f.exists():
            return None
        return json.loads(f.read_text())
    pl_disk_rows = []
    for job_dir in sorted(MULTIRUN_ROOT.iterdir(), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not job_dir.is_dir() or not job_dir.name.isdigit():
            continue
        meta = _read_overrides(job_dir)
        metrics = _load_pl_metrics(job_dir)
        if meta is None or metrics is None:
            continue
        qa = metrics.get('qa_probing') or {}
        leak = metrics.get('leakage') or {}
        helpf = metrics.get('helpfulness') or {}
        adj = metrics.get('adjusted_leakage') or {}
        pl_disk_rows.append({'job': int(job_dir.name), **meta, 'pl_format_adherence': leak.get('agent_action_format_rate'), 'pl_n_total': leak.get('total'), 'pl_n_parseable': (leak.get('total') or 0) - (leak.get('skipped_no_action_format') or 0), 'pl_qa_accuracy': qa.get('accuracy'), 'pl_leakage_rate': leak.get('leakage_rate_among_parseable'), 'pl_leakage_rate_overall': leak.get('leakage_rate_overall_with_default_zero'), 'pl_helpful_rate': helpf.get('helpful_rate_among_parseable'), 'pl_helpful_rate_overall': helpf.get('helpful_rate_overall_with_default_zero'), 'pl_adjusted_leakage_rate': adj.get('adjusted_leakage_rate'), 'pl_adj_n_helpful_judged': adj.get('total_helpful_and_judged')})
    pl_disk = pd.DataFrame(pl_disk_rows)
    print(f'Loaded PrivacyLens metrics from {len(pl_disk)} multirun jobs (expect 32)')
    RATE_COLS = [_c for _c in pl_disk.columns if _c.startswith('pl_') and _c.endswith(('_rate', '_adherence', '_overall', 'accuracy'))]
    for _c in RATE_COLS:
        pl_disk[_c] = pl_disk[_c] * 100
    # Convert rates [0,1] to percent for display consistency with the rest of the notebook
    display(pl_disk.sort_values(['backbone', 'variant']).round(2))  # Headline trust signal — both leakage and helpfulness emit this; they agree  # Counts (per the standard 493-sample PrivacyLens eval set)  # QA-probing (sanity-independent — does not use agent_action format)  # Leakage / helpfulness — *_among_parseable reflects only the tiny subset  # that produced valid agent actions; *_overall_with_default_zero applies  # default-no-leak/default-not-helpful to unparseable responses.  # Adjusted leakage = leak rate among helpful & judged. With ~16  # parseable / 0 helpful, this is essentially undefined for this sweep.
    return (pl_disk,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### How badly did sanity fail?

    `agent_action_format_rate` is the fraction of the 493 PrivacyLens prompts where the model returned a parseable agent-action JSON. The sanity-stage threshold is **0.9**. Bars below show the gap.
    """)
    return


@app.cell
def _(BACKBONE_TITLE, TOGGLES, display, out_dir, pl_disk):
    import matplotlib.pyplot as plt
    SANITY_THRESHOLD_PCT = 90.0
    adh = pl_disk['pl_format_adherence']
    print(f'Format adherence (%) — sanity threshold = {SANITY_THRESHOLD_PCT}%')
    print(f'  n runs:  {len(adh)}')
    print(f'  min:     {adh.min():.2f}')
    print(f'  median:  {adh.median():.2f}')
    print(f'  max:     {adh.max():.2f}')
    print(f'  passing sanity (>= {SANITY_THRESHOLD_PCT}%): {int((adh >= SANITY_THRESHOLD_PCT).sum())} / {len(adh)}')
    print('\nBy backbone:')
    display(pl_disk.groupby('backbone')['pl_format_adherence'].agg(['count', 'min', 'median', 'max']).round(2))

    def _bitmask(r):
        return r['ctx'] * 8 + r['appr'] * 4 + r['norms'] * 2 + r['conf'] * 1
    pl_disk_1 = pl_disk.assign(_bbo=pl_disk['backbone'].map({'base': 0, 'instruct': 1}), _bbm=pl_disk.apply(_bitmask, axis=1)).sort_values(['_bbo', '_bbm']).drop(columns=['_bbo', '_bbm']).reset_index(drop=True)

    def _label(r):
        return ''.join(('1' if r[t] else '0' for t in TOGGLES))
    pl_disk_1['vlabel'] = pl_disk_1.apply(_label, axis=1)
    _fig, _axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
    backbone_color = {'base': '#4C72B0', 'instruct': '#DD8452'}
    for _ax, _bb in zip(_axes, ['base', 'instruct']):
        _sub = pl_disk_1[pl_disk_1['backbone'] == _bb]
        _ax.bar(_sub['vlabel'], _sub['pl_format_adherence'], color=backbone_color[_bb], edgecolor='white', linewidth=0.5, zorder=3)
        _ax.axhline(SANITY_THRESHOLD_PCT, color='crimson', lw=1, ls='--', label=f'sanity threshold ({SANITY_THRESHOLD_PCT:.0f}%)', zorder=2)
        _ax.set_title(BACKBONE_TITLE[_bb], fontweight='bold')
        _ax.set_xlabel('variant (ctx,appr,norms,conf)')
        _ax.set_xticklabels(_sub['vlabel'], rotation=60, ha='right', fontfamily='monospace', fontsize=8)
        _ax.set_ylim(0, 100)
        _ax.grid(axis='y', alpha=0.25, lw=0.5)
        _ax.legend(loc='upper right', fontsize=8, framealpha=0.95)
    _axes[0].set_ylabel('agent_action_format_rate (%)')
    _fig.suptitle('PrivacyLens — agent-action format adherence per ablation variant', fontweight='bold')
    _fig.savefig(out_dir / 'sft_pair_ablation_pl_sanity.pdf')
    plt.show()
    return pl_disk_1, plt


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. Pivot per-dagspace metrics into one row per (backbone, variant)
    """)
    return


@app.cell
def _(DS_PREFIX, df_raw, display, pd):
    # De-duplicate: keep last run per (backbone, variant, dagspace) by created_at
    df_dedup = df_raw.sort_values('created_at').drop_duplicates(subset=['backbone', 'variant', 'dagspace'], keep='last')
    metric_cols_all = [_c for _c in df_dedup.columns if _c.startswith(('gc_', 'pl_', 'vlm_', 'ca_', 'cirl_', 'eval/'))]

    def _pivot_dagspace(df, dagspace):
        _sub = df[df['dagspace'] == dagspace]
        if _sub.empty:
            return pd.DataFrame()
        prefix = DS_PREFIX.get(dagspace, dagspace)
        keep = [_c for _c in metric_cols_all if _sub[_c].notna().any()]
        out = _sub[['backbone', 'variant', 'ctx', 'appr', 'norms', 'conf'] + keep].copy()
        rename = {_c: f'{prefix}/{_c[5:]}' for _c in keep if _c.startswith('eval/')}
        return out.rename(columns=rename)
    df_table = pd.DataFrame()
    for ds in sorted(df_dedup['dagspace'].dropna().unique()):
        piv = _pivot_dagspace(df_dedup, ds)
        if piv.empty:
            continue
        if df_table.empty:
            df_table = piv
        else:
            keys = ['backbone', 'variant', 'ctx', 'appr', 'norms', 'conf']
            overlap = [_c for _c in piv.columns if _c in df_table.columns and _c not in keys]
            if overlap:
                piv = piv.drop(columns=overlap)
            df_table = df_table.merge(piv, on=keys, how='outer')
    df_table = df_table.copy()
    metric_out_cols = [_c for _c in df_table.columns if _c not in {'backbone', 'variant', 'ctx', 'appr', 'norms', 'conf'}]
    for _c in metric_out_cols:
        if pd.api.types.is_numeric_dtype(df_table[_c]):
            df_table[_c] = df_table[_c] * 100
    sort_keys = pd.DataFrame({'_bbo': df_table['backbone'].map({'base': 0, 'instruct': 1}), '_bbm': df_table['ctx'].astype(int) * 8 + df_table['appr'].astype(int) * 4 + df_table['norms'].astype(int) * 2 + df_table['conf'].astype(int)}, index=df_table.index)
    # Defragment after wide-merge so subsequent .insert calls don't warn
    df_table = df_table.assign(**sort_keys).sort_values(['_bbo', '_bbm']).drop(columns=['_bbm', '_bbo']).reset_index(drop=True)
    print(f'Full pivoted table: {df_table.shape}')
    # Convert [0,1] → percent for any metric col
    # Sort: backbone (base, instruct), then by toggle bitmask (000 → 1111)
    display(df_table.round(2))
    return (df_table,)


@app.cell
def _(df_table, pl_disk_1):
    _pl_metric_cols = [_c for _c in pl_disk_1.columns if _c.startswith('pl_') and (not _c.endswith(('_n_total', '_n_parseable', '_adj_n_helpful_judged')))]
    _pl_merge = pl_disk_1[['backbone', 'variant'] + _pl_metric_cols].copy()
    df_table_1 = df_table.merge(_pl_merge, on=['backbone', 'variant'], how='left').copy()
    print(f'Merged {len(_pl_metric_cols)} PrivacyLens columns into df_table → shape {df_table_1.shape}')
    return (df_table_1,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. Headline metric set

    Same set as `zero_shot_vs_sft.ipynb` so this notebook composes with the others.
    """)
    return


@app.cell
def _(df_table_1):
    PAPER_METRICS = [('gc_applicability_f1', 'App F1', False), ('gc_compliance_f1', 'Comp F1', False), ('pl_qa_accuracy', 'QA Acc', False), ('pl_adjusted_leakage_rate', 'Adj Leak', True), ('pl_helpful_rate', 'Helpful', False), ('pl_format_adherence', 'PL Format', False), ('ca/pearson_r', 'Pearson r', False), ('cirl_accuracy', 'CIRL Acc', False), ('vlm/Q7/accuracy', 'Q7 Acc', False)]
    UNRELIABLE_METRICS = {'pl_qa_accuracy', 'pl_adjusted_leakage_rate', 'pl_helpful_rate', 'pl_format_adherence'}
    _missing = [(_c, l) for _c, l, _ in PAPER_METRICS if _c not in df_table_1.columns]
    PAPER_METRICS = [(_c, l, lb) for _c, l, lb in PAPER_METRICS if _c in df_table_1.columns]
    if _missing:
        print(f'Headline metrics absent from table (skipped): {_missing}')
    print(f'Headline metrics in use: {[l for _, l, _ in PAPER_METRICS]}')
    print(f'Unreliable (sanity-failed): {sorted(UNRELIABLE_METRICS & {_c for _c, _, _ in PAPER_METRICS})}')
    metric_cols = [_c for _c, _, _ in PAPER_METRICS]
    label_by_col = {_c: l for _c, l, _ in PAPER_METRICS}
    lower_better = {_c for _c, _, lb in PAPER_METRICS if lb}
    return PAPER_METRICS, label_by_col, lower_better, metric_cols


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 4. Toggle main effects

    For each backbone × metric × toggle T, the **main effect** is the mean of all 8 variants where T=True minus the mean of the 8 where T=False. A positive number means *including this CI metadata field in the SFT target* helps that metric (or hurts it for lower-is-better metrics, where a negative main effect is good).
    """)
    return


@app.cell
def _(
    PAPER_METRICS,
    TOGGLES,
    df_table_1,
    display,
    label_by_col,
    lower_better,
    metric_cols,
    pd,
):
    def main_effects(df, metrics, toggles):
        """Return long-form DataFrame: one row per (backbone, metric, toggle)."""
        out = []
        for backbone, _sub in df.groupby('backbone'):
            for m in metrics:
                if m not in _sub.columns:
                    continue
                for t in toggles:
                    on = _sub.loc[_sub[t], m].mean()
                    off = _sub.loc[~_sub[t], m].mean()
                    out.append({'backbone': backbone, 'metric': label_by_col.get(m, m), 'metric_col': m, 'toggle': t, 'on_mean': on, 'off_mean': off, 'effect': on - off, 'lower_is_better': m in lower_better})
        return pd.DataFrame(out)
    df_eff = main_effects(df_table_1, metric_cols, TOGGLES)
    eff_table = df_eff.pivot_table(index=['backbone', 'metric'], columns='toggle', values='effect')[TOGGLES]
    metric_order = [l for _, l, _ in PAPER_METRICS]
    eff_table = eff_table.reindex(pd.MultiIndex.from_product([['base', 'instruct'], metric_order], names=['backbone', 'metric'])).dropna(how='all')
    print('Main effect (Δ percentage points): on - off')
    display(eff_table.round(2))
    return (eff_table,)


@app.cell
def _(NB_DIR, TOGGLES, eff_table, np, pd, plt):
    import matplotlib.colors as mcolors
    plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'DejaVu Serif'], 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight', 'axes.spines.top': False, 'axes.spines.right': False})
    BACKBONES_1 = ['base', 'instruct']
    BACKBONE_TITLE_1 = {'base': 'Qwen3.5-9B-Base', 'instruct': 'Qwen3.5-9B-Instruct'}
    _fig, _axes = plt.subplots(1, len(BACKBONES_1), figsize=(4.6 * len(BACKBONES_1), 4.6), sharey=True)
    if not isinstance(_axes, (list, np.ndarray)):
        _axes = [_axes]
    vmax = float(np.nanmax(np.abs(eff_table.values))) or 1.0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax, vcenter=0, vmax=vmax)
    cmap = plt.get_cmap('RdBu_r')
    for _ax, _bb in zip(_axes, BACKBONES_1):
        if _bb not in eff_table.index.get_level_values(0):
            _ax.set_visible(False)
            continue
        _sub = eff_table.loc[_bb]
        im = _ax.imshow(_sub.values, cmap=cmap, norm=norm, aspect='auto')
        _ax.set_xticks(range(len(TOGGLES)))
        _ax.set_xticklabels(TOGGLES)
        _ax.set_yticks(range(len(_sub.index)))
        _ax.set_yticklabels(_sub.index)
        _ax.set_title(BACKBONE_TITLE_1[_bb], fontweight='bold')
        for _i in range(_sub.shape[0]):
            for _j in range(_sub.shape[1]):
                _v = _sub.values[_i, _j]
                if pd.isna(_v):
                    continue
                _ax.text(_j, _i, f'{_v:+.1f}', ha='center', va='center', fontsize=8, color='black' if abs(_v) < vmax * 0.55 else 'white')
    _fig.colorbar(im, ax=_axes, fraction=0.025, pad=0.02, label='Δ percentage points (on − off)')
    _fig.suptitle('SFT pair-format ablation — toggle main effects', fontweight='bold')
    out_dir_1 = NB_DIR / 'tables'
    out_dir_1.mkdir(exist_ok=True)
    _fig.savefig(out_dir_1 / 'sft_pair_ablation_main_effects.pdf')
    _fig.savefig(out_dir_1 / 'sft_pair_ablation_main_effects.png')
    plt.show()
    return BACKBONES_1, BACKBONE_TITLE_1, out_dir_1


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 5. Per-variant heatmap

    16 variants × headline metrics, per backbone. Each cell colored by **within-metric, within-backbone z-score** so signals from differently-scaled metrics are comparable.
    """)
    return


@app.cell
def _(
    BACKBONES_1,
    BACKBONE_TITLE_1,
    TOGGLES,
    df_table_1,
    label_by_col,
    lower_better,
    metric_cols,
    np,
    out_dir_1,
    pd,
    plt,
):
    def _label(row):
        return ''.join(('1' if row[t] else '0' for t in TOGGLES))
    df_table_1['vlabel'] = df_table_1.apply(_label, axis=1)
    _fig, _axes = plt.subplots(1, len(BACKBONES_1), figsize=(5.2 * len(BACKBONES_1), 5.6), sharey=True)
    if not isinstance(_axes, (list, np.ndarray)):
        _axes = [_axes]
    for _ax, _bb in zip(_axes, BACKBONES_1):
        _sub = df_table_1[df_table_1['backbone'] == _bb].sort_values('vlabel')[['vlabel'] + metric_cols].set_index('vlabel')
        if _sub.empty:
            _ax.set_visible(False)
            continue
        z = _sub.copy()
        for _c in z.columns:
            sign = -1 if _c in lower_better else 1
            col = z[_c] * sign
            mu, sd = (col.mean(), col.std(ddof=0))
            z[_c] = (col - mu) / sd if sd and (not np.isnan(sd)) else 0.0
        _ax.imshow(z.values, cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
        _ax.set_xticks(range(len(metric_cols)))
        _ax.set_xticklabels([label_by_col[_c] for _c in metric_cols], rotation=35, ha='right')
        _ax.set_yticks(range(len(_sub.index)))
        _ax.set_yticklabels(_sub.index, fontfamily='monospace')
        _ax.set_title(BACKBONE_TITLE_1[_bb], fontweight='bold')
        for _i in range(_sub.shape[0]):
            for _j, _c in enumerate(metric_cols):
                _v = _sub.values[_i, _j]
                if pd.isna(_v):
                    continue
                _ax.text(_j, _i, f'{_v:.0f}', ha='center', va='center', fontsize=7, color='black' if abs(z.values[_i, _j]) < 1.1 else 'white')
    _axes[0].set_ylabel('variant (ctx,appr,norms,conf)')
    _fig.suptitle('SFT pair-format ablation — per-variant performance (color: within-metric z-score, sign-flipped for ↓ metrics)', fontweight='bold', y=1.02)
    _fig.savefig(out_dir_1 / 'sft_pair_ablation_per_variant.pdf')
    _fig.savefig(out_dir_1 / 'sft_pair_ablation_per_variant.png')
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 6. Best variant per (backbone, metric)
    """)
    return


@app.cell
def _(BACKBONES_1, PAPER_METRICS, df_table_1, display, np, pd):
    best_rows = []
    for _bb in BACKBONES_1:
        _sub = df_table_1[df_table_1['backbone'] == _bb]
        if _sub.empty:
            continue
        for _c, l, lb in PAPER_METRICS:
            if _c not in _sub.columns or _sub[_c].isna().all():
                best_rows.append({'backbone': _bb, 'metric': l, 'best_variant': None, 'best_value': np.nan, 'all_True_value': np.nan, 'all_False_value': np.nan, 'spread': np.nan})
                continue
            idx = _sub[_c].idxmin() if lb else _sub[_c].idxmax()
            best_v = _sub.loc[idx, 'vlabel']
            best_val = _sub.loc[idx, _c]
            all_T = _sub.loc[_sub['vlabel'] == '1111', _c]
            all_F = _sub.loc[_sub['vlabel'] == '0000', _c]
            best_rows.append({'backbone': _bb, 'metric': l, 'best_variant': best_v, 'best_value': best_val, 'all_True_value': float(all_T.iloc[0]) if len(all_T) else np.nan, 'all_False_value': float(all_F.iloc[0]) if len(all_F) else np.nan, 'spread': _sub[_c].max() - _sub[_c].min()})
    df_best = pd.DataFrame(best_rows)
    display(df_best.round(2))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 7. LaTeX — main-effects table
    """)
    return


@app.cell
def _(BACKBONES_1, BACKBONE_TITLE_1, TOGGLES, eff_table, out_dir_1, pd):
    def build_main_effects_latex(eff_table: pd.DataFrame) -> str:
        metric_cols_display = ['\\textbf{' + t + '}' for t in TOGGLES]
        lines = []
        lines.append('\\begin{table}[t]')
        lines.append('\\centering')
        lines.append('\\caption{SFT pair-format ablation main effects (\\%-points). Each cell is the mean over the 8 variants with the toggle on minus the mean over the 8 variants with it off. Adj Leak is lower-is-better; for that row, negative values mean the field helps.}')
        lines.append('\\label{tab:sft-pair-ablation-main-effects}')
        lines.append('\\begin{tabular}{ll' + 'c' * len(TOGGLES) + '}')
        lines.append('\\toprule')
        lines.append('\\textbf{Backbone} & \\textbf{Metric} & ' + ' & '.join(metric_cols_display) + ' \\\\')
        lines.append('\\midrule')
        for _bb in BACKBONES_1:
            if _bb not in eff_table.index.get_level_values(0):
                continue
            _sub = eff_table.loc[_bb]
            for _i, (metric, row) in enumerate(_sub.iterrows()):
                backbone_cell = f'\\multirow{{{len(_sub)}}}{{*}}{{{BACKBONE_TITLE_1[_bb]}}}' if _i == 0 else ''
                cells = [backbone_cell, metric]
                for t in TOGGLES:
                    _v = row[t]
                    if pd.isna(_v):
                        cells.append('---')
                        continue
                    helpful = _v < 0 if metric == 'Adj Leak' else _v > 0
                    txt = f'{_v:+.1f}'
                    if helpful:
                        txt = f'\\textcolor{{teal}}{{{txt}}}'
                    elif _v != 0:
                        txt = f'\\textcolor{{red}}{{{txt}}}'
                    cells.append(txt)
                lines.append(' & '.join(cells) + ' \\\\')
            lines.append('\\midrule')
        if lines[-1] == '\\midrule':
            lines[-1] = '\\bottomrule'
        else:
            lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        lines.append('\\end{table}')
        return '\n'.join(lines)
    latex_main = build_main_effects_latex(eff_table)
    _out_path = out_dir_1 / 'sft_pair_ablation_main_effects.tex'
    _out_path.write_text(latex_main)
    print(f'Saved {_out_path}')
    print(latex_main)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 8. LaTeX — full per-variant table

    32 rows × headline metrics. Best value per metric per backbone is bolded.
    """)
    return


@app.cell
def _(
    BACKBONES_1,
    BACKBONE_TITLE_1,
    PAPER_METRICS,
    df_table_1,
    label_by_col,
    np,
    out_dir_1,
    pd,
):
    def build_per_variant_latex(df_t: pd.DataFrame) -> str:
        cols = [_c for _c, _, _ in PAPER_METRICS]
        headers = [label_by_col[_c] for _c in cols]
        lines = []
        lines.append('\\begin{table}[t]')
        lines.append('\\centering')
        lines.append('\\small')
        lines.append('\\caption{Per-variant SFT pair-format ablation (\\%). Variant code = (ctx, appr, norms, conf). Best per (backbone, metric) bolded; $\\downarrow$ = lower is better.}')
        lines.append('\\label{tab:sft-pair-ablation-per-variant}')
        lines.append('\\begin{tabular}{ll' + 'c' * len(cols) + '}')
        lines.append('\\toprule')
        col_labels = {'Adj Leak': 'Adj Leak $\\downarrow$', 'Pearson r': 'Pearson $r$'}
        header_row = ['\\textbf{Backbone}', '\\textbf{Variant}'] + [col_labels.get(h, h) for h in headers]
        lines.append(' & '.join(header_row) + ' \\\\')
        lines.append('\\midrule')
        for _bb in BACKBONES_1:
            _sub = df_t[df_t['backbone'] == _bb].sort_values('vlabel')
            if _sub.empty:
                continue
            best_idx = {}
            for _c, _, lb in PAPER_METRICS:
                if _c in _sub.columns and _sub[_c].notna().any():
                    best_idx[_c] = _sub[_c].idxmin() if lb else _sub[_c].idxmax()
            first = True
            for idx, row in _sub.iterrows():
                cells = []
                cells.append(f'\\multirow{{{len(_sub)}}}{{*}}{{{BACKBONE_TITLE_1[_bb]}}}' if first else '')
                cells.append(f'\\texttt{{{row['vlabel']}}}')
                for _c in cols:
                    _v = row.get(_c, np.nan)
                    if pd.isna(_v):
                        cells.append('---')
                        continue
                    txt = f'{_v:.1f}'
                    if best_idx.get(_c) == idx:
                        txt = f'\\textbf{{{txt}}}'
                    cells.append(txt)
                lines.append(' & '.join(cells) + ' \\\\')
                first = False
            lines.append('\\midrule')
        if lines[-1] == '\\midrule':
            lines[-1] = '\\bottomrule'
        else:
            lines.append('\\bottomrule')
        lines.append('\\end{tabular}')
        lines.append('\\end{table}')
        return '\n'.join(lines)
    latex_full = build_per_variant_latex(df_table_1)
    _out_path = out_dir_1 / 'sft_pair_ablation_per_variant.tex'
    _out_path.write_text(latex_full)
    print(f'Saved {_out_path}')
    print(latex_full)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. Save the wide CSV for downstream use
    """)
    return


@app.cell
def _(df_table_1, out_dir_1):
    csv_path = out_dir_1 / 'sft_pair_ablation_metrics.csv'
    df_table_1.to_csv(csv_path, index=False)
    print(f'Saved {csv_path} ({df_table_1.shape[0]} rows × {df_table_1.shape[1]} cols)')
    return


if __name__ == "__main__":
    app.run()
