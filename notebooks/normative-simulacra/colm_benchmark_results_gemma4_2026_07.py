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
    # COLM benchmark-results table — Gemma-4-judged canonical set (2026-07)

    Same template as `colm_benchmark_results_qwen36_2026_06.py`, retargeted at the
    **2026-07-16/17 canonical-set sweeps**: `eval_canonical_instruct` (Zero-shot)
    and `eval_canonical_sft_gemma4` (SFT), both judged by **Gemma-4-31B-it**.

    ### What changed vs the Qwen3.6 notebook

    1. **Judge is Gemma-4-31B-it, and it is read from the artifacts, not the
       config.** The sweeps' `multirun.yaml` still carries the stale OmegaConf
       default `${oc.env:JUDGE_MODEL,Qwen3.6-27B}`, so the old notebook's
       `multirun.yaml` regex would mislabel every one of these runs as Qwen3.6.
       Here the judge is resolved from
       `privacylens_eval/outputs/*_judge_batch/manifest.json` → `model`, which
       records what actually served (`judge_export.py` resolves it from
       `/v1/models` at export time). All three sweeps resolve to
       `Gemma-4-31B-it`. **PrivacyLens rows are therefore NOT comparable with the
       pre-07-16 Qwen3.6-judged column.**
    2. **Rows are the canonical 11 × {Zero-shot, SFT}.** The canonical 13 minus
       `cirl/base` and `context-reasoner/ppo`, which are comparison baselines and
       were not SFT targets. Zero-shot = the `<family>/instruct` checkpoint, which
       is byte-identically the pre-SFT weights each `sft-canonical` adapter was
       trained on (verified 2026-07-17 in the sweep yaml header) — so
       Zero-shot↔SFT is a clean paired contrast, not a cross-checkpoint one.
       No GRPO row: no canonical-set GRPO eval exists yet.
    3. **CIRL column is vignette probing accuracy, not trajectory completeness.**
       These sweeps run `cirl_vignettes` (`compute_metrics` → `accuracy`), which
       is judge-free. The paper's `cirl_trajectory` completeness metric was not
       run here, so this column answers a different question than the paper's
       CIRL Comp. — do not paste it into `benchmark_results.tex` unchanged.
    4. **Judge-gating relaxed to a note.** In the Qwen3.6 notebook judged columns
       were filtered to one judge because the corpus mixed judges. Here every
       candidate run is Gemma-4-judged, so the filter is a no-op assertion — the
       notebook still checks it and will loudly flag any non-Gemma run that
       sneaks into the scan.

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
    | CIRL Acc | `cirl_vignettes/cirl_vignettes` | `compute_metrics` | `accuracy` | no |
    | Q7 | `vlm_geoprivacy/vlm_geoprivacy_bench` | `compute_metrics` | `per_question.Q7.accuracy` | no |
    | MMLU | `mmlu/mmlu` | `compute_metrics` | `overall_accuracy` | no |

    **Known gaps** (surfaced explicitly in the coverage cell, so a blank cell is
    never mistaken for a zero). Resolved as of 2026-07-18 — note that two of
    these are *findings*, not infrastructure failures, and will not be filled by
    re-running:

    *Repaired (2026-07-18 sweeps, now folded in):*

    - **`llama3.1-8b/sft-canonical` GoldCoin** — cancelled mid-run, not a model
      failure. Re-run in `eval_canonical_repair`; filled.
    - **`gpt-oss-20b/sft-canonical`** — every benchmark died at engine init:
      vLLM 0.25.0 routes this MoE model through the fused-MoE LoRA manager and,
      in eager mode, hands the *attention* LoRA shrink kernel a non-contiguous
      tensor (`assert inputs.is_contiguous()` → `EngineDeadError`). Fixed by
      `enforce_eager: false` (isolated 2026-07-18, job 24462). GoldCoin + CIRL
      now land; PrivacyLens does not — see below.
    - **VLM Q7 for gemma-4** — the whole gemma-4 line is any-to-any multimodal
      but `gemma-4` was missing from eval_all's `_VLM_FAMILIES`, so those cells
      silently self-skipped as "text-only". Fixed, and backfilled by
      `eval_gemma4_q7_backfill`. Q7 remains blank for the genuinely text-only
      models (Phi-4, Llama, OpenThinker, GPT-OSS).
    *NOT repairable — reported as findings:*

    - **`harc-llama3.1-8b/instruct` ConfAIde r — 31.6% Tier-2b refusal rate.**
      Two distinct causes, unpicked in that order. First, token truncation: this
      checkpoint opens with a preamble ("I'm just an AI, I don't have personal
      opinions...") and the budget expired mid-sentence — 70% of tier2a, 65% of
      tier2b, 84% of tier3_control `length`-truncated, `parseable_rate=0.0714`.
      Raising those three budgets to 128 lifted tier2a to 0.7653 (passes) and
      tier2b to 0.6837. What that fix revealed is the actual finding: **every**
      remaining unparsed row is an explicit refusal ("I can't provide a rating
      for this scenario") — 23/23 on tier2a, 31/31 on tier2b. More tokens will
      not help; the 9 still-truncated tier2b rows all parsed fine. Cell left
      blank rather than computing r over the 68% it agreed to rate, since the
      declined items are plausibly the more sensitive ones — a self-selected
      subset, not a random one.

    - **`openthinker3-7b/sft-canonical` PrivacyLens** — emits JSON instead of
      the ReAct `Thought:/Action:` format on **493/493** vignettes (instruct:
      401/493 carry `Action:`). Zero parseable actions → zero judge requests.
      Deterministic at temperature 0.
    - **`gpt-oss-20b/sft-canonical` PrivacyLens** — emits an **empty harmony
      final channel on 492/1479 (33.3%)** of QA probes, vs 0.3% for its instruct
      baseline. All 492 have non-empty reasoning: the model reasons to the answer
      and never states it. `finish_reason=stop`, so this is channel discipline,
      not truncation. Do **not** repair this by reading the answer out of the
      reasoning channel — that re-introduces scoring the model's own CoT, a
      harmony bug already fixed once.

    **Q7 caveat.** gemma-4-E2B and E4B collapse to a single class on Q7
    (782/783 and 760/783 of predictions), so their 21.3 / 23.8 are base rates,
    not skill. gemma-4-12B discriminates properly (predictions spread
    401/176/206, Q7=64.2), which is what confirms the collapse is model capacity
    rather than a prompt defect.
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
        / "colm_benchmark_results_gemma4_2026_07"
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    # The Gemma-4-judged canonical-set sweeps. Restricting the scan to these (as
    # opposed to globbing all of multirun/) keeps a stray older eval_all from
    # silently supplying a Qwen3.6-judged cell.
    SWEEP_GLOBS = [
        "*_eval_canonical_instruct/*",
        "*_eval_canonical_sft_gemma4/*",
        # 2026-07-18 repair/extension sweeps. All Gemma-4-judged (or judge-free),
        # so they compose with the originals under the same per-cell latest-wins
        # rule rather than needing a separate table.
        "*_eval_canonical_repair/*",        # llama3.1-8b GoldCoin (cancelled),
                                            # harc-llama3.1-8b ConfAIde (parse fail)
        "*_eval_canonical_gptoss_refix/*",  # gpt-oss SFT, after the enforce_eager fix
        "*_eval_gemma4_q7_backfill/*",      # VLM Q7 for the six gemma-4 cells
        "*_eval_teacher_gemma4_31b/*",      # the teacher/judge as a subject
    ]

    # Judge every judged cell must have come from. Matched as a substring of the
    # served-model path recorded in the judge-batch manifest.
    EXPECTED_JUDGE = "Gemma-4-31B-it"

    # Multiruns whose judge CANNOT be verified from artifacts, but was attested
    # out-of-band. The `*_2gpu` pipelines use `judge.mode: live` with
    # `model_name: default` — the served model is resolved per-request and never
    # written to disk, so there is no `*_judge_batch/manifest.json` to read (the
    # stages are `*_judge_inference/` and carry only results.parquet).
    #
    # Without this, those cells silently vanish from judged columns: the judge
    # resolves to None, fails the EXPECTED_JUDGE test, and gets filtered out
    # looking exactly like "this benchmark never ran".
    #
    # Each entry records HOW the judge was established. Do not add a run here on
    # the strength of its config alone — the whole reason this notebook reads
    # manifests is that config lies (the stale `${oc.env:JUDGE_MODEL,
    # Qwen3.6-27B}` default). Attest only from a live observation.
    JUDGE_ATTESTED_MULTIRUNS = {
        "2026-07-18_eval_teacher_gemma4_31b/10-04-59":
            "curl http://klara.tech.cornell.edu:8002/v1/models immediately "
            "before launch (2026-07-18) returned exactly one served model: "
            "/share/pierson/matt/zoo/models/Gemma-4-31B-it. The sweep pins "
            "JUDGE_SERVER_URL to that host and the server ran uninterrupted "
            "(job 920636) across the run.",
    }

    # Which Tier-2 sub-metric backs the ConfAIde "r" column — Tier-2b Pearson,
    # same as the paper (Tier-2a is a different, much higher number).
    CONFAIDE_R_SUBDIR = "compute_metrics_tier2b"

    # The teacher/judge model, evaluated as a subject. Its label carries the
    # warning inline so the row cannot be copied out of the table without it.
    TEACHER_ROW = "Gemma-4-31B-it (teacher/judge — self-judged)"

    # ── Rows: the canonical 11, in size-then-family order ─────────────────────
    ROW_ORDER = [
        ("Qwen3.5-2B", ["Zero-shot", "SFT"]),
        ("Qwen3.5-4B", ["Zero-shot", "SFT"]),
        ("Qwen3.5-9B", ["Zero-shot", "SFT"]),
        ("Gemma-4-E2B", ["Zero-shot", "SFT"]),
        ("Gemma-4-E4B", ["Zero-shot", "SFT"]),
        ("Gemma-4-12B", ["Zero-shot", "SFT"]),
        ("OpenThinker3-7B", ["Zero-shot", "SFT"]),
        ("Llama-3.1-8B", ["Zero-shot", "SFT"]),
        ("HARC-Llama-3.1-8B", ["Zero-shot", "SFT"]),
        ("Phi-4", ["Zero-shot", "SFT"]),
        ("GPT-OSS-20B", ["Zero-shot", "SFT"]),
        # Reference ceiling, NOT one of the canonical 11 and not a paired
        # Zero-shot/SFT contrast — this is the teacher that generated the SFT
        # data and the judge that scores every PrivacyLens row. See
        # TEACHER_ROW / the self-judging warning below before quoting its
        # judged columns.
        (TEACHER_ROW, ["Reference"]),
    ]

    # ── model= override string  →  (display model, condition) ─────────────────
    # "Zero-shot" = <family>/instruct, verified to be the exact pre-SFT weights
    # for the paired <family>/sft-canonical adapter.
    _FAMILIES = [
        ("qwen3.5-2b", "Qwen3.5-2B"),
        ("qwen3.5-4b", "Qwen3.5-4B"),
        ("qwen3.5-9b", "Qwen3.5-9B"),
        ("gemma-4-e2b", "Gemma-4-E2B"),
        ("gemma-4-e4b", "Gemma-4-E4B"),
        ("gemma-4-12b", "Gemma-4-12B"),
        ("openthinker3-7b", "OpenThinker3-7B"),
        ("llama3.1-8b", "Llama-3.1-8B"),
        ("harc-llama3.1-8b", "HARC-Llama-3.1-8B"),
        ("phi-4", "Phi-4"),
        ("gpt-oss-20b", "GPT-OSS-20B"),
    ]
    OVERRIDE_TO_ROW = {}
    for _slug, _disp in _FAMILIES:
        OVERRIDE_TO_ROW[f"{_slug}/instruct"] = (_disp, "Zero-shot")
        OVERRIDE_TO_ROW[f"{_slug}/sft-canonical"] = (_disp, "SFT")
    # The teacher is deliberately NOT in _FAMILIES: it has no sft-canonical
    # counterpart, and labelling it "Zero-shot" would invite reading it as a
    # twelfth paired row rather than a reference ceiling.
    OVERRIDE_TO_ROW["gemma-4-31b/instruct"] = (TEACHER_ROW, "Reference")

    # Judged columns for THIS row are self-judged (judge == subject), so they
    # are an optimistic bound, not a like-for-like score. Used below to mark the
    # affected cells and to footnote the table.
    SELF_JUDGED_ROWS = {TEACHER_ROW}

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
        # NOTE: vignette probing accuracy (judge-free), NOT the paper's
        # trajectory completeness. Different metric, different question.
        ("CIRL", "Acc", "cirl_vignettes", "cirl_vignettes",
         "compute_metrics", "accuracy", False, False, "pct"),
        ("VLM", "Q7", "vlm_geoprivacy", "vlm_geoprivacy_bench",
         "compute_metrics", "per_question.Q7.accuracy", False, False, "pct"),
        ("MMLU", "Acc", "mmlu", "mmlu",
         "compute_metrics", "overall_accuracy", False, False, "pct"),
    ]
    return (
        COLUMNS,
        EXPECTED_JUDGE,
        JUDGE_ATTESTED_MULTIRUNS,
        MULTIRUN_GLOB_ROOT,
        OVERRIDE_TO_ROW,
        REPORT_DIR,
        ROW_ORDER,
        SELF_JUDGED_ROWS,
        SWEEP_GLOBS,
        TEACHER_ROW,
        dt,
        re,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase A — Scan the canonical-set multiruns

    Walk every `multirun/<date>_eval_canonical_{instruct,sft_gemma4}/<HH-MM-SS>/<idx>/`
    sub-run. For each, read the `model=` override (→ row identity) and the
    **served judge from the judge-batch manifest**, then record one row per
    (sub-run × benchmark column) with the metric value if present.
    """)
    return


@app.cell
def _(COLUMNS, EXPECTED_JUDGE, JUDGE_ATTESTED_MULTIRUNS, MULTIRUN_GLOB_ROOT, OVERRIDE_TO_ROW, SWEEP_GLOBS, dt, re):
    import json as _json

    # Extracts only the <date>/<time> stamp — it must NOT also gate on the sweep
    # name. SWEEP_GLOBS above is the single place that decides which sweeps are
    # in scope; duplicating that filter here silently dropped every sweep whose
    # name was not `_eval_canonical_*` (the q7 backfill and the teacher run were
    # globbed, then discarded by this regex, leaving Q7 blank with no warning).
    _MR_RE = re.compile(r"(\d{4}-\d{2}-\d{2})_\w+/(\d{2}-\d{2}-\d{2})$")

    def _parse_mr_dt(mr_dir):
        m = _MR_RE.search(str(mr_dir))
        if not m:
            return None
        return dt.datetime.strptime(f"{m.group(1)} {m.group(2)}", "%Y-%m-%d %H-%M-%S")

    def _served_judge(sub_dir):
        """The judge that actually served, from the PrivacyLens judge-batch
        manifest (`judge_export.py` writes the model it resolved from
        /v1/models). Deliberately NOT read from multirun.yaml: that file carries
        the stale OmegaConf default `${oc.env:JUDGE_MODEL,Qwen3.6-27B}` and would
        mislabel every one of these runs.

        Returns None when the judge cannot be established from artifacts. That
        happens for the `*_2gpu` pipelines, which run `judge.mode: live` with
        `model_name: default`: the served model is resolved at request time and
        written nowhere, so those runs leave `*_judge_inference/` stages with no
        manifest at all. Callers must treat None as "unverified", NOT as
        "wrong judge" — see JUDGE_ATTESTED_MULTIRUNS.
        """
        for man in sorted(sub_dir.glob(
            "privacylens/privacylens_eval/outputs/*_judge_batch/manifest.json"
        )):
            try:
                m = _json.loads(man.read_text()).get("model")
            except (ValueError, OSError):
                continue
            if m:
                return m.rstrip("/").split("/")[-1]
        return None  # no judged benchmark ran in this sub-run

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
    _mr_dirs = sorted(
        {p for g in SWEEP_GLOBS for p in MULTIRUN_GLOB_ROOT.glob(g)}
    )
    for _mr in _mr_dirs:
        if not _mr.is_dir():
            continue
        _mrdt = _parse_mr_dt(_mr)
        if _mrdt is None:
            continue
        for _sub in sorted(p for p in _mr.iterdir() if p.is_dir() and p.name.isdigit()):
            _ov = _override_model(_sub)
            if _ov is None or _ov not in OVERRIDE_TO_ROW:
                continue
            _model, _cond = OVERRIDE_TO_ROW[_ov]
            _mr_key = str(_mr.relative_to(MULTIRUN_GLOB_ROOT))
            _judge = _served_judge(_sub)
            # Fall back to an explicit out-of-band attestation only when the
            # artifacts genuinely carry no manifest (live-mode pipelines).
            # `judge_src` keeps the distinction visible in the provenance table
            # so an attested cell is never mistaken for a verified one.
            if _judge is None and _mr_key in JUDGE_ATTESTED_MULTIRUNS:
                _judge, _judge_src = EXPECTED_JUDGE, "attested"
            elif _judge is None:
                _judge_src = "unresolved"
            else:
                _judge_src = "manifest"
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
                    "judge": _judge, "judge_src": _judge_src, "mr_dt": _mrdt,
                    "multirun": _mr_key,
                    "override": _ov,
                })

    import pandas as _pd
    scan = _pd.DataFrame(rows)
    print(f"{len(scan)} (sub-run × column) observations across "
          f"{scan['multirun'].nunique()} multiruns")
    print(f"multiruns: {sorted(scan['multirun'].unique())}")
    print(f"judges seen: {sorted(scan['judge'].dropna().unique())}")
    print(f"judge provenance: "
          f"{scan.groupby('judge_src').size().to_dict()}")
    # Judged observations whose judge could be established by NEITHER a manifest
    # nor an attestation get dropped downstream. Say so here — a silent drop is
    # indistinguishable from "that benchmark never ran", which is exactly how
    # the teacher row's PrivacyLens cells went missing on the first pass.
    _unres = scan[scan["judged"] & (scan["judge_src"] == "unresolved")]
    if len(_unres):
        print(f"\n!! {len(_unres)} JUDGED observations have an unresolvable "
              f"judge and will be DROPPED (not a data gap — a provenance gap):")
        for (_m, _c, _mr_), _g in _unres.groupby(["model", "condition", "multirun"]):
            print(f"   {_m} / {_c}  [{_mr_}]  cols={sorted(_g['col'])}")
        print("   Fix: add the multirun to JUDGE_ATTESTED_MULTIRUNS with "
              "evidence, or re-run under a batch-mode pipeline that writes a "
              "judge manifest.")
    scan
    return (scan,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase B — Judge assertion + per-cell latest-run selection

    Every judged cell must come from a `Gemma-4-31B-it` run; the assertion below
    is expected to be a no-op, and exists so a re-run against a differently-judged
    sweep fails loudly instead of silently mixing judges. Then, for each
    (model, condition, column), take the most recent qualifying multirun.
    """)
    return


@app.cell
def _(EXPECTED_JUDGE, scan):
    import pandas as _pd

    # Loud check: any judged observation not from the expected judge is dropped
    # AND reported. With the current corpus this should print nothing.
    _judged = scan[scan["judged"]]
    _bad = _judged[~_judged["judge"].fillna("").str.contains(EXPECTED_JUDGE)]
    if len(_bad):
        print(f"!! {len(_bad)} judged observations NOT judged by {EXPECTED_JUDGE} "
              f"— dropped. Judges: {sorted(_bad['judge'].fillna('<none>').unique())}")
        print(_bad[["model", "condition", "col", "judge", "multirun"]].to_string())
    else:
        print(f"OK: all {len(_judged)} judged observations came from {EXPECTED_JUDGE}")

    _elig = scan[
        (~scan["judged"])
        | (scan["judge"].fillna("").str.contains(EXPECTED_JUDGE))
    ].copy()
    # Latest run per (model, condition, col) — the 07-17 sweeps supersede the
    # partial 07-16 wave cell-by-cell rather than wholesale.
    picked = (
        _elig.sort_values("mr_dt")
        .drop_duplicates(subset=["model", "condition", "col_id"], keep="last")
        .copy()
    )
    print(f"{len(picked)} cells filled")
    picked[["model", "condition", "group", "col", "value", "judge",
            "multirun"]].sort_values(["model", "condition", "col"])
    return (picked,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase B2 — Coverage: which cells are missing, and why

    A blank cell in the table below is never a zero. This grid separates
    "benchmark never ran for this model" from "ran and produced nothing", so the
    cancelled `llama3.1-8b` GoldCoin cell and the failed `gpt-oss-20b` SFT row
    read as gaps rather than results.
    """)
    return


@app.cell
def _(COLUMNS, ROW_ORDER, picked):
    import pandas as _pd

    _have = {
        (r["model"], r["condition"], r["col_id"]) for _, r in picked.iterrows()
    }
    _col_ids = [(f"{c[0]}::{c[1]}", f"{c[0]} {c[1]}") for c in COLUMNS]

    _grid = []
    for (_mdl, _conds) in ROW_ORDER:
        for _cond in _conds:
            _row = {"model": _mdl, "condition": _cond}
            _n = 0
            for _cid, _label in _col_ids:
                _ok = (_mdl, _cond, _cid) in _have
                _row[_label] = "✓" if _ok else "—"
                _n += _ok
            _row["filled"] = f"{_n}/{len(_col_ids)}"
            _grid.append(_row)
    coverage = _pd.DataFrame(_grid)

    _missing = [
        f"{r['model']}/{r['condition']}"
        for r in _grid if r["filled"].startswith("0/")
    ]
    if _missing:
        print(f"Rows with NO data at all: {', '.join(_missing)}")
    coverage
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase C — Assemble the table + markdown

    Lay the cells out in row/column order, format (percentages ×100; Help on the
    0–3 scale), bold the best per column, render, and save to
    `benchmark_results_gemma4.md` plus a provenance parquet.
    """)
    return


@app.cell
def _(COLUMNS, EXPECTED_JUDGE, REPORT_DIR, ROW_ORDER, SELF_JUDGED_ROWS, picked):
    import pandas as _pd

    # Unique column id = "group::col" (GoldCoin "Comp." would otherwise collide
    # with any other "Comp.").
    # specs: (col_id, group, col, judged, lower_better, scale). `judged` is
    # carried so self-judged cells can be marked (see _fmt).
    _col_specs = [(f"{c[0]}::{c[1]}", c[0], c[1], c[6], c[7], c[8]) for c in COLUMNS]

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
                # The teacher is a reference ceiling, not a competitor in the
                # canonical-11 comparison — and on judged columns it grades
                # itself. Excluded from best-per-column so it cannot take a
                # bold away from a fairly-scored row.
                if mdl in SELF_JUDGED_ROWS:
                    continue
                v = _lut.get((mdl, cond, cid))
                if v is None:
                    continue
                out[(mdl, cond)] = v * 100.0 if scale == "pct" else v
        return out

    # Best-per-column, over the canonical rows only (see _scaled).
    _best = {}
    for (cid, grp, col, judged, lo, scale) in _col_specs:
        vals = _scaled(cid, scale)
        if not vals:
            continue
        _best[cid] = (min if lo else max)(vals.values())

    def _fmt(cid, judged, lo, scale, mdl, cond):
        v = _lut.get((mdl, cond, cid))
        if v is None:
            return "—"
        sv = v * 100.0 if scale == "pct" else v
        spec = "{:.1f}" if scale == "pct" else "{:.2f}"
        s = spec.format(sv)
        if cid in _best and abs(sv - _best[cid]) < 1e-9:
            s = f"**{s}**"
        # Dagger every judged cell where the judge IS the subject, so the
        # caveat travels with the number if the row is copied out.
        if judged and mdl in SELF_JUDGED_ROWS:
            s = f"{s}†"
        return s

    _hdr_cols = [f"{grp} {col}" for (cid, grp, col, judged, lo, scale) in _col_specs]

    _lines = []
    _lines.append("| Model | Cond. | " + " | ".join(_hdr_cols) + " |")
    _lines.append("|" + "---|" * (2 + len(_col_specs)))
    _dropped = []
    for (mdl, conds) in ROW_ORDER:
        _shown = 0
        for cond in conds:
            _cells = [_fmt(cid, judged, lo, scale, mdl, cond)
                      for (cid, grp, col, judged, lo, scale) in _col_specs]
            # Drop (model, condition) rows with no data at all rather than
            # emitting an all-"—" row sourced from nothing.
            if all(c == "—" for c in _cells):
                _dropped.append(f"{mdl} / {cond}")
                continue
            _mcell = mdl if _shown == 0 else ""
            _lines.append(f"| {_mcell} | {cond} | " + " | ".join(_cells) + " |")
            _shown += 1
    table_md = "\n".join(_lines)
    if _dropped:
        print("Dropped (no data for any column): " + ", ".join(_dropped))

    _legend = (
        "\n\n*Percentages (×100) except **Help** (mean helpfulness, 0–3). "
        "↓ = lower is better. Best per column in **bold**. "
        "GoldCoin Appl./Comp. = applicability/compliance macro-F1. "
        "PrivacyLens: QA Acc, Lk = leakage rate, Adj Lk = adjusted leakage, "
        "Helpful = helpful rate, Help = mean helpfulness. ConfAIde r = Tier-2b "
        "Pearson. **CIRL Acc = vignette probing accuracy (judge-free), NOT the "
        "paper's trajectory completeness.** VLM Q7 = location-granularity "
        "accuracy; the Qwen3.5 and gemma-4 families ran it, the rest are "
        "text-only. "
        f"All judged columns (PrivacyLens Lk/Adj Lk/Helpful/Help) use "
        f"{EXPECTED_JUDGE} — not comparable with the pre-07-16 Qwen3.6-27B "
        "PrivacyLens column. Zero-shot = the pre-SFT `<family>/instruct` "
        "checkpoint each SFT adapter was trained from. Most recent run per cell.*"
        "\n\n*† = **self-judged**: on that row the judge and the subject are "
        "the same weights, so the value is an optimistic bound rather than a "
        "like-for-like score. The teacher row is also excluded from "
        "best-per-column bolding — it is a reference ceiling (and a larger "
        "model), not a competitor in the canonical-11 comparison. Its "
        "gold-scored columns (GoldCoin, QA Acc, ConfAIde r, CIRL, Q7, MMLU) "
        "carry no self-judging concern.*"
    )
    _dropped_note = (
        f"\n\n*Rows omitted (no run produced any metric): "
        f"{', '.join(_dropped)}.*" if _dropped else ""
    )
    full_md = (
        f"## COLM benchmark results — canonical set, {EXPECTED_JUDGE}-judged "
        "(2026-07)\n\n"
        + table_md + _legend + _dropped_note + "\n"
    )

    (REPORT_DIR / "benchmark_results_gemma4.md").write_text(full_md)

    # Provenance: judge + source multirun per filled cell.
    _prov_rows = []
    for (mdl, conds) in ROW_ORDER:
        for cond in conds:
            for (cid, grp, col, judged, lo, scale) in _col_specs:
                if (mdl, cond, cid) in _prov:
                    j, mr = _prov[(mdl, cond, cid)]
                    _prov_rows.append({
                        "model": mdl, "condition": cond,
                        "column": f"{grp} {col}",
                        "judge": j, "multirun": mr,
                        "value": _lut[(mdl, cond, cid)],
                        # Explicit column so a downstream consumer of the
                        # parquet cannot miss what the table's † encodes.
                        "self_judged": bool(judged and mdl in SELF_JUDGED_ROWS),
                    })
    provenance = _pd.DataFrame(_prov_rows)
    provenance.to_parquet(
        REPORT_DIR / "benchmark_results_provenance.parquet", index=False
    )
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
    ## Phase D — SFT effect (paired Zero-shot → SFT delta)

    The reason these two sweeps are row-aligned: same 11 checkpoints, same
    benchmarks, same judge, LoRA on/off. Delta is signed so **positive = SFT
    better** on every column (leakage columns are negated, since lower is
    better there). Only cells present in *both* conditions are differenced.
    """)
    return


@app.cell
def _(COLUMNS, REPORT_DIR, ROW_ORDER, picked):
    import pandas as _pd

    _lut2 = {
        (r["model"], r["condition"], r["col_id"]): r["value"]
        for _, r in picked.iterrows()
    }
    _specs = [(f"{c[0]}::{c[1]}", f"{c[0]} {c[1]}", c[7], c[8]) for c in COLUMNS]

    _drows = []
    for (_mdl2, _conds2) in ROW_ORDER:
        _row2 = {"model": _mdl2}
        for _cid2, _label2, _lo2, _scale2 in _specs:
            _z = _lut2.get((_mdl2, "Zero-shot", _cid2))
            _s = _lut2.get((_mdl2, "SFT", _cid2))
            if _z is None or _s is None:
                _row2[_label2] = None
                continue
            _mult = 100.0 if _scale2 == "pct" else 1.0
            _d = (_s - _z) * _mult
            # Negate so positive always means "SFT is better".
            _row2[_label2] = -_d if _lo2 else _d
        _drows.append(_row2)
    sft_delta = _pd.DataFrame(_drows).set_index("model")

    print("Paired Zero-shot → SFT delta (positive = SFT better; "
          "leakage columns sign-flipped). Blank = one side missing.")
    print(sft_delta.round(1).to_string())
    print("\nMean delta per column (over models with both conditions):")
    print(sft_delta.mean(axis=0).round(2).to_string())
    sft_delta.to_parquet(REPORT_DIR / "sft_delta.parquet")
    sft_delta.round(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Provenance — which run & judge backs each cell
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
      `tables/colm_benchmark_results_gemma4_2026_07/benchmark_results_gemma4.md`)
      is the canonical-set results table under the Gemma-4-31B-it judge.
    - **Before quoting any PrivacyLens number**, note the judge changed on
      2026-07-16. These rows are internally comparable (Zero-shot vs SFT, model
      vs model) but *not* comparable with the Qwen3.6-judged column in
      `benchmark_results.tex`. Also note the judge (Gemma-4-31B-it) is the same
      model that generated the SFT training data — an accepted coupling of the
      gemma consolidation, recorded in the sweep yaml, not a bug.
    - **CIRL Acc ≠ the paper's CIRL Comp.** Vignette probing accuracy here;
      trajectory completeness there. Do not swap one for the other.
    - **Blank cells** are gaps, not zeros — see the coverage grid. To close the
      known ones: re-run `llama3.1-8b/sft-canonical` GoldCoin (cancelled), and
      diagnose `gpt-oss-20b/sft-canonical` (all benchmarks failed at dispatch;
      its `instruct` counterpart ran clean, so suspect the adapter, not the
      base checkpoint).
    - To switch the ConfAIde *r* sub-metric (Tier-2a ↔ Tier-2b) edit
      `CONFAIDE_R_SUBDIR`. To add a GRPO row, run a canonical-set eval on the
      GRPO checkpoint and add its `model=` override to `OVERRIDE_TO_ROW`.
    """)
    return


if __name__ == "__main__":
    app.run()
