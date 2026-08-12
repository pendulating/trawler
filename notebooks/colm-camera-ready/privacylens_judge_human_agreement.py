import marimo

__generated_with = "0.23.14"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # PrivacyLens leakage judge vs the human expert — camera-ready

    Built 2026-08-08 for the COLM 2026 camera-ready. Replaces Phase E of
    `notebooks/normative-simulacra/judge_validation_rebuttal_2026_05_30.py` as
    the source of `\autoref{tab:judge-human-agreement}`
    (`B_additional-results.tex`, `app:judge-validation`). That notebook does nine
    other things; this one does only the human calibration, so the table in the
    paper has a single short file behind it.

    ## What changes versus the printed table

    1. **n = 36 → the full annotated set.** The paper's table was computed off
       the 2026-06-01 annotation CSV, which had rows 0–35 marked. The expert
       went on to finish rows 36–50, so the set on disk is **51 records** and the
       printed $n{=}36$ is a stale subset of it. Every number moves.
    2. **The adopted judge is Gemma-4-31B-it, not Qwen3.6-27B.** Production
       switched on 2026-07-16 (`scripts/judge_server.sub` defaults to
       Gemma-4-31B-it; every judge-server launch since serves it), so the
       `(adopted)` marker moves. Both are still in the table — the point of the
       ablation is that they agree.
    3. **Small-n is answered instead of conceded.** The appendix currently
       concedes "the expert set is small". Two additions make that honest rather
       than fatal: a percentile **bootstrap CI on $\kappa$** (10k resamples over
       records), and an **exact McNemar test** against the incumbent on the same
       51 records. The paired test is the one that matters — all four judges
       score the *same* records, so the comparison need not go through κ's
       standard error.

    The direction of the finding does not change: the incumbent
    Qwen3-32B-AWQ is lenient and misses real leaks, and the three consensus
    judges track the expert.

    ## Provenance

    | artifact | path |
    |---|---|
    | expert annotations | `annotation_runs/annot_privacylens_agent_action_inference_seed777_n100_50annotated.csv` |
    | row → record map | `privacylens_audit_n100_seed777.html` (`sampled_indices`, seed 777, n=100) |
    | Qwen3-32B-AWQ verdicts | `multirun/2026-03-30_eval_all/22-41-52/{2,3,4}` |
    | gpt-5.2 verdicts | same Mar30 actions, offline OpenAI-Batch rejudge (`judge_batches/*/results_gpt52.parquet`) |
    | Gemma-4-31B-it verdicts | `multirun/2026-04-20_eval_all/18-15-21/4` |
    | Qwen3.6-27B verdicts | `multirun/2026-04-24_eval_all/10-13-47/4` |

    Judge identity for each sweep was established by matching the port in
    `privacylens_eval.log` to the preceding `.slurm_jobs/judge-server/*.out`
    launch header; that audit lives in the 05-30 notebook and is not repeated.
    Sub-runs 2/3/4 of the Mar30 multirun are Base/SFT/GRPO, and the expert
    annotated all three; the paper table (and this one) reports **Base**, the
    scope where all four judges have coverage.

    ## The one caveat this table cannot remove

    Qwen3-32B-AWQ and gpt-5.2 scored the **byte-identical Mar30 actions** the
    expert read. Gemma-4 and Qwen3.6 judged their own April re-runs of the same
    model and prompt, and re-sampling means only **30 of 51** of their actions
    are byte-identical to the annotated one. For the other 21 the judge is
    scoring a *different* action than the expert graded, so disagreement there
    mixes action variance into judge variance.

    Both scopes are therefore computed: `Base (all)` is the headline (it is the
    full annotated set, and it is what the paper's $n$ should say), and
    `Base (byte-identical)` is the clean-but-smaller check. They agree —
    Gemma-4 κ 0.648 → 0.661, Qwen3.6 0.606 → 0.595 — which is why the headline
    is reportable. Rejudging the 51 Mar30 actions with the current
    Gemma-4 judge server would close the gap properly; until then the caption
    carries the disclosure.

    Tables → `tables/judge_human_agreement/`.
    """)
    return


@app.cell
def _():
    import json
    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd

    PROJECT_ROOT = Path("/share/pierson/matt/UAIR")
    NB_DIR = PROJECT_ROOT / "notebooks/colm-camera-ready"
    TAB_DIR = NB_DIR / "tables/judge_human_agreement"
    TAB_DIR.mkdir(parents=True, exist_ok=True)
    PAPER_TAB_DIR = PROJECT_ROOT / "papers/colm26_normative-simulacra/tables"

    ANNOT_CSV = PROJECT_ROOT / (
        "annotation_runs/"
        "annot_privacylens_agent_action_inference_seed777_n100_50annotated.csv"
    )
    AUDIT_HTML = PROJECT_ROOT / "privacylens_audit_n100_seed777.html"

    # The Mar30 multirun the expert read, and its Base/SFT/GRPO sub-runs.
    MAR30 = PROJECT_ROOT / "multirun/2026-03-30_eval_all/22-41-52"
    SUBRUN = {"Base": 2, "SFT": 3, "GRPO": 4}
    # The April re-runs, Base only — these are the two judges whose actions can
    # drift from what the expert saw.
    APR = {
        "Gemma-4-31B-it": PROJECT_ROOT / "multirun/2026-04-20_eval_all/18-15-21/4",
        "Qwen3.6-27B": PROJECT_ROOT / "multirun/2026-04-24_eval_all/10-13-47/4",
    }

    # Table order follows the printed table; the `(adopted)` marker is the one
    # thing that moved (Gemma-4 since 2026-07-16).
    JUDGE_ORDER = ["Qwen3-32B-AWQ", "Gemma-4-31B-it", "Qwen3.6-27B", "gpt-5.2"]
    JUDGE_LABEL = {
        "Qwen3-32B-AWQ": "Qwen3-32B-AWQ (original)",
        "Gemma-4-31B-it": "Gemma-4-31B-it (adopted)",
        "Qwen3.6-27B": "Qwen3.6-27B",
        "gpt-5.2": "gpt-5.2",
    }
    INCUMBENT = "Qwen3-32B-AWQ"

    SCOPE_ALL = "Base (all)"
    SCOPE_IDENT = "Base (byte-identical)"
    SCOPE_POOLED = "Pooled Base+SFT+GRPO"

    N_BOOT = 10_000
    BOOT_SEED = 20260808  # fixed so the printed CIs are reproducible

    def save_table(df, name, index=False):
        out = TAB_DIR / f"{name}.csv"
        df.to_csv(out, index=index)
        print(f"[table] {out}")
        return df

    def save_tex(body, name, also_paper=False):
        """Write a complete `\\input`-able LaTeX table.

        `also_paper` off by default: running this notebook should never edit the
        paper tree as a side effect. Flip it when the table is meant to land.
        """
        out = TAB_DIR / f"{name}.tex"
        out.write_text(body.rstrip("\n") + "\n")
        print(f"[latex] {out}")
        if also_paper:
            (PAPER_TAB_DIR / f"{name}.tex").write_text(body.rstrip("\n") + "\n")
            print(f"[paper] {PAPER_TAB_DIR / name}.tex")

    return (
        ANNOT_CSV,
        APR,
        AUDIT_HTML,
        BOOT_SEED,
        INCUMBENT,
        JUDGE_LABEL,
        JUDGE_ORDER,
        MAR30,
        N_BOOT,
        SCOPE_ALL,
        SCOPE_IDENT,
        SCOPE_POOLED,
        SUBRUN,
        json,
        np,
        pd,
        re,
        save_table,
        save_tex,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Expert annotations

    The annotation CSV is keyed by `row_idx` — the position within the audit's
    100-record sample, not the PrivacyLens record id. The audit HTML carries the
    `sampled_indices` list it was built from, so the join key is recovered from
    the artifact rather than reconstructed from a seed.

    Vocabulary is the judges' own: leakage ∈ {`leak`, `no-leak`}, helpfulness ∈
    {`helpful`, `not-helpful`}, one verdict per
    (record × {Base, SFT, GRPO} × {leakage, helpfulness}) cell.
    """)
    return


@app.cell
def _(ANNOT_CSV, AUDIT_HTML, json, pd, re):
    _sampled = json.loads(
        re.search(r'"sampled_indices":(\[[0-9,\s]+\])',
                  AUDIT_HTML.read_text(errors="ignore")).group(1)
    )

    _csv = pd.read_csv(ANNOT_CSV)
    annot = _csv[_csv["expert_verdict"].notna()].copy()
    annot["record_id"] = annot["row_idx"].map(lambda i: str(_sampled[int(i)]))
    annot["judge_type"] = annot["judge"].map(
        {"Leakage Judge": "leakage", "Helpfulness Judge": "helpfulness"})

    # Counts are derived, never hardcoded, so the table tracks the CSV if the
    # expert annotates further. The assertions pin what is on disk today: rows
    # 0-50 complete for leakage on all three models, one helpfulness cell on
    # GRPO left blank.
    N_RECORDS = int(annot["record_id"].nunique())
    N_BASE_LEAK = int(annot.query("model == 'Base' and judge_type == 'leakage'")
                      ["record_id"].nunique())
    assert N_RECORDS == 51, N_RECORDS
    assert N_BASE_LEAK == N_RECORDS, (N_BASE_LEAK, N_RECORDS)
    assert set(annot["judge_type"]) == {"leakage", "helpfulness"}
    assert annot["row_idx"].max() == 50, annot["row_idx"].max()

    _leak = annot.query("judge_type == 'leakage' and model == 'Base'")
    N_EXPERT_LEAKS = int((_leak["expert_verdict"] == "leak").sum())
    EXPERT_LEAK_RATE = N_EXPERT_LEAKS / N_BASE_LEAK

    print(f"annotated cells {len(annot)} over {N_RECORDS} records "
          f"x {annot['model'].nunique()} models x 2 judge types")
    print(annot.groupby(["judge_type", "model"]).size().to_string())
    print(f"\nexpert leak rate on Base: {EXPERT_LEAK_RATE:.3f} "
          f"({N_EXPERT_LEAKS}/{N_BASE_LEAK})")
    return (
        EXPERT_LEAK_RATE,
        N_BASE_LEAK,
        N_EXPERT_LEAKS,
        N_RECORDS,
        annot,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Judge verdicts, aligned to the annotated records

    One row per (record × model × judge type × judge LLM). `action_identical`
    marks whether the action that judge scored is byte-identical to the one the
    expert read; it can only be false for the two April judges.

    Two checks run as assertions:

    1. The audit's own `judge_verdict` column **is** the Qwen3-32B-AWQ verdict,
       so re-reading it from the Mar30 parquet must reproduce the CSV exactly.
       Anything below 1.000 means the record join is wrong and no number here is
       meaningful.
    2. No annotated record may have empty judge text or a failed parse. The
       2026-08-07 PrivacyLens judge audit found fabricated verdicts and silent
       judge truncation elsewhere in this codebase; these 51 records are clean,
       and this asserts it rather than assuming it.
    """)
    return


@app.cell
def _(APR, JUDGE_ORDER, MAR30, SUBRUN, annot, pd):
    # 305 annotated cells x 4 judges means the loop below asks for the same
    # parquet ~100 times; memoised on (judge, model, judge_type), which is the
    # whole key, so one read per file.
    _CACHE = {}

    def _verdicts(judge, model, judge_type):
        """Per-record verdict frame for one judge × model × judge type."""
        key = (judge, model, judge_type)
        if key in _CACHE:
            return _CACHE[key]
        _CACHE[key] = _out = _read_verdicts(judge, model, judge_type)
        return _out

    def _read_verdicts(judge, model, judge_type):
        stem = ("leakage_judge_inference" if judge_type == "leakage"
                else "helpfulness_judge_inference")
        if judge == "Qwen3-32B-AWQ":
            path = (MAR30 / str(SUBRUN[model])
                    / "privacylens/privacylens_eval/outputs" / stem
                    / "results.parquet")
        elif judge == "gpt-5.2":
            batch = ("leakage_judge_batch" if judge_type == "leakage"
                     else "helpfulness_judge_batch")
            path = (MAR30 / str(SUBRUN[model])
                    / "privacylens/privacylens_eval/outputs/judge_batches"
                    / batch / "results_gpt52.parquet")
        elif model == "Base":
            path = (APR[judge] / "privacylens/privacylens_eval/outputs" / stem
                    / "results.parquet")
        else:
            return None  # the April sweeps only re-ran Base
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df["record_id"] = df["record_id"].astype(str)
        if judge_type == "leakage":
            df["verdict"] = df["leak_flag"].map(
                lambda x: "leak" if x else "no-leak")
            df["judge_text"] = df["leak_judge_text"]
        else:
            df["verdict"] = df["helpfulness_binary"].map(
                lambda x: "helpful" if x else "not-helpful")
            df["judge_text"] = df.get("helpfulness_judge_text")
        keep = ["record_id", "verdict", "generated_action", "judge_text"]
        for extra in ("parse_status", "leakage_judged"):
            if extra in df.columns:
                keep.append(extra)
        return df[keep]

    # The actions the expert actually read, for the drift check.
    _expert_action = {
        m: _verdicts("Qwen3-32B-AWQ", m, "leakage")
        .set_index("record_id")["generated_action"].to_dict()
        for m in SUBRUN
    }

    _rows = []
    for _, _a in annot.iterrows():
        for _J in JUDGE_ORDER:
            _v = _verdicts(_J, _a["model"], _a["judge_type"])
            if _v is None:
                continue
            _v = _v.set_index("record_id")
            if _a["record_id"] not in _v.index:
                continue
            _r = _v.loc[_a["record_id"]]
            _rows.append({
                "record_id": _a["record_id"],
                "model": _a["model"],
                "judge_type": _a["judge_type"],
                "judge_llm": _J,
                "expert": _a["expert_verdict"],
                "judge_verdict": _r["verdict"],
                "agree": _r["verdict"] == _a["expert_verdict"],
                "action_identical": (
                    True if _J not in APR
                    else str(_r["generated_action"])
                    == str(_expert_action[_a["model"]].get(_a["record_id"]))
                ),
                "judge_text_empty": (
                    _r["judge_text"] is None
                    or str(_r["judge_text"]).strip() == ""
                ),
                "parse_status": _r.get("parse_status", "n/a"),
            })
    align = pd.DataFrame(_rows)

    # Check 1 — our Mar30 verdict must reproduce the CSV's judge_verdict column.
    _q32 = align.query("judge_llm == 'Qwen3-32B-AWQ'").merge(
        annot[["record_id", "model", "judge_type", "judge_verdict"]],
        on=["record_id", "model", "judge_type"], how="inner",
        suffixes=("", "_csv"))
    _repro = float((_q32["judge_verdict"] == _q32["judge_verdict_csv"]).mean())
    assert _repro == 1.0, f"Mar30 parquet disagrees with the audit CSV: {_repro:.3f}"

    # Check 2 — no fabricated / truncated judge output among annotated records.
    assert not align["judge_text_empty"].any(), (
        align.loc[align["judge_text_empty"], ["judge_llm", "record_id"]])
    _bad_parse = align.query("parse_status not in ['n/a', 'parsed']")
    assert _bad_parse.empty, _bad_parse

    _apr_leak = align[align["judge_llm"].isin(APR)
                      & (align["judge_type"] == "leakage")]
    DRIFT = (_apr_leak
             .drop_duplicates(["judge_llm", "record_id", "model"])
             .groupby("judge_llm")["action_identical"]
             .agg(["sum", "size"]))
    N_IDENTICAL = int(DRIFT["sum"].min())

    print(f"aligned rows {len(align)} | "
          f"{align['record_id'].nunique()} records x "
          f"{align['judge_llm'].nunique()} judges")
    print(f"check: Mar30 parquet == audit CSV verdicts {_repro:.3f}")
    print("check: empty judge text 0, bad parse 0")
    print("\naction byte-identical to the annotated action (April judges):")
    print(DRIFT.to_string())
    return DRIFT, N_IDENTICAL, align


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Agreement metrics

    Per judge: raw accuracy, Cohen's $\kappa$ with a percentile bootstrap CI
    over records, and — for leakage — precision/recall/F1 on the safety-relevant
    `leak` class plus the two error counts that say *which way* a judge is
    wrong. `missed` is the count of expert-confirmed leaks the judge called
    `no-leak`; that is the failure mode that would flatter a privacy method.

    The bootstrap resamples **records**, which is the sampling unit, and
    discards resamples where either rater is constant (κ undefined). The
    fraction discarded is reported so a CI computed off a thin effective sample
    cannot pass unnoticed.
    """)
    return


@app.cell
def _(BOOT_SEED, N_BOOT, np):
    from sklearn.metrics import cohen_kappa_score, precision_recall_fscore_support

    def _binary_code(expert, judge):
        """Code both raters against one reference level; return (code, n).

        For binary labelings only the 2x2 pattern determines κ, so which level
        is called 1 is irrelevant as long as both raters use the same one.
        `code = 2*e + j` in 0..3 indexes that 2x2 table.
        """
        e_raw, j_raw = np.asarray(expert), np.asarray(judge)
        levels = set(e_raw.tolist()) | set(j_raw.tolist())
        assert len(levels) <= 2, f"kappa fast path is binary-only, got {levels}"
        ref = e_raw[0]
        return (2 * (e_raw == ref).astype(np.int64)
                + (j_raw == ref).astype(np.int64)), len(e_raw)

    def _kappa_from_counts(cnt, n):
        """κ from 2x2 counts laid out as [c00, c01, c10, c11] along the last axis.

        Returns (kappa, defined_mask). A resample in which either rater is
        constant leaves κ undefined-as-agreement (one margin is degenerate) and
        is masked out rather than scored as 0, which would drag every interval
        toward zero.
        """
        c00, c01, c10, c11 = (cnt[..., k] for k in range(4))
        po = (c00 + c11) / n
        pe = ((c00 + c01) * (c00 + c10) + (c10 + c11) * (c01 + c11)) / n ** 2
        ok = (((c00 + c01) > 0) & ((c10 + c11) > 0)
              & ((c00 + c10) > 0) & ((c01 + c11) > 0))
        with np.errstate(invalid="ignore", divide="ignore"):
            kap = (po - pe) / (1 - pe)
        return kap, ok

    def kappa_ci(expert, judge, n_boot=N_BOOT, seed=BOOT_SEED):
        """Percentile bootstrap CI for Cohen's κ, resampling records.

        κ comes from the closed form above rather than 10,000
        `cohen_kappa_score` calls per row of the table; the two agree exactly on
        binary data (self-tested below) and the vectorised version turns minutes
        into milliseconds. The retained-resample fraction is returned so a thin
        effective sample is visible rather than implied.
        """
        code, n = _binary_code(expert, judge)
        rng = np.random.default_rng(seed)
        idx = rng.integers(0, n, size=(n_boot, n))
        drawn = code[idx]
        cnt = np.stack([(drawn == k).sum(1) for k in range(4)], axis=1)
        kap, ok = _kappa_from_counts(cnt, n)
        draws = kap[ok]
        if not len(draws):
            return float("nan"), float("nan"), 0.0
        lo, hi = np.percentile(draws, [2.5, 97.5])
        return float(lo), float(hi), len(draws) / n_boot

    def kappa_fast(expert, judge):
        code, n = _binary_code(expert, judge)
        cnt = np.array([(code == k).sum() for k in range(4)])
        kap, ok = _kappa_from_counts(cnt, n)
        return float(kap) if ok else float("nan")

    # Self-test: the closed form must reproduce sklearn on random binary data,
    # including the lopsided-margin cases the real tables contain.
    _rng = np.random.default_rng(0)
    for _p in (0.5, 0.75, 0.95):
        _e = _rng.choice(["leak", "no-leak"], 60, p=[_p, 1 - _p])
        _j = np.where(_rng.random(60) < 0.8, _e,
                      np.where(_e == "leak", "no-leak", "leak"))
        if len(set(_e)) > 1 and len(set(_j)) > 1:
            assert abs(kappa_fast(_e, _j) - cohen_kappa_score(_e, _j)) < 1e-12, _p

    def agreement(df, judge_type):
        e = df["expert"].tolist()
        j = df["judge_verdict"].tolist()
        if not e:
            return None
        out = {
            "n": len(e),
            "accuracy": sum(a == b for a, b in zip(e, j)) / len(e),
            "cohen_kappa": (cohen_kappa_score(e, j)
                            if len(set(e)) > 1 and len(set(j)) > 1
                            else float("nan")),
        }
        lo, hi, frac = kappa_ci(e, j)
        out.update(kappa_lo=lo, kappa_hi=hi, boot_valid_frac=frac)
        if judge_type == "leakage":
            p, r, f, _ = precision_recall_fscore_support(
                e, j, labels=["leak"], average=None, zero_division=0)
            out.update(
                leak_precision=float(p[0]),
                leak_recall=float(r[0]),
                leak_f1=float(f[0]),
                missed_leaks=int(sum(a == "leak" and b == "no-leak"
                                     for a, b in zip(e, j))),
                false_alarms=int(sum(a == "no-leak" and b == "leak"
                                     for a, b in zip(e, j))),
            )
        return out

    return agreement, cohen_kappa_score, kappa_ci, kappa_fast


@app.cell
def _(
    JUDGE_ORDER,
    SCOPE_ALL,
    SCOPE_IDENT,
    SCOPE_POOLED,
    agreement,
    align,
    pd,
    save_table,
):
    # Pooled is reported only for the two judges with Base+SFT+GRPO coverage;
    # including the April judges there would silently mean "Base only" under a
    # label that says otherwise.
    # Boolean masks rather than `.query("... @var ...")` throughout: pandas
    # resolves `@name` by frame inspection and cannot see marimo's cell-local
    # underscore names, which fails at runtime rather than at parse time.
    _FULL_COVERAGE = ["Qwen3-32B-AWQ", "gpt-5.2"]
    _SCOPES = [
        (SCOPE_ALL, lambda d: d[d["model"] == "Base"], JUDGE_ORDER),
        (SCOPE_IDENT,
         lambda d: d[(d["model"] == "Base") & d["action_identical"]], JUDGE_ORDER),
        (SCOPE_POOLED, lambda d: d, _FULL_COVERAGE),
    ]

    _rows = []
    for _scope, _sel, _judges in _SCOPES:
        _d = _sel(align)
        for _jt in ("leakage", "helpfulness"):
            for _J in _judges:
                _m = agreement(
                    _d[(_d["judge_type"] == _jt) & (_d["judge_llm"] == _J)], _jt)
                if _m:
                    _rows.append({"scope": _scope, "judge_type": _jt,
                                  "judge_llm": _J, **_m})
    agree_table = pd.DataFrame(_rows)
    save_table(agree_table, "judge_human_agreement")

    _cols = ["judge_llm", "n", "accuracy", "cohen_kappa", "kappa_lo", "kappa_hi",
             "leak_precision", "leak_recall", "leak_f1", "missed_leaks",
             "false_alarms"]
    _is_leak = agree_table["judge_type"] == "leakage"
    for _s in (SCOPE_ALL, SCOPE_IDENT, SCOPE_POOLED):
        print(f"\n=== {_s} — leakage")
        print(agree_table[_is_leak & (agree_table["scope"] == _s)]
              [_cols].round(3).to_string(index=False))
    print("\n=== helpfulness (all scopes)")
    print(agree_table[~_is_leak]
          [["scope", "judge_llm", "n", "accuracy", "cohen_kappa",
            "kappa_lo", "kappa_hi"]].round(3).to_string(index=False))
    agree_table
    return (agree_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The paired test

    Every judge scored the same records, so "is this judge closer to the expert
    than the incumbent?" is a paired question and does not have to be answered
    through κ's standard error — which is what makes $n{=}51$ enough to say
    something. McNemar's exact test on the discordant pairs: $b$ = records where
    the challenger agrees with the expert and the incumbent does not, $c$ = the
    reverse.

    ### Why not a t-test

    A paired t-test on the per-record agreement indicators is not wrong — it
    estimates the same quantity, since the mean paired difference *is* $(b-c)/n$
    — but it is a normal approximation to this test, and here it is the
    friendlier one:

    | judge vs incumbent | $b$/$c$ | McNemar exact | paired $t$ | Wilcoxon |
    |---|---|---|---|---|
    | Gemma-4-31B-it | 16/4 | 0.012 | 0.006 | 0.007 |
    | Qwen3.6-27B | 15/4 | 0.019 | 0.010 | 0.012 |
    | gpt-5.2 | 15/2 | 0.002 | 0.001 | 0.002 |

    The paired differences take three values ($-1$, $0$, $+1$) and **31 of 51
    are exactly 0**; a t-test asserts those are approximately normal, while the
    exact binomial on the 20 informative records assumes nothing. McNemar is
    also the conservative column, and choosing the kinder approximation when the
    exact test is available is the kind of thing a reviewer notices. It is
    additionally the standard test for comparing two classifiers on one test set
    (Dietterich 1998), so it is not an exotic choice.

    If a reader wants an effect size rather than a $p$, the paired accuracy gain
    over the incumbent with a bootstrap CI is +0.24 [+0.08, +0.39] (Gemma-4),
    +0.22 [+0.06, +0.37] (Qwen3.6), +0.26 [+0.12, +0.39] (gpt-5.2). Note the
    same interval on **Δκ** includes zero for Qwen3.6 (+0.26 [−0.02, +0.52]), so
    Δaccuracy is the defensible version of that framing and Δκ is not.
    """)
    return


@app.cell
def _(INCUMBENT, JUDGE_ORDER, align, pd, save_table):
    from scipy.stats import binomtest

    def mcnemar(df):
        """Exact McNemar of each judge vs the incumbent on expert agreement."""
        wide = df.pivot_table(index="record_id", columns="judge_llm",
                              values="agree", aggfunc="first").astype(bool)
        rows = []
        for J in JUDGE_ORDER:
            if J == INCUMBENT or J not in wide.columns:
                continue
            b = int((wide[J] & ~wide[INCUMBENT]).sum())
            c = int((~wide[J] & wide[INCUMBENT]).sum())
            rows.append({
                "judge_llm": J,
                "n_records": int(len(wide)),
                "challenger_right_incumbent_wrong": b,
                "incumbent_right_challenger_wrong": c,
                "mcnemar_p": (binomtest(b, b + c, 0.5).pvalue
                              if b + c else float("nan")),
            })
        return pd.DataFrame(rows)

    mcnemar_table = mcnemar(align[(align["model"] == "Base")
                                  & (align["judge_type"] == "leakage")])
    save_table(mcnemar_table, "judge_human_mcnemar")
    print(mcnemar_table.round(4).to_string(index=False))
    mcnemar_table
    return (mcnemar_table,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The paper table

    Same shape as the printed `tab:judge-human-agreement`, with two columns
    added (bootstrap CI on $\kappa$, McNemar $p$ against the incumbent) and the
    drift disclosure attached to the two April judges' $n$.
    """)
    return


@app.cell
def _(
    DRIFT,
    EXPERT_LEAK_RATE,
    INCUMBENT,
    JUDGE_LABEL,
    JUDGE_ORDER,
    N_EXPERT_LEAKS,
    N_IDENTICAL,
    N_RECORDS,
    SCOPE_ALL,
    SCOPE_IDENT,
    agree_table,
    mcnemar_table,
    save_tex,
):
    _leak = (agree_table[(agree_table["scope"] == SCOPE_ALL)
                         & (agree_table["judge_type"] == "leakage")]
             .set_index("judge_llm"))
    _mc = mcnemar_table.set_index("judge_llm")
    _drifted = {j for j in DRIFT.index if DRIFT.loc[j, "sum"] < DRIFT.loc[j, "size"]}

    _lines = []
    for _J in JUDGE_ORDER:
        _r = _leak.loc[_J]
        _n = f"{int(_r['n'])}$^{{\\dagger}}$" if _J in _drifted else f"{int(_r['n'])}"
        _p = ("---" if _J == INCUMBENT
              else f"{_mc.loc[_J, 'mcnemar_p']:.3f}")
        # Two decimals: at n=51 the third is noise — the bootstrap CIs span
        # 0.4-0.5 of a kappa unit — and printing it costs the width the CI
        # column needs. The CSV keeps full precision.
        _lines.append(
            f"{JUDGE_LABEL[_J]} & {_n} & {_r['accuracy']:.2f} & "
            f"{_r['cohen_kappa']:.2f} & "
            f"[{_r['kappa_lo']:.2f}, {_r['kappa_hi']:.2f}] & "
            f"{_r['leak_precision']:.2f} & {_r['leak_recall']:.2f} & "
            f"{_r['leak_f1']:.2f} & {int(_r['missed_leaks'])} & {_p} \\\\"
        )

    _inc = _leak.loc[INCUMBENT]
    _best = _leak.drop(index=INCUMBENT)["cohen_kappa"]
    # How far the drift-restricted scope moves the two April judges' kappa —
    # derived, so the caption's claim cannot drift from the table it cites.
    _ident = (agree_table[(agree_table["scope"] == SCOPE_IDENT)
                          & (agree_table["judge_type"] == "leakage")]
              .set_index("judge_llm")["cohen_kappa"])
    _kappa_shift = max(abs(_ident[_j] - _leak.loc[_j, "cohen_kappa"])
                       for _j in _drifted)
    # Derived, not asserted in prose: the "entirely lenient" reading holds only
    # while the incumbent raises no false alarms, so the sentence tracks the data.
    _fa = int(_inc["false_alarms"])
    _lenient = ("zero false alarms, i.e.\\ its errors are entirely in the "
                "lenient direction" if _fa == 0
                else f"{_fa} false alarms")
    _caption = (
        r"\textbf{Leakage-judge agreement with the human expert} "
        rf"($n{{=}}{N_RECORDS}$ expert-annotated Qwen3.5-9B base actions; the "
        # `%` is a LaTeX comment: every percentage in a caption is written with
        # an explicit `\%`, never through an f-string `:.0%` format.
        rf"expert calls {round(100 * EXPERT_LEAK_RATE)}\% of them leaks). "
        r"Higher accuracy / "
        r"$\kappa$ / leak-F1 = closer to the human; leak-R is recall on "
        r"expert-confirmed leaks and \emph{missed} counts expert-confirmed leaks "
        r"the judge called no-leak --- the lenient failure mode that would "
        r"flatter any privacy method. $\kappa$ intervals are 10{,}000-resample "
        r"percentile bootstraps over records. $p$ is an exact McNemar test "
        r"against the incumbent on the same records, which is the paired "
        r"comparison the small sample supports. The adopted Gemma-4-31B-it, the "
        r"prior Qwen3.6-27B, and the independent gpt-5.2 all track the expert "
        rf"($\kappa$ {_best.min():.2f}--{_best.max():.2f}) and each beats the "
        rf"incumbent on paired agreement ($p \leq {_mc['mcnemar_p'].max():.2f}$); "
        rf"the original Qwen3-32B-AWQ reaches $\kappa$ {_inc['cohen_kappa']:.2f} "
        rf"and misses {int(_inc['missed_leaks'])} of {N_EXPERT_LEAKS} "
        rf"expert-confirmed leaks while raising {_lenient}. "
        rf"$^{{\dagger}}$Gemma-4 and Qwen3.6 judged April re-runs of the same "
        rf"model and prompt, so {N_IDENTICAL} of {N_RECORDS} of their actions "
        r"are byte-identical to the annotated one; restricting to those moves "
        rf"their $\kappa$ by at most {_kappa_shift:.02f} (see "
        r"\texttt{judge\_human\_agreement.csv})."
    )

    _tex = "\n".join([
        r"\begin{table}[ht]",
        r"\centering",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3pt}",
        r"\begin{tabular}{@{}lrccccccrr@{}}",
        r"\toprule",
        # `$\kappa$` rather than `Cohen's $\kappa$`: that column's width was set
        # by its header, and the caption already names the statistic.
        r"\textbf{Judge} & $n$ & Acc. & $\kappa$ & 95\% CI & leak P & "
        r"leak R & leak F1 & missed & $p$ \\",
        r"\midrule",
        "\n".join(_lines),
        r"\bottomrule",
        r"\end{tabular}",
        rf"\caption{{{_caption}}}",
        r"\label{tab:judge-human-agreement}",
        r"\end{table}",
    ])
    save_tex(_tex, "judge_human_agreement", also_paper=True)
    print(_tex)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## What to change in the appendix""")
    return


@app.cell
def _(
    DRIFT,
    INCUMBENT,
    N_RECORDS,
    SCOPE_ALL,
    agree_table,
    mcnemar_table,
    mo,
):
    _scoped = agree_table[agree_table["scope"] == SCOPE_ALL]
    _l = (_scoped[_scoped["judge_type"] == "leakage"]
          .set_index("judge_llm"))
    _h = (_scoped[_scoped["judge_type"] == "helpfulness"]
          .set_index("judge_llm"))
    _mc = mcnemar_table.set_index("judge_llm")

    def _f(j, c):
        return _l.loc[j, c]

    mo.md(f"""
    `app:judge-validation` in `B_additional-results.tex` needs four edits. Every
    number below is from the cells above, at the full annotated set.

    1. **The "Human-expert calibration" paragraph.** Replace the 36-record
       sentence. On **{N_RECORDS}** expert-annotated records the adopted
       Gemma-4-31B-it agrees at κ **{_f('Gemma-4-31B-it', 'cohen_kappa'):.2f}**
       (accuracy {_f('Gemma-4-31B-it', 'accuracy'):.0%}), Qwen3.6-27B at
       **{_f('Qwen3.6-27B', 'cohen_kappa'):.2f}**, gpt-5.2 at
       **{_f('gpt-5.2', 'cohen_kappa'):.2f}**; the incumbent Qwen3-32B-AWQ
       reaches **{_f(INCUMBENT, 'cohen_kappa'):.2f}** and recovers only
       **{_f(INCUMBENT, 'leak_recall'):.0%}** of expert-confirmed leaks
       ({int(_f(INCUMBENT, 'missed_leaks'))} missed,
       {int(_f(INCUMBENT, 'false_alarms'))} false alarms). The printed κ values
       (0.79 / 0.79 / 0.86 / 0.47) were all computed on the 36-row subset and
       every one of them is now wrong.

    2. **Stop conceding small-n without the paired test.** The sentence "the
       expert set is small ($n{{=}}36$, single-annotator)" should keep
       *single-annotator* and drop the implication that the comparison is
       underpowered: each consensus judge beats the incumbent on the same
       records at McNemar
       p = {", ".join(f"{_mc.loc[j, 'mcnemar_p']:.3f}" for j in _mc.index)}
       (Gemma-4 / Qwen3.6 / gpt-5.2).

    3. **Move the `(adopted)` marker to Gemma-4-31B-it.** Production has served
       Gemma-4-31B-it since 2026-07-16; the table and the surrounding prose
       still call Qwen3.6-27B "the judge we adopt".

    4. **The helpfulness caveat survives, with better numbers.** κ is
       {_h['cohen_kappa'].min():.2f}–{_h['cohen_kappa'].max():.2f} across judges
       (accuracy {_h['accuracy'].min():.0%}–{_h['accuracy'].max():.0%}); all
       four over-call "helpful". That is a rubric problem, not a judge problem,
       exactly as written.

    Also note: `figures/judge_human_agreement.pdf` and its caption still say
    $n{{=}}36$, so the figure is now inconsistent with the table it sits beside.
    It is a four-bar accuracy/κ chart and everything it needs is in
    `judge_human_agreement.csv`.

    **Not closed by this notebook.** Gemma-4 and Qwen3.6 scored their own April
    re-runs, so only {int(DRIFT['sum'].min())} of {N_RECORDS} of their actions
    are the ones the expert read. Re-judging the {N_RECORDS} Mar30 actions on
    the current Gemma-4 judge server would make all four columns byte-identical
    and retire the $^\\dagger$ footnote.
    """)
    return


if __name__ == "__main__":
    app.run()
