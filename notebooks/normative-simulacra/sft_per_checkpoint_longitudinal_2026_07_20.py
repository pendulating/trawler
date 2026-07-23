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
    # SFT per-checkpoint eval — comparative + longitudinal (2026-07-20)

    Results for the **2026-07-19 canonical per-checkpoint SFT eval**
    (`multirun/2026-07-19_eval_sft_per_checkpoint_all/22-48-47`, the JOINT
    32-cell run that supersedes the holed fragments from earlier that day),
    joined with the **canonical instruct baselines**
    (`multirun/2026-07-17_eval_canonical_instruct/21-21-52`) as the
    epoch-0 anchor of each family's training trajectory.

    Instruct = the exact pre-SFT weights every `sft-canonical` adapter was
    trained from (verified 2026-07-17), so instruct → ckpt171 → ckpt342 →
    ckpt513 is a true longitudinal series (epochs 0–3 of the same weights;
    GPT-OSS uses its own step count 86/172/258 = epochs 1–3).

    **The sweep may still be filling in.** The scan below reads live from
    disk — re-run the notebook (or just re-execute Phase A) to pick up arms
    that finished since. The progress cell states exactly which of the 32
    cells are complete / partial / not yet dispatched, so a blank cell is
    never mistaken for a result.

    **Phase D2 adds the empirical noise floor** from the 2026-07-21
    judge-free N=3 variance sweep (same checkpoints, 3 reps each), so
    cross-checkpoint deltas can be read against measured run-to-run
    variance instead of the single GoldCoin-rerun anecdote. The
    longitudinal figure (Phase E) carries the rep ranges as whiskers.

    ### Source of truth, and where W&B fits

    Numbers are read from each benchmark's `metrics.json` on disk — same as
    `colm_benchmark_results_gemma4_2026_07.py`, and for the same reason: the
    judge is verified from the judge-batch **manifests** (config lies — the
    stale `${oc.env:JUDGE_MODEL,Qwen3.6-27B}` default), and disk stays
    current while the sweep runs. W&B runs cannot be scoped to one multirun
    dir reliably (the `eval_all_run:` tag only exists when `WANDB_GROUP` was
    exported at launch), so Phase F instead *fetches* the cohort's W&B runs
    (matched on `config.model.lora_path` ∋ `2026-07-19_sft_canonical` +
    created-at cutoff) and cross-checks them against the disk scan — run
    states, coverage, and PrivacyLens summary numbers.

    ### Comparability caveats (read before quoting)

    1. **Everything here is template-overhaul + DFT era** (SFT relaunched
       2026-07-18/19). Not protocol-comparable with keeper-era SFT evals,
       and **not the same adapters** as the 07-17 `eval_canonical_sft_gemma4`
       SFT rows.
    2. **CIRL changed benchmark between the two sweeps.** The instruct sweep
       ran vignette probing (judge-free `accuracy`); the per-checkpoint sweep
       runs **trajectory** probing (judged Integrity / Utility / Complete).
       The trajectory columns therefore have **no epoch-0 point**, and the
       vignette column exists **only** at epoch 0. Do not read one as the
       other.
    3. **ConfAIde is tier-2a/2b only** in the per-checkpoint sweep
       (`confaide_tier2_only`); tier-3 exists only on the instruct side and
       is not tabulated here.
    4. **Engine kwargs differ** (per-benchmark `max_model_len`/`max_num_seqs`
       tuned for 6-wide parallelism in the ckpt sweep). Sampling params are
       unchanged, so metrics should be unaffected unless a prompt got
       truncated at the smaller `max_model_len` — treat sub-point cross-sweep
       gaps as noise (GoldCoin re-run noise alone is ~1pt).
    5. **VLM Q7 epoch-0 gaps for gemma-4**: the instruct-side gemma-4 Q7
       cells came from the separate `eval_gemma4_q7_backfill` sweep, which is
       deliberately out of scope here. Missing-at-epoch-0 is expected for
       those rows (user-accepted gap).
    6. **Judge = Gemma-4-31B-it** for every judged cell (PrivacyLens both
       sweeps, CIRL trajectory in the ckpt sweep), verified from manifests
       below. It is also the SFT teacher — accepted coupling of the gemma
       consolidation.
    7. **Gemma-4-12B rows are a different era.** Its whole E0–E3 series
       comes from the 2026-07-23 post-parity-review family rerun
       (ckpt513 finished training after the record above froze, and the
       07-21 parity reviews changed goldcoin/vlm/confaide semantics in
       between). Within-family 12B trends are clean; comparing 12B
       *levels* against the other ten families crosses the review
       boundary. Its CIRL column is CIRL-729 (net/leak/util), not the
       retired trajectory flow.
    """)
    return


@app.cell
def _():
    import json
    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd

    SFT_SWEEP_DIR = Path(
        "/share/pierson/matt/UAIR/multirun/"
        "2026-07-19_eval_sft_per_checkpoint_all/22-48-47"
    )
    # 2026-07-20 15:06 klara job-launch wedge: arms 18-19 and 21-31 of the
    # original sweep were cancelled and relaunched protocol-identically as
    # this restart sweep (see eval_sft_per_checkpoint_restart21_2026_07_20
    # sweep yaml header). Higher precedence: where both sweeps carry a cell
    # (openthinker 342/513 partial arms), the restart wins.
    RESTART_SWEEP_DIR = Path(
        "/share/pierson/matt/UAIR/multirun/"
        "2026-07-20_eval_sft_per_checkpoint_restart/15-14-55"
    )
    INSTRUCT_SWEEP_DIR = Path(
        "/share/pierson/matt/UAIR/multirun/"
        "2026-07-17_eval_canonical_instruct/21-21-52"
    )
    # gemma-4-12b E0-E3 rerun under POST-parity-review code (Matt's call
    # 2026-07-23: ckpt513 landed after the 07-19 record froze, and the
    # parity reviews changed goldcoin/vlm/confaide semantics in between —
    # a lone new-era E3 cell could not join its family's old-era rows, so
    # the WHOLE family reran for an internally consistent series).
    # Highest precedence: these rows REPLACE every gemma-4-12b row from
    # the sweeps above. Era caveat: 12b deltas vs OTHER families are
    # cross-era; within-family E0→E3 comparisons are clean.
    POSTREVIEW_12B_SWEEP_DIR = Path(
        "/share/pierson/matt/UAIR/multirun/"
        "2026-07-23_eval_12b_family_postreview/09-48-49"
    )
    REPORT_DIR = (
        Path(__file__).resolve().parent
        / "tables"
        / "sft_per_checkpoint_longitudinal_2026_07_20"
    )
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    EXPECTED_JUDGE = "Gemma-4-31B-it"

    # ── Families: model= override slug → display name, size-then-family order ──
    FAMILIES = [
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
    SLUG_TO_DISPLAY = dict(FAMILIES)
    DISPLAY_ORDER = [d for _s, d in FAMILIES]

    # ── Checkpoint step → epoch. 513 total steps = 3 epochs for everyone
    # except GPT-OSS (own step count: 258 total = 3 epochs).
    DEFAULT_EPOCHS = {171: 1, 342: 2, 513: 3}
    EPOCHS_BY_SLUG = {"gpt-oss-20b": {86: 1, 172: 2, 258: 3}}

    # ── The 32 expected sweep cells (sweeper roster of 22-48-47).
    # gemma-4-12b ckpt513 was still training at launch — deliberately absent;
    # a follow-up run owes it.
    EXPECTED_ROSTER = []
    for _slug, _disp in FAMILIES:
        _steps = sorted(EPOCHS_BY_SLUG.get(_slug, DEFAULT_EPOCHS))
        if _slug == "gemma-4-12b":
            _steps = [171, 342]
        for _st in _steps:
            EXPECTED_ROSTER.append((_slug, _st))
    assert len(EXPECTED_ROSTER) == 32, len(EXPECTED_ROSTER)

    # ── Metric registry: where each number lives on disk.
    # (mid, group, label, bench_dir, inner, subdir, dotted key,
    #  judged, lower_is_better, scale "pct"|"raw")
    # The same registry is probed in both sweeps; whichever metrics.json
    # exists gets read. CIRL trajectory only exists in the ckpt sweep,
    # CIRL vignette accuracy only in the instruct sweep (see intro caveat 2).
    METRICS = [
        ("gc_appl", "GoldCoin", "Appl.", "goldcoin", "goldcoin_hipaa",
         "compute_metrics_applicability", "macro_f1", False, False, "pct"),
        ("gc_comp", "GoldCoin", "Comp.", "goldcoin", "goldcoin_hipaa",
         "compute_metrics_compliance", "macro_f1", False, False, "pct"),
        ("pl_qa", "PrivacyLens", "QA Acc", "privacylens", "privacylens_eval",
         "compute_metrics", "qa_probing.accuracy", False, False, "pct"),
        ("pl_lk", "PrivacyLens", "Lk↓", "privacylens", "privacylens_eval",
         "compute_metrics", "leakage.leakage_rate_overall_with_default_zero",
         True, True, "pct"),
        ("pl_adjlk", "PrivacyLens", "Adj Lk↓", "privacylens",
         "privacylens_eval", "compute_metrics",
         "adjusted_leakage.adjusted_leakage_rate", True, True, "pct"),
        ("pl_helpful", "PrivacyLens", "Helpful", "privacylens",
         "privacylens_eval", "compute_metrics",
         "helpfulness.helpful_rate_overall_with_default_zero",
         True, False, "pct"),
        ("pl_help", "PrivacyLens", "Help", "privacylens", "privacylens_eval",
         "compute_metrics", "helpfulness.mean_score_overall_with_default_zero",
         True, False, "raw"),
        ("ca_2a", "ConfAIde", "r 2a", "confaide", "confaide",
         "compute_metrics_tier2a", "pearson_r", False, False, "pct"),
        ("ca_2b", "ConfAIde", "r 2b", "confaide", "confaide",
         "compute_metrics_tier2b", "pearson_r", False, False, "pct"),
        ("cirl_integ", "CIRL-traj", "Integrity", "cirl_vignettes",
         "cirl_vignettes", "compute_trajectory_metrics",
         "integrity_overall_with_default_no_leak", True, False, "pct"),
        ("cirl_util", "CIRL-traj", "Utility", "cirl_vignettes",
         "cirl_vignettes", "compute_trajectory_metrics",
         "utility_overall_with_default_zero", True, False, "pct"),
        ("cirl_comp", "CIRL-traj", "Complete", "cirl_vignettes",
         "cirl_vignettes", "compute_trajectory_metrics",
         "complete_overall_with_default_zero", True, False, "pct"),
        ("cirl_vig", "CIRL-vig", "Acc (E0 only)", "cirl_vignettes",
         "cirl_vignettes", "compute_metrics", "accuracy",
         False, False, "pct"),
        # CIRL-729 (2026-07-21 benchmark swap): the real action benchmark,
        # judge-free deterministic scoring under the eval_all key `cirl`.
        # Absent from all pre-swap sweeps (blank there); populated by the
        # variance restart + future canonical re-runs.
        ("cirl_leak", "CIRL-729", "Leak↓", "cirl", "cirl",
         "compute_metrics", "leakage.leakage_rate", False, True, "pct"),
        ("cirl_util", "CIRL-729", "Utility", "cirl", "cirl",
         "compute_metrics", "utility.utility_rate", False, False, "pct"),
        # net_score averages EVERY row (strict-unparseable → -1, the
        # paper's own penalty) so it stays meaningful for models that
        # ignore the <think>/<answer> format entirely; the leak/util
        # rates above are conditional-on-parsed and are dropped by the
        # scans when parseable_rate == 0 (see the guard in each scan).
        ("cirl_net", "CIRL-729", "Net", "cirl", "cirl",
         "compute_metrics", "net_score", False, False, "raw"),
        ("vlm_q7", "VLM", "Q7", "vlm_geoprivacy", "vlm_geoprivacy_bench",
         "compute_metrics", "per_question.Q7.accuracy", False, False, "pct"),
        ("mmlu", "MMLU", "Acc", "mmlu", "mmlu",
         "compute_metrics", "overall_accuracy", False, False, "pct"),
    ]
    METRIC_LABELS = {m[0]: f"{m[1]} {m[2]}" for m in METRICS}

    # ── W&B cross-check parameters (Phase F) ──────────────────────────────────
    WANDB_ENTITY = "uair"
    WANDB_PROJECT = "eval-all"
    # Every adapter in the cohort lives under a 2026-07-19_sft_canonical*
    # training multirun — this substring of config.model.lora_path is the
    # run-matcher (family tags only carry model_family, not the checkpoint).
    SFT_LORA_MARKER = "2026-07-19_sft_canonical"
    # The canonical sweep launched 2026-07-19 22:48:47 America/New_York
    # (EDT, UTC-4) = 2026-07-20T02:48:47Z. The holed fragment sweeps from
    # earlier that evening (19-20-27, 21-21-50, 21-50-31 local) used the SAME
    # adapters, so lora_path alone cannot exclude their W&B runs — this
    # created-at cutoff does.
    WANDB_CREATED_CUTOFF = "2026-07-20T02:48:00+00:00"
    # config.model.model_source basename → display (for W&B run matching).
    MODEL_SOURCE_TO_DISPLAY = {
        "Qwen3.5-2B": "Qwen3.5-2B",
        "Qwen3.5-4B": "Qwen3.5-4B",
        "Qwen3.5-9B": "Qwen3.5-9B",
        "Gemma-4-E2B-it": "Gemma-4-E2B",
        "Gemma-4-E4B-it": "Gemma-4-E4B",
        "gemma-4-12B-it": "Gemma-4-12B",
        "OpenThinker3-7B": "OpenThinker3-7B",
        "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
        "HARC-Llama-3.1-8B-Instruct": "HARC-Llama-3.1-8B",
        "Phi-4": "Phi-4",
        "GPT-OSS-20B": "GPT-OSS-20B",
    }
    return (
        DEFAULT_EPOCHS,
        DISPLAY_ORDER,
        EPOCHS_BY_SLUG,
        EXPECTED_JUDGE,
        EXPECTED_ROSTER,
        INSTRUCT_SWEEP_DIR,
        METRICS,
        METRIC_LABELS,
        MODEL_SOURCE_TO_DISPLAY,
        REPORT_DIR,
        RESTART_SWEEP_DIR,
        SFT_LORA_MARKER,
        SFT_SWEEP_DIR,
        SLUG_TO_DISPLAY,
        WANDB_CREATED_CUTOFF,
        WANDB_ENTITY,
        WANDB_PROJECT,
        json,
        np,
        pd,
        re,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase A — Scan both sweeps from disk

    Walk `<sweep>/<idx>/`, read the `model=` override (→ family + checkpoint),
    resolve the **served judge from the judge-batch manifests** (PrivacyLens
    `*_judge_batch/manifest.json`, CIRL `judge_*_batch/manifest.json`), and
    record one observation per (cell × metric) found on disk.
    """)
    return


@app.cell
def _(
    DEFAULT_EPOCHS,
    EPOCHS_BY_SLUG,
    INSTRUCT_SWEEP_DIR,
    METRICS,
    SFT_SWEEP_DIR,
    SLUG_TO_DISPLAY,
    json,
    pd,
    re,
):
    _CKPT_RE = re.compile(r"^(?P<slug>[\w.\-]+)/sft-canonical-ckpt(?P<step>\d+)$")
    _INSTR_RE = re.compile(r"^(?P<slug>[\w.\-]+)/instruct$")

    def _override_model(sub_dir):
        # The model override lives in any benchmark's .hydra/overrides.yaml.
        for _ov in sub_dir.glob("*/.hydra/overrides.yaml"):
            for _line in _ov.read_text(errors="ignore").splitlines():
                _line = _line.strip().lstrip("- ").strip()
                if _line.startswith("model="):
                    return _line.split("=", 1)[1]
        return None

    def _served_judges(sub_dir):
        """Every judge model recorded by a judge-batch manifest in this
        sub-run. Read from artifacts, never from config (the stale
        `${oc.env:JUDGE_MODEL,Qwen3.6-27B}` default lies)."""
        _found = set()
        for _pat in (
            "privacylens/privacylens_eval/outputs/*judge_batch/manifest.json",
            "cirl_vignettes/cirl_vignettes/outputs/judge_*_batch/manifest.json",
        ):
            for _man in sub_dir.glob(_pat):
                try:
                    _m = json.loads(_man.read_text()).get("model")
                except (ValueError, OSError):
                    continue
                if _m:
                    _found.add(str(_m).rstrip("/").split("/")[-1])
        return _found

    def _dotted(_d, _path):
        _cur = _d
        for _part in _path.split("."):
            if not isinstance(_cur, dict) or _part not in _cur:
                return None
            _cur = _cur[_part]
        return _cur

    def _scan_sweep(sweep_dir, sweep_label, precedence=0):
        _rows = []
        if not sweep_dir.is_dir():
            print(f"!! sweep dir missing: {sweep_dir}")
            return _rows
        for _sub in sorted(
            (p for p in sweep_dir.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        ):
            _ovr = _override_model(_sub)
            if _ovr is None:
                continue
            _m_ck = _CKPT_RE.match(_ovr)
            _m_in = _INSTR_RE.match(_ovr)
            if _m_ck:
                _slug = _m_ck.group("slug")
                _step = int(_m_ck.group("step"))
                _epoch = EPOCHS_BY_SLUG.get(_slug, DEFAULT_EPOCHS).get(_step)
                if _epoch is None:
                    print(f"!! unmapped checkpoint step {_step} for {_slug} "
                          f"({_sub}) — skipped")
                    continue
                _cond = f"ckpt{_step}"
            elif _m_in:
                _slug, _step, _epoch, _cond = _m_in.group("slug"), 0, 0, "instruct"
            else:
                continue
            if _slug not in SLUG_TO_DISPLAY:
                print(f"!! unknown model slug {_slug!r} in {_sub} — skipped")
                continue
            _judges = _served_judges(_sub)
            _judge = _judges.pop() if len(_judges) == 1 else None
            if _judges and _judge is None:
                print(f"!! MIXED judges in {_sub}: cannot attribute")
            for (_mid, _grp, _lab, _bd, _inner, _subdir, _key,
                 _judged, _lo, _scale) in METRICS:
                _mp = _sub / _bd / _inner / "outputs" / _subdir / "metrics.json"
                if not _mp.exists():
                    continue
                try:
                    _mjson = json.loads(_mp.read_text())
                    _val = _dotted(_mjson, _key)
                except (ValueError, OSError):
                    continue
                if _val is None:
                    continue
                # CIRL-729 conditional-on-parsed rates are degenerate when
                # nothing parsed (0.0 over n=0) — only net_score carries
                # signal there (strict-unparseable rows score -1).
                if (_mid in ("cirl_leak", "cirl_util")
                        and (_mjson.get("parseable") or 0)
                            < (_mjson.get("total") or 729) // 2):
                    continue
                _rows.append({
                    "model": SLUG_TO_DISPLAY[_slug], "slug": _slug,
                    "condition": _cond, "step": _step, "epoch": _epoch,
                    "metric": _mid, "value": float(_val),
                    "judged": _judged, "lower_better": _lo, "scale": _scale,
                    "judge": _judge if _judged else None,
                    "sweep": sweep_label, "arm": int(_sub.name),
                    "sweep_dir": str(sweep_dir), "precedence": precedence,
                })
        return _rows

    scan = pd.DataFrame(
        _scan_sweep(SFT_SWEEP_DIR, "sft_per_checkpoint", precedence=0)
        # Restart supersedes the wedge-holed original arms cell-by-cell.
        + _scan_sweep(RESTART_SWEEP_DIR, "sft_restart21", precedence=1)
        + _scan_sweep(INSTRUCT_SWEEP_DIR, "canonical_instruct", precedence=0)
        # Post-review 12b family rerun supersedes ALL gemma-4-12b rows
        # (incl. its E0 instruct anchor) — see POSTREVIEW_12B_SWEEP_DIR.
        + _scan_sweep(POSTREVIEW_12B_SWEEP_DIR, "12b_postreview",
                      precedence=2)
    )
    print(
        f"{len(scan)} (cell × metric) observations — "
        f"{scan[scan.sweep == 'sft_per_checkpoint']['arm'].nunique()} ckpt arms"
        f" + {scan[scan.sweep == 'sft_restart21']['arm'].nunique()} restart arms, "
        f"{scan[scan.sweep == 'canonical_instruct']['arm'].nunique()} instruct arms"
    )
    print(f"judges seen on judged cells: "
          f"{sorted(scan.loc[scan.judged, 'judge'].dropna().unique())}")
    scan
    return (scan,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase A2 — Sweep progress: which of the 32 cells are in

    The per-checkpoint sweep runs 6-wide over 32 arms; anything not yet
    dispatched simply has no `<idx>/` dir. `failures.json` (written per arm
    at completion) separates "arm finished" from "arm still running".
    """)
    return


@app.cell
def _(EXPECTED_ROSTER, SLUG_TO_DISPLAY, json, pd, scan):
    from pathlib import Path as _Path

    # (slug, step) -> (sweep_dir, arm), highest-precedence sweep wins — a
    # cell holed in the wedged original but rerun in the restart reports the
    # restart arm's status.
    _arm_of = {}
    _sft = scan[scan.sweep.isin(["sft_per_checkpoint", "sft_restart21"])]
    for (_slug, _step, _sd, _arm, _prec), _ in _sft.groupby(
        ["slug", "step", "sweep_dir", "arm", "precedence"]
    ):
        _key = (_slug, _step)
        if _key not in _arm_of or _prec >= _arm_of[_key][2]:
            _arm_of[_key] = (_sd, _arm, _prec)

    _rows = []
    for _slug, _step in EXPECTED_ROSTER:
        _hit = _arm_of.get((_slug, _step))
        _n = len(_sft[(_sft.slug == _slug) & (_sft.step == _step)])
        if _hit is None:
            _arm = None
            _status = "NOT DISPATCHED YET"
        else:
            _sd, _arm, _prec = _hit
            _fj = _Path(_sd) / str(_arm) / "failures.json"
            if _fj.exists():
                try:
                    _f = json.loads(_fj.read_text())
                    _failed = _f.get("failed") or []
                    _status = ("complete" if _f.get("success") and not _failed
                               else f"FAILED: {_failed}")
                except (ValueError, OSError):
                    _status = "complete? (failures.json unreadable)"
            else:
                _status = "running (no failures.json yet)"
        _rows.append({
            "model": SLUG_TO_DISPLAY[_slug], "ckpt": _step,
            "arm": _arm, "metrics_found": _n, "status": _status,
        })
    progress = pd.DataFrame(_rows)
    _done = (progress.status == "complete").sum()
    _pend = (progress.status == "NOT DISPATCHED YET").sum()
    print(f"sweep progress: {_done}/32 arms complete, "
          f"{(32 - _done - _pend)} running/failed, {_pend} not yet dispatched")
    print("(gemma-4-12b is absent from this 32-cell roster tracker by "
          "design — its whole E0-E3 series now comes from the 2026-07-23 "
          "post-review family rerun; see intro caveat 7)")
    progress
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase B — Judge assertion + cell selection

    Every judged observation must come from Gemma-4-31B-it; anything else is
    dropped **loudly**. Then one value per (model, epoch, metric) — within
    these two sweeps that key is already unique, the dedupe is a guard.
    """)
    return


@app.cell
def _(EXPECTED_JUDGE, scan):
    _judged = scan[scan.judged]
    _bad = _judged[~_judged["judge"].fillna("").str.contains(EXPECTED_JUDGE)]
    if len(_bad):
        print(f"!! {len(_bad)} judged observations NOT judged by "
              f"{EXPECTED_JUDGE} — DROPPED:")
        print(_bad[["model", "condition", "metric", "judge", "sweep", "arm"]]
              .to_string())
    else:
        print(f"OK: all {len(_judged)} judged observations came from "
              f"{EXPECTED_JUDGE}")

    cells_df = (
        scan[(~scan.judged)
             | (scan["judge"].fillna("").str.contains(EXPECTED_JUDGE))]
        .sort_values(["precedence", "arm"])
        .drop_duplicates(subset=["model", "epoch", "metric"], keep="last")
        .copy()
    )
    print(f"{len(cells_df)} cells")
    cells_df
    return (cells_df,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase B2 — Coverage grid

    ✓ = value on disk. Expected blanks: CIRL-traj at E0 and CIRL-vig at E1–E3
    (benchmark changed between sweeps), VLM Q7 for text-only families and for
    gemma-4 at E0 (backfill sweep out of scope), gemma-4-12b E3 (not in
    roster), plus everything the sweep hasn't reached yet.
    """)
    return


@app.cell
def _(DISPLAY_ORDER, METRICS, METRIC_LABELS, cells_df, pd):
    _have = set(zip(cells_df["model"], cells_df["epoch"], cells_df["metric"]))
    _epochs_present = sorted(cells_df["epoch"].unique())
    _rows = []
    for _mdl in DISPLAY_ORDER:
        for _ep in _epochs_present:
            if not len(cells_df[(cells_df.model == _mdl)
                                & (cells_df.epoch == _ep)]):
                continue
            _row = {"model": _mdl, "epoch": _ep}
            for _m in METRICS:
                _row[METRIC_LABELS[_m[0]]] = (
                    "✓" if (_mdl, _ep, _m[0]) in _have else "—"
                )
            _rows.append(_row)
    coverage = pd.DataFrame(_rows)
    coverage
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase C — Comparative table

    One row per (model × epoch), instruct (E0) first. **Bold = best epoch
    within that model** for the column (min for the ↓ columns) — the
    longitudinal question "which checkpoint should we keep" read per cell.
    Percentages ×100; Help on the 0–3 scale.
    """)
    return


@app.cell
def _(DISPLAY_ORDER, METRICS, METRIC_LABELS, REPORT_DIR, cells_df, mo):
    _lut = {
        (r["model"], r["epoch"], r["metric"]): r["value"]
        for _, r in cells_df.iterrows()
    }
    _cond_of = {
        (r["model"], r["epoch"]): r["condition"]
        for _, r in cells_df.iterrows()
    }

    def _scaled(_v, _scale):
        return _v * 100.0 if _scale == "pct" else _v

    _lines = ["| Model | Ckpt | " +
              " | ".join(METRIC_LABELS[_m[0]] for _m in METRICS) + " |",
              "|" + "---|" * (2 + len(METRICS))]
    for _mdl in DISPLAY_ORDER:
        _eps = sorted({e for (m, e, _mi) in _lut if m == _mdl})
        if not _eps:
            continue
        # best epoch per metric within this model
        _best = {}
        for _m in METRICS:
            _vals = {e: _lut[(_mdl, e, _m[0])] for e in _eps
                     if (_mdl, e, _m[0]) in _lut}
            if _vals:
                _pick = min if _m[8] else max
                _best[_m[0]] = _pick(_vals.values())
        for _i, _ep in enumerate(_eps):
            _cells = []
            for _m in METRICS:
                _v = _lut.get((_mdl, _ep, _m[0]))
                if _v is None:
                    _cells.append("—")
                    continue
                _sv = _scaled(_v, _m[9])
                _s = ("{:.1f}" if _m[9] == "pct" else "{:.2f}").format(_sv)
                if _m[0] in _best and abs(_v - _best[_m[0]]) < 1e-12:
                    _s = f"**{_s}**"
                _cells.append(_s)
            _cond = _cond_of.get((_mdl, _ep), f"E{_ep}")
            _label = f"{_cond} (E{_ep})"
            _lines.append(
                f"| {_mdl if _i == 0 else ''} | {_label} | "
                + " | ".join(_cells) + " |"
            )
    comparative_md = "\n".join(_lines) + (
        "\n\n*Percentages ×100 except **Help** (mean helpfulness, 0–3). "
        "↓ = lower is better. **Bold = best epoch within the model** per "
        "column. E0 = the pre-SFT `instruct` checkpoint (2026-07-17 canonical "
        "sweep); E1–E3 = SFT epoch checkpoints (2026-07-19 per-checkpoint "
        "sweep, template-overhaul + DFT era). CIRL-traj (judged Integrity/"
        "Utility/Complete) exists only at E1–E3; CIRL-vig accuracy only at "
        "E0 — different benchmarks, never compare across them. ConfAIde r = "
        "tier-2a/2b Pearson. Judged columns use Gemma-4-31B-it (also the SFT "
        "teacher). Blank = gap (see coverage grid), never zero.*\n"
    )
    (REPORT_DIR / "comparative_table.md").write_text(
        "## SFT per-checkpoint comparative table (2026-07-20)\n\n"
        + comparative_md
    )
    print(f"saved {REPORT_DIR / 'comparative_table.md'}")
    mo.md(comparative_md)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase D — SFT effect over training: deltas vs the instruct anchor

    Signed so **positive = better than instruct** everywhere (↓ columns are
    negated). Two views: the final available checkpoint, and the best
    checkpoint per (model, metric). Only metrics with an E0 value can be
    differenced — the CIRL-traj columns can't (no anchor).
    """)
    return


@app.cell
def _(DISPLAY_ORDER, METRICS, METRIC_LABELS, REPORT_DIR, cells_df, pd):
    _lut = {
        (r["model"], r["epoch"], r["metric"]): r["value"]
        for _, r in cells_df.iterrows()
    }

    def _delta_rows(pick_final):
        _rows = []
        for _mdl in DISPLAY_ORDER:
            _row = {"model": _mdl}
            for _m in METRICS:
                _mid, _lo, _scale = _m[0], _m[8], _m[9]
                _z = _lut.get((_mdl, 0, _mid))
                _sft = {e: _lut[(_mdl, e, _mid)] for e in (1, 2, 3)
                        if (_mdl, e, _mid) in _lut}
                if _z is None or not _sft:
                    _row[METRIC_LABELS[_mid]] = None
                    continue
                if pick_final:
                    _v = _sft[max(_sft)]
                else:
                    _v = (min if _lo else max)(_sft.values())
                _mult = 100.0 if _scale == "pct" else 1.0
                _d = (_v - _z) * _mult
                _row[METRIC_LABELS[_mid]] = -_d if _lo else _d
            _rows.append(_row)
        return (pd.DataFrame(_rows).set_index("model")
                .dropna(axis=1, how="all").dropna(axis=0, how="all"))

    delta_final = _delta_rows(pick_final=True)
    delta_best = _delta_rows(pick_final=False)

    print("Final-available-checkpoint − instruct "
          "(positive = SFT better; ↓ columns sign-flipped):")
    print(delta_final.round(1).to_string())
    print("\nBest-checkpoint − instruct:")
    print(delta_best.round(1).to_string())
    print("\nMean delta per column (final ckpt):")
    print(delta_final.mean(axis=0).round(2).to_string())
    delta_final.to_parquet(REPORT_DIR / "delta_final_vs_instruct.parquet")
    delta_best.to_parquet(REPORT_DIR / "delta_best_vs_instruct.parquet")
    delta_final.round(1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase D2 — Benchmark variance: the empirical noise floor

    Scans the **2026-07-21 judge-free N=3 variance sweep**
    (`multirun/*_eval_judgefree_variance/*` + the in-process gpt-oss
    companion `multirun/*_eval_judgefree_variance_gptoss/*`): the same 43
    instruct/checkpoint configs, each run 3× over the judge-free benchmarks
    (GoldCoin, ConfAIde 2a/2b, **CIRL-729** action leakage/utility, VLM Q7,
    MMLU). The first attempt (arms 0-~50, pre-CIRL-swap and
    pre-parity-review metric semantics) was cancelled and moved to
    `..._CANCELLED_precirlswap/` — deliberately outside these globs; do
    not pool it.

    Rep design (see the sweep yaml header): the **sampled** benchmarks
    (GoldCoin temp 0.2, VLM Q7 temp 0.2) rep over `variance_seed`
    101/102/103 — sampling variance. The **greedy** benchmarks (ConfAIde,
    CIRL-729, MMLU, temp 0) ignore the seed — their spread is pure engine
    nondeterminism (batching order, kernel numerics). The two are reported
    separately; both contribute to what a "different" number means.

    The **GoldCoin top-up sweep** (seeds 104–108, one server boot per
    model, reps as `goldcoin_sNNN/` dirs inside each arm) extends N for
    the configs whose N=3 range exceeded the quotable threshold; its reps
    pool into the same per-cell stats below.

    Caveats: (1) reps run **server-mode** with the family yaml's engine
    kwargs and seeds ≠ the canonical 42/1, so rep **dispersion** is the
    transferable quantity — rep *means* may sit slightly off the canonical
    points. (2) On top of that, the variance reps run **post-parity-review
    code** (2026-07-21: goldcoin/vlm/confaide upstream-parity denominators)
    while the canonical sweeps above predate it — another reason to
    transfer only the dispersion, never the means, and to expect visible
    whisker-center offsets for models with unparseable rows. (3) The sweep
    may be in progress: cells with <2 reps are excluded from spread stats
    and counted below. (4) CIRL-729 noise says nothing about the judged
    CIRL-trajectory columns (nor PrivacyLens; both judged, out of scope) —
    and those trajectory columns are themselves the OLD
    PrivacyLens-under-CIRL-protocol, superseded by the benchmark swap.
    """)
    return


@app.cell
def _(DEFAULT_EPOCHS, EPOCHS_BY_SLUG, METRICS, SLUG_TO_DISPLAY, json, pd, re):
    from pathlib import Path as _VPath

    _MULTIRUN = _VPath("/share/pierson/matt/UAIR/multirun")
    # Main (server-mode) half + in-process gpt-oss companion + the GoldCoin
    # seed top-up (one arm per model, reps as goldcoin_sNNN benchmark dirs —
    # see eval_judgefree_variance_topup_goldcoin_2026_07_21.yaml). Globs keep
    # the notebook working before each sweep exists — absent ones just
    # report themselves absent.
    _VARIANCE_GLOBS = [
        "*_eval_judgefree_variance/*",
        "*_eval_judgefree_variance_gptoss/*",
        "*_eval_judgefree_variance_topup/*",
    ]
    _vdirs = sorted(
        {p for g in _VARIANCE_GLOBS for p in _MULTIRUN.glob(g) if p.is_dir()}
    )
    if not _vdirs:
        print("!! no variance sweep dirs found under multirun/ — "
              "has the 2026-07-21 sweep started?")
    if not any("gptoss" in str(p) for p in _vdirs):
        print("(gpt-oss companion sweep not on disk yet — runs after the "
              "server-mode half; GPT-OSS-20B rows will be absent until then)")

    _V_CKPT_RE = re.compile(
        r"^(?P<slug>[\w.\-]+)/sft-canonical-ckpt(?P<step>\d+)$")
    _V_INSTR_RE = re.compile(r"^(?P<slug>[\w.\-]+)/instruct$")

    def _v_dotted(_d, _path):
        _cur = _d
        for _part in _path.split("."):
            if not isinstance(_cur, dict) or _part not in _cur:
                return None
            _cur = _cur[_part]
        return _cur

    _rows = []
    _arms_seen = 0
    for _vdir in _vdirs:
        for _sub in sorted(
            (p for p in _vdir.iterdir() if p.is_dir() and p.name.isdigit()),
            key=lambda p: int(p.name),
        ):
            # Arm-level overrides carry model= AND variance_seed= (the rep id).
            _ov = _sub / ".hydra" / "overrides.yaml"
            if not _ov.exists():
                continue
            _model_ovr, _seed = None, None
            for _line in _ov.read_text(errors="ignore").splitlines():
                _line = _line.strip().lstrip("- ").strip()
                if _line.startswith("model="):
                    _model_ovr = _line.split("=", 1)[1]
                elif _line.startswith("variance_seed="):
                    _seed = int(_line.split("=", 1)[1])
            # The top-up sweep sweeps model only (no variance_seed
            # override); its rep seeds live in the benchmark dir names.
            if _model_ovr is None:
                continue
            _m_ck = _V_CKPT_RE.match(_model_ovr)
            _m_in = _V_INSTR_RE.match(_model_ovr)
            if _m_ck:
                _slug = _m_ck.group("slug")
                _step = int(_m_ck.group("step"))
                _epoch = EPOCHS_BY_SLUG.get(_slug, DEFAULT_EPOCHS).get(_step)
                if _epoch is None:
                    continue
            elif _m_in:
                _slug, _step, _epoch = _m_in.group("slug"), 0, 0
            else:
                continue
            if _slug not in SLUG_TO_DISPLAY:
                continue
            _arms_seen += 1
            # Judge-free subset resolves naturally: judged metrics have no
            # metrics.json in this sweep.
            for (_mid, _grp, _lab, _bd, _inner, _subdir, _key,
                 _judged, _lo, _scale) in METRICS:
                # Rep dirs: plain `<bd>/` (N=3 sweep, seed from the arm's
                # variance_seed override) and/or `<bd>_sNNN/` (top-up
                # sweep, seed from the dir suffix).
                _rep_dirs = []
                if _seed is not None and (_sub / _bd).is_dir():
                    _rep_dirs.append((_sub / _bd, _seed))
                for _rd in _sub.glob(f"{_bd}_s*"):
                    _sfx = _rd.name[len(_bd) + 2:]
                    if _rd.is_dir() and _sfx.isdigit():
                        _rep_dirs.append((_rd, int(_sfx)))
                for _rdir, _rseed in _rep_dirs:
                    _mp = (_rdir / _inner / "outputs" / _subdir
                           / "metrics.json")
                    if not _mp.exists():
                        continue
                    try:
                        _mjson = json.loads(_mp.read_text())
                        _val = _v_dotted(_mjson, _key)
                    except (ValueError, OSError):
                        continue
                    if _val is None:
                        continue
                    # Conditional-on-parsed rates need a real denominator:
                    # a model parsing 1/729 items swings leak 0→0.67 on
                    # one row (observed). Require majority-parseable;
                    # cirl_net carries the signal below that.
                    if (_mid in ("cirl_leak", "cirl_util")
                            and (_mjson.get("parseable") or 0)
                            < (_mjson.get("total") or 729) // 2):
                        continue
                    _rows.append({
                        "model": SLUG_TO_DISPLAY[_slug], "slug": _slug,
                        "step": _step, "epoch": _epoch, "metric": _mid,
                        "seed": _rseed, "value": float(_val),
                        "lower_better": _lo, "scale": _scale,
                        "sweep_dir": str(_vdir), "arm": int(_sub.name),
                    })
    variance_df = pd.DataFrame(_rows)
    # 43 configs x 3 seeds: 117 server-mode + 12 gpt-oss companion arms;
    # top-up arms (one per model, 5 goldcoin reps inside) come on top.
    print(f"variance sweep: {_arms_seen} arms on disk (129 = complete N=3; "
          f"more = top-up landed), {len(variance_df)} (rep × metric) "
          "observations")
    if len(variance_df):
        _rep_counts = (variance_df.groupby(["model", "epoch", "metric"])
                       .size().value_counts().to_dict())
        print(f"reps per (model, epoch, metric): {_rep_counts}")
    variance_df
    return (variance_df,)


@app.cell
def _(METRIC_LABELS, REPORT_DIR, np, pd, variance_df):
    # Seeded (sampled) vs greedy rep types — which noise source a spread
    # measures (see the sweep yaml's variance-design header).
    _VARIANCE_REP_TYPE = {
        "gc_appl": "sampled", "gc_comp": "sampled", "vlm_q7": "sampled",
        "ca_2a": "greedy", "ca_2b": "greedy", "cirl_vig": "greedy",
        "cirl_leak": "greedy", "cirl_util": "greedy", "cirl_net": "greedy",
        "mmlu": "greedy",
    }

    if not len(variance_df):
        variance_stats = pd.DataFrame()
        noise_floor = pd.DataFrame()
        print("(no variance data yet — noise floor unavailable)")
    else:
        _g = variance_df.groupby(["model", "epoch", "metric"])
        variance_stats = _g["value"].agg(
            n_reps="count", rep_mean="mean", rep_std="std",
            rep_min="min", rep_max="max",
        ).reset_index()
        variance_stats["rep_range"] = (
            variance_stats["rep_max"] - variance_stats["rep_min"])
        _scale_of = dict(zip(variance_df["metric"], variance_df["scale"]))
        _mult = variance_stats["metric"].map(
            lambda m: 100.0 if _scale_of.get(m) == "pct" else 1.0)
        for _c in ("rep_mean", "rep_std", "rep_min", "rep_max", "rep_range"):
            variance_stats[_c + "_disp"] = variance_stats[_c] * _mult
        variance_stats["rep_type"] = variance_stats["metric"].map(
            _VARIANCE_REP_TYPE)

        _full = variance_stats[variance_stats.n_reps >= 2]
        _sgl = variance_stats[variance_stats.n_reps < 2]
        print(f"{len(_full)} cells with ≥2 reps (spread measurable), "
              f"{len(_sgl)} single-rep cells excluded from the floor")

        # Per-metric noise floor, display units (pct ×100; Help-style raw
        # metrics keep their scale). max_range = the conservative "a gap
        # this small is indistinguishable from a re-run" threshold.
        noise_floor = (_full.groupby("metric").agg(
            rep_type=("rep_type", "first"),
            n_cells=("metric", "count"),
            median_std=("rep_std_disp", "median"),
            max_std=("rep_std_disp", "max"),
            median_range=("rep_range_disp", "median"),
            max_range=("rep_range_disp", "max"),
        ).round(2).reset_index())
        noise_floor["metric_label"] = noise_floor["metric"].map(METRIC_LABELS)
        noise_floor = noise_floor.sort_values(
            ["rep_type", "median_range"], ascending=[False, False])
        print("\nPer-metric noise floor (display units, cells with ≥2 reps):")
        print(noise_floor[["metric_label", "rep_type", "n_cells",
                           "median_std", "max_std", "median_range",
                           "max_range"]].to_string(index=False))

        # The worst offenders — cells whose reps spread the most.
        _worst = _full.nlargest(8, "rep_range_disp")
        print("\nWidest cells (rep range, display units):")
        print(_worst[["model", "epoch", "metric", "n_reps",
                      "rep_range_disp"]].round(2).to_string(index=False))

        variance_stats.to_parquet(
            REPORT_DIR / "variance_cells.parquet", index=False)
        noise_floor.to_parquet(
            REPORT_DIR / "variance_noise_floor.parquet", index=False)
        _md = ["## Judge-free benchmark noise floor (2026-07-21 N=3 "
               "variance sweep)\n",
               "| Metric | Rep type | Cells | Median σ | Max σ | "
               "Median range | Max range |",
               "|---|---|---|---|---|---|---|"]
        for _, _r in noise_floor.iterrows():
            _md.append(
                f"| {_r.metric_label} | {_r.rep_type} | {_r.n_cells} | "
                f"{_r.median_std} | {_r.max_std} | {_r.median_range} | "
                f"{_r.max_range} |")
        _md.append(
            "\n*Display units (pct ×100). `sampled` = reps vary "
            "`sampling_params.seed` (101/102/103); `greedy` = temp-0 reps, "
            "spread is engine nondeterminism. Range = max−min over ≤3 reps. "
            "A cross-checkpoint gap below a metric's max range is "
            "indistinguishable from re-run noise.*\n")
        (REPORT_DIR / "variance_noise_floor.md").write_text("\n".join(_md))
        print(f"\nsaved {REPORT_DIR / 'variance_noise_floor.md'} (+ parquets)")
    noise_floor if len(variance_df) else variance_stats
    return noise_floor, variance_stats


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ### Phase D2b — Is the "best checkpoint" call above the noise?

    For each (model, metric) with a noise estimate: the margin between the
    best and runner-up epoch (canonical values, Phase C) vs that cell
    family's own measured rep range (falling back to the metric-level
    median range where the model's variance cells haven't landed yet).
    `within_noise=True` means the bold cell in Phase C should be read as a
    tie, not a winner.
    """)
    return


@app.cell
def _(
    METRICS,
    METRIC_LABELS,
    REPORT_DIR,
    cells_df,
    noise_floor,
    pd,
    variance_stats,
):
    if not len(variance_stats):
        margin_vs_noise = pd.DataFrame()
        print("(no variance data yet — margin check skipped)")
    else:
        _floor_of = dict(zip(noise_floor["metric"], noise_floor["median_range"]))
        _meta = {m[0]: (m[8], m[9]) for m in METRICS}  # mid -> (lower, scale)
        _rows = []
        for (_mdl, _mid), _sub in cells_df.groupby(["model", "metric"]):
            if _mid not in _floor_of or len(_sub) < 2:
                continue
            _lo, _scale = _meta[_mid]
            _mult = 100.0 if _scale == "pct" else 1.0
            _vals = _sub.set_index("epoch")["value"] * _mult
            _srt = _vals.sort_values(ascending=_lo)
            _margin = abs(_srt.iloc[0] - _srt.iloc[1])
            # Prefer the model's own measured spread for this metric.
            _own = variance_stats[
                (variance_stats.model == _mdl)
                & (variance_stats.metric == _mid)
                & (variance_stats.n_reps >= 2)]
            _noise = (_own["rep_range_disp"].max() if len(_own)
                      else _floor_of[_mid])
            _rows.append({
                "model": _mdl, "metric": METRIC_LABELS[_mid],
                "best_epoch": int(_srt.index[0]),
                "runner_up_epoch": int(_srt.index[1]),
                "margin": round(_margin, 2),
                "noise": round(float(_noise), 2),
                "noise_source": "own cells" if len(_own) else "metric median",
                "within_noise": bool(_margin <= _noise),
            })
        margin_vs_noise = pd.DataFrame(_rows)
        if len(margin_vs_noise):
            _n_tie = int(margin_vs_noise["within_noise"].sum())
            print(f"{_n_tie}/{len(margin_vs_noise)} best-checkpoint calls "
                  "are within noise (read as ties)")
            margin_vs_noise.to_parquet(
                REPORT_DIR / "margin_vs_noise.parquet", index=False)
    margin_vs_noise
    return (margin_vs_noise,)


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase D2c — Variance visualizations

    Three figures over the Phase D2 noise data, all saved to `REPORT_DIR`:

    1. **Noise-floor overview** (`variance_noise_overview.png`) — one column
       per judge-free metric, a dot per cell showing its N=3 rep **range**
       (display units), median marked. Metrics are grouped **sampled** (left,
       seed-varied — GoldCoin/VLM) vs **greedy** (right, temp-0 engine
       nondeterminism — ConfAIde/CIRL-729/MMLU), split by a dashed rule.
       Two hues + two marker shapes so the split survives grayscale. This is
       the "how noisy is each benchmark" headline. CIRL-729 Net is raw
       (−1..1), flagged on its tick — its numeric range is not comparable to
       the pct-point ranges beside it.
    2. **Widest cells detail** (`variance_widest_cells.png`) — the 12 cells
       with the largest rep range, each row showing its individual reps as
       dots **around the cell mean** (x = value − mean), so a 20-point
       GoldCoin range reads as dots spread ±10. Row label carries the mean;
       the range is annotated at right.
    3. **Margin vs noise** (`variance_margin_vs_noise.png`) — from
       `margin_vs_noise`: for each (model, metric) the best-vs-runner-up epoch
       **margin** (blue) beside that cell's measured **noise** (orange). Rows
       where the margin sits within noise (a tie, not a winner) are shaded and
       tagged. Sorted margin/noise ascending, so the shakiest calls are at
       the top.
    """)
    return


@app.cell
def _(METRIC_LABELS, REPORT_DIR, noise_floor, np, variance_stats):
    import matplotlib as _mpl

    _mpl.use("Agg")
    import matplotlib.pyplot as _plt

    _SAMPLED_C, _GREEDY_C = "#2a78d6", "#1baf7a"
    _SAMPLED_MK, _GREEDY_MK = "o", "s"

    if not len(variance_stats) or not len(noise_floor):
        _fig, _ax = _plt.subplots(figsize=(8, 3))
        _ax.text(0.5, 0.5, "no variance data yet", ha="center", va="center",
                 fontsize=10, color="#555555")
        _ax.axis("off")
    else:
        _full = variance_stats[variance_stats.n_reps >= 2]
        # noise_floor is already sorted sampled-then-greedy, median_range desc.
        _order = list(noise_floor["metric"])
        _rt_of = dict(zip(noise_floor["metric"], noise_floor["rep_type"]))
        _rng = np.random.default_rng(0)
        _fig, _ax = _plt.subplots(
            figsize=(max(8.0, 1.05 * len(_order)), 4.7))
        for _x, _mid in enumerate(_order):
            _cells = _full[_full.metric == _mid]["rep_range_disp"].to_numpy()
            if not len(_cells):
                continue
            _rt = _rt_of[_mid]
            _c = _SAMPLED_C if _rt == "sampled" else _GREEDY_C
            _mk = _SAMPLED_MK if _rt == "sampled" else _GREEDY_MK
            _jit = _rng.uniform(-0.13, 0.13, size=len(_cells))
            _ax.scatter(np.full(len(_cells), _x) + _jit, _cells, s=28,
                        color=_c, marker=_mk, alpha=0.75, edgecolor="white",
                        linewidth=0.5, zorder=3)
            _med = float(np.median(_cells))
            _ax.hlines(_med, _x - 0.28, _x + 0.28, color="#333333",
                       linewidth=1.7, zorder=4)
            _ax.annotate(f"{_med:.1f}", (_x, _med), textcoords="offset points",
                         xytext=(0, 7), ha="center", fontsize=7,
                         color="#333333")
        # Dashed rule where the rep type changes (sampled | greedy).
        _rts = [_rt_of[m] for m in _order]
        for _k in range(1, len(_order)):
            if _rts[_k] != _rts[_k - 1]:
                _ax.axvline(_k - 0.5, color="#999999", linestyle="--",
                            linewidth=0.9, zorder=1)
        _labels = []
        for _mid in _order:
            _lab = METRIC_LABELS.get(_mid, _mid)
            if _mid == "cirl_net":
                _lab += "\n(raw −1..1)"
            _labels.append(_lab)
        _ax.set_xticks(range(len(_order)))
        _ax.set_xticklabels(_labels, rotation=35, ha="right", fontsize=8)
        _ax.set_ylabel("per-cell rep range (display units)", fontsize=9)
        _ax.set_title("Benchmark noise floor — N=3 rep range per cell "
                      "(median marked)", fontsize=10)
        _ax.set_ylim(bottom=-0.05 * max(1.0, _full["rep_range_disp"].max()))
        _ax.grid(True, alpha=0.25, linewidth=0.5, axis="y")
        for _sp in ("top", "right"):
            _ax.spines[_sp].set_visible(False)
        _ax.tick_params(labelsize=8)
        _hs = [
            _plt.Line2D([], [], color=_SAMPLED_C, marker=_SAMPLED_MK,
                        linestyle="none", markersize=6,
                        markeredgecolor="white", label="sampled (seed-varied)"),
            _plt.Line2D([], [], color=_GREEDY_C, marker=_GREEDY_MK,
                        linestyle="none", markersize=6,
                        markeredgecolor="white",
                        label="greedy (engine nondeterminism)"),
        ]
        _ax.legend(handles=_hs, fontsize=8, frameon=False, loc="upper right")
        _fig.tight_layout()
        _fig.savefig(REPORT_DIR / "variance_noise_overview.png",
                     dpi=150, bbox_inches="tight")
        print(f"saved {REPORT_DIR / 'variance_noise_overview.png'}")
    _fig
    return


@app.cell
def _(METRIC_LABELS, REPORT_DIR, np, variance_df, variance_stats):
    import matplotlib as _mpl

    _mpl.use("Agg")
    import matplotlib.pyplot as _plt

    _SAMPLED_C, _GREEDY_C = "#2a78d6", "#1baf7a"
    _SAMPLED_MK, _GREEDY_MK = "o", "s"
    _REP_TYPE = {
        "gc_appl": "sampled", "gc_comp": "sampled", "vlm_q7": "sampled",
        "ca_2a": "greedy", "ca_2b": "greedy", "cirl_vig": "greedy",
        "cirl_leak": "greedy", "cirl_util": "greedy", "cirl_net": "greedy",
        "mmlu": "greedy",
    }

    if not len(variance_stats) or not (variance_stats.n_reps >= 2).any():
        _fig, _ax = _plt.subplots(figsize=(8, 3))
        _ax.text(0.5, 0.5, "no variance data yet", ha="center", va="center",
                 fontsize=10, color="#555555")
        _ax.axis("off")
    else:
        _scale_of = dict(zip(variance_df["metric"], variance_df["scale"]))
        _full = variance_stats[variance_stats.n_reps >= 2]
        _top = _full.nlargest(12, "rep_range_disp").reset_index(drop=True)
        _fig, _ax = _plt.subplots(
            figsize=(8.8, max(4.0, 0.52 * len(_top) + 1.2)))
        _rows_plot = []
        _xmax = 0.0
        for _i, _r in _top.iterrows():
            _mid = _r["metric"]
            _mult = 100.0 if _scale_of.get(_mid) == "pct" else 1.0
            _reps = variance_df[(variance_df.model == _r["model"])
                                & (variance_df.epoch == _r["epoch"])
                                & (variance_df.metric == _mid)]
            _vals = _reps["value"].to_numpy() * _mult
            _mean = float(np.mean(_vals)) if len(_vals) else 0.0
            _dev = _vals - _mean
            # +1 leaves an empty y=1 lane at the bottom for the legend.
            _y = len(_top) - _i + 1
            _rt = _REP_TYPE.get(_mid, "greedy")
            _rows_plot.append((_y, _dev, _rt, float(_r["rep_range_disp"]),
                               _r["model"], int(_r["epoch"]),
                               METRIC_LABELS.get(_mid, _mid), _mean))
            _xmax = max(_xmax,
                        float(np.max(np.abs(_dev))) if len(_dev) else 0.0)
        _annx = _xmax * 1.12 + 0.5
        _yticks, _ylabels = [], []
        for _y, _dev, _rt, _rrange, _mdl, _ep, _mlab, _mean in _rows_plot:
            _c = _SAMPLED_C if _rt == "sampled" else _GREEDY_C
            _mk = _SAMPLED_MK if _rt == "sampled" else _GREEDY_MK
            _ax.scatter(_dev, np.full(len(_dev), _y), s=38, color=_c,
                        marker=_mk, alpha=0.82, edgecolor="white",
                        linewidth=0.5, zorder=3)
            _ax.plot([0, 0], [_y - 0.32, _y + 0.32], color="#333333",
                     linewidth=1.4, zorder=2)
            _ax.annotate(f"range {_rrange:.1f}", (_annx, _y), va="center",
                         ha="left", fontsize=7, color="#555555")
            _yticks.append(_y)
            _ylabels.append(f"{_mdl} E{_ep} · {_mlab}  "
                            f"(μ={_mean:.1f})")
        _ax.axvline(0, color="#bbbbbb", linewidth=0.9, zorder=1)
        _ax.set_yticks(_yticks)
        _ax.set_yticklabels(_ylabels, fontsize=7.5)
        _ax.set_ylim(0.3, len(_top) + 1.7)
        _ax.set_xlim(-_xmax * 1.15 - 0.5, _annx + _xmax * 0.6 + 2.0)
        _ax.set_xlabel("rep value − cell mean (display units)",
                       fontsize=9)
        _ax.set_title("Widest variance cells — individual reps around "
                      "the cell mean", fontsize=10)
        _ax.grid(True, alpha=0.25, linewidth=0.5, axis="x")
        for _sp in ("top", "right", "left"):
            _ax.spines[_sp].set_visible(False)
        _ax.tick_params(labelsize=8, length=0)
        _hs = [
            _plt.Line2D([], [], color=_SAMPLED_C, marker=_SAMPLED_MK,
                        linestyle="none", markersize=6,
                        markeredgecolor="white", label="sampled (seed-varied)"),
            _plt.Line2D([], [], color=_GREEDY_C, marker=_GREEDY_MK,
                        linestyle="none", markersize=6,
                        markeredgecolor="white",
                        label="greedy (engine nondeterminism)"),
        ]
        _ax.legend(handles=_hs, fontsize=8, frameon=False, loc="lower left")
        _fig.tight_layout()
        _fig.savefig(REPORT_DIR / "variance_widest_cells.png",
                     dpi=150, bbox_inches="tight")
        print(f"saved {REPORT_DIR / 'variance_widest_cells.png'}")
    _fig
    return


@app.cell
def _(REPORT_DIR, margin_vs_noise, np):
    import matplotlib as _mpl

    _mpl.use("Agg")
    import matplotlib.pyplot as _plt

    _MARGIN_C, _NOISE_C = "#2a78d6", "#eb6834"

    if not len(margin_vs_noise):
        _fig, _ax = _plt.subplots(figsize=(8, 3))
        _ax.text(0.5, 0.5, "no margin/noise data yet", ha="center",
                 va="center", fontsize=10, color="#555555")
        _ax.axis("off")
    else:
        _d = margin_vs_noise.copy()
        _d["_ratio"] = _d["margin"] / _d["noise"].replace(0, np.nan)
        # Ties (within_noise) grouped contiguously at top, then margin/noise
        # ratio ascending so the shakiest non-tie calls sit just below them.
        _d = _d.sort_values(
            ["within_noise", "_ratio"], ascending=[False, True]
        ).reset_index(drop=True)
        _fig, _ax = _plt.subplots(
            figsize=(8.6, max(3.6, 0.34 * len(_d) + 1.2)))
        _yticks, _ylabels = [], []
        for _i, _r in _d.iterrows():
            _y = len(_d) - _i
            if bool(_r["within_noise"]):
                _ax.axhspan(_y - 0.42, _y + 0.42, color="#e34948",
                            alpha=0.07, zorder=0)
            _ax.plot([_r["margin"], _r["noise"]], [_y, _y], color="#cccccc",
                     linewidth=1.3, zorder=1)
            _ax.scatter(_r["noise"], _y, s=42, color=_NOISE_C, marker="D",
                        edgecolor="white", linewidth=0.5, zorder=3)
            _ax.scatter(_r["margin"], _y, s=44, color=_MARGIN_C, marker="o",
                        edgecolor="white", linewidth=0.5, zorder=4)
            _yticks.append(_y)
            _lab = f"{_r['model']} · {_r['metric']}"
            if bool(_r["within_noise"]):
                _lab += "  (tie)"
            _ylabels.append(_lab)
        _ax.set_yticks(_yticks)
        _ax.set_yticklabels(_ylabels, fontsize=7.5)
        _ax.set_ylim(0.3, len(_d) + 0.7)
        _ax.set_xlim(left=0)
        _ax.set_xlabel("display units (pct points; CIRL Net raw)", fontsize=9)
        _ax.set_title("Best-checkpoint margin vs measured rep noise",
                      fontsize=10)
        _ax.grid(True, alpha=0.25, linewidth=0.5, axis="x")
        for _sp in ("top", "right", "left"):
            _ax.spines[_sp].set_visible(False)
        _ax.tick_params(labelsize=8, length=0)
        _hs = [
            _plt.Line2D([], [], color=_MARGIN_C, marker="o", linestyle="none",
                        markersize=6, markeredgecolor="white",
                        label="best-vs-runner-up margin"),
            _plt.Line2D([], [], color=_NOISE_C, marker="D", linestyle="none",
                        markersize=6, markeredgecolor="white",
                        label="measured rep noise"),
        ]
        _ax.legend(handles=_hs, fontsize=8, frameon=False, loc="lower right")
        _fig.tight_layout()
        _fig.savefig(REPORT_DIR / "variance_margin_vs_noise.png",
                     dpi=150, bbox_inches="tight")
        print(f"saved {REPORT_DIR / 'variance_margin_vs_noise.png'}")
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase E — Longitudinal small multiples

    One panel per metric, x = epoch (0 = instruct anchor), one line per
    model. Color encodes the model *group* (hue) and size within group
    (lightness); markers disambiguate for CVD/print. CIRL-vig is excluded
    (single epoch — nothing longitudinal to show).

    Where the variance sweep (Phase D2) has ≥2 reps for a point, a thin
    whisker spans the **rep range**, centered on the canonical value —
    a noise-scale annotation, not a CI of that exact run (reps use
    different seeds and the server-mode engine path).
    """)
    return


@app.cell
def _(
    DISPLAY_ORDER,
    METRICS,
    METRIC_LABELS,
    REPORT_DIR,
    cells_df,
    np,
    variance_stats,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _shade(_hex, _f):
        """Blend a hex color toward white by fraction _f (0 = unchanged)."""
        _r, _g, _b = (int(_hex[i:i + 2], 16) for i in (1, 3, 5))
        return tuple((c + (255 - c) * _f) / 255 for c in (_r, _g, _b))

    # Hue = family group, lightness = size within group (small → light),
    # marker = model. Base hues from the validated categorical palette.
    _STYLE = {
        "Qwen3.5-2B":        (_shade("#2a78d6", 0.50), "o"),
        "Qwen3.5-4B":        (_shade("#2a78d6", 0.25), "s"),
        "Qwen3.5-9B":        (_shade("#2a78d6", 0.00), "^"),
        "Gemma-4-E2B":       (_shade("#008300", 0.50), "o"),
        "Gemma-4-E4B":       (_shade("#008300", 0.25), "s"),
        "Gemma-4-12B":       (_shade("#008300", 0.00), "^"),
        "Llama-3.1-8B":      (_shade("#eb6834", 0.30), "D"),
        "HARC-Llama-3.1-8B": (_shade("#eb6834", 0.00), "v"),
        "OpenThinker3-7B":   (_shade("#4a3aa7", 0.00), "P"),
        "Phi-4":             (_shade("#1baf7a", 0.00), "X"),
        "GPT-OSS-20B":       (_shade("#e34948", 0.00), "*"),
    }

    _plot_metrics = [m for m in METRICS if m[0] != "cirl_vig"]
    _ncols = 4
    _nrows = int(np.ceil(len(_plot_metrics) / _ncols))
    fig, _axes = plt.subplots(
        _nrows, _ncols, figsize=(3.4 * _ncols, 2.6 * _nrows), sharex=True
    )
    _axes = np.atleast_2d(_axes)

    for _i, _m in enumerate(_plot_metrics):
        _ax = _axes[_i // _ncols][_i % _ncols]
        _mid, _lo, _scale = _m[0], _m[8], _m[9]
        for _mdl in DISPLAY_ORDER:
            _sub = (cells_df[(cells_df.model == _mdl)
                             & (cells_df.metric == _mid)]
                    .sort_values("epoch"))
            if not len(_sub):
                continue
            _y = _sub["value"] * (100.0 if _scale == "pct" else 1.0)
            _c, _mk = _STYLE[_mdl]
            _ax.plot(_sub["epoch"], _y, color=_c, marker=_mk,
                     markersize=4.5, linewidth=1.6, label=_mdl,
                     markeredgecolor="white", markeredgewidth=0.5)
            # Noise whiskers from the variance sweep (rep range, centered
            # on the canonical point) — recessive: thin, translucent,
            # series-colored, no marker of their own.
            if len(variance_stats):
                _vs = variance_stats[
                    (variance_stats.model == _mdl)
                    & (variance_stats.metric == _mid)
                    & (variance_stats.n_reps >= 2)]
                if len(_vs):
                    _vlut = _vs.set_index("epoch")["rep_range_disp"]
                    _eps_v = [e for e in _sub["epoch"] if e in _vlut.index]
                    if _eps_v:
                        _yv = [float(_y[_sub["epoch"] == e].iloc[0])
                               for e in _eps_v]
                        _err = [float(_vlut[e]) / 2.0 for e in _eps_v]
                        _ax.errorbar(_eps_v, _yv, yerr=_err, fmt="none",
                                     ecolor=_c, alpha=0.45, elinewidth=1.0,
                                     capsize=2.0, capthick=1.0, zorder=1)
        _ax.set_title(METRIC_LABELS[_mid] + (" (lower better)" if _lo else ""),
                      fontsize=9)
        _ax.set_xticks([0, 1, 2, 3])
        _ax.grid(True, alpha=0.25, linewidth=0.5)
        for _sp in ("top", "right"):
            _ax.spines[_sp].set_visible(False)
        _ax.tick_params(labelsize=8)
    for _j in range(len(_plot_metrics), _nrows * _ncols):
        _axes[_j // _ncols][_j % _ncols].axis("off")
    for _ax in _axes[-1]:
        _ax.set_xlabel("epoch (0 = instruct)", fontsize=8)

    _handles, _labels = [], []
    for _mdl in DISPLAY_ORDER:
        _c, _mk = _STYLE[_mdl]
        _handles.append(plt.Line2D([], [], color=_c, marker=_mk,
                                   markersize=5, linewidth=1.6,
                                   markeredgecolor="white",
                                   markeredgewidth=0.5))
        _labels.append(_mdl)
    fig.legend(_handles, _labels, loc="center left",
               bbox_to_anchor=(1.0, 0.5), fontsize=8, frameon=False)
    fig.suptitle("SFT training trajectory — canonical cohort, "
                 "template-overhaul + DFT era (E0 = instruct)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.98, 0.97))
    fig.savefig(REPORT_DIR / "longitudinal_small_multiples.png",
                dpi=150, bbox_inches="tight")
    print(f"saved {REPORT_DIR / 'longitudinal_small_multiples.png'}")
    fig
    return


@app.cell
def _(REPORT_DIR, cells_df):
    # Tidy data for downstream use (paper figures, ad-hoc slicing).
    cells_df.to_parquet(REPORT_DIR / "cells.parquet", index=False)
    print(f"saved {REPORT_DIR / 'cells.parquet'} ({len(cells_df)} rows)")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md("""
    ## Phase F — W&B cross-check (optional)

    Fetches the cohort's runs from `uair/eval-all` (same API pattern as
    `fetch_wandb_runs.py`) and caches them next to the tables. Matched on
    `config.model.lora_path` containing the 2026-07-19 SFT training multirun
    plus the created-at cutoff that excludes the holed fragment sweeps.
    Checks: run states (crashed/running runs the disk scan can't see), and
    PrivacyLens summary numbers vs the disk metrics.

    Press the button to fetch (needs W&B auth); otherwise the last cache is
    used if present.
    """)
    return


@app.cell
def _(mo):
    wandb_fetch_btn = mo.ui.run_button(label="Fetch / refresh W&B runs")
    wandb_fetch_btn
    return (wandb_fetch_btn,)


@app.cell
def _(
    MODEL_SOURCE_TO_DISPLAY,
    REPORT_DIR,
    SFT_LORA_MARKER,
    WANDB_CREATED_CUTOFF,
    WANDB_ENTITY,
    WANDB_PROJECT,
    json,
    pd,
    re,
    wandb_fetch_btn,
):
    _cache = REPORT_DIR / "wandb_runs.json"

    if wandb_fetch_btn.value:
        import wandb as _wandb

        _api = _wandb.Api()
        # Server-side date filter keeps the pull small; lora_path matching
        # happens client-side (nested-config filters are unreliable).
        try:
            _raw = _api.runs(
                f"{WANDB_ENTITY}/{WANDB_PROJECT}",
                filters={"createdAt": {"$gte": "2026-07-20"}},
            )
            _raw = list(_raw)
        except Exception as _exc:
            print(f"filtered fetch failed ({_exc}); falling back to full pull")
            _raw = list(_api.runs(f"{WANDB_ENTITY}/{WANDB_PROJECT}"))
        _kept = []
        for _r in _raw:
            _mcfg = (_r.config or {}).get("model") or {}
            _lp = str(_mcfg.get("lora_path") or "")
            if SFT_LORA_MARKER not in _lp:
                continue
            _kept.append({
                "run_id": _r.id, "run_name": _r.name, "state": _r.state,
                "created_at": str(_r.created_at), "tags": list(_r.tags),
                "lora_path": _lp,
                "model_source": str(_mcfg.get("model_source") or ""),
                "summary": {k: v for k, v in _r.summary.items()
                            if not k.startswith("_")},
            })
        _cache.write_text(json.dumps(_kept, indent=1, default=str))
        print(f"fetched {len(_raw)} runs, kept {len(_kept)} cohort runs "
              f"→ {_cache}")

    if not _cache.exists():
        print("No W&B cache yet — press the fetch button above (or skip; "
              "the disk scan is the source of truth).")
        wandb_df = pd.DataFrame()
    else:
        _runs = json.loads(_cache.read_text())
        _rows = []
        for _r in _runs:
            if str(_r["created_at"]) < WANDB_CREATED_CUTOFF.replace(
                    "+00:00", ""):
                continue  # fragment-sweep run (same adapters, earlier launch)
            _mstep = re.search(r"checkpoint-(\d+)$", _r["lora_path"])
            _base = _r["model_source"].rstrip("/").split("/")[-1]
            _bench = next((t.split(":", 1)[1] for t in _r["tags"]
                           if t.startswith("bench:")), None)
            _rows.append({
                "run_id": _r["run_id"], "run_name": _r["run_name"],
                "state": _r["state"], "created_at": _r["created_at"],
                "model": MODEL_SOURCE_TO_DISPLAY.get(_base, _base),
                "step": int(_mstep.group(1)) if _mstep else None,
                "bench": _bench,
                "pl_qa": _r["summary"].get("compute_metrics/eval/qa_accuracy"),
                "pl_lk": _r["summary"].get("compute_metrics/eval/leakage_rate"),
                "pl_adjlk": _r["summary"].get(
                    "compute_metrics/eval/adjusted_leakage_rate"),
                "pl_helpful": _r["summary"].get(
                    "compute_metrics/eval/helpful_rate"),
            })
        wandb_df = pd.DataFrame(_rows)
        if len(wandb_df):
            print(f"{len(wandb_df)} cohort W&B runs after the created-at "
                  f"cutoff ({WANDB_CREATED_CUTOFF})")
            _bad_states = wandb_df[~wandb_df.state.isin(
                ["finished", "running"])]
            if len(_bad_states):
                print(f"!! {len(_bad_states)} runs in a bad state:")
                print(_bad_states[["model", "step", "bench", "state",
                                   "run_name"]].to_string())
            else:
                print("run states: "
                      f"{wandb_df.groupby('state').size().to_dict()}")
    wandb_df
    return (wandb_df,)


@app.cell
def _(cells_df, pd, wandb_df):
    # Numeric cross-check where a clean W&B↔disk key mapping exists
    # (PrivacyLens). Anything > 1e-6 apart is flagged.
    if not len(wandb_df):
        print("(no W&B cache — cross-check skipped)")
        wandb_xcheck = pd.DataFrame()
    else:
        _pl = wandb_df[wandb_df.bench == "privacylens"].copy()
        _pairs = [("pl_qa", "pl_qa"), ("pl_lk", "pl_lk"),
                  ("pl_adjlk", "pl_adjlk"), ("pl_helpful", "pl_helpful")]
        _rows = []
        for _, _wr in _pl.iterrows():
            for _wcol, _mid in _pairs:
                _wv = _wr[_wcol]
                if _wv is None or pd.isna(_wv):
                    continue
                _dm = cells_df[(cells_df.model == _wr["model"])
                               & (cells_df.step == _wr["step"])
                               & (cells_df.metric == _mid)]
                _dv = float(_dm["value"].iloc[0]) if len(_dm) else None
                _rows.append({
                    "model": _wr["model"], "step": _wr["step"],
                    "metric": _mid, "wandb": float(_wv), "disk": _dv,
                    "match": (_dv is not None
                              and abs(_dv - float(_wv)) < 1e-6),
                })
        wandb_xcheck = pd.DataFrame(_rows)
        if len(wandb_xcheck):
            _mism = wandb_xcheck[~wandb_xcheck["match"]]
            if len(_mism):
                print(f"!! {len(_mism)} W&B↔disk mismatches "
                      "(disk None = metrics not on disk yet, or a genuinely "
                      "different number — investigate before quoting):")
                print(_mism.to_string())
            else:
                print(f"OK: all {len(wandb_xcheck)} comparable PrivacyLens "
                      "values agree between W&B and disk")
    wandb_xcheck
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Read the result

    - **Comparative table** (Phase C, saved to
      `tables/sft_per_checkpoint_longitudinal_2026_07_20/comparative_table.md`):
      bold marks the best epoch *within each model* — the checkpoint-selection
      question. **Delta tables** (Phase D) give the same thing as signed
      gains over the instruct anchor.
    - **Longitudinal figure** (Phase E, `longitudinal_small_multiples.png`):
      the shape of training — where SFT keeps improving through epoch 3
      vs. where it peaks at epoch 1–2 and erodes, and what it costs on
      MMLU/GoldCoin. Whiskers = N=3 rep ranges from the variance sweep.
    - **Noise floor** (Phase D2, `variance_noise_floor.md`): per-metric
      run-to-run spread — quote a cross-checkpoint gap only if it clears
      the metric's range; Phase D2b flags which best-checkpoint calls are
      actually ties. Sampled (GoldCoin/VLM, seed-varied) and greedy
      (ConfAIde/CIRL-vig/MMLU, engine-nondeterminism) spreads are separate
      quantities. No noise estimate exists for the judged columns
      (PrivacyLens, CIRL-traj).
    - **Everything is template-overhaul + DFT era** — do not compare against
      keeper-era SFT numbers or the 07-17 `eval_canonical_sft_gemma4` rows
      (different adapters).
    - **CIRL-traj has no epoch-0 anchor** (instruct sweep ran vignette
      probing instead). If a zero-shot trajectory anchor is needed, run
      `cirl_trajectory_async` on the instruct checkpoints and re-scan.
    - Known format-discipline regressions from the canonical SFT
      (OpenThinker ReAct→JSON on PrivacyLens, GPT-OSS empty harmony final
      channel) will surface here as missing/degraded PrivacyLens cells at
      E1–E3 — findings, not infrastructure failures; leave them blank.
    - Re-run the notebook while the sweep finishes; Phase A2 tracks the
      32-cell roster. gemma-4-12b ckpt513 needs its follow-up sweep before
      this table is complete.
    """)
    return


if __name__ == "__main__":
    app.run()
