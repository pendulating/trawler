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
    # How far does each training stage actually move the weights?

    Built 2026-08-07 for the COLM 2026 camera-ready.
    Companion to `distilled_grounding.py`, which measures what the stages do to
    *behaviour*; this measures what they do to the *parameters*.

    ## The question

    SFT, GRPO, and KTO are three very different objectives applied in sequence to
    the same backbone. Do they move the weights by comparable amounts? The paper
    treats the RL stage as the one that installs normative reasoning, so it
    matters whether that stage is a large parameter change or a small one.

    This is directly measurable: all three camera-ready adapters are LoRA with
    **identical geometry** — `r=64`, `alpha=128`, the same 12 target modules,
    hence the same scaling `alpha/r = 2`. So `||ΔW||` is comparable across them
    without any normalisation games.

    ## What is being compared

    | stage | adapter | its base |
    |---|---|---|
    | SFT | `sft-canonical` | stock Qwen3.5-9B |
    | GRPO | `m2-full` checkpoint-450 | the merged SFT |
    | KTO | `k3-verdict` final | the merged SFT (byte-identical to GRPO's) |

    The updates are **sequential, not competing**: the RL deltas sit on top of a
    base that has already absorbed the SFT delta. So "GRPO is smaller than SFT"
    means the RL stage adds less on top, not that it lost a race.

    ## Method

    For a LoRA pair `(A, B)`, the effective update is `ΔW = (alpha/r)·B·A`. Every
    quantity below avoids materialising `ΔW` (which would be `out x in` per
    module, ~250 large matrices) by working through `r x r` Gram matrices:

    - `||ΔW||_F² = scale²·tr[(BᵀB)(AAᵀ)]`
    - `<ΔW₁, ΔW₂>_F = scale²·tr[(B₁ᵀB₂)(A₂A₁ᵀ)]`
    - singular values of `ΔW` = singular values of `R_B R_Aᵀ` (both `r x r`, from
      QR of `B` and `Aᵀ`), since `rank(ΔW) ≤ r`

    All exact, all cheap.
    """)
    return


@app.cell
def _():
    import glob
    import json
    import re
    from pathlib import Path

    import numpy as np
    import pandas as pd
    import torch
    from safetensors import safe_open

    PROJECT_ROOT = Path("/share/pierson/matt/UAIR")
    NB_DIR = PROJECT_ROOT / "notebooks/colm-camera-ready"
    TAB_DIR = NB_DIR / "tables/stage_weight_deltas"
    FIG_DIR = NB_DIR / "figures/stage_weight_deltas"
    for _d in (TAB_DIR, FIG_DIR):
        _d.mkdir(parents=True, exist_ok=True)

    BACKBONE = "/share/pierson/matt/zoo/models/Qwen3.5-9B"

    #: The three camera-ready adapters — the paths the model yamls under
    #: dagspaces/common/conf/model/qwen3.5-9b/ resolve to.
    ADAPTERS = {
        "SFT": PROJECT_ROOT / (
            "multirun/2026-07-15_sft_canonical_gemma4/00-07-44/2/"
            "sft_only/outputs/sft/checkpoint"),
        "GRPO": PROJECT_ROOT / (
            "multirun/2026-07-28_grpo_m2_full/21-31-11/cell=full/"
            "grpo_only_online_external/outputs/grpo/checkpoint/checkpoint-450"),
        "KTO": PROJECT_ROOT / (
            "multirun/2026-08-01_k3_arms_b/18-55-02/1/"
            "kto_only/outputs/kto/checkpoint"),
    }
    STAGES = ["SFT", "GRPO", "KTO"]

    #: Optimiser settings, read from each run's OWN frozen .hydra record rather
    #: than from the live conf/ tree, which has moved on since these runs.
    HPARAMS = {
        # NOT "DFT": the adapter analysed here is the 2026-07-15 run, whose
        # frozen .hydra/config.yaml carries no `loss_type`, i.e. TRL stock NLL.
        # `conf/training/sft/default.yaml` gained `loss_type: dft` on 2026-07-18,
        # AFTER this run — read the run record, not the current config.
        "SFT":  {"lr": 2.0e-5, "beta": None, "objective": "cross-entropy (NLL)"},
        "GRPO": {"lr": 2.0e-5, "beta": 0.02, "objective": "group-relative PG"},
        "KTO":  {"lr": 5.0e-6, "beta": 0.10, "objective": "prospect-theoretic"},
    }

    def save_table(df, name, index=True):
        out = TAB_DIR / f"{name}.csv"
        df.to_csv(out, index=index)
        print(f"[table] {out}")
        return df

    return (ADAPTERS, BACKBONE, FIG_DIR, HPARAMS, PROJECT_ROOT, STAGES, glob,
            json, np, pd, re, safe_open, save_table, torch)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Loading

    Adapter keys use the PEFT prefix `base_model.model.model.layers.*`; the
    backbone shards use `model.language_model.layers.*` (the Qwen3.5 VLM
    architecture). They must be remapped to join — the same mismatch
    `_remap_lora_keys_for_vlm` handles at inference time. Joining without the
    remap yields **zero matches silently**, which reads as "no data" rather than
    as an error, so the join is asserted below.
    """)
    return


@app.cell
def _(re, safe_open, torch):
    def canon_key(key: str) -> str:
        """PEFT adapter key -> backbone key namespace."""
        return key.replace("base_model.model.model.", "model.language_model.")

    def load_lora(path) -> dict:
        """{module: (A, B)} in float64, keys canonicalised to the backbone's."""
        _A, _B = {}, {}
        with safe_open(str(path / "adapter_model.safetensors"), framework="pt") as _f:
            for _k in _f.keys():
                _mod = canon_key(re.sub(r"\.lora_[AB]\.weight$", "", _k))
                if _k.endswith("lora_A.weight"):
                    _A[_mod] = _f.get_tensor(_k).double()
                elif _k.endswith("lora_B.weight"):
                    _B[_mod] = _f.get_tensor(_k).double()
        return {_m: (_A[_m], _B[_m]) for _m in sorted(set(_A) & set(_B))}

    def delta_fro(A, B, scale: float) -> float:
        """||scale·B·A||_F without forming B·A."""
        return scale * float(torch.sqrt(torch.clamp(
            torch.trace((B.T @ B) @ (A @ A.T)), min=0.0)))

    def delta_svals(A, B, scale: float):
        """Singular values of scale·B·A (exact; rank <= r)."""
        _, _rb = torch.linalg.qr(B, mode="reduced")     # B  = Q_B R_B
        _, _ra = torch.linalg.qr(A.T, mode="reduced")   # Aᵀ = Q_A R_A
        return scale * torch.linalg.svdvals(_rb @ _ra.T)

    def delta_dot(A1, B1, A2, B2, scale: float) -> float:
        """<scale·B₁A₁, scale·B₂A₂>_F without forming either product."""
        return scale * scale * float(torch.trace((B1.T @ B2) @ (A2 @ A1.T)))

    def participation_ratio(svals) -> float:
        """(Σσ)²/Σσ² — 1 = rank-one, r = flat spectrum.

        NOT the field-standard measure; reported alongside the two that are
        (`stable_rank`, `effective_rank`) so the notebook is comparable to
        published spectral analyses rather than to itself only.
        """
        _s = svals[svals > 0]
        return float(_s.sum() ** 2 / (_s ** 2).sum()) if len(_s) else 0.0

    def stable_rank(svals) -> float:
        """||A||_F² / ||A||_2² — the standard stable rank (Rudelson & Vershynin).

        Widely used in the DNN spectral literature; lower means more energy in
        the leading singular direction.
        """
        _s = svals[svals > 0]
        return float((_s ** 2).sum() / _s[0] ** 2) if len(_s) else 0.0

    def effective_rank(svals) -> float:
        """exp(H(p)) with p = σ/Σσ — Roy & Vetterli (2007) effective rank.

        The standard entropy-based definition. Quoted in the paper in preference
        to the participation ratio.
        """
        _s = svals[svals > 0]
        if len(_s) == 0:
            return 0.0
        _p = _s / _s.sum()
        return float(torch.exp(-(_p * torch.log(_p)).sum()))

    def module_type(m: str) -> str:
        return m.split(".")[-1]

    def module_layer(m: str) -> int:
        _g = re.search(r"layers\.(\d+)\.", m)
        return int(_g.group(1)) if _g else -1

    return (canon_key, delta_dot, delta_fro, delta_svals, effective_rank,
            load_lora, module_layer, module_type, participation_ratio,
            stable_rank)


@app.cell
def _(ADAPTERS, STAGES, json, load_lora):
    LORA = {_s: load_lora(ADAPTERS[_s]) for _s in STAGES}
    SCALE = {}
    for _st in STAGES:
        _cfg = json.loads((ADAPTERS[_st] / "adapter_config.json").read_text())
        SCALE[_st] = _cfg["lora_alpha"] / _cfg["r"]
        print(f"{_st:5s} r={_cfg['r']} alpha={_cfg['lora_alpha']} "
              f"scale={SCALE[_st]:.1f} modules={len(LORA[_st])}")

    # Geometry must match, or ||ΔW|| comparisons are meaningless.
    assert len(set(SCALE.values())) == 1, SCALE
    _sets = [set(LORA[_s]) for _s in STAGES]
    assert _sets[0] == _sets[1] == _sets[2], "adapters cover different modules"
    MODULES = sorted(_sets[0])
    print(f"\nidentical geometry across all three; {len(MODULES)} shared modules")
    return LORA, MODULES, SCALE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Total update magnitude""")
    return


@app.cell
def _(BACKBONE, MODULES, glob, re, safe_open):
    #: Depth-spread sample of layers for the relative denominator. Sampled
    #: rather than exhaustive because reading all 17.5 GB buys nothing — the
    #: ratio is stable across depth (see the depth profile below).
    SAMPLE_LAYERS = {0, 8, 16, 24, 32}
    _want = set(MODULES)

    BASE_NORMS = {}
    for _fp in sorted(glob.glob(BACKBONE + "/*.safetensors")):
        with safe_open(_fp, framework="pt") as _f:
            for _k in _f.keys():
                _lm = re.search(r"layers\.(\d+)\.", _k)
                if not _lm or int(_lm.group(1)) not in SAMPLE_LAYERS:
                    continue
                if not _k.endswith(".weight"):
                    continue
                _base = _k[: -len(".weight")]
                if _base not in _want:
                    continue
                BASE_NORMS[_base] = float(_f.get_tensor(_k).float().norm())

    assert BASE_NORMS, (
        "no adapter module matched a backbone tensor — the VLM key remap in "
        "canon_key() is wrong, and every relative figure would be empty"
    )
    print(f"base-weight norms for {len(BASE_NORMS)} modules "
          f"(layers {sorted(SAMPLE_LAYERS)})")
    return BASE_NORMS, SAMPLE_LAYERS


@app.cell
def _(BASE_NORMS, LORA, MODULES, SCALE, STAGES, delta_fro, np, pd, save_table):
    PER_MODULE = {}
    _rows = []
    for _s in STAGES:
        _d = {_m: delta_fro(*LORA[_s][_m], SCALE[_s]) for _m in MODULES}
        PER_MODULE[_s] = _d
        _rel = np.array([_d[_m] / BASE_NORMS[_m]
                         for _m in _d if _m in BASE_NORMS])
        _rows.append({
            "stage": _s,
            "n_modules": len(_d),
            "total_fro": float(np.sqrt(sum(_v ** 2 for _v in _d.values()))),
            "rel_median_pct": float(np.median(_rel)) * 100,
            "rel_mean_pct": float(_rel.mean()) * 100,
            "rel_max_pct": float(_rel.max()) * 100,
        })
    TOTALS = pd.DataFrame(_rows).set_index("stage")
    TOTALS["vs_SFT"] = TOTALS["total_fro"] / TOTALS.loc["SFT", "total_fro"]
    save_table(TOTALS, "totals")
    TOTALS
    return PER_MODULE, TOTALS


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### Is this just the learning-rate budget?

    A crude expectation for how far a stage travels is `lr x optimiser steps`.
    Comparing that budget against what actually landed isolates the part of the
    difference the schedule does *not* explain. `damping < 1` means the stage
    travelled **less** than its budget predicts.
    """)
    return


@app.cell
def _(ADAPTERS, HPARAMS, STAGES, TOTALS, json, pd, save_table):
    def final_steps(path):
        """Optimiser steps from the run's own trainer_state, not from config."""
        _cands = sorted(path.glob("checkpoint-*"),
                        key=lambda p: int(p.name.rsplit("-", 1)[1]))
        for _p in [path] + _cands[::-1]:
            _f = _p / "trainer_state.json"
            if _f.exists():
                _st = json.loads(_f.read_text())
                return int(_st["global_step"]), float(_st.get("epoch", float("nan")))
        return None, None

    _rows = []
    for _s in STAGES:
        _steps, _epochs = final_steps(ADAPTERS[_s])
        _hp = HPARAMS[_s]
        _rows.append({
            "stage": _s,
            "objective": _hp["objective"],
            "lr": _hp["lr"],
            "beta_kl": _hp["beta"],
            "steps": _steps,
            "epochs": _epochs,
            "budget": _hp["lr"] * (_steps or 0),
            "measured_fro": TOTALS.loc[_s, "total_fro"],
        })
    BUDGET = pd.DataFrame(_rows).set_index("stage")
    BUDGET["budget_vs_SFT"] = BUDGET["budget"] / BUDGET.loc["SFT", "budget"]
    BUDGET["measured_vs_SFT"] = (BUDGET["measured_fro"]
                                 / BUDGET.loc["SFT", "measured_fro"])
    BUDGET["damping"] = BUDGET["measured_vs_SFT"] / BUDGET["budget_vs_SFT"]
    save_table(BUDGET, "budget_vs_measured")
    BUDGET
    return BUDGET, final_steps


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Where the update lands: by module type and by depth

    If RL were learning something structurally different from SFT, the *shape*
    of the update should differ — different module types, different depths. If
    it is the same direction taken more cautiously, the shape should be a scaled
    copy. The Spearman ρ between per-module profiles quantifies that.
    """)
    return


@app.cell
def _(MODULES, PER_MODULE, STAGES, module_layer, module_type, np, pd,
      save_table):
    _rows = []
    for _t in sorted({module_type(_m) for _m in MODULES}):
        _row = {"module": _t}
        for _s in STAGES:
            _row[_s] = float(np.sqrt(sum(
                PER_MODULE[_s][_m] ** 2 for _m in MODULES
                if module_type(_m) == _t)))
        _rows.append(_row)
    BY_TYPE = pd.DataFrame(_rows).set_index("module").sort_values(
        "SFT", ascending=False)
    BY_TYPE["GRPO/SFT"] = BY_TYPE["GRPO"] / BY_TYPE["SFT"]
    BY_TYPE["KTO/SFT"] = BY_TYPE["KTO"] / BY_TYPE["SFT"]
    save_table(BY_TYPE, "by_module_type")

    _rows = []
    for _L in sorted({module_layer(_m) for _m in MODULES if module_layer(_m) >= 0}):
        _row = {"layer": _L}
        for _s in STAGES:
            _row[_s] = float(np.sqrt(sum(
                PER_MODULE[_s][_m] ** 2 for _m in MODULES
                if module_layer(_m) == _L)))
        _rows.append(_row)
    BY_LAYER = pd.DataFrame(_rows).set_index("layer")
    save_table(BY_LAYER, "by_layer")

    print("Spearman rho of the per-module magnitude profile (shape agreement):")
    for _a, _b in (("SFT", "GRPO"), ("SFT", "KTO"), ("GRPO", "KTO")):
        _x = np.array([PER_MODULE[_a][_m] for _m in MODULES])
        _y = np.array([PER_MODULE[_b][_m] for _m in MODULES])
        _rho = float(np.corrcoef(_x.argsort().argsort(),
                                 _y.argsort().argsort())[0, 1])
        print(f"  {_a:5s} vs {_b:5s}  rho = {_rho:.4f}")
    BY_TYPE
    return BY_LAYER, BY_TYPE


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Direction: does RL continue SFT's update, or rotate away from it?

    Magnitude cannot answer whether the RL stage is "more of the same". The
    Frobenius cosine between updates can:

    - `cos ≈ +1` — RL pushes further along the direction SFT already took
    - `cos ≈ 0` — RL moves in a direction SFT left untouched (orthogonal)
    - `cos < 0` — RL partially *undoes* SFT

    The two are anchored at different points (RL starts from the merged SFT), so
    this compares update *directions* in a shared parameter space, not two paths
    from one origin.

    **A cosine near zero is the null, not a finding.** Two unrelated directions
    in a space this large are near-orthogonal by construction, so the raw number
    is uninterpretable on its own. The next cell measures the chance level
    empirically — cosines between *mismatched* modules of the same shape, which
    share every structural property except being the same parameter block — and
    the real pairs are only informative relative to that.
    """)
    return


@app.cell
def _(LORA, MODULES, PER_MODULE, SCALE, delta_dot, np, pd, save_table):
    _rows = []
    for _a, _b in (("SFT", "GRPO"), ("SFT", "KTO"), ("GRPO", "KTO")):
        _cos, _num = [], 0.0
        for _m in MODULES:
            _A1, _B1 = LORA[_a][_m]
            _A2, _B2 = LORA[_b][_m]
            _dot = delta_dot(_A1, _B1, _A2, _B2, SCALE[_a])
            _num += _dot
            _na, _nb = PER_MODULE[_a][_m], PER_MODULE[_b][_m]
            if _na > 0 and _nb > 0:
                _cos.append(_dot / (_na * _nb))
        _den = (np.sqrt(sum(PER_MODULE[_a][_m] ** 2 for _m in MODULES))
                * np.sqrt(sum(PER_MODULE[_b][_m] ** 2 for _m in MODULES)))
        _cos = np.array(_cos)
        _rows.append({
            "pair": f"{_a} vs {_b}",
            "global_cos": _num / _den,
            "per_module_mean": float(_cos.mean()),
            "per_module_median": float(np.median(_cos)),
            "frac_positive": float((_cos > 0).mean()),
        })
    COSINE = pd.DataFrame(_rows).set_index("pair")
    save_table(COSINE, "update_direction_cosine")
    COSINE
    return (COSINE,)


@app.cell
def _(COSINE, LORA, MODULES, PER_MODULE, SCALE, delta_dot, np, pd, save_table):
    # Empirical chance level: same two adapters, but pair module i of one with a
    # DIFFERENT module j of the other, restricted to identical shape. Those two
    # updates share dimensionality, module type, and training run — everything
    # except being the same parameter block — so their cosine is what "unrelated"
    # looks like here.
    _by_shape = {}
    for _m in MODULES:
        _A, _B = LORA["SFT"][_m]
        _by_shape.setdefault((_B.shape[0], _A.shape[1]), []).append(_m)

    _rng = np.random.default_rng(0)
    _rows = []
    for _a, _b in (("SFT", "GRPO"), ("SFT", "KTO"), ("GRPO", "KTO")):
        _null = []
        for _shape, _group in _by_shape.items():
            if len(_group) < 2:
                continue
            for _m in _group:
                _other = _rng.choice([_o for _o in _group if _o != _m])
                _A1, _B1 = LORA[_a][_m]
                _A2, _B2 = LORA[_b][_other]
                _na, _nb = PER_MODULE[_a][_m], PER_MODULE[_b][_other]
                if _na > 0 and _nb > 0:
                    _null.append(
                        delta_dot(_A1, _B1, _A2, _B2, SCALE[_a]) / (_na * _nb))
        _null = np.abs(np.array(_null))
        _obs = abs(COSINE.loc[f"{_a} vs {_b}", "per_module_median"])
        _rows.append({
            "pair": f"{_a} vs {_b}",
            "observed_abs_cos": _obs,
            "null_abs_cos_median": float(np.median(_null)),
            "null_abs_cos_p95": float(np.quantile(_null, 0.95)),
            "n_null": len(_null),
            # >1 would mean the real pairing is more aligned than chance.
            "obs_over_null": _obs / float(np.median(_null)),
        })
    COSINE_NULL = pd.DataFrame(_rows).set_index("pair")
    save_table(COSINE_NULL, "update_direction_cosine_null")
    COSINE_NULL
    return (COSINE_NULL,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Spectral concentration

    `ΔW` is rank-`r` by construction, so a hard rank says nothing. The
    participation ratio `(Σσ)²/Σσ²` says how much of the update lives in a few
    directions: 1 means rank-one, 64 means a flat spectrum across the full
    budget. A stage concentrating its update in fewer directions is making a
    more targeted edit.
    """)
    return


@app.cell
def _(LORA, MODULES, SCALE, STAGES, delta_svals, effective_rank, np,
      participation_ratio, pd, save_table, stable_rank):
    SPECTRA = {}
    _rows = []
    for _s in STAGES:
        _pr, _sr, _er, _top1 = [], [], [], []
        for _m in MODULES:
            _sv = delta_svals(*LORA[_s][_m], SCALE[_s])
            _pr.append(participation_ratio(_sv))
            _sr.append(stable_rank(_sv))
            _er.append(effective_rank(_sv))
            _tot = float((_sv ** 2).sum())
            _top1.append(float(_sv[0] ** 2 / _tot) if _tot > 0 else np.nan)
        SPECTRA[_s] = np.array(_pr)
        _rows.append({
            "stage": _s,
            # the two standard measures, quoted in preference to the third
            "stable_rank_median": float(np.median(_sr)),
            "effective_rank_median": float(np.median(_er)),
            "participation_median": float(np.median(_pr)),
            "top1_energy_median": float(np.median(_top1)),
        })
    SPECTRUM = pd.DataFrame(_rows).set_index("stage")
    save_table(SPECTRUM, "spectral_concentration")
    SPECTRUM
    return SPECTRA, SPECTRUM


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## The dissociation

    Joined against the behavioural result from `distilled_grounding.py`: κ
    between an arm's own ungrounded appropriateness label and the norm-grounded
    one, on the 503-chunk doubly-held-out set. `base` has no adapter, so no ΔW
    row — it is the κ floor the SFT delta is measured against.
    """)
    return


@app.cell
def _(PROJECT_ROOT, TOTALS, pd, save_table):
    _p = (PROJECT_ROOT / "notebooks/colm-camera-ready/tables/"
          "distilled_grounding/per_arm_by_chunk_set.csv")
    DISSOC = None
    if _p.exists():
        _k = pd.read_csv(_p)
        _k = _k[_k.chunk_set == "double-heldout"].set_index("arm")["kappa"]
        DISSOC = pd.DataFrame({
            "delta_fro": [float("nan"),
                          TOTALS.loc["SFT", "total_fro"],
                          TOTALS.loc["GRPO", "total_fro"],
                          TOTALS.loc["KTO", "total_fro"]],
            "kappa": [_k.get("base"), _k.get("sft"),
                      _k.get("m2-full"), _k.get("k3-verdict")],
        }, index=["base (no adapter)", "SFT", "GRPO", "KTO"])
        DISSOC["kappa_gain_over_prev"] = DISSOC["kappa"].diff()
        save_table(DISSOC, "delta_vs_kappa")
    else:
        print(f"distilled_grounding tables not built: {_p}")
    DISSOC
    return (DISSOC,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""## Figure""")
    return


@app.cell
def _(BY_LAYER, FIG_DIR, SPECTRA, STAGES, TOTALS):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "DejaVu Serif"],
        "font.size": 9,
        "axes.titlesize": 9,
        "legend.fontsize": 7.5,
        "text.usetex": False,
    })
    COLORS = {"SFT": "#4C72B0", "GRPO": "#B4443B", "KTO": "#8172B2"}

    def save_fig(fig, name, pad_inches=0.0):
        for _ext in ("png", "pdf"):
            fig.savefig(FIG_DIR / f"{name}.{_ext}", dpi=300,
                        bbox_inches="tight", pad_inches=pad_inches)
        print(f"[fig] {FIG_DIR / name}.png|.pdf")

    _fig, _axes = plt.subplots(1, 3, figsize=(9.4, 2.8))

    _ax = _axes[0]
    _ax.bar(STAGES, [TOTALS.loc[_s, "total_fro"] for _s in STAGES],
            color=[COLORS[_s] for _s in STAGES])
    for _i, _s in enumerate(STAGES):
        _ax.text(_i, TOTALS.loc[_s, "total_fro"],
                 f"{TOTALS.loc[_s, 'total_fro']:.2f}", ha="center", va="bottom",
                 fontsize=8)
    _ax.set_ylabel(r"total $\|\Delta W\|_F$")
    _ax.set_title("update magnitude")

    _ax = _axes[1]
    for _s in STAGES:
        _ax.plot(BY_LAYER.index, BY_LAYER[_s], color=COLORS[_s], lw=1.3,
                 label=_s)
    _ax.set_xlabel("layer")
    _ax.set_ylabel(r"$\|\Delta W\|_F$")
    _ax.set_title("depth profile")
    _ax.legend(frameon=False)

    _ax = _axes[2]
    _bp = _ax.boxplot([SPECTRA[_s] for _s in STAGES], showfliers=False,
                      medianprops=dict(color="black"))
    _ax.set_xticklabels(STAGES)
    _ax.set_ylabel("participation ratio")
    _ax.set_title("spectral concentration")

    for _a in _axes:
        _a.spines[["top", "right"]].set_visible(False)
    _fig.tight_layout()
    save_fig(_fig, "fig_stage_weight_deltas")
    _fig
    return COLORS, matplotlib, plt, save_fig


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Depth profile as a replication

    Two prior results claim SFT concentrates its update in the upper layers
    while RL spreads its update evenly across depth. We test both here on our
    own runs, so the claim in the paper rests on our numbers rather than on
    theirs.
    """)
    return


@app.cell
def _(BY_LAYER, STAGES, np, pd, save_table):
    _n = len(BY_LAYER)
    _z = BY_LAYER / BY_LAYER.mean()          # shape only; magnitude divided out
    _thirds = {"lower": slice(0, _n // 3),
               "middle": slice(_n // 3, 2 * _n // 3),
               "upper": slice(2 * _n // 3, _n)}
    _rows = []
    for _s in STAGES:
        _row = {"stage": _s,
                # Pearson r against layer index: is the update depth-graded?
                "corr_with_depth": float(np.corrcoef(np.arange(_n),
                                                     BY_LAYER[_s])[0, 1]),
                # CV of the shape-only profile: how uneven across depth
                "cv_across_depth": float(_z[_s].std() / _z[_s].mean())}
        for _name, _sl in _thirds.items():
            _row[f"norm_{_name}"] = float(_z[_s].iloc[_sl].mean())
        _rows.append(_row)
    DEPTH = pd.DataFrame(_rows).set_index("stage")
    DEPTH["flatter_than_SFT"] = (DEPTH.loc["SFT", "cv_across_depth"]
                                 / DEPTH["cv_across_depth"])
    save_table(DEPTH, "depth_profile_replication")
    DEPTH
    return (DEPTH,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Positioning against prior parameter-space analyses

    A parallel literature compares SFT and RL directly in parameter space. We
    read our measurements against it for two reasons: to establish which of our
    statistics are standard, and to separate what we replicate from what is new.

    **Task vectors are the standard object, and near-orthogonality is the
    standard finding.** Ilharco et al. define a 'task vector' as the difference
    between fine-tuned and base parameters, then measure its norm and its cosine
    against other task vectors. Each of our per-stage updates is a task vector in
    exactly this sense, so both `||ΔW||_F` and the pairwise cosines are standard
    instruments. Ilharco et al. also report that task vectors for different tasks
    sit close to orthogonal. Our cosines fall at or below an empirical chance
    level (0.46 to 0.53 times the null median, against a null 95th percentile
    roughly 30 times larger than any observed value), which is consistent with
    their result rather than a departure from it. What the test does license is
    narrower and still useful: it can detect strong alignment, and it finds none,
    so we can say the RL stage does not continue the direction SFT established.

    **Two of our three spectral measures are standard; one is ours.** Stable rank
    and the entropy-based effective rank of Roy and Vetterli are established
    descriptors of a spectrum. The participation ratio is our own choice, and we
    report it only beside the other two. All three agree on the ordering, so the
    conclusion does not depend on the choice: SFT produces the most concentrated
    update (stable rank 1.63, with 61.4% of the squared energy in the leading
    singular direction); GRPO the most diffuse (3.79, and 26.4%); and KTO between
    them (1.83, and 54.8%).

    **We replicate the depth profile on a new model family.** Prior work reports
    that SFT concentrates its update in the upper layers, whereas RL touches each
    layer about equally. Our SFT update rises almost monotonically with layer
    index (Pearson r = 0.966), carrying 1.22 times the mean norm in the upper
    third against 0.73 in the lower. Both RL stages are far flatter: GRPO's
    coefficient of variation across depth is 0.045 against SFT's 0.225, i.e. 5.0
    times flatter, and KTO's is 0.074. We observe this on Qwen3.5-9B with a
    GRPO/KTO pair, where the prior reports use different families and different
    objectives, so we read it as independent support for a general property of
    the two objectives.

    **The magnitude ordering agrees, but our instrument differs.** Mukherjee et
    al. report that RL updates only 5% to 30% of parameters while SFT updates are
    dense, across seven RL algorithms (including GRPO) and ten models. Our RL
    updates are likewise much smaller than our SFT update: 0.45 times for GRPO
    and 0.15 times for KTO, by Frobenius norm. However, we cannot test their
    claim. Update sparsity counts parameters whose value moves; every stage we
    train is LoRA, whose update is dense within a rank-64 factorization by
    construction, so the sparsity statistic is undefined here rather than merely
    unmeasured. For the same reason we cannot evaluate their finding that RL
    updates are close to full-rank, since our rank is capped at 64 a priori.
    **We therefore make no sparsity claim.**

    **A LoRA-specific confound applies to all three of our stages.**
    Shuttleworth et al. show that LoRA introduces 'intruder dimensions', high-
    ranking singular vectors absent from the pre-trained model, which full
    fine-tuning does not produce, and demonstrate causally that these dimensions
    drive forgetting. Every stage we report is LoRA at rank 64, so all three may
    carry intruder dimensions, and the general-knowledge degradation we observe
    on MMLU after treatment is consistent with that mechanism. We flag this as a
    limitation on the generality of the comparison: what we measure is how three
    objectives behave under a shared low-rank parameterization, not how they
    would behave under full fine-tuning.

    **Two measurements here appear to be new.** First, we are not aware of a
    prior comparison of GRPO and KTO updates trained from a byte-identical base:
    ours are mutually near-orthogonal, yet the two policies generate identical
    reasoning text on 91.7% of 2,993 chunks, so orthogonal parameter updates need
    not produce distinguishable behavior. Second, prior work relates weight
    travel to task accuracy, whereas we relate it to a construct-alignment
    measure, Cohen's kappa between a policy's own ungrounded appropriateness
    judgment and the norm-grounded one. On that measure the ordering inverts:
    GRPO moves the weights 0.45 times as far as SFT and buys 2.0 times the kappa
    gain, which is approximately 4.4 times the alignment per unit of parameter
    travel.

    **Collectively, these comparisons support a direction-over-magnitude
    reading.** Two prior analyses of SFT-then-RL pipelines find that singular
    values shift by at most 0.005 during fine-tuning while singular vectors
    rotate by 25 to 90 degrees, and that the directional shift, not the change in
    magnitude, governs downstream behavior. Our dissociation is the same
    phenomenon observed through a different instrument: the stage that moves the
    weights least improves norm alignment most.

    ### References

    - Ilharco et al., 'Editing Models with Task Arithmetic', ICLR 2023
      (`arXiv:2212.04089`)
    - Mukherjee et al., 'Reinforcement Learning Finetunes Small Subnetworks in
      Large Language Models', NeurIPS 2025 (`arXiv:2505.11711`)
    - Shuttleworth et al., 'LoRA vs Full Fine-tuning: An Illusion of
      Equivalence', NeurIPS 2025 (`arXiv:2410.21228`)
    - Chu et al., 'SFT Memorizes, RL Generalizes: A Comparative Study of
      Foundation Model Post-training', ICML 2025 (`arXiv:2501.17161`)
    - 'RL Fine-Tuning Heals OOD Forgetting in SFT' (`arXiv:2509.12235`)
    - 'RL Is Neither a Panacea Nor a Mirage' (`arXiv:2508.16546`)
    - Roy and Vetterli, 'The effective rank: a measure of effective
      dimensionality', EUSIPCO 2007
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Findings

    **1. SFT moves the weights far more than either RL stage.** Total
    `||ΔW||_F`: SFT 6.08, GRPO 2.72 (0.45x), KTO 0.88 (0.15x). As a fraction of
    the base weights that is 0.41% / 0.22% / 0.07% (median over sampled modules).

    **2. Only about half of that gap is the learning-rate schedule.** An
    `lr x steps` budget predicts GRPO at 0.83x and KTO at 0.29x of SFT; measured
    is 0.45x and 0.15x. Both RL stages land at `damping ≈ 0.50` — they travel
    half as far as their budget allows. Two forces account for it, and both are
    specific to RL: group-relative advantage is mean-zero by construction, so
    gradients cancel when a group's rewards are flat (m2-full *failed* its
    `reward_trend` gate at +0.0057 against a 0.02 bar — that flatness is visible
    in the weights); and the KL term (β=0.02 GRPO, β=0.1 KTO) explicitly pulls
    ΔW toward zero. SFT's DFT cross-entropy has neither.

    **3. All three allocate magnitude across modules almost identically**
    (Spearman ρ 0.92–0.94 on the per-module profile; `gate_proj`/`up_proj`
    largest, `k_proj`/`v_proj` smallest, in the same order every time). Where
    the update goes is a property of the architecture, not the objective.

    **4. But the directions are mutually orthogonal.** Observed |cos| is
    0.00007–0.0001 against an empirical chance level of 0.00014–0.00019
    (mismatched same-shape modules) — i.e. `obs/null ≈ 0.46–0.53`, at or below
    chance, and far under the null's 95th percentile of ~0.003. The test *can*
    detect strong alignment and finds none. So the RL stage is **not** "SFT's
    direction, taken more cautiously"; it edits a direction SFT left alone. The
    same holds between GRPO and KTO, despite those two producing 91.7% identical
    text.

    **5. The updates have different shapes.** SFT is the most concentrated
    (participation ratio 21.8, top singular direction carrying 61% of the
    energy); GRPO is the most diffuse (41.6, top direction only 26%); KTO sits
    between (27.8, 55%). SFT makes a large, nearly rank-one edit per module;
    GRPO makes a smaller edit spread across many directions — again what the two
    objectives predict, a single consistent target versus many weakly-correlated
    advantage-weighted pushes.

    **6. The dissociation.** Weight travel and behavioural gain run *opposite*:

    | stage | `||ΔW||_F` | κ (double-heldout) | 95% CI | κ gain |
    |---|---|---|---|---|
    | base | — | 0.029 | [−0.010, +0.064] | — |
    | SFT | 6.08 | 0.060 | [+0.041, +0.078] | +0.030 |
    | GRPO | 2.72 | 0.120 | [+0.082, +0.145] | +0.060 |
    | KTO | 0.88 | 0.110 | [+0.075, +0.140] | −0.010 vs GRPO |

    The base CI straddles zero: without an adapter the backbone's agreement
    with grounding is what its label prior alone would produce.

    GRPO buys **2.0x the κ gain for 0.45x the weight change** — roughly 4.4x the
    alignment per unit of parameter travel. That is mechanistically sensible
    rather than paradoxical: SFT imitates the teacher's *ungrounded* labels,
    which are themselves only κ≈0.07 with the grounded label, so a large step
    toward a nearly-unrelated target cannot buy alignment. RL's reward is
    *derived from* the grounding, so a small, KL-anchored, well-aimed step moves
    it efficiently. **Direction matters more than magnitude** — and finding (4)
    says that direction is genuinely new, not more of SFT.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Caveats

    - **`||ΔW||_F` measures parameter travel, not functional change.** A small
      update concentrated in a sensitive subspace can move behaviour more than a
      large diffuse one. Nothing here licenses "SFT changed the model more" in
      any behavioural sense — and `distilled_grounding.py` shows the opposite on
      norm alignment, which is the point of the dissociation section.
    - **The relative figures use a sampled denominator** (5 layers of the stock
      backbone). The RL stages' true base is the merged SFT, which differs from
      stock by well under 1%, so the ratios hold at the precision reported.
    - **Sequential, not competing.** The RL deltas are measured from a base that
      already contains the SFT delta.
    - **One checkpoint per stage.** These are endpoints, not trajectories. A
      stage that travelled far and came back would look identical to one that
      barely moved.
    - **KTO's `k3-verdict` is the label-only rung** of the supervision-depth
      ladder. The `citation` and `scrutinize` arms are not measured here and may
      sit elsewhere on this scale.
    """)
    return


if __name__ == "__main__":
    app.run()
