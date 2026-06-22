#!/usr/bin/env python3
"""Generate publishable artifacts for the top100-corpus scaling appendix.

Faithful de-marimo'd reproduction of the analysis in
  norm_distribution_top100_vs_fiction10_2026_06.py  and
  norm_yield_gap_qwen36_2026_05.py
that SAVES figures + tables to disk (the marimo notebooks render inline only).

Compares two Raz-norm extractions:
  fiction10 (Qwen2.5-72B-AWQ, 10 books)  vs  top100 (Qwen3.6-27B, 97 books).

Outputs -> notebooks/normative-simulacra/tables/top100_scaling_2026_06/
  corpus_summary.csv         headline corpus/extraction stats
  axis_divergence.csv        per-axis TVD/JSD, pooled and same-7-book overlap
  top100_norm_attributes.pdf multi-panel attribute distributions (pooled)
  top100_divergence_axes.pdf JSD by axis: pooled (model+corpus) vs overlap (model only)
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

F10 = Path("/share/pierson/matt/n2s4cir/data/fiction10/abstracted_norms.parquet")
TOP = Path(
    "/share/pierson/matt/UAIR/multirun/2026-05-26_historical_norms/05-00-38/0"
    "/role_abstraction_standalone_qwen36/outputs/role_abstraction/abstracted_norms.parquet"
)
CHUNKS = Path("/share/pierson/matt/zoo/datasets/gutenberg_cache/chunks_top100_fiction_en.parquet")

OUT = Path(__file__).resolve().parent / "tables" / "top100_scaling_2026_06"
OUT.mkdir(parents=True, exist_ok=True)

COLOR = {"fiction10": "#1f77b4", "top100": "#ff7f0e"}
LABEL = {
    "fiction10": "fiction10 (Qwen2.5-72B, 10 books)",
    "top100": "top100 (Qwen3.6-27B, 97 books)",
}


def load(path, corpus):
    df = pd.read_parquet(path)
    df["corpus"] = corpus
    df["gutenberg_id"] = df["gutenberg_id"].astype(str)
    return df


f_raw = load(F10, "fiction10")
t_raw = load(TOP, "top100")
shared = [c for c in f_raw.columns if c in set(t_raw.columns) and c != "corpus"]
combined = pd.concat(
    [f_raw[shared + ["corpus"]], t_raw[shared + ["corpus"]]], ignore_index=True
)
# valid norm = has a deontic force assigned
norms = combined[combined["raz_normative_force"].notna()].copy()
chunks = pd.read_parquet(CHUNKS, columns=["gutenberg_id", "chunk_id"])


def axis_table(field, data=None):
    src = norms if data is None else data
    sub = src[[field, "corpus"]].copy()
    sub = sub[sub[field].notna()]
    ct = sub.groupby(["corpus", field]).size().unstack("corpus", fill_value=0)
    for c in ("fiction10", "top100"):
        if c not in ct.columns:
            ct[c] = 0
    ct = ct[["fiction10", "top100"]]
    prop = ct.div(ct.sum(axis=0), axis=1)
    out = pd.DataFrame(
        {"fiction10_p": prop["fiction10"], "top100_p": prop["top100"]}
    )
    return out.sort_values("fiction10_p", ascending=False)


def divergence(field, data=None):
    t = axis_table(field, data=data)
    p, q = t["fiction10_p"].to_numpy(), t["top100_p"].to_numpy()
    tvd = 0.5 * np.abs(p - q).sum()

    def kl(a, b):
        m = a > 0
        return np.sum(a[m] * np.log2(a[m] / b[m]))

    mid = 0.5 * (p + q)
    jsd = 0.5 * kl(p, mid) + 0.5 * kl(q, mid)
    return dict(field=field, tvd=tvd, jsd=jsd)


def plot_axis(ax, field, title, order=None, data=None):
    t = axis_table(field, data=data)
    if order is not None:
        t = t.reindex([c for c in order if c in t.index])
    cats = [str(c) for c in t.index]
    y = np.arange(len(cats))
    h = 0.38
    ax.barh(y - h / 2, t["fiction10_p"], h, label=LABEL["fiction10"], color=COLOR["fiction10"])
    ax.barh(y + h / 2, t["top100_p"], h, label=LABEL["top100"], color=COLOR["top100"])
    ax.set_yticks(y)
    ax.set_yticklabels(cats, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("proportion of norms")
    d = divergence(field, data=data)
    ax.set_title(f"{title}\nTVD={d['tvd']:.3f}  JSD={d['jsd']:.3f}", fontsize=10)
    ax.grid(True, axis="x", alpha=0.3)


# ---- corpus summary ----
def summarize(df, name, n_chunks=None):
    valid = df[df["raz_normative_force"].notna()]
    nb = df["gutenberg_id"].nunique()
    return dict(
        corpus=name,
        books=nb,
        chunks=n_chunks if n_chunks is not None else np.nan,
        valid_norms=len(valid),
        norms_per_book=len(valid) / nb,
        info_flow_rate=(valid["raz_governs_info_flow"] == True).mean(),
        quality_pass_rate=(valid["norm_quality_passed"] == True).mean(),
    )


summary = pd.DataFrame(
    [
        summarize(f_raw, "fiction10"),  # fiction10 chunk count comes from the paper table (2,216)
        summarize(t_raw, "top100", n_chunks=len(chunks)),
    ]
)
summary.loc[summary["corpus"] == "fiction10", "chunks"] = 2216
summary.to_csv(OUT / "corpus_summary.csv", index=False)
print("=== corpus_summary ===")
print(summary.to_string(index=False))

# ---- same-book overlap (model-only effect) ----
f10_books = set(norms.loc[norms["corpus"] == "fiction10", "gutenberg_id"])
top_books = set(norms.loc[norms["corpus"] == "top100", "gutenberg_id"])
overlap_ids = sorted(f10_books & top_books, key=int)
norms_overlap = norms[norms["gutenberg_id"].isin(overlap_ids)].copy()
print(f"\noverlap books: {len(overlap_ids)}")

AXES = [
    "raz_normative_force",
    "raz_governs_info_flow",
    "raz_norm_source",
    "raz_context",
    "raz_confidence_qual",
    "norm_quality_passed",
]
rows = []
for f in AXES:
    dp = divergence(f)
    do = divergence(f, data=norms_overlap)
    rows.append(
        dict(field=f, jsd_pooled=dp["jsd"], tvd_pooled=dp["tvd"],
             jsd_overlap=do["jsd"], tvd_overlap=do["tvd"],
             ratio_overlap_over_pooled=do["jsd"] / dp["jsd"] if dp["jsd"] else np.nan)
    )
axis_div = pd.DataFrame(rows).sort_values("jsd_pooled", ascending=False).reset_index(drop=True)
axis_div.to_csv(OUT / "axis_divergence.csv", index=False)
print("\n=== axis_divergence ===")
print(axis_div.to_string(index=False))

# ---- Figure A: attribute distributions (pooled) ----
FORCE = ["obligatory", "recommended", "permitted", "discouraged", "prohibited"]
figA, axesA = plt.subplots(2, 2, figsize=(12, 8))
plot_axis(axesA[0, 0], "raz_normative_force", "Normative force", order=FORCE)
plot_axis(axesA[0, 1], "raz_governs_info_flow", "Governs information flow")
plot_axis(axesA[1, 0], "raz_norm_source", "Norm source")
plot_axis(axesA[1, 1], "norm_quality_passed", "Generalizability gate passed")
axesA[0, 0].legend(loc="lower right", fontsize=8)
figA.tight_layout()
figA.savefig(OUT / "top100_norm_attributes.pdf", bbox_inches="tight")
print("\nsaved top100_norm_attributes.pdf")

# ---- Figure B: JSD by axis, pooled vs overlap ----
o = axis_div.sort_values("jsd_pooled")
y = np.arange(len(o))
h = 0.38
figB, axB = plt.subplots(figsize=(8.5, 4.2))
axB.barh(y - h / 2, o["jsd_pooled"], h, label="pooled (model + corpus)", color="#bbbbbb")
axB.barh(y + h / 2, o["jsd_overlap"], h, label="overlap (model only, 7 books)", color="#d62728")
axB.set_yticks(y)
axB.set_yticklabels(o["field"], fontsize=9)
axB.set_xlabel("Jensen-Shannon divergence (base 2)")
axB.set_title("Norm-distribution shift by axis: top100 vs fiction10")
axB.legend(loc="lower right", fontsize=8)
axB.grid(True, axis="x", alpha=0.3)
figB.tight_layout()
figB.savefig(OUT / "top100_divergence_axes.pdf", bbox_inches="tight")
print("saved top100_divergence_axes.pdf")

# overlap info-flow / quality rates (for prose)
ifr = axis_table("raz_governs_info_flow", data=norms_overlap)
qp = axis_table("norm_quality_passed", data=norms_overlap)
print("\n=== same-7-book overlap rates ===")
print(f"info-flow:  fiction10={ifr.loc[True,'fiction10_p']:.3f}  top100={ifr.loc[True,'top100_p']:.3f}")
print(f"quality:    fiction10={qp.loc[True,'fiction10_p']:.3f}  top100={qp.loc[True,'top100_p']:.3f}")
print(f"\nartifacts written to {OUT}")
