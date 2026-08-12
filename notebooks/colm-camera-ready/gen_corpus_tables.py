"""Emit the camera-ready corpus tables (LaTeX) for both text corpora.

Writes three tables straight into the paper tree so the numbers can never drift
from the parquets they came from:

    tables/corpus_scaling.tex     — fiction10 vs top100 side by side (tab:corpus-scaling)
    tables/source_texts.tex       — fiction10 per-book extraction stats (tab:source-texts)
    tables/top100_corpus.tex      — the 100-book holdout listing (tab:top100-corpus)

Source of truth:
    notebooks/colm-camera-ready/tables/corpus_descriptives/per_book_yield.csv
        (written by corpus_descriptives_two_corpora.py, read off the Gemma-4
         extraction parquets)
    notebooks/colm-camera-ready/tables/norm_distribution/book_meta.csv
        (Gutenberg catalog metadata, written by
         norm_distribution_top100_vs_fiction10.py)

Run:
    .venv-vllm025cu129/bin/python notebooks/colm-camera-ready/gen_corpus_tables.py
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PAPER_TABLES = HERE.parent.parent / "papers/colm26_normative-simulacra/tables"
YIELD = HERE / "tables/corpus_descriptives/per_book_yield.csv"
META = HERE / "tables/norm_distribution/book_meta.csv"

books = pd.read_csv(YIELD)
meta = pd.read_csv(META)
books["gutenberg_id"] = books["gutenberg_id"].astype(int)
meta["gid"] = meta["gid"].astype(int)
# Merge on (corpus, gid): the seven books present in both corpora have one
# metadata row per corpus, so joining on gid alone would duplicate them.
df = books.merge(
    meta[["gid", "corpus", "title", "author", "birth_year", "death_year"]],
    left_on=["corpus", "gutenberg_id"],
    right_on=["corpus", "gid"],
    how="left",
)
assert len(df) == len(books), f"metadata merge changed row count: {len(books)} -> {len(df)}"


def clean_title(s: str) -> str:
    """Strip MARC subfield markers and bracketed alternates from catalog titles."""
    s = re.sub(r"\s*:\s*\$[a-z]\s*", ": ", str(s))
    s = re.sub(r"\s*\[[^\]]*\]", "", s)
    s = re.sub(r"\s+", " ", s).strip().rstrip(":,;")
    return s


def clean_author(s) -> str:
    """`Surname, Given (Expansion), honorific` -> `Given Surname`.

    Gutenberg catalog names carry parenthetical name expansions and trailing
    titles of nobility ("Tolstoy, Leo, graf"); both are dropped.
    """
    if not isinstance(s, str) or not s.strip():
        return "---"
    s = re.sub(r"\s*\([^)]*\)", "", s).strip()
    parts = [p.strip() for p in s.split(",")]
    if len(parts) == 1:
        return parts[0]
    return f"{parts[1]} {parts[0]}".strip()


# Accented characters are written as LaTeX control sequences rather than raw
# UTF-8: the paper has no inputenc/fontenc line and round-trips through
# Overleaf, so escaping is the safe form.
_ACCENTS = {
    "à": r"\`{a}", "á": r"\'{a}", "â": r"\^{a}", "ä": r'\"{a}', "ã": r"\~{a}", "å": r"\aa{}",
    "è": r"\`{e}", "é": r"\'{e}", "ê": r"\^{e}", "ë": r'\"{e}',
    "ì": r"\`{i}", "í": r"\'{i}", "î": r"\^{i}", "ï": r'\"{i}',
    "ò": r"\`{o}", "ó": r"\'{o}", "ô": r"\^{o}", "ö": r'\"{o}', "õ": r"\~{o}", "ø": r"\o{}",
    "ù": r"\`{u}", "ú": r"\'{u}", "û": r"\^{u}", "ü": r'\"{u}',
    "ñ": r"\~{n}", "ç": r"\c{c}", "ß": r"\ss{}", "æ": r"\ae{}", "œ": r"\oe{}",
    "É": r"\'{E}", "Á": r"\'{A}", "Ö": r'\"{O}', "Ü": r'\"{U}', "Ä": r'\"{A}',
    "–": "--", "—": "---", "’": "'", "‘": "`", "“": "``", "”": "''", "…": r"\ldots{}",
}


def tex(s: str) -> str:
    for a, b in [
        ("\\", r"\textbackslash{}"),
        ("&", r"\&"),
        ("%", r"\%"),
        ("$", r"\$"),
        ("#", r"\#"),
        ("_", r"\_"),
        ("{", r"\{"),
        ("}", r"\}"),
        ("~", r"\textasciitilde{}"),
        ("^", r"\textasciicircum{}"),
    ]:
        s = s.replace(a, b)
    for a, b in _ACCENTS.items():
        s = s.replace(a, b)
    unhandled = {c for c in s if ord(c) > 127}
    assert not unhandled, f"unescaped non-ASCII {unhandled!r} in {s!r}"
    return s


df["disp_title"] = df["title"].map(clean_title)
df["disp_author"] = df["author"].map(clean_author)


def n(x) -> str:
    return f"{int(round(x)):,}".replace(",", "{,}")


def pct(x) -> str:
    return f"{100 * x:.1f}\\%"


# --------------------------------------------------------------------------
# 1. Corpus-scaling comparison
# --------------------------------------------------------------------------
rows = []
for corpus, label in (("fiction10", "fiction10"), ("top100", "top100")):
    d = df[df.corpus == corpus]
    rows.append(
        {
            "label": label,
            "books": len(d),
            "chunks": d.chunks.sum(),
            "norms": d.norms.sum(),
            "flows": d.flows.sum(),
            "norms_per_book": d.norms.sum() / len(d),
            "norms_per_chunk": d.norms.sum() / d.chunks.sum(),
            "flows_per_chunk": d.flows.sum() / d.chunks.sum(),
            "info_flow": d.ci_norms.sum() / d.norms.sum(),
        }
    )
sc = pd.DataFrame(rows)

# Quality-gate rate is pooled over norms, read back from the axis table.
gate = pd.read_csv(HERE / "tables/norm_distribution/axis_norm_quality_passed.csv")
gate_rate = {
    "fiction10": float(gate.loc[gate.norm_quality_passed == True, "fiction10_p"].iloc[0]),  # noqa: E712
    "top100": float(gate.loc[gate.norm_quality_passed == True, "top100_p"].iloc[0]),  # noqa: E712
}

# Transposed layout (metrics as rows): two corpora against nine measures is far
# narrower than nine columns, and puts the two numbers being compared adjacent.
sc = sc.set_index("label")
METRIC_ROWS = [
    ("Books", lambda c: f"{sc.loc[c, 'books']:d}", None),
    ("Chunks", lambda c: n(sc.loc[c, "chunks"]), None),
    ("Norms extracted", lambda c: n(sc.loc[c, "norms"]), None),
    ("CI flow tuples extracted", lambda c: n(sc.loc[c, "flows"]), r"\midrule"),
    ("Norms per book", lambda c: n(sc.loc[c, "norms_per_book"]), None),
    ("Norms per chunk", lambda c: f"{sc.loc[c, 'norms_per_chunk']:.2f}", None),
    ("Flows per chunk", lambda c: f"{sc.loc[c, 'flows_per_chunk']:.2f}", r"\midrule"),
    ("Info-flow rate", lambda c: pct(sc.loc[c, "info_flow"]), None),
    ("Generalizability rate", lambda c: pct(gate_rate[c]), None),
]
body_lines = []
for name, fn, after in METRIC_ROWS:
    body_lines.append(f"{name} & {fn('fiction10')} & {fn('top100')} \\\\")
    if after:
        body_lines.append(after)
body = "\n".join(body_lines)

(PAPER_TABLES / "corpus_scaling.tex").write_text(
    r"""\begin{table}[ht]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\caption{The two normative corpora, extracted by the \textbf{same teacher under the same prompts}
(Gemma-4-31B-it, fiction reasoning + extraction prompts, both runs post-dating the 2026-07-12
prompt-wiring fix). \textbf{fiction10} is the training corpus of \autoref{tab:source-texts};
\textbf{top100} is the 100 most-downloaded English-language fiction works on Project Gutenberg, used
as a book-level holdout and as the scaling demonstration for the claim that the pipeline extends to
the full Gutenberg catalogue. Chunks are 6{,}000-character segments with 1{,}000-character overlap.
Info-flow rate = share of norms governing information transmission (the CI-relevant slice);
generalizability rate = share passing the no-character/plot-leakage quality gate. Every per-chunk
yield and distributional rate is preserved at $10\times$ the reading list.}
\label{tab:corpus-scaling}
\begin{tabular}{lrr}
\toprule
& \textbf{fiction10} & \textbf{top100} \\
& \textit{(training corpus)} & \textit{(book-level holdout)} \\
\midrule
"""
    + body
    + r"""
\bottomrule
\end{tabular}
\end{table}
"""
)
print(f"[tex] {PAPER_TABLES / 'corpus_scaling.tex'}")

# --------------------------------------------------------------------------
# 2. fiction10 source texts
# --------------------------------------------------------------------------
YEAR = {
    "1984": 1949,
    "541": 1920,
    "11": 1865,
    "1399": 1878,
    "1023": 1852,
    "1184": 1844,
    "135": 1862,
    "145": 1871,
    "4078": 1890,
    "1342": 1813,
}
SHORT_TITLE = {
    "1984": "1984",
    "541": "The Age of Innocence",
    "11": "Alice's Adventures in Wonderland",
    "1399": "Anna Karenina",
    "1023": "Bleak House",
    "1184": "The Count of Monte Cristo",
    "135": "Les Mis\\'{e}rables",
    "145": "Middlemarch",
    "4078": "Dorian Gray",
    "1342": "Pride and Prejudice",
}

f10 = df[df.corpus == "fiction10"].copy()
f10["year"] = f10.gutenberg_id.astype(str).map(YEAR)
f10 = f10.sort_values("year")
lines = [
    f"{SHORT_TITLE[str(r.gutenberg_id)]} & {tex(r.disp_author)} & {r.year} & {r.gutenberg_id} & "
    f"{n(r.chunks)} & {n(r.norms)} & {n(r.ci_norms)} & {n(r.flows)} \\\\"
    for r in f10.itertuples()
]
total = (
    r"\textbf{Total} & & & & \textbf{"
    + n(f10.chunks.sum())
    + r"} & \textbf{"
    + n(f10.norms.sum())
    + r"} & \textbf{"
    + n(f10.ci_norms.sum())
    + r"} & \textbf{"
    + n(f10.flows.sum())
    + r"} \\"
)

(PAPER_TABLES / "source_texts.tex").write_text(
    r"""\begin{table}[ht]
\centering
\footnotesize
\setlength{\tabcolsep}{4pt}
\begin{tabular}{llrlrrrr}
\toprule
Title & Author & Year & ID & Chunks & Norms & CI norms & Flows \\
\midrule
"""
    + "\n".join(lines)
    + "\n\\midrule\n"
    + total
    + r"""
\bottomrule
\end{tabular}
\caption{The \textit{fiction10} training corpus and its extraction statistics (Gemma-4-31B-it
teacher). ID: Project Gutenberg identifier. Chunks: 6{,}000-character segments with
1{,}000-character overlap; the extraction LLM processes each chunk independently with a fresh
context window. Norms: Raz-anatomy norms with an assigned deontic force. CI norms: the subset that
governs information flow --- the slice the reward grounds against. Flows: CI information-flow
tuples, extracted in a separate pass over the same chunks.}
\label{tab:source-texts}\label{tab:extraction-stats}
\end{table}
"""
)
print(f"[tex] {PAPER_TABLES / 'source_texts.tex'}")

# --------------------------------------------------------------------------
# 3. top100 corpus listing (two-column landscape table)
# --------------------------------------------------------------------------
t100 = df[df.corpus == "top100"].sort_values("disp_title").reset_index(drop=True)
half = (len(t100) + 1) // 2
left, right = t100.iloc[:half], t100.iloc[half:].reset_index(drop=True)


def cell(r) -> str:
    if r is None:
        return " & & & & "
    return (
        f"{tex(r.disp_title)} & {tex(r.disp_author)} & {r.gutenberg_id} & "
        f"{n(r.chunks)} & {n(r.norms)}"
    )


pair_lines = []
for i in range(half):
    a = cell(left.iloc[i])
    b = cell(right.iloc[i]) if i < len(right) else " & & & & "
    pair_lines.append(f"{a} & {b} \\\\")

(PAPER_TABLES / "top100_corpus.tex").write_text(
    r"""\begin{table}[p]
\centering
\tiny
\setlength{\tabcolsep}{3pt}
\caption{The \textit{top100} corpus: the 100 most-downloaded English-language fiction works on
Project Gutenberg, extracted with the same Gemma-4-31B-it teacher and the same prompts as
\textit{fiction10}. ID: Project Gutenberg identifier; Chunks: 6{,}000-character segments with
1{,}000-character overlap; Norms: extracted Raz-anatomy norms. Seven titles (Alice's Adventures in
Wonderland, Anna Karenina, Bleak House, The Count of Monte Cristo, Les Mis\'{e}rables, Middlemarch,
Pride and Prejudice) also appear in \textit{fiction10}; their two independent extractions give the
run-to-run noise floor used in \autoref{fig:corpus-divergence}b.}
\label{tab:top100-corpus}
\begin{tabular}{p{5.2cm}p{3.1cm}rrr@{\hskip 18pt}p{5.2cm}p{3.1cm}rrr}
\toprule
Title & Author & ID & Chunks & Norms & Title & Author & ID & Chunks & Norms \\
\midrule
"""
    + "\n".join(pair_lines)
    + r"""
\bottomrule
\end{tabular}
\end{table}
"""
)
print(f"[tex] {PAPER_TABLES / 'top100_corpus.tex'}")

print("\nfiction10 totals:", f10[["chunks", "norms", "ci_norms", "flows"]].sum().to_dict())
print("top100 totals:", t100[["chunks", "norms", "ci_norms", "flows"]].sum().to_dict())
print("gate rates:", gate_rate)
