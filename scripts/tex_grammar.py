#!/usr/bin/env python
"""Grammar, punctuation, capitalization, and LaTeX typography for the manuscript.

    scripts/tex_grammar.py                  # human-readable report
    scripts/tex_grammar.py --json           # for copyedit.py
    scripts/tex_grammar.py 04_results.tex   # one file
    scripts/tex_grammar.py --no-languagetool   # skip the slow pass

Three passes, each doing what the others cannot
-----------------------------------------------
1. LanguageTool: real parser-based grammar. Subject-verb agreement, article
   choice, tense, comma splices, sentence-initial capitalization. This is the
   only pass that reads a sentence as a sentence, and nothing else in the
   toolchain replaces it. It runs against the de-TeXed text from tex_plain, so
   its offsets land straight on the real source.

2. codespell: a curated typo dictionary with almost no false positives, which
   is exactly the complement to Vale's dictionary spell check. Vale flags any
   word not in its dictionary, so a manuscript full of `PrivacyLens`, `deontic`
   and `Nissenbaum` needs a long allowlist; codespell instead flags only known
   misspellings ("recieve", "seperate"), so it finds real typos in technical
   prose that a dictionary check drowns in noise.

3. Hand-written LaTeX typography rules. These catch defects that are invisible
   to a prose checker because they live in the markup: a missing `~` that lets
   "Figure" and its number split across a line break, a bare `%` that comments
   out the rest of the line, straight quotes that render as ”like this”.

Spelling is deliberately split: LanguageTool's own spell rule is switched off
here so it does not duplicate (and disagree with) Vale's. Run scripts/prose_lint.py
for the dictionary pass.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tex_plain import to_prose, strip_comments, offset_to_linecol, is_standin  # noqa: E402
from tex_integrity import PAPER, resolve_inputs, Finding, SEVERITY_ORDER  # noqa: E402

# --------------------------------------------------------------------------
# LanguageTool configuration
# --------------------------------------------------------------------------

# Rules switched off, each with the reason it misfires on this input.
DISABLED_RULES = {
    # De-TeXing replaces markup with spaces, so runs of whitespace are an
    # artifact of this pipeline, never a defect in the manuscript.
    "WHITESPACE_RULE",
    "COMMA_PARENTHESIS_WHITESPACE",
    "SENTENCE_WHITESPACE",
    "DOUBLE_PUNCTUATION",
    # LaTeX source correctly uses ``quoted'' and -- ; a plain-text checker
    # reads both as typographic errors.
    "EN_QUOTES",
    "DASH_RULE",
    "TYPOGRAPHY",
    # Quotation marks that open in one \textit{} and close in another are
    # normal LaTeX and unpaired to a plain-text reader.
    "EN_UNPAIRED_QUOTES", "EN_UNPAIRED_BRACKETS",
    # Vale owns dictionary spelling (see the module docstring).
    "MORFOLOGIK_RULE_EN_US",
    # Fires on every "Figure 1" stand-in this pipeline inserts.
    "UPPERCASE_SENTENCE_START_TITLE",
    # Style opinions that fight academic register and the house guide.
    "PASSIVE_VOICE", "PASSIVE_SENTENCE", "WORDINESS", "REP_PASSIVE_VOICE",
    "TOO_LONG_SENTENCE", "EN_WORD_COHERENCY", "ARTICLE_MISSING",
}

# Only these LanguageTool categories are reported. The rest are style advice
# that duplicates Vale, or typography that the LaTeX rules below own.
KEEP_CATEGORIES = {
    "GRAMMAR",        # agreement, tense, verb form
    "CASING",         # capitalization
    "PUNCTUATION",    # commas, apostrophes, hyphenation
    "CONFUSED_WORDS", # affect/effect, its/it's, principle/principal
    "MISC",           # repeated words
    "SEMANTICS",      # contradictions, impossible comparisons
    "COLLOCATIONS",
    "COMPOUNDING",
}

# LanguageTool issue types mapped onto this tool's severities.
SEVERITY_BY_ISSUE = {
    "grammar": "error",
    "misspelling": "error",
    "duplication": "error",
    "typographical": "warning",
    "inconsistency": "warning",
    "style": "info",
}

# --------------------------------------------------------------------------
# LaTeX typography rules
# --------------------------------------------------------------------------

# Nouns that must not be separated from their number by a line break. LaTeX
# needs an explicit `~` for that; a plain space is a latent bad break.
TIED_NOUNS = ("Figure", "Figures", "Table", "Tables", "Section", "Sections",
              "Appendix", "Appendices", "Equation", "Equations", "Algorithm",
              "Theorem", "Lemma", "Definition", "Step", "Chapter", "Part")

# --------------------------------------------------------------------------
# Material that is quoted, not authored
# --------------------------------------------------------------------------
# prompts/ and traces/ reproduce verbatim model input and output. Their
# straight quotes, sentence fragments, markdown asterisks and JSON are the
# artifact being documented: "correcting" them would misreport what was run.
# They are excluded from every prose rule and still checked for the markup
# defects (a missing \% breaks the build regardless of who wrote the text).
VERBATIM_DIRS = ("prompts", "traces")
VERBATIM_MARKER = "DO NOT EDIT BY HAND"


def is_verbatim(path: Path, raw: str) -> bool:
    """True when a file reproduces material rather than making an argument."""
    return path.parent.name in VERBATIM_DIRS or VERBATIM_MARKER in raw[:400]


def project_vocabulary() -> set[str]:
    """Words Vale has already been told are correct.

    Reusing Vale's accept list keeps one source of truth for project
    vocabulary, so adding "unparseable" there silences it everywhere instead
    of only in the dictionary pass.
    """
    accept = (Path("/share/pierson/matt/UAIR") / ".vale" / "styles" / "config" /
              "vocabularies" / "NormSim" / "accept.txt")
    if not accept.is_file():
        return set()
    words: set[str] = set()
    for line in accept.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            # Entries may be regexes ("PrivacyLens(es)?"); take the literal stem.
            words.add(re.split(r"[\(\[\\|?*+]", line)[0].lower())
    return words


@dataclass
class GrammarFinding(Finding):
    """A Finding with an optional suggested replacement."""
    suggestion: str = ""


def _tex_typography(path: Path, raw: str, prose: str, verbatim: bool
                    ) -> list[GrammarFinding]:
    """LaTeX-specific defects that a prose checker cannot see.

    Markup rules read `raw`, because the defect is in the markup. Prose rules
    read `prose`, the de-TeXed text, so that a straight quote inside
    \\texttt{{"key": 1}} or a tabular cell is never mistaken for authored prose.
    Both share the same offsets, so either can report against the real line.
    """
    out: list[GrammarFinding] = []
    text = strip_comments(raw)

    def add(sev, code, pos, msg, suggestion=""):
        line, _ = offset_to_linecol(text, pos)
        line_start = text.rfind("\n", 0, pos) + 1
        line_end = text.find("\n", pos)
        line_end = len(text) if line_end == -1 else line_end
        col = pos - line_start
        src = text[line_start:line_end]
        ctx = ("..." if col > 55 else "") + src[max(0, col - 55): col + 55].strip()
        out.append(GrammarFinding(sev, code, path.name, line, msg, ctx, suggestion))

    # --- unescaped percent ------------------------------------------------
    # `50%` silently comments out the rest of the line. Scanned line by line
    # on the raw text: the first unescaped % on a line opens the comment, and
    # if the character before it is a digit that is nearly always a percent
    # sign missing its backslash. [ \t]* rather than \s*, because \s crosses
    # newlines and would pair a number at the end of one line with the comment
    # marker that opens the next.
    for ln_no, ln in enumerate(raw.split("\n"), 1):
        pm = re.search(r"(?<!\\)%", ln)
        if not pm or not ln[: pm.start()].strip():
            continue  # no comment, or a whole-line comment
        if re.search(r"\d[ \t]*$", ln[: pm.start()]):
            out.append(GrammarFinding(
                "error", "UNESCAPED_PERCENT", path.name, ln_no,
                "A `%` right after a number is almost certainly a percent sign that is "
                "missing its backslash. As written it opens a comment and silently "
                "deletes the rest of the line from the PDF.",
                ln.strip()[:110], r"\%"))

    if verbatim:
        return out

    # --- a plain space where a tie is required ----------------------------
    noun_alt = "|".join(TIED_NOUNS)
    for m in re.finditer(r"\b(" + noun_alt + r") +(\\(?:auto)?(?:C|c)?ref\b|\d)", text):
        add("warning", "MISSING_TIE", m.start(),
            f"\"{m.group(1)}\" is separated from its number by a normal space, so a "
            "line break can fall between them. Use a tie: "
            f"`{m.group(1)}~`.",
            f"{m.group(1)}~{m.group(2)}")

    # --- straight quotes ---------------------------------------------------
    # Reported once per line: a footnote listing ten deontic terms produces
    # twenty of these, and twenty identical checkboxes is not a review.
    for ln_no, ln in enumerate(prose.split("\n"), 1):
        n_quotes = ln.count('"')
        if not n_quotes:
            continue
        out.append(GrammarFinding(
            "warning", "STRAIGHT_QUOTE", path.name, ln_no,
            f"{n_quotes} straight `\"` on this line. A straight quote renders as a "
            "right-hand curly quote on both sides; use ``opening'' and closing''.",
            " ".join(ln.split())[:110]))

    # --- space before punctuation -----------------------------------------
    # Read from `text`, not `prose`: "context~\citep{x}." is correct LaTeX, but
    # de-TeXing blanks the citation and leaves "context      .", which would be
    # reported as a space before the full stop on nearly every citing sentence.
    for m in re.finditer(r"[A-Za-z\)\}] +([,;:.!?])(?=\s|$)", text):
        add("warning", "SPACE_BEFORE_PUNCT", m.start(),
            f"Space before `{m.group(1)}`.")

    # --- abbreviation followed by an inter-sentence space ------------------
    # TeX reads the period in "e.g." as a full stop and sets a wider space
    # after it. `\ ` (or `\@`) forces the normal inter-word space. Read from
    # `text`, not `prose`: de-TeXing turns an already-correct `i.e.\ ` into two
    # plain spaces, so checking the prose would flag exactly the sentences that
    # have already been fixed.
    for m in re.finditer(r"\b(e\.g\.|i\.e\.|cf\.|et al\.|vs\.|Fig\.|Eq\.|Sec\.) +[a-z0-9(]",
                         text):
        add("info", "ABBREV_SPACING", m.start(),
            f"`{m.group(1)}` mid-sentence is followed by a plain space, so TeX sets a "
            "wider inter-sentence space. Write "
            f"`{m.group(1)}\\ ` to keep it inter-word.")

    # --- doubled words across a line break --------------------------------
    # LanguageTool sees these too, but only within a sentence it can parse;
    # this catches the "the\nthe" that survives de-TeXing.
    for m in re.finditer(r"\b([A-Za-z]{2,})\s+\1\b", prose):
        if m.group(1).lower() in ("that", "had", "very"):  # legitimate doubling
            continue
        add("error", "DOUBLED_WORD", m.start(),
            f"The word \"{m.group(1)}\" appears twice in a row.")

    return out


def _casing_drift(files: list[Path], texts: dict[Path, str],
                  raws: dict[Path, str]) -> list[GrammarFinding]:
    """Terms written with inconsistent capitalization across the manuscript.

    Camera-ready copy should settle on one spelling of "contextual integrity"
    and one of "Section". Sentence-initial occurrences are excluded, because
    those are capitalized by grammar rather than by choice.
    """
    variants: dict[str, dict[str, list[tuple[Path, int]]]] = {}
    for f in files:
        prose = texts[f]
        # Headings are Title Case by convention, so every content word in them
        # would look like drift against its lowercase use in a sentence.
        # HEADING_CASE owns heading capitalization; blank them here. The spans
        # have to be located in the RAW text, because to_prose has already
        # removed the \section wrapper and left only the bare title. Offsets
        # are identical in both strings, which is what makes this work.
        prose_body = list(prose)
        for hm in re.finditer(r"\\(?:sub)*section\*?\s*\{[^}]*\}|\\paragraph\*?\s*\{[^}]*\}",
                              strip_comments(raws[f])):
            for k in range(hm.start(), min(hm.end(), len(prose_body))):
                if prose_body[k] != "\n":
                    prose_body[k] = " "
        prose_body = "".join(prose_body)
        words = list(re.finditer(r"\b[A-Za-z][A-Za-z\-']*\b", prose_body))
        capped = [bool(w.group(0)[0].isupper()) for w in words]

        for idx, m in enumerate(words):
            word = m.group(0)
            if len(word) < 4 or word.isupper():
                continue  # acronyms are NormSim.Acronyms' business
            # A cross-reference noun is capitalized only when it names a
            # numbered object: "Figure 3" but "that figure places every panel",
            # "Appendix B" but "the appendix gives their derivation". Both are
            # correct, so only compare occurrences that carry a number. Without
            # this the rule reports every generic use as drift.
            if word.lower() in ("figure", "table", "section", "appendix",
                                "equation", "algorithm", "chapter"):
                after = prose_body[m.end():m.end() + 12].lstrip()
                if not re.match(r"[0-9~]|\\(?:auto)?(?:C|c)?ref", after):
                    continue
            # Sentence-initial capitals are grammar, not a naming choice. The
            # look-back has to skip arbitrary whitespace, because de-TeXing
            # leaves a long run of spaces where "\citep{...}." used to be, and
            # a fixed-width window would only ever see those spaces.
            prev = prose_body[:m.start()].rstrip()
            if not prev or prev[-1] in ".!?:;":
                continue
            # A capital inside a run of capitals is part of a proper-noun
            # phrase ("Group Relative Policy Optimization"), where the capital
            # is required and says nothing about how the bare word is styled.
            if capped[idx]:
                prev_adjacent = (idx > 0 and capped[idx - 1]
                                 and prose_body[words[idx - 1].end():m.start()].strip() == "")
                next_adjacent = (idx + 1 < len(words) and capped[idx + 1]
                                 and prose_body[m.end():words[idx + 1].start()].strip() == "")
                if prev_adjacent or next_adjacent:
                    continue
            variants.setdefault(word.lower(), {}).setdefault(word, []).append((f, m.start()))

    out: list[GrammarFinding] = []
    for key, forms in sorted(variants.items()):
        if len(forms) < 2:
            continue
        counts = {k: len(v) for k, v in forms.items()}
        total = sum(counts.values())
        if total < 4:  # too rare to call a convention
            continue
        majority = max(counts, key=counts.get)
        for form, places in forms.items():
            if form == majority or counts[form] > counts[majority]:
                continue
            # Report the minority form once per file, at its first occurrence.
            seen_files: set[Path] = set()
            for f, pos in places:
                if f in seen_files:
                    continue
                seen_files.add(f)
                line, _ = offset_to_linecol(texts[f], pos)
                out.append(GrammarFinding(
                    "warning", "CASING_DRIFT", f.name, line,
                    f"\"{form}\" ({counts[form]}x) disagrees with the dominant "
                    f"\"{majority}\" ({counts[majority]}x). Pick one for the camera-ready.",
                    "", majority))
    return out


# British spellings, paired with their American counterparts. Only stems that
# are unambiguous are listed: "analyse/analyze" is deliberately absent, because
# "analyses" is also the correct American plural of "analysis" and would fire on
# every occurrence of it.
SPELLING_VARIANTS: tuple[tuple[str, str], ...] = (
    ("behaviour", "behavior"), ("colour", "color"), ("centre", "center"),
    ("neighbour", "neighbor"), ("favour", "favor"), ("labour", "labor"),
    ("honour", "honor"), ("rumour", "rumor"), ("odour", "odor"),
    ("normalise", "normalize"), ("recognise", "recognize"),
    ("organise", "organize"), ("summarise", "summarize"),
    ("characterise", "characterize"), ("emphasise", "emphasize"),
    ("generalise", "generalize"), ("minimise", "minimize"),
    ("maximise", "maximize"), ("utilise", "utilize"), ("categorise", "categorize"),
    ("modelling", "modeling"), ("labelled", "labeled"), ("labelling", "labeling"),
    ("travelled", "traveled"), ("cancelled", "canceled"),
    ("defence", "defense"), ("licence", "license"), ("practise", "practice"),
    ("judgement", "judgment"), ("acknowledgement", "acknowledgment"),
    ("grey", "gray"), ("aeroplane", "airplane"), ("programme", "program"),
    ("towards", "toward"), ("whilst", "while"), ("amongst", "among"),
)


def _spelling_variety(files: list[Path], texts: dict[Path, str]
                      ) -> list[GrammarFinding]:
    """Words spelled in the minority English variety for this document.

    A camera-ready should be internally consistent, and a lone "behaviour" in an
    otherwise American paper is the sort of thing a reader notices and a
    dictionary spell checker never reports, because both spellings are correct.
    The house variety is inferred from the document rather than assumed, so this
    works either way round, and only the minority side is reported.
    """
    hits: dict[str, list[tuple[Path, int]]] = {"gb": [], "us": []}
    detail: list[tuple[str, str, Path, int]] = []  # variety, word, file, offset

    for f in files:
        prose = texts[f]
        for gb, us in SPELLING_VARIANTS:
            for variety, stem in (("gb", gb), ("us", us)):
                for m in re.finditer(rf"\b{stem}[a-z]*\b", prose, re.IGNORECASE):
                    hits[variety].append((f, m.start()))
                    detail.append((variety, m.group(0), f, m.start()))

    total = len(hits["gb"]) + len(hits["us"])
    if total < 5:
        return []
    minority = "gb" if len(hits["gb"]) <= len(hits["us"]) else "us"
    majority = "us" if minority == "gb" else "gb"
    label = {"gb": "British", "us": "American"}
    # A near-even split means the document has no house variety to appeal to.
    if len(hits[minority]) > 0.35 * total:
        return []

    out: list[GrammarFinding] = []
    for variety, word, f, pos in detail:
        if variety != minority:
            continue
        counterpart = next(
            (us if variety == "gb" else gb)
            for gb, us in SPELLING_VARIANTS
            if word.lower().startswith(gb if variety == "gb" else us)
        )
        line, _ = offset_to_linecol(texts[f], pos)
        out.append(GrammarFinding(
            "warning", "SPELLING_VARIETY", f.name, line,
            f"\"{word}\" is {label[minority]} spelling; this manuscript is "
            f"{label[majority]} ({len(hits[majority])} of {total} variant words). "
            "Pick one variety for the camera-ready.",
            "", counterpart))
    return out


# Words that stay capitalized in a sentence-case heading: proper nouns,
# acronyms, benchmark and model names, and named statistical tests.
HEADING_PROPER_NOUNS = {
    "CI", "CI-RL", "GRPO", "KTO", "SFT", "DFT", "LLM", "MMLU", "VLM", "PCA", "QR",
    "PrivacyLens", "GoldCoin", "HIPAA", "ConfAIde", "Gemma", "Qwen", "CRediT",
    "Raz", "Project", "Gutenberg", "Group-Relative", "Policy", "Optimization",
    "English", "American", "British", "Nissenbaum", "Wilcoxon", "Cohen",
}


def _heading_case(files: list[Path], raws: dict[Path, str]) -> list[GrammarFinding]:
    """Headings that are not in COLM's mandated sentence case.

    The COLM 2026 style file is explicit, and says the same thing for all three
    heading levels:

        First level headings are in lower case (except for first word and
        proper nouns), bold face, flush left and in point size 12.

    This rule therefore *enforces* sentence case rather than inferring the
    document's dominant style. An earlier version inferred it, and on this
    manuscript that was actively harmful: the appendices were written in Title
    Case, so the majority style was Title Case, and the rule recommended
    "correcting" the body headings *away* from what the venue requires. House
    style is not a vote. The same rule applies to figure captions and table
    titles, which this manuscript already satisfies.
    """
    headings: list[tuple[Path, int, str]] = []  # file, line, title
    for f in files:
        text = strip_comments(raws[f])
        for m in re.finditer(r"\\(?:sub){0,2}section\*?\s*\{([^}]*)\}", text):
            title = m.group(1).strip()
            if not title:
                continue
            line, _ = offset_to_linecol(text, m.start())
            headings.append((f, line, title))

    out: list[GrammarFinding] = []
    for f, line, title in headings:
        # Compare word by word, skipping the first (always capitalized), any
        # word carrying LaTeX markup or math, and the proper-noun allowlist.
        words = title.split(" ")
        offenders = []
        for i, w in enumerate(words):
            if i == 0 or "\\" in w or "$" in w:
                continue
            core = w.strip("()?:,.\u2019'")
            if not core or not core[0].isupper():
                continue
            if core in HEADING_PROPER_NOUNS:
                continue
            # An all-caps token is an acronym, not a capitalization choice.
            if core.isupper():
                continue
            # Hyphenated compounds: judge each part on its own.
            parts = core.split("-")
            if all(p in HEADING_PROPER_NOUNS or p.isupper() for p in parts if p):
                continue
            offenders.append(core)
        if offenders:
            out.append(GrammarFinding(
                "warning", "HEADING_CASE", f.name, line,
                f"Heading \"{title}\" is not sentence case. COLM requires headings in "
                "lower case except for the first word and proper nouns; "
                f"{', '.join(repr(o) for o in offenders)} "
                f"{'is' if len(offenders) == 1 else 'are'} capitalized. "
                "If a word here is a proper noun, add it to HEADING_PROPER_NOUNS.",
                title))
    return out


def _languagetool(files: list[Path], texts: dict[Path, str],
                  raws: dict[Path, str]) -> list[GrammarFinding]:
    try:
        import language_tool_python as lt
    except ImportError:
        return [GrammarFinding(
            "info", "LANGUAGETOOL_MISSING", "-", 1,
            "language-tool-python is not installed, so the grammar pass was skipped. "
            "Install it with: uv pip install --python .venv/bin/python language-tool-python")]

    out: list[GrammarFinding] = []
    tool = lt.LanguageTool("en-US")
    try:
        tool.disabled_rules = DISABLED_RULES
        for f in files:
            prose = texts[f]
            raw = raws[f]
            for m in tool.check(prose):
                if m.rule_id in DISABLED_RULES or m.category not in KEEP_CATEGORIES:
                    continue
                line, col = offset_to_linecol(prose, m.offset)
                matched = prose[m.offset: m.offset + m.error_length]
                # A "defect" made entirely of the spaces this pipeline inserted
                # is an artifact, not a finding.
                if not matched.strip():
                    continue
                # Nor is a complaint about one of this pipeline's own stand-ins.
                # Two ways that happens: the flagged span *is* a stand-in ("x :
                # the leader ties..." reads as a lowercase sentence opening),
                # or the stand-in is the word the rule judged the span against
                # ("a $2.15{:}1$ asymmetry" becomes "a x asymmetry", and LT
                # hears "ex", a vowel, so it asks for "an").
                if is_standin(raw, m.offset):
                    continue
                nxt = re.compile(r"\s*(\S)").match(prose, m.offset + m.error_length)
                if nxt and is_standin(raw, nxt.start(1)):
                    continue
                sev = SEVERITY_BY_ISSUE.get(m.rule_issue_type, "info")
                repl = ", ".join(m.replacements[:3])
                # Quote the real source line, not the de-TeXed one: the reader
                # has to find this spot in the .tex file with CTRL-F, and the
                # prose version has the markup blanked out. Same offsets, so
                # the same line index is correct in both.
                raw_lines = raw.split("\n")
                src_line = raw_lines[line - 1] if line - 1 < len(raw_lines) else ""
                out.append(GrammarFinding(
                    sev, f"LT:{m.rule_id}", f.name, line,
                    m.message.rstrip("."),
                    " ".join(src_line[max(0, col - 56): col + 56].split()),
                    repl))
    finally:
        tool.close()
    return out


def _codespell(files: list[Path], texts: dict[Path, str]) -> list[GrammarFinding]:
    try:
        from codespell_lib._codespell import build_dict, _data_root
    except ImportError:
        return []
    # Load codespell's dictionary directly rather than shelling out, so offsets
    # stay in this process and map back through tex_plain. Only the base
    # `dictionary.txt` is used: it is the unambiguous-errors table. The `rare`
    # and `informal` tables trade precision for recall, which is the wrong
    # trade in a manuscript full of deliberate technical vocabulary.
    misspellings: dict = {}
    base = Path(_data_root) / "dictionary.txt"
    if not base.is_file():
        return []
    build_dict(str(base), misspellings, set())

    allowed = project_vocabulary()
    out: list[GrammarFinding] = []
    for f in files:
        prose = texts[f]
        for m in re.finditer(r"\b[A-Za-z][A-Za-z']*\b", prose):
            word = m.group(0)
            entry = misspellings.get(word.lower())
            if entry is None or word.lower() in allowed:
                continue
            line, col = offset_to_linecol(prose, m.start())
            src_line = prose.split("\n")[line - 1]
            fix = getattr(entry, "data", "") or ""
            out.append(GrammarFinding(
                "error", "TYPO", f.name, line,
                f"\"{word}\" is a known misspelling. {entry.reason or ''}".strip(),
                " ".join(src_line[max(0, col - 56): col + 56].split()),
                fix.strip().rstrip(",")))
    return out


# --------------------------------------------------------------------------

def collect(names: list[str] | None = None, use_languagetool: bool = True
            ) -> tuple[list[GrammarFinding], dict]:
    all_files = resolve_inputs(PAPER)
    if names:
        wanted = {Path(n).name for n in names}
        files = [f for f in all_files if f.name in wanted]
    else:
        files = all_files

    raws = {f: f.read_text(encoding="utf-8", errors="replace") for f in files}
    texts = {f: to_prose(raws[f]) for f in files}

    # Verbatim material is checked for markup defects but never for prose.
    authored = [f for f in files if not is_verbatim(f, raws[f])]
    quoted = [f for f in files if f not in authored]

    findings: list[GrammarFinding] = []
    for f in files:
        findings += _tex_typography(f, raws[f], texts[f], verbatim=f in quoted)
    findings += _casing_drift(authored, texts, raws)
    findings += _spelling_variety(authored, texts)
    findings += _heading_case(authored, raws)
    findings += _codespell(authored, texts)
    if use_languagetool:
        findings += _languagetool(authored, texts, raws)

    findings.sort(key=lambda x: (SEVERITY_ORDER[x.severity], x.code, x.file, x.line))
    return findings, {
        "files": [f.name for f in files],
        "authored": [f.name for f in authored],
        "verbatim_excluded": [f.name for f in quoted],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-languagetool", action="store_true")
    args = ap.parse_args()

    findings, meta = collect(args.files, use_languagetool=not args.no_languagetool)

    if args.json:
        print(json.dumps({"findings": [asdict(f) for f in findings], **meta}, indent=2))
    else:
        by_code: dict[str, list[GrammarFinding]] = {}
        for f in findings:
            by_code.setdefault(f.code, []).append(f)
        for code, items in sorted(by_code.items(),
                                  key=lambda kv: (SEVERITY_ORDER[kv[1][0].severity],
                                                  -len(kv[1]))):
            print(f"\n{items[0].severity.upper()}  {code}  ({len(items)})")
            for x in items[:40]:
                print(f"  {x.file}:{x.line}  {x.message}")
                if x.context:
                    print(f"      {x.context}")
                if x.suggestion:
                    print(f"      -> {x.suggestion}")
            if len(items) > 40:
                print(f"  ... and {len(items) - 40} more")
        print(f"\n{len(findings)} findings across {len(meta['files'])} files.")

    sys.exit(1 if any(f.severity == "error" for f in findings) else 0)


if __name__ == "__main__":
    main()
