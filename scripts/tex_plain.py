#!/usr/bin/env python
"""Turn LaTeX into plain prose *without moving a single character*.

    from tex_plain import to_prose
    prose = to_prose(source)     # len(prose) == len(source), always

Why offsets have to be preserved
--------------------------------
Every downstream checker (LanguageTool, codespell, the typography rules) reports
a character offset into whatever text it was handed. If de-TeXing shortened the
text, each of those offsets would have to be mapped back through an edit script
to reach the real file, and any bug in that map silently points the author at the
wrong line. Instead, markup is *overwritten in place*: a construct is replaced by
a same-length run of spaces, or by a same-length stand-in word. Line numbers,
column numbers, and byte offsets are then identical in both texts, and mapping a
finding back to the source is a no-op.

Stand-ins, not deletions
------------------------
Blanking a citation turns "shown by \\citet{doe} to hold" into "shown by
            to hold", which reads to a grammar checker as a missing subject and
generates a false positive. So constructs that *render as words in the PDF* are
replaced by words of the same width:

    \\citet{doe23}   ->  "Author       "     (a noun, as it renders)
    \\autoref{fig:x} ->  "Figure 1      "    (as hyperref renders it)
    $\\alpha$        ->  "x       "          (math reads as a noun phrase)

Constructs that render as nothing (comments, \\label, \\vspace) are blanked, and
environments with no prose in them (tabular, tikzpicture, equation, verbatim) are
blanked wholesale so their contents never reach a prose checker.

The one thing this cannot do is expand user macros. \\ours or \\normsim is
replaced by a generic stand-in noun, which is right for grammar and wrong for
spelling; spelling is Vale's job, and Vale has its own LaTeX handling.
"""

from __future__ import annotations

import re

# --- Environments whose bodies hold no prose ------------------------------
# Blanked in full, \begin and \end included. `figure`/`table` are deliberately
# absent: their \caption{} text is prose and gets checked.
BLANK_ENVIRONMENTS = (
    "equation", "align", "gather", "multline", "eqnarray", "displaymath", "math",
    "tabular", "tabularx", "tabulary", "array", "matrix", "bmatrix", "pmatrix",
    "tikzpicture", "verbatim", "lstlisting", "minted", "algorithm", "algorithmic",
    "thebibliography", "filecontents", "picture",
)

# --- Commands that render as nothing --------------------------------------
# Command plus all of its brace/bracket arguments is blanked.
DROP_WITH_ARGS = (
    "label", "vspace", "hspace", "setlength", "addtolength", "input", "include",
    "includegraphics", "bibliographystyle", "bibliography", "graphicspath",
    "usepackage", "documentclass", "newcommand", "renewcommand", "providecommand",
    "definecolor", "DeclareMathOperator", "declaretheorem", "newtheorem",
    "hypersetup", "newcolumntype", "captionsetup", "pagestyle", "thispagestyle",
    "index", "nocite", "phantomsection", "addcontentsline", "resizebox",
    "newtcolorbox", "tcbuselibrary", "usetikzlibrary", "setcounter", "refstepcounter",
)

# --- Commands whose brace argument IS prose and should be kept ------------
# The wrapper is blanked, the argument text survives in place.
KEEP_ARGUMENT = (
    "emph", "textit", "textbf", "textsc", "underline", "text", "mbox",
    "caption", "captionof", "title", "author", "thanks",
    "section", "subsection", "subsubsection", "paragraph", "subparagraph",
    "textrm", "textsf", "textnormal", "uline", "so", "highlight",
)

# --- Commands whose argument is prose, but a *separate sentence* ----------
# A footnote is spliced into the middle of its host sentence in the source and
# set apart from it in the PDF. Left inline it reads to a grammar checker as one
# enormous run-on, so the braces become full stops: the host sentence ends, the
# note becomes its own sentence, and both get checked on their own terms.
SENTENCE_BREAK_ARGUMENT = ("footnote", "footnotetext", "epigraph", "marginpar")

# --- Stand-in words, chosen so the sentence still parses -------------------
# Each maps a command to the word it renders as. The word is truncated or
# space-padded to the exact width of the construct it replaces.
CITE_COMMANDS = ("citet", "citeauthor", "citeyearpar", "citealt", "citealp",
                 "citep", "cite", "citenum", "citeyear", "Citet", "Citep")
REF_COMMANDS = ("autoref", "Cref", "cref", "vref", "nameref", "ref",
                "eqref", "pageref", "subref")

# What each cross-reference command renders as, by label prefix. hyperref's
# \autoref picks the word from the counter, so "fig:" becomes "Figure".
REF_WORD_BY_PREFIX = {
    "fig": "Figure", "tab": "Table", "sec": "Section", "eq": "Equation",
    "app": "Appendix", "alg": "Algorithm", "thm": "Theorem", "lem": "Lemma",
    "def": "Definition", "prop": "Proposition", "cor": "Corollary",
    "line": "Line", "item": "Item", "ex": "Example",
}

_COMMENT = re.compile(r"(?<!\\)%")


def _blank(buf: list[str], start: int, end: int) -> None:
    """Overwrite buf[start:end] with spaces, leaving newlines in place.

    Newlines survive so that line numbering, and Vale/LanguageTool's notion of
    where a sentence ends, both stay faithful to the source.
    """
    for i in range(start, end):
        if buf[i] != "\n":
            buf[i] = " "


def _write(buf: list[str], start: int, end: int, word: str) -> None:
    """Blank buf[start:end], then lay `word` down at the start of that span.

    If the span is shorter than the word the word is truncated: correctness of
    offsets outranks readability of the stand-in.
    """
    _blank(buf, start, end)
    for i, ch in enumerate(word[: end - start]):
        if buf[start + i] != "\n":
            buf[start + i] = ch


def _match_group(text: str, i: int, opener: str, closer: str) -> int:
    """Index just past the group starting at text[i], or i if none starts there.

    Counts nesting and honours backslash-escaped braces.
    """
    if i >= len(text) or text[i] != opener:
        return i
    depth = 0
    j = i
    while j < len(text):
        ch = text[j]
        if ch == "\\":
            j += 2
            continue
        if ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return j + 1
        j += 1
    return len(text)  # unbalanced; consume the rest


def _skip_args(text: str, i: int) -> int:
    """Index just past every [optional] and {required} argument starting at i."""
    while i < len(text):
        if text[i] == "{":
            i = _match_group(text, i, "{", "}")
        elif text[i] == "[":
            i = _match_group(text, i, "[", "]")
        elif text[i] == "*":
            i += 1
        else:
            break
    return i


def strip_comments(text: str) -> str:
    """Blank LaTeX comment bodies, preserving line and column positions.

    An unescaped % opens a comment that runs to end of line; `\\%` is a literal
    percent sign. Python's re has lookbehind, which is exactly why this step
    lives here and not in .vale.ini (see scripts/prose_lint.py).
    """
    out = []
    for line in text.split("\n"):
        m = _COMMENT.search(line)
        if m:
            out.append(line[: m.start()] + " " * (len(line) - m.start()))
        else:
            out.append(line)
    return "\n".join(out)


def _ref_word(keys: str) -> str:
    """The word hyperref renders a cross-reference as, from its label prefix."""
    first = keys.split(",")[0].strip()
    prefix = first.split(":")[0].lower()
    return REF_WORD_BY_PREFIX.get(prefix, "Section") + " 1"


def to_prose(text: str) -> str:
    """De-TeX `text` into prose of exactly the same length.

    Comments and no-prose environments are blanked; rendering commands become
    same-width stand-in words; prose-carrying arguments survive in place.
    """
    text = strip_comments(text)
    buf = list(text)

    # 1. Whole environments with no prose inside. Done first so that commands
    #    inside them are never considered.
    for env in BLANK_ENVIRONMENTS:
        pattern = re.compile(
            r"\\begin\{" + env + r"\*?\}.*?\\end\{" + env + r"\*?\}", re.DOTALL
        )
        for m in pattern.finditer(text):
            _blank(buf, m.start(), m.end())

    # 2. Display and inline math. Display first: `$$` must not be read as two
    #    empty inline groups.
    for m in re.finditer(r"\\\[.*?\\\]|\$\$.*?\$\$", text, re.DOTALL):
        _write(buf, m.start(), m.end(), "x")
    for m in re.finditer(r"(?<!\\)\$(?:\\.|[^$\\])*(?<!\\)\$", text):
        # Inline math almost always occupies a noun slot ("$R$ is the reward").
        _write(buf, m.start(), m.end(), "x")

    # 3. Commands, scanned left to right so an outer command is handled before
    #    the commands nested in its arguments.
    i = 0
    n = len(text)
    while i < n:
        if text[i] != "\\":
            i += 1
            continue
        # Accents first: \'{e} is one letter, and blanking it would split
        # "Mis\'{e}rables" into the word "Mis", which a spell checker then
        # reports as a typo. The accented letter is kept, unaccented.
        acc = re.compile(r"\\([`'^\"~=.]|[uvHcktrbd](?![a-zA-Z]))\s*(?:\{([A-Za-z])\}|([A-Za-z]))"
                         ).match(text, i)
        if acc:
            _write(buf, acc.start(), acc.end(), acc.group(2) or acc.group(3))
            i = acc.end()
            continue

        m = re.compile(r"\\([a-zA-Z@]+)\*?").match(text, i)
        if not m:
            # \\, \%, \&, \_ and friends: two characters, no name.
            if i + 1 < n and text[i + 1] == "\\":
                _blank(buf, i, i + 2)  # line break renders as whitespace
            elif i + 1 < n and text[i + 1] in "%&_#${}":
                _write(buf, i, i + 2, text[i + 1])  # escaped literal
            else:
                _blank(buf, i, i + 2)  # any other escape renders as one glyph
            i += 2
            continue

        name = m.group(1)
        after = m.end()

        if name in DROP_WITH_ARGS:
            _blank(buf, i, _skip_args(text, after))
            i = _skip_args(text, after)
        elif name in CITE_COMMANDS:
            end = _skip_args(text, after)
            # \citet renders as a sentence constituent ("Author (2024) shows"),
            # \citep as a parenthetical that can simply go.
            word = "Author" if name.lower().startswith("citet") else ""
            _write(buf, i, end, word)
            i = end
        elif name in REF_COMMANDS:
            end = _skip_args(text, after)
            keys = ""
            g = re.compile(r"\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}").match(text, after)
            if g:
                keys = g.group(1)
            _write(buf, i, end, _ref_word(keys))
            i = end
        elif name in ("texttt", "verb", "lstinline", "url", "path"):
            # Code and URLs: a noun to the grammar checker, invisible to the
            # spell checker. Keeping the literal text would flood both.
            _write(buf, i, _skip_args(text, after), "code")
            i = _skip_args(text, after)
        elif name == "href":
            # \href{url}{text}: drop the URL, keep the visible text.
            url_end = _match_group(text, after, "{", "}")
            _blank(buf, i, url_end)
            i = url_end
        elif name in SENTENCE_BREAK_ARGUMENT:
            arg_start = after
            while arg_start < n and text[arg_start] in " \t":
                arg_start += 1
            if arg_start < n and text[arg_start] == "[":
                arg_start = _match_group(text, arg_start, "[", "]")
            _blank(buf, i, arg_start)
            if arg_start < n and text[arg_start] == "{":
                end = _match_group(text, arg_start, "{", "}")
                _write(buf, arg_start, arg_start + 1, ".")
                _write(buf, end - 1, end, ".")
                i = arg_start + 1
            else:
                i = arg_start
        elif name in KEEP_ARGUMENT:
            # Blank the command name and any optional arg; the brace argument
            # is prose and is left untouched, as are its own delimiters, which
            # become spaces.
            arg_start = after
            while arg_start < n and text[arg_start] in " \t":
                arg_start += 1
            if arg_start < n and text[arg_start] == "[":
                arg_start = _match_group(text, arg_start, "[", "]")
            _blank(buf, i, arg_start)
            if arg_start < n and text[arg_start] == "{":
                end = _match_group(text, arg_start, "{", "}")
                _blank(buf, arg_start, arg_start + 1)
                _blank(buf, end - 1, end)
            i = arg_start
        elif name in ("begin", "end"):
            _blank(buf, i, _skip_args(text, after))
            i = _skip_args(text, after)
        else:
            # Unknown command. Its name is blanked; its arguments are left as
            # prose, because for user macros like \fix{...} the argument
            # usually is the sentence.
            _blank(buf, i, after)
            i = after

    # 4. LaTeX quotes. ``like this'' is correct source but reads to a plain-text
    #    checker as four unpaired marks, so it becomes real curly quotes. Done
    #    on the buffer, one character wide plus a pad, so offsets hold.
    joined = "".join(buf)
    for m in re.finditer(r"``", joined):
        _write(buf, m.start(), m.end(), "“")
    for m in re.finditer(r"''", joined):
        _write(buf, m.start(), m.end(), "”")

    # 5. Residual markup that is punctuation to TeX but noise to a checker.
    for idx, ch in enumerate(buf):
        if ch in "&~":
            buf[idx] = " "  # ~ is a non-breaking space, & a column separator
    return "".join(buf)


def is_standin(raw: str, offset: int) -> bool:
    """True when the prose at `offset` was invented by to_prose, not written.

    Stand-ins are laid down at the first character of the construct they
    replace, so a prose word that begins where the source has a `$` or a `\\`
    is "x" or "Figure 1" or "Author", never something the author typed. Callers
    use this to drop findings about their own placeholders.
    """
    return 0 <= offset < len(raw) and raw[offset] in "$\\"


def offset_to_linecol(text: str, offset: int) -> tuple[int, int]:
    """1-based (line, column) for a character offset into `text`."""
    line = text.count("\n", 0, offset) + 1
    last_nl = text.rfind("\n", 0, offset)
    return line, offset - last_nl


if __name__ == "__main__":  # pragma: no cover - manual inspection aid
    import sys

    src = open(sys.argv[1], encoding="utf-8").read()
    out = to_prose(src)
    assert len(out) == len(src), (len(out), len(src))
    sys.stdout.write(out)
