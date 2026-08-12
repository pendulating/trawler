#!/usr/bin/env python
"""Find every cross-reference that will render as `??` (or `[?]`) in the PDF.

    scripts/tex_integrity.py                # human-readable report
    scripts/tex_integrity.py --json         # machine-readable, for copyedit.py
    scripts/tex_integrity.py --build        # also compile and read LaTeX's log

Why both a static pass and a build pass
---------------------------------------
LaTeX itself is the ground truth: it prints "Reference `x' undefined" and sets
`??` in the output. But its log is only as fresh as the last compile, and this
manuscript is edited in Overleaf and locally at the same time, so the .aux on
disk is routinely older than the .tex files. Trusting a stale log is how a `??`
reaches a camera-ready deadline.

So the static pass is the default: it parses the sources directly and needs no
toolchain, no compile, and no warm .aux. `--build` then runs latexmk and folds
LaTeX's own verdict in, which catches the residue a regex cannot see (labels
produced by macro expansion, `\\cref` ranges, hyperref anchor collisions).

Only files the document actually builds are checked. `\\input` is followed from
00_main.tex, so scraps.tex and TADA_1pager.tex, which are in the directory but
not in the paper, never produce findings.

Checks, by severity
-------------------
error    UNDEFINED_REF     -> renders as `??`
         UNDEFINED_CITE    -> renders as `[?]`
         DUPLICATE_LABEL   -> silently resolves to the wrong number
         LITERAL_QQ        -> `??` typed straight into the source
         MISSING_GRAPHIC   -> \\includegraphics of a file that is not there
warning  LABEL_BEFORE_CAPTION -> float numbered as the preceding float
         FLOAT_NO_LABEL       -> a float that cannot be referenced
         UNREFERENCED_FLOAT   -> a float the prose never points at
         BARE_REF             -> "see \\ref{fig:x}" renders "see 3", no "Figure"
info     UNUSED_LABEL         -> label defined, never referenced
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tex_plain import strip_comments  # noqa: E402

ROOT = Path("/share/pierson/matt/UAIR")
PAPER = ROOT / "papers" / "colm26_normative-simulacra"
MAIN = "00_main.tex"

REF_CMDS = r"autoref|Cref|cref|vref|nameref|eqref|pageref|subref|ref"
CITE_CMDS = r"cite[a-zA-Z]*"
FLOAT_ENVS = ("figure", "table", "wrapfigure", "sidewaystable", "sidewaysfigure",
              "algorithm", "listing")
GRAPHIC_EXTS = ("", ".pdf", ".png", ".jpg", ".jpeg", ".eps", ".PDF", ".PNG", ".JPG")

# Words that make a bare \ref read correctly ("Table~\ref{...}"). Checked case
# insensitively against the text immediately preceding the command.
REF_LEAD_WORDS = ("figure", "table", "section", "appendix", "equation", "eq",
                  "algorithm", "theorem", "lemma", "definition", "proposition",
                  "corollary", "step", "line", "item", "part", "panel", "row",
                  "and", "or", "to", "through", "-", "--", ",")

SEVERITY_ORDER = {"error": 0, "warning": 1, "info": 2}


@dataclass
class Finding:
    severity: str
    code: str
    file: str
    line: int
    message: str
    context: str = ""

    def sort_key(self):
        return (SEVERITY_ORDER[self.severity], self.code, self.file, self.line)


# --------------------------------------------------------------------------
# Document assembly
# --------------------------------------------------------------------------

def resolve_inputs(paper: Path, main: str = MAIN) -> list[Path]:
    """Every .tex file the build actually reads, in \\input order, main first.

    Recursive, so a table \\input from inside a figure environment is included.
    A file is visited once even if \\input twice.
    """
    ordered: list[Path] = []
    seen: set[Path] = set()

    def visit(path: Path) -> None:
        if path in seen or not path.is_file():
            return
        seen.add(path)
        ordered.append(path)
        text = strip_comments(path.read_text(encoding="utf-8", errors="replace"))
        for m in re.finditer(r"\\(?:input|include)\s*\{([^}]+)\}", text):
            name = m.group(1).strip()
            child = paper / (name if name.endswith(".tex") else name + ".tex")
            visit(child)

    visit(paper / main)
    return ordered


def bib_keys(paper: Path, files: list[Path]) -> set[str]:
    """Keys defined in every .bib the document loads."""
    names: list[str] = []
    for f in files:
        text = strip_comments(f.read_text(encoding="utf-8", errors="replace"))
        for m in re.finditer(r"\\(?:bibliography|addbibresource)\s*\{([^}]+)\}", text):
            names += [n.strip() for n in m.group(1).split(",")]
    keys: set[str] = set()
    for n in names:
        bib = paper / (n if n.endswith(".bib") else n + ".bib")
        if not bib.is_file():
            continue
        text = bib.read_text(encoding="utf-8", errors="replace")
        keys |= {m.group(1).strip() for m in re.finditer(r"@\w+\s*\{\s*([^,\s]+)\s*,", text)}
    return keys


# --------------------------------------------------------------------------
# Static analysis
# --------------------------------------------------------------------------

def _line_of(text: str, pos: int) -> int:
    return text.count("\n", 0, pos) + 1


def _context(text: str, pos: int, width: int = 55) -> str:
    """A one-line slice of source around `pos`, for CTRL-F."""
    line_start = text.rfind("\n", 0, pos) + 1
    line_end = text.find("\n", pos)
    line_end = len(text) if line_end == -1 else line_end
    line = text[line_start:line_end]
    col = pos - line_start
    start, end = max(0, col - width), min(len(line), col + width)
    return ("..." if start else "") + line[start:end].strip() + ("..." if end < len(line) else "")


def analyze(paper: Path, files: list[Path]) -> list[Finding]:
    out: list[Finding] = []
    texts = {f: strip_comments(f.read_text(encoding="utf-8", errors="replace")) for f in files}

    # --- pass 1: inventory ------------------------------------------------
    labels: dict[str, list[tuple[Path, int]]] = {}
    refs: list[tuple[str, Path, int, int, str]] = []  # key, file, line, pos, cmd
    cites: list[tuple[str, Path, int, int]] = []

    for f, text in texts.items():
        for m in re.finditer(r"\\label\s*\{([^}]*)\}", text):
            labels.setdefault(m.group(1).strip(), []).append((f, _line_of(text, m.start())))
        for m in re.finditer(r"\\(" + REF_CMDS + r")\*?\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}", text):
            # \cref takes a comma-separated list; every key must resolve.
            for key in m.group(2).split(","):
                key = key.strip()
                if key:
                    refs.append((key, f, _line_of(text, m.start()), m.start(), m.group(1)))
        for m in re.finditer(r"\\(" + CITE_CMDS + r")\*?\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}", text):
            if m.group(1) in ("citation",):
                continue
            for key in m.group(2).split(","):
                key = key.strip()
                if key:
                    cites.append((key, f, _line_of(text, m.start()), m.start()))

    known_cites = bib_keys(paper, files)

    # --- UNDEFINED_REF: the `??` maker ------------------------------------
    for key, f, line, pos, cmd in refs:
        if key not in labels:
            out.append(Finding(
                "error", "UNDEFINED_REF", f.name, line,
                f"\\{cmd}{{{key}}} has no matching \\label; this renders as `??`.",
                _context(texts[f], pos),
            ))

    # --- DUPLICATE_LABEL ---------------------------------------------------
    for key, places in sorted(labels.items()):
        if len(places) > 1:
            where = ", ".join(f"{p.name}:{ln}" for p, ln in places)
            out.append(Finding(
                "error", "DUPLICATE_LABEL", places[0][0].name, places[0][1],
                f"\\label{{{key}}} is defined {len(places)} times ({where}); "
                "references to it silently resolve to the last one.",
            ))

    # --- UNDEFINED_CITE ----------------------------------------------------
    if known_cites:
        for key, f, line, pos in cites:
            if key not in known_cites:
                out.append(Finding(
                    "error", "UNDEFINED_CITE", f.name, line,
                    f"\\cite key `{key}` is in no loaded .bib; this renders as `[?]`.",
                    _context(texts[f], pos),
                ))

    # --- LITERAL_QQ --------------------------------------------------------
    # A `??` already sitting in the source, usually pasted back in from a PDF.
    # Skip anything inside a verbatim-ish command, and skip `???` in prose.
    for f, text in texts.items():
        for m in re.finditer(r"(?<![?\\])\?\?(?!\?)", text):
            out.append(Finding(
                "error", "LITERAL_QQ", f.name, _line_of(text, m.start()),
                "A literal `??` in the source. If this stands for a reference, "
                "it will print as `??` whether or not the build succeeds.",
                _context(text, m.start()),
            ))

    # --- MISSING_GRAPHIC ---------------------------------------------------
    graphics_dirs = [paper]
    for f, text in texts.items():
        for m in re.finditer(r"\\graphicspath\s*\{((?:\{[^}]*\}\s*)+)\}", text):
            for d in re.findall(r"\{([^}]*)\}", m.group(1)):
                graphics_dirs.append(paper / d)
    for f, text in texts.items():
        for m in re.finditer(r"\\includegraphics\s*(?:\[[^\]]*\])*\s*\{([^}]*)\}", text):
            name = m.group(1).strip()
            if "#" in name:  # inside a macro definition, not a real path
                continue
            found = any((d / (name + ext)).is_file() for d in graphics_dirs for ext in GRAPHIC_EXTS)
            if not found:
                out.append(Finding(
                    "error", "MISSING_GRAPHIC", f.name, _line_of(text, m.start()),
                    f"\\includegraphics{{{name}}} resolves to no file under "
                    + ", ".join(str(d.relative_to(paper)) or "." for d in graphics_dirs) + ".",
                    _context(text, m.start()),
                ))

    # --- float-level checks ------------------------------------------------
    referenced = {key for key, *_ in refs}
    for f, text in texts.items():
        for env in FLOAT_ENVS:
            pattern = re.compile(
                r"\\begin\{" + env + r"\*?\}(.*?)\\end\{" + env + r"\*?\}", re.DOTALL)
            for m in pattern.finditer(text):
                body = m.group(1)
                line = _line_of(text, m.start())
                lab = re.search(r"\\label\s*\{([^}]*)\}", body)
                cap = re.search(r"\\caption(?:of\s*\{[^}]*\})?\s*(?:\[[^\]]*\])?\s*\{", body)

                if not lab:
                    # A float with neither caption nor label is being used for
                    # layout (the CRediT contributions table, say), not as a
                    # numbered object, so there is nothing to reference.
                    if cap:
                        out.append(Finding(
                            "warning", "FLOAT_NO_LABEL", f.name, line,
                            f"This {env} is captioned but has no \\label, so nothing "
                            "can reference it.",
                        ))
                    continue
                key = lab.group(1).strip()

                # \label must follow \caption: the label captures whatever the
                # counter held when it was set, so a label above the caption
                # numbers the float as the *previous* one.
                if cap and lab.start() < cap.start():
                    out.append(Finding(
                        "warning", "LABEL_BEFORE_CAPTION", f.name,
                        line + body[: lab.start()].count("\n"),
                        f"\\label{{{key}}} appears before \\caption in this {env}; "
                        "it will take the preceding float's number. Move it after the caption.",
                    ))
                if key not in referenced:
                    out.append(Finding(
                        "warning", "UNREFERENCED_FLOAT", f.name, line,
                        f"This {env} (\\label{{{key}}}) is never referenced in the text. "
                        "Floats are placed relative to their first mention.",
                    ))

    # --- BARE_REF ----------------------------------------------------------
    # `\ref` prints a bare number. Unless the preceding word supplies the noun,
    # the sentence reads "see 3". `\autoref` supplies its own noun and is exempt.
    for key, f, line, pos, cmd in refs:
        if cmd != "ref":
            continue
        before = texts[f][max(0, pos - 24):pos]
        tail = re.sub(r"[~\s\(\[]+$", "", before)
        # A section/paragraph sign is a noun too: "\S\ref{sec:x}" sets "§4".
        if re.search(r"(\\S|\\P|§|¶)$", tail):
            continue
        word = re.search(r"([A-Za-z\-,]+)$", tail.lower())
        if not word or word.group(1) not in REF_LEAD_WORDS:
            out.append(Finding(
                "warning", "BARE_REF", f.name, line,
                f"\\ref{{{key}}} prints only a number and is not preceded by a noun "
                "(\"Figure\", \"Table\", ...). Use \\autoref, or name the object.",
                _context(texts[f], pos),
            ))

    # --- UNUSED_LABEL ------------------------------------------------------
    for key, places in sorted(labels.items()):
        if key not in referenced:
            f, line = places[0]
            out.append(Finding(
                "info", "UNUSED_LABEL", f.name, line,
                f"\\label{{{key}}} is never referenced.",
            ))

    return out


# --------------------------------------------------------------------------
# Build cross-check
# --------------------------------------------------------------------------

BUILD_DIR = Path("/tmp") / f"copyedit-build-{Path.home().name}"


def build_findings(paper: Path, do_build: bool) -> tuple[list[Finding], str]:
    """LaTeX's own verdict, plus a note on how much it can be trusted."""
    log = paper / (Path(MAIN).stem + ".log")
    out: list[Finding] = []

    if do_build:
        # -outdir keeps every artifact out of the source tree. Building in
        # place would overwrite the author's .pdf and .aux, and a copyediting
        # tool has no business doing that to a live manuscript.
        BUILD_DIR.mkdir(parents=True, exist_ok=True)
        # -f keeps going past errors so the log records every undefined
        # reference, not just the ones before the first failure.
        proc = subprocess.run(
            ["latexmk", "-pdf", "-f", "-interaction=nonstopmode",
             f"-outdir={BUILD_DIR}", MAIN],
            cwd=paper, capture_output=True, text=True,
        )
        note = f"compiled fresh with latexmk (exit {proc.returncode}), artifacts in {BUILD_DIR}"
        log = BUILD_DIR / (Path(MAIN).stem + ".log")
        if not log.is_file():
            return [Finding("error", "BUILD_FAILED", MAIN, 1,
                            f"latexmk produced no log.\n{proc.stdout[-2000:]}")], note
    elif not log.is_file():
        return [], "no build log on disk; static checks only"
    else:
        # A log older than the sources describes a document that no longer
        # exists, so say so rather than reporting its warnings as current.
        newest = max((f.stat().st_mtime for f in paper.glob("*.tex")), default=0)
        if log.stat().st_mtime < newest:
            return [], ("build log is OLDER than the .tex sources: its warnings are "
                        "stale and were skipped. Re-run with --build.")
        note = "build log is newer than all sources"

    text = log.read_text(encoding="utf-8", errors="replace")
    for m in re.finditer(r"LaTeX Warning: (Reference|Citation) [`']([^']*)' on page (\S+) undefined",
                         text):
        kind, key, page = m.groups()
        out.append(Finding(
            "error", f"BUILD_UNDEFINED_{kind.upper()}", MAIN, 1,
            f"LaTeX reports {kind.lower()} `{key}` undefined on page {page}.",
        ))
    for m in re.finditer(r"LaTeX Warning: Label [`']([^']*)' multiply defined", text):
        out.append(Finding("error", "BUILD_DUPLICATE_LABEL", MAIN, 1,
                           f"LaTeX reports label `{m.group(1)}` multiply defined."))
    for m in re.finditer(r"^! (.+)$", text, re.M):
        out.append(Finding("error", "BUILD_ERROR", MAIN, 1, m.group(1).strip()))

    out += pdf_findings(log.with_suffix(".pdf"))
    return out, note


def pdf_findings(pdf: Path) -> list[Finding]:
    """Scan the rendered pages for `??` and `[?]`, the last word on the matter.

    Everything upstream reasons about what LaTeX *should* produce. This reads
    what it did produce, so it also catches markers that arrive by routes the
    source scan cannot model: a `??` inside a generated table, a bad reference
    in an included PDF figure, or a citation the .bst dropped.
    """
    if not pdf.is_file():
        return []
    try:
        from pypdf import PdfReader
    except ImportError:
        return [Finding("info", "PDF_SCAN_SKIPPED", pdf.name, 1,
                        "pypdf is not installed, so the rendered PDF was not scanned "
                        "for `??`. Install it with: uv pip install --python .venv/bin/python pypdf")]
    out: list[Finding] = []
    reader = PdfReader(str(pdf))
    for page_no, page in enumerate(reader.pages, 1):
        text = page.extract_text() or ""
        for m in re.finditer(r"\?\?|\[\?\]", text):
            snippet = " ".join(text[max(0, m.start() - 70): m.end() + 40].split())
            out.append(Finding(
                "error", "PDF_UNRESOLVED_MARKER", pdf.name, page_no,
                f"Page {page_no} of the rendered PDF contains "
                f"`{m.group(0)}`: an unresolved reference or citation.",
                snippet,
            ))
    return out


def collect(do_build: bool = False) -> tuple[list[Finding], dict]:
    files = resolve_inputs(PAPER)
    findings = analyze(PAPER, files)
    build, note = build_findings(PAPER, do_build)
    findings += build
    findings.sort(key=lambda f: f.sort_key())
    meta = {"files": [f.name for f in files], "build_note": note}
    return findings, meta


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--build", action="store_true", help="compile and read LaTeX's log")
    args = ap.parse_args()

    findings, meta = collect(args.build)

    if args.json:
        print(json.dumps({"findings": [asdict(f) for f in findings], **meta}, indent=2))
    else:
        print(f"{len(meta['files'])} files in the build; {meta['build_note']}.")
        by_code: dict[str, list[Finding]] = {}
        for f in findings:
            by_code.setdefault(f.code, []).append(f)
        for code, items in sorted(by_code.items(),
                                  key=lambda kv: (SEVERITY_ORDER[kv[1][0].severity], kv[0])):
            print(f"\n{items[0].severity.upper()}  {code}  ({len(items)})")
            for f in items:
                print(f"  {f.file}:{f.line}  {f.message}")
                if f.context:
                    print(f"      {f.context}")
        n_err = sum(1 for f in findings if f.severity == "error")
        print(f"\n{len(findings)} findings, {n_err} of them errors.")

    sys.exit(1 if any(f.severity == "error" for f in findings) else 0)


if __name__ == "__main__":
    main()
