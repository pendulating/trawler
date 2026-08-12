#!/usr/bin/env python
"""One copyediting pass over the COLM manuscript, in one report.

    scripts/copyedit.py                 # write COPYEDIT_REVIEW.md
    scripts/copyedit.py --build         # also compile, and read LaTeX's verdict
    scripts/copyedit.py --fast          # skip LanguageTool (seconds, not minutes)
    scripts/copyedit.py --gate          # exit non-zero if anything blocking remains

What this runs, and why each is here
------------------------------------
reference integrity  scripts/tex_integrity.py  every `??` the PDF can print
grammar and spelling scripts/tex_grammar.py    LanguageTool, codespell, typography
house style          scripts/prose_lint.py     Vale: the style guide as rules

The three overlap by design at exactly one point (spelling) and are wired so
they do not duplicate: Vale owns dictionary spelling, codespell owns known
typos, LanguageTool's own speller is off. Everywhere else they are disjoint.

The report is ordered by what it costs to be wrong, not by which tool found it.
A `??` in the camera-ready is unrecoverable; a passive-voice suggestion is a
matter of taste. So findings are grouped into tiers and the tiers are ordered:
fix tier 1 completely, read tier 4 selectively.

Coherency is not in here. Whether an argument holds together is not a thing a
linter can score, so this script emits the *checklist* for that pass (see
COHERENCY_PROMPT) and leaves the judgment to a reader.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import tex_grammar  # noqa: E402
import tex_integrity  # noqa: E402
from tex_integrity import PAPER, ROOT  # noqa: E402

# (tier heading, why it matters, [codes]) in the order they should be worked.
# "Vale:<Rule>" entries match Vale check names; everything else matches a code
# from tex_integrity or tex_grammar. "LT:*" matches any LanguageTool rule.
TIERS: list[tuple[str, str, list[str]]] = [
    (
        "Blocking: this renders wrong in the PDF",
        "Each of these changes what a reader sees. Nothing else in the report "
        "matters until this section is empty.",
        ["UNDEFINED_REF", "UNDEFINED_CITE", "DUPLICATE_LABEL", "LITERAL_QQ",
         "MISSING_GRAPHIC", "UNESCAPED_PERCENT", "BUILD_FAILED", "BUILD_ERROR",
         "BUILD_UNDEFINED_REFERENCE", "BUILD_UNDEFINED_CITATION",
         "BUILD_DUPLICATE_LABEL", "PDF_UNRESOLVED_MARKER",
         "Vale:proselint.Annotations"],
    ),
    (
        "Grammar and spelling",
        "Defects in the English. LanguageTool findings are parser-based and "
        "occasionally wrong about technical phrasing, so read the quoted line "
        "before accepting one.",
        ["TYPO", "DOUBLED_WORD", "LT:*",
         "Vale:Vale.Spelling", "Vale:proselint.Spelling", "Vale:NormSim.Terms",
         "Vale:NormSim.Acronyms", "Vale:Vale.Terms", "Vale:Vale.Avoid"],
    ),
    (
        "Punctuation, typography, and capitalization",
        "Surface presentation. Cheap to fix, and the most visible class of "
        "defect in a camera-ready.",
        ["STRAIGHT_QUOTE", "SPACE_BEFORE_PUNCT", "MISSING_TIE", "ABBREV_SPACING",
         "HEADING_CASE", "CASING_DRIFT", "SPELLING_VARIETY",
         "Vale:NormSim.EmDash", "Vale:NormSim.LatinAbbrev"],
    ),
    (
        "Cross-references and floats",
        "Not wrong, but worth a look: a float nothing points at lands wherever "
        "LaTeX likes, and a bare number reads badly.",
        ["BARE_REF", "UNREFERENCED_FLOAT", "FLOAT_NO_LABEL",
         "LABEL_BEFORE_CAPTION", "UNUSED_LABEL"],
    ),
    (
        "House style and word choice",
        "Rules from `academic-writing-mini.md`. Suggestions, not defects: a "
        "passive construction is correct in plenty of methods sentences.",
        ["Vale:NormSim.Filler", "Vale:write-good.Weasel", "Vale:write-good.ThereIs",
         "Vale:proselint.Uncomparables", "Vale:proselint.Hedging",
         "Vale:write-good.Passive", "Vale:*"],
    ),
]

COHERENCY_PROMPT = """\
Mechanical checks cannot tell whether the argument holds, and they read one
sentence at a time, so they cannot see a term spelled three ways across three
files. That pass is a read-through, and its findings live in
`COHERENCY_REVIEW.md`.

Read the body once end to end, against these questions:

- [ ] **Topic sentences.** Does every paragraph open with the claim it goes on
      to support? Read only the first sentence of each paragraph in order: that
      sequence should be a readable summary of the section.
- [ ] **Promise and payoff.** Does everything the introduction says the paper
      will show actually get shown, in the section it points at? Does anything
      in the results arrive without having been set up?
- [ ] **Terminology.** Is each construct named the same way throughout? A term
      that appears as three near-synonyms reads as three different things.
- [ ] **Claim strength.** Does each claim's hedging match its evidence? Flag
      both directions: an unhedged claim resting on one measurement, and a
      hedged claim that the data actually settles.
- [ ] **Given/new order.** Does each sentence start from what the reader
      already knows and end on what is new? Reversals are the usual cause of a
      paragraph that is individually clear and collectively confusing.
- [ ] **Transitions.** Do the connectives assert real relations? "However" over
      a non-contrast and "Therefore" over a non-consequence both mislead.
- [ ] **Dead cross-references.** Does text that points at a section still
      describe what that section now says?
"""


def _vale_alerts(names: list[str]) -> tuple[list[dict], str]:
    """Vale findings, or an explanation of why there are none.

    `names` is the same authored-file set the other two passes use, so all
    three report on the same document. Vale's own default glob only reaches
    the top-level chapter files, which would leave table captions unlinted.
    """
    try:
        from prose_lint import collect_alerts
    except ImportError as exc:
        return [], f"Vale pass unavailable: {exc}"
    try:
        alerts, files = collect_alerts(names)
    except SystemExit as exc:
        return [], f"Vale did not run: {exc}"
    return alerts, f"Vale linted {len(files)} files"


def _norm_vale(alert: dict) -> dict:
    return {
        "severity": {"error": "error", "warning": "warning"}.get(
            alert.get("Severity", "suggestion"), "info"),
        "code": f"Vale:{alert['Check']}",
        "file": Path(alert["File"]).name,
        "line": alert["Line"],
        "message": alert["Message"],
        "context": (alert.get("Match") or "").replace("\n", " ").strip(),
        "suggestion": "",
    }


def _matches(code: str, pattern: str) -> bool:
    if pattern.endswith("*"):
        return code.startswith(pattern[:-1])
    return code == pattern


def _fence(text: str) -> str:
    """Inline code span that survives the backticks LaTeX prose is full of."""
    text = " ".join(text.split())
    longest, run = 0, 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    f = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{f}{pad}{text}{pad}{f}"


def build_report(rows: list[dict], meta: dict) -> str:
    by_code: dict[str, list[dict]] = {}
    for r in rows:
        by_code.setdefault(r["code"], []).append(r)

    out: list[str] = []
    w = out.append
    w("# Copyedit review: `colm26_normative-simulacra`")
    w("")
    w(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"- Paper HEAD: `{meta['head']}` (plus uncommitted working-tree edits)")
    w(f"- Files in the build: {meta['n_files']}"
      f" ({meta['n_authored']} authored, {len(meta['verbatim'])} verbatim)")
    w(f"- Reference integrity: {meta['build_note']}")
    w(f"- {meta['vale_note']}")
    if meta["verbatim"]:
        w(f"- Excluded from prose rules (reproduced material, not authored): "
          f"{', '.join('`' + v + '`' for v in meta['verbatim'])}")
    w("")
    w("> **A snapshot.** Line numbers drift as the manuscript is edited. "
      "Regenerate with `scripts/copyedit.py` before working through it.")
    w("")

    n_block = sum(1 for r in rows if r["_tier"] == 0)
    w("## Status")
    w("")
    w("| Tier | Findings |")
    w("| --- | ---: |")
    for i, (heading, _, _) in enumerate(TIERS):
        w(f"| {heading} | {sum(1 for r in rows if r['_tier'] == i)} |")
    w(f"| **Total** | **{len(rows)}** |")
    w("")
    if n_block == 0:
        w("**No blocking defects.** Every cross-reference, citation and float "
          "resolves; nothing will print as `??`.")
    else:
        w(f"**{n_block} blocking defects.** These render incorrectly in the PDF.")
    w("")

    for i, (heading, blurb, patterns) in enumerate(TIERS):
        tier_rows = [r for r in rows if r["_tier"] == i]
        if not tier_rows:
            continue
        w("---")
        w("")
        w(f"## {heading} ({len(tier_rows)})")
        w("")
        w(blurb)
        w("")
        codes = sorted({r["code"] for r in tier_rows},
                       key=lambda c: (-len(by_code[c]), c))
        for code in codes:
            items = [r for r in by_code[code] if r["_tier"] == i]
            if not items:
                continue
            w(f"### `{code}` ({len(items)})")
            w("")
            per_file: dict[str, list[dict]] = {}
            for r in items:
                per_file.setdefault(r["file"], []).append(r)
            for fname in sorted(per_file):
                w(f"**`{fname}`**")
                w("")
                for r in sorted(per_file[fname], key=lambda x: x["line"]):
                    w(f"- [ ] **L{r['line']}** — {r['message']}")
                    if r.get("context"):
                        w(f"  - {_fence(r['context'])}")
                    if r.get("suggestion"):
                        w(f"  - suggested: {_fence(r['suggestion'])}")
                w("")

    w("---")
    w("")
    w("## Coherency (manual)")
    w("")
    w(COHERENCY_PROMPT)
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(ROOT / "COPYEDIT_REVIEW.md"))
    ap.add_argument("--build", action="store_true",
                    help="compile the document and fold in LaTeX's own warnings")
    ap.add_argument("--fast", action="store_true", help="skip the LanguageTool pass")
    ap.add_argument("--gate", action="store_true",
                    help="exit non-zero when any blocking finding remains")
    args = ap.parse_args()

    print("reference integrity ...", flush=True)
    integrity, imeta = tex_integrity.collect(do_build=args.build)
    print(f"  {len(integrity)} findings", flush=True)

    print("grammar, spelling, typography ...", flush=True)
    grammar, gmeta = tex_grammar.collect(use_languagetool=not args.fast)
    print(f"  {len(grammar)} findings", flush=True)

    print("house style (Vale) ...", flush=True)
    authored_rel = [
        str(f.relative_to(PAPER))
        for f in tex_integrity.resolve_inputs(PAPER)
        if not tex_grammar.is_verbatim(f, f.read_text(encoding="utf-8", errors="replace"))
    ]
    alerts, vale_note = _vale_alerts(authored_rel)
    print(f"  {len(alerts)} findings", flush=True)

    rows: list[dict] = []
    for f in integrity + grammar:
        d = {"severity": f.severity, "code": f.code, "file": f.file,
             "line": f.line, "message": f.message, "context": f.context,
             "suggestion": getattr(f, "suggestion", "")}
        rows.append(d)
    rows += [_norm_vale(a) for a in alerts]

    # Assign each row to the first tier that claims its code; anything
    # unclaimed lands in the last tier rather than vanishing.
    for r in rows:
        r["_tier"] = len(TIERS) - 1
        for i, (_, _, patterns) in enumerate(TIERS):
            if any(_matches(r["code"], p) for p in patterns):
                r["_tier"] = i
                break

    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"],
                          capture_output=True, text=True, cwd=PAPER).stdout.strip()
    meta = {
        "head": head or "unknown",
        "n_files": len(imeta["files"]),
        "n_authored": len(gmeta["authored"]),
        "verbatim": gmeta["verbatim_excluded"],
        "build_note": imeta["build_note"],
        "vale_note": vale_note,
    }

    Path(args.out).write_text(build_report(rows, meta), encoding="utf-8")
    n_block = sum(1 for r in rows if r["_tier"] == 0)
    print(f"\nwrote {args.out}: {len(rows)} findings, {n_block} blocking")

    if args.gate and n_block:
        sys.exit(1)


if __name__ == "__main__":
    main()
