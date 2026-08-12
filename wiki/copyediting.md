# Copyediting the manuscript

How the COLM camera-ready gets checked for reference integrity, English, and
coherency. Entry point: `scripts/copyedit.sh`.

## Run it

```bash
scripts/copyedit.sh --fast     # ~20 s: everything except LanguageTool
scripts/copyedit.sh            # ~4 min: adds LanguageTool and a fresh compile
scripts/copyedit.sh --gate     # same, but exits non-zero on a blocking defect
```

Output is `COPYEDIT_REVIEW.md` in the repo root: one checklist, tiered by what
it costs to be wrong, with enough of each source line to find the spot with
CTRL-F. Regenerate before working through it, because line numbers drift.

The compile runs with `-outdir=/tmp/copyedit-build-$USER`, so the author's
`00_main.pdf` and `.aux` are never overwritten.

## The four passes

| Pass | Script | Catches |
| --- | --- | --- |
| Reference integrity | `tex_integrity.py` | anything that prints `??` or `[?]` |
| Grammar and spelling | `tex_grammar.py` | agreement, articles, typos, capitalization, en-GB/en-US drift |
| House style | `prose_lint.py` (Vale) | `academic-writing-mini.md` as rules |
| Coherency and voice | a human read-through | argument, terminology drift, slop |

The first three are wired so they do not duplicate each other. They overlap at
exactly one point, spelling, and that is split deliberately: **Vale** owns
dictionary spelling (with the allowlist at
`.vale/styles/config/vocabularies/NormSim/accept.txt`), **codespell** owns known
misspellings, and LanguageTool's own speller is switched off. Adding a word to
Vale's `accept.txt` silences it in both passes.

## Reference integrity: the `??` check

Two independent checks, because neither alone is trustworthy:

- **Static.** Parses the sources, matches every `\ref`/`\autoref`/`\cref`
  against every `\label` and every `\cite` key against the loaded `.bib` files.
  Needs no compile and no warm `.aux`, which matters because this manuscript is
  edited in Overleaf and locally at once, so the `.aux` on disk is routinely
  older than the `.tex` files. **A stale log is how a `??` reaches a deadline.**
- **Build (`--build`).** Compiles, reads LaTeX's own "Reference `x' undefined"
  warnings, then extracts the text of every rendered page with `pypdf` and
  greps it for `??` and `[?]`. That last step is the only check that sees what
  actually printed, so it also catches markers arriving through routes a source
  scan cannot model.

It also reports duplicate labels (which silently resolve to the wrong number),
`\label` placed before `\caption` inside a float (which numbers the float as
the previous one), floats nothing references, and missing graphics files.

Only files the document really builds are checked: `\input` is followed from
`00_main.tex`, so `scraps.tex` and `TADA_1pager.tex` never produce findings.

## House style is not a vote

`HEADING_CASE` **enforces** COLM's rule rather than inferring the document's
dominant style, and that distinction cost a round of wrong edits. The COLM
style file says, for all three heading levels:

> First level headings are in lower case (except for first word and proper
> nouns), bold face, flush left and in point size 12.

An earlier version of the rule inferred the majority style instead. On this
manuscript the appendices were written in Title Case, so Title Case won the
vote, and the rule recommended "correcting" the body headings *away* from what
the venue requires. When a venue specifies a convention, encode the convention;
only infer where the venue is silent. The same lower-case rule applies to
figure captions and table titles (audited clean 2026-08-09).

Proper nouns that stay capitalized live in `HEADING_PROPER_NOUNS` in
`tex_grammar.py`; add to it rather than loosening the rule.

## Two things worth knowing before you extend this

**Offsets are preserved, everywhere.** `tex_plain.py` de-TeXes by overwriting
markup *in place* with same-length runs of spaces, or with same-width stand-in
words (`\autoref{fig:x}` becomes `Figure 1`, `$\alpha$` becomes `x`,
`\citet{doe}` becomes `Author`). The de-TeXed string is the same length as the
source, so a checker's character offset maps back to the real line with no
translation step. Keep that invariant: it is what makes the grammar findings
point at the right place. `tex_plain.is_standin()` exists so a checker can tell
its own placeholder from something the author typed, and drop findings about it.

**Verbatim material is not copyedited.** `prompts/` and `traces/` reproduce
model input and output; their straight quotes, fragments, and JSON are the
artifact being documented, so "correcting" them would misreport what was run.
Those files are still checked for markup defects that break the build. Files are
recognized as verbatim by directory, or by a `DO NOT EDIT BY HAND` marker in
their first lines.

## The coherency pass

A linter reads one sentence at a time, so it cannot see that the 10-novel corpus
is called `fiction-10`, `fiction 10`, and `fiction10` in three different files,
or that a promise made in the introduction is never paid off. That pass is a
read-through against `academic-writing-mini.md` and the `no-ai-slop` skill, and
its findings live in `COHERENCY_REVIEW.md`. `COPYEDIT_REVIEW.md` ends with the
checklist it works from.

Record what you deliberately did *not* change, and why. `COHERENCY_REVIEW.md`
has a "deliberately not flagged" section for constructions that look like
defects but are house style: the one-word `Why?`, the `First,… Second,…`
signposting, the colon pivots. Without it, the next pass re-litigates them.

## Related

- `.vale.ini` — Vale config, with the LaTeX handling and each disabled rule's
  reason. Read the comments before changing `TokenIgnores`: the ordering is
  load-bearing and RE2 has no lookbehind.
- `academic-writing-mini.md` — the house style these rules encode.
- `CONGRUENCE.md`, `MANUSCRIPT_STALE_AUDIT.md` — paper-vs-code factual audits.
  Different job: those ask whether a number is *true*, this asks whether the
  prose is *right*.
