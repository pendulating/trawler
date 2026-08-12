#!/usr/bin/env python
"""Turn Vale's JSON output into a reviewable checklist.

    scripts/prose_report.py            # writes PROSE_REVIEW.md in the repo root
    scripts/prose_report.py -o /tmp/x.md

Every item carries a verbatim slice of the source line so the spot can be found
with CTRL-F in the .tex file. The manuscript is edited concurrently, so the
report records the git HEAD and file mtimes it was generated against; regenerate
rather than trusting line numbers from an old copy.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prose_lint import PAPER, ROOT, collect_alerts  # noqa: E402

# (heading, blurb, [rule names]) in the order they should be worked through.
TIERS: list[tuple[str, str, list[str]]] = [
    (
        "Unfinished content",
        "Placeholders that will render into the PDF. Fix these first.",
        ["proselint.Annotations"],
    ),
    (
        "Spelling and terminology",
        "Typos, unknown words, and project names that drift from their canonical spelling. "
        "A word flagged here that is actually correct belongs in "
        "`.vale/styles/config/vocabularies/NormSim/accept.txt`.",
        [
            "Vale.Spelling",
            "Vale.Avoid",
            "Vale.Terms",
            "proselint.Spelling",
            "NormSim.Terms",
            "NormSim.Acronyms",
        ],
    ),
    (
        "Punctuation and house style",
        "Rules taken from `academic-writing-mini.md`: no em dashes, `e.g.,` / `i.e.,` take a "
        "trailing comma, and stock academic filler gets tightened.",
        ["NormSim.EmDash", "NormSim.LatinAbbrev", "NormSim.Filler"],
    ),
    (
        "Word choice",
        "Hedges and vague intensifiers that weaken a claim, plus absolutes used comparatively.",
        ["write-good.Weasel", "write-good.ThereIs", "proselint.Uncomparables"],
    ),
    (
        "Passive voice",
        "The style guide asks for `we` as the active subject of research actions. This is the "
        "largest and lowest-confidence group: passive voice is correct in plenty of methods "
        "sentences, so treat it as a reading list rather than a defect list.",
        ["write-good.Passive"],
    ),
]


def inline_code(text: str) -> str:
    """Wrap text in an inline code span, fencing around any backticks it holds.

    LaTeX uses ` as an opening quote, so a naive single-backtick span breaks
    on a lot of real manuscript prose.
    """
    text = text.replace("\n", " ").strip()
    longest, run = 0, 0
    for ch in text:
        run = run + 1 if ch == "`" else 0
        longest = max(longest, run)
    fence = "`" * (longest + 1)
    pad = " " if text.startswith("`") or text.endswith("`") else ""
    return f"{fence}{pad}{text}{pad}{fence}"


def context_for(line_text: str, match: str, width: int = 60) -> str:
    """A slice of the raw source line centred on the match, for CTRL-F."""
    stripped = line_text.rstrip("\n")
    # A match can straddle a wrapped line ("is\nblanked"); the alert is anchored
    # to the line the match starts on, so locate using that first segment.
    head = match.split("\n")[0] if match else ""
    idx = stripped.find(head) if head else -1
    if idx == -1:
        snippet = stripped[: width * 2]
        return snippet + ("..." if len(stripped) > width * 2 else "")
    start = max(0, idx - width)
    end = min(len(stripped), idx + len(head) + width)
    return ("..." if start else "") + stripped[start:end] + ("..." if end < len(stripped) else "")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("-o", "--out", default=str(ROOT / "PROSE_REVIEW.md"))
    args = ap.parse_args()

    alerts, files = collect_alerts()

    # Cache source lines so each alert can quote its own line.
    lines: dict[str, list[str]] = {}
    for f in files:
        lines[f.name] = f.read_text(encoding="utf-8", errors="replace").splitlines()

    # Flatten: rule -> [(filename, alert)]
    by_rule: dict[str, list[tuple[str, dict]]] = {}
    for a in alerts:
        by_rule.setdefault(a["Check"], []).append((Path(a["File"]).name, a))
    total = len(alerts)

    try:
        head = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=PAPER,
        ).stdout.strip() or "unknown"
    except OSError:
        head = "unknown"

    out: list[str] = []
    w = out.append
    w("# Prose review: `colm26_normative-simulacra`")
    w("")
    w(f"{total} suggestions from Vale across {len(files)} manuscript files. ")
    w("Each item quotes enough of its source line to be found with CTRL-F.")
    w("")
    w("> **This is a snapshot.** The manuscript is under active edit, so line")
    w("> numbers drift. Regenerate with `scripts/prose_report.py` before working")
    w("> through it, and re-lint with `scripts/lint_prose.sh`.")
    w("")
    w(f"- Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    w(f"- Paper submodule HEAD: `{head}` (plus uncommitted working-tree edits)")
    w(f"- Config: `.vale.ini`, custom rules in `.vale/styles/NormSim/`")
    w("")
    w("Counts by rule:")
    w("")
    w("| Rule | Count |")
    w("| --- | ---: |")
    for rule, items in sorted(by_rule.items(), key=lambda kv: -len(kv[1])):
        w(f"| `{rule}` | {len(items)} |")
    w("")

    seen: set[str] = set()
    for heading, blurb, rules in TIERS:
        present = [r for r in rules if r in by_rule]
        if not present:
            continue
        count = sum(len(by_rule[r]) for r in present)
        w("---")
        w("")
        w(f"## {heading} ({count})")
        w("")
        w(blurb)
        w("")
        for rule in present:
            seen.add(rule)
            items = by_rule[rule]
            w(f"### `{rule}` ({len(items)})")
            w("")
            # Group by file, then order by line.
            per_file: dict[str, list[dict]] = {}
            for fname, a in items:
                per_file.setdefault(fname, []).append(a)
            for fname in sorted(per_file):
                w(f"**`{fname}`**")
                w("")
                for a in sorted(per_file[fname], key=lambda x: x["Line"]):
                    src = lines.get(fname, [])
                    line_text = src[a["Line"] - 1] if 0 < a["Line"] <= len(src) else ""
                    ctx = context_for(line_text, a.get("Match", ""))
                    w(f"- [ ] **L{a['Line']}** — {a['Message']}")
                    if ctx:
                        w(f"  - {inline_code(ctx)}")
                w("")

    leftovers = sorted(set(by_rule) - seen)
    if leftovers:
        w("---")
        w("")
        w("## Uncategorised")
        w("")
        w("Rules that fired but are not in this script's `TIERS` table; add them there.")
        w("")
        for rule in leftovers:
            w(f"### `{rule}` ({len(by_rule[rule])})")
            w("")
            for fname, a in sorted(by_rule[rule], key=lambda kv: (kv[0], kv[1]["Line"])):
                src = lines.get(fname, [])
                line_text = src[a["Line"] - 1] if 0 < a["Line"] <= len(src) else ""
                ctx = context_for(line_text, a.get("Match", ""))
                w(f"- [ ] **`{fname}` L{a['Line']}** — {a['Message']}")
                if ctx:
                    w(f"  - {inline_code(ctx)}")
            w("")

    Path(args.out).write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"wrote {args.out} ({total} items across {len(by_rule)} rules)")


if __name__ == "__main__":
    main()
