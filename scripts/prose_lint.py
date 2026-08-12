#!/usr/bin/env python
"""Run Vale over the manuscript, with LaTeX comments stripped first.

    scripts/prose_lint.py                 # one line per alert
    scripts/prose_lint.py --summary       # counts per rule
    scripts/prose_lint.py --json          # raw alerts, for tooling
    scripts/prose_lint.py 03_methods.tex  # specific file(s)

Why the preprocessing pass exists
---------------------------------
Comments must not be linted: the manuscript carries commented-out paragraphs and
figure blocks that would otherwise produce alerts for text that never reaches the
PDF. The obvious way to do that is a TokenIgnores entry in .vale.ini, but it does
not work. Vale's regex engine is Go RE2, which has no lookbehind, so a pattern
cannot say "a % that is not preceded by a backslash" and therefore cannot tell a
comment from an escaped percent in "50\\%". Anchoring with (?m)^ to catch only
whole-line comments then behaves inconsistently inside Vale's Markdown block
parsing, and -- worse -- its presence made Vale *drop* genuine alerts later in
the file (a real "[TODO: cite]" went unreported whenever a comment appeared
earlier in the same file).

So comments are removed here instead, where Python's re does support lookbehind.
Comment bodies are overwritten with spaces rather than deleted, which keeps every
line number and column identical to the original file, so Vale's positions point
straight back into the real source.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

ROOT = Path("/share/pierson/matt/UAIR")
PAPER = ROOT / "papers" / "colm26_normative-simulacra"
VALE = ROOT / ".venv" / "bin" / "vale"
CONFIG = ROOT / ".vale.ini"

# An unescaped % starts a comment and runs to end of line. A %% is still a
# comment (the first % opens it), but \% is a literal percent sign.
COMMENT = re.compile(r"(?<!\\)%")


def strip_comments(text: str) -> str:
    """Blank out LaTeX comment bodies, preserving line and column positions."""
    out = []
    for line in text.split("\n"):
        m = COMMENT.search(line)
        if m:
            # Keep everything up to and including the '%', pad the rest with
            # spaces so downstream column numbers still line up.
            line = line[: m.start()] + " " * (len(line) - m.start())
        out.append(line)
    return "\n".join(out)


def target_files(names: list[str]) -> list[Path]:
    if names:
        # Relative subdirectory paths are preserved ("tables/canon.tex"), so a
        # caller can lint files the default glob does not reach. A bare name
        # still resolves against the paper root, as it always has.
        return [PAPER / n for n in names]
    return sorted(PAPER.glob("[0-9][0-9]_*.tex")) + sorted(PAPER.glob("[A-E]_*.tex"))


def collect_alerts(names: list[str] | None = None) -> tuple[list[dict], list[Path]]:
    """Lint the manuscript and return (alerts, files).

    Each alert is Vale's dict plus a "File" key naming the real source file.
    """
    files = [f for f in target_files(names or []) if f.is_file()]
    if not files:
        return [], []
    if not VALE.exists():
        sys.exit(
            f"vale not found at {VALE}\n"
            "install it with: uv pip install --python .venv/bin/python vale"
        )

    # Style packages are downloaded artifacts; fetch on a fresh checkout.
    if not (ROOT / ".vale" / "styles" / "proselint").is_dir():
        subprocess.run([str(VALE), "--config", str(CONFIG), "sync"], cwd=ROOT, check=False)

    alerts: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="prose-lint-") as tmp:
        shadow_dir = Path(tmp)
        shadow_to_real: dict[str, Path] = {}
        shadow_paths: list[Path] = []
        for f in files:
            # Flatten subdirectories into the shadow name so that
            # tables/canon.tex and canon.tex cannot collide on basename.
            try:
                stem = str(f.relative_to(PAPER)).replace("/", "__")
            except ValueError:
                stem = f.name
            shadow = shadow_dir / stem
            shadow.write_text(
                strip_comments(f.read_text(encoding="utf-8", errors="replace")),
                encoding="utf-8",
            )
            shadow_to_real[shadow.name] = f
            shadow_paths.append(shadow)

        proc = subprocess.run(
            [
                str(VALE),
                "--config", str(CONFIG),
                "--output=JSON",
                "--no-exit",
                *[str(s) for s in shadow_paths],
            ],
            capture_output=True,
            text=True,
            cwd=ROOT,
        )
        if not proc.stdout.strip():
            sys.exit(f"vale produced no output.\n{proc.stderr}")
        data = json.loads(proc.stdout)

    for path, file_alerts in data.items():
        real = shadow_to_real.get(Path(path).name)
        if real is None:
            continue
        for a in file_alerts:
            a["File"] = str(real)
            alerts.append(a)

    alerts.sort(key=lambda a: (a["File"], a["Line"], a.get("Span", [0])[0]))
    return alerts, files


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="manuscript file names (default: all)")
    ap.add_argument("--summary", action="store_true", help="counts per rule")
    ap.add_argument("--json", action="store_true", help="raw alerts as JSON")
    args = ap.parse_args()

    alerts, files = collect_alerts(args.files)

    if args.json:
        print(json.dumps(alerts, indent=2))
    elif args.summary:
        for rule, n in Counter(a["Check"] for a in alerts).most_common():
            print(f"{n:6d}  {rule}")
        print(f"{len(alerts):6d}  TOTAL across {len(files)} files")
    else:
        for a in alerts:
            rel = Path(a["File"]).name
            print(f"{rel}:{a['Line']}:{a.get('Span',[0])[0]}:{a['Check']}:{a['Message']}")

    # Match Vale's convention: non-zero when anything was found.
    sys.exit(1 if alerts else 0)


if __name__ == "__main__":
    main()
