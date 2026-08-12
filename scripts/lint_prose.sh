#!/usr/bin/env bash
# Vale prose lint for the COLM 2026 manuscript.
#
#   scripts/lint_prose.sh                 # one line per alert
#   scripts/lint_prose.sh --summary       # counts per rule
#   scripts/lint_prose.sh 03_methods.tex  # lint specific file(s)
#
# Thin wrapper over scripts/prose_lint.py, which strips LaTeX comments before
# handing the text to Vale (see that file for why that cannot be done in
# .vale.ini). For a reviewable checklist, use scripts/prose_report.py instead.
#
# Exits non-zero when anything is found, so this is safe to wire into CI.
set -uo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/prose_lint.py" "$@"
