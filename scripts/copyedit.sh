#!/usr/bin/env bash
# Full copyedit pass over the COLM 2026 manuscript.
#
#   scripts/copyedit.sh              # every pass, fresh compile  (~4 min)
#   scripts/copyedit.sh --fast       # skip LanguageTool          (~20 s)
#   scripts/copyedit.sh --gate       # exit non-zero on a blocking defect
#
# Writes COPYEDIT_REVIEW.md in the repo root. The compile happens in
# /tmp, so the author's own .pdf and .aux are never touched.
#
# The companion read-through (coherency, terminology drift, voice) is not
# mechanizable and lives in COHERENCY_REVIEW.md; see wiki/copyediting.md for
# how the two fit together.
set -uo pipefail

PROJECT_ROOT=/share/pierson/matt/UAIR
exec "$PROJECT_ROOT/.venv/bin/python" "$PROJECT_ROOT/scripts/copyedit.py" "$@"
