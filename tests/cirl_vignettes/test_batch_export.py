"""Tests for ``dagspaces/cirl_vignettes/stages`` batch-export entrypoints.

Verifies that ``export_leakage_judge_batch`` and
``export_helpfulness_judge_batch`` skip rows lacking ``Action:`` (the
upstream-faithful gate) and emit the correct per-secret leakage fanout
and per-row helpfulness fanout.

The judge-client mechanics live in
``tests/common/test_judge_client_export.py``; this file isolates the
cirl_vignettes-specific fanout shapes and the skip-on-no-action gate.

Migrated from ``scripts/test_batch_export.py::test_cirl_vignettes_stages``
on 2026-05-12.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pytest


@pytest.fixture(autouse=True)
def _scrub_openai_api_key(monkeypatch):
    """Batch-export must work without credentials locally."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _make_cirl_df():
    """3-row CIRL DataFrame exercising the no-action skip path.

    Row 0: 2 secrets, valid ``Action:`` → 2 leakage requests + 1 helpfulness.
    Row 1: 1 secret, NO ``Action:`` → 0 requests of either kind (skipped).
    Row 2: 1 secret, valid ``Action:`` → 1 leakage + 1 helpfulness.

    Totals: 3 leakage fanout requests, 2 helpfulness requests.
    """
    import pandas as pd

    return pd.DataFrame([
        {
            "user_name": "Alice",
            "user_instruction": "Reply to Bob.",
            "executable_trajectory": "[...]",
            "sensitive_info_items": json.dumps(["HIV status", "home address"]),
            "final_action_generated": "Action: send_email(body='...')",
        },
        {
            "user_name": "Carol",
            "user_instruction": "Reply to Dave.",
            "executable_trajectory": "[...]",
            "sensitive_info_items": json.dumps(["SSN"]),
            "final_action_generated": "no action here",  # skipped
        },
        {
            "user_name": "Eve",
            "user_instruction": "Reply to Frank.",
            "executable_trajectory": "[...]",
            "sensitive_info_items": json.dumps(["phone"]),
            "final_action_generated": "Action: send_email(body='call 555-1212')",
        },
    ])


def _batch_export_cfg():
    """Minimal cfg with ``judge.mode=batch_export``."""
    from omegaconf import OmegaConf
    return OmegaConf.create({
        "judge": {
            "mode": "batch_export",
            "temperature": 0.0,
            "max_tokens": 256,
            "batch": {
                "target_model": "gpt-4o-mini",
                "target_endpoint": "/v1/chat/completions",
            },
        },
    })


# ---------------------------------------------------------------------------
# Leakage judge fanout — per (row, secret), skipping no-action rows
# ---------------------------------------------------------------------------

class TestExportLeakageJudgeBatch:
    def test_fanout_skips_no_action_rows(self, tmp_path):
        """row 0 × 2 secrets + row 2 × 1 secret = 3 requests; row 1 is
        skipped because its ``final_action_generated`` lacks ``Action:``.
        Skip indices are preserved in the custom_id (``..._judge:0:0``,
        ``..._judge:0:1``, ``..._judge:2:0``) so finalize can
        reconstruct row alignment after the merge."""
        from dagspaces.cirl_vignettes.stages.judge_leakage import (
            export_leakage_judge_batch,
        )

        out_dir = str(tmp_path / "cirl_leakage")
        export_leakage_judge_batch(_make_cirl_df(), _batch_export_cfg(), out_dir)

        lines = _load_jsonl(os.path.join(out_dir, "requests.jsonl"))
        assert len(lines) == 3
        cids = [line["custom_id"] for line in lines]
        assert cids == [
            "cirl_vignettes:judge_leakage:0:0",
            "cirl_vignettes:judge_leakage:0:1",
            "cirl_vignettes:judge_leakage:2:0",
        ]

    def test_items_parquet_records_row_secret_mapping(self, tmp_path):
        """``items.parquet`` carries the reverse-lookup
        (judge_custom_id → row_idx, sub_idx, secret) used by the
        finalize step to OR-aggregate per-secret answers back into a
        single ``leak_judge_text`` per row."""
        import pandas as pd
        from dagspaces.cirl_vignettes.stages.judge_leakage import (
            export_leakage_judge_batch,
        )

        out_dir = str(tmp_path / "cirl_leakage")
        export_leakage_judge_batch(_make_cirl_df(), _batch_export_cfg(), out_dir)
        items_df = pd.read_parquet(os.path.join(out_dir, "items.parquet"))
        assert len(items_df) == 3
        assert set(items_df.columns) == {"judge_custom_id", "row_idx", "sub_idx", "secret"}


# ---------------------------------------------------------------------------
# Helpfulness judge fanout — per row, skipping no-action rows
# ---------------------------------------------------------------------------

class TestExportHelpfulnessJudgeBatch:
    def test_one_request_per_row_skips_no_action(self, tmp_path):
        """rows 0 and 2 have valid actions → 2 requests. Row 1 skipped."""
        from dagspaces.cirl_vignettes.stages.judge_helpfulness import (
            export_helpfulness_judge_batch,
        )

        out_dir = str(tmp_path / "cirl_helpfulness")
        export_helpfulness_judge_batch(_make_cirl_df(), _batch_export_cfg(), out_dir)
        lines = _load_jsonl(os.path.join(out_dir, "requests.jsonl"))
        assert len(lines) == 2
        cids = [line["custom_id"] for line in lines]
        assert cids == [
            "cirl_vignettes:judge_helpfulness:0",
            "cirl_vignettes:judge_helpfulness:2",
        ]
