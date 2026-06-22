"""Tests for ``JudgeClient.export_batch_jsonl`` and ``merge_batch_output``.

Verifies that the batch-export path runs offline (no API key, no live
OpenAI client) and that the merge-roundtrip correctly reattaches
per-line responses to the pending parquet by ``custom_id``.

Per-dagspace fanout tests (privacylens / cirl_vignettes) live in
``tests/privacylens/test_batch_export.py`` and
``tests/cirl_vignettes/test_batch_export.py`` respectively.

Migrated from ``scripts/test_batch_export.py`` on 2026-05-12. The
original ``test_vllm_rejected`` test was deleted at migration time:
upstream production deliberately removed the vLLM rejection
(``judge_client.py:380-387`` — async-judge sidecar consumes the same
JSONL shape as the OpenAI Batch API). Replaced with
``test_vllm_provider_allowed_for_batch_export`` (regression-pins the
fix) and ``test_non_openai_non_vllm_provider_warns_but_succeeds``
(pins the anthropic/gemini WARN-but-continue behavior).
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List

import pytest


@pytest.fixture(autouse=True)
def _scrub_openai_api_key(monkeypatch):
    """Batch-export must work without any credentials locally so the
    JSONL can be handed off to a user who submits from their own account.
    Force-remove ``OPENAI_API_KEY`` for every test in this module."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _count_lines(path: str) -> int:
    with open(path) as f:
        return sum(1 for line in f if line.strip())


def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# JudgeClient.export_batch_jsonl
# ---------------------------------------------------------------------------

class TestJudgeClientExport:
    def test_vllm_provider_allowed_for_batch_export(self, tmp_path):
        """vLLM provider must be accepted by ``export_batch_jsonl``.

        The async-mode judge sidecar (``dagspaces/common/batch_api.py``)
        consumes the exact same OpenAI-Batch-shaped JSONL the production
        code emits here. Earlier production refused ``provider='vllm'``
        which forced callers to fake ``provider='openai'`` and stuff
        OpenAI-only model names like ``gpt-5.2`` into ``body.model`` —
        every request then 404'd against the vLLM endpoint. The rejection
        was removed in ``judge_client.py:380-387`` with an explicit
        comment; this test pins the new behavior so a future revert is
        caught immediately rather than silently breaking the async-judge
        pipeline.
        """
        from dagspaces.common.judge_client import JudgeClient

        client = JudgeClient(
            base_url="http://localhost:8002/v1",
            model_name="qwen3-8b-instruct",  # explicit, not the 'default' sentinel
        )
        assert client.provider == "vllm"

        out_path = str(tmp_path / "vllm_requests.jsonl")
        manifest = client.export_batch_jsonl(
            items=[{"x": 1}],
            build_messages_fn=lambda item: [{"role": "user", "content": "hi"}],
            output_path=out_path,
            custom_id_fn=lambda item, idx: f"req-{idx}",
        )

        assert manifest["count"] == 1
        assert manifest["provider"] == "vllm"
        assert manifest["model"] == "qwen3-8b-instruct"
        # Confirm the JSONL line landed with the vLLM model in body.model —
        # this is precisely the regression scenario the rejection used to
        # block (and that re-enabling vLLM was meant to support).
        lines = _load_jsonl(out_path)
        assert len(lines) == 1
        assert lines[0]["body"]["model"] == "qwen3-8b-instruct"

    def test_non_openai_non_vllm_provider_warns_but_succeeds(self, tmp_path, capfd):
        """Anthropic/Gemini share the OpenAI-compat JSONL shape but need
        their own Batch endpoints downstream. Production emits a WARN
        on stderr/stdout but does not raise (``judge_client.py:388-395``).
        Pin both halves: (1) export succeeds, (2) WARN is printed so
        operators notice they need to re-point the downstream consumer.
        """
        from dagspaces.common.judge_client import JudgeClient

        client = JudgeClient(
            base_url="https://api.anthropic.com/v1",
            model_name="claude-3-5-sonnet-20241022",
            offline=True,  # skips the commercial-provider API-key requirement
        )
        assert client.provider == "anthropic"

        out_path = str(tmp_path / "anthropic_requests.jsonl")
        manifest = client.export_batch_jsonl(
            items=[{"x": 1}],
            build_messages_fn=lambda item: [{"role": "user", "content": "hi"}],
            output_path=out_path,
            custom_id_fn=lambda item, idx: f"req-{idx}",
        )

        assert manifest["count"] == 1
        assert manifest["provider"] == "anthropic"
        captured = capfd.readouterr()
        combined = captured.out + captured.err
        assert "WARN" in combined, (
            "non-openai/non-vllm provider must emit a WARN so operators "
            "know to re-point the downstream consumer at the right "
            "batch endpoint"
        )
        assert "anthropic" in combined, "WARN should name the actual provider"

    def test_basic_export_writes_jsonl_with_unique_ids_and_schema(self, tmp_path):
        """Offline mode: no API key, no live OpenAI client constructed."""
        from dagspaces.common.judge_client import JudgeClient

        client = JudgeClient(
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o-mini",
            temperature=0.0,
            max_tokens=256,
            offline=True,
        )
        assert client.provider == "openai"
        assert client._client is None, "offline mode should not build an OpenAI client"
        assert client._api_key == "", "offline mode should work without an API key"

        items = [{"text": f"hello {i}"} for i in range(3)]
        out_path = str(tmp_path / "requests.jsonl")

        def build_messages(item: Dict[str, Any]):
            return [
                {"role": "system", "content": "You are a judge."},
                {"role": "user", "content": item["text"]},
            ]

        manifest = client.export_batch_jsonl(
            items=items,
            build_messages_fn=build_messages,
            output_path=out_path,
            custom_id_fn=lambda item, idx: f"smoke:{idx}",
            json_schema={
                "type": "object",
                "properties": {"answer": {"type": "string"}},
            },
            schema_name="SmokeResult",
        )

        assert manifest["count"] == 3
        assert manifest["model"] == "gpt-4o-mini"
        assert manifest["provider"] == "openai"
        assert manifest["schema_name"] == "SmokeResult"
        assert _count_lines(out_path) == 3

        lines = _load_jsonl(out_path)
        seen_ids = set()
        for i, line in enumerate(lines):
            assert line["method"] == "POST"
            assert line["url"] == "/v1/chat/completions"
            assert line["custom_id"] == f"smoke:{i}"
            seen_ids.add(line["custom_id"])

            body = line["body"]
            assert body["model"] == "gpt-4o-mini"
            assert body["temperature"] == 0.0
            assert body["max_tokens"] == 256
            assert body["messages"][0]["role"] == "system"
            assert body["messages"][1]["content"] == f"hello {i}"
            assert body["response_format"]["type"] == "json_schema"
            assert body["response_format"]["json_schema"]["name"] == "SmokeResult"
        assert len(seen_ids) == 3

    def test_duplicate_custom_ids_rejected(self, tmp_path):
        from dagspaces.common.judge_client import JudgeClient

        client = JudgeClient(
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o-mini",
            offline=True,
        )
        with pytest.raises(ValueError, match="duplicate custom_id"):
            client.export_batch_jsonl(
                items=[{"i": 0}, {"i": 1}],
                build_messages_fn=lambda _item: [{"role": "user", "content": "x"}],
                output_path=str(tmp_path / "dup.jsonl"),
                custom_id_fn=lambda _item, _idx: "same-id",
            )

    def test_offline_mode_blocks_live_calls(self):
        """Offline JudgeClient must refuse ``health_check`` and ``_call_single``.

        Without this guard, a misconfigured pipeline could silently
        fall back from batch-export to live-judging mid-run, defeating
        the whole "export only, hand off the JSONL" workflow."""
        from dagspaces.common.judge_client import JudgeClient

        client = JudgeClient(
            base_url="https://api.openai.com/v1",
            model_name="gpt-4o-mini",
            offline=True,
        )
        with pytest.raises(RuntimeError, match=r"(?i)offline"):
            client.health_check()
        with pytest.raises(RuntimeError, match=r"(?i)offline"):
            client._call_single([{"role": "user", "content": "hi"}])

    def test_default_model_name_rejected(self, tmp_path):
        """``model_name='default'`` is a sentinel for the vLLM probe path
        and is meaningless for batch export — operators must spell out
        the target model so typos like ``gpt-5.2`` fail fast."""
        from dagspaces.common.judge_client import JudgeClient

        client = JudgeClient(
            base_url="https://api.openai.com/v1",
            model_name="default",
            offline=True,
        )
        with pytest.raises(ValueError, match="explicit judge.model_name|explicit model_name"):
            client.export_batch_jsonl(
                items=[{"i": 0}],
                build_messages_fn=lambda _item: [{"role": "user", "content": "x"}],
                output_path=str(tmp_path / "nope.jsonl"),
                custom_id_fn=lambda _item, idx: f"req-{idx}",
            )


# ---------------------------------------------------------------------------
# merge_batch_output
# ---------------------------------------------------------------------------

class TestMergeBatchOutput:
    def test_roundtrip_matches_by_custom_id_with_missing(self, tmp_path):
        """Output JSONL may be missing some custom_ids (rate-limit drops,
        moderation rejects, etc.). ``merge_batch_output`` must attach
        responses to the pending rows where present, leave the rest
        empty, and report counts for the finalize step's sanity check."""
        import pandas as pd
        from dagspaces.common.batch_api import merge_batch_output

        pending = pd.DataFrame([
            {"judge_custom_id": "smoke:0", "payload": "A"},
            {"judge_custom_id": "smoke:1", "payload": "B"},
            {"judge_custom_id": "smoke:2", "payload": "C"},
        ])
        pending_path = str(tmp_path / "pending.parquet")
        pending.to_parquet(pending_path, index=False)

        output_lines = [
            {
                "custom_id": "smoke:0",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{
                            "index": 0,
                            "message": {"role": "assistant",
                                        "content": '{"answer": "Yes"}'},
                            "finish_reason": "stop",
                        }],
                    },
                },
                "error": None,
            },
            {
                "custom_id": "smoke:2",
                "response": {
                    "status_code": 200,
                    "body": {
                        "choices": [{
                            "message": {"role": "assistant",
                                        "content": '{"answer": "No"}'},
                        }],
                    },
                },
                "error": None,
            },
        ]
        output_path = str(tmp_path / "output.jsonl")
        with open(output_path, "w") as f:
            for line in output_lines:
                f.write(json.dumps(line) + "\n")

        results_path = str(tmp_path / "results.parquet")
        stats = merge_batch_output(
            pending_parquet=pending_path,
            output_jsonl=output_path,
            text_column="leak_judge_text",
            out_parquet=results_path,
        )
        assert stats["matched"] == 2
        assert stats["missing"] == 1

        merged = pd.read_parquet(results_path)
        assert merged.loc[0, "leak_judge_text"] == '{"answer": "Yes"}'
        assert merged.loc[1, "leak_judge_text"] == "", (
            "smoke:1 was missing from the output JSONL; merged cell must be empty"
        )
        assert merged.loc[2, "leak_judge_text"] == '{"answer": "No"}'
