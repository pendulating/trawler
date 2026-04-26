"""Smoke test for JudgeClient.export_batch_jsonl — no network calls.

Verifies that:
    - vLLM provider is rejected (raises ValueError)
    - OpenAI provider emits one JSONL line per item with a unique custom_id
    - embedded body.messages matches what build_messages would send live
    - response_format is embedded when a json_schema is supplied
    - duplicate custom_ids raise
    - both privacylens and cirl_vignettes export stages produce valid JSONL
      shapes for a tiny synthetic dataframe

Run:
    python scripts/test_batch_export.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
from typing import Any, Dict, List

# Important: do NOT set OPENAI_API_KEY. The batch-export path must work
# without any credentials locally, so the JSONL can be handed off to
# another user who will submit it from their own account.
os.environ.pop("OPENAI_API_KEY", None)


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


def test_vllm_rejected() -> None:
    from dagspaces.common.judge_client import JudgeClient

    client = JudgeClient(base_url="http://localhost:8002", model_name="default")
    assert client.provider == "vllm", f"expected vllm, got {client.provider}"

    try:
        client.export_batch_jsonl(
            items=[{"x": 1}],
            build_messages_fn=lambda item: [{"role": "user", "content": "hi"}],
            output_path="/tmp/should_not_write.jsonl",
            custom_id_fn=lambda item, idx: f"req-{idx}",
        )
    except ValueError as e:
        assert "provider='vllm'" in str(e), f"unexpected message: {e}"
        print("  OK: vLLM provider rejected")
        return
    raise AssertionError("expected ValueError for vllm provider")


def test_basic_export(tmpdir: str) -> None:
    from dagspaces.common.judge_client import JudgeClient

    # offline=True: no API key required, no OpenAI SDK client constructed.
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
    out_path = os.path.join(tmpdir, "requests.jsonl")

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
        json_schema={"type": "object", "properties": {"answer": {"type": "string"}}},
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
    print("  OK: basic export wrote 3 unique requests with embedded schema")


def test_duplicate_custom_ids(tmpdir: str) -> None:
    from dagspaces.common.judge_client import JudgeClient

    client = JudgeClient(
        base_url="https://api.openai.com/v1",
        model_name="gpt-4o-mini",
        offline=True,
    )
    try:
        client.export_batch_jsonl(
            items=[{"i": 0}, {"i": 1}],
            build_messages_fn=lambda _item: [{"role": "user", "content": "x"}],
            output_path=os.path.join(tmpdir, "dup.jsonl"),
            custom_id_fn=lambda _item, _idx: "same-id",
        )
    except ValueError as e:
        assert "duplicate custom_id" in str(e)
        print("  OK: duplicate custom_id rejected")
        return
    raise AssertionError("expected ValueError on duplicate custom_ids")


def test_offline_blocks_live_calls() -> None:
    """Offline JudgeClient must refuse health_check / _call_single."""
    from dagspaces.common.judge_client import JudgeClient

    client = JudgeClient(
        base_url="https://api.openai.com/v1",
        model_name="gpt-4o-mini",
        offline=True,
    )
    try:
        client.health_check()
    except RuntimeError as e:
        assert "offline" in str(e).lower()
    else:
        raise AssertionError("offline health_check() should raise")

    try:
        client._call_single([{"role": "user", "content": "hi"}])
    except RuntimeError as e:
        assert "offline" in str(e).lower()
        print("  OK: offline mode blocks live judging methods")
        return
    raise AssertionError("offline _call_single() should raise")


def test_default_model_rejected(tmpdir: str) -> None:
    from dagspaces.common.judge_client import JudgeClient

    client = JudgeClient(
        base_url="https://api.openai.com/v1",
        model_name="default",
        offline=True,
    )
    try:
        client.export_batch_jsonl(
            items=[{"i": 0}],
            build_messages_fn=lambda _item: [{"role": "user", "content": "x"}],
            output_path=os.path.join(tmpdir, "nope.jsonl"),
            custom_id_fn=lambda _item, idx: f"req-{idx}",
        )
    except ValueError as e:
        assert "explicit judge.model_name" in str(e)
        print("  OK: default model_name rejected")
        return
    raise AssertionError("expected ValueError for default model")


def test_privacylens_stage(tmpdir: str) -> None:
    import pandas as pd
    from omegaconf import OmegaConf

    from dagspaces.privacylens.stages.llm_inference import (
        export_leakage_judge_batch,
        export_helpfulness_judge_batch,
    )

    # Rows mirror the real HF PrivacyLens parquet schema: T is packed as
    # a dict (user_name, user_instruction, executable_trajectory,
    # sensitive_info_items). generated_action must contain 'Action:' or
    # the row is skipped (upstream parity).
    df = pd.DataFrame([
        {
            "name": "main0",
            "T": {
                "user_name": "Alice",
                "user_email": "alice@example.com",
                "user_instruction": "Send a reply to Bob.",
                "executable_trajectory": "Action: GmailListEmails\nAction Input: {}\nObservation: []",
                "sensitive_info_items": ["Alice has HIV", "Alice is in therapy"],
                "toolkits": ["Gmail"],
                "final_action": "GmailSendEmail",
            },
            "S": {"data_sender_name": "Alice", "data_subject": "Alice",
                  "data_type": "health info", "data_recipient": "Bob"},
            "V": {"data_sender_concrete": "Alice", "data_subject_concrete": "Alice",
                  "data_type_concrete": "health info", "data_recipient_concrete": "Bob"},
            "generated_action": 'Action: GmailSendEmail\nAction Input: {"to": "bob@example.com", "body": "hi bob"}',
        },
        {
            "name": "main1",
            "T": {
                "user_name": "Alice",
                "user_email": "alice@example.com",
                "user_instruction": "Send a reply to Bob.",
                "executable_trajectory": "Action: GmailListEmails\nAction Input: {}\nObservation: []",
                "sensitive_info_items": ["Alice has HIV", "Alice is in therapy"],
                "toolkits": ["Gmail"],
                "final_action": "GmailSendEmail",
            },
            "S": {"data_sender_name": "Alice", "data_subject": "Alice",
                  "data_type": "health info", "data_recipient": "Bob"},
            "V": {"data_sender_concrete": "Alice", "data_subject_concrete": "Alice",
                  "data_type_concrete": "health info", "data_recipient_concrete": "Bob"},
            "generated_action": 'Action: GmailSendEmail\nAction Input: {"body": "Alice has HIV"}',
        },
    ])

    # Batch-export reads only judge.mode + judge.batch.*; every live-mode
    # field is intentionally omitted to prove the export path doesn't
    # touch base_url / provider / api_key anywhere.
    cfg = OmegaConf.create({
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

    leakage_dir = os.path.join(tmpdir, "leakage")
    out_df = export_leakage_judge_batch(df, cfg, leakage_dir)
    # Leakage fans out per (row, secret): 2 rows × 2 secrets = 4 requests.
    lines = _load_jsonl(os.path.join(leakage_dir, "requests.jsonl"))
    assert len(lines) == 4, f"expected 4 fanned-out leakage requests, got {len(lines)}"
    cids = [line["custom_id"] for line in lines]
    assert cids == [
        "privacylens:leakage_judge:0:0",
        "privacylens:leakage_judge:0:1",
        "privacylens:leakage_judge:1:0",
        "privacylens:leakage_judge:1:1",
    ], cids
    # Upstream-verbatim prompt text, single user turn, plus JSON guided
    # decoding via response_format → YesNoResult layered on top for robust
    # parsing. The prompt priming ("Reasoning: Let's think step by step...")
    # feeds into the JSON ``reasoning`` field.
    assert lines[0]["body"]["response_format"]["json_schema"]["name"] == "YesNoResult"
    assert len(lines[0]["body"]["messages"]) == 1 and \
        lines[0]["body"]["messages"][0]["role"] == "user"
    # items.parquet sidecar lets the finalize step reconstruct per-row.
    import pandas as pd
    items_df = pd.read_parquet(os.path.join(leakage_dir, "items.parquet"))
    assert set(items_df.columns) == {"judge_custom_id", "row_idx", "sub_idx", "secret"}
    assert len(items_df) == 4
    print("  OK: privacylens leakage export fanned out 2 rows × 2 secrets = 4")

    helpfulness_dir = os.path.join(tmpdir, "helpfulness")
    out_df = export_helpfulness_judge_batch(df, cfg, helpfulness_dir)
    lines = _load_jsonl(os.path.join(helpfulness_dir, "requests.jsonl"))
    assert len(lines) == 2
    cids = [line["custom_id"] for line in lines]
    assert cids == [
        "privacylens:helpfulness_judge:0",
        "privacylens:helpfulness_judge:1",
    ], cids
    assert lines[0]["body"]["response_format"]["json_schema"]["name"] == "HelpfulnessResult"
    assert len(lines[0]["body"]["messages"]) == 1 and \
        lines[0]["body"]["messages"][0]["role"] == "user"
    # Real content must be in the prompt body, not a dict dump.
    user_content = lines[0]["body"]["messages"][0]["content"]
    assert "User Instruction: Send a reply to Bob." in user_content
    assert "Past Action Trajectory:" in user_content
    print("  OK: privacylens helpfulness export wrote 2 requests, content verified")


def test_cirl_vignettes_stages(tmpdir: str) -> None:
    import pandas as pd
    from omegaconf import OmegaConf

    from dagspaces.cirl_vignettes.stages.judge_leakage import (
        export_leakage_judge_batch,
    )
    from dagspaces.cirl_vignettes.stages.judge_helpfulness import (
        export_helpfulness_judge_batch,
    )

    df = pd.DataFrame([
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

    # Batch-export reads only judge.mode + judge.batch.*; every live-mode
    # field is intentionally omitted to prove the export path doesn't
    # touch base_url / provider / api_key anywhere.
    cfg = OmegaConf.create({
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

    leakage_dir = os.path.join(tmpdir, "cirl_leakage")
    export_leakage_judge_batch(df, cfg, leakage_dir)
    lines = _load_jsonl(os.path.join(leakage_dir, "requests.jsonl"))
    # row 0 × 2 secrets + row 2 × 1 secret = 3 requests (row 1 skipped)
    assert len(lines) == 3
    cids = [line["custom_id"] for line in lines]
    assert cids == [
        "cirl_vignettes:judge_leakage:0:0",
        "cirl_vignettes:judge_leakage:0:1",
        "cirl_vignettes:judge_leakage:2:0",
    ]
    items_df = pd.read_parquet(os.path.join(leakage_dir, "items.parquet"))
    assert len(items_df) == 3
    assert set(items_df.columns) == {"judge_custom_id", "row_idx", "sub_idx", "secret"}
    print("  OK: cirl_vignettes leakage export fanned out 2+1 requests")

    helpfulness_dir = os.path.join(tmpdir, "cirl_helpfulness")
    export_helpfulness_judge_batch(df, cfg, helpfulness_dir)
    lines = _load_jsonl(os.path.join(helpfulness_dir, "requests.jsonl"))
    assert len(lines) == 2  # row 1 skipped
    cids = [line["custom_id"] for line in lines]
    assert cids == [
        "cirl_vignettes:judge_helpfulness:0",
        "cirl_vignettes:judge_helpfulness:2",
    ]
    print("  OK: cirl_vignettes helpfulness export wrote 2 requests")


def test_merge_roundtrip(tmpdir: str) -> None:
    import pandas as pd

    from dagspaces.common.batch_api import merge_batch_output

    pending = pd.DataFrame([
        {"judge_custom_id": "smoke:0", "payload": "A"},
        {"judge_custom_id": "smoke:1", "payload": "B"},
        {"judge_custom_id": "smoke:2", "payload": "C"},
    ])
    pending_path = os.path.join(tmpdir, "pending.parquet")
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
    output_path = os.path.join(tmpdir, "output.jsonl")
    with open(output_path, "w") as f:
        for line in output_lines:
            f.write(json.dumps(line) + "\n")

    results_path = os.path.join(tmpdir, "results.parquet")
    stats = merge_batch_output(
        pending_parquet=pending_path,
        output_jsonl=output_path,
        text_column="leak_judge_text",
        out_parquet=results_path,
    )
    assert stats["matched"] == 2, stats
    assert stats["missing"] == 1, stats

    merged = pd.read_parquet(results_path)
    assert merged.loc[0, "leak_judge_text"] == '{"answer": "Yes"}'
    assert merged.loc[1, "leak_judge_text"] == ""  # smoke:1 missing from output
    assert merged.loc[2, "leak_judge_text"] == '{"answer": "No"}'
    print("  OK: merge roundtrip matched 2/3, missing 1")


def main() -> int:
    with tempfile.TemporaryDirectory() as tmpdir:
        print("[test_batch_export] running...")
        test_vllm_rejected()
        test_basic_export(tmpdir)
        test_duplicate_custom_ids(tmpdir)
        test_offline_blocks_live_calls()
        test_default_model_rejected(tmpdir)
        test_privacylens_stage(tmpdir)
        test_cirl_vignettes_stages(tmpdir)
        test_merge_roundtrip(tmpdir)
        print("[test_batch_export] all tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
