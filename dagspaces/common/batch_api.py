"""OpenAI Batch API helper for judge-stage batch exports.

Thin wrapper around the ``openai`` SDK's Files + Batches endpoints that
matches Trawler's batch-export flow: the eval pipeline writes a
``requests.jsonl`` + ``manifest.json`` pair in batch-export mode
(see ``JudgeClient.export_batch_jsonl``), you submit the JSONL here,
poll for completion, download the output, and the pipeline resumes
from a batch-ingest stage.

CLI::

    # 1. Upload + create the batch
    python -m dagspaces.common.batch_api submit \\
        outputs/.../leakage_judge_export/requests.jsonl

    # 2. Poll status
    python -m dagspaces.common.batch_api status <batch_id>

    # 3. Download output.jsonl once status == completed
    python -m dagspaces.common.batch_api fetch <batch_id> \\
        -o outputs/.../leakage_judge_export/output.jsonl

    # 4. Join output.jsonl back into pending.parquet to produce results.parquet
    python -m dagspaces.common.batch_api merge \\
        --pending   outputs/.../leakage_judge_export/pending.parquet \\
        --output    outputs/.../leakage_judge_export/output.jsonl \\
        --text-column leak_judge_text \\
        --out       outputs/.../leakage_judge_export/results.parquet

The ``submit`` command writes the batch id back into the adjacent
``manifest.json`` so downstream tools can find it without copy-paste.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, Iterable, List, Optional

__all__ = [
    "submit_batch",
    "get_batch_status",
    "fetch_batch_output",
    "merge_batch_output",
    "load_jsonl",
    "extract_content",
]


def _client(api_key_env: Optional[str] = None):
    """Build an OpenAI SDK client using OPENAI_API_KEY (or a custom env var)."""
    from openai import OpenAI

    key = None
    if api_key_env:
        key = os.environ.get(api_key_env)
    key = key or os.environ.get("OPENAI_API_KEY")
    if not key:
        raise RuntimeError(
            "No OpenAI API key found. Set OPENAI_API_KEY or pass "
            "--api-key-env <NAME>."
        )
    return OpenAI(api_key=key)


# ---------------------------------------------------------------------------
# JSONL helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: str) -> List[Dict[str, Any]]:
    """Read a JSONL file into a list of dicts, skipping blank lines."""
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _update_manifest(jsonl_path: str, updates: Dict[str, Any]) -> Optional[str]:
    """Merge ``updates`` into the sibling ``manifest.json`` if one exists."""
    manifest_path = os.path.join(os.path.dirname(jsonl_path), "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
    except Exception:
        manifest = {}
    manifest.update(updates)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return manifest_path


# ---------------------------------------------------------------------------
# Batch operations
# ---------------------------------------------------------------------------

def submit_batch(
    jsonl_path: str,
    endpoint: str = "/v1/chat/completions",
    completion_window: str = "24h",
    metadata: Optional[Dict[str, str]] = None,
    api_key_env: Optional[str] = None,
) -> Dict[str, Any]:
    """Upload a JSONL file and create a batch job. Returns batch metadata."""
    client = _client(api_key_env)
    if not os.path.exists(jsonl_path):
        raise FileNotFoundError(jsonl_path)

    size_mb = os.path.getsize(jsonl_path) / 1024 / 1024
    print(f"[batch_api] uploading {jsonl_path} ({size_mb:.1f} MB)...", flush=True)
    with open(jsonl_path, "rb") as fh:
        file_obj = client.files.create(file=fh, purpose="batch")
    print(f"[batch_api] file uploaded: {file_obj.id}", flush=True)

    print(f"[batch_api] creating batch (endpoint={endpoint}, "
          f"window={completion_window})...", flush=True)
    batch = client.batches.create(
        input_file_id=file_obj.id,
        endpoint=endpoint,
        completion_window=completion_window,
        metadata=metadata or None,
    )
    print(f"[batch_api] batch created: {batch.id} (status={batch.status})",
          flush=True)

    info = {
        "batch_id": batch.id,
        "input_file_id": file_obj.id,
        "endpoint": endpoint,
        "status": batch.status,
        "created_at": batch.created_at,
    }
    mf = _update_manifest(jsonl_path, info)
    if mf:
        print(f"[batch_api] updated manifest: {mf}", flush=True)
    return info


def get_batch_status(batch_id: str, api_key_env: Optional[str] = None) -> Dict[str, Any]:
    """Retrieve the current status of a batch."""
    client = _client(api_key_env)
    batch = client.batches.retrieve(batch_id)
    counts = getattr(batch, "request_counts", None)
    return {
        "batch_id": batch.id,
        "status": batch.status,
        "endpoint": batch.endpoint,
        "created_at": batch.created_at,
        "in_progress_at": getattr(batch, "in_progress_at", None),
        "completed_at": getattr(batch, "completed_at", None),
        "failed_at": getattr(batch, "failed_at", None),
        "expired_at": getattr(batch, "expired_at", None),
        "output_file_id": getattr(batch, "output_file_id", None),
        "error_file_id": getattr(batch, "error_file_id", None),
        "request_counts": {
            "total": getattr(counts, "total", None) if counts else None,
            "completed": getattr(counts, "completed", None) if counts else None,
            "failed": getattr(counts, "failed", None) if counts else None,
        },
    }


def fetch_batch_output(
    batch_id: str,
    output_path: str,
    error_path: Optional[str] = None,
    api_key_env: Optional[str] = None,
) -> Dict[str, Any]:
    """Download output.jsonl (and error file if present) for a completed batch."""
    client = _client(api_key_env)
    batch = client.batches.retrieve(batch_id)
    if batch.status != "completed":
        print(f"[batch_api] WARN: batch status is {batch.status!r}, "
              f"not 'completed'. Downloading whatever is available.",
              flush=True)

    result: Dict[str, Any] = {"batch_id": batch_id, "status": batch.status}
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    out_id = getattr(batch, "output_file_id", None)
    if out_id:
        print(f"[batch_api] downloading output_file_id={out_id} -> "
              f"{output_path}", flush=True)
        resp = client.files.content(out_id)
        # SDK returns an HttpxBinaryResponseContent with a .read() method.
        with open(output_path, "wb") as f:
            f.write(resp.read())
        result["output_path"] = output_path
    else:
        print("[batch_api] no output_file_id available", flush=True)

    err_id = getattr(batch, "error_file_id", None)
    if err_id:
        err_path = error_path or (output_path + ".errors")
        print(f"[batch_api] downloading error_file_id={err_id} -> "
              f"{err_path}", flush=True)
        resp = client.files.content(err_id)
        with open(err_path, "wb") as f:
            f.write(resp.read())
        result["error_path"] = err_path

    return result


# ---------------------------------------------------------------------------
# Merge: join output.jsonl back into pending parquet
# ---------------------------------------------------------------------------

def extract_content(response_line: Dict[str, Any]) -> str:
    """Pull the assistant message content out of a Batch API output line."""
    err = response_line.get("error")
    if err:
        return json.dumps({"error": err})
    resp = response_line.get("response") or {}
    body = resp.get("body") or {}
    choices = body.get("choices") or []
    if not choices:
        return json.dumps({"error": "empty_choices", "status": resp.get("status_code")})
    msg = (choices[0] or {}).get("message") or {}
    return msg.get("content") or ""


def merge_batch_output(
    pending_parquet: str,
    output_jsonl: str,
    text_column: str,
    out_parquet: str,
    custom_id_column: str = "judge_custom_id",
) -> Dict[str, Any]:
    """Join a Batch API output JSONL into a pending parquet dataframe.

    The pending parquet must carry a ``judge_custom_id`` column written by
    the export stage. Each output line's ``custom_id`` is joined against it,
    and the assistant message content is written into ``text_column``. Any
    row whose ``custom_id`` is missing from the output file gets an empty
    string in ``text_column`` (same shape live mode produces on judge error).
    """
    import pandas as pd

    df = pd.read_parquet(pending_parquet)
    if custom_id_column not in df.columns:
        raise ValueError(
            f"{pending_parquet} is missing the {custom_id_column!r} column; "
            f"was it written by a batch-export stage?"
        )

    lines = load_jsonl(output_jsonl)
    by_cid: Dict[str, str] = {}
    failed: List[str] = []
    for line in lines:
        cid = line.get("custom_id")
        if not cid:
            continue
        content = extract_content(line)
        by_cid[cid] = content
        if line.get("error"):
            failed.append(cid)

    merged = df.copy()
    merged[text_column] = merged[custom_id_column].map(by_cid).fillna("")

    os.makedirs(os.path.dirname(out_parquet) or ".", exist_ok=True)
    merged.to_parquet(out_parquet, index=False)

    stats = {
        "pending_rows": int(len(df)),
        "output_rows": int(len(lines)),
        "matched": int((merged[text_column].astype(bool)).sum()),
        "missing": int((~merged[text_column].astype(bool)).sum()),
        "failed_custom_ids": failed,
        "out": out_parquet,
    }
    print(
        f"[batch_api] merged {stats['matched']}/{stats['pending_rows']} rows "
        f"into {out_parquet} (missing={stats['missing']}, "
        f"failed={len(failed)})",
        flush=True,
    )
    return stats


# ---------------------------------------------------------------------------
# CLI dispatch
# ---------------------------------------------------------------------------

def _cmd_submit(args: argparse.Namespace) -> int:
    info = submit_batch(
        args.jsonl,
        endpoint=args.endpoint,
        completion_window=args.window,
        api_key_env=args.api_key_env,
    )
    print(json.dumps(info, indent=2, default=str))
    return 0


def _cmd_status(args: argparse.Namespace) -> int:
    info = get_batch_status(args.batch_id, api_key_env=args.api_key_env)
    print(json.dumps(info, indent=2, default=str))
    return 0 if info["status"] in ("completed", "in_progress", "validating", "finalizing") else 1


def _cmd_fetch(args: argparse.Namespace) -> int:
    info = fetch_batch_output(
        args.batch_id,
        output_path=args.out,
        error_path=args.error_out,
        api_key_env=args.api_key_env,
    )
    print(json.dumps(info, indent=2, default=str))
    return 0


def _cmd_merge(args: argparse.Namespace) -> int:
    stats = merge_batch_output(
        pending_parquet=args.pending,
        output_jsonl=args.output,
        text_column=args.text_column,
        out_parquet=args.out,
        custom_id_column=args.custom_id_column,
    )
    print(json.dumps(stats, indent=2, default=str))
    return 0


def main(argv: Optional[Iterable[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m dagspaces.common.batch_api",
        description="OpenAI Batch API helpers for Trawler judge exports.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--api-key-env", default=None,
        help="Environment variable holding the OpenAI API key "
             "(default: OPENAI_API_KEY).",
    )

    p_submit = sub.add_parser("submit", parents=[common],
                              help="Upload a JSONL and create a batch.")
    p_submit.add_argument("jsonl", help="Path to the requests.jsonl file.")
    p_submit.add_argument("--endpoint", default="/v1/chat/completions")
    p_submit.add_argument("--window", default="24h")
    p_submit.set_defaults(func=_cmd_submit)

    p_status = sub.add_parser("status", parents=[common],
                              help="Retrieve the status of a batch.")
    p_status.add_argument("batch_id")
    p_status.set_defaults(func=_cmd_status)

    p_fetch = sub.add_parser("fetch", parents=[common],
                             help="Download a completed batch's output file.")
    p_fetch.add_argument("batch_id")
    p_fetch.add_argument("-o", "--out", required=True,
                         help="Destination path for output.jsonl.")
    p_fetch.add_argument("--error-out", default=None,
                         help="Optional destination for the error file.")
    p_fetch.set_defaults(func=_cmd_fetch)

    p_merge = sub.add_parser("merge",
                             help="Join output.jsonl into a pending parquet.")
    p_merge.add_argument("--pending", required=True,
                         help="Path to pending.parquet from the export stage.")
    p_merge.add_argument("--output", required=True,
                         help="Path to the downloaded output.jsonl.")
    p_merge.add_argument("--text-column", required=True,
                         help="Name of the judge text column to fill "
                              "(e.g. leak_judge_text).")
    p_merge.add_argument("--out", required=True,
                         help="Destination path for the merged results.parquet.")
    p_merge.add_argument("--custom-id-column", default="judge_custom_id")
    p_merge.set_defaults(func=_cmd_merge)

    args = parser.parse_args(list(argv) if argv is not None else None)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
