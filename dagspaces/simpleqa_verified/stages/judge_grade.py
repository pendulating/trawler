"""Judge-grading stages: live and async-export paths.

Two distinct stage functions; the pipeline yaml selects which one runs
based on ``judge.mode``:

- :func:`judge_grade_live` (``judge.mode=live``) — calls the live judge
  server inline via :class:`dagspaces.common.judge_client.JudgeClient`,
  appends ``judge_response`` + ``verdict`` columns, writes the parquet
  directly. ``compute_metrics`` consumes the result.

- :func:`judge_grade_batch_export` (``judge.mode=async`` or
  ``batch_export``) — writes ``requests.jsonl`` + ``items.parquet`` +
  ``pending.parquet`` + ``manifest.json``. The eval_all sidecar drains
  the manifest into ``output.jsonl``; ``finalize_async`` then joins on
  ``custom_id`` and emits metrics.

Both paths use the same grader template + JSON schema, so live ↔ async
parity is byte-identical on the judge prompt.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..prompts import (
    SIMPLEQA_GRADE_SCHEMA,
    build_grader_messages,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def parse_grade_letter(text: str) -> str:
    """Extract ``A``/``B``/``C`` from a judge response.

    Returns ``"unparseable"`` when neither structured JSON nor a
    leading single-letter answer is present. Downstream metrics treat
    unparseable rows as defaulted (not silently as not_attempted).
    """
    s = str(text or "").strip()
    if not s:
        return "unparseable"

    # Structured-output path: vLLM with guided_decoding emits
    # ``{"grade": "A"}``. Tolerate surrounding text by scanning for the
    # last JSON object.
    try:
        start = s.find("{")
        end = s.rfind("}") + 1
        if start >= 0 and end > start:
            obj = json.loads(s[start:end])
            if isinstance(obj, dict):
                g = str(obj.get("grade", "")).strip().upper()
                if g in ("A", "B", "C"):
                    return g
    except (json.JSONDecodeError, TypeError, ValueError):
        pass

    # Fallback: the SimpleQA grader prompt asks for a bare single-letter
    # response. Look for an UPPERCASE A/B/C as a standalone token (word
    # boundary). Don't case-fold first — that would match the 'a' in
    # "answer" or "incorrect" → false positive.
    m = re.search(r"\b([ABC])\b", s)
    if m:
        return m.group(1)
    return "unparseable"


def letter_to_verdict(letter: str) -> str:
    """Map A/B/C/unparseable → verdict label."""
    return {
        "A": "correct",
        "B": "incorrect",
        "C": "not_attempted",
    }.get(letter, "unparseable")


def _build_grader_item(row: Dict[str, Any]) -> Dict[str, Any]:
    """Shape used by both live judge_batch and async export_batch_jsonl."""
    return {
        "row_idx": int(row["question_id"]),
        "question": str(row["question"]),
        "gold_answer": str(row["gold_answer"]),
        "predicted_answer": str(row["generated_text"]),
    }


def _build_messages(item: Dict[str, Any]):
    return build_grader_messages(
        question=item["question"],
        gold_target=item["gold_answer"],
        predicted_answer=item["predicted_answer"],
    )


# ---------------------------------------------------------------------------
# Live judging
# ---------------------------------------------------------------------------

def _build_live_judge_client(cfg: DictConfig):
    """Build a live :class:`JudgeClient` from ``cfg.judge.*``."""
    from dagspaces.common.judge_client import JudgeClient

    judge_cfg = OmegaConf.select(cfg, "judge", default=None)
    if judge_cfg is None:
        raise RuntimeError(
            "[simpleqa_verified] judge.* section missing — set judge.base_url "
            "and judge.model_name (or use the async pipeline)."
        )

    url = str(getattr(judge_cfg, "base_url", "") or "")
    if not url:
        url = os.environ.get("JUDGE_SERVER_URL", "")
    if not url:
        raise RuntimeError(
            "[simpleqa_verified] No judge endpoint. Set judge.base_url or "
            "JUDGE_SERVER_URL. (sbatch scripts/judge_server.sub for the "
            "in-cluster vLLM judge.)"
        )

    client = JudgeClient(
        base_url=url,
        model_name=str(getattr(judge_cfg, "model_name", "default") or "default"),
        max_workers=int(getattr(judge_cfg, "max_workers", 8) or 8),
        temperature=float(getattr(judge_cfg, "temperature", 0.0) or 0.0),
        max_tokens=int(getattr(judge_cfg, "max_tokens", 256) or 256),
        provider=(getattr(judge_cfg, "provider", None) or None),
        api_key=(getattr(judge_cfg, "api_key", None) or None),
        api_key_env=(getattr(judge_cfg, "api_key_env", None) or None),
    )
    if not client.health_check():
        raise RuntimeError(f"[simpleqa_verified] judge endpoint not reachable at {url}")
    print(
        f"[simpleqa_verified] Judge OK: {url} "
        f"(provider={client.provider}, model={client.model_name})",
        flush=True,
    )
    return client


def judge_grade_live(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    """Live-judge each row; append ``judge_response``, ``grade_letter``, ``verdict``."""
    client = _build_live_judge_client(cfg)

    items = [_build_grader_item(r) for _, r in df.iterrows()]
    responses = client.judge_batch(
        items=items,
        build_messages_fn=_build_messages,
        json_schema=SIMPLEQA_GRADE_SCHEMA,
    )

    out = df.copy()
    out["judge_response"] = responses
    out["grade_letter"] = out["judge_response"].apply(parse_grade_letter)
    out["verdict"] = out["grade_letter"].apply(letter_to_verdict)
    out["parse_status"] = out["verdict"].apply(
        lambda v: "unparseable" if v == "unparseable" else "parsed"
    )

    n = len(out)
    n_unp = int((out["verdict"] == "unparseable").sum())
    print(
        f"[judge_grade_live] {n} rows judged | unparseable={n_unp} "
        f"({n_unp / max(n, 1) * 100:.1f}%) | "
        f"distribution: {out['verdict'].value_counts().to_dict()}",
        flush=True,
    )
    return out


# ---------------------------------------------------------------------------
# Async / batch-export
# ---------------------------------------------------------------------------

def _write_batch_manifest(output_dir: str, manifest: Dict[str, Any]) -> str:
    path = os.path.join(output_dir, "manifest.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, default=str)
    return path


def judge_grade_batch_export(
    df: pd.DataFrame, cfg: DictConfig, output_dir: str,
) -> pd.DataFrame:
    """Emit the judge JSONL bundle the sidecar drains.

    Writes ``requests.jsonl`` + ``items.parquet`` + ``pending.parquet`` +
    ``manifest.json`` under ``output_dir``. The sidecar discovers this
    manifest via the adjacency check in :mod:`dagspaces.eval_all.judge_sidecar`
    and writes ``output.jsonl`` + ``done.flag`` next to them; the
    ``finalize_async`` stage then joins on ``custom_id``.

    Returns ``pending.parquet`` (passed-through input + grader fields
    populated with empty placeholders so downstream consumers can ingest
    the parquet directly).
    """
    from dagspaces.common.judge_export import resolve_export_client, resolve_export_endpoint

    client, info = resolve_export_client(
        cfg, dagspace="simpleqa_verified", default_max_tokens=256,
    )

    items = [_build_grader_item(r) for _, r in df.iterrows()]

    def custom_id_fn(item: Dict[str, Any], _idx: int) -> str:
        return f"simpleqa_verified:judge_grade:{item['row_idx']}"

    os.makedirs(output_dir, exist_ok=True)
    jsonl_path = os.path.join(output_dir, "requests.jsonl")
    manifest = client.export_batch_jsonl(
        items=items,
        build_messages_fn=_build_messages,
        output_path=jsonl_path,
        custom_id_fn=custom_id_fn,
        json_schema=SIMPLEQA_GRADE_SCHEMA,
        schema_name="SimpleQAGrade",
        endpoint_url=resolve_export_endpoint(cfg),
    )

    items_rows = [
        {
            "judge_custom_id": custom_id_fn(item, i),
            "row_idx": item["row_idx"],
        }
        for i, item in enumerate(items)
    ]
    items_df = pd.DataFrame(items_rows)
    items_path = os.path.join(output_dir, "items.parquet")
    items_df.to_parquet(items_path, index=False)

    manifest.update({
        "dagspace": "simpleqa_verified",
        "stage": "judge_grade",
        "text_column": "judge_response",
        "items_parquet": items_path,
        "fanout": "per-row",
        **{k: v for k, v in info.items() if k not in manifest},
    })
    _write_batch_manifest(output_dir, manifest)

    # pending.parquet: passthrough input + empty grader columns. The
    # finalize stage fills in judge_response + verdict from output.jsonl.
    pending = df.copy()
    pending["judge_response"] = ""
    pending["grade_letter"] = ""
    pending["verdict"] = ""
    pending_path = os.path.join(output_dir, "pending.parquet")
    pending.to_parquet(pending_path, index=False)

    print(
        f"[judge_grade_batch_export] wrote {manifest['count']} requests to "
        f"{jsonl_path} (mode={info.get('mode')}, model={manifest.get('model')})",
        flush=True,
    )
    return pending
