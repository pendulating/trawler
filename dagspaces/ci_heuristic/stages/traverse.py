"""Heuristic traversal stage: run cases through a ladder level (L0-L4).

The chain (L2+) runs as 9 sequential *batched* rounds: every case's step-k
prompt is built from that case's accumulated state, all cases generate step k
together, artifacts are parsed and threaded into state, then step k+1. This
preserves vLLM batching within a step while keeping the chain dependency
across steps.

NOTE on engine lifecycle: each round calls dagspaces.common.vllm_inference.
run_vllm_inference, which in in-process mode re-initializes the engine per
round (9 loads per pipeline). Prefer server mode (model.server_url pointing
at a long-lived vLLM OpenAI server) for chain runs at scale; in-process is
fine for debug/small models.

Parse failures degrade, not abort: the failed step's artifact becomes
{"parse_error": ...}, downstream steps still run (scorers filter on
parse_status), so one bad step doesn't void a whole case.

`run_inference` is injectable for tests (signature mirrors the common
run_vllm_inference).
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional, Tuple

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..prompts import (
    build_l0_prompt,
    build_l1_prompt,
    build_step_prompt,
    render_exemplar,
)
from ..schemas import STEP_ORDER, STEP_SCHEMAS, L0Verdict, L1Traversal

logger = logging.getLogger(__name__)

LADDER_LEVELS = ("l0", "l1", "l2", "l3", "l4")


def _parse_json_artifact(text: str) -> Tuple[Dict[str, Any], str]:
    """Extract and parse the JSON object from generated text.

    Returns (artifact, parse_status) where parse_status is
    'parsed' | 'recovered' (needed brace-slicing) | 'unparseable'.
    """
    if not text or not text.strip():
        return {"parse_error": "empty"}, "unparseable"
    raw = text.strip()
    try:
        return json.loads(raw), "parsed"
    except json.JSONDecodeError:
        pass
    start, end = raw.find("{"), raw.rfind("}") + 1
    if 0 <= start < end:
        try:
            return json.loads(raw[start:end]), "recovered"
        except json.JSONDecodeError:
            pass
    return {"parse_error": raw[:500]}, "unparseable"


def _default_run_inference(df: pd.DataFrame, cfg: Any, preprocess, postprocess, stage_name: str) -> pd.DataFrame:
    from dagspaces.common.vllm_inference import run_vllm_inference

    return run_vllm_inference(df, cfg, preprocess=preprocess, postprocess=postprocess, stage_name=stage_name)


def _generate_round(
    cases: pd.DataFrame,
    cfg: Any,
    sys_usr: List[Tuple[str, str]],
    schema_model,
    stage_name: str,
    run_inference: Callable,
) -> List[Tuple[str, str]]:
    """One batched generation round. Returns [(generated_text, case_id), ...] aligned to cases order."""
    round_df = cases[["case_id"]].copy()
    round_df["_sys"] = [p[0] for p in sys_usr]
    round_df["_usr"] = [p[1] for p in sys_usr]

    sp_node = getattr(cfg, "sampling_params", None)
    if OmegaConf.is_config(sp_node):
        sp_base = OmegaConf.to_container(sp_node, resolve=True)
    else:
        sp_base = dict(sp_node or {})
    guided = {"guided_decoding": {"json": schema_model.model_json_schema()}}

    def preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        row["messages"] = [
            {"role": "system", "content": row["_sys"]},
            {"role": "user", "content": row["_usr"]},
        ]
        row["sampling_params"] = {**sp_base, **guided}
        return row

    def postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        return row

    out = run_inference(round_df, cfg, preprocess=preprocess, postprocess=postprocess, stage_name=stage_name)
    by_id = dict(zip(out["case_id"], out["generated_text"]))
    return [(str(by_id.get(cid, "")), cid) for cid in cases["case_id"]]


def _load_exemplar(cfg: Any) -> Optional[str]:
    path = str(OmegaConf.select(cfg, "ladder.exemplar_path") or "")
    if not path:
        raise ValueError("ladder level l4 requires ladder.exemplar_path (a contaminated Tier A gold file)")
    gold = json.load(open(path))
    return render_exemplar(gold)


def run_traversal(
    df: pd.DataFrame,
    cfg: DictConfig,
    run_inference: Callable | None = None,
) -> pd.DataFrame:
    """Traverse all cases at the configured ladder level.

    Returns a long-format DataFrame: one row per case x step (chain levels)
    or per case (monolithic levels), with columns case_id, tier, ladder_level,
    step, prompt_sys, prompt_usr, generated_text, artifact_json, parse_status.
    """
    run_inference = run_inference or _default_run_inference
    level = str(OmegaConf.select(cfg, "ladder.level") or "")
    if level not in LADDER_LEVELS:
        raise ValueError(f"ladder.level must be one of {LADDER_LEVELS}, got {level!r}")

    tier_by_id = dict(zip(df["case_id"], df["tier"]))
    records: List[Dict[str, Any]] = []

    if level in ("l0", "l1"):
        builder = build_l0_prompt if level == "l0" else build_l1_prompt
        schema = L0Verdict if level == "l0" else L1Traversal
        prompts = [builder(p) for p in df["practice_input"]]
        results = _generate_round(df, cfg, prompts, schema, f"traverse_{level}", run_inference)
        for (text, cid), (sys_p, usr_p) in zip(results, prompts):
            artifact, status = _parse_json_artifact(text)
            records.append({
                "case_id": cid, "tier": tier_by_id[cid], "ladder_level": level,
                "step": "monolithic", "prompt_sys": sys_p, "prompt_usr": usr_p,
                "generated_text": text, "artifact_json": json.dumps(artifact),
                "parse_status": status,
            })
        return pd.DataFrame(records)

    # Chain levels: l2 (bare), l3 (+guiding questions), l4 (+exemplar)
    include_gq = level in ("l3", "l4")
    exemplar = _load_exemplar(cfg) if level == "l4" else None

    state: Dict[str, Dict[str, Any]] = {cid: {} for cid in df["case_id"]}

    for step in STEP_ORDER:
        prompts = [
            build_step_prompt(
                practice_input=row.practice_input,
                step=step,
                state=state[row.case_id],
                include_guiding_questions=include_gq,
                exemplar=exemplar,
            )
            for row in df.itertuples()
        ]
        results = _generate_round(
            df, cfg, prompts, STEP_SCHEMAS[step], f"traverse_{level}_{step}", run_inference
        )
        n_ok = 0
        for (text, cid), (sys_p, usr_p) in zip(results, prompts):
            artifact, status = _parse_json_artifact(text)
            state[cid][step] = artifact
            n_ok += status != "unparseable"
            records.append({
                "case_id": cid, "tier": tier_by_id[cid], "ladder_level": level,
                "step": step, "prompt_sys": sys_p, "prompt_usr": usr_p,
                "generated_text": text, "artifact_json": json.dumps(artifact),
                "parse_status": status,
            })
        logger.info(f"[traverse:{level}] {step}: {n_ok}/{len(df)} parseable")

    return pd.DataFrame(records)
