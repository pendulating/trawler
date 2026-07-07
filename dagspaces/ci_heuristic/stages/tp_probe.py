"""TP-elicitation probe stage (E2): 'this information flow is fine IF ___'.

Kumar et al.'s method for surfacing transmission principles, run as a
standalone probe: for each case's flow, elicit every condition that would
make the flow appropriate; each condition is a candidate TP. Scored later
against Tier B constructed TPs / Tier A gold TP lists.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

import pandas as pd
from omegaconf import DictConfig

from ..prompts import build_tp_elicitation_prompt
from ..schemas import TPElicitation
from .traverse import _default_run_inference, _generate_round, _parse_json_artifact

logger = logging.getLogger(__name__)


def run_tp_probe(
    df: pd.DataFrame,
    cfg: DictConfig,
    run_inference: Callable | None = None,
) -> pd.DataFrame:
    """Probe each case's flow for transmission principles.

    Uses `flow_statement` when present (Tier B carries it), falling back to
    the full practice_input. Returns df + generated_text, conditions_json,
    n_conditions, parse_status.
    """
    run_inference = run_inference or _default_run_inference

    flows = [
        str(row.get("flow_statement") or "") or str(row["practice_input"])
        for _, row in df.iterrows()
    ]
    prompts = [build_tp_elicitation_prompt(flow) for flow in flows]
    results = _generate_round(df, cfg, prompts, TPElicitation, "tp_probe", run_inference)

    out = df.copy()
    texts, conditions_json, n_conditions, statuses = [], [], [], []
    for text, _cid in results:
        artifact, status = _parse_json_artifact(text)
        conditions = artifact.get("conditions", []) if isinstance(artifact, dict) else []
        if not isinstance(conditions, list):
            conditions, status = [], "unparseable"
        texts.append(text)
        conditions_json.append(json.dumps(conditions))
        n_conditions.append(len(conditions))
        statuses.append(status)

    out["generated_text"] = texts
    out["conditions_json"] = conditions_json
    out["n_conditions"] = n_conditions
    out["parse_status"] = statuses

    parseable = sum(s != "unparseable" for s in statuses)
    logger.info(f"[tp_probe] {parseable}/{len(out)} parseable, "
                f"mean conditions={pd.Series(n_conditions).mean():.2f}")
    return out
