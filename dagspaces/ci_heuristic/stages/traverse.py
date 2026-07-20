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
from collections.abc import Callable
from typing import Any

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

LADDER_LEVELS = ("l0", "l1", "l2", "l3", "l4", "l5")


def _parse_json_artifact(text: str) -> tuple[dict[str, Any], str]:
    """Extract and parse the JSON object from generated text.

    Returns (artifact, parse_status) where parse_status is
    'parsed' | 'recovered' (needed brace-slicing) | 'unparseable'.
    """
    from dagspaces.common.json_extraction import extract_json_from_text

    if not text or not text.strip():
        return {"parse_error": "empty"}, "unparseable"
    raw = text.strip()
    # Fast path: clean JSON
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj, "parsed"
    except (json.JSONDecodeError, ValueError):
        pass
    # Delegate to canonical extractor (outermost span)
    obj, _ = extract_json_from_text(raw)
    if obj is not None:
        return obj, "recovered"
    return {"parse_error": raw[:500]}, "unparseable"


def _default_run_inference(
    df: pd.DataFrame, cfg: Any, preprocess, postprocess, stage_name: str
) -> pd.DataFrame:
    from dagspaces.common.vllm_inference import run_vllm_inference

    return run_vllm_inference(
        df, cfg, preprocess=preprocess, postprocess=postprocess, stage_name=stage_name
    )


def _generate_round(
    cases: pd.DataFrame,
    cfg: Any,
    sys_usr: list[tuple[str, str]],
    schema_model,
    stage_name: str,
    run_inference: Callable,
) -> list[tuple[str, str]]:
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

    def preprocess(row: dict[str, Any]) -> dict[str, Any]:
        row["messages"] = [
            {"role": "system", "content": row["_sys"]},
            {"role": "user", "content": row["_usr"]},
        ]
        row["sampling_params"] = {**sp_base, **guided}
        return row

    def postprocess(row: dict[str, Any]) -> dict[str, Any]:
        return row

    out = run_inference(
        round_df,
        cfg,
        preprocess=preprocess,
        postprocess=postprocess,
        stage_name=stage_name,
    )
    by_id = dict(zip(out["case_id"], out["generated_text"]))
    return [(str(by_id.get(cid, "")), cid) for cid in cases["case_id"]]


def _load_exemplar(cfg: Any) -> str | None:
    path = str(OmegaConf.select(cfg, "ladder.exemplar_path") or "")
    if not path:
        raise ValueError(
            "ladder level l4 requires ladder.exemplar_path (a contaminated Tier A gold file)"
        )
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
    records: list[dict[str, Any]] = []

    if level in ("l0", "l1"):
        builder = build_l0_prompt if level == "l0" else build_l1_prompt
        schema = L0Verdict if level == "l0" else L1Traversal
        prompts = [builder(p) for p in df["practice_input"]]
        results = _generate_round(
            df, cfg, prompts, schema, f"traverse_{level}", run_inference
        )
        for (text, cid), (sys_p, usr_p) in zip(results, prompts):
            artifact, status = _parse_json_artifact(text)
            records.append(
                {
                    "case_id": cid,
                    "tier": tier_by_id[cid],
                    "ladder_level": level,
                    "step": "monolithic",
                    "prompt_sys": sys_p,
                    "prompt_usr": usr_p,
                    "generated_text": text,
                    "artifact_json": json.dumps(artifact),
                    "parse_status": status,
                }
            )
        return pd.DataFrame(records)

    # Chain levels: l2 (bare), l3 (+guiding questions), l4 (+exemplar),
    # l5 (l3 + deliberative structures at steps 5/7/8/9)
    include_gq = level in ("l3", "l4", "l5")
    exemplar = _load_exemplar(cfg) if level == "l4" else None
    deliberative = level == "l5"

    state: dict[str, dict[str, Any]] = {cid: {} for cid in df["case_id"]}

    def _record(
        step: str,
        cid: str,
        sys_p: str,
        usr_p: str,
        text: str,
        artifact: dict[str, Any],
        status: str,
    ) -> None:
        records.append(
            {
                "case_id": cid,
                "tier": tier_by_id[cid],
                "ladder_level": level,
                "step": step,
                "prompt_sys": sys_p,
                "prompt_usr": usr_p,
                "generated_text": text,
                "artifact_json": json.dumps(artifact),
                "parse_status": status,
            }
        )

    for step in STEP_ORDER:
        if deliberative and step in ("s5", "s7", "s8", "s9"):
            _run_deliberative_step(df, cfg, step, state, _record, run_inference, level)
            continue
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
            df,
            cfg,
            prompts,
            STEP_SCHEMAS[step],
            f"traverse_{level}_{step}",
            run_inference,
        )
        n_ok = 0
        for (text, cid), (sys_p, usr_p) in zip(results, prompts):
            artifact, status = _parse_json_artifact(text)
            state[cid][step] = artifact
            n_ok += status != "unparseable"
            _record(step, cid, sys_p, usr_p, text, artifact, status)
        logger.info(f"[traverse:{level}] {step}: {n_ok}/{len(df)} parseable")

    return pd.DataFrame(records)


def _member_round(
    df: pd.DataFrame,
    cfg: Any,
    prompts: list[tuple[str, str]],
    member_ids: list[str],
    schema_model,
    stage_name: str,
    run_inference: Callable,
) -> list[tuple[str, str, str]]:
    """One batched round over cases x members. Returns (text, case_id, member_id)
    triples. Rows are given composite ids so _generate_round can align them."""
    round_df = pd.DataFrame(
        {
            "case_id": [
                f"{cid}::{mid}" for cid, mid in zip(df["case_id_expanded"], member_ids)
            ],
        }
    )
    results = _generate_round(
        round_df, cfg, prompts, schema_model, stage_name, run_inference
    )
    out = []
    for text, composite in results:
        cid, mid = composite.rsplit("::", 1)
        out.append((text, cid, mid))
    return out


def _run_deliberative_step(
    df: pd.DataFrame,
    cfg: Any,
    step: str,
    state: dict[str, dict[str, Any]],
    _record,
    run_inference: Callable,
    level: str,
) -> None:
    """L5 handling for steps 5/7/8/9 (see deliberation.py for the design)."""
    from omegaconf import OmegaConf as _OC

    from ..deliberation import (
        NORM_POPULATION,
        NormExpectation,
        aggregate_expectations,
        build_moderator_prompt,
        build_norm_elicitation_prompts,
        build_norm_synthesis_prompt,
        build_s8_analyst_prompt,
        build_stakeholder_prompt,
        merge_factor_artifacts,
        stakeholder_set,
    )

    n_personas = int(_OC.select(cfg, "ladder.s5_n_personas") or len(NORM_POPULATION))
    s7_structure = str(_OC.select(cfg, "ladder.s7_structure") or "ensemble")
    include_marginalized = bool(
        _OC.select(cfg, "ladder.include_marginalized")
        if _OC.select(cfg, "ladder.include_marginalized") is not None
        else True
    )

    if step == "s5":
        personas = NORM_POPULATION[:n_personas]
        # Round A: elicit expectations, cases x personas in one batch
        flat_prompts, flat_cids, flat_mids = [], [], []
        for row in df.itertuples():
            for p, prompt in zip(
                personas,
                build_norm_elicitation_prompts(
                    row.practice_input, state[row.case_id], personas
                ),
            ):
                flat_prompts.append(prompt)
                flat_cids.append(row.case_id)
                flat_mids.append(p.id)
        exp_df = pd.DataFrame({"case_id_expanded": flat_cids})
        results = _member_round(
            exp_df,
            cfg,
            flat_prompts,
            flat_mids,
            NormExpectation,
            f"traverse_{level}_s5_elicit",
            run_inference,
        )
        by_case: dict[str, list[dict[str, Any]]] = {cid: [] for cid in df["case_id"]}
        for (text, cid, mid), (sys_p, usr_p) in zip(results, flat_prompts):
            artifact, status = _parse_json_artifact(text)
            by_case[cid].append(artifact)
            _record(f"s5:elicit:{mid}", cid, sys_p, usr_p, text, artifact, status)
        # Round B: synthesize S5 artifact per case, stats injected
        synth_prompts = []
        stats_by_case = {}
        for row in df.itertuples():
            stats = aggregate_expectations(by_case[row.case_id])
            stats_by_case[row.case_id] = stats
            synth_prompts.append(
                build_norm_synthesis_prompt(
                    row.practice_input, state[row.case_id], stats
                )
            )
        results = _generate_round(
            df,
            cfg,
            synth_prompts,
            STEP_SCHEMAS["s5"],
            f"traverse_{level}_s5_synth",
            run_inference,
        )
        for (text, cid), (sys_p, usr_p) in zip(results, synth_prompts):
            artifact, status = _parse_json_artifact(text)
            artifact["_population_stats"] = {
                k: v for k, v in stats_by_case[cid].items() if k != "expectations"
            }
            state[cid]["s5"] = artifact
            _record("s5", cid, sys_p, usr_p, text, artifact, status)
        return

    if step == "s7":
        panel = stakeholder_set(include_marginalized)
        member_artifacts: dict[str, list[tuple[str, dict[str, Any]]]] = {
            cid: [] for cid in df["case_id"]
        }
        prior_texts: dict[str, list[str]] = {cid: [] for cid in df["case_id"]}

        if s7_structure in ("ensemble", "chain"):
            # ensemble: one batch, no shared info; chain: sequential members,
            # each batched across cases, seeing this case's prior members.
            member_seq = [panel] if s7_structure == "chain" else [panel]
            if s7_structure == "ensemble":
                flat_prompts, flat_cids, flat_mids = [], [], []
                for row in df.itertuples():
                    for p in panel:
                        flat_prompts.append(
                            build_stakeholder_prompt(
                                row.practice_input, state[row.case_id], p
                            )
                        )
                        flat_cids.append(row.case_id)
                        flat_mids.append(p.id)
                exp_df = pd.DataFrame({"case_id_expanded": flat_cids})
                results = _member_round(
                    exp_df,
                    cfg,
                    flat_prompts,
                    flat_mids,
                    STEP_SCHEMAS["s7"],
                    f"traverse_{level}_s7_ensemble",
                    run_inference,
                )
                for (text, cid, mid), (sys_p, usr_p) in zip(results, flat_prompts):
                    artifact, status = _parse_json_artifact(text)
                    member_artifacts[cid].append((mid, artifact))
                    _record(
                        f"s7:member:{mid}", cid, sys_p, usr_p, text, artifact, status
                    )
            else:  # chain
                for p in panel:
                    prompts = [
                        build_stakeholder_prompt(
                            row.practice_input,
                            state[row.case_id],
                            p,
                            prior_responses=prior_texts[row.case_id] or None,
                        )
                        for row in df.itertuples()
                    ]
                    results = _generate_round(
                        df,
                        cfg,
                        prompts,
                        STEP_SCHEMAS["s7"],
                        f"traverse_{level}_s7_chain_{p.id}",
                        run_inference,
                    )
                    for (text, cid), (sys_p, usr_p) in zip(results, prompts):
                        artifact, status = _parse_json_artifact(text)
                        member_artifacts[cid].append((p.id, artifact))
                        prior_texts[cid].append(text)
                        _record(
                            f"s7:member:{p.id}",
                            cid,
                            sys_p,
                            usr_p,
                            text,
                            artifact,
                            status,
                        )
        elif s7_structure == "debate":
            from ..deliberation import DEBATE_INSTRUCTIONS
            from ..deliberation import Persona as _Persona

            debaters = [
                _Persona(
                    "defender",
                    "an advocate who believes the practice is on balance defensible",
                ),
                _Persona(
                    "critic",
                    "an advocate who believes the practice is on balance harmful",
                ),
            ]
            cycles = 2
            for cycle in range(cycles):
                for d in debaters:
                    prompts = []
                    for row in df.itertuples():
                        prior = prior_texts[row.case_id]
                        combo = DEBATE_INSTRUCTIONS[d.id] if prior else "{prior}"
                        prompts.append(
                            build_stakeholder_prompt(
                                row.practice_input,
                                state[row.case_id],
                                d,
                                prior_responses=prior[-1:] or None,
                                combination_template=combo,
                            )
                        )
                    results = _generate_round(
                        df,
                        cfg,
                        prompts,
                        STEP_SCHEMAS["s7"],
                        f"traverse_{level}_s7_debate{cycle}_{d.id}",
                        run_inference,
                    )
                    for (text, cid), (sys_p, usr_p) in zip(results, prompts):
                        artifact, status = _parse_json_artifact(text)
                        prior_texts[cid].append(text)
                        if cycle == cycles - 1:  # only final positions merge
                            member_artifacts[cid].append((d.id, artifact))
                        _record(
                            f"s7:member:{d.id}:c{cycle}",
                            cid,
                            sys_p,
                            usr_p,
                            text,
                            artifact,
                            status,
                        )
        else:
            raise ValueError(f"Unknown s7_structure {s7_structure!r}")

        for cid in df["case_id"]:
            merged = merge_factor_artifacts(member_artifacts[cid])
            state[cid]["s7"] = merged
            _record("s7", cid, "", "(merged from members)", "", merged, "parsed")
        return

    # s8 analyst / s9 moderator: single dedicated-prompt rounds
    builder = build_s8_analyst_prompt if step == "s8" else build_moderator_prompt
    prompts = [
        builder(row.practice_input, state[row.case_id]) for row in df.itertuples()
    ]
    results = _generate_round(
        df,
        cfg,
        prompts,
        STEP_SCHEMAS[step],
        f"traverse_{level}_{step}_deliberative",
        run_inference,
    )
    for (text, cid), (sys_p, usr_p) in zip(results, prompts):
        artifact, status = _parse_json_artifact(text)
        state[cid][step] = artifact
        _record(step, cid, sys_p, usr_p, text, artifact, status)
