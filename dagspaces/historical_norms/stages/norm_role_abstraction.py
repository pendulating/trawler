# Role abstraction stage: rewrite character-specific norms into functional
# social roles using a focused LLM rewrite pass.
#
# Runs after norm_extraction.  Each norm is independently rewritten so that
# norm_subject, norm_act, condition_of_application, and norm_articulation
# use generalizable social roles instead of named characters.  Metadata
# fields (normative_force, context, confidence, etc.) are preserved exactly.

import json
import pandas as pd
from omegaconf import OmegaConf
from typing import Any, Dict, List

from dagspaces.common.vllm_inference import run_vllm_inference
from ..ci_schema import RoleAbstractionResult

try:
    from json_repair import repair_json
    _JSON_REPAIR_OK = True
except ImportError:
    _JSON_REPAIR_OK = False


# Columns that carry over from extraction unchanged (not rewritten by LLM)
_PASSTHROUGH_COLUMNS = [
    "raz_normative_force", "raz_norm_source", "raz_governs_info_flow",
    "raz_info_flow_note", "raz_confidence_qual", "raz_confidence_quant",
    "raz_context",
]

# Columns that the LLM rewrites
_REWRITTEN_COLUMNS = [
    "raz_prescriptive_element", "raz_norm_subject", "raz_norm_act",
    "raz_condition_of_application", "raz_norm_articulation",
]


def _extract_json(gen_text: str):
    """Parse JSON from LLM output, with optional repair."""
    obj = None
    parse_error = None
    json_text = gen_text

    if "{" in gen_text:
        start = gen_text.find("{")
        end = gen_text.rfind("}") + 1
        if start < end:
            json_text = gen_text[start:end]

    try:
        obj = json.loads(json_text)
    except json.JSONDecodeError as e:
        parse_error = e
        if _JSON_REPAIR_OK:
            try:
                repaired = repair_json(json_text, return_objects=True)
                if isinstance(repaired, dict):
                    obj = repaired
            except Exception as repair_err:
                parse_error = f"JSON repair failed: {repair_err}"
    return obj, parse_error


def run_norm_role_abstraction_stage(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Rewrite norms to use functional social roles instead of character names.

    Args:
        df: Input DataFrame with raz_* columns from norm_extraction.
        cfg: Hydra config with prompt_role_abstraction, model, sampling_params.

    Returns:
        DataFrame with the same schema as input, but raz_norm_subject,
        raz_norm_act, raz_condition_of_application, and raz_norm_articulation
        rewritten to use social roles.  Adds role_rationale column.
        Original values preserved in orig_raz_* columns.
    """
    required = ["raz_norm_subject", "raz_norm_articulation"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[role_abstraction] Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Filter to rows that have actual norms
    pre_filter = len(df)
    mask = (
        df["raz_norm_articulation"].notna()
        & (df["raz_norm_articulation"] != "")
        & df["raz_norm_subject"].notna()
        & (df["raz_norm_subject"] != "")
    )
    valid_df = df[mask].copy()
    skip_df = df[~mask].copy()
    post_filter = len(valid_df)
    if pre_filter != post_filter:
        print(f"[role_abstraction] Filtered {pre_filter - post_filter} rows "
              f"with empty/null norm fields ({post_filter} remaining)")
    if post_filter == 0:
        print("[role_abstraction] No valid norms to abstract, returning as-is")
        df["role_rationale"] = None
        df["role_abstraction_failed"] = None
        return df

    # Load prompt config
    prompt_cfg = (
        OmegaConf.select(cfg, "prompt_role_abstraction")
        or OmegaConf.select(cfg, "prompt")
    )
    if prompt_cfg is None:
        raise RuntimeError(
            "[role_abstraction] No prompt config found at "
            "'prompt_role_abstraction' or 'prompt'."
        )

    system_prompt = str(OmegaConf.select(prompt_cfg, "system_prompt"))
    prompt_template = str(OmegaConf.select(prompt_cfg, "prompt_template"))
    print(f"[role_abstraction] Loaded prompt "
          f"(system: {len(system_prompt)} chars, template: {len(prompt_template)} chars)")

    # Sampling params with guided decoding
    sampling_params = dict(
        OmegaConf.to_container(
            OmegaConf.select(cfg, "sampling_params"),
            resolve=True,
        ) or {}
    )
    sampling_params.setdefault("temperature", 0.0)
    sampling_params.setdefault("max_tokens", 1024)
    json_schema = RoleAbstractionResult.model_json_schema()
    sampling_params["guided_decoding"] = {"json": json_schema}

    def _format_prompt(row: Dict[str, Any]) -> str:
        return (
            prompt_template
            .replace("{{prescriptive_element}}", str(row.get("raz_prescriptive_element") or ""))
            .replace("{{norm_subject}}", str(row.get("raz_norm_subject") or ""))
            .replace("{{norm_act}}", str(row.get("raz_norm_act") or ""))
            .replace("{{condition_of_application}}", str(row.get("raz_condition_of_application") or "null"))
            .replace("{{normative_force}}", str(row.get("raz_normative_force") or ""))
            .replace("{{norm_articulation}}", str(row.get("raz_norm_articulation") or ""))
            .replace("{{context}}", str(row.get("raz_context") or ""))
        )

    def _preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        result_row = dict(row)
        user_prompt = _format_prompt(result_row)
        result_row["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result_row["sampling_params"] = sampling_params
        return result_row

    def _postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        result_row = dict(row)
        result_row.pop("messages", None)
        result_row.pop("sampling_params", None)
        result_row.pop("usage", None)

        gen_text = result_row.get("generated_text", "{}")
        obj, parse_error = _extract_json(gen_text)

        if obj is not None:
            norm = obj.get("norm", obj)

            # Preserve originals before overwriting
            for col in _REWRITTEN_COLUMNS:
                result_row[f"orig_{col}"] = result_row.get(col)

            # Overwrite with rewritten values
            result_row["raz_prescriptive_element"] = (
                norm.get("prescriptive_element")
                or result_row.get("orig_raz_prescriptive_element")
            )
            result_row["raz_norm_subject"] = (
                norm.get("norm_subject")
                or result_row.get("orig_raz_norm_subject")
            )
            result_row["raz_norm_act"] = (
                norm.get("norm_act")
                or result_row.get("orig_raz_norm_act")
            )
            result_row["raz_condition_of_application"] = (
                norm.get("condition_of_application")
                # Preserve None if the LLM outputs null for unconditional norms
            )
            result_row["raz_norm_articulation"] = (
                norm.get("norm_articulation")
                or result_row.get("orig_raz_norm_articulation")
            )

            # Metadata fields: prefer original values (LLM should pass through,
            # but we trust the extraction stage's values over the rewrite)
            for col in _PASSTHROUGH_COLUMNS:
                field_name = col.replace("raz_", "")
                llm_val = norm.get(field_name)
                # Only use LLM value if original is missing
                if result_row.get(col) is None and llm_val is not None:
                    result_row[col] = llm_val

            result_row["role_rationale"] = norm.get("role_rationale", "")
            result_row["role_abstraction_failed"] = False
        else:
            print(f"[role_abstraction] JSON parse error: {parse_error}")
            result_row["role_rationale"] = None
            result_row["role_abstraction_failed"] = True
            # Preserve original columns as-is on failure
            for col in _REWRITTEN_COLUMNS:
                result_row[f"orig_{col}"] = result_row.get(col)

        return result_row

    result_df = run_vllm_inference(
        df=valid_df,
        cfg=cfg,
        preprocess=_preprocess,
        postprocess=_postprocess,
        stage_name="role_abstraction",
    )

    print(f"[role_abstraction] Completed inference, {len(result_df)} results")

    # Report success/failure stats
    if "role_abstraction_failed" in result_df.columns:
        n_failed = int(result_df["role_abstraction_failed"].sum())
        n_ok = len(result_df) - n_failed
        print(f"[role_abstraction] Success: {n_ok}/{len(result_df)}, "
              f"failed: {n_failed}")

    # Re-attach skipped rows (those with empty norms that weren't processed)
    if len(skip_df) > 0:
        skip_df["role_rationale"] = None
        skip_df["role_abstraction_failed"] = None
        for col in _REWRITTEN_COLUMNS:
            skip_df[f"orig_{col}"] = skip_df.get(col)
        result_df = pd.concat([result_df, skip_df], ignore_index=True)
        print(f"[role_abstraction] Re-attached {len(skip_df)} skipped rows, "
              f"total: {len(result_df)}")

    return result_df
