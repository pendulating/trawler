# Norm consolidation from pre-computed cluster labels (Phase B only).
#
# Accepts a DataFrame with raz_* columns and a pre-computed cluster
# label column (e.g. from Qwen3-Embedding-8B + HDBSCAN), then runs
# only the LLM merge step to produce canonical norms.
#
# This replaces the embed→cluster→merge pipeline when higher-quality
# embeddings are computed offline (e.g. via scripts/recluster_norms_qwen3emb.py).

import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List

from omegaconf import OmegaConf
from dagspaces.common.vllm_inference import run_vllm_inference

from ._utils import extract_json
from .norm_consolidation import (
    _CONSOLIDATED_COLUMNS,
    _build_cluster_df,
    _singleton_result,
    _merged_result,
    _fallback_result,
)


def run_norm_consolidation_from_clusters_stage(
    df: pd.DataFrame, cfg: Any
) -> pd.DataFrame:
    """Consolidate norms using pre-computed cluster labels + LLM merging.

    The input DataFrame must have:
      - raz_* columns (from norm_extraction)
      - a cluster label column (default: ``qwen3emb_cluster``, configurable
        via ``consolidation.cluster_column``). Label -1 = singleton.
      - ``gutenberg_id`` (for grouping / provenance)

    Args:
        df: Input DataFrame with raz_* columns and cluster labels.
        cfg: Hydra config with prompt_consolidation, model, sampling_params.

    Returns:
        DataFrame with one row per consolidated canonical norm.
    """
    # Resolve cluster column name
    consolidation_cfg = OmegaConf.select(cfg, "consolidation") or {}
    cluster_col = str(
        OmegaConf.select(consolidation_cfg, "cluster_column")
        or "qwen3emb_cluster"
    )
    group_by_col = str(
        OmegaConf.select(consolidation_cfg, "group_by") or "gutenberg_id"
    )

    for required in ["raz_norm_articulation", cluster_col, group_by_col]:
        if required not in df.columns:
            raise ValueError(
                f"[norm_consolidation_from_clusters] Missing column '{required}'. "
                f"Available: {list(df.columns)}"
            )

    # Filter rows with valid norm articulations
    pre_filter = len(df)
    mask = df["raz_norm_articulation"].notna() & (df["raz_norm_articulation"] != "")
    df = df[mask].reset_index(drop=True)
    post_filter = len(df)
    if pre_filter != post_filter:
        print(f"[norm_consol_clusters] Filtered {pre_filter - post_filter} rows "
              f"with empty articulations ({post_filter} remaining)")
    if post_filter == 0:
        return pd.DataFrame(columns=_CONSOLIDATED_COLUMNS)

    df["_source_norm_id"] = range(len(df))

    # Use pre-computed labels directly
    labels = df[cluster_col].values.astype(int)
    groups = df[group_by_col].fillna("unknown")

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_singletons = int((labels == -1).sum())
    print(f"[norm_consol_clusters] Using pre-computed clusters from '{cluster_col}': "
          f"{n_clusters} clusters, {n_singletons} singletons")

    # Build cluster DataFrame
    df["_group_key"] = groups.values
    cluster_df = _build_cluster_df(df, labels)
    cluster_df["group_key"] = cluster_df["member_indices"].apply(
        lambda idxs: str(df.iloc[idxs[0]]["_group_key"]) if idxs else "unknown"
    )

    singletons = cluster_df[cluster_df["cluster_size"] == 1].copy()
    multi = cluster_df[cluster_df["cluster_size"] > 1].copy()
    print(f"[norm_consol_clusters] {len(singletons)} singletons (pass-through), "
          f"{len(multi)} multi-norm clusters (LLM merge)")

    results: List[Dict[str, Any]] = []

    # Pass-through singletons
    for _, crow in singletons.iterrows():
        idx = crow["member_indices"][0]
        orig = df.iloc[idx]
        results.append(_singleton_result(crow, orig, crow["group_key"]))

    # LLM merge for multi-norm clusters
    if len(multi) > 0:
        prompt_cfg = (
            OmegaConf.select(cfg, "prompt_consolidation")
            or OmegaConf.select(cfg, "prompt")
        )
        if prompt_cfg is None:
            raise RuntimeError(
                "[norm_consol_clusters] No prompt config found at "
                "'prompt_consolidation' or 'prompt'."
            )

        system_prompt = str(OmegaConf.select(prompt_cfg, "system_prompt"))
        prompt_template = str(OmegaConf.select(prompt_cfg, "prompt_template"))

        sampling_params = dict(
            OmegaConf.to_container(
                OmegaConf.select(cfg, "sampling_params"), resolve=True,
            ) or {}
        )
        sampling_params.setdefault("temperature", 0.0)
        sampling_params.setdefault("max_tokens", 2048)

        from ..ci_schema import ConsolidatedNormResult
        json_schema = ConsolidatedNormResult.model_json_schema()
        sampling_params["guided_decoding"] = {"json": json_schema}

        _MAX_MEMBERS_PER_PROMPT = 25

        def _preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
            result_row = dict(row)
            member_norms_json = row["member_norms_json"]
            cluster_size = row["cluster_size"]

            if cluster_size > _MAX_MEMBERS_PER_PROMPT:
                members = json.loads(member_norms_json)
                members.sort(
                    key=lambda m: m.get("confidence_quant") or 0,
                    reverse=True,
                )
                members = members[:_MAX_MEMBERS_PER_PROMPT]
                member_norms_json = json.dumps(members, ensure_ascii=False)
                print(f"[norm_consol_clusters] Truncated cluster "
                      f"({cluster_size} -> {_MAX_MEMBERS_PER_PROMPT} members)")

            user_prompt = prompt_template.replace(
                "{{member_norms}}", member_norms_json
            ).replace(
                "{{cluster_size}}", str(min(cluster_size, _MAX_MEMBERS_PER_PROMPT))
            )
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
            obj, parse_error = extract_json(gen_text)
            if obj is not None:
                result_row["merged_norm"] = obj
                result_row["merge_failed"] = False
            else:
                print(f"[norm_consol_clusters] JSON parse error: {parse_error}")
                result_row["merged_norm"] = None
                result_row["merge_failed"] = True
                result_row["merge_error"] = str(parse_error)
            return result_row

        merged_df = run_vllm_inference(
            df=multi,
            cfg=cfg,
            preprocess=_preprocess,
            postprocess=_postprocess,
            stage_name="norm_consol_clusters",
        )

        for _, mrow in merged_df.iterrows():
            merged = mrow.get("merged_norm")
            group_key = mrow.get("group_key", "unknown")
            member_idx = mrow.get("member_indices", [])

            if isinstance(merged, dict):
                results.append(_merged_result(mrow, merged, group_key))
            else:
                print(f"[norm_consol_clusters] Cluster {mrow['cluster_id']}: "
                      f"merge failed, using highest-confidence member")
                best_idx = member_idx[0] if member_idx else 0
                if member_idx:
                    confidences = [
                        df.iloc[i].get("raz_confidence_quant", 0) or 0
                        for i in member_idx
                    ]
                    best_idx = member_idx[int(np.argmax(confidences))]
                orig = df.iloc[best_idx]
                results.append(_fallback_result(mrow, orig, group_key))

    out_df = pd.DataFrame(results)

    # Summary per group
    if len(out_df) > 0 and "group_key" in out_df.columns:
        for gk, gdf in out_df.groupby("group_key"):
            n_merged = int((gdf["cluster_size"] > 1).sum())
            n_single = int((gdf["cluster_size"] == 1).sum())
            n_input = int((groups == gk).sum())
            print(f"[norm_consol_clusters] '{gk}': {n_input} norms -> "
                  f"{len(gdf)} canonical ({n_merged} merged, {n_single} singletons)")

    print(f"[norm_consol_clusters] Total: {post_filter} norms -> "
          f"{len(out_df)} canonical norms")
    return out_df
