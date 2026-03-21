# Norm consolidation stage: embed -> cluster -> LLM merge.
#
# Phase A: Embed whole-norm articulations with sentence-transformers,
#          cluster with HDBSCAN on cosine distance.
# Phase B: For each cluster, prompt vLLM to produce a single canonical
#          merged norm with abstracted components + provenance list.
#
# Clustering is partitioned by:
#   1. group_by column (default: gutenberg_id) — norms from different
#      source texts are never merged
#   2. normative_force — norms with different deontic operators are
#      definitionally distinct and must never be merged

import json
import pandas as pd
import numpy as np
from typing import Any, Dict, List, Optional
from omegaconf import OmegaConf

from dagspaces.common.vllm_inference import run_vllm_inference


# ---------------------------------------------------------------------------
# Column definitions for consolidated output
# ---------------------------------------------------------------------------

_CONSOLIDATED_COLUMNS = [
    "group_key",
    "cluster_id",
    "cluster_size",
    "canonical_prescriptive_element",
    "canonical_norm_subject",
    "canonical_norm_act",
    "canonical_condition_of_application",
    "canonical_normative_force",
    "canonical_norm_articulation",
    "canonical_context",
    "canonical_governs_info_flow",
    "canonical_info_flow_note",
    "consolidation_rationale",
    "abstraction_map",
    "source_norm_ids",
    "source_norm_articulations",
    "mean_confidence",
]


# ---------------------------------------------------------------------------
# Phase A: Embedding + Clustering
# ---------------------------------------------------------------------------

def _build_norm_text(row: Dict[str, Any]) -> str:
    """Build a text representation of a norm for embedding.

    Concatenates the Raz tuple components and articulation into a single
    string that captures the full semantic content of the norm.
    """
    parts = []
    subj = row.get("raz_norm_subject") or ""
    pe = row.get("raz_prescriptive_element") or ""
    act = row.get("raz_norm_act") or ""
    cond = row.get("raz_condition_of_application") or ""
    art = row.get("raz_norm_articulation") or ""

    # Primary: the natural-language articulation
    if art:
        parts.append(art)

    # Secondary: structured tuple as reinforcement
    tuple_str = f"{subj} {pe} {act}"
    if cond:
        tuple_str += f" when {cond}"
    parts.append(tuple_str.strip())

    ctx = row.get("raz_context") or ""
    if ctx:
        parts.append(f"[context: {ctx}]")

    force = row.get("raz_normative_force") or ""
    if force:
        parts.append(f"[force: {force}]")

    return " | ".join(parts)


def _embed_norms(texts: List[str], model_name: str = "all-MiniLM-L6-v2",
                 batch_size: int = 256) -> np.ndarray:
    """Embed norm texts using sentence-transformers.

    Returns an (N, D) numpy array of L2-normalised embeddings.
    """
    from sentence_transformers import SentenceTransformer

    print(f"[norm_consolidation] Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"[norm_consolidation] Embedding {len(texts)} norms "
          f"(batch_size={batch_size})...")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    return np.asarray(embeddings)


def _cluster_embeddings(
    embeddings: np.ndarray,
    min_cluster_size: int = 3,
    min_samples: int = 2,
    cluster_selection_epsilon: float = 0.0,
) -> np.ndarray:
    """Cluster embeddings with HDBSCAN on cosine distance.

    Returns an integer label array (length N). Label -1 = singleton/noise.
    """
    import hdbscan

    print(f"[norm_consolidation] Clustering {len(embeddings)} embeddings "
          f"(min_cluster_size={min_cluster_size}, "
          f"min_samples={min_samples})...")

    # HDBSCAN with euclidean on L2-normed vectors (equivalent to cosine).
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
        cluster_selection_epsilon=cluster_selection_epsilon,
        cluster_selection_method="eom",
    )
    labels = clusterer.fit_predict(embeddings)

    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise = int((labels == -1).sum())
    print(f"[norm_consolidation] Found {n_clusters} clusters, "
          f"{n_noise} singletons/noise")
    return labels


def _cluster_within_deontic_partitions(
    df: pd.DataFrame,
    embeddings: np.ndarray,
    min_cluster_size: int,
    min_samples: int,
    cluster_selection_epsilon: float,
    global_cluster_offset: int = 0,
) -> tuple[np.ndarray, int]:
    """Cluster embeddings, partitioned by normative_force.

    Returns (labels array, next available cluster offset).
    """
    labels = np.full(len(df), -1, dtype=int)
    force_col = df["raz_normative_force"].fillna("unknown")
    partitions = force_col.unique()
    print(f"[norm_consolidation]   Deontic partitions: {list(partitions)}")

    for force_val in partitions:
        part_mask = (force_col == force_val).values
        part_indices = np.where(part_mask)[0]
        if len(part_indices) == 0:
            continue

        part_embeddings = embeddings[part_indices]

        if len(part_indices) < min_cluster_size:
            print(f"[norm_consolidation]     {force_val}: {len(part_indices)} norms "
                  f"(< min_cluster_size={min_cluster_size}, all singletons)")
            continue

        part_labels = _cluster_embeddings(
            part_embeddings,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
        )

        for i, idx in enumerate(part_indices):
            if part_labels[i] >= 0:
                labels[idx] = part_labels[i] + global_cluster_offset

        n_part_clusters = len(set(part_labels)) - (1 if -1 in part_labels else 0)
        global_cluster_offset += max(n_part_clusters, 0)
        print(f"[norm_consolidation]     {force_val}: {len(part_indices)} norms -> "
              f"{n_part_clusters} clusters")

    return labels, global_cluster_offset


# ---------------------------------------------------------------------------
# Phase B: LLM-based merging
# ---------------------------------------------------------------------------

def _build_cluster_df(df: pd.DataFrame, labels: np.ndarray) -> pd.DataFrame:
    """Attach cluster labels and build one row per cluster for LLM merging.

    Singletons (label == -1) are kept as their own 1-element clusters.
    Returns a DataFrame where each row represents one cluster, with columns:
      cluster_id, cluster_size, member_norms (JSON list of norm dicts),
      member_indices (list of original df indices).
    """
    df = df.copy()
    df["_cluster_label"] = labels

    # Assign unique IDs to singletons so each gets its own cluster
    next_id = int(labels.max()) + 1 if len(labels) > 0 else 0
    singleton_ids = []
    for label in labels:
        if label == -1:
            singleton_ids.append(next_id)
            next_id += 1
        else:
            singleton_ids.append(int(label))
    df["_cluster_id"] = singleton_ids

    cluster_rows = []
    for cid, group in df.groupby("_cluster_id"):
        member_norms = []
        for _, row in group.iterrows():
            member_norms.append({
                "norm_articulation": row.get("raz_norm_articulation"),
                "prescriptive_element": row.get("raz_prescriptive_element"),
                "norm_subject": row.get("raz_norm_subject"),
                "norm_act": row.get("raz_norm_act"),
                "condition_of_application": row.get("raz_condition_of_application"),
                "normative_force": row.get("raz_normative_force"),
                "context": row.get("raz_context"),
                "governs_info_flow": row.get("raz_governs_info_flow"),
                "info_flow_note": row.get("raz_info_flow_note"),
                "confidence_quant": row.get("raz_confidence_quant"),
            })
        cluster_rows.append({
            "cluster_id": int(cid),
            "cluster_size": len(group),
            "member_norms_json": json.dumps(member_norms, ensure_ascii=False),
            "member_articulations": [
                str(row.get("raz_norm_articulation") or "")
                for _, row in group.iterrows()
            ],
            "member_indices": list(group.index),
            "mean_confidence": group["raz_confidence_quant"].mean()
            if "raz_confidence_quant" in group.columns
            else None,
        })

    return pd.DataFrame(cluster_rows)


try:
    from json_repair import repair_json
    _JSON_REPAIR_OK = True
except ImportError:
    _JSON_REPAIR_OK = False


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


def _singleton_result(
    crow: Any,
    orig: Any,
    group_key: str,
) -> Dict[str, Any]:
    """Build a result dict for a singleton (no LLM merge needed)."""
    idx = crow["member_indices"][0]
    return {
        "group_key": group_key,
        "cluster_id": crow["cluster_id"],
        "cluster_size": 1,
        "canonical_prescriptive_element": orig.get("raz_prescriptive_element"),
        "canonical_norm_subject": orig.get("raz_norm_subject"),
        "canonical_norm_act": orig.get("raz_norm_act"),
        "canonical_condition_of_application": orig.get("raz_condition_of_application"),
        "canonical_normative_force": orig.get("raz_normative_force"),
        "canonical_norm_articulation": orig.get("raz_norm_articulation"),
        "canonical_context": orig.get("raz_context"),
        "canonical_governs_info_flow": orig.get("raz_governs_info_flow"),
        "canonical_info_flow_note": orig.get("raz_info_flow_note"),
        "consolidation_rationale": "singleton — no merge needed",
        "abstraction_map": None,
        "source_norm_ids": json.dumps([int(idx)]),
        "source_norm_articulations": json.dumps(
            [str(orig.get("raz_norm_articulation") or "")],
            ensure_ascii=False,
        ),
        "mean_confidence": crow["mean_confidence"],
    }


def _merged_result(
    mrow: Any,
    merged: Dict[str, Any],
    group_key: str,
) -> Dict[str, Any]:
    """Build a result dict from a successful LLM merge."""
    norm = merged.get("canonical_norm", {})
    return {
        "group_key": group_key,
        "cluster_id": mrow["cluster_id"],
        "cluster_size": mrow["cluster_size"],
        "canonical_prescriptive_element": norm.get("prescriptive_element"),
        "canonical_norm_subject": norm.get("norm_subject"),
        "canonical_norm_act": norm.get("norm_act"),
        "canonical_condition_of_application": norm.get("condition_of_application"),
        "canonical_normative_force": merged.get("normative_force"),
        "canonical_norm_articulation": merged.get("canonical_articulation"),
        "canonical_context": merged.get("context"),
        "canonical_governs_info_flow": merged.get("governs_information_flow"),
        "canonical_info_flow_note": merged.get("information_flow_note"),
        "consolidation_rationale": merged.get("consolidation_rationale"),
        "abstraction_map": json.dumps(
            merged.get("abstraction_map", {}), ensure_ascii=False
        ) if merged.get("abstraction_map") else None,
        "source_norm_ids": json.dumps(
            [int(i) for i in mrow.get("member_indices", [])]
        ),
        "source_norm_articulations": json.dumps(
            [str(a) for a in mrow.get("member_articulations", [])],
            ensure_ascii=False,
        ),
        "mean_confidence": mrow.get("mean_confidence"),
    }


def _fallback_result(
    mrow: Any,
    orig: Any,
    group_key: str,
) -> Dict[str, Any]:
    """Build a result dict when LLM merge failed (use best member)."""
    return {
        "group_key": group_key,
        "cluster_id": mrow["cluster_id"],
        "cluster_size": mrow["cluster_size"],
        "canonical_prescriptive_element": orig.get("raz_prescriptive_element"),
        "canonical_norm_subject": orig.get("raz_norm_subject"),
        "canonical_norm_act": orig.get("raz_norm_act"),
        "canonical_condition_of_application": orig.get("raz_condition_of_application"),
        "canonical_normative_force": orig.get("raz_normative_force"),
        "canonical_norm_articulation": orig.get("raz_norm_articulation"),
        "canonical_context": orig.get("raz_context"),
        "canonical_governs_info_flow": orig.get("raz_governs_info_flow"),
        "canonical_info_flow_note": orig.get("raz_info_flow_note"),
        "consolidation_rationale": "LLM merge failed — used highest-confidence member",
        "abstraction_map": None,
        "source_norm_ids": json.dumps(
            [int(i) for i in mrow.get("member_indices", [])]
        ),
        "source_norm_articulations": json.dumps(
            [str(a) for a in mrow.get("member_articulations", [])],
            ensure_ascii=False,
        ),
        "mean_confidence": mrow.get("mean_confidence"),
    }


# ---------------------------------------------------------------------------
# Main stage function
# ---------------------------------------------------------------------------

def run_norm_consolidation_stage(df: pd.DataFrame, cfg: Any) -> pd.DataFrame:
    """Consolidate similar norms via embedding clustering + LLM merging.

    Norms are first grouped by a source column (default: ``gutenberg_id``)
    so that clustering only happens within each book/text. Within each
    group, norms are further partitioned by ``normative_force`` (deontic
    operator) before embedding-based clustering, since norms with
    different deontic operators are definitionally distinct.

    Args:
        df: Input DataFrame with raz_* columns from norm_extraction.
        cfg: Hydra config with prompt_consolidation, model, sampling_params,
             and consolidation.* knobs.

    Returns:
        DataFrame with one row per consolidated canonical norm, plus
        provenance columns linking back to source norms.
    """
    required = ["raz_norm_articulation"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[norm_consolidation] Missing required columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Filter rows with valid norm articulations
    pre_filter = len(df)
    mask = df["raz_norm_articulation"].notna() & (df["raz_norm_articulation"] != "")
    df = df[mask].reset_index(drop=True)
    post_filter = len(df)
    if pre_filter != post_filter:
        print(f"[norm_consolidation] Filtered {pre_filter - post_filter} rows "
              f"with empty/null raz_norm_articulation ({post_filter} remaining)")
    if post_filter == 0:
        print("[norm_consolidation] No valid norms to consolidate")
        return pd.DataFrame(columns=_CONSOLIDATED_COLUMNS)

    # Assign a stable source norm ID to each input row
    df["_source_norm_id"] = range(len(df))

    # ------------------------------------------------------------------
    # Read consolidation config
    # ------------------------------------------------------------------
    consolidation_cfg = OmegaConf.select(cfg, "consolidation") or {}
    embedding_model = str(
        OmegaConf.select(consolidation_cfg, "embedding_model")
        or "all-MiniLM-L6-v2"
    )
    min_cluster_size = int(
        OmegaConf.select(consolidation_cfg, "min_cluster_size") or 3
    )
    min_samples = int(
        OmegaConf.select(consolidation_cfg, "min_samples") or 2
    )
    cluster_selection_epsilon = float(
        OmegaConf.select(consolidation_cfg, "cluster_selection_epsilon") or 0.0
    )
    group_by_col = str(
        OmegaConf.select(consolidation_cfg, "group_by") or "gutenberg_id"
    )

    # ------------------------------------------------------------------
    # Validate group_by column
    # ------------------------------------------------------------------
    if group_by_col not in df.columns:
        raise ValueError(
            f"[norm_consolidation] group_by column '{group_by_col}' not found "
            f"in input DataFrame. Available columns: {list(df.columns)}. "
            f"Set consolidation.group_by to a valid column name."
        )

    groups = df[group_by_col].fillna("unknown")
    unique_groups = groups.unique()
    print(f"[norm_consolidation] Grouping by '{group_by_col}': "
          f"{len(unique_groups)} groups ({list(unique_groups)})")

    # ------------------------------------------------------------------
    # Phase A: Embed ALL norms once (shared across groups)
    # ------------------------------------------------------------------
    norm_texts = [_build_norm_text(row) for row in df.to_dict("records")]
    embeddings = _embed_norms(norm_texts, model_name=embedding_model)

    # ------------------------------------------------------------------
    # Phase A (cont): Cluster within each group x deontic partition
    # ------------------------------------------------------------------
    labels = np.full(len(df), -1, dtype=int)
    global_cluster_offset = 0

    for gval in unique_groups:
        group_mask = (groups == gval).values
        group_indices = np.where(group_mask)[0]
        if len(group_indices) == 0:
            continue

        print(f"[norm_consolidation] Group '{gval}': {len(group_indices)} norms")

        group_df = df.iloc[group_indices].copy()
        group_embeddings = embeddings[group_indices]

        group_labels, global_cluster_offset = _cluster_within_deontic_partitions(
            group_df,
            group_embeddings,
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_epsilon=cluster_selection_epsilon,
            global_cluster_offset=global_cluster_offset,
        )

        # Write back into the global labels array
        for i, idx in enumerate(group_indices):
            labels[idx] = group_labels[i]

    total_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    total_singletons = int((labels == -1).sum())
    print(f"[norm_consolidation] Total across all groups: {total_clusters} clusters, "
          f"{total_singletons} singletons")

    # ------------------------------------------------------------------
    # Phase B: LLM merge per cluster
    # ------------------------------------------------------------------
    # Attach group_key to each row so it propagates into cluster_df
    df["_group_key"] = groups.values
    cluster_df = _build_cluster_df(df, labels)

    # Propagate group_key to cluster rows (all members in a cluster share
    # the same group, so just take the first member's value)
    cluster_df["group_key"] = cluster_df["member_indices"].apply(
        lambda idxs: str(df.iloc[idxs[0]]["_group_key"]) if idxs else "unknown"
    )

    singletons = cluster_df[cluster_df["cluster_size"] == 1].copy()
    multi = cluster_df[cluster_df["cluster_size"] > 1].copy()

    print(f"[norm_consolidation] {len(singletons)} singleton clusters "
          f"(pass-through), {len(multi)} multi-norm clusters (LLM merge)")

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
                "[norm_consolidation] No prompt config found at "
                "'prompt_consolidation' or 'prompt'."
            )

        system_prompt = str(OmegaConf.select(prompt_cfg, "system_prompt"))
        prompt_template = str(OmegaConf.select(prompt_cfg, "prompt_template"))

        sampling_params = dict(
            OmegaConf.to_container(
                OmegaConf.select(cfg, "sampling_params"),
                resolve=True,
            ) or {}
        )
        sampling_params.setdefault("temperature", 0.0)
        sampling_params.setdefault("max_tokens", 2048)

        from ..ci_schema import ConsolidatedNormResult
        json_schema = ConsolidatedNormResult.model_json_schema()
        sampling_params["guided_decoding"] = {"json": json_schema}

        # Max member norms to include in a single merge prompt.
        # Larger clusters are truncated to the highest-confidence members
        # to stay within the model's context window (32768 tokens).
        _MAX_MEMBERS_PER_PROMPT = 25

        def _preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
            result_row = dict(row)
            member_norms_json = row["member_norms_json"]
            cluster_size = row["cluster_size"]

            # Truncate oversized clusters to fit context window
            if cluster_size > _MAX_MEMBERS_PER_PROMPT:
                members = json.loads(member_norms_json)
                # Sort by confidence (descending), keep top N
                members.sort(
                    key=lambda m: m.get("confidence_quant") or 0,
                    reverse=True,
                )
                members = members[:_MAX_MEMBERS_PER_PROMPT]
                member_norms_json = json.dumps(members, ensure_ascii=False)
                print(f"[norm_consolidation] Truncated cluster "
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
            obj, parse_error = _extract_json(gen_text)
            if obj is not None:
                result_row["merged_norm"] = obj
                result_row["merge_failed"] = False
            else:
                print(f"[norm_consolidation] JSON parse error: {parse_error}")
                result_row["merged_norm"] = None
                result_row["merge_failed"] = True
                result_row["merge_error"] = str(parse_error)
            return result_row

        merged_df = run_vllm_inference(
            df=multi,
            cfg=cfg,
            preprocess=_preprocess,
            postprocess=_postprocess,
            stage_name="norm_consolidation",
        )

        # Unpack LLM results
        for _, mrow in merged_df.iterrows():
            merged = mrow.get("merged_norm")
            group_key = mrow.get("group_key", "unknown")
            member_idx = mrow.get("member_indices", [])

            if isinstance(merged, dict):
                results.append(_merged_result(mrow, merged, group_key))
            else:
                print(f"[norm_consolidation] Cluster {mrow['cluster_id']}: "
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
            print(f"[norm_consolidation] '{gk}': {n_input} norms -> "
                  f"{len(gdf)} canonical ({n_merged} merged, {n_single} singletons)")

    print(f"[norm_consolidation] Total: {post_filter} norms -> "
          f"{len(out_df)} canonical norms")
    return out_df
