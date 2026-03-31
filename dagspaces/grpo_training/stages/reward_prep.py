"""Reward preparation stage: pre-compute R_ground judge scores.

Uses Qwen2.5-72B (via run_vllm_inference) to evaluate normative grounding
of training prompts against their normative universes. Uses pre-computed
Qwen3-Embedding-8B embeddings from the norm_universe stage for retrieval.
"""

import hashlib
import json
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

from dagspaces.common.vllm_inference import run_vllm_inference
from dagspaces.common.stage_utils import extract_last_json
from .norm_universe import EMBED_INSTRUCTION


def _make_prompt_id(text: str) -> str:
    """Create a stable hash identifier for a prompt."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def _load_precomputed_embeddings(
    embeddings_dir: str,
    norm_universes: Dict[str, list],
) -> Dict[str, np.ndarray]:
    """Load per-book pre-computed Qwen3-Embedding-8B embeddings.

    Falls back to None if embeddings aren't available.
    """
    emb_by_source: Dict[str, np.ndarray] = {}
    for source_id in norm_universes:
        npy_path = os.path.join(embeddings_dir, f"{source_id}.npy")
        if os.path.exists(npy_path):
            emb_by_source[source_id] = np.load(npy_path)
    return emb_by_source


def run_reward_prep_stage(
    sft_df: pd.DataFrame,
    norm_universes: Dict[str, list],
    cfg: Any,
    embeddings_dir: Optional[str] = None,
) -> pd.DataFrame:
    """Pre-compute R_ground judge evaluations.

    For each training prompt:
    1. Pair with its correct N_hat_b (normative universe)
    2. Also pair with 1-2 contrastive N_hat_b' (from different sources)
    3. Run Qwen2.5-72B judge to evaluate grounding quality
    4. Cache results for GRPO training lookup

    Args:
        sft_df: SFT pairs DataFrame with messages, source_id columns.
        norm_universes: Dict mapping source_id -> list of norm dicts.
        cfg: Hydra config with judge_model and prompt sections.
        embeddings_dir: Dir with per-book .npy embeddings from norm_universe stage.

    Returns:
        DataFrame with columns: prompt_id, source_id, is_contrastive,
        judge_score, judge_response.
    """
    contrastive_ratio = float(
        OmegaConf.select(cfg, "training.grpo.contrastive_ratio") or 0.1
    )

    # Load judge prompt template
    prompt_cfg = OmegaConf.select(cfg, "prompt_reward_judge") or OmegaConf.select(cfg, "prompt")
    if prompt_cfg is None:
        raise RuntimeError("[reward_prep] No prompt config found at 'prompt_reward_judge'")

    system_prompt = str(OmegaConf.select(prompt_cfg, "system_prompt") or "")
    prompt_template = str(OmegaConf.select(prompt_cfg, "prompt_template") or "")

    # ---------------------------------------------------------------
    # Load pre-computed embeddings or embed on-the-fly
    # ---------------------------------------------------------------
    TOP_K_NORMS = 3

    norm_embeddings_by_source: Dict[str, np.ndarray] = {}
    retrieval_model = None

    if embeddings_dir and os.path.isdir(embeddings_dir):
        # Use pre-computed Qwen3-Embedding-8B embeddings
        norm_embeddings_by_source = _load_precomputed_embeddings(
            embeddings_dir, norm_universes
        )
        loaded = sum(len(v) for v in norm_embeddings_by_source.values())
        print(f"[reward_prep] Loaded pre-computed embeddings for "
              f"{len(norm_embeddings_by_source)} books ({loaded} vectors)")

        # Load the same embedding model for query encoding
        embedding_model_path = str(
            OmegaConf.select(cfg, "embedding_model.model_source", default=None) or OmegaConf.select(cfg, "model.embedding_model_source", default=None)
            or OmegaConf.select(cfg, "norm_universe.embedding_model")
            or ""
        )
        if embedding_model_path:
            from sentence_transformers import SentenceTransformer
            print(f"[reward_prep] Loading query embedding model: {embedding_model_path}")
            retrieval_model = SentenceTransformer(
                embedding_model_path,
                device="cuda:0",
                tokenizer_kwargs={"padding_side": "left"},
            )
    else:
        # Fallback: embed universe norms on-the-fly with whatever model is available
        try:
            from sentence_transformers import SentenceTransformer
            model_name = str(
                OmegaConf.select(cfg, "embedding_model.model_source", default=None) or OmegaConf.select(cfg, "model.embedding_model_source", default=None)
                or OmegaConf.select(cfg, "reward_prep.retrieval_model")
                or "all-MiniLM-L6-v2"
            )
            print(f"[reward_prep] No pre-computed embeddings. Embedding on-the-fly with {model_name}")
            retrieval_model = SentenceTransformer(model_name)
            for source_id, norms in norm_universes.items():
                texts = [
                    n.get("norm_articulation") or n.get("canonical_norm_articulation") or str(n)
                    for n in norms
                ]
                embs = retrieval_model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
                norm_embeddings_by_source[source_id] = np.asarray(embs)
            print(f"[reward_prep] Indexed {sum(len(v) for v in norm_embeddings_by_source.values())} norms")
        except Exception as e:
            print(f"[reward_prep] Retrieval unavailable ({e}), will truncate universes")

    _use_retrieval = bool(norm_embeddings_by_source) and retrieval_model is not None

    def _retrieve_relevant_norms(
        query: str, source_id: str, contrastive_source: Optional[str] = None,
    ) -> str:
        """Retrieve top-k norms from the universe most relevant to a query.

        Args:
            query: Free-text query built from a single flow's fields.
            source_id: The source book's ID.
            contrastive_source: If set, retrieve from this (wrong) source instead.
        """
        target_id = contrastive_source or source_id
        norms = norm_universes.get(target_id, [])
        if not norms:
            return "[]"

        if len(norms) <= TOP_K_NORMS:
            return json.dumps(norms, ensure_ascii=False, indent=1)

        if _use_retrieval and target_id in norm_embeddings_by_source:
            query_emb = retrieval_model.encode(
                [EMBED_INSTRUCTION + query],
                normalize_embeddings=True,
            )
            sims = np.dot(norm_embeddings_by_source[target_id], query_emb.T).flatten()
            top_indices = np.argsort(sims)[-TOP_K_NORMS:][::-1]
            selected = [norms[i] for i in top_indices]
        else:
            selected = norms[:TOP_K_NORMS]

        return json.dumps(selected, ensure_ascii=False, indent=1)

    def _flow_to_query(flow: Dict[str, Any]) -> str:
        """Build a retrieval query from a single flow's CI tuple fields."""
        parts = []
        for key in ("sender", "recipient", "information_type",
                     "context", "transmission_principle", "subject"):
            val = flow.get(key, "")
            if val:
                parts.append(str(val))
        # Also include invoked norms as query signal
        invoked = flow.get("norms_invoked", [])
        if isinstance(invoked, list):
            parts.extend(str(n) for n in invoked)
        return " ".join(parts) if parts else "information flow"

    # ---------------------------------------------------------------
    # Build per-flow evaluation rows
    # ---------------------------------------------------------------
    eval_rows: List[Dict[str, Any]] = []
    no_flow_rows: List[Dict[str, Any]] = []
    all_source_ids = list(norm_universes.keys())

    for _, row in sft_df.iterrows():
        messages_raw = row.get("messages", "[]")
        if isinstance(messages_raw, str):
            messages = json.loads(messages_raw)
        else:
            messages = messages_raw

        source_id = str(row.get("source_id", ""))
        if not source_id or source_id not in norm_universes:
            continue

        chunk_text = ""
        completion_text = ""
        for msg in messages:
            if msg.get("role") == "user":
                chunk_text = msg.get("content", "")
            elif msg.get("role") == "assistant":
                completion_text = msg.get("content", "")

        if not chunk_text or not completion_text:
            continue

        prompt_id = _make_prompt_id(chunk_text)

        # Parse completion to extract individual flows
        try:
            completion = json.loads(completion_text)
            flows = completion.get("flows", [])
        except (json.JSONDecodeError, AttributeError):
            flows = []

        # No-flow completions: score 1.0 directly, no judge call
        if not flows:
            no_flow_rows.append({
                "prompt_id": prompt_id,
                "source_id": source_id,
                "is_contrastive": False,
                "judge_score": 1.0,
                "norm_match_score": 1.0,
                "governance_score": 1.0,
                "judge_response": json.dumps({"no_flows": True}),
            })
            continue

        # Explode into per-flow eval rows
        for flow_idx, flow in enumerate(flows):
            flow_query = _flow_to_query(flow)
            flow_json = json.dumps(flow, ensure_ascii=False, indent=1)
            relevant_norms = _retrieve_relevant_norms(flow_query, source_id)

            eval_rows.append({
                "prompt_id": prompt_id,
                "source_id": source_id,
                "flow_index": flow_idx,
                "is_contrastive": False,
                "chunk_text": chunk_text,
                "flow_json": flow_json,
                "norm_universe_json": relevant_norms,
            })

            # Contrastive pairing: same flow, norms from wrong source
            if random.random() < contrastive_ratio and len(all_source_ids) > 1:
                wrong_sources = [s for s in all_source_ids if s != source_id]
                wrong_source = random.choice(wrong_sources)
                contrastive_norms = _retrieve_relevant_norms(
                    flow_query, source_id, contrastive_source=wrong_source,
                )
                eval_rows.append({
                    "prompt_id": prompt_id,
                    "source_id": source_id,
                    "flow_index": flow_idx,
                    "is_contrastive": True,
                    "contrastive_source": wrong_source,
                    "chunk_text": chunk_text,
                    "flow_json": flow_json,
                    "norm_universe_json": contrastive_norms,
                })

    eval_df = pd.DataFrame(eval_rows)
    n_flows = len(eval_df[eval_df["is_contrastive"] == False]) if len(eval_df) else 0
    n_contrastive = len(eval_df[eval_df["is_contrastive"] == True]) if len(eval_df) else 0
    print(f"[reward_prep] Built {len(eval_df)} per-flow judge rows "
          f"({n_flows} correct, {n_contrastive} contrastive, "
          f"{len(no_flow_rows)} no-flow prompts scored 1.0)")

    if len(eval_df) == 0:
        return pd.DataFrame(columns=[
            "prompt_id", "source_id", "is_contrastive",
            "judge_score", "judge_response",
        ])

    # Free retrieval model before loading judge
    del retrieval_model
    import gc, torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Override model config to use the judge model
    judge_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    judge_model = OmegaConf.select(cfg, "judge_model")
    if judge_model:
        OmegaConf.update(judge_cfg, "model", judge_model)

    sampling_params = {
        "temperature": 0.0,
        "max_tokens": 512,
    }

    from ..schemas import FlowGovernanceJudgment
    json_schema = FlowGovernanceJudgment.model_json_schema()
    sampling_params["guided_decoding"] = {"json": json_schema}

    def _preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        result_row = dict(row)
        user_prompt = prompt_template.replace(
            "{{chunk_text}}", str(row.get("chunk_text", ""))
        ).replace(
            "{{flow_json}}", str(row.get("flow_json", "{}"))
        ).replace(
            "{{norm_universe_json}}", str(row.get("norm_universe_json", "[]"))
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
        gen_text = result_row.get("generated_text", "")

        parsed = extract_last_json(gen_text)
        if parsed and isinstance(parsed, dict):
            nm = float(parsed.get("norm_match_score", 0.0))
            gov = float(parsed.get("governance_score", 0.0))
            result_row["norm_match_score"] = nm
            result_row["governance_score"] = gov
            result_row["judge_score"] = 0.5 * nm + 0.5 * gov
            result_row["judge_response"] = json.dumps(parsed, ensure_ascii=False)
        else:
            result_row["norm_match_score"] = 0.0
            result_row["governance_score"] = 0.0
            result_row["judge_score"] = 0.0
            result_row["judge_response"] = gen_text

        return result_row

    result_df = run_vllm_inference(
        df=eval_df,
        cfg=judge_cfg,
        preprocess=_preprocess,
        postprocess=_postprocess,
        stage_name="reward_prep",
    )

    # -------------------------------------------------------------------
    # Aggregate per-flow scores back to prompt level
    # -------------------------------------------------------------------
    group_cols = ["prompt_id", "source_id", "is_contrastive"]
    score_cols = ["judge_score", "norm_match_score", "governance_score"]

    # Ensure score columns exist
    for col in score_cols:
        if col not in result_df.columns:
            result_df[col] = 0.0

    agg_df = (
        result_df
        .groupby(group_cols, as_index=False)[score_cols]
        .mean()
    )

    # Collect per-flow judge responses as JSON array per prompt
    if "judge_response" in result_df.columns:
        resp_agg = (
            result_df
            .groupby(group_cols, as_index=False)["judge_response"]
            .agg(lambda x: json.dumps(list(x)))
        )
        agg_df = agg_df.merge(resp_agg, on=group_cols, how="left")

    # Append no-flow rows (pre-scored 1.0, no judge call)
    if no_flow_rows:
        no_flow_df = pd.DataFrame(no_flow_rows)
        agg_df = pd.concat([agg_df, no_flow_df], ignore_index=True)

    keep_cols = [
        "prompt_id", "source_id", "is_contrastive",
        "judge_score", "norm_match_score", "governance_score",
        "judge_response",
    ]
    out_cols = [c for c in keep_cols if c in agg_df.columns]
    out_df = agg_df[out_cols].copy()

    correct_mask = out_df["is_contrastive"] == False
    contrastive_mask = out_df["is_contrastive"] == True
    correct_mean = out_df.loc[correct_mask, "judge_score"].mean()
    contrastive_mean = out_df.loc[contrastive_mask, "judge_score"].mean() if contrastive_mask.any() else float("nan")
    nm_mean = out_df.loc[correct_mask, "norm_match_score"].mean() if "norm_match_score" in out_df.columns else float("nan")
    gov_mean = out_df.loc[correct_mask, "governance_score"].mean() if "governance_score" in out_df.columns else float("nan")
    print(f"[reward_prep] Judge evaluation complete. "
          f"Mean score (correct): {correct_mean:.3f} "
          f"(norm_match={nm_mean:.3f}, governance={gov_mean:.3f}), "
          f"Mean score (contrastive): {contrastive_mean:.3f}")

    return out_df
