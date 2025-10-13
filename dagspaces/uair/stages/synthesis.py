"""Cluster synthesis stage: Aggregate and synthesize insights from topic clusters."""

from typing import Any, Dict, List, Optional, Tuple
import os
import json
import logging
import hashlib
import pandas as pd
import numpy as np
from omegaconf import OmegaConf
from dataclasses import dataclass

try:
    import ray  # noqa: F401
    from ray.data.llm import build_llm_processor, vLLMEngineProcessorConfig  # type: ignore
    _RAY_OK = True
except Exception:
    _RAY_OK = False

_VLLM_LOGS_SILENCED = False


def _maybe_silence_vllm_logs() -> None:
    """Silence vLLM logs when RULE_TUPLES_SILENT is set."""
    global _VLLM_LOGS_SILENCED
    if _VLLM_LOGS_SILENCED:
        return
    try:
        from dagspaces.uair.logging_filters import PatternModuloFilter
        lg = logging.getLogger("vllm")
        try:
            n = int(os.environ.get("UAIR_VLLM_LOG_EVERY", "10") or "10")
        except Exception:
            n = 10
        lg.setLevel(logging.INFO)
        try:
            existing_filters = getattr(lg, "filters", [])
            if not any(getattr(f, "__class__", object).__name__ == "PatternModuloFilter" for f in existing_filters):
                lg.addFilter(PatternModuloFilter(mod=n, pattern="Elapsed time for batch"))
        except Exception:
            pass
        if os.environ.get("RULE_TUPLES_SILENT"):
            lg.setLevel(logging.ERROR)
        _VLLM_LOGS_SILENCED = True
    except Exception:
        pass


def _ensure_ray_init(cfg) -> None:
    """Initialize Ray with appropriate memory settings."""
    try:
        import ray  # type: ignore
        if not ray.is_initialized():
            # Reuse Ray init logic from other stages
            cpus_alloc = None
            try:
                cpt = os.environ.get("SLURM_CPUS_PER_TASK")
                if cpt is not None and str(cpt).strip() != "":
                    cpus_alloc = int(cpt)
            except Exception:
                cpus_alloc = None
            
            try:
                job_mem_gb = int(getattr(cfg.runtime, "job_memory_gb", 64) or 64)
            except Exception:
                job_mem_gb = 64
            
            try:
                obj_store_bytes = int(max(1, job_mem_gb) * (1024 ** 3) * 0.90)
            except Exception:
                obj_store_bytes = int(64 * (1024 ** 3) * 0.90)
            
            try:
                if cpus_alloc is not None and int(cpus_alloc) > 0:
                    ray.init(log_to_driver=True, object_store_memory=obj_store_bytes, num_cpus=int(cpus_alloc))
                else:
                    ray.init(log_to_driver=True, object_store_memory=obj_store_bytes)
            except Exception:
                try:
                    ray.init(log_to_driver=True)
                except Exception:
                    pass
    except Exception:
        pass


def _detect_num_gpus() -> int:
    """Detect number of allocated GPUs."""
    try:
        cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        if cuda_visible and cuda_visible.strip():
            gpu_indices = [x.strip() for x in cuda_visible.split(",") if x.strip()]
            if gpu_indices:
                return len(gpu_indices)
    except Exception:
        pass
    
    try:
        slurm_gpus = os.environ.get("SLURM_GPUS_PER_NODE") or os.environ.get("SLURM_GPUS_ON_NODE")
        if slurm_gpus:
            try:
                if ":" in slurm_gpus:
                    return int(slurm_gpus.split(":")[-1])
                return int(slurm_gpus)
            except Exception:
                pass
    except Exception:
        pass
    
    try:
        import torch  # type: ignore
        if torch.cuda.is_available():
            count = torch.cuda.device_count()
            if count > 0:
                return count
    except Exception:
        pass
    
    return 1


def _detect_gpu_type() -> str:
    """Detect GPU type."""
    try:
        import torch  # type: ignore
        if torch.cuda.is_available() and torch.cuda.device_count() > 0:
            gpu_name = torch.cuda.get_device_name(0).lower()
            
            if "a6000" in gpu_name:
                return "rtx_a6000"
            elif "a5000" in gpu_name:
                return "rtx_a5000"
            elif "a100" in gpu_name:
                return "a100"
            elif "v100" in gpu_name:
                return "v100"
            elif "a40" in gpu_name:
                return "a40"
            
            return "unknown"
    except Exception:
        pass
    
    return "unknown"


def _apply_gpu_aware_batch_settings(engine_kwargs: Dict[str, Any], cfg) -> Dict[str, Any]:
    """Apply GPU-type-aware batch settings."""
    GPU_BATCH_SETTINGS = {
        "rtx_a6000": {"batch_size": 4, "max_num_seqs": 4},
        "rtx_a5000": {"batch_size": 2, "max_num_seqs": 2},
        "a100": {"batch_size": 8, "max_num_seqs": 8},
        "v100": {"batch_size": 4, "max_num_seqs": 4},
        "a40": {"batch_size": 4, "max_num_seqs": 4},
    }
    
    gpu_type = _detect_gpu_type()
    gpu_settings = GPU_BATCH_SETTINGS.get(gpu_type, {})
    
    if "max_num_seqs" not in engine_kwargs and gpu_settings:
        try:
            engine_kwargs["max_num_seqs"] = gpu_settings["max_num_seqs"]
            if not os.environ.get("RULE_TUPLES_SILENT"):
                print(f"Auto-set max_num_seqs={gpu_settings['max_num_seqs']} for {gpu_type}")
        except Exception:
            pass
    
    return gpu_settings


def _filter_vllm_engine_kwargs(ek: Dict[str, Any]) -> Dict[str, Any]:
    """Filter unsupported vLLM engine kwargs."""
    try:
        import vllm as _v
        accepted = None
        try:
            fields = getattr(getattr(_v, "AsyncEngineArgs", None), "__dataclass_fields__", None)
            if isinstance(fields, dict) and fields:
                accepted = set(fields.keys())
        except Exception:
            accepted = None
        
        if accepted is None:
            try:
                import inspect as _inspect
                sig = _inspect.signature(_v.AsyncEngineArgs.__init__)
                accepted = set(k for k in sig.parameters.keys() if k != "self")
            except Exception:
                accepted = None
        
        if accepted:
            return {k: v for k, v in ek.items() if k in accepted}
    except Exception:
        pass
    
    ek = dict(ek)
    for k in ("use_v2_block_manager",):
        ek.pop(k, None)
    return ek


def _to_json_str(value: Any) -> Optional[str]:
    """Serialize to JSON string."""
    try:
        if value is None:
            return None
        return json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        try:
            return str(value)
        except Exception:
            return None


def _serialize_arrow_unfriendly_in_row(row: Dict[str, Any], columns: List[str]) -> None:
    """In-place convert nested columns to JSON strings."""
    for col in columns:
        if col in row:
            val = row.get(col)
            if isinstance(val, (dict, list, tuple)):
                row[col] = _to_json_str(val)


def _extract_last_json(text: str) -> Optional[Dict[str, Any]]:
    """Extract the last JSON object from text.

    First tries to parse the whole string; on failure, finds the last {...} block.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    try:
        import re as _re
        snippets = _re.findall(r"\{[\s\S]*\}", text)
        for snip in reversed(snippets or []):
            try:
                obj = json.loads(snip)
                if isinstance(obj, dict):
                    return obj
            except Exception:
                continue
    except Exception:
        pass
    return None


@dataclass
class SynthesisConfig:
    """Configuration for synthesis stage."""
    strategy: str = "hierarchical_map_reduce"
    map_article_tokens: int = 2048
    reduce_batch_tokens: int = 4096
    synthesis_tokens: int = 6144
    articles_per_map_batch: int = 10
    max_reduce_depth: int = 3
    min_topic_prob: float = 0.3
    max_articles_per_cluster: int = 100
    top_k_articles: int = 10  # OPTIMIZATION: Reduced from 50 to 10 for faster preprocessing
    enable_citations: bool = True
    citation_threshold: float = 0.7
    
    @classmethod
    def from_cfg(cls, cfg) -> "SynthesisConfig":
        """Extract synthesis config from Hydra config."""
        try:
            synthesis_cfg = getattr(cfg, "synthesis", None)
            if synthesis_cfg is None:
                return cls()
            
            return cls(
                strategy=str(getattr(synthesis_cfg, "strategy", "hierarchical_map_reduce")),
                map_article_tokens=int(getattr(synthesis_cfg, "map_article_tokens", 2048)),
                reduce_batch_tokens=int(getattr(synthesis_cfg, "reduce_batch_tokens", 4096)),
                synthesis_tokens=int(getattr(synthesis_cfg, "synthesis_tokens", 6144)),
                articles_per_map_batch=int(getattr(synthesis_cfg, "articles_per_map_batch", 10)),
                max_reduce_depth=int(getattr(synthesis_cfg, "max_reduce_depth", 3)),
                min_topic_prob=float(getattr(synthesis_cfg, "min_topic_prob", 0.3)),
                max_articles_per_cluster=int(getattr(synthesis_cfg, "max_articles_per_cluster", 100)),
                top_k_articles=int(getattr(synthesis_cfg, "top_k_articles", 50)),
                enable_citations=bool(getattr(synthesis_cfg, "enable_citations", True)),
                citation_threshold=float(getattr(synthesis_cfg, "citation_threshold", 0.7)),
            )
        except Exception:
            return cls()


def _load_synthesis_schema(cfg) -> Optional[Dict[str, Any]]:
    """Load synthesis JSON schema."""
    try:
        schema_path = str(getattr(cfg, "synthesis", {}).get("schema_path", ""))
        if schema_path and os.path.exists(schema_path):
            with open(schema_path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    
    # Fallback: Basic schema
    return {
        "type": "object",
        "properties": {
            "primary_risk_type": {"type": "string"},
            "risk_confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "common_themes": {"type": "array", "items": {"type": "string"}},
            "synthesis_summary": {"type": "string"},
        },
        "required": ["primary_risk_type", "synthesis_summary"]
    }


def _extract_article_insights(row: Dict[str, Any], cfg) -> Dict[str, Any]:
    """Map phase: Extract structured insights from single article.
    
    This would be called via LLM in production; here we use a placeholder.
    """
    # Placeholder for LLM extraction
    # In real implementation, this would call vLLM with guided decoding
    
    article_hash = hashlib.sha1(str(row.get("article_text", "")).encode()).hexdigest()[:8]
    
    # Simulated extraction (replace with actual LLM call)
    insights = {
        "article_id": row.get("article_id"),
        "extracted_risk_type": "placeholder",
        "confidence": 0.8,
        "key_entities": ["entity1", "entity2"],
        "technology_type": "nlp",
        "key_quote": row.get("chunk_text", "")[:200] if row.get("chunk_text") else "",
    }
    
    return {**row, **insights}


def _group_articles_by_cluster(df: pd.DataFrame, synthesis_cfg: SynthesisConfig) -> pd.DataFrame:
    """Group articles by cluster/topic_id and prepare for synthesis."""
    
    # Filter by topic probability
    df_filtered = df[df["topic_prob"] >= synthesis_cfg.min_topic_prob].copy()
    
    # Group by topic_id
    grouped = []
    for topic_id, group in df_filtered.groupby("topic_id"):
        # Sort by topic_prob descending
        group_sorted = group.sort_values("topic_prob", ascending=False)
        
        # Limit articles per cluster
        if len(group_sorted) > synthesis_cfg.max_articles_per_cluster:
            group_sorted = group_sorted.head(synthesis_cfg.max_articles_per_cluster)
        
        # Aggregate metadata
        cluster_data = {
            "cluster_id": int(topic_id),
            "num_articles": len(group_sorted),
            "articles": group_sorted.to_dict("records"),
            "cluster_top_terms": group_sorted["topic_top_terms"].iloc[0] if "topic_top_terms" in group_sorted.columns else "",
            "avg_topic_prob": float(group_sorted["topic_prob"].mean()),
            "date_range": _extract_date_range(group_sorted),
            "countries": _extract_countries(group_sorted),
        }
        
        grouped.append(cluster_data)
    
    return pd.DataFrame(grouped)


def _extract_date_range(df: pd.DataFrame) -> str:
    """Extract date range from articles."""
    try:
        if "year" in df.columns:
            years = df["year"].dropna()
            if len(years) > 0:
                return f"{years.min()}-{years.max()}"
    except Exception:
        pass
    return "unknown"


def _extract_countries(df: pd.DataFrame) -> str:
    """Extract unique countries from articles."""
    try:
        if "country" in df.columns:
            countries = df["country"].dropna().unique()
            if len(countries) > 0:
                return ", ".join(sorted(countries[:5]))
    except Exception:
        pass
    return "global"


def _hierarchical_reduce(
    articles: List[Dict[str, Any]], 
    synthesis_cfg: SynthesisConfig,
    depth: int = 0
) -> Dict[str, Any]:
    """Hierarchical reduce: Recursively batch and summarize articles."""
    
    if depth >= synthesis_cfg.max_reduce_depth:
        # Max depth reached, return summary of current batch
        return _direct_summary(articles)
    
    # If small enough, synthesize directly
    if len(articles) <= synthesis_cfg.articles_per_map_batch:
        return _direct_summary(articles)
    
    # Otherwise, batch and recurse
    batch_size = synthesis_cfg.articles_per_map_batch
    batches = [articles[i:i+batch_size] for i in range(0, len(articles), batch_size)]
    
    # Recursively process batches
    intermediate_summaries = [
        _hierarchical_reduce(batch, synthesis_cfg, depth + 1)
        for batch in batches
    ]
    
    # Reduce intermediate summaries
    return _reduce_summaries(intermediate_summaries)


def _direct_summary(articles: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Directly summarize a batch of articles (placeholder for LLM call)."""
    
    # Placeholder: In production, this would call vLLM
    num_articles = len(articles)
    
    # Extract common patterns (placeholder logic)
    all_entities = []
    for art in articles:
        if "key_entities" in art and isinstance(art["key_entities"], list):
            all_entities.extend(art["key_entities"])
    
    return {
        "num_articles": num_articles,
        "summary": f"Summary of {num_articles} articles",
        "common_entities": list(set(all_entities))[:5],
        "themes": ["theme1", "theme2"],
    }


def _reduce_summaries(summaries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Reduce multiple summaries into one (placeholder for LLM call)."""
    
    total_articles = sum(s.get("num_articles", 0) for s in summaries)
    
    # Merge entities
    all_entities = []
    for s in summaries:
        if "common_entities" in s:
            all_entities.extend(s["common_entities"])
    
    return {
        "num_articles": total_articles,
        "summary": f"Aggregated summary of {len(summaries)} batches covering {total_articles} articles",
        "common_entities": list(set(all_entities))[:10],
        "themes": ["meta_theme1", "meta_theme2"],
    }


def _synthesize_cluster_final(
    cluster_data: Dict[str, Any],
    reduced_insights: Dict[str, Any],
    cfg,
    schema: Dict[str, Any]
) -> Dict[str, Any]:
    """Final synthesis with guided decoding (placeholder for vLLM call)."""
    
    cluster_id = cluster_data["cluster_id"]
    num_articles = cluster_data["num_articles"]
    
    # Placeholder synthesis
    # In production: Call vLLM with guided decoding using schema
    
    synthesis = {
        "cluster_id": cluster_id,
        "num_articles": num_articles,
        "primary_risk_type": "Privacy Violations",  # Placeholder
        "risk_confidence": 0.75,
        "risk_evidence": _to_json_str(["Evidence 1", "Evidence 2"]),
        "common_themes": _to_json_str(reduced_insights.get("themes", [])),
        "temporal_patterns": _to_json_str({"trend": "increasing"}),
        "geographic_distribution": _to_json_str({"primary_countries": [cluster_data.get("countries", "")]}),
        "synthesis_summary": f"This cluster of {num_articles} articles focuses on...",
        "key_entities": _to_json_str(reduced_insights.get("common_entities", [])),
        "technology_types": _to_json_str(["NLP", "Computer Vision"]),
        "impact_areas": _to_json_str(["healthcare", "finance"]),
        "cluster_top_terms": cluster_data.get("cluster_top_terms", ""),
        "avg_topic_prob": cluster_data.get("avg_topic_prob", 0.0),
    }
    
    return synthesis


def run_synthesis_stage(df, cfg, logger=None, articles_df=None):
    """Cluster synthesis stage: Aggregate and synthesize topic cluster insights.
    
    Input: Output from topic stage with cluster assignments + original articles with text
    Output: Cluster-level synthesis with structured insights
    
    Args:
        df: pandas DataFrame with topic modeling results (cluster assignments)
        cfg: Hydra config
        logger: Optional WandbLogger for logging
        articles_df: pandas DataFrame with original articles (must have article_text and article_id)
    
    Returns:
        pandas DataFrame with cluster syntheses
    """
    
    # Initialize Ray
    _ensure_ray_init(cfg)
    
    # Load configuration
    synthesis_cfg = SynthesisConfig.from_cfg(cfg)
    schema = _load_synthesis_schema(cfg)
    
    # Convert to pandas if needed
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["cluster_id", "num_articles", "synthesis_summary"])
    
    # Handle Ray Dataset (convert to pandas for now)
    is_ray_ds = hasattr(df, "map_batches") and hasattr(df, "count") and _RAY_OK
    if is_ray_ds:
        out = df.to_pandas()
    else:
        out = df.copy()
    
    # Validate required columns from topic stage output
    required_cols = ["topic_id", "topic_prob"]
    missing = [col for col in required_cols if col not in out.columns]
    if missing:
        raise ValueError(f"Input missing required columns: {missing}")
    
    # Join with original articles to get text
    if articles_df is not None:
        print(f"Joining {len(out)} cluster assignments with {len(articles_df)} articles...", flush=True)
        
        # Validate/ensure article identifiers exist on articles_df
        if "article_id" not in articles_df.columns:
            # Generate deterministic article_id consistent with classify: hash article_path else text
            try:
                df_a = articles_df.copy()
                def _gen_article_id(r: pd.Series) -> str:
                    try:
                        src = r.get("article_path")
                        if not isinstance(src, str) or src.strip() == "":
                            src = r.get("article_text") or r.get("chunk_text") or ""
                        return hashlib.sha1(str(src).encode("utf-8")).hexdigest()
                    except Exception:
                        return hashlib.sha1(str(r.get("article_text") or r.get("chunk_text") or "").encode("utf-8")).hexdigest()
                df_a["article_id"] = df_a.apply(_gen_article_id, axis=1)
                articles_df = df_a
            except Exception:
                raise ValueError("articles_df must have 'article_id' column or fields to derive it ('article_path' or text)")
        
        # Determine text column in articles
        text_col = None
        if "article_text" in articles_df.columns:
            text_col = "article_text"
        elif "chunk_text" in articles_df.columns:
            text_col = "chunk_text"
        else:
            raise ValueError("articles_df must have 'article_text' or 'chunk_text' column")
        
        # Join cluster assignments with article text
        # Keep only necessary columns from articles to save memory
        articles_subset = articles_df[["article_id", text_col]].copy()
        
        # Ensure article_id exists in cluster data
        if "article_id" not in out.columns:
            # Try to extract from unit_id if present
            if "unit_id" in out.columns:
                print("Extracting article_id from unit_id...", flush=True)
                out["article_id"] = out["unit_id"].apply(
                    lambda x: str(x).split("__")[0] if isinstance(x, str) and "__" in str(x) else str(x)
                )
            else:
                raise ValueError("Cluster data must have 'article_id' or 'unit_id' column")
        
        # Perform join
        out = out.merge(articles_subset, on="article_id", how="inner", suffixes=("", "_from_articles"))
        
        # Use the text column from articles
        if text_col == "article_text" and "article_text" in out.columns:
            out["chunk_text"] = out["article_text"]
        elif "chunk_text_from_articles" in out.columns:
            out["chunk_text"] = out["chunk_text_from_articles"]
        
        print(f"After join: {len(out)} rows with text", flush=True)
    else:
        # Fallback: work with what we have
        print("Warning: No articles_df provided. Working with cluster metadata only.", flush=True)
        if "topic_top_terms" in out.columns:
            out["chunk_text"] = out["topic_top_terms"].apply(
                lambda x: f"Cluster terms: {x}" if x else "No terms"
            )
        else:
            out["chunk_text"] = "No text available"
    
    # Group articles by cluster
    print(f"Grouping {len(out)} articles into clusters...", flush=True)
    clustered = _group_articles_by_cluster(out, synthesis_cfg)
    print(f"Processing {len(clustered)} clusters", flush=True)

    # LLM-guided synthesis via Ray vLLM when available
    if _RAY_OK:
        ek = dict(getattr(cfg.model, "engine_kwargs", {}))
        ek.setdefault("max_model_len", 4096)
        ek.setdefault("gpu_memory_utilization", 0.85)
        if "tensor_parallel_size" not in ek:
            try:
                ek["tensor_parallel_size"] = _detect_num_gpus()
                if not os.environ.get("RULE_TUPLES_SILENT"):
                    print(f"Auto-detected {ek['tensor_parallel_size']} GPU(s) for tensor parallelism")
            except Exception:
                ek.setdefault("tensor_parallel_size", 1)
        gpu_settings = _apply_gpu_aware_batch_settings(ek, cfg)
        ek.setdefault("enable_prefix_caching", True)
        ek.setdefault("use_v2_block_manager", True)
        ek.setdefault("tokenizer_mode", "auto")
        ek.setdefault("trust_remote_code", True)
        ek.setdefault("dtype", "auto")
        ek.setdefault("kv_cache_dtype", "auto")
        ek = _filter_vllm_engine_kwargs(ek)
        # Explicitly set guided decoding backend per Ray docs
        ek.setdefault("guided_decoding_backend", "xgrammar")
        try:
            batch_size_cfg = getattr(cfg.model, "batch_size", None)
            if batch_size_cfg is not None:
                batch_size = int(batch_size_cfg)
            elif gpu_settings and "batch_size" in gpu_settings:
                batch_size = int(gpu_settings["batch_size"])  # type: ignore[index]
                if not os.environ.get("RULE_TUPLES_SILENT"):
                    print(f"Auto-set batch_size={batch_size} for {_detect_gpu_type()}")
            else:
                # OPTIMIZATION: Increased default from 4 to 16 for synthesis
                # Synthesis prompts are shorter (no hierarchical reduce), so we can batch more
                batch_size = 16
        except Exception:
            batch_size = 16
        try:
            concurrency = int(getattr(cfg.model, "concurrency", 1) or 1)
        except Exception:
            concurrency = 1
        engine_config = vLLMEngineProcessorConfig(
            model_source=str(getattr(cfg.model, "model_source")),
            runtime_env={
                "env_vars": {
                    "VLLM_LOGGING_LEVEL": str(os.environ.get("VLLM_LOGGING_LEVEL", "WARNING")),
                    "WANDB_DISABLE_SERVICE": str(os.environ.get("WANDB_DISABLE_SERVICE", "true")),
                    "WANDB_SILENT": str(os.environ.get("WANDB_SILENT", "true")),
                }
            },
            engine_kwargs=ek,
            concurrency=int(concurrency),
            batch_size=int(batch_size),
        )

        try:
            sp_src = getattr(cfg, "sampling_params_synthesis", getattr(cfg, "sampling_params", {}))
            sampling_params = OmegaConf.to_container(sp_src, resolve=True) if isinstance(sp_src, (dict,)) or hasattr(sp_src, "_get_node") else dict(sp_src)
        except Exception:
            sampling_params = dict(getattr(cfg, "sampling_params", {}))
        try:
            if schema:
                # Ensure schema is plain JSON-serializable dict
                sampling_params["guided_decoding"] = {"json": dict(schema)}
        except Exception:
            pass

        try:
            system_prompt = str(getattr(cfg.prompt_synthesis, "system_prompt", ""))
        except Exception:
            system_prompt = ""
        if not system_prompt:
            system_prompt = "You are an expert AI safety researcher. Return a JSON conforming to the schema."
        try:
            user_template = str(getattr(cfg.prompt_synthesis, "synthesis_prompt", ""))
        except Exception:
            user_template = ""
        if not user_template:
            # Simplified prompt - let LLM do aggregation directly from top articles
            user_template = (
                "Synthesize insights from this cluster of {num_articles} articles.\n\n"
                "**Cluster Metadata:**\n"
                "- ID: {cluster_id}\n"
                "- Top Terms: {cluster_top_terms}\n"
                "- Date Range: {date_range}\n"
                "- Countries: {countries}\n\n"
                "**Top Articles (sorted by relevance):**\n{top_articles}\n\n"
                "Generate a structured synthesis following the schema."
            )

        ds = ray.data.from_pandas(clustered)

        def _format_top_articles(arts: List[Dict[str, Any]], k: int = 5) -> str:
            try:
                if not arts:
                    return _to_json_str([]) or "[]"
                try:
                    arts_sorted = sorted(arts, key=lambda r: float(r.get("topic_prob", 0.0)), reverse=True)
                except Exception:
                    arts_sorted = list(arts)
                top = arts_sorted[:max(1, int(k))]
                formatted = []
                for a in top:
                    aid = a.get("article_id") or a.get("unit_id") or None
                    txt = a.get("chunk_text") or a.get("article_text") or ""
                    try:
                        # OPTIMIZATION: Reduced snippet from 240 to 150 chars for faster processing
                        snip = str(txt)[:150]
                    except Exception:
                        snip = str(txt)
                    formatted.append({"article_id": aid, "quote": snip})
                return _to_json_str(formatted) or "[]"
            except Exception:
                return "[]"

        def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
            _maybe_silence_vllm_logs()
            # OPTIMIZATION: Skip expensive hierarchical reduce in preprocessing
            # Instead, pass top articles directly to LLM for synthesis
            articles_list = row.get("articles")
            if articles_list is None:
                articles_list = []
            top_articles_str = _format_top_articles(list(articles_list), int(getattr(synthesis_cfg, "top_k_articles", 5)))
            
            # Simple metadata aggregation (fast)
            num_articles = row.get("num_articles", 0)
            cluster_terms = row.get("cluster_top_terms")
            if cluster_terms is None:
                cluster_terms = []
            cluster_terms_str = ", ".join(cluster_terms) if isinstance(cluster_terms, list) else str(cluster_terms)
            
            try:
                date_range_val = row.get("date_range")
                if date_range_val is None:
                    date_range_val = "unknown"
                countries_val = row.get("countries")
                if countries_val is None:
                    countries_val = "global"
                
                utext = user_template.format(
                    cluster_id=row.get("cluster_id"),
                    num_articles=num_articles,
                    cluster_top_terms=cluster_terms_str,
                    date_range=str(date_range_val),
                    countries=str(countries_val),
                    aggregated_insights="",  # Skip pre-aggregation (let LLM handle it)
                    top_articles=top_articles_str,
                )
            except Exception:
                utext = f"Synthesize cluster {row.get('cluster_id')} with {num_articles} articles"
            
            base = {k: v for k, v in row.items() if k not in {"articles", "messages", "sampling_params", "generated_text"}}
            base.update({
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": utext},
                ],
                "sampling_params": sampling_params,
            })
            return base

        def _post(row: Dict[str, Any]) -> Dict[str, Any]:
            txt = row.get("generated_text")
            obj = _extract_last_json(txt if isinstance(txt, str) else "") or {}
            out: Dict[str, Any] = {
                "cluster_id": row.get("cluster_id"),
                "num_articles": row.get("num_articles"),
                "cluster_top_terms": row.get("cluster_top_terms"),
                "avg_topic_prob": row.get("avg_topic_prob"),
                "date_range": row.get("date_range"),
                "countries": row.get("countries"),
            }
            for k in (
                "primary_risk_type",
                "risk_confidence",
                "common_themes",
                "key_entities",
                "technology_types",
                "geographic_distribution",
                "temporal_patterns",
                "synthesis_summary",
                "impact_areas",
                "supporting_evidence",
            ):
                if k in obj:
                    out[k] = obj.get(k)
            _serialize_arrow_unfriendly_in_row(out, [
                "common_themes",
                "key_entities",
                "technology_types",
                "geographic_distribution",
                "temporal_patterns",
                "impact_areas",
                "supporting_evidence",
            ])
            try:
                if "risk_confidence" in out and out["risk_confidence"] is not None:
                    out["risk_confidence"] = float(out["risk_confidence"])  # type: ignore[assignment]
            except Exception:
                pass
            return out

        processor = build_llm_processor(engine_config, preprocess=_pre, postprocess=_post)
        result_df = processor(ds).to_pandas()
    else:
        # Fallback: placeholder without hardcoded risk fields
        syntheses = []
        for _, row in clustered.iterrows():
            cluster_data = row.to_dict()
            articles = cluster_data.pop("articles", [])
            try:
                reduced_insights = _hierarchical_reduce(articles, synthesis_cfg, depth=0)
                syntheses.append({
                    "cluster_id": cluster_data.get("cluster_id"),
                    "num_articles": cluster_data.get("num_articles"),
                    "primary_risk_type": None,
                    "risk_confidence": None,
                    "common_themes": _to_json_str(reduced_insights.get("themes", [])),
                    "key_entities": _to_json_str(reduced_insights.get("common_entities", [])),
                    "synthesis_summary": None,
                    "cluster_top_terms": cluster_data.get("cluster_top_terms", ""),
                    "avg_topic_prob": cluster_data.get("avg_topic_prob", 0.0),
                })
            except Exception:
                continue
        result_df = pd.DataFrame(syntheses)
    
    # Log results to wandb
    if logger and len(result_df) > 0:
        try:
            prefer_cols = [
                "cluster_id", "num_articles", "primary_risk_type", 
                "risk_confidence", "synthesis_summary"
            ]
            logger.log_table(
                result_df,
                "synthesis/clusters",
                prefer_cols=prefer_cols,
                panel_group="inspect_results"
            )
        except Exception as e:
            print(f"Warning: Failed to log synthesis results to wandb: {e}", flush=True)
    
    print(f"Synthesis complete: {len(result_df)} cluster summaries generated", flush=True)
    
    return result_df

