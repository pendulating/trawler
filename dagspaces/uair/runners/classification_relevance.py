"""Relevance classification stage runner."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict

import pandas as pd
from omegaconf import DictConfig, OmegaConf

from ..orchestrator import (
    StageExecutionContext,
    StageResult,
    _collect_outputs,
    _convert_to_pandas_if_needed,
    _safe_log_table,
    prepare_stage_input,
)
from ..stages.classify_relevance import run_classification_relevance
from .base import StageRunner


class ClassificationRelevanceRunner(StageRunner):
    """Runner for the classify_relevance stage."""
    
    stage_name = "classify_relevance"

    def run(self, context: StageExecutionContext) -> StageResult:
        """Execute the classify_relevance stage."""
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        # Ensure proper profile
        try:
            OmegaConf.update(cfg, "runtime.classification_profile", "relevance", merge=True)
        except Exception:
            pass
        # Note: Prompt injection and profile setup are handled internally by run_classification_relevance
        # Support streaming (Ray Datasets) since the stage implementation supports it
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        out = run_classification_relevance(in_obj, cfg)
        
        out = _convert_to_pandas_if_needed(out)
        
        # Calculate row count
        row_count = None
        if isinstance(out, pd.DataFrame):
            row_count = len(out)
        elif isinstance(out, pd.Series):
            row_count = len(out)
        elif hasattr(out, "__len__"):
            try:
                row_count = len(out)
            except Exception:
                pass
        
        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            # Save generic results if requested
            if "results" in context.output_paths:
                out.to_parquet(context.output_paths["results"], index=False)
            # Save "all" output (all classified articles)
            if "all" in context.output_paths:
                out.to_parquet(context.output_paths["all"], index=False)
            # Save "relevant" output (only relevant articles)
            if "relevant" in context.output_paths and "is_relevant" in out.columns:
                relevant_df = out[out["is_relevant"] == True].copy()
                relevant_df.to_parquet(context.output_paths["relevant"], index=False)
        
        # Log results table and keyword statistics to wandb
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                try:
                    print(
                        f"[classify_relevance] Logging columns to wandb ({len(out.columns)} total): {list(out.columns)}",
                        flush=True,
                    )
                except Exception:
                    pass
                prefer_cols = [
                    "article_id",
                    "is_relevant",
                    "relevance_answer",
                    "classification_mode",
                    "relevant_keyword",
                    "matched_keywords",
                    "keyword_match_count",
                    "article_text",
                    "article_path",
                    "country",
                    "year",
                ]
                # Filter to only columns that exist
                prefer_cols = [c for c in prefer_cols if c in out.columns]
                _safe_log_table(context.logger, out, "classify_relevance/results", prefer_cols=prefer_cols, panel_group="inspect_results")
                
                # Log keyword filtering statistics
                if "relevant_keyword" in out.columns:
                    total_articles = len(out)
                    articles_with_keywords = out["relevant_keyword"].sum() if "relevant_keyword" in out.columns else 0
                    articles_without_keywords = total_articles - articles_with_keywords
                    keyword_presence_rate = articles_with_keywords / total_articles if total_articles > 0 else 0
                    
                    keyword_stats = {
                        "classify_relevance/keyword_filtering/total_articles": total_articles,
                        "classify_relevance/keyword_filtering/articles_with_keywords": int(articles_with_keywords),
                        "classify_relevance/keyword_filtering/articles_without_keywords": int(articles_without_keywords),
                        "classify_relevance/keyword_filtering/keyword_presence_rate": keyword_presence_rate,
                    }
                    
                    # Average matches per article (for articles with matches)
                    if "keyword_match_count" in out.columns:
                        articles_with_matches = out[out["keyword_match_count"] > 0]
                        if len(articles_with_matches) > 0:
                            avg_matches = articles_with_matches["keyword_match_count"].mean()
                            max_matches = articles_with_matches["keyword_match_count"].max()
                            keyword_stats["classify_relevance/keyword_filtering/avg_matches_per_article"] = avg_matches
                            keyword_stats["classify_relevance/keyword_filtering/max_matches_in_article"] = int(max_matches)
                    
                    context.logger.log_metrics(keyword_stats)
                    print(f"[classify_relevance] Keyword filtering: {articles_with_keywords}/{total_articles} articles ({keyword_presence_rate:.1%}) have keywords", flush=True)
                    
                    # Log top matched keywords if available
                    if "matched_keywords" in out.columns:
                        try:
                            # Flatten all matched keywords and count frequency
                            all_keywords = []
                            for kws in out["matched_keywords"].dropna():
                                if isinstance(kws, list):
                                    all_keywords.extend(kws)
                            
                            if all_keywords:
                                keyword_freq = Counter(all_keywords)
                                top_10_keywords = keyword_freq.most_common(10)
                                
                                # Log as wandb summary (for easy viewing)
                                for i, (keyword, count) in enumerate(top_10_keywords, 1):
                                    context.logger.set_summary(f"classify_relevance/top_keywords/{i:02d}_{keyword}", count)
                                
                                print(f"[classify_relevance] Top 5 keywords: {', '.join([f'{kw}({cnt})' for kw, cnt in top_10_keywords[:5]])}", flush=True)
                        except Exception as e:
                            print(f"Warning: Failed to log top keywords: {e}", flush=True)
                
                # Log relevance statistics
                if "is_relevant" in out.columns:
                    try:
                        total_articles = len(out)
                        relevant_count = out["is_relevant"].sum() if "is_relevant" in out.columns else 0
                        relevance_rate = relevant_count / total_articles if total_articles > 0 else 0
                        context.logger.log_metrics({
                            "classify_relevance/total_articles": total_articles,
                            "classify_relevance/relevant_count": int(relevant_count),
                            "classify_relevance/relevance_rate": relevance_rate,
                        })
                        print(f"[classify_relevance] Relevance: {relevant_count}/{total_articles} articles ({relevance_rate:.1%}) are relevant", flush=True)
                    except Exception as e:
                        print(f"Warning: Failed to log relevance statistics: {e}", flush=True)
            except Exception as e:
                print(f"Warning: Failed to log classify_relevance results or statistics to wandb: {e}", flush=True)
        
        metadata: Dict[str, Any] = {
            "rows": row_count,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)

