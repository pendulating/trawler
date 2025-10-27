from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

import pandas as pd
from omegaconf import DictConfig, OmegaConf
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hydra.core.hydra_config import HydraConfig

from .config_schema import (
    PipelineGraphSpec,
    PipelineNodeSpec,
    load_pipeline_graph,
    resolve_output_root,
)
from .stages.classify import run_classification_stage
from .stages.decompose import run_decomposition_stage
from .stages.decompose_nbl import run_decomposition_stage_nbl
from .stages.taxonomy import run_taxonomy_stage
from .stages.topic import run_topic_stage
from .stages.verify import run_verification_stage
from .stages.verify_nbl import run_verification_stage_nbl
from .stages.synthesis import run_synthesis_stage
from .wandb_logger import WandbLogger

try:
    import ray  # type: ignore

    _RAY_AVAILABLE = True
except Exception:  # pragma: no cover - Ray optional dependency
    ray = None  # type: ignore
    _RAY_AVAILABLE = False

try:
    import submitit  # type: ignore
    _SUBMITIT_AVAILABLE = True
except Exception:
    submitit = None  # type: ignore
    _SUBMITIT_AVAILABLE = False


@dataclass
class StageExecutionContext:
    cfg: DictConfig
    node: PipelineNodeSpec
    inputs: Dict[str, str]
    output_paths: Dict[str, str]
    output_dir: str
    output_root: str
    logger: Optional['WandbLogger'] = None


@dataclass
class StageResult:
    outputs: Dict[str, str]
    metadata: Dict[str, Any] = field(default_factory=dict)


class StageRunner:
    stage_name: str

    def run(self, context: StageExecutionContext) -> StageResult:
        raise NotImplementedError


class ArtifactRegistry:
    def __init__(self) -> None:
        self._artifacts: Dict[str, str] = {}

    def register_source(self, name: str, path: str) -> None:
        self._artifacts[name] = path

    def register_outputs(self, node_key: str, outputs: Mapping[str, str]) -> None:
        for out_name, out_path in outputs.items():
            self._artifacts[f"{node_key}.{out_name}"] = out_path

    def resolve(self, ref: str) -> str:
        if ref in self._artifacts:
            return self._artifacts[ref]
        candidate = os.path.abspath(os.path.expanduser(ref))
        if os.path.exists(candidate) or os.path.isabs(ref):
            return candidate
        raise KeyError(f"Unknown artifact reference '{ref}'")

    def resolve_output_path(self, path: str, output_root: str, node_key: str) -> str:
        if not path:
            raise ValueError(f"Node '{node_key}' output path is empty")
        resolved = path
        if not os.path.isabs(resolved):
            resolved = os.path.join(output_root, resolved)
        return os.path.abspath(resolved)


def clone_config(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=False))  # type: ignore[return-value]


def merge_overrides(base_cfg: DictConfig, overrides: Optional[Mapping[str, Any]]) -> DictConfig:
    if not overrides:
        return base_cfg
    # Apply each override using OmegaConf.update to properly handle dot notation
    for key, value in overrides.items():
        OmegaConf.update(base_cfg, key, value, merge=True)
    return base_cfg


def ensure_section(cfg: DictConfig, section: str) -> None:
    if OmegaConf.select(cfg, section) is None:
        OmegaConf.update(cfg, section, {}, merge=True)


def common_parent(paths: Iterable[str]) -> Optional[str]:
    try:
        parents = [os.path.dirname(p) for p in paths]
        if not parents:
            return None
        return os.path.commonpath(parents)
    except Exception:
        return None


def prepare_node_config(base_cfg: DictConfig, node: PipelineNodeSpec, output_dir: str) -> DictConfig:
    cfg_copy = clone_config(base_cfg)
    cfg_copy = merge_overrides(cfg_copy, node.overrides)
    ensure_section(cfg_copy, "runtime")
    OmegaConf.update(cfg_copy, "runtime.stage", node.stage, merge=True)
    OmegaConf.update(cfg_copy, "runtime.output_dir", output_dir, merge=True)
    OmegaConf.update(cfg_copy, "runtime.output_csv", None, merge=True)
    return cfg_copy


def _load_parquet_dataset(parquet_path: str, columns: Mapping[str, str], debug: bool, sample_n: Optional[int]) -> pd.DataFrame:
    if not isinstance(parquet_path, str) or parquet_path.strip() == "":
        raise ValueError("data.parquet_path is required")
    if not os.path.isabs(parquet_path):
        parquet_path = os.path.abspath(parquet_path)
    df = pd.read_parquet(parquet_path)
    col_map = {
        columns.get("article_text", "article_text"): "article_text",
        columns.get("article_path", "article_path"): "article_path",
        columns.get("country", "country"): "country",
        columns.get("year", "year"): "year",
        columns.get("article_id", "article_id"): "article_id",
    }
    present = {src: dst for src, dst in col_map.items() if src in df.columns}
    if present:
        df = df.rename(columns=present)
    if "article_text" not in df.columns and "chunk_text" not in df.columns:
        raise RuntimeError("Parquet missing required text column (article_text) or chunk_text")

    def _safe_str(x: Any) -> str:
        if x is None:
            return ""
        try:
            return "" if (isinstance(x, float) and pd.isna(x)) else str(x).strip()
        except Exception:
            return str(x) if x is not None else ""

    for column in ("article_path", "country", "year", "article_id"):
        if column not in df.columns:
            df[column] = None
        else:
            try:
                df[column] = df[column].apply(_safe_str)
            except Exception:
                pass

    legacy_cols = [
        "name",
        "public_description",
        "subscribers",
        "rule_text",
        "rule_index",
        "total_rules_count",
    ]
    drop_now = [col for col in legacy_cols if col in df.columns]
    if drop_now:
        df = df.drop(columns=drop_now)

    if debug and isinstance(sample_n, int) and sample_n > 0:
        try:
            n = min(int(sample_n), int(len(df)))
        except Exception:
            n = int(sample_n)
        try:
            seed_env = os.environ.get("UAIR_SAMPLE_SEED", "777")
            seed = int(seed_env) if seed_env is not None else 777
        except Exception:
            seed = 777
        try:
            df = df.sample(n=n, random_state=seed).reset_index(drop=True)
        except Exception:
            df = df.head(n)
    return df


def _prepare_streaming_dataset(dataset_path: str, columns: Mapping[str, str], cfg: DictConfig, stage: str) -> tuple[Optional[Any], bool]:
    if not _RAY_AVAILABLE:
        return None, False
    streaming_allowed = stage in {"classify", "taxonomy", "verification"}
    if not streaming_allowed:
        return None, False
    try:
        if not os.path.isabs(dataset_path):
            dataset_path = os.path.abspath(dataset_path)
        if not ray.is_initialized():
            namespace = os.environ.get("RAY_NAMESPACE") or os.environ.get("WANDB_GROUP") or "uair"
            try:
                ray.init(log_to_driver=True, namespace=str(namespace))
            except Exception:
                ray.init(log_to_driver=True)
        ds = ray.data.read_parquet(dataset_path)
        col_map = {
            columns.get("article_text", "article_text"): "article_text",
            columns.get("article_path", "article_path"): "article_path",
            columns.get("country", "country"): "country",
            columns.get("year", "year"): "year",
            columns.get("article_id", "article_id"): "article_id",
        }

        def _ensure_canon(row: Dict[str, Any]) -> Dict[str, Any]:
            out = dict(row)
            for src, dst in col_map.items():
                if dst not in out and src in row:
                    out[dst] = row.get(src)
            return out

        try:
            ds = ds.map(_ensure_canon)
        except Exception:
            pass
        debug = bool(getattr(cfg.runtime, "debug", False))
        sample_n = getattr(cfg.runtime, "sample_n", None)
        if debug and isinstance(sample_n, int) and sample_n > 0:
            try:
                ds = ds.limit(max(1, int(sample_n)))
            except Exception:
                pass
        return ds, True
    except Exception:
        return None, False


def prepare_stage_input(cfg: DictConfig, dataset_path: str, stage: str) -> tuple[Optional[pd.DataFrame], Optional[Any], bool]:
    debug = bool(getattr(cfg.runtime, "debug", False))
    sample_n = getattr(cfg.runtime, "sample_n", None)
    columns = dict(getattr(cfg.data, "columns", {})) if getattr(cfg, "data", None) else {}
    streaming_enabled = bool(getattr(cfg.runtime, "streaming_io", False))
    ds = None
    use_streaming = False
    if streaming_enabled:
        ds, use_streaming = _prepare_streaming_dataset(dataset_path, columns, cfg, stage)
    df = None
    if not use_streaming:
        df = _load_parquet_dataset(dataset_path, columns, debug=debug, sample_n=sample_n)
    return df, ds, use_streaming


def _collect_outputs(context: StageExecutionContext, optional: Mapping[str, bool]) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for key, path in context.output_paths.items():
        if os.path.exists(path):
            resolved[key] = path
        else:
            if optional.get(key, False):
                continue
            raise FileNotFoundError(
                f"Expected output '{key}' for node '{context.node.key}' at '{path}' not found"
            )
    return resolved


class ClassificationRunner(StageRunner):
    stage_name = "classify"

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        # Ensure base stage uses 'relevance' profile (avoid EU/RB vague gating here)
        try:
            OmegaConf.update(cfg, "runtime.classification_profile", "relevance", merge=True)
        except Exception:
            pass
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        out = run_classification_stage(in_obj, cfg)
        
        # Convert Ray Dataset to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()
        
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
            # Scrub any stray too_vague_to_process for base relevance stage
            try:
                if "too_vague_to_process" in out.columns:
                    out["too_vague_to_process"] = False
            except Exception:
                pass
            # Save generic results if requested (profile-agnostic)
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
                # Include profile-specific columns when present
                if any(c in out.columns for c in ("eu_ai_label", "eu_ai_desc", "eu_ai_relevant_text", "eu_ai_reason")):
                    prefer_cols = [
                        "article_id",
                        "too_vague_to_process",
                        "eu_valid_input_count",
                        "eu_ai_label",
                        "eu_ai_desc",
                        "eu_ai_relevant_text",
                        "eu_ai_reason",
                        "classification_mode",
                        "article_path",
                        "country",
                        "year",
                    ]
                else:
                    prefer_cols = [
                        "article_id",
                        "too_vague_to_process",
                        "eu_valid_input_count",
                        "is_relevant",
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
                context.logger.log_table(out, "classify/results", prefer_cols=prefer_cols, panel_group="inspect_results")
                
                # Log keyword filtering statistics
                if "relevant_keyword" in out.columns:
                    total_articles = len(out)
                    articles_with_keywords = out["relevant_keyword"].sum() if "relevant_keyword" in out.columns else 0
                    articles_without_keywords = total_articles - articles_with_keywords
                    keyword_presence_rate = articles_with_keywords / total_articles if total_articles > 0 else 0
                    
                    keyword_stats = {
                        "classify/keyword_filtering/total_articles": total_articles,
                        "classify/keyword_filtering/articles_with_keywords": int(articles_with_keywords),
                        "classify/keyword_filtering/articles_without_keywords": int(articles_without_keywords),
                        "classify/keyword_filtering/keyword_presence_rate": keyword_presence_rate,
                    }
                    
                    # Average matches per article (for articles with matches)
                    if "keyword_match_count" in out.columns:
                        articles_with_matches = out[out["keyword_match_count"] > 0]
                        if len(articles_with_matches) > 0:
                            avg_matches = articles_with_matches["keyword_match_count"].mean()
                            max_matches = articles_with_matches["keyword_match_count"].max()
                            keyword_stats["classify/keyword_filtering/avg_matches_per_article"] = avg_matches
                            keyword_stats["classify/keyword_filtering/max_matches_in_article"] = int(max_matches)
                    
                    context.logger.log_metrics(keyword_stats)
                    print(f"[classify] Keyword filtering: {articles_with_keywords}/{total_articles} articles ({keyword_presence_rate:.1%}) have keywords", flush=True)
                    
                    # Log top matched keywords if available
                    if "matched_keywords" in out.columns:
                        try:
                            # Flatten all matched keywords and count frequency
                            from collections import Counter
                            all_keywords = []
                            for kws in out["matched_keywords"].dropna():
                                if isinstance(kws, list):
                                    all_keywords.extend(kws)
                            
                            if all_keywords:
                                keyword_freq = Counter(all_keywords)
                                top_10_keywords = keyword_freq.most_common(10)
                                
                                # Log as wandb summary (for easy viewing)
                                for i, (keyword, count) in enumerate(top_10_keywords, 1):
                                    context.logger.set_summary(f"classify/top_keywords/{i:02d}_{keyword}", count)
                                
                                print(f"[classify] Top 5 keywords: {', '.join([f'{kw}({cnt})' for kw, cnt in top_10_keywords[:5]])}", flush=True)
                        except Exception as e:
                            print(f"Warning: Failed to log top keywords: {e}", flush=True)
            except Exception as e:
                print(f"Warning: Failed to log classify results or keyword statistics to wandb: {e}", flush=True)
        
        metadata: Dict[str, Any] = {
            "rows": row_count,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


class TaxonomyRunner(StageRunner):
    stage_name = "taxonomy"

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        out = run_taxonomy_stage(in_obj, cfg)
        
        # Convert to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()
        
        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            for output_name, output_path in context.output_paths.items():
                out.to_parquet(output_path, index=False)
        
        # Log results table to wandb (in inspect_results panel group)
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                prefer_cols = ["article_id", "chunk_id", "taxonomy_json", "chunk_text", "article_path"]
                context.logger.log_table(out, "taxonomy/results", prefer_cols=prefer_cols, panel_group="inspect_results")
            except Exception as e:
                print(f"Warning: Failed to log taxonomy results table to wandb: {e}", flush=True)
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


class DecomposeRunner(StageRunner):
    stage_name = "decompose"

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        out = run_decomposition_stage(in_obj, cfg)
        
        # Convert to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()
        
        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            for output_name, output_path in context.output_paths.items():
                out.to_parquet(output_path, index=False)
        
        # Log results table to wandb (in inspect_results panel group)
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                prefer_cols = ["article_id", "chunk_id", "chunk_text", "article_path"]
                context.logger.log_table(out, "decompose/results", prefer_cols=prefer_cols, panel_group="inspect_results")
            except Exception as e:
                print(f"Warning: Failed to log decompose results table to wandb: {e}", flush=True)
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


class DecomposeNBLRunner(StageRunner):
    stage_name = "decompose_nbl"

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        out = run_decomposition_stage_nbl(in_obj, cfg)
        
        # Convert to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()
        
        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            for output_name, output_path in context.output_paths.items():
                out.to_parquet(output_path, index=False)
        
        # Log results table to wandb (in inspect_results panel group)
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                prefer_cols = ["article_id", "chunk_id", "chunk_text", "article_path"]
                context.logger.log_table(out, "decompose_nbl/results", prefer_cols=prefer_cols, panel_group="inspect_results")
            except Exception as e:
                print(f"Warning: Failed to log decompose_nbl results table to wandb: {e}", flush=True)
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


class TopicRunner(StageRunner):
    stage_name = "topic"

    @staticmethod
    def _resolve_topic_path(path: str) -> str:
        if os.path.isdir(path):
            candidates = [
                os.path.join(path, "classify_relevant.parquet"),
                os.path.join(path, "classify_all.parquet"),
                os.path.join(path, "results.parquet"),
            ]
            for cand in candidates:
                if os.path.exists(cand):
                    return cand
        return path

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        resolved_path = self._resolve_topic_path(dataset_path)
        
        # Debug: Check what's being loaded
        print(f"[TopicRunner] Input path: {dataset_path}", flush=True)
        print(f"[TopicRunner] Resolved path: {resolved_path}", flush=True)
        if os.path.exists(resolved_path):
            import pandas as pd
            try:
                df_check = pd.read_parquet(resolved_path)
                print(f"[TopicRunner] File has {len(df_check)} rows", flush=True)
                print(f"[TopicRunner] Columns: {list(df_check.columns)}", flush=True)
            except Exception as e:
                print(f"[TopicRunner] Could not read file for debugging: {e}", flush=True)
        else:
            print(f"[TopicRunner] WARNING: File does not exist!", flush=True)
        
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", resolved_path, merge=True)
        df, ds, use_streaming = prepare_stage_input(cfg, resolved_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        
        # Debug: Check what prepare_stage_input returned
        if df is not None:
            print(f"[TopicRunner] DataFrame loaded with {len(df)} rows", flush=True)
        elif ds is not None:
            print(f"[TopicRunner] Ray Dataset loaded (streaming)", flush=True)
        
        out = run_topic_stage(in_obj, cfg, logger=context.logger)
        
        # Convert to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()
        
        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            for output_name, output_path in context.output_paths.items():
                out.to_parquet(output_path, index=False)
        
        # Log results table and plots to wandb
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                # Log results table (in inspect_results panel group)
                prefer_cols = ["unit_id", "topic_id", "topic_prob", "topic_top_terms", "article_keywords", "plot_x", "plot_y"]
                context.logger.log_table(out, "topic/results", prefer_cols=prefer_cols, panel_group="inspect_results")
            except Exception as e:
                print(f"Warning: Failed to log topic results table to wandb: {e}", flush=True)
            
            try:
                # Log plotly visualization
                from .stages.topic_plot import log_cluster_scatter_plotly_to_wandb
                log_cluster_scatter_plotly_to_wandb(out, context.logger, title="topic_cluster_map")
            except Exception as e:
                print(f"Warning: Failed to log topic plot to wandb: {e}", flush=True)
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


class VerificationRunner(StageRunner):
    stage_name = "verification"

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        out = run_verification_stage(in_obj, cfg)
        
        # Convert to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()
        
        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            for output_name, output_path in context.output_paths.items():
                out.to_parquet(output_path, index=False)
        
        # Log results table to wandb (in inspect_results panel group)
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                prefer_cols = ["article_id", "chunk_id", "chunk_text", "verification_result", "article_path"]
                context.logger.log_table(out, "verification/results", prefer_cols=prefer_cols, panel_group="inspect_results")
            except Exception as e:
                print(f"Warning: Failed to log verification results table to wandb: {e}", flush=True)
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


class VerificationNBLRunner(StageRunner):
    stage_name = "verify_nbl"

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_obj = ds if use_streaming and ds is not None else df
        out = run_verification_stage_nbl(in_obj, cfg)
        
        # Convert to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()
        
        # Prepare optional doc-level aggregation and merge core flags into row-level output
        docs_df = None
        if isinstance(out, pd.DataFrame):
            try:
                import pandas as _pd  # local import to avoid top-level dependency coupling
                # Preferred: read docs_verification written by stage implementation (if present)
                try:
                    results_path = context.output_paths.get("results")
                    if results_path:
                        out_dir = os.path.dirname(results_path)
                        cand_file = os.path.join(out_dir, "docs_verification.parquet")
                        cand_dir = os.path.join(out_dir, "docs_verification")
                        if os.path.exists(cand_file):
                            docs_df = _pd.read_parquet(cand_file)
                        elif os.path.isdir(cand_dir):
                            docs_df = _pd.read_parquet(cand_dir)
                except Exception:
                    docs_df = None
                # Fallback: compute simple doc-level view from the results DataFrame
                if docs_df is None and "article_id" in out.columns:
                    try:
                        def _reduce(df_in: _pd.DataFrame) -> _pd.DataFrame:
                            if df_in.empty:
                                return _pd.DataFrame([])
                            any_tuple = bool(df_in.get("ver_tuples_any_verified", _pd.Series([], dtype=bool)).any())
                            # core tuple verified: require per-field verified flags
                            core_ok = True
                            for f in ("deployment_domain", "deployment_purpose", "deployment_capability"):
                                col = f"ver_tuple_{f}_verified"
                                if col in df_in.columns:
                                    try:
                                        v_any = bool(df_in[col].astype(bool).any())
                                    except Exception:
                                        v_any = False
                                    core_ok = core_ok and v_any
                                else:
                                    core_ok = False
                            return _pd.DataFrame([
                                {
                                    "article_id": df_in.get("article_id", _pd.Series([None])).iloc[0],
                                    "doc_any_component_verified": any_tuple,
                                    "core_tuple_verified": bool(core_ok),
                                }
                            ])
                        docs_df = out.groupby("article_id", dropna=False).apply(_reduce).reset_index(drop=True)
                    except Exception:
                        docs_df = None
                # Merge core flags into row-level frame for downstream gating
                if docs_df is not None and "article_id" in out.columns and "article_id" in docs_df.columns:
                    try:
                        out = out.merge(docs_df[["article_id", "core_tuple_verified"]], on="article_id", how="left")
                    except Exception:
                        pass
            except Exception:
                docs_df = None

        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            for output_name, output_path in context.output_paths.items():
                if output_name == "docs":
                    # Emit doc-level table if available
                    try:
                        if docs_df is not None and len(docs_df):
                            docs_df.to_parquet(output_path, index=False)
                        else:
                            # If not available, synthesize minimal docs view with article_id and core flag
                            if "article_id" in out.columns and "core_tuple_verified" in out.columns:
                                tmp_docs = out[["article_id", "core_tuple_verified"]].drop_duplicates()
                                tmp_docs.to_parquet(output_path, index=False)
                    except Exception:
                        pass
                else:
                    out.to_parquet(output_path, index=False)
        
        # Log verify_nbl results table with all columns (logger handles filtering of heavy/internal)
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                context.logger.log_table(out, "verify_nbl/results", prefer_cols=None, panel_group="inspect_results")
            except Exception as e:
                print(f"Warning: Failed to log verify_nbl results table to wandb: {e}", flush=True)
            # Log document-level aggregation in inspect_results; build from memory if needed
            try:
                # docs_df may have been computed above; if not, try to synthesize a minimal one
                if 'docs_df' not in locals() or docs_df is None:
                    try:
                        import pandas as _pd
                        if "article_id" in out.columns:
                            docs_df = out[["article_id", "core_tuple_verified"]].drop_duplicates()
                    except Exception:
                        docs_df = None
                if docs_df is not None and len(docs_df):
                    context.logger.log_table(docs_df, "verify_nbl/docs", prefer_cols=None, panel_group="inspect_results")
            except Exception as e:
                print(f"Warning: verify_nbl docs logging encountered an error: {e}", flush=True)

            # Plot run-level verification rates by input and doc-level
            try:
                import pandas as _pd
                import matplotlib.pyplot as _plt  # type: ignore

                df_res = out
                df_docs = docs_df if 'docs_df' in locals() else None

                # Helper: presence check for scalars/lists
                def _present_series(s: _pd.Series) -> _pd.Series:
                    try:
                        ss = s.astype(str).str.strip()
                        ssl = ss.str.lower()
                        not_empty = ss != ""
                        not_none = ~ssl.isin(["none", "null"])
                        not_empty_json = ~ss.isin(["[]", "{}"])
                        return not_empty & not_none & not_empty_json
                    except Exception:
                        return _pd.Series([False] * len(s))

                scalar_fields = [
                    "deployment_domain",
                    "deployment_purpose",
                    "deployment_capability",
                    "deployment_space",
                    "identity_of_ai_deployer",
                    "identity_of_ai_subject",
                    "identity_of_ai_developer",
                    "location_of_ai_deployer",
                    "location_of_ai_subject",
                    "date_and_time_of_event",
                ]
                list_fields = [
                    "list_of_harms_that_occurred",
                    "list_of_risks_that_occurred",
                    "list_of_benefits_that_occurred",
                ]

                labels = []
                percents = []
                # Scalars: percent verified among present inputs
                for f in scalar_fields:
                    try:
                        if f in df_res.columns:
                            present = _present_series(df_res[f])
                            ver_col = f"ver_tuple_{f}_verified"
                            if ver_col in df_res.columns:
                                verified = present & df_res[ver_col].astype(bool)
                                present_n = int(present.sum())
                                verified_n = int(verified.sum())
                                pct = (verified_n / present_n * 100.0) if present_n > 0 else None
                                labels.append(f)
                                percents.append(pct)
                    except Exception:
                        pass
                # Lists: percent any verified among present lists
                for f in list_fields:
                    try:
                        if f in df_res.columns:
                            present = _present_series(df_res[f])
                            any_col = f"ver_tuple_{f}_any"
                            if any_col in df_res.columns:
                                verified = present & df_res[any_col].astype(bool)
                                present_n = int(present.sum())
                                verified_n = int(verified.sum())
                                pct = (verified_n / present_n * 100.0) if present_n > 0 else None
                                labels.append(f)
                                percents.append(pct)
                    except Exception:
                        pass

                # Build bar plot for input percentages (filter out None)
                try:
                    labels_plot = []
                    percents_plot = []
                    for l, p in zip(labels, percents):
                        if p is not None:
                            labels_plot.append(l)
                            percents_plot.append(p)
                    if labels_plot:
                        fig, ax = _plt.subplots(figsize=(10, max(3, int(len(labels_plot) * 0.4))))
                        ax.barh(labels_plot, percents_plot, color="#4e79a7")
                        ax.set_xlabel("Percent verified (%)")
                        ax.set_title("Verification rate by input")
                        for i, v in enumerate(percents_plot):
                            ax.text(v + 1, i, f"{v:.1f}%", va='center')
                        _plt.tight_layout()
                        context.logger.log_plot("verify_nbl/percent_verified_by_input", fig)
                except Exception as e:
                    print(f"Warning: failed to log percent_verified_by_input plot: {e}", flush=True)

                # Doc-level percentages
                try:
                    if df_docs is not None and len(df_docs):
                        vals = {}
                        for key in ("doc_any_component_verified", "core_tuple_verified"):
                            if key in df_docs.columns:
                                try:
                                    vals[key] = float(df_docs[key].astype(bool).mean() * 100.0)
                                except Exception:
                                    pass
                        if vals:
                            fig2, ax2 = _plt.subplots(figsize=(6, 3))
                            keys = list(vals.keys())
                            vals_list = [vals[k] for k in keys]
                            ax2.bar(keys, vals_list, color="#59a14f")
                            ax2.set_ylabel("Percent of articles (%)")
                            ax2.set_title("Doc-level verification rates")
                            for i, v in enumerate(vals_list):
                                ax2.text(i, v + 1, f"{v:.1f}%", ha='center')
                            _plt.tight_layout()
                            context.logger.log_plot("verify_nbl/doc_level_percentages", fig2)
                except Exception as e:
                    print(f"Warning: failed to log doc_level_percentages plot: {e}", flush=True)
            except Exception as e:
                print(f"Warning: verify_nbl run-level plots error: {e}", flush=True)
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": bool(use_streaming),
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)

class SynthesisRunner(StageRunner):
    stage_name = "synthesis"

    def run(self, context: StageExecutionContext) -> StageResult:
        # Synthesis requires clusters; articles are optional (enables text-aware join when provided)
        clusters_path = context.inputs.get("clusters")
        articles_path = context.inputs.get("articles")
        
        if not clusters_path:
            raise ValueError(f"Node '{context.node.key}' requires 'clusters' input (topic assignments)")
        
        cfg = context.cfg
        
        # Load both datasets
        try:
            df_clusters = pd.read_parquet(clusters_path)
            print(f"Loaded {len(df_clusters)} rows from clusters: {clusters_path}", flush=True)
        except Exception as e:
            raise ValueError(f"Failed to load clusters from '{clusters_path}': {e}")
        
        df_articles = None
        if articles_path:
            try:
                if os.path.exists(articles_path):
                    df_articles = pd.read_parquet(articles_path)
                    print(f"Loaded {len(df_articles)} rows from articles: {articles_path}", flush=True)
                else:
                    print(f"Articles path not found; proceeding without articles: {articles_path}", flush=True)
            except Exception as e:
                print(f"Warning: Failed to load articles from '{articles_path}': {e}; proceeding without articles.", flush=True)
        
        # Pass both to synthesis stage
        out = run_synthesis_stage(df_clusters, cfg, logger=context.logger, articles_df=df_articles)
        
        # Convert to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()
        
        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            for output_name, output_path in context.output_paths.items():
                out.to_parquet(output_path, index=False)
        
        # Log results table to wandb (in inspect_results panel group)
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                prefer_cols = ["cluster_id", "num_articles", "primary_risk_type", "risk_confidence", "synthesis_summary"]
                context.logger.log_table(out, "synthesis/results", prefer_cols=prefer_cols, panel_group="inspect_results")
            except Exception as e:
                print(f"Warning: Failed to log synthesis results table to wandb: {e}", flush=True)
        
        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": False,  # Synthesis uses dual-input join, not streaming
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


class ClassificationEUActRunner(StageRunner):
    stage_name = "classify_eu_act"

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        # Inject EU AI Act prompt into cfg.prompt for this stage only
        try:
            base_dir = os.path.dirname(__file__)
            prompt_path = os.path.join(base_dir, "conf", "prompt", "eu_ai_act_classification.yaml")
            if os.path.exists(prompt_path):
                eu_prompt_cfg = OmegaConf.load(prompt_path)
                try:
                    sys_p = eu_prompt_cfg.get("system_prompt")
                    usr_p = eu_prompt_cfg.get("prompt_template")
                except Exception:
                    sys_p = None
                    usr_p = None
                ensure_section(cfg, "prompt")
                if sys_p:
                    OmegaConf.update(cfg, "prompt.system_prompt", sys_p, merge=True)
                if usr_p:
                    OmegaConf.update(cfg, "prompt.prompt_template", usr_p, merge=True)
        except Exception:
            pass
        # Always load as pandas for gating; streaming not supported for this custom stage
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_df = df if df is not None else (ds.to_pandas() if hasattr(ds, "to_pandas") else None)
        if in_df is None:
            raise RuntimeError("Failed to load input dataset for classify_eu_act")
        # Gate by core_tuple_verified == True
        try:
            total_rows = len(in_df)
        except Exception:
            total_rows = None
        try:
            if "core_tuple_verified" in in_df.columns:
                mask = in_df["core_tuple_verified"].fillna(False).astype(bool)
            else:
                mask = pd.Series([False] * len(in_df))
            gated_df = in_df[mask].copy()
        except Exception:
            gated_df = in_df.iloc[0:0].copy()
        # Prefer one row per article to avoid duplicate classifications
        try:
            if "article_id" in gated_df.columns and len(gated_df):
                # Keep the first occurrence per article_id
                gated_df = gated_df.sort_values(by=["article_id"]).drop_duplicates(subset=["article_id"], keep="first")
        except Exception:
            pass

        # Run classification with EU AI Act profile (assumed set via overrides)
        out = run_classification_stage(gated_df, cfg)

        # Convert Ray Dataset to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()

        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            if "results" in context.output_paths:
                out.to_parquet(context.output_paths["results"], index=False)

        # Log results
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                prefer_cols = [
                    c for c in [
                        "article_id",
                        "eu_ai_label",
                        "eu_ai_desc",
                        "eu_ai_relevant_text",
                        "eu_ai_reason",
                        "classification_mode",
                        "article_path",
                        "country",
                        "year",
                    ] if c in out.columns
                ]
                context.logger.log_table(out, "classify_eu_act/results", prefer_cols=prefer_cols, panel_group="inspect_results")
                # Simple counts
                try:
                    context.logger.log_metrics({
                        "classify_eu_act/input_rows": int(total_rows or 0),
                        "classify_eu_act/gated_rows": int(len(gated_df)),
                        "classify_eu_act/output_rows": int(len(out)),
                    })
                except Exception:
                    pass
            except Exception as e:
                print(f"Warning: Failed to log classify_eu_act results to wandb: {e}", flush=True)

        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": False,
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)


class ClassificationRisksBenefitsRunner(StageRunner):
    stage_name = "classify_risk_and_benefits"

    def run(self, context: StageExecutionContext) -> StageResult:
        dataset_path = context.inputs.get("dataset")
        if not dataset_path:
            raise ValueError(f"Node '{context.node.key}' requires 'dataset' input")
        cfg = context.cfg
        OmegaConf.update(cfg, "data.parquet_path", dataset_path, merge=True)
        # Ensure proper profile
        try:
            OmegaConf.update(cfg, "runtime.classification_profile", "risks_and_benefits", merge=True)
            OmegaConf.update(cfg, "runtime.use_llm_classify", True, merge=True)
        except Exception:
            pass
        # Inject risks & benefits prompt into cfg.prompt for this stage only
        try:
            base_dir = os.path.dirname(__file__)
            prompt_path = os.path.join(base_dir, "conf", "prompt", "classify_risks_and_benefits.yaml")
            if os.path.exists(prompt_path):
                rb_prompt_cfg = OmegaConf.load(prompt_path)
                try:
                    sys_p = rb_prompt_cfg.get("system_prompt")
                    usr_p = rb_prompt_cfg.get("prompt_template")
                except Exception:
                    sys_p = None
                    usr_p = None
                ensure_section(cfg, "prompt")
                if sys_p:
                    OmegaConf.update(cfg, "prompt.system_prompt", sys_p, merge=True)
                if usr_p:
                    OmegaConf.update(cfg, "prompt.prompt_template", usr_p, merge=True)
        except Exception:
            pass
        df, ds, use_streaming = prepare_stage_input(cfg, dataset_path, self.stage_name)
        in_df = df if df is not None else (ds.to_pandas() if hasattr(ds, "to_pandas") else None)
        if in_df is None:
            raise RuntimeError("Failed to load input dataset for classify_risk_and_benefits")
        # Prefer one row per article to avoid duplicate classifications
        try:
            if "article_id" in in_df.columns and len(in_df):
                in_df = in_df.sort_values(by=["article_id"]).drop_duplicates(subset=["article_id"], keep="first")
        except Exception:
            pass
        out = run_classification_stage(in_df, cfg)

        # Convert Ray Dataset to pandas if needed
        if hasattr(out, "to_pandas"):
            out = out.to_pandas()

        # Save outputs to disk
        if isinstance(out, pd.DataFrame):
            if "results" in context.output_paths:
                out.to_parquet(context.output_paths["results"], index=False)

        # Log results
        if isinstance(out, pd.DataFrame) and context.logger:
            try:
                prefer_cols = [
                    c for c in [
                        "article_id",
                        "risks_benefits_json",
                        "classification_mode",
                        "article_path",
                        "country",
                        "year",
                    ] if c in out.columns
                ]
                context.logger.log_table(out, "classify_risk_and_benefits/results", prefer_cols=prefer_cols, panel_group="inspect_results")
            except Exception as e:
                print(f"Warning: Failed to log classify_risk_and_benefits results to wandb: {e}", flush=True)

        metadata: Dict[str, Any] = {
            "rows": len(out) if isinstance(out, pd.DataFrame) else None,
            "streaming": False,
        }
        outputs = _collect_outputs(
            context,
            {name: spec.optional for name, spec in context.node.outputs.items()},
        )
        return StageResult(outputs=outputs, metadata=metadata)

_STAGE_REGISTRY: Dict[str, StageRunner] = {
    "classify": ClassificationRunner(),
    "classify_eu_act": ClassificationEUActRunner(),
    "classify_risk_and_benefits": ClassificationRisksBenefitsRunner(),
    "decompose": DecomposeRunner(),
    "decompose_nbl": DecomposeNBLRunner(),
    "taxonomy": TaxonomyRunner(),
    "topic": TopicRunner(),
    "verification": VerificationRunner(),
    "verify_nbl": VerificationNBLRunner(),
    "synthesis": SynthesisRunner(),
}


def _ensure_output_dirs(paths: Iterable[str]) -> None:
    for path in paths:
        os.makedirs(os.path.dirname(path), exist_ok=True)


def _node_optional_outputs(node: PipelineNodeSpec) -> Dict[str, bool]:
    return {name: spec.optional for name, spec in node.outputs.items()}


def _node_output_paths(node: PipelineNodeSpec, registry: ArtifactRegistry, output_root: str) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for out_name, spec in node.outputs.items():
        resolved[out_name] = registry.resolve_output_path(spec.path, output_root, node.key)
    _ensure_output_dirs(resolved.values())
    return resolved


def _node_inputs(node: PipelineNodeSpec, registry: ArtifactRegistry) -> Dict[str, str]:
    resolved: Dict[str, str] = {}
    for alias, ref in node.inputs.items():
        resolved[alias] = registry.resolve(ref)
    return resolved


def _print_status(payload: Dict[str, Any]) -> None:
    try:
        print(json.dumps(payload, indent=2))
    except Exception:
        pass


def _load_launcher_config(cfg: DictConfig, launcher_name: str) -> Optional[DictConfig]:
    """Load a launcher configuration from Hydra config."""
    try:
        # Find the config path - use the location of this file as reference
        config_path = os.path.join(os.path.dirname(__file__), "conf")
        
        if not os.path.exists(config_path):
            # Try to get from hydra runtime
            hydra_cfg = getattr(cfg, "hydra", None)
            if hydra_cfg:
                runtime_cfg = getattr(hydra_cfg, "runtime", None)
                if runtime_cfg:
                    sources = getattr(runtime_cfg, "config_sources", [])
                    for source in sources:
                        if hasattr(source, "provider") and source.provider == "main":
                            config_path = source.path
                            break
        
        if not config_path or not os.path.exists(config_path):
            raise ValueError(f"Could not find config directory")
            
        launcher_file = os.path.join(config_path, "hydra", "launcher", f"{launcher_name}.yaml")
        if not os.path.exists(launcher_file):
            raise ValueError(f"Launcher config file not found: {launcher_file}")
        
        # Load the launcher config
        launcher_cfg = OmegaConf.load(launcher_file)
        # Resolve interpolations with the main config as context
        launcher_cfg = OmegaConf.merge({"runtime": cfg.get("runtime", {})}, launcher_cfg)
        return launcher_cfg
    except Exception as e:
        raise ValueError(f"Failed to load launcher config '{launcher_name}': {e}") from e


def _create_submitit_executor(launcher_cfg: DictConfig, job_name: str, log_folder: str) -> Any:
    """Create a submitit executor from launcher configuration."""
    if not _SUBMITIT_AVAILABLE or submitit is None:
        raise RuntimeError("submitit is not available but is required for SLURM job submission")
    
    executor = submitit.AutoExecutor(folder=log_folder)
    
    # Map launcher config to submitit parameters
    executor.update_parameters(
        timeout_min=int(launcher_cfg.get("timeout_min", 120)),
        slurm_partition=str(launcher_cfg.get("partition", "pierson")),
        slurm_mem=f"{int(launcher_cfg.get('mem_gb', 8))}GB",
        slurm_cpus_per_task=int(launcher_cfg.get("cpus_per_task", 2)),
        slurm_gpus_per_node=int(launcher_cfg.get("gpus_per_node", 0)),
        slurm_nodes=int(launcher_cfg.get("nodes", 1)),
        slurm_tasks_per_node=int(launcher_cfg.get("tasks_per_node", 1)),
        slurm_array_parallelism=int(launcher_cfg.get("array_parallelism", 1)),
        name=job_name,
        slurm_additional_parameters=launcher_cfg.get("additional_parameters", {}),
        slurm_setup=launcher_cfg.get("setup", []),
    )
    
    return executor


def execute_stage_job(context_data: Dict[str, Any]) -> Dict[str, Any]:
    """Execute a single stage - designed to be submitted as a SLURM job."""
    # Reconstruct context from serialized data
    from omegaconf import OmegaConf
    cfg = OmegaConf.create(context_data["cfg"])
    node_dict = context_data["node"]
    
    # Reconstruct PipelineNodeSpec
    from .config_schema import PipelineNodeSpec, OutputSpec
    outputs = {}
    for out_key, out_val in node_dict.get("outputs", {}).items():
        outputs[out_key] = OutputSpec.from_config(out_key, out_val)
    
    node = PipelineNodeSpec(
        key=node_dict["key"],
        stage=node_dict["stage"],
        depends_on=node_dict.get("depends_on", []),
        inputs=node_dict.get("inputs", {}),
        outputs=outputs,
        overrides=node_dict.get("overrides", {}),
        launcher=node_dict.get("launcher"),
        parallel_group=node_dict.get("parallel_group"),
        max_attempts=node_dict.get("max_attempts", 1),
        retry_backoff_s=node_dict.get("retry_backoff_s", 0.0),
        wandb_suffix=node_dict.get("wandb_suffix"),
    )
    
    context = StageExecutionContext(
        cfg=cfg,
        node=node,
        inputs=context_data["inputs"],
        output_paths=context_data["output_paths"],
        output_dir=context_data["output_dir"],
        output_root=context_data["output_root"],
    )
    
    # Get the stage runner
    stage_registry = dict(_STAGE_REGISTRY)
    runner = stage_registry.get(node.stage)
    if runner is None:
        raise ValueError(f"No runner registered for stage '{node.stage}' (node '{node.key}')")
    
    # Execute stage with wandb logging context
    wandb_run_id = node.wandb_suffix or node.key
    run_config = {
        "node": node.key,
        "stage": node.stage,
        "inputs": list(context.inputs.keys()),
        "outputs": list(context.output_paths.keys()),
    }
    
    with WandbLogger(cfg, stage=node.stage, run_id=wandb_run_id, run_config=run_config) as logger:
        try:
            # Update context with logger
            context.logger = logger
            
            # Execute the stage
            _print_status({"node": node.key, "stage": node.stage, "status": "running", "inputs": context.inputs})
            stage_start = time.time()
            
            result = runner.run(context)
            
            # Log completion metrics
            duration_s = time.time() - stage_start
            try:
                logger.set_summary(f"{node.stage}/status", "completed")
            except Exception:
                pass
            logger.log_metrics({
                f"{node.stage}/duration_s": duration_s,
                f"{node.stage}/rows_processed": result.metadata.get("rows", 0),
            })
            
            return {
                "outputs": result.outputs,
                "metadata": result.metadata,
            }
        except Exception as e:
            # Log failure
            try:
                logger.set_summary(f"{node.stage}/status", "failed")
                logger.set_summary(f"{node.stage}/error", str(e))
            except Exception:
                pass
            raise


def run_experiment(cfg: DictConfig) -> None:
    # Execute entire pipeline with wandb logging context
    with WandbLogger(cfg, stage="orchestrator", run_id="monitor", run_config={"type": "pipeline"}) as logger:
        try:
            # Get the parent/monitor group ID to pass to child jobs
            # This ensures all stages in one pipeline run are grouped together
            parent_group = logger.wb_config.group if logger.wb_config else None
            if parent_group:
                # Set in environment so child jobs can inherit it
                os.environ["WANDB_GROUP"] = parent_group
                print(f"[orchestrator] Setting WANDB_GROUP={parent_group} for child stages", flush=True)
            
            graph_spec: PipelineGraphSpec = load_pipeline_graph(cfg)
            output_root = resolve_output_root(graph_spec, cfg)
            os.makedirs(output_root, exist_ok=True)
            registry = ArtifactRegistry()
            for source_key, source in graph_spec.sources.items():
                path = source.path
                if not os.path.isabs(path):
                    path = os.path.abspath(os.path.expanduser(path))
                registry.register_source(source_key, path)
            manifest: Dict[str, Any] = {
                "output_root": output_root,
                "nodes": {},
            }
            stage_registry = dict(_STAGE_REGISTRY)
            ordered_nodes = graph_spec.topological_order()
            pipeline_start = time.time()
            
            # Log pipeline structure to wandb: numeric to charts; structure to config
            logger.log_metrics({
                "orchestrator/total_nodes": len(ordered_nodes),
            })
            try:
                logger.set_config({
                    "orchestrator": {
                        "node_order": ordered_nodes,
                        "total_nodes": len(ordered_nodes),
                    }
                })
            except Exception:
                pass
            
            for node_key in ordered_nodes:
                node = graph_spec.nodes[node_key]
                runner = stage_registry.get(node.stage)
                if runner is None:
                    raise ValueError(f"No runner registered for stage '{node.stage}' (node '{node.key}')")
                inputs = _node_inputs(node, registry)
                output_paths = _node_output_paths(node, registry, output_root)
                output_dir = common_parent(output_paths.values())
                if not output_dir:
                    output_dir = os.path.join(output_root, node.key)
                os.makedirs(output_dir, exist_ok=True)
                node_cfg = prepare_node_config(cfg, node, output_dir)
                context = StageExecutionContext(
                    cfg=node_cfg,
                    node=node,
                    inputs=inputs,
                    output_paths=output_paths,
                    output_dir=output_dir,
                    output_root=output_root,
                )
                
                node_start = time.time()
                
                # Check if this node should be launched as a separate SLURM job
                if node.launcher:
                    _print_status({"node": node.key, "stage": node.stage, "status": "submitting", "launcher": node.launcher, "inputs": inputs})
                    try:
                        launcher_cfg = _load_launcher_config(cfg, node.launcher)
                    except ValueError as e:
                        raise ValueError(f"Could not load launcher config '{node.launcher}' for node '{node.key}': {e}") from e
                    
                    # Create submitit executor - store logs in the Hydra multirun directory
                    # Structure: multirun/YYYY-MM-DD/HH-MM-SS/0/.slurm_jobs/STAGE_NAME/
                    log_folder = None
                    try:
                        # Priority 1: Use HydraConfig to get runtime output directory
                        hydra_cfg = HydraConfig.get()
                        if hydra_cfg and hydra_cfg.runtime and hydra_cfg.runtime.output_dir:
                            hydra_output_dir = hydra_cfg.runtime.output_dir
                            log_folder = os.path.join(hydra_output_dir, ".slurm_jobs", node.key)
                            _print_status({"debug": "using_hydra_output_dir", "log_folder": log_folder})
                    except Exception as e:
                        _print_status({"debug": "hydra_config_error", "error": str(e)})
                    
                    # Priority 2: Fall back to output_root
                    if not log_folder:
                        log_folder = os.path.join(output_root, ".slurm_jobs", node.key)
                        _print_status({"debug": "using_output_root_fallback", "log_folder": log_folder, "output_root": output_root})
                    
                    log_folder = os.path.abspath(log_folder)
                    os.makedirs(log_folder, exist_ok=True)
                    job_name = f"UAIR-{node.key}"
                    executor = _create_submitit_executor(launcher_cfg, job_name, log_folder)
                    
                    # Ensure child job uses parent's W&B group for proper grouping
                    # Submitit doesn't auto-inherit env vars, so we need to explicitly set them
                    if parent_group:
                        # Method 1: Set environment variable on executor
                        # This ensures it's available in the SLURM job's environment
                        try:
                            # Get current setup commands and prepend WANDB_GROUP export
                            current_setup = list(launcher_cfg.get("setup", []))
                            # Insert explicit WANDB_GROUP export at the beginning (after shebang/source commands)
                            # Find insertion point (after source commands)
                            insert_idx = 0
                            for i, cmd in enumerate(current_setup):
                                if "source" in cmd or "export HYDRA_FULL_ERROR" in cmd:
                                    insert_idx = i + 1
                            # Insert WANDB_GROUP export
                            wandb_group_export = f"export WANDB_GROUP={parent_group}"
                            if wandb_group_export not in current_setup:
                                current_setup.insert(insert_idx, wandb_group_export)
                                executor.update_parameters(slurm_setup=current_setup)
                                _print_status({"debug": "injected_wandb_group", "group": parent_group, "node": node.key})
                        except Exception as e:
                            _print_status({"debug": "failed_to_inject_wandb_group", "error": str(e)})
                    
                    # Prepare serializable context data
                    context_data = {
                        "cfg": OmegaConf.to_container(node_cfg, resolve=True),
                        "node": {
                            "key": node.key,
                            "stage": node.stage,
                            "depends_on": node.depends_on,
                            "inputs": node.inputs,
                            "outputs": {k: {"path": v.path, "type": v.type, "optional": v.optional} for k, v in node.outputs.items()},
                            "overrides": node.overrides,
                            "launcher": node.launcher,
                            "parallel_group": node.parallel_group,
                            "max_attempts": node.max_attempts,
                            "retry_backoff_s": node.retry_backoff_s,
                            "wandb_suffix": node.wandb_suffix,
                        },
                        "inputs": inputs,
                        "output_paths": output_paths,
                        "output_dir": output_dir,
                        "output_root": output_root,
                    }
                    
                    # Submit the job
                    job = executor.submit(execute_stage_job, context_data)
                    _print_status({"node": node.key, "stage": node.stage, "status": "submitted", "job_id": job.job_id})
                    
                    # Wait for the job to complete
                    try:
                        job_result = job.result()  # This blocks until the job completes
                        result = StageResult(
                            outputs=job_result["outputs"],
                            metadata=job_result["metadata"],
                        )
                    except Exception as exc:
                        _print_status({"node": node.key, "stage": node.stage, "status": "failed", "job_id": job.job_id, "error": str(exc)})
                        raise
                else:
                    # Run locally in the current process
                    _print_status({"node": node.key, "stage": node.stage, "status": "running", "inputs": inputs})
                    try:
                        result = runner.run(context)
                    except Exception as exc:
                        _print_status({"node": node.key, "stage": node.stage, "status": "failed", "error": str(exc)})
                        raise
                
                registry.register_outputs(node.key, result.outputs)
                duration = time.time() - node_start
                manifest["nodes"][node.key] = {
                    "stage": node.stage,
                    "inputs": inputs,
                    "outputs": result.outputs,
                    "metadata": result.metadata,
                    "duration_s": duration,
                }
                _print_status({
                    "node": node.key,
                    "stage": node.stage,
                    "status": "completed",
                    "duration_s": round(duration, 3),
                    "outputs": result.outputs,
                })
            
            manifest_path = os.path.join(output_root, "pipeline_manifest.json")
            try:
                with open(manifest_path, "w", encoding="utf-8") as fh:
                    json.dump(manifest, fh, indent=2)
            except Exception:
                pass
            total_duration = time.time() - pipeline_start
            
            # Log final pipeline metrics to wandb
            try:
                logger.set_summary("orchestrator/status", "completed")
            except Exception:
                pass
            logger.log_metrics({
                "orchestrator/total_duration_s": round(total_duration, 3),
                "orchestrator/nodes_completed": len(manifest["nodes"]),
            })
            
            _print_status({
                "pipeline": {
                    "output_root": output_root,
                    "nodes": ordered_nodes,
                    "duration_s": round(total_duration, 3),
                    "manifest": manifest_path,
                }
            })
        except Exception as e:
            try:
                logger.set_summary("orchestrator/status", "failed")
                logger.set_summary("orchestrator/error", str(e))
            except Exception:
                pass
            raise
