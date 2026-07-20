"""Trawler pipeline configuration schema.

This module defines the dataclasses and utilities for parsing pipeline
configuration from Hydra YAML configs into typed Python objects.

Key classes:
- PipelineGraphSpec: Top-level pipeline definition with sources and nodes
- PipelineNodeSpec: Individual stage node with inputs, outputs, dependencies
- ArtifactSpec/SourceSpec/OutputSpec: Data artifact specifications
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from omegaconf import DictConfig, OmegaConf


def _to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, DictConfig):
        return OmegaConf.to_container(obj, resolve=True)  # type: ignore[return-value]
    if isinstance(obj, dict):
        return dict(obj)
    raise TypeError(f"Unsupported mapping type: {type(obj)!r}")


def _infer_artifact_type(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in {".parquet"}:
        return "parquet"
    if ext in {".csv"}:
        return "csv"
    if ext in {".json"}:
        return "json"
    if ext in {".ndjson"}:
        return "ndjson"
    if ext in {".txt"}:
        return "text"
    if ext in {".yaml", ".yml"}:
        return "yaml"
    if ext in {".log"}:
        return "log"
    if ext in {".pb", ".bin"}:
        return "binary"
    return "dir" if (not ext or ext == "") else ext.lstrip(".")


@dataclass
class ArtifactSpec:
    key: str
    type: str
    path: str
    optional: bool = False

    @classmethod
    def from_config(cls, key: str, value: Any) -> ArtifactSpec:
        if isinstance(value, str):
            artifact_path = value
            art_type = _infer_artifact_type(artifact_path)
            return cls(key=key, type=art_type, path=artifact_path, optional=False)
        data = _to_dict(value)
        if "path" not in data:
            raise ValueError(f"Artifact '{key}' is missing required 'path'")
        path = str(data["path"])
        art_type = str(data.get("type") or _infer_artifact_type(path))
        optional = bool(data.get("optional", False))
        return cls(key=key, type=art_type, path=path, optional=optional)


@dataclass
class SourceSpec(ArtifactSpec):
    pass


@dataclass
class OutputSpec(ArtifactSpec):
    pass


@dataclass
class PipelineNodeSpec:
    key: str
    stage: str
    depends_on: list[str] = field(default_factory=list)
    inputs: dict[str, str] = field(default_factory=dict)
    outputs: dict[str, OutputSpec] = field(default_factory=dict)
    overrides: dict[str, Any] = field(default_factory=dict)
    launcher: str | None = None
    parallel_group: str | None = None
    max_attempts: int = 1
    retry_backoff_s: float = 0.0
    wandb_suffix: str | None = None

    @classmethod
    def from_config(cls, key: str, value: Any) -> PipelineNodeSpec:
        data = _to_dict(value)
        stage = str(data.get("stage")) if data.get("stage") is not None else None
        if not stage:
            raise ValueError(f"Pipeline node '{key}' must define a 'stage'")
        depends_on_val = data.get("depends_on", [])
        if isinstance(depends_on_val, str):
            depends_on = [depends_on_val]
        else:
            depends_on = [str(dep) for dep in depends_on_val]
        inputs_val = data.get("inputs", {})
        inputs = {str(k): str(v) for k, v in _to_dict(inputs_val).items()} if inputs_val else {}
        outputs_val = data.get("outputs", {})
        outputs: dict[str, OutputSpec] = {}
        for out_key, out_value in _to_dict(outputs_val).items():
            outputs[str(out_key)] = OutputSpec.from_config(str(out_key), out_value)
        overrides_val = data.get("overrides", {})
        overrides = _to_dict(overrides_val) if overrides_val else {}
        launcher = data.get("launcher")
        parallel_group = data.get("parallel_group")
        max_attempts = int(data.get("max_attempts", 1) or 1)
        retry_backoff_s = float(data.get("retry_backoff_s", 0.0) or 0.0)
        wandb_suffix = data.get("wandb_suffix")
        return cls(
            key=key,
            stage=str(stage),
            depends_on=depends_on,
            inputs=inputs,
            outputs=outputs,
            overrides=overrides,
            launcher=str(launcher) if launcher else None,
            parallel_group=str(parallel_group) if parallel_group else None,
            max_attempts=max_attempts,
            retry_backoff_s=retry_backoff_s,
            wandb_suffix=str(wandb_suffix) if wandb_suffix else None,
        )


@dataclass
class PipelineGraphSpec:
    sources: dict[str, SourceSpec] = field(default_factory=dict)
    nodes: dict[str, PipelineNodeSpec] = field(default_factory=dict)
    output_root: str | None = None
    allow_partial: bool = False

    def topological_order(self) -> list[str]:
        indegree: dict[str, int] = {node_id: 0 for node_id in self.nodes.keys()}
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    raise ValueError(f"Node '{node.key}' depends on unknown node '{dep}'")
                indegree[node.key] += 1
        ready = [node_id for node_id, degree in indegree.items() if degree == 0]
        ordered: list[str] = []
        while ready:
            current = ready.pop(0)
            ordered.append(current)
            for node in self.nodes.values():
                if current in node.depends_on:
                    indegree[node.key] -= 1
                    if indegree[node.key] == 0:
                        ready.append(node.key)
        if len(ordered) != len(self.nodes):
            missing = set(self.nodes.keys()) - set(ordered)
            raise ValueError(f"Cycle detected in pipeline graph; unresolved nodes: {sorted(missing)}")
        return ordered


def load_pipeline_graph(cfg: DictConfig) -> PipelineGraphSpec:
    if "pipeline" not in cfg:
        raise ValueError("Configuration is missing required 'pipeline' section")
    pipeline_section = cfg.pipeline
    sources_cfg = getattr(pipeline_section, "sources", {})
    sources: dict[str, SourceSpec] = {}
    for src_key, src_val in _to_dict(sources_cfg).items():
        sources[src_key] = SourceSpec.from_config(src_key, src_val)
    graph_cfg = getattr(pipeline_section, "graph", None)
    if graph_cfg is None:
        raise ValueError("'pipeline.graph' must be defined in the configuration")
    nodes_cfg = getattr(graph_cfg, "nodes", None)
    if nodes_cfg is None:
        raise ValueError("'pipeline.graph.nodes' must be defined in the configuration")
    nodes: dict[str, PipelineNodeSpec] = {}
    for node_key, node_val in _to_dict(nodes_cfg).items():
        nodes[node_key] = PipelineNodeSpec.from_config(node_key, node_val)
    output_root = getattr(pipeline_section, "output_root", None)
    allow_partial = bool(getattr(pipeline_section, "allow_partial", False))
    return PipelineGraphSpec(
        sources=sources,
        nodes=nodes,
        output_root=str(output_root) if output_root else None,
        allow_partial=allow_partial,
    )


def resolve_output_root(graph_spec: PipelineGraphSpec, cfg: DictConfig) -> str:
    root = graph_spec.output_root
    if root:
        resolved = os.path.abspath(os.path.expanduser(root))
    else:
        # Fallback to runtime.output_root or hydra.run.dir if provided
        runtime_root = getattr(getattr(cfg, "runtime", object()), "output_root", None)
        if runtime_root:
            resolved = os.path.abspath(os.path.expanduser(str(runtime_root)))
        else:
            hydra_cfg = getattr(cfg, "hydra", None)
            try:
                hydra_run_dir = getattr(getattr(hydra_cfg, "run", object()), "dir", None)
            except Exception:
                hydra_run_dir = None
            if hydra_run_dir:
                resolved = os.path.abspath(os.path.expanduser(str(hydra_run_dir)))
            else:
                resolved = os.path.abspath(os.getcwd())
    _assert_unique_in_multirun(resolved)
    return resolved


def _assert_unique_in_multirun(output_root: str) -> None:
    """Refuse to start if a sweep job's output_root is shared across jobs.

    Reason: prior to 2026-04-28 every pipeline yaml used
    ``${hydra:run.dir}/<name>`` for ``pipeline.output_root``. The
    ``${hydra:run.dir}`` resolver returns ``hydra.run.dir`` (the
    *run-mode* template), which is identical for every job in a sweep
    (same ``now()`` timestamp, no per-job subdir). All sweep jobs
    therefore resolved to the same path and raced/overwrote each
    other's outputs — wasting an entire 16-job SFT ablation that took
    ~75 GPU-hours to produce.

    How to apply: in MULTIRUN mode, ``hydra.runtime.output_dir`` is the
    only Hydra-provided path that includes the sweep subdir. If the
    resolved ``output_root`` is not nested under that path, this is the
    same bug recurring — fail fast with a pointer at the YAML fix
    rather than letting compute silently corrupt itself.
    """
    try:
        from hydra.core.hydra_config import HydraConfig
        from hydra.types import RunMode
        hc = HydraConfig.get()
        if hc.mode != RunMode.MULTIRUN:
            return
        runtime_dir = os.path.abspath(str(hc.runtime.output_dir))
    except Exception:
        # Hydra not initialized (e.g. unit tests) — nothing to check.
        return
    try:
        common = os.path.commonpath([output_root, runtime_dir])
    except ValueError:
        # commonpath raises on mixed absolute/relative; treat as mismatch.
        common = ""
    if common != runtime_dir:
        raise RuntimeError(
            "pipeline.output_root collision risk in MULTIRUN mode.\n"
            f"  resolved output_root:  {output_root}\n"
            f"  hydra.runtime.output_dir: {runtime_dir}\n"
            "Every sweep job would write to the same output_root, "
            "overwriting each other's checkpoints/outputs.\n"
            "Fix: in your pipeline yaml, use "
            "${hydra:runtime.output_dir}/<name> (NOT ${hydra:run.dir}/<name>) "
            "for pipeline.output_root."
        )


def iter_topologically(nodes: dict[str, PipelineNodeSpec]) -> Iterable[PipelineNodeSpec]:
    graph = PipelineGraphSpec(nodes=nodes)
    for node_id in graph.topological_order():
        yield nodes[node_id]
