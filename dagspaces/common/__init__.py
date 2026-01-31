"""Trawler common module - shared infrastructure for all dagspaces.

This module contains shared code used across all dagspaces:
- config_schema: Pipeline graph configuration dataclasses
- runners/base: Base StageRunner class
"""

from .config_schema import (
    ArtifactSpec,
    SourceSpec,
    OutputSpec,
    PipelineNodeSpec,
    PipelineGraphSpec,
    load_pipeline_graph,
    resolve_output_root,
    iter_topologically,
)

__all__ = [
    "ArtifactSpec",
    "SourceSpec", 
    "OutputSpec",
    "PipelineNodeSpec",
    "PipelineGraphSpec",
    "load_pipeline_graph",
    "resolve_output_root",
    "iter_topologically",
]
