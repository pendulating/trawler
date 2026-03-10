"""Base classes and shared utilities for historical_norms stage runners."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dagspaces.common.runners.base import StageRunner

if TYPE_CHECKING:
    from ..orchestrator import StageExecutionContext, StageResult

__all__ = ["StageRunner"]

