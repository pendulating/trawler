"""Tests for ``dagspaces.common.config_schema.resolve_output_root``.

The 2026-04-25 SFT pair-ablation sweep wasted 16 fully-trained checkpoints
because every job in the sweep resolved ``pipeline.output_root`` to the
same path (``${hydra:run.dir}/sft_only`` returns the run-mode template,
which is identical across sweep jobs — no per-job subdir). All 16 jobs
serialized writes into the same checkpoint directory and overwrote each
other.

Coverage:

- single-run mode + valid path → returns the path unchanged
- MULTIRUN + output_root nested under runtime.output_dir → returns it
- MULTIRUN + output_root NOT nested → raises with explicit fix pointer
- HydraConfig not initialized (e.g. unit tests / library use) → no-op
"""

from __future__ import annotations

import os
from unittest.mock import MagicMock, patch

import pytest
from omegaconf import OmegaConf

from dagspaces.common.config_schema import (
    PipelineGraphSpec,
    resolve_output_root,
)


def _spec(output_root: str) -> PipelineGraphSpec:
    return PipelineGraphSpec(sources={}, nodes={}, output_root=output_root)


@pytest.fixture
def cfg():
    return OmegaConf.create({"runtime": {"output_root": None}})


class TestNoHydra:
    """When HydraConfig isn't initialized (unit tests), guard is a no-op."""

    def test_no_hydra_returns_path(self, cfg):
        spec = _spec("/tmp/anywhere/foo")
        assert resolve_output_root(spec, cfg) == "/tmp/anywhere/foo"


class TestRunMode:
    def test_run_mode_passes_through(self, cfg):
        from hydra.types import RunMode
        with patch("dagspaces.common.config_schema.HydraConfig", create=True), \
             patch("hydra.core.hydra_config.HydraConfig.get") as mock_get:
            mock_get.return_value = MagicMock(
                mode=RunMode.RUN,
                runtime=MagicMock(output_dir="/tmp/run/foo"),
            )
            spec = _spec("/tmp/run/foo/sft_only")
            assert resolve_output_root(spec, cfg) == "/tmp/run/foo/sft_only"


class TestMultirunMode:
    def test_nested_under_runtime_passes(self, cfg):
        """Sweep job whose output_root is correctly under runtime.output_dir."""
        from hydra.types import RunMode
        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_get:
            mock_get.return_value = MagicMock(
                mode=RunMode.MULTIRUN,
                runtime=MagicMock(
                    output_dir="/tmp/multirun/exp/HHMM/ctx-True_appr-True"
                ),
            )
            spec = _spec(
                "/tmp/multirun/exp/HHMM/ctx-True_appr-True/sft_only"
            )
            assert resolve_output_root(spec, cfg).endswith(
                "ctx-True_appr-True/sft_only"
            )

    def test_shared_path_raises(self, cfg):
        """The 2026-04-25 bug: output_root resolves above the sweep subdir,
        so every sweep job would race on the same path."""
        from hydra.types import RunMode
        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_get:
            mock_get.return_value = MagicMock(
                mode=RunMode.MULTIRUN,
                runtime=MagicMock(
                    output_dir="/tmp/multirun/exp/HHMM/ctx-True_appr-True"
                ),
            )
            spec = _spec("/tmp/multirun/exp/HHMM/sft_only")  # <-- no subdir
            with pytest.raises(RuntimeError, match="collision risk in MULTIRUN"):
                resolve_output_root(spec, cfg)

    def test_unrelated_path_raises(self, cfg):
        """Hardcoded absolute path that ignores runtime.output_dir entirely."""
        from hydra.types import RunMode
        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_get:
            mock_get.return_value = MagicMock(
                mode=RunMode.MULTIRUN,
                runtime=MagicMock(
                    output_dir="/tmp/multirun/exp/HHMM/ctx-True_appr-True"
                ),
            )
            spec = _spec("/share/checkpoints/sft")
            with pytest.raises(RuntimeError, match="collision risk in MULTIRUN"):
                resolve_output_root(spec, cfg)

    def test_error_message_points_at_yaml_fix(self, cfg):
        from hydra.types import RunMode
        with patch("hydra.core.hydra_config.HydraConfig.get") as mock_get:
            mock_get.return_value = MagicMock(
                mode=RunMode.MULTIRUN,
                runtime=MagicMock(output_dir="/tmp/multirun/exp/HHMM/sub"),
            )
            spec = _spec("/tmp/multirun/exp/HHMM/sft_only")
            with pytest.raises(RuntimeError) as exc:
                resolve_output_root(spec, cfg)
            msg = str(exc.value)
            assert "${hydra:runtime.output_dir}" in msg
            assert "${hydra:run.dir}" in msg  # tells you what NOT to use
