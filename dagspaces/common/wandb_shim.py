"""Build a dagspace's W&B logger pair from its defaults.

Every dagspace needs its own W&B project name, experiment name, and
full-column stage set, but the wiring around those values is the same
everywhere: subclass :class:`WandbConfig` to bake the defaults into
``from_hydra_config``, then subclass :class:`WandbLogger` so that its
``__init__`` installs that config.

Before 2026-08-12 all eleven dagspaces wrote both subclasses out. That was 808
lines, of which only the default values differed.

Use it like this::

    from dagspaces.common.wandb_logger import (
        collect_compute_metadata,
        ensure_local_tmpdir,
        pipeline_run_id,
    )
    from dagspaces.common.wandb_shim import make_wandb_shim

    ensure_local_tmpdir("confaide")

    WandbConfig, WandbLogger = make_wandb_shim(
        "confaide",
        default_project="confaide",
        default_experiment_name="CONFAIDE",
        full_column_stages=frozenset({"llm_inference"}),
    )

Two module-level details a dagspace must keep
---------------------------------------------
1. ``ensure_local_tmpdir("<dagspace>")`` runs at IMPORT time. It points TMPDIR
   at a writable local path (``/scratch`` or ``/tmp``) instead of a ``/share``
   network mount. Keep the call in the dagspace module, not here, so it stays
   visible and fires once per dagspace import.
2. ``pipeline_run_id`` must be re-exported.
   ``common/orchestrator.make_wandb_logger`` imports the dagspace's
   ``wandb_logger`` module by name and calls ``wl.pipeline_run_id(...)`` on it.
   A shim that drops the re-export breaks every W&B run of that dagspace.

``tests/common/test_wandb_shim.py`` asserts both, per dagspace, along with the
resolved project names — a wrong project name silently splits a benchmark's
dashboard history into two series.
"""

from __future__ import annotations

from typing import Any

from dagspaces.common.wandb_logger import WandbConfig as _WandbConfigBase
from dagspaces.common.wandb_logger import WandbLogger as _WandbLoggerBase

__all__ = ["make_wandb_shim"]


def make_wandb_shim(
    dagspace: str, **defaults: Any
) -> tuple[type[_WandbConfigBase], type[_WandbLoggerBase]]:
    """Return ``(WandbConfig, WandbLogger)`` with this dagspace's defaults.

    Args:
        dagspace: The dagspace name. It becomes the ``dagspace_name`` default,
            which tags every run.
        **defaults: The ``from_hydra_config`` keyword defaults for this
            dagspace — ``default_project``, ``default_experiment_name``,
            ``env_var_prefix``, ``full_column_stages``, and so on. They apply
            with ``setdefault``, so an explicit caller argument still wins.

    Returns:
        The two classes, for the dagspace module to bind and export.
    """
    shim_defaults = dict(defaults)
    shim_defaults.setdefault("dagspace_name", dagspace)

    class WandbConfig(_WandbConfigBase):
        f"""WandbConfig with the {dagspace} defaults."""

        @classmethod
        def from_hydra_config(cls, cfg, **kwargs) -> "WandbConfig":  # type: ignore[override]
            for key, value in shim_defaults.items():
                kwargs.setdefault(key, value)
            return super().from_hydra_config(cfg, **kwargs)

    class WandbLogger(_WandbLoggerBase):
        f"""WandbLogger that installs the {dagspace} WandbConfig."""

        def __init__(
            self,
            cfg,
            stage: str,
            run_id: str | None = None,
            run_config: dict[str, Any] | None = None,
            *,
            wandb_id: str | None = None,
            resume: str | None = None,
        ) -> None:
            super().__init__(
                cfg, stage=stage, run_id=run_id, run_config=run_config,
                wandb_id=wandb_id, resume=resume,
            )
            self.wb_config = WandbConfig.from_hydra_config(cfg)

    WandbConfig.__qualname__ = "WandbConfig"
    WandbLogger.__qualname__ = "WandbLogger"
    WandbConfig.__module__ = f"dagspaces.{dagspace}.wandb_logger"
    WandbLogger.__module__ = f"dagspaces.{dagspace}.wandb_logger"
    # Recorded so a test can assert the defaults without composing a config.
    WandbConfig.trawler_shim_defaults = dict(shim_defaults)
    return WandbConfig, WandbLogger
