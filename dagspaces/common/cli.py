"""One Hydra entry point for every dagspace.

Each dagspace exposes ``python -m dagspaces.<name>.cli``. All of them do the
same three things: load the site environment, print the composed config, then
call the dagspace's run function. Before 2026-08-12 each dagspace wrote that
out. Two of the twelve copies had already lost the comment that gives the
reason for the ``ensure_dotenv()`` call site, which is the drift this module
stops.

A dagspace ``cli.py`` is then nine lines: a one-line module docstring naming
the dagspace, followed by::

    from dagspaces.common.cli import make_cli

    from .orchestrator import run_experiment

    main = make_cli(run_experiment)

    if __name__ == "__main__":
        main()

That module docstring becomes the entry point's help text, so a dagspace
states its purpose one time. Pass ``description=`` only when the module
docstring says more than that — ``eval_all`` does, because it also carries
usage lines.

How the config path stays correct
---------------------------------
``hydra.main(config_path="conf")`` does NOT resolve ``conf`` against the file
that CALLS the decorator. It resolves it against the decorated FUNCTION, in
``hydra._internal.utils.detect_calling_file_or_module_from_task_function``:

* a real package name in ``task_function.__module__`` gives ``pkg://<mdl>``;
* ``None`` or ``"__main__"`` falls back to ``inspect.getfile(task_function)``,
  the file that holds the function's code.

Both routes point at THIS module for a function built here, so a relative
``conf`` would make every dagspace read ``dagspaces/common/conf``. Verified
2026-08-12: that is not a theory, it is what the first draft of this file did.

The package route cannot work either: no dagspace ``conf/`` directory has an
``__init__.py``, so ``pkg://dagspaces.mmlu.conf`` does not resolve. A dagspace
launched with ``-m`` reaches its configs through the FILE route alone, and the
hand-written entry points depended on that.

:func:`make_cli` therefore sidesteps the detection completely. It reads the
caller's ``__file__``, joins ``config_path`` to that directory, and hands
Hydra an ABSOLUTE path. The resulting search path is byte-identical to the
one the hand-written entry points produced::

    | main | file:///share/pierson/matt/UAIR/dagspaces/mmlu/conf |

One behavior improves. The old form resolved through the decorated function's
identity, so ``main()`` failed with "Primary config module not found" if a
caller imported the module under its package name and called it directly. The
absolute path does not care how the module was reached, so that now works.
This was NOT a live submitit defect: the launcher restores the pickled
``Singleton`` state and calls ``run_job(task_function=...)``, so a worker
re-uses the search path the parent computed and never re-runs the detection.

``tests/common/test_cli_factory.py`` asserts the search path, per dagspace.
"""

from __future__ import annotations

import inspect
import os
from collections.abc import Callable
from typing import Any

import hydra
from omegaconf import DictConfig, OmegaConf

from dagspaces.common.stage_utils import ensure_dotenv

__all__ = ["make_cli"]


def _caller_config_dir(caller_globals: dict[str, Any], config_path: str) -> str:
    """Resolve ``config_path`` against the calling module's directory.

    Raises:
        RuntimeError: if the caller has no ``__file__``, or the directory does
            not exist. Composing against the wrong config directory is worse
            than a hard failure here.
    """
    caller_file = caller_globals.get("__file__")
    if not caller_file:
        raise RuntimeError(
            "make_cli() needs the calling module's __file__ to locate its "
            "conf/ directory. Call it from a dagspace cli.py module."
        )
    config_dir = os.path.normpath(
        os.path.join(os.path.dirname(os.path.abspath(caller_file)), config_path)
    )
    if not os.path.isdir(config_dir):
        raise RuntimeError(
            f"make_cli() resolved config_path={config_path!r} to {config_dir!r}, "
            f"which is not a directory."
        )
    return config_dir


def make_cli(
    run: Callable[[DictConfig], Any],
    *,
    description: str | None = None,
    config_path: str = "conf",
    config_name: str = "config",
) -> Callable[..., Any]:
    """Build a dagspace's Hydra entry point.

    Args:
        run: The dagspace run function, e.g. ``orchestrator.run_experiment``.
            It takes the composed config.
        description: The help text for this dagspace. It defaults to the
            calling module's docstring, so a dagspace states its purpose one
            time. Pass a value only when the module docstring says more.
        config_path: The config directory, relative to the CALLING module's
            file. See this module's docstring for the reason it must resolve
            here and not inside Hydra.
        config_name: The root config file, without the ``.yaml`` suffix.

    Returns:
        The decorated ``main`` for the calling module to export.
    """
    caller_globals = inspect.stack()[1].frame.f_globals
    config_dir = _caller_config_dir(caller_globals, config_path)

    def main(cfg: DictConfig) -> None:
        # ensure_dotenv is idempotent. It MUST run inside main() so that it
        # fires in submitit-launched workers as well. A worker imports this
        # module rather than executing __main__, so a __main__-guarded call
        # would skip it, and the downstream ${oc.env:...} interpolations
        # would then fail.
        ensure_dotenv()
        print(OmegaConf.to_yaml(cfg))
        run(cfg)

    main.__doc__ = description or caller_globals.get("__doc__") or ""
    main.__qualname__ = "main"

    decorated = hydra.main(
        version_base=None, config_path=config_dir, config_name=config_name
    )(main)
    # Record the resolved directory on the entry point. It makes the wrong-conf
    # failure mode visible without a Hydra run, and it is what
    # tests/common/test_cli_factory.py asserts on.
    decorated.trawler_config_dir = config_dir
    return decorated
