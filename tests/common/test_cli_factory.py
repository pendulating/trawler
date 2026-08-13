"""Guard: every dagspace entry point must find its OWN conf/ directory.

``dagspaces/common/cli.py`` builds all twelve entry points. Hydra resolves a
relative ``config_path`` against the decorated function, which for a shared
factory is ``dagspaces/common/cli.py`` — so a naive version silently points
every dagspace at ``dagspaces/common/conf``. The factory passes an absolute
path instead. These tests pin that.
"""

from __future__ import annotations

import importlib
import os

import pytest

DAGSPACES = [
    "ci_heuristic",
    "cirl",
    "confaide",
    "eval_all",
    "goldcoin_hipaa",
    "grpo_training",
    "historical_norms",
    "mmlu",
    "privacylens",
    "simpleqa_verified",
    "vlm_geoprivacy_aug",
    "vlm_geoprivacy_bench",
]

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _config_path_of(dagspace: str) -> str:
    """The config directory the factory handed Hydra for this dagspace."""
    mod = importlib.import_module(f"dagspaces.{dagspace}.cli")
    return mod.main.trawler_config_dir


@pytest.mark.parametrize("dagspace", DAGSPACES)
def test_entry_point_points_at_its_own_conf(dagspace):
    config_path = _config_path_of(dagspace)

    expected = os.path.join(REPO_ROOT, "dagspaces", dagspace, "conf")
    assert config_path == expected, (
        f"{dagspace} resolves config_path to {config_path!r}. A relative "
        f"config_path in the shared factory resolves against "
        f"dagspaces/common/, which composes the WRONG dagspace's configs."
    )
    assert os.path.isdir(config_path)
    assert os.path.isfile(os.path.join(config_path, "config.yaml"))


@pytest.mark.parametrize("dagspace", DAGSPACES)
def test_entry_point_is_not_the_common_conf(dagspace):
    """The exact failure mode the factory was written to avoid."""
    common_conf = os.path.join(REPO_ROOT, "dagspaces", "common", "conf")
    assert _config_path_of(dagspace) != common_conf


@pytest.mark.parametrize("dagspace", DAGSPACES)
def test_entry_point_exposes_main_and_a_description(dagspace):
    mod = importlib.import_module(f"dagspaces.{dagspace}.cli")
    assert callable(mod.main)
    doc = mod.main.__wrapped__.__doc__
    assert doc and doc.strip(), f"{dagspace} entry point has no help text"


def test_make_cli_refuses_a_missing_conf_directory(tmp_path):
    """A wrong config_path must fail loudly, not compose the wrong configs."""
    from dagspaces.common.cli import make_cli

    with pytest.raises(RuntimeError, match="not a directory"):
        make_cli(lambda cfg: None, config_path="no_such_dir")


def test_make_cli_uses_the_caller_module_docstring():
    """A dagspace states its purpose once, in the module docstring."""
    import types

    from dagspaces.common.cli import make_cli

    fake = types.ModuleType("fake_dagspace_cli")
    fake.__doc__ = "Fake dagspace CLI."
    fake.__file__ = os.path.join(REPO_ROOT, "dagspaces", "mmlu", "cli.py")

    src = (
        "from dagspaces.common.cli import make_cli\n"
        "main = make_cli(lambda cfg: None)\n"
    )
    exec(compile(src, fake.__file__, "exec"), fake.__dict__)
    assert fake.main.__wrapped__.__doc__ == "Fake dagspace CLI."

    # An explicit description wins, for eval_all's multi-line usage docstring.
    src2 = (
        "from dagspaces.common.cli import make_cli\n"
        "main = make_cli(lambda cfg: None, description='Explicit.')\n"
    )
    exec(compile(src2, fake.__file__, "exec"), fake.__dict__)
    assert fake.main.__wrapped__.__doc__ == "Explicit."


def test_every_dagspace_cli_uses_the_factory():
    """No dagspace may go back to a hand-written @hydra.main entry point."""
    for dagspace in DAGSPACES:
        path = os.path.join(REPO_ROOT, "dagspaces", dagspace, "cli.py")
        src = open(path).read()
        assert "make_cli(" in src, f"{dagspace}/cli.py stopped using make_cli"
        assert "@hydra.main" not in src, (
            f"{dagspace}/cli.py declares its own @hydra.main again. Use "
            f"make_cli so the ensure_dotenv rationale stays in one place."
        )
