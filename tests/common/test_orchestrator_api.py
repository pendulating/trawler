"""Guard: production code must not import private names from common/orchestrator.

Until 2026-08-12, twelve helpers in ``dagspaces/common/orchestrator.py``
carried a leading underscore while three dagspaces and the shared runner base
imported them anyway:

    collect_outputs  save_stage_outputs  node_inputs  node_output_paths
    create_submitit_executor  load_launcher_config  submit_slurm_job
    sanitize_cuda_visible_devices  NoOpLogger  safe_log_table
    print_status  inject_prompt_from_file

A private marker nothing honours is worse than no marker: it tells a reader
that a function is safe to change when four other modules depend on its
signature. They are public names now, and ``__all__`` records the surface.

Tests may still reach into internals — that is a deliberate exception.
"""

from __future__ import annotations

import ast
import glob

import pytest

import dagspaces.common.orchestrator as orch

MODULE = "dagspaces.common.orchestrator"

# Private helpers that only tests touch. Keeping them private is correct.
TEST_ONLY_PRIVATES = {"_mirror_stage_metrics", "_rebuild_node"}

PRODUCTION_GLOBS = ["dagspaces/**/*.py", "scripts/**/*.py"]


def _production_files():
    out = []
    for pat in PRODUCTION_GLOBS:
        out.extend(glob.glob(pat, recursive=True))
    return sorted(f for f in out if f != "dagspaces/common/orchestrator.py")


def test_no_production_module_imports_a_private_orchestrator_name():
    offenders = []
    for f in _production_files():
        for node in ast.walk(ast.parse(open(f).read())):
            if isinstance(node, ast.ImportFrom) and node.module == MODULE:
                for alias in node.names:
                    if alias.name.startswith("_"):
                        offenders.append(f"{f}:{node.lineno} imports {alias.name}")
    assert not offenders, (
        "production code imports private names from common/orchestrator:\n  "
        + "\n  ".join(offenders)
        + "\n\nIf the helper is genuinely shared, drop the underscore and add "
          "it to __all__. If it is internal, stop importing it."
    )


def test_all_is_declared_and_complete():
    assert hasattr(orch, "__all__"), "common/orchestrator must declare __all__"
    missing = [n for n in orch.__all__ if not hasattr(orch, n)]
    assert not missing, f"__all__ names that do not exist: {missing}"


def test_every_externally_imported_name_is_in_all():
    """__all__ must cover what other modules actually import.

    A name imported elsewhere but absent from __all__ is API that nobody
    declared, which is how the twelve underscore helpers happened.
    """
    imported = set()
    for f in _production_files():
        for node in ast.walk(ast.parse(open(f).read())):
            if isinstance(node, ast.ImportFrom) and node.module == MODULE:
                imported.update(
                    a.name for a in node.names if not a.name.startswith("_")
                )
    undeclared = sorted(imported - set(orch.__all__))
    assert not undeclared, (
        f"imported by other modules but missing from __all__: {undeclared}"
    )


@pytest.mark.parametrize(
    "name",
    ["collect_outputs", "save_stage_outputs", "node_inputs", "node_output_paths",
     "create_submitit_executor", "load_launcher_config", "submit_slurm_job",
     "sanitize_cuda_visible_devices", "NoOpLogger", "safe_log_table",
     "print_status", "inject_prompt_from_file"],
)
def test_promoted_helper_is_public_and_declared(name):
    """The twelve that were private. Renaming one back breaks four dagspaces."""
    assert hasattr(orch, name), f"{name} disappeared from common/orchestrator"
    assert name in orch.__all__, f"{name} must stay in __all__"
    assert not hasattr(orch, f"_{name}"), (
        f"_{name} came back as an alias. Keep one spelling, or importers "
        f"drift between the two."
    )


def test_test_only_privates_stay_private():
    """These are reached only by tests, so they are correctly not API."""
    for name in TEST_ONLY_PRIVATES:
        assert hasattr(orch, name), f"{name} vanished"
        assert name not in orch.__all__, (
            f"{name} is test-only and must not be advertised as public API"
        )


def test_the_shared_runner_base_uses_public_names():
    """common/runners/base.py was one of the four importers."""
    src = open("dagspaces/common/runners/base.py").read()
    assert "save_stage_outputs" in src and "collect_outputs" in src
    assert "_save_stage_outputs" not in src and "_collect_outputs" not in src
