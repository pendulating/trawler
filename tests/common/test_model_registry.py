"""Tests for dagspaces.common.model_registry.resolve_model_source.

The resolver must redirect a model load path to the node-local /scratch
mirror ONLY when TRAWLER_MODEL_REGISTRY is set, the source directory is
visible, and the mirror's .sync_complete marker records this exact source
(the activate_stage_venv.sh marker convention). Everything else must pass
through unchanged — a wrong redirect loads the wrong weights.
"""

import pytest

from dagspaces.common.model_registry import resolve_model_source


@pytest.fixture()
def zoo_and_registry(tmp_path, monkeypatch):
    zoo = tmp_path / "zoo" / "models"
    reg = tmp_path / "scratch" / "registry" / "models"
    src = zoo / "Qwen3.5-9B"
    src.mkdir(parents=True)
    (src / "config.json").write_text("{}")
    mirror = reg / "Qwen3.5-9B"
    mirror.mkdir(parents=True)
    (mirror / "config.json").write_text("{}")
    monkeypatch.setenv("TRAWLER_MODEL_REGISTRY", str(reg))
    return src, mirror


def _write_marker(mirror, src):
    (mirror / ".sync_complete").write_text(
        f"src={src}\nhost=testhost\ndate=2026-07-19\nfiles=1 bytes=2\n"
    )


class TestResolveModelSource:
    def test_redirects_when_marker_matches(self, zoo_and_registry):
        src, mirror = zoo_and_registry
        _write_marker(mirror, src)
        assert resolve_model_source(str(src)) == str(mirror)

    def test_trailing_slash_source_still_matches(self, zoo_and_registry):
        src, mirror = zoo_and_registry
        _write_marker(mirror, src)
        assert resolve_model_source(str(src) + "/") == str(mirror)

    def test_no_marker_falls_back_to_source(self, zoo_and_registry):
        src, _mirror = zoo_and_registry
        assert resolve_model_source(str(src)) == str(src)

    def test_marker_for_different_source_is_rejected(self, zoo_and_registry):
        # Basename collision: a mirror synced from some OTHER path with the
        # same basename must never be substituted.
        src, mirror = zoo_and_registry
        _write_marker(mirror, "/somewhere/else/Qwen3.5-9B")
        assert resolve_model_source(str(src)) == str(src)

    def test_env_unset_is_noop(self, zoo_and_registry, monkeypatch):
        src, mirror = zoo_and_registry
        _write_marker(mirror, src)
        monkeypatch.delenv("TRAWLER_MODEL_REGISTRY")
        assert resolve_model_source(str(src)) == str(src)

    def test_missing_source_dir_is_never_redirected(self, zoo_and_registry):
        src, mirror = zoo_and_registry
        _write_marker(mirror, src.parent / "Ghost-Model")
        ghost = src.parent / "Ghost-Model"
        assert resolve_model_source(str(ghost)) == str(ghost)

    def test_hub_id_passes_through(self, zoo_and_registry):
        assert resolve_model_source("Qwen/Qwen3.5-9B") == "Qwen/Qwen3.5-9B"

    def test_empty_and_none_pass_through(self, zoo_and_registry):
        assert resolve_model_source("") == ""
        assert resolve_model_source(None) == ""
