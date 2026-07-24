"""ensure_importable_sentence_transformers: torchcodec breakage shim.

torchcodec raises RuntimeError (not ImportError) at import when no compatible
FFmpeg shared library loads (e.g. klara: system FFmpeg 8 needs a newer GLIBCXX
than the anaconda-base libstdc++). sentence-transformers guards its torchcodec
import with ``except (ImportError, OSError)`` only, so that RuntimeError kills
every ``import sentence_transformers`` — this shim stubs torchcodec so the
guard sees a ModuleNotFoundError and degrades to the no-audio/video path.
"""

from __future__ import annotations

import importlib
import sys
import types

import pytest

from dagspaces.common.stage_utils import ensure_importable_sentence_transformers


@pytest.fixture
def no_torchcodec(monkeypatch):
    """Ensure torchcodec (and submodules) absent from sys.modules.

    Teardown also removes any stub the shim-under-test installed — it writes
    to sys.modules directly (untracked by monkeypatch), and a leaked stub
    poisons later tests that import datasets/TRL in this process.
    """
    for name in [m for m in sys.modules if m == "torchcodec" or m.startswith("torchcodec.")]:
        monkeypatch.delitem(sys.modules, name)
    yield monkeypatch
    for name in [m for m in sys.modules if m == "torchcodec" or m.startswith("torchcodec.")]:
        del sys.modules[name]


class TestBrokenTorchcodec:
    def test_runtimeerror_import_gets_stubbed(self, no_torchcodec):
        real_import = importlib.import_module

        def _broken_import(name, *args, **kwargs):
            if name == "torchcodec":
                raise RuntimeError("Could not load libtorchcodec")
            return real_import(name, *args, **kwargs)

        no_torchcodec.setattr(importlib, "import_module", _broken_import)

        ensure_importable_sentence_transformers()

        assert isinstance(sys.modules["torchcodec"], types.ModuleType)
        # The stub has no __path__, so the exact import sentence-transformers
        # guards (`from torchcodec.decoders import ...`) must raise an
        # ImportError subclass — the branch its except clause catches.
        with pytest.raises(ImportError):
            from torchcodec.decoders import AudioDecoder  # noqa: F401
        # transformers probes availability via find_spec, which raises
        # ValueError on a module whose __spec__ is None — the stub must
        # carry a real spec.
        assert importlib.util.find_spec("torchcodec") is not None

    def test_stub_survives_repeat_calls(self, no_torchcodec):
        no_torchcodec.setattr(
            importlib, "import_module",
            lambda name, *a, **k: (_ for _ in ()).throw(RuntimeError("boom")),
        )
        ensure_importable_sentence_transformers()
        stub = sys.modules["torchcodec"]
        ensure_importable_sentence_transformers()
        assert sys.modules["torchcodec"] is stub


class TestWorkingTorchcodec:
    def test_noop_when_already_imported(self, monkeypatch):
        marker = types.ModuleType("torchcodec")
        marker.MARKER = "real"
        monkeypatch.setitem(sys.modules, "torchcodec", marker)
        ensure_importable_sentence_transformers()
        assert sys.modules["torchcodec"] is marker

    def test_clean_import_not_replaced(self, no_torchcodec):
        real = types.ModuleType("torchcodec")

        def _ok_import(name, *args, **kwargs):
            if name == "torchcodec":
                sys.modules["torchcodec"] = real
                return real
            return importlib.import_module(name, *args, **kwargs)

        no_torchcodec.setattr(importlib, "import_module", _ok_import)
        ensure_importable_sentence_transformers()
        assert sys.modules["torchcodec"] is real
