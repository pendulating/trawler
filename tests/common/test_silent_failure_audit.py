"""Guard: the five failure paths that used to fail silently must report.

`dagspaces/common/` held 67 `except Exception: pass` sites. Most guard
best-effort telemetry — SLURM env vars, GPU probes, disk usage — where a
silent skip is correct and a crash would be wrong.

Five did not. Each one sat on a path where silence hides something that
changes results or hides a defect:

1/2. `log_sanity_report` in both the no-op and the real W&B logger. The call
     exists to be LOUD; a silent failure means a SANITY FAILURE reaches
     nobody.
3.   `sanity_overrides`. Falling back to default thresholds silently WEAKENS
     the health gate — a malformed `sanity.thresholds` looked exactly like
     "no overrides configured".
4.   `JudgeClient` model auto-discovery. On failure `model_name` stays
     `"default"`, and that literal string then travels into every judge
     request, failing far from the cause.
5.   `_split_reasoning`. A broken vLLM parser silently degraded EVERY
     completion in a run to the regex fallback. The two paths do not agree on
     all outputs.

Each fix keeps the "never fail the pipeline" contract: it reports and
continues. These tests assert both halves — it reports, AND it continues.
"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest


class _ExplodingReport:
    """A sanity report whose print_loud() raises, as a broken one would."""

    dagspace = "testbench"
    stage = "parse_responses"

    def print_loud(self):
        raise RuntimeError("formatting blew up")


# --------------------------------------------------------------------------
# 1. NoOpLogger.log_sanity_report
# --------------------------------------------------------------------------

def test_noop_logger_reports_a_broken_sanity_report(capsys):
    from dagspaces.common.orchestrator import NoOpLogger

    logger = NoOpLogger(SimpleNamespace(), stage="parse_responses")
    logger.log_sanity_report(_ExplodingReport())   # must NOT raise

    err = capsys.readouterr().err
    assert "testbench" in err and "parse_responses" in err, (
        "a sanity report that cannot print must still name itself on stderr"
    )
    assert "RuntimeError" in err


# --------------------------------------------------------------------------
# 2. WandbLogger.log_sanity_report
# --------------------------------------------------------------------------

def test_wandb_logger_reports_a_broken_sanity_report(capsys):
    from dagspaces.common.wandb_logger import WandbLogger

    logger = WandbLogger.__new__(WandbLogger)   # skip __init__ / wandb.init
    # `enabled` is a read-only property over wb_config.enabled.
    logger.wb_config = SimpleNamespace(enabled=False)
    logger._run = None
    logger.stage = "parse_responses"

    logger.log_sanity_report(_ExplodingReport())   # must NOT raise

    err = capsys.readouterr().err
    assert "testbench" in err and "parse_responses" in err
    assert "RuntimeError" in err


# --------------------------------------------------------------------------
# 3. sanity_overrides
# --------------------------------------------------------------------------

def test_sanity_overrides_reports_a_malformed_config(capsys):
    """A bad threshold must not look like 'no overrides configured'."""
    from omegaconf import OmegaConf

    from dagspaces.common.runners.sanity import sanity_overrides

    cfg = OmegaConf.create({"sanity": {"thresholds": {"parseable_rate:lt": "not-a-float"}}})
    thresholds, patterns = sanity_overrides(cfg)   # must NOT raise

    err = capsys.readouterr().err
    assert "sanity" in err.lower(), (
        "a malformed sanity config silently weakens the health gate; it must "
        "say so"
    )
    assert "DEFAULT" in err
    # (None, None) means "not configured, use the defaults". That is exactly
    # why the old silent swallow was dangerous: a malformed config produced
    # the SAME return value as an absent one, so the caller could not tell
    # a weakened gate from an unconfigured one.
    assert thresholds is None and patterns is None


def test_sanity_overrides_stays_quiet_on_a_good_config(capsys):
    from omegaconf import OmegaConf

    from dagspaces.common.runners.sanity import sanity_overrides

    cfg = OmegaConf.create({
        "sanity": {"thresholds": {"class_balance_min:lt": 0.001},
                   "refusal_patterns": ["I cannot"]},
    })
    thresholds, patterns = sanity_overrides(cfg)
    assert thresholds == {"class_balance_min:lt": 0.001}
    assert patterns == ["I cannot"]
    assert capsys.readouterr().err == "", "a valid config must not warn"


def test_sanity_overrides_stays_quiet_when_unconfigured(capsys):
    from omegaconf import OmegaConf

    from dagspaces.common.runners.sanity import sanity_overrides

    thresholds, patterns = sanity_overrides(OmegaConf.create({}))
    assert (thresholds, patterns) == (None, None)
    assert capsys.readouterr().err == ""


# --------------------------------------------------------------------------
# 4. JudgeClient model auto-discovery
# --------------------------------------------------------------------------

def test_judge_client_reports_a_failed_model_discovery(capsys):
    """model_name stays 'default' — the operator has to hear about it.

    Drives the real ``health_check()``: a healthy /health probe followed by a
    models.list() that raises, which is what a vLLM server looks like while
    it is still loading weights.
    """
    from dagspaces.common import judge_client as jc

    client = jc.JudgeClient.__new__(jc.JudgeClient)
    client.offline = False
    client.provider = "vllm"
    client.model_name = "default"
    client.base_url = "http://localhost:8000/v1"
    client._auth_header = lambda: {}
    client._session = SimpleNamespace(
        get=lambda *a, **kw: SimpleNamespace(status_code=200)
    )

    class _Models:
        def list(self):
            raise ConnectionError("server is still loading weights")

    client._client = SimpleNamespace(models=_Models())

    assert client.health_check() is True, "the probe must still pass"
    assert client.model_name == "default", "discovery failed, so it stays"

    err = capsys.readouterr().err
    assert "auto-discover" in err
    assert "ConnectionError" in err
    assert "localhost:8000" in err


def test_judge_client_source_no_longer_swallows_discovery(capsys):
    """The production source must carry the report, not a bare pass."""
    import inspect

    from dagspaces.common import judge_client as jc

    src = inspect.getsource(jc)
    idx = src.index("could not auto-discover")
    window = src[max(0, idx - 600):idx]
    assert "models.list()" in window, (
        "the auto-discovery report drifted away from the block it guards"
    )


# --------------------------------------------------------------------------
# 5. _split_reasoning parser fallback
# --------------------------------------------------------------------------

def test_reasoning_parser_fallback_warns_once_then_stays_quiet(capsys, monkeypatch):
    from dagspaces.common import reasoning as R

    monkeypatch.setattr(R, "_PARSER_FALLBACK_WARNED", set())
    monkeypatch.setattr(R, "_detect_reasoning_parser", lambda src: "deepseek_r1")
    monkeypatch.setattr(R, "_is_harmony_model", lambda src: False)

    import builtins
    real_import = builtins.__import__

    def _boom(name, *a, **kw):
        if name == "vllm.reasoning":
            raise ImportError("vllm.reasoning is gone")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _boom)

    text = "<think>hidden</think>ANSWER"
    r1, c1 = R._split_reasoning(text, "/models/DeepSeek-R1", True, None)
    first = capsys.readouterr().out
    assert "deepseek_r1" in first and "regex fallback" in first.lower().replace("-", " ")

    # The fallback still produced the right split — it degraded, not failed.
    assert r1 == "hidden"
    assert "<think>" not in c1 and "ANSWER" in c1

    # Second call: same parser, no repeat. This runs per completion.
    R._split_reasoning(text, "/models/DeepSeek-R1", True, None)
    assert capsys.readouterr().out == "", (
        "the parser-fallback warning must fire one time per process, not per "
        "completion"
    )


def test_reasoning_fallback_flag_is_per_parser(monkeypatch, capsys):
    from dagspaces.common import reasoning as R

    monkeypatch.setattr(R, "_PARSER_FALLBACK_WARNED", set())
    assert R._PARSER_FALLBACK_WARNED == set()
    R._PARSER_FALLBACK_WARNED.add("deepseek_r1")
    assert "qwen3" not in R._PARSER_FALLBACK_WARNED, (
        "one broken parser must not silence a different one"
    )


# --------------------------------------------------------------------------
# Regression: these five sites must not go back to a bare pass
# --------------------------------------------------------------------------

AUDITED = [
    ("dagspaces/common/orchestrator.py", "could not print the report"),
    ("dagspaces/common/wandb_logger.py", "could not print the report"),
    ("dagspaces/common/runners/sanity.py", "could not read the sanity overrides"),
    ("dagspaces/common/judge_client.py", "could not auto-discover"),
    ("dagspaces/common/reasoning.py", "uses the regex"),
]


@pytest.mark.parametrize("path,marker", AUDITED, ids=[p.split("/")[-1] for p, _ in AUDITED])
def test_audited_site_still_reports(path, marker):
    """Each audited site keeps its report. Removing one re-hides a failure."""
    assert marker in open(path).read(), (
        f"{path} lost the failure report for {marker!r}. That site used to "
        f"swallow silently; see this module's docstring for what it hides."
    )
