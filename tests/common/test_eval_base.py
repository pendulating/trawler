"""Contract tests for the shared benchmark stage runners.

These lock the StageResult shape that every benchmark dagspace produced
BEFORE the 2026-08-12 migration to ``dagspaces/common/runners/eval_base.py``.
The orchestrator, the W&B logger, and the ArtifactRegistry all read that
shape, so a change here is a change to every benchmark at once.
"""

from __future__ import annotations

import json
import os
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from dagspaces.common.runners.eval_base import (
    EvalLoadRunner,
    EvalMetricsRunner,
    EvalParseRunner,
    EvalStageRunner,
    runtime_sample_n,
    write_dataset,
)


def _context(tmp_path, *, inputs=None, outputs=None, cfg=None):
    """A StageExecutionContext good enough for a DataFrame-in/out stage."""
    out_dir = str(tmp_path / "out")
    os.makedirs(out_dir, exist_ok=True)
    return SimpleNamespace(
        cfg=cfg if cfg is not None else SimpleNamespace(runtime=None),
        node=SimpleNamespace(key="test_node", outputs={}),
        inputs=inputs or {},
        output_paths=outputs or {"dataset": os.path.join(out_dir, "d.parquet")},
        output_dir=out_dir,
        output_root=str(tmp_path),
    )


@pytest.fixture
def input_parquet(tmp_path):
    path = tmp_path / "in.parquet"
    pd.DataFrame({"q": ["a", "b", "c"], "generated_text": ["x", "y", "z"]}).to_parquet(
        path, index=False
    )
    return str(path)


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------

def test_write_dataset_creates_the_parent_directory(tmp_path):
    target = str(tmp_path / "deep" / "deeper" / "d.parquet")
    returned = write_dataset(pd.DataFrame({"a": [1, 2]}), target)
    assert returned == target
    assert os.path.exists(target)
    assert len(pd.read_parquet(target)) == 2


def test_write_dataset_drops_the_index():
    """The hand-written runners all passed index=False. Keep that."""
    import tempfile

    df = pd.DataFrame({"a": [1, 2]}, index=[7, 9])
    with tempfile.TemporaryDirectory() as d:
        path = write_dataset(df, os.path.join(d, "d.parquet"))
        assert list(pd.read_parquet(path).columns) == ["a"]


@pytest.mark.parametrize(
    "runtime,expected",
    [
        (None, None),
        (SimpleNamespace(sample_n=None), None),
        (SimpleNamespace(sample_n=5), 5),
        (SimpleNamespace(sample_n="7"), 7),
        (SimpleNamespace(), None),
    ],
)
def test_runtime_sample_n(runtime, expected):
    assert runtime_sample_n(SimpleNamespace(runtime=runtime)) == expected


# --------------------------------------------------------------------------
# EvalStageRunner
# --------------------------------------------------------------------------

def test_eval_stage_runner_reads_transforms_writes(tmp_path, input_parquet):
    class Doubler(EvalStageRunner):
        stage_name = "doubler"

        def transform(self, df, context):
            return pd.concat([df, df], ignore_index=True)

    ctx = _context(tmp_path, inputs={"dataset": input_parquet})
    result = Doubler().run(ctx)

    assert set(result.outputs) == {"dataset"}
    assert result.metadata == {"rows": 6}
    assert len(pd.read_parquet(result.outputs["dataset"])) == 6


def test_eval_stage_runner_merges_stage_metadata(tmp_path, input_parquet):
    class WithTier(EvalStageRunner):
        stage_name = "with_tier"

        def transform(self, df, context):
            return df

        def stage_metadata(self, context, df):
            return {"tier": "2a"}

    result = WithTier().run(_context(tmp_path, inputs={"dataset": input_parquet}))
    assert result.metadata == {"rows": 3, "tier": "2a"}


def test_stage_metadata_sees_the_written_dataframe(tmp_path, input_parquet):
    """ci_heuristic summarises its own output (per-step parse rates) here."""
    seen = {}

    class Summariser(EvalStageRunner):
        stage_name = "summariser"

        def transform(self, df, context):
            return df.head(2)

        def stage_metadata(self, context, df):
            seen["n"] = len(df)
            return {"n_seen": len(df)}

    result = Summariser().run(_context(tmp_path, inputs={"dataset": input_parquet}))
    assert seen["n"] == 2, "the hook must get the OUTPUT frame, not the input"
    assert result.metadata == {"rows": 2, "n_seen": 2}


def test_eval_stage_runner_requires_transform(tmp_path, input_parquet):
    class Bare(EvalStageRunner):
        stage_name = "bare"

    with pytest.raises(NotImplementedError, match="transform"):
        Bare().run(_context(tmp_path, inputs={"dataset": input_parquet}))


def test_eval_stage_runner_honours_custom_keys(tmp_path):
    src = tmp_path / "src.parquet"
    pd.DataFrame({"a": [1]}).to_parquet(src, index=False)

    class Custom(EvalStageRunner):
        stage_name = "custom"
        input_key = "cases"
        output_key = "scored"

        def transform(self, df, context):
            return df

    ctx = _context(
        tmp_path,
        inputs={"cases": str(src)},
        outputs={"scored": str(tmp_path / "o" / "s.parquet")},
    )
    result = Custom().run(ctx)
    assert set(result.outputs) == {"scored"}


# --------------------------------------------------------------------------
# EvalLoadRunner
# --------------------------------------------------------------------------

def test_eval_load_runner_takes_no_input(tmp_path):
    class Loader(EvalLoadRunner):
        stage_name = "load_dataset"

        def load(self, context):
            return pd.DataFrame({"a": [1, 2, 3, 4]})

        def stage_metadata(self, context, df):
            return {"k_shot": 5}

    result = Loader().run(_context(tmp_path))
    assert result.metadata == {"rows": 4, "k_shot": 5}
    assert len(pd.read_parquet(result.outputs["dataset"])) == 4


# --------------------------------------------------------------------------
# EvalParseRunner
# --------------------------------------------------------------------------

def test_eval_parse_runner_reports_health(tmp_path, input_parquet, monkeypatch):
    calls: dict[str, Any] = {}

    def fake_health(df, **kwargs):
        calls.update(kwargs)
        calls["n_rows"] = len(df)
        return "REPORT"

    def fake_log(context, report, metadata):
        calls["logged_report"] = report
        metadata["health"] = "ok"

    monkeypatch.setattr(
        "dagspaces.common.eval_sanity.compute_parse_health", fake_health
    )
    monkeypatch.setattr(
        "dagspaces.common.runners.sanity.log_sanity_to_context", fake_log
    )
    monkeypatch.setattr(
        "dagspaces.common.runners.sanity.sanity_overrides",
        lambda cfg: ({"t": 1.0}, ["refuse"]),
    )
    monkeypatch.setattr(
        "dagspaces.common.runners.sanity.task_model_name", lambda cfg: "some-model"
    )

    class Parser(EvalParseRunner):
        stage_name = "parse_responses"
        health_dagspace = "testbench"
        label_col = "prediction"

        def transform(self, df, context):
            out = df.head(2).copy()
            out["prediction"] = ["A", "B"]
            return out

    result = Parser().run(_context(tmp_path, inputs={"dataset": input_parquet}))

    # expected_input_n must be the INPUT row count, not the output count —
    # that is how the report detects a stage that dropped rows.
    assert calls["expected_input_n"] == 3
    assert calls["n_rows"] == 2
    assert calls["dagspace"] == "testbench"
    assert calls["stage"] == "parse_responses"
    assert calls["label_col"] == "prediction"
    assert calls["model"] == "some-model"
    assert calls["thresholds"] == {"t": 1.0}
    assert calls["refusal_patterns"] == ["refuse"]
    assert calls["logged_report"] == "REPORT"
    # log_sanity_to_context mutates the live metadata dict, so its keys must
    # survive into the returned StageResult.
    assert result.metadata == {"rows": 2, "health": "ok"}


def test_eval_parse_runner_default_label_col_matches_the_library(monkeypatch):
    """An unset label_col must reach compute_parse_health as None.

    The vlm_geoprivacy parse stages depend on this: their predictions live in
    seven per-question columns, so there is no single label column.
    """
    import inspect

    from dagspaces.common.eval_sanity import compute_parse_health

    library_default = inspect.signature(compute_parse_health).parameters[
        "label_col"
    ].default
    assert EvalParseRunner.label_col == library_default


def test_eval_parse_runner_demands_a_dagspace_name(tmp_path, input_parquet):
    class NoName(EvalParseRunner):
        stage_name = "parse_responses"

        def transform(self, df, context):
            return df

    with pytest.raises(ValueError, match="health_dagspace"):
        NoName().run(_context(tmp_path, inputs={"dataset": input_parquet}))


# --------------------------------------------------------------------------
# EvalMetricsRunner
# --------------------------------------------------------------------------

def test_eval_metrics_runner_writes_json_and_parquet(tmp_path, input_parquet):
    class Metrics(EvalMetricsRunner):
        stage_name = "compute_metrics"

        def compute(self, df, context):
            return {"accuracy": 0.5, "n": len(df)}

        def to_dataframe(self, metrics):
            return pd.DataFrame([metrics])

    result = Metrics().run(_context(tmp_path, inputs={"dataset": input_parquet}))

    assert set(result.outputs) == {"dataset", "metrics_json"}
    assert os.path.basename(result.outputs["metrics_json"]) == "metrics.json"
    with open(result.outputs["metrics_json"]) as f:
        assert json.load(f) == {"accuracy": 0.5, "n": 3}

    # The orchestrator logs from metadata["metrics"] — this key is the contract.
    assert result.metadata["metrics"] == {"accuracy": 0.5, "n": 3}
    assert result.metadata["rows"] == 1


def test_eval_metrics_runner_serialises_non_json_values(tmp_path, input_parquet):
    """The hand-written runners all passed default=str. Keep that."""
    import numpy as np

    class Metrics(EvalMetricsRunner):
        stage_name = "compute_metrics"

        def compute(self, df, context):
            return {"arr": np.int64(3), "when": pd.Timestamp("2026-08-12")}

        def to_dataframe(self, metrics):
            return pd.DataFrame([{"k": 1}])

    result = Metrics().run(_context(tmp_path, inputs={"dataset": input_parquet}))
    with open(result.outputs["metrics_json"]) as f:
        loaded = json.load(f)
    assert loaded["when"] == "2026-08-12 00:00:00"


# --------------------------------------------------------------------------
# The migrated dagspaces
# --------------------------------------------------------------------------

MIGRATED = [
    ("cirl", "prediction", "cirl"),
    ("confaide", "prediction", "confaide"),
    ("goldcoin_hipaa", "prediction", "goldcoin"),
    ("mmlu", "prediction_letter", "mmlu"),
]


@pytest.mark.parametrize("dagspace,label_col,health_name", MIGRATED, ids=[m[0] for m in MIGRATED])
def test_migrated_parse_runners_keep_their_health_identity(
    dagspace, label_col, health_name
):
    """The health report's dagspace name and label column are per-benchmark.

    These values feed the eval-health dashboards. A rename silently splits a
    benchmark's history into two series.
    """
    import importlib

    mod = importlib.import_module(f"dagspaces.{dagspace}.runners.eval_stages")
    runner = mod.ParseResponsesRunner()
    assert isinstance(runner, EvalParseRunner)
    assert runner.health_dagspace == health_name
    assert runner.label_col == label_col


def test_vlm_parse_runners_carry_no_label_column():
    """vlm_geoprivacy predictions span Q1_pred..Q7_pred — there is no one label."""
    from dagspaces.vlm_geoprivacy_bench.runners.eval_stages import (
        ParseFreeformRunner,
        ParseMCQRunner,
    )

    assert ParseMCQRunner.label_col is None
    assert ParseFreeformRunner.label_col is None
    assert ParseMCQRunner.health_dagspace == "vlm_geoprivacy"


def test_aug_parse_runners_extend_bench_but_rename_the_health_report():
    from dagspaces.vlm_geoprivacy_aug.runners.eval_stages import (
        ParseMCQRunner as AugParse,
    )
    from dagspaces.vlm_geoprivacy_bench.runners.eval_stages import (
        ParseMCQRunner as BenchParse,
    )

    assert issubclass(AugParse, BenchParse)
    assert AugParse.health_dagspace == "vlm_geoprivacy_aug"
    assert BenchParse.health_dagspace == "vlm_geoprivacy"


def test_ci_heuristic_runners_use_the_shared_bases():
    """ci_heuristic reports per-step parse health, so it is not an EvalParseRunner."""
    from dagspaces.ci_heuristic.runners.eval_stages import (
        LoadCasesRunner,
        ScoreTraversalRunner,
        TPProbeRunner,
        TraverseRunner,
    )

    assert issubclass(LoadCasesRunner, EvalLoadRunner)
    assert issubclass(TraverseRunner, EvalStageRunner)
    assert issubclass(TPProbeRunner, EvalStageRunner)
    assert not issubclass(TraverseRunner, EvalParseRunner)
    # Two node inputs, and one call returns both the metrics and the frame.
    assert not issubclass(ScoreTraversalRunner, EvalMetricsRunner)


def test_every_migrated_registry_still_resolves():
    """Each dagspace must still hand the orchestrator a full stage registry."""
    import importlib

    expected = {
        "cirl": 4, "confaide": 4, "goldcoin_hipaa": 4, "mmlu": 4,
        "simpleqa_verified": 6, "vlm_geoprivacy_bench": 7,
        "vlm_geoprivacy_aug": 9, "ci_heuristic": 4,
    }
    for dagspace, n in expected.items():
        registry = importlib.import_module(
            f"dagspaces.{dagspace}.runners"
        ).get_stage_registry()
        assert len(registry) == n, f"{dagspace}: {sorted(registry)}"
        for stage, runner in registry.items():
            assert runner.stage_name == stage, (
                f"{dagspace}.{stage} runner reports stage_name="
                f"{runner.stage_name!r} — the registry key and the runner "
                f"must agree, or the health report is filed under the wrong name."
            )
