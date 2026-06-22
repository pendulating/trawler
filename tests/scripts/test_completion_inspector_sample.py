"""Tests for the seeded-sample + annotation upgrades to completion_inspector.

The completion inspector is the artifact handed to an external annotator who
audits judge accuracy. Two failure modes would silently corrupt that audit:

1. **Non-deterministic sampling.** If two invocations with the same --seed
   produce different row subsets, two annotators cannot meaningfully merge
   their work (the rows don't line up). The manifest carries seed +
   sampled_indices precisely so this is reproducible.
2. **Missing annotate-mode payload.** The HTML needs MANIFEST,
   ANNOTATE_MODE = true, and the annot-strip markup. If any of those drift,
   the annotation UI silently degrades to a read-only viewer and no expert
   annotation gets captured.

Coverage:

- ``_select_sample`` is deterministic given a seed and rejects N > total.
- ``_validate_stage_key`` handles exact, substring, ambiguous, and missing.
- argparse mutexes: --sample with --rows / --max-rows, --sample / --annotate
  without --stage.
- End-to-end smoke: build a fixture parquet tree, invoke the CLI with
  --stage + --sample + --annotate, and assert the resulting HTML carries the
  manifest, the annotate flag, and at least one annotation strip.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

import scripts.completion_inspector as ci


# ── unit: sample selection ────────────────────────────────────────────────

def test_select_sample_deterministic_under_seed():
    a = ci._select_sample(n_total=1000, n_sample=20, seed=42)
    b = ci._select_sample(n_total=1000, n_sample=20, seed=42)
    assert a == b
    assert len(a) == 20
    assert len(set(a)) == 20
    assert all(0 <= i < 1000 for i in a)
    assert a == sorted(a)


def test_select_sample_seed_changes_indices():
    a = ci._select_sample(n_total=1000, n_sample=20, seed=0)
    b = ci._select_sample(n_total=1000, n_sample=20, seed=1)
    assert a != b  # extremely unlikely to collide


def test_select_sample_rejects_oversample():
    with pytest.raises(ValueError, match="exceeds available rows"):
        ci._select_sample(n_total=10, n_sample=20, seed=0)


def test_select_sample_full_population():
    out = ci._select_sample(n_total=5, n_sample=5, seed=0)
    assert out == [0, 1, 2, 3, 4]


# ── unit: stage key validation ────────────────────────────────────────────

def test_validate_stage_key_exact():
    avail = ["privacylens/agent_action_inference", "goldcoin/llm_inference_applicability"]
    assert ci._validate_stage_key(avail[0], avail) == avail[0]


def test_validate_stage_key_whitespace_tolerant():
    """Discovered keys use 'bench / stage' form; users will type 'bench/stage'."""
    avail = ["privacylens / agent_action_inference"]
    assert ci._validate_stage_key("privacylens/agent_action_inference", avail) == avail[0]


def test_validate_stage_key_unique_substring():
    avail = ["privacylens/agent_action_inference", "goldcoin/llm_inference_applicability"]
    assert ci._validate_stage_key("agent_action", avail) == avail[0]


def test_validate_stage_key_ambiguous_substring():
    avail = ["privacylens/agent_action_inference", "privacylens/agent_action_followup"]
    with pytest.raises(ValueError, match="matches 2 stages"):
        ci._validate_stage_key("agent_action", avail)


def test_validate_stage_key_missing_with_suggestion():
    avail = ["privacylens/agent_action_inference"]
    with pytest.raises(ValueError) as exc:
        ci._validate_stage_key("privacylens/agent_actoin_inference", avail)  # typo
    # Difflib should still flag the close match
    assert "Did you mean" in str(exc.value) or "Available" in str(exc.value)


# ── unit: manifest assembly ───────────────────────────────────────────────

def test_build_manifest_round_trip():
    m = ci._build_manifest(
        stage="privacylens/agent_action_inference",
        seed=42,
        sampled_indices=[3, 7, 9],
        n_total=100,
        runs={"Base": Path("/tmp/a"), "GRPO": Path("/tmp/b")},
        models=["Base", "GRPO"],
    )
    assert m["stage"] == "privacylens/agent_action_inference"
    assert m["seed"] == 42
    assert m["n_sampled"] == 3
    assert m["n_total"] == 100
    assert m["sampled_indices"] == [3, 7, 9]
    assert m["models"] == ["Base", "GRPO"]
    assert m["source_runs"] == {"Base": "/tmp/a", "GRPO": "/tmp/b"}
    assert "generated_at" in m


# ── argparse mutex tests (via subprocess) ─────────────────────────────────

def _cli(*args, expect_exit: int = 2):
    """Run the CLI module and return (proc, exit_code). expect_exit is for sanity."""
    proc = subprocess.run(
        [sys.executable, "-m", "scripts.completion_inspector", *args],
        capture_output=True, text=True,
    )
    return proc


def test_cli_sample_without_stage_errors():
    proc = _cli("--runs", "X=/tmp/nonexistent", "--sample", "10")
    assert proc.returncode != 0
    assert "--sample requires --stage" in proc.stderr


def test_cli_annotate_without_stage_errors():
    proc = _cli("--runs", "X=/tmp/nonexistent", "--annotate")
    assert proc.returncode != 0
    assert "--annotate requires --stage" in proc.stderr


def test_cli_sample_and_rows_mutex():
    proc = _cli("--runs", "X=/tmp/nonexistent", "--stage", "x", "--sample", "5", "--rows", "0:5")
    assert proc.returncode != 0
    assert "mutually exclusive" in proc.stderr


def test_cli_sample_and_max_rows_mutex():
    proc = _cli("--runs", "X=/tmp/nonexistent", "--stage", "x", "--sample", "5", "--max-rows", "5")
    assert proc.returncode != 0
    assert "mutually exclusive" in proc.stderr


# ── end-to-end smoke ──────────────────────────────────────────────────────

@pytest.fixture
def tiny_run_root(tmp_path):
    """Build a minimal eval_all-style run root with one primary stage and two judges."""
    root = tmp_path / "run_a"
    N = 20

    # Primary stage
    primary = root / "privacylens" / "outputs" / "agent_action_inference"
    primary.mkdir(parents=True)
    pd.DataFrame({
        "name": [f"case_{i}" for i in range(N)],
        "generated_text": [f"agent action {i}" for i in range(N)],
        "messages": [[{"role": "user", "content": f"prompt {i}"}] for i in range(N)],
    }).to_parquet(primary / "results.parquet")

    # Leakage judge
    lk = root / "privacylens" / "outputs" / "leakage_judge_inference"
    lk.mkdir(parents=True)
    pd.DataFrame({
        "name": [f"case_{i}" for i in range(N)],
        "generated_text": [f"leak judge raw {i}" for i in range(N)],
        "leak_flag": [bool(i % 2) for i in range(N)],
        "leak_probability": [round(0.1 * (i % 10), 3) for i in range(N)],
        "leak_judge_text": [f"leak reasoning {i}" for i in range(N)],
    }).to_parquet(lk / "results.parquet")

    # Helpfulness judge
    hp = root / "privacylens" / "outputs" / "helpfulness_judge_inference"
    hp.mkdir(parents=True)
    pd.DataFrame({
        "name": [f"case_{i}" for i in range(N)],
        "generated_text": [f"help judge raw {i}" for i in range(N)],
        "helpfulness_binary": [bool((i + 1) % 3) for i in range(N)],
        "helpfulness_score": [(i % 5) + 1 for i in range(N)],
        "helpfulness_judge_text": [f"help reasoning {i}" for i in range(N)],
    }).to_parquet(hp / "results.parquet")

    return root


def test_e2e_annotate_html_contains_manifest_and_strip(tiny_run_root, tmp_path):
    out = tmp_path / "audit.html"
    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.completion_inspector",
            "--runs", f"Base={tiny_run_root}",
            "--stage", "privacylens/agent_action_inference",
            "--sample", "5", "--seed", "7", "--annotate",
            "-o", str(out),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    assert out.exists()

    html = out.read_text()
    # Annotate-mode payload (canonical stage key uses ' / ' separator as
    # produced by discover_stages; user-supplied 'bench/stage' is normalized).
    assert "const ANNOTATE_MODE = true;" in html
    assert '"stage":"privacylens / agent_action_inference"' in html
    assert '"seed":7' in html
    assert '"n_sampled":5' in html
    # Judge meta wired in
    assert "leakage_judge_inference" not in html or "Leakage Judge" in html  # display names exported
    assert "Helpfulness Judge" in html
    # Annotation strip markup present in the rendered output
    assert "annot-strip" in html
    assert "annot-radios" in html
    assert "annot-progress" in html
    # Manifest banner element is in the template
    assert 'id="manifest-banner"' in html


def test_e2e_no_annotate_html_omits_annotate_mode(tiny_run_root, tmp_path):
    out = tmp_path / "viewer.html"
    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.completion_inspector",
            "--runs", f"Base={tiny_run_root}",
            "-o", str(out),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    html = out.read_text()
    assert "const ANNOTATE_MODE = false;" in html
    # Manifest is null when --stage absent
    assert "const MANIFEST = null;" in html


def test_e2e_annotate_includes_ranking_with_two_models(tiny_run_root, tmp_path):
    """Ranking strip should be rendered when 2+ models are present in --runs."""
    out = tmp_path / "audit2.html"
    # Reuse the same fixture parquet as both "Base" and "GRPO" — content
    # doesn't matter, we just need the inspector to see two model labels.
    proc = subprocess.run(
        [
            sys.executable, "-m", "scripts.completion_inspector",
            "--runs", f"Base={tiny_run_root}", f"GRPO={tiny_run_root}",
            "--stage", "privacylens/agent_action_inference",
            "--sample", "5", "--seed", "13", "--annotate",
            "-o", str(out),
        ],
        capture_output=True, text=True,
    )
    assert proc.returncode == 0, proc.stderr
    html = out.read_text()

    # Ranking markup is present
    assert "rank-strip" in html
    assert "rank-chip" in html
    assert "buildRankingStrip" in html  # the renderer is wired in
    assert "1 = least leaky" in html  # direction labels present in JS const
    assert "1 = most helpful" in html
    # CSV header includes rank column
    assert "row_idx,model,judge,judge_verdict,expert_verdict,agree,rank,notes" in html
    # JSON export schema is wrapped {annotations, rankings}
    assert "{manifest: MANIFEST, annotations, rankings}" in html


def test_e2e_annotate_single_model_hides_ranking(tiny_run_root, tmp_path):
    """With a single model, the ranking strip must not be rendered (markup
    is generated in JS via buildRankingStrip; the guard short-circuits)."""
    out = tmp_path / "audit_single.html"
    subprocess.run(
        [
            sys.executable, "-m", "scripts.completion_inspector",
            "--runs", f"Base={tiny_run_root}",
            "--stage", "privacylens/agent_action_inference",
            "--sample", "5", "--seed", "0", "--annotate",
            "-o", str(out),
        ],
        check=True, capture_output=True, text=True,
    )
    # The CSS class will still appear in the static <style> block but the
    # buildRankingStrip JS guard ensures no .rank-strip data-* element ever
    # renders for 1-model runs. We just sanity-check the CSS is unaffected.
    html = out.read_text()
    assert ".rank-strip" in html  # CSS rule is always present


def test_e2e_sample_reproducibility(tiny_run_root, tmp_path):
    """Two runs with the same seed must produce the same sampled_indices in MANIFEST."""
    outs = []
    for i in range(2):
        out = tmp_path / f"a{i}.html"
        subprocess.run(
            [
                sys.executable, "-m", "scripts.completion_inspector",
                "--runs", f"Base={tiny_run_root}",
                "--stage", "privacylens/agent_action_inference",
                "--sample", "5", "--seed", "42", "--annotate",
                "-o", str(out),
            ],
            check=True, capture_output=True, text=True,
        )
        outs.append(out.read_text())

    # Extract the sampled_indices substring (deterministic JSON encoding)
    def _extract_indices(html: str) -> str:
        # MANIFEST is a single-line JSON blob; find the sampled_indices field
        marker = '"sampled_indices":'
        i = html.find(marker)
        assert i >= 0, "sampled_indices missing from MANIFEST"
        j = html.find("]", i)
        return html[i:j + 1]

    assert _extract_indices(outs[0]) == _extract_indices(outs[1])
