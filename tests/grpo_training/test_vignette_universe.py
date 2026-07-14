"""Tests for the separate judgment-vignette universe knob.

Covers ``_resolve_vignette_universes`` — the single-variable lever that lets the
judgment vignettes be drawn from a DIFFERENT (e.g. more force-balanced top100)
norm universe while R_ground grounding + the CI-extraction prompts keep the
original universe. See ``stages/grpo_training.py`` and the v11 field notes.

The selection logic is pure; the heavy training path
(``run_grpo_training_stage``) is exercised on the cluster, not here.
"""

from __future__ import annotations

import json

import pytest

from dagspaces.grpo_training.stages.grpo_training import (
    _generate_vignettes,
    _resolve_vignette_universes,
)

# Minimal prompt template using the placeholders _generate_vignettes substitutes.
_VIG_TMPL = "Scenario: {{scenario}} Subject: {{subject}} Act: {{act}}"


def _norm(force: str, govern: bool = True) -> dict:
    return {
        "normative_force": force,
        "governs_info_flow": govern,
        "norm_subject": "a clerk",
        "norm_act": "disclose the record",
        "condition_of_application": "asked by a stranger",
        "context": "an office",
    }


# A skewed grounding universe (4 yes : 1 no among info-flow norms) and a balanced
# one (1 yes : 1 no) — the whole point of pointing vignettes at a separate corpus.
_SKEWED = {
    "bookA": [
        _norm("obligatory"),
        _norm("obligatory"),
        _norm("recommended"),
        _norm("recommended"),
        _norm("prohibited"),
    ]
}
_BALANCED = {
    "bookB": [
        _norm("obligatory"),
        _norm("prohibited"),
        _norm("recommended"),
        _norm("discouraged"),
    ]
}


def test_empty_path_returns_grounding_universe_object():
    """Unset path ⇒ vignettes use the grounding universe (identity, historical)."""
    out = _resolve_vignette_universes(_SKEWED, "")
    assert out is _SKEWED


def test_nonexistent_path_raises():
    """A truthy-but-nonexistent path must fail loud, not silently fall back: a
    silent fallback makes a 'balanced-vignette' arm a byte-identical copy of its
    control while the run metadata certifies the configured (missing) corpus."""
    with pytest.raises(FileNotFoundError):
        _resolve_vignette_universes(_SKEWED, "/no/such/universe.json")


def test_directory_path_falls_back_to_grounding(tmp_path):
    """A directory (the os.path.abspath("") → CWD trap) must not be opened."""
    out = _resolve_vignette_universes(_SKEWED, str(tmp_path))
    assert out is _SKEWED


def test_valid_json_file_is_loaded(tmp_path):
    p = tmp_path / "vignette_universe.json"
    p.write_text(json.dumps(_BALANCED), encoding="utf-8")
    out = _resolve_vignette_universes(_SKEWED, str(p))
    assert out is not _SKEWED
    assert set(out.keys()) == {"bookB"}
    assert len(out["bookB"]) == 4


def _yes_no(vignettes):
    yes = sum(1 for v in vignettes if v["gold_judgment"] == "yes")
    no = sum(1 for v in vignettes if v["gold_judgment"] == "no")
    return yes, no


def test_separate_universe_rebalances_vignette_gold():
    """The probe's reason for existing: a separate universe shifts the vignette
    yes:no balance without touching grounding."""
    skewed_yes, skewed_no = _yes_no(_generate_vignettes(_SKEWED, _VIG_TMPL))
    bal_yes, bal_no = _yes_no(_generate_vignettes(_BALANCED, _VIG_TMPL))

    assert (skewed_yes, skewed_no) == (4, 1)   # 4:1 permissive
    assert (bal_yes, bal_no) == (2, 2)         # 1:1 balanced
    # The balanced universe yields a strictly larger "no" share.
    assert bal_no / (bal_yes + bal_no) > skewed_no / (skewed_yes + skewed_no)
