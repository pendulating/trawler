"""Prompt-wiring regression guard for the historical_norms dagspace.

Context (2026-07-12): from 2026-03-09 (fb7256e) until 2026-07-12, config.yaml
loaded the active prompt keys as `prompt@prompt_*:` GROUP defaults positioned
after the pipeline entry in the defaults list. Hydra merges later defaults
over earlier ones, so those group defaults silently clobbered every
pipeline-body prompt selection (`prompt_ci_reasoning: ${prompt_ci_reasoning_fiction}`)
— ALL fiction extraction runs (norms + flows, fiction10 + top100) actually
ran with the prescriptive-text prompts. The fix makes the active keys body
keys in config.yaml, which pipeline bodies legitimately override.

These tests compose every pipeline config exactly as the CLI would and assert
the *effective* prompt is the one the pipeline declares. If someone
reintroduces an active-key group default (or renames a variant), this fails.
"""

from pathlib import Path

import pytest
import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import dagspaces.historical_norms

CONF_DIR = Path(dagspaces.historical_norms.__file__).parent / "conf"
PIPELINE_DIR = CONF_DIR / "pipeline"

# The four runtime prompt keys stages resolve, and the config packages their
# named variants live under (interpolation targets in pipeline bodies).
ACTIVE_PROMPT_KEYS = (
    "prompt_reasoning",
    "prompt_extraction",
    "prompt_ci_reasoning",
    "prompt_ci_extraction",
)


def _compose(pipeline: str):
    with initialize_config_dir(config_dir=str(CONF_DIR), version_base="1.3"):
        return compose(config_name="config", overrides=[f"pipeline={pipeline}"])


def _declared_prompt_selections(pipeline_yaml: Path) -> dict[str, str]:
    """Map active prompt keys a pipeline body sets to the package they select.

    E.g. `prompt_ci_reasoning: ${prompt_ci_reasoning_fiction}` ->
    {"prompt_ci_reasoning": "prompt_ci_reasoning_fiction"}.
    """
    raw = yaml.safe_load(pipeline_yaml.read_text()) or {}
    out = {}
    for key in ACTIVE_PROMPT_KEYS:
        val = raw.get(key)
        if isinstance(val, str) and val.startswith("${") and val.endswith("}"):
            out[key] = val[2:-1]
    return out


ALL_PIPELINES = sorted(p.stem for p in PIPELINE_DIR.glob("*.yaml"))
PIPELINES_WITH_SELECTIONS = [
    p for p in ALL_PIPELINES if _declared_prompt_selections(PIPELINE_DIR / f"{p}.yaml")
]


class TestDeclaredPromptSelectionsAreEffective:
    """The generic invariant: what a pipeline declares is what it gets."""

    @pytest.mark.parametrize("pipeline", PIPELINES_WITH_SELECTIONS)
    def test_pipeline_prompt_selection_wins(self, pipeline):
        declared = _declared_prompt_selections(PIPELINE_DIR / f"{pipeline}.yaml")
        cfg = _compose(pipeline)
        for active_key, target_pkg in declared.items():
            effective = OmegaConf.select(cfg, f"{active_key}.name")
            intended = OmegaConf.select(cfg, f"{target_pkg}.name")
            assert intended is not None, (
                f"{pipeline}: interpolation target '{target_pkg}' has no "
                f"'name' field — add one to its prompt yaml"
            )
            assert effective == intended, (
                f"{pipeline}: declares {active_key} -> {target_pkg} "
                f"(prompt '{intended}') but composition resolved prompt "
                f"'{effective}'. A config-level group default is clobbering "
                f"the pipeline's selection again (see module docstring)."
            )


class TestKnownVariants:
    """Explicit spot checks for the pipelines the 2026-03→07 bug shipped on,
    plus the prescriptive defaults that must NOT flip to fiction."""

    @pytest.mark.parametrize(
        "pipeline,key,expected",
        [
            ("COLM_flows_fiction", "prompt_ci_reasoning", "ci_reasoning_fiction"),
            ("COLM_flows_fiction", "prompt_ci_extraction", "ci_extraction_fiction"),
            ("COLM_flows_reasoning_prefetched_qwen36", "prompt_ci_reasoning", "ci_reasoning_fiction"),
            ("COLM_norms_fiction_prefetched_qwen36", "prompt_reasoning", "norm_reasoning_fiction"),
            ("COLM_norms_fiction_prefetched_qwen36", "prompt_extraction", "norm_extraction_fiction"),
            ("COLM_norms_prescriptive", "prompt_reasoning", "norm_reasoning_prescriptive"),
            ("ci_extraction_religious", "prompt_ci_reasoning", "ci_reasoning_prescriptive"),
            # Base pipeline sets nothing -> config.yaml prescriptive defaults.
            ("norm_extraction", "prompt_reasoning", "norm_reasoning_prescriptive"),
            ("norm_extraction", "prompt_ci_reasoning", "ci_reasoning_prescriptive"),
        ],
    )
    def test_effective_prompt(self, pipeline, key, expected):
        cfg = _compose(pipeline)
        assert OmegaConf.select(cfg, f"{key}.name") == expected

    def test_fiction_and_prescriptive_prompts_differ(self):
        """Sanity: the two flows-reasoning variants are actually different
        prompts (guards against a copy-paste collapse of the yamls)."""
        cfg = _compose("COLM_flows_fiction")
        fict = OmegaConf.select(cfg, "prompt_ci_reasoning_fiction.system_prompt")
        presc = OmegaConf.select(cfg, "prompt_ci_reasoning_prescriptive.system_prompt")
        assert fict and presc and fict != presc
        assert "literary texts" in fict
        assert "prescriptive and religious texts" in presc
