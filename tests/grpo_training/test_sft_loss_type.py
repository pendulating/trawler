"""Guard the SFT loss_type knob across the training/sft config group.

The files in ``dagspaces/grpo_training/conf/training/sft/`` are full
standalone copies, not overlays: Hydra config-group selection replaces the
whole group, so a key present only in ``default.yaml`` never reaches runs
launched with ``training/sft=gpt_oss`` (the gpt-oss canonical-sweep cell),
``sft_27b``, or the flow-ablation variants. Every group file must therefore
declare ``loss_type`` explicitly. Added 2026-07-18 with the switch from stock
SFT (TRL chunked_nll) to Dynamic Fine-Tuning (``loss_type: dft``,
arXiv:2508.05629) for all new SFT runs.
"""

from pathlib import Path

import pytest
import yaml

SFT_CONF_DIR = (
    Path(__file__).resolve().parents[2]
    / "dagspaces" / "grpo_training" / "conf" / "training" / "sft"
)

SFT_GROUP_FILES = sorted(SFT_CONF_DIR.glob("*.yaml"))


def _load(path: Path) -> dict:
    return yaml.safe_load(path.read_text()) or {}


def test_group_dir_exists_and_nonempty():
    assert SFT_GROUP_FILES, f"no yaml files found under {SFT_CONF_DIR}"


@pytest.mark.parametrize("path", SFT_GROUP_FILES, ids=lambda p: p.name)
def test_every_group_file_declares_loss_type(path):
    cfg = _load(path)
    assert "loss_type" in cfg, (
        f"{path.name} does not declare loss_type. Group files are full "
        "standalone copies — without the key, runs selecting this variant "
        "silently fall back to TRL's stock loss instead of DFT. Add "
        "`loss_type: dft` (or an explicit null to opt back into stock SFT)."
    )


@pytest.mark.parametrize("path", SFT_GROUP_FILES, ids=lambda p: p.name)
def test_loss_type_values_are_valid(path):
    # TRL 1.8.0 accepts nll | dft | chunked_nll; null means "let TRL default"
    # (stock pre-2026-07-18 behaviour) via the sft_training.py fallthrough.
    value = _load(path).get("loss_type")
    assert value in (None, "nll", "dft", "chunked_nll"), (
        f"{path.name}: loss_type={value!r} is not a valid TRL 1.8.0 value"
    )


def test_default_is_dft():
    assert _load(SFT_CONF_DIR / "default.yaml").get("loss_type") == "dft"
