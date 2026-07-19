#!/usr/bin/env python3
"""Generate `<family>/sft-canonical.yaml` model YAMLs from the canonical SFT sweeps.

For each model in the 2026-07-15 canonical SFT sweeps, emit a model YAML that
wires that model's LoRA adapter into vLLM:

    dagspaces/common/conf/model/<family>/sft-canonical.yaml

Each doc is derived from the model's own source YAML (the `<family>/<variant>`
recorded in that job's `.hydra/overrides.yaml`, i.e. `instruct` for this sweep)
rather than a hardcoded template, so per-family quirks survive: gpt-oss's
`sampling_params.temperature: 1.0`, phi-4's `force_answer_format`,
openthinker3's `enable_thinking: true`, gemma-4's `thinking_mode`.

Run selection: a job counts as COMPLETE only if it produced
`sft_only/outputs/sft/checkpoint/adapter_config.json`. When a model has more
than one complete run (retries), the most recent by `<date>_<sweep>/<time>`
wins. Incomplete runs are ignored, never silently preferred.

Usage::

    # Dry-run by default.
    python scripts/build_canonical_sft_model_yamls.py
    python scripts/build_canonical_sft_model_yamls.py --write --force

    # Custom roots:
    python scripts/build_canonical_sft_model_yamls.py --write \\
        multirun/2026-07-15_sft_canonical_gemma4 \\
        multirun/2026-07-15_sft_canonical_gemma4_gptoss
"""
from __future__ import annotations

import argparse
import copy
import re
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
MODEL_CONF_DIR = REPO / "dagspaces/common/conf/model"

DEFAULT_ROOTS = [
    REPO / "multirun/2026-07-15_sft_canonical_gemma4",
    REPO / "multirun/2026-07-15_sft_canonical_gemma4_gptoss",
]

OUT_VARIANT = "sft-canonical"

MODEL_OVERRIDE_RE = re.compile(r"^-?\s*model=(\S+)\s*$")


def read_job_model(job: Path) -> str | None:
    """Return the `family/variant` this job trained, from its .hydra record."""
    ov = job / ".hydra" / "overrides.yaml"
    if not ov.exists():
        return None
    for line in ov.read_text().splitlines():
        m = MODEL_OVERRIDE_RE.match(line.strip())
        if m:
            return m.group(1)
    return None


def discover_runs(roots: list[Path]) -> dict[str, tuple[str, Path, Path]]:
    """Map family/variant -> (run_key, job_dir, checkpoint_dir) for complete runs.

    run_key is `<root_name>/<time_dir>`, which sorts chronologically because the
    root name is date-prefixed. Later keys win.
    """
    best: dict[str, tuple[str, Path, Path]] = {}
    incomplete: list[tuple[str, str]] = []

    for root in roots:
        for time_dir in sorted(p for p in root.iterdir() if p.is_dir()):
            for job in sorted(p for p in time_dir.iterdir() if p.is_dir()):
                choice = read_job_model(job)
                if choice is None:
                    continue
                ckpt = job / "sft_only" / "outputs" / "sft" / "checkpoint"
                run_key = f"{root.name}/{time_dir.name}/{job.name}"
                if not (ckpt / "adapter_config.json").exists():
                    incomplete.append((choice, run_key))
                    continue
                prev = best.get(choice)
                if prev is None or run_key > prev[0]:
                    best[choice] = (run_key, job, ckpt)

    if incomplete:
        print(f"Ignoring {len(incomplete)} incomplete run(s) (no adapter_config.json):")
        for choice, run_key in sorted(incomplete):
            print(f"  - {choice:<28} {run_key}")
        print()

    return best


def load_source_model(choice: str) -> tuple[str, dict]:
    """Load the `<family>/<variant>.yaml` a sweep cell was trained from."""
    if "/" not in choice:
        raise ValueError(f"unexpected model choice {choice!r} (expected family/variant)")
    family, variant = choice.split("/", 1)
    src = MODEL_CONF_DIR / family / f"{variant}.yaml"
    if not src.exists():
        raise FileNotFoundError(
            f"source model yaml not found: {src} — needed as template for the LoRA variant"
        )
    doc = yaml.safe_load(src.read_text())
    if "model" not in doc:
        raise ValueError(f"{src} has no top-level 'model' key")
    return family, doc


# Base models whose architecture is Mixture-of-Experts. Only these take vLLM's
# fused-MoE LoRA path, which is what makes eager mode fatal (see build_doc).
# Matched against `model.model_family`.
MOE_MODEL_FAMILIES = {"gpt-oss"}


def family_is_moe(model: dict) -> bool:
    """True if this model's base architecture is MoE.

    Keyed on `model_family` rather than sniffing the checkpoint, so the rule is
    visible in the config rather than implicit in a weights probe. Add a family
    here when introducing another MoE base (e.g. a Qwen3 MoE or DeepSeek).
    """
    return str(model.get("model_family", "")).strip() in MOE_MODEL_FAMILIES


def build_doc(source_doc: dict, checkpoint: Path) -> dict:
    """Add the LoRA wiring to a copy of the source model doc."""
    doc = copy.deepcopy(source_doc)
    model = doc["model"]

    # Insert lora_path right after model_family to match sft-ci.yaml layout.
    new_model: dict = {}
    for k, v in model.items():
        new_model[k] = v
        if k == "model_family":
            new_model["lora_path"] = str(checkpoint)
    if "lora_path" not in new_model:
        new_model["lora_path"] = str(checkpoint)
    doc["model"] = new_model

    eng = new_model.setdefault("engine_kwargs", {})
    eng["enable_lora"] = True
    eng["max_lora_rank"] = 64
    # enforce_eager must stay OFF for MoE base models. vLLM 0.25.0 routes any MoE
    # model through the fused-MoE LoRA manager ("MoE model detected..."), and in
    # EAGER mode the *attention* LoRA shrink kernel is handed a non-contiguous
    # tensor -> `assert inputs.is_contiguous()` (lora_shrink_op.py:182) ->
    # EngineDeadError, before a single token is produced. Isolated 2026-07-18
    # (job 24462): base alone PASSES, +adapter eager FAILS, +adapter compiled
    # PASSES. Killed all 5 benchmarks of gpt-oss-20b/sft-canonical in
    # 2026-07-17_eval_canonical_sft_gemma4. Same bug family as vllm#26976/#28640.
    # Note the <family>/instruct baselines never set enforce_eager either, so
    # leaving it off also keeps the SFT cell engine-comparable with its own
    # zero-shot row. Revisit if vLLM fixes the kernel.
    if family_is_moe(model):
        eng["enforce_eager"] = False
    else:
        eng["enforce_eager"] = True
    # sft-ci.yaml / sft-contentless-v6.yaml use max_num_seqs=16 with LoRA.
    eng["max_num_seqs"] = 16
    return doc


def render_yaml(doc: dict, choice: str, run_key: str, job: Path) -> str:
    body = yaml.safe_dump(doc, sort_keys=False, default_flow_style=False, indent=2)
    # yaml.safe_dump emits `thinking_mode: false` where the source said `off`
    # (YAML 1.1 folds `off` to bool). Restore the documented spelling.
    body = body.replace("thinking_mode: false", "thinking_mode: off")
    return (
        "# @package _global_\n"
        "# Auto-generated by scripts/build_canonical_sft_model_yamls.py — do not edit by hand.\n"
        "#\n"
        f"# {choice} SFT'd on the 2026-07-12 Gemma-4-31B-teacher fiction10 flows\n"
        "# with the contentless-v6 recipe (sft/default.yaml + negative_selection=contentless).\n"
        f"# Source run: {run_key}\n"
        f"# Artifact:   {job}\n"
        + body
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("roots", nargs="*", type=Path, help="Sweep root dirs.")
    ap.add_argument("--write", action="store_true", help="Actually write files.")
    ap.add_argument("--force", action="store_true", help="Overwrite existing YAMLs.")
    args = ap.parse_args(argv)

    roots = args.roots or DEFAULT_ROOTS
    roots = [r if r.is_absolute() else (REPO / r).resolve() for r in roots]
    for root in roots:
        if not root.is_dir():
            print(f"ERROR: not a directory: {root}", file=sys.stderr)
            return 2

    runs = discover_runs(roots)
    if not runs:
        print("ERROR: no complete runs found", file=sys.stderr)
        return 2

    written: list[str] = []
    n_exists = 0
    for choice in sorted(runs):
        run_key, job, ckpt = runs[choice]
        family, source_doc = load_source_model(choice)
        doc = build_doc(source_doc, ckpt.resolve())
        text = render_yaml(doc, choice, run_key, job.resolve())
        out_path = MODEL_CONF_DIR / family / f"{OUT_VARIANT}.yaml"

        if out_path.exists() and not args.force:
            print(f"  EXISTS {out_path.relative_to(REPO)} (use --force to overwrite)")
            n_exists += 1
            continue

        if args.write:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(text)
            print(f"  WROTE  {out_path.relative_to(REPO)}   <- {run_key}")
        else:
            print(f"  PLAN   {out_path.relative_to(REPO)}   <- {run_key}")
        written.append(f"{family}/{OUT_VARIANT}")

    verb = "wrote" if args.write else "would write"
    print(f"\nSummary: {verb} {len(written)} yaml(s); skipped {n_exists} already-exists.")
    if not args.write:
        print("(dry run; pass --write to persist)")
    print("\nSweep model list:")
    print("        " + ",\n        ".join(written))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
