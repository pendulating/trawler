from __future__ import annotations

import argparse
import json
from pathlib import Path

if __package__ in {None, ""}:
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[1]))

try:
    from .orchestrator import run_step
except ImportError:  # direct file execution
    from prompt_dev.orchestrator import run_step


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prompt development cycle runner.")
    parser.add_argument("--step", required=True, help="Step id: 1..12, step1, or step01")
    parser.add_argument("--theory", required=True, choices=["CI", "IG"])
    parser.add_argument("--config", default=None)
    parser.add_argument("--run-id", default=None)
    return parser


def _normalize_step(step: str) -> str:
    value = (step or "").strip().lower()
    if value.isdigit():
        num = int(value)
        if 1 <= num <= 12:
            return f"step{num:02d}"
    if value.startswith("step"):
        suffix = value[4:]
        if suffix.isdigit():
            num = int(suffix)
            if 1 <= num <= 12:
                return f"step{num:02d}"
    raise ValueError(f"Invalid --step value '{step}'. Use 1..12, step1, or step01.")


def _resolve_config_path(user_config: str | None) -> str:
    candidates: list[Path] = []
    if user_config:
        candidates.append(Path(user_config))
    else:
        candidates.append(Path("prompt_dev/conf/config.yaml"))
        candidates.append(Path("conf/config.yaml"))
        candidates.append(Path(__file__).resolve().parent / "conf" / "config.yaml")
    for cand in candidates:
        if cand.exists():
            return str(cand)
    raise FileNotFoundError("Could not find config file. Pass --config explicitly.")


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    step_key = _normalize_step(args.step)
    config_path = _resolve_config_path(args.config)
    out = run_step(step_key=step_key, theory=args.theory, config_path=config_path, run_id=args.run_id)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()

