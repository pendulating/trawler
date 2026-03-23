from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from prompt_dev.io import now_utc_iso, read_json, write_json
from prompt_dev.steps._step_common import run_from_cli
from tqdm.auto import tqdm


def run(ctx) -> dict:
    pbar = tqdm(total=3, desc="step06 expert revision check", unit="task")
    required = ctx.artifact("step_06_K_star.json")
    pbar.update(1)
    if not required.exists():
        overview = read_json(ctx.artifact("step_02_overview_yT.json"))
        defs = read_json(ctx.artifact("step_05_definitions_dT.json"))
        pbar.update(1)
        template = {
            "theory": ctx.theory,
            "generated_at": now_utc_iso(),
            "instructions": (
                "Expert step required: edit this file into step_06_K_star.json with "
                "finalized overview and component definitions."
            ),
            "overview": overview["yT"],
            "definitions": defs["definitions"],
        }
        template_path = ctx.artifact("step_06_K_star_TEMPLATE.json")
        write_json(template_path, template)
        pbar.update(1)
        pbar.close()
        raise RuntimeError(
            "Missing expert artifact step_06_K_star.json. "
            f"Template created at {template_path}."
        )

    payload = read_json(required)
    pbar.update(1)
    if "overview" not in payload or "definitions" not in payload:
        pbar.close()
        raise ValueError("step_06_K_star.json must include keys: overview, definitions")
    pbar.update(1)
    pbar.close()
    return {"artifact": str(required), "validated": True}


if __name__ == "__main__":
    run_from_cli("step06", "Step 06: Expert-in-the-loop revision")

