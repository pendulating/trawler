from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from prompt_dev.io import now_utc_iso, read_json, write_json
from prompt_dev.prompts import p_define
from prompt_dev.schemas import DefinitionRecord
from prompt_dev.steps._step_common import run_from_cli
from tqdm.auto import tqdm


def run(ctx) -> dict:
    overview = read_json(ctx.artifact("step_02_overview_yT.json"))["yT"]
    components = read_json(ctx.artifact("step_03_components_C.json"))["components"]
    sampling = ctx.cfg["sampling"]
    client = ctx.get_client()

    records: list[dict] = []
    pbar = tqdm(total=len(components) + 1, desc="step04 define components", unit="component")
    for component in components:
        component_name = component["name"]
        component_summary = component.get("summary", "")
        out = client.chat(
            p_define(component_name, component_summary, overview, theory=ctx.theory),
            temperature=ctx.temperature(),
            top_p=float(sampling["default_top_p"]),
            max_tokens=int(sampling["default_max_tokens"]),
        )
        rec = DefinitionRecord(component=component_name, draft=out["text"])
        records.append(rec.model_dump())
        pbar.update(1)

    payload = {
        "theory": ctx.theory,
        "generated_at": now_utc_iso(),
        "definitions": records,
    }
    path = ctx.artifact("step_04_definitions_d0.json")
    write_json(path, payload)
    pbar.update(1)
    pbar.close()
    return {"artifact": str(path), "definition_count": len(records)}


if __name__ == "__main__":
    run_from_cli("step04", "Step 04: Recursive component definition")

