from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from prompt_dev.io import now_utc_iso, read_json, write_json
from prompt_dev.orchestrator import extract_last_json_object
from prompt_dev.schemas import Component
from prompt_dev.prompts import p_decompose
from prompt_dev.steps._step_common import run_from_cli
from tqdm.auto import tqdm


def _fallback_components(raw_text: str) -> list[dict]:
    lines = [ln.strip("-* \t") for ln in (raw_text or "").splitlines() if ln.strip()]
    candidates = []
    for line in lines[:12]:
        if len(line) < 3:
            continue
        name = line.split(":")[0].strip()
        if name:
            candidates.append({"name": name[:120], "summary": line[:400]})
    return candidates


def run(ctx) -> dict:
    source = read_json(ctx.artifact("step_02_overview_yT.json"))
    overview = source["yT"]
    client = ctx.get_client()
    sampling = ctx.cfg["sampling"]
    pbar = tqdm(total=4, desc="step03 decompose components", unit="task")
    raw = client.chat(
        p_decompose(overview, theory=ctx.theory),
        temperature=ctx.temperature(),
        top_p=float(sampling["default_top_p"]),
        max_tokens=int(sampling["default_max_tokens"]),
    )
    pbar.update(1)
    parsed = extract_last_json_object(raw["text"]) or {}
    components = parsed.get("components", [])
    if not isinstance(components, list) or not components:
        components = _fallback_components(raw["text"])

    validated: list[dict] = []
    for item in tqdm(components, desc="step03 validate components", unit="component"):
        try:
            rec = Component.model_validate(item)
            validated.append(rec.model_dump())
        except Exception:
            continue
    pbar.update(2)
    if not validated:
        raise ValueError("Step 03 failed: no valid components could be parsed.")

    payload = {
        "theory": ctx.theory,
        "generated_at": now_utc_iso(),
        "components": validated,
        "raw_generation": raw["text"],
    }
    path = ctx.artifact("step_03_components_C.json")
    write_json(path, payload)
    pbar.update(1)
    pbar.close()
    return {"artifact": str(path), "component_count": len(validated)}


if __name__ == "__main__":
    run_from_cli("step03", "Step 03: Sub-component decomposition")

