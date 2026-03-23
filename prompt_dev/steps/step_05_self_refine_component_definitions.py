from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from prompt_dev.io import now_utc_iso, read_json, write_json
from prompt_dev.prompts import p_feedback, p_refine
from prompt_dev.schemas import DefinitionRecord
from prompt_dev.steps._step_common import run_from_cli
from tqdm.auto import tqdm


def run(ctx) -> dict:
    step4 = read_json(ctx.artifact("step_04_definitions_d0.json"))
    records = step4["definitions"]
    t_max = int(ctx.cfg["iterations"]["definition_refine_t"])
    sampling = ctx.cfg["sampling"]
    client = ctx.get_client()

    refined_records: list[dict] = []
    total_actions = len(records) * max(t_max, 1) * 2 + 1
    pbar = tqdm(total=total_actions, desc="step05 refine definitions", unit="task")
    for rec in records:
        current = rec["draft"]
        feedback_iterations = []
        for t in range(t_max):
            fb = client.chat(
                p_feedback(current),
                temperature=ctx.temperature(),
                top_p=float(sampling["default_top_p"]),
                max_tokens=int(sampling["default_max_tokens"]),
            )
            pbar.update(1)
            refined = client.chat(
                p_refine(current, fb["text"], theory=ctx.theory),
                temperature=ctx.temperature(),
                top_p=float(sampling["default_top_p"]),
                max_tokens=int(sampling["default_max_tokens"]),
            )
            pbar.update(1)
            current = refined["text"]
            feedback_iterations.append({"iteration": t + 1, "feedback": fb["text"], "refined": current})

        out_rec = DefinitionRecord(
            component=rec["component"],
            draft=rec["draft"],
            refined=current,
            feedback_iterations=feedback_iterations,
        )
        refined_records.append(out_rec.model_dump())

    payload = {
        "theory": ctx.theory,
        "generated_at": now_utc_iso(),
        "t_max": t_max,
        "definitions": refined_records,
    }
    path = ctx.artifact("step_05_definitions_dT.json")
    write_json(path, payload)
    pbar.update(1)
    pbar.close()
    return {"artifact": str(path), "definition_count": len(refined_records), "iterations": t_max}


if __name__ == "__main__":
    run_from_cli("step05", "Step 05: Self-refine component definitions")

