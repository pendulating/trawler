from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from prompt_dev.io import now_utc_iso, read_json, write_json
from prompt_dev.prompts import p_feedback, p_refine
from prompt_dev.steps._step_common import run_from_cli
from tqdm.auto import tqdm


def run(ctx) -> dict:
    source = read_json(ctx.artifact("step_01_overview_y0.json"))
    current = source["y0"]
    iterations = []
    t_max = int(ctx.cfg["iterations"]["overview_refine_t"])
    sampling = ctx.cfg["sampling"]
    client = ctx.get_client()

    pbar = tqdm(total=(t_max * 2) + 1, desc="step02 refine overview", unit="task")
    for t in range(t_max):
        feedback_out = client.chat(
            p_feedback(current),
            temperature=ctx.temperature(),
            top_p=float(sampling["default_top_p"]),
            max_tokens=int(sampling["default_max_tokens"]),
        )
        pbar.update(1)
        refined_out = client.chat(
            p_refine(current, feedback_out["text"], theory=ctx.theory),
            temperature=ctx.temperature(),
            top_p=float(sampling["default_top_p"]),
            max_tokens=int(sampling["default_max_tokens"]),
        )
        pbar.update(1)
        iterations.append(
            {
                "iteration": t + 1,
                "feedback": feedback_out["text"],
                "refined": refined_out["text"],
            }
        )
        current = refined_out["text"]

    payload = {
        "theory": ctx.theory,
        "generated_at": now_utc_iso(),
        "t_max": t_max,
        "yT": current,
        "iterations": iterations,
    }
    path = ctx.artifact("step_02_overview_yT.json")
    write_json(path, payload)
    pbar.update(1)
    pbar.close()
    return {"artifact": str(path), "iterations": t_max}


if __name__ == "__main__":
    run_from_cli("step02", "Step 02: Self-refine theoretical overview")

