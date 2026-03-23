from __future__ import annotations

import json

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from prompt_dev.io import now_utc_iso, read_json, write_json
from prompt_dev.prompts import p_cot
from prompt_dev.schemas import GroundedAnswer
from prompt_dev.steps._step_common import run_from_cli
from tqdm.auto import tqdm


def _load_exemplars(ctx) -> str:
    local_path = ctx.artifact("step_11_exemplars.json")
    if local_path.exists():
        return json.dumps(read_json(local_path), ensure_ascii=True)
    configured = ctx.cfg.get("exemplars_path")
    if configured:
        from pathlib import Path

        p = Path(str(configured))
        if p.exists():
            return p.read_text(encoding="utf-8")
    return "[]"


def run(ctx) -> dict:
    k_star = read_json(ctx.artifact("step_06_K_star.json"))
    q_star = read_json(ctx.artifact("step_09_Q_star_10.json"))
    k_star_text = json.dumps(k_star, ensure_ascii=True)
    exemplars_text = _load_exemplars(ctx)
    sampling = ctx.cfg["sampling"]
    client = ctx.get_client()

    answers: list[dict] = []
    pbar = tqdm(total=len(q_star["questions"]) + 1, desc="step11 grounded inference", unit="question")
    for q in q_star["questions"]:
        prompt = p_cot(k_star_text, exemplars_text, q["question"], theory=ctx.theory)
        out = client.chat(
            prompt,
            temperature=ctx.temperature(),
            top_p=float(sampling["default_top_p"]),
            max_tokens=int(sampling["default_max_tokens"]),
        )
        rec = GroundedAnswer(
            question_id=q["id"],
            question=q["question"],
            answer_hat=out["text"],
            reasoning_trace=out["text"],
        )
        answers.append(rec.model_dump())
        pbar.update(1)

    payload = {
        "theory": ctx.theory,
        "generated_at": now_utc_iso(),
        "answers_hat": answers,
    }
    path = ctx.artifact("step_11_grounded_answers_hatA.json")
    write_json(path, payload)
    pbar.update(1)
    pbar.close()
    return {"artifact": str(path), "answer_count": len(answers)}


if __name__ == "__main__":
    run_from_cli("step11", "Step 11: Grounded inference with CoT")

