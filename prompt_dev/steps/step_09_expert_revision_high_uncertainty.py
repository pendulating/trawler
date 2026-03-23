from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from prompt_dev.io import now_utc_iso, read_json, write_json
from prompt_dev.schemas import QuestionRecord
from prompt_dev.steps._step_common import run_from_cli
from tqdm.auto import tqdm


def run(ctx) -> dict:
    pbar = tqdm(total=3, desc="step09 expert question revision", unit="task")
    required = ctx.artifact("step_09_Q_star_10.json")
    pbar.update(1)
    if not required.exists():
        ranked = read_json(ctx.artifact("step_08_uncertainty_ranked.json"))["ranked_questions"]
        template_q = [
            {
                "id": item["id"],
                "question": item["question"],
                "rationale": item.get("rationale"),
                "uncertainty": item["uncertainty"],
            }
            for item in ranked[:10]
        ]
        template = {
            "theory": ctx.theory,
            "generated_at": now_utc_iso(),
            "instructions": "Expert review required: pare/merge into 10 final questions.",
            "questions": template_q,
        }
        template_path = ctx.artifact("step_09_Q_star_10_TEMPLATE.json")
        write_json(template_path, template)
        pbar.update(2)
        pbar.close()
        raise RuntimeError(
            "Missing expert artifact step_09_Q_star_10.json. "
            f"Template created at {template_path}."
        )

    payload = read_json(required)
    pbar.update(1)
    questions = payload.get("questions", [])
    if len(questions) != 10:
        pbar.close()
        raise ValueError("step_09_Q_star_10.json must contain exactly 10 questions.")
    for q in tqdm(questions, desc="step09 validate expert questions", unit="question"):
        QuestionRecord.model_validate(q)
    pbar.update(1)
    pbar.close()
    return {"artifact": str(required), "validated": True, "question_count": 10}


if __name__ == "__main__":
    run_from_cli("step09", "Step 09: Expert revision of high-uncertainty questions")

