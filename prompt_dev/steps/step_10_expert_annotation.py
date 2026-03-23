from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from prompt_dev.io import now_utc_iso, read_json, write_json
from prompt_dev.steps._step_common import run_from_cli
from tqdm.auto import tqdm


def run(ctx) -> dict:
    pbar = tqdm(total=3, desc="step10 expert annotation check", unit="task")
    required = ctx.artifact("step_10_A_star_10.json")
    pbar.update(1)
    if not required.exists():
        q_star = read_json(ctx.artifact("step_09_Q_star_10.json"))
        template_answers = []
        for q in tqdm(q_star["questions"], desc="step10 build annotation template", unit="question"):
            template_answers.append(
                {
                    "id": q["id"],
                    "question": q["question"],
                    "answer": "",
                    "notes": "",
                }
            )
        template = {
            "theory": ctx.theory,
            "generated_at": now_utc_iso(),
            "instructions": "Expert annotation required: author ground-truth answers.",
            "answers": template_answers,
        }
        template_path = ctx.artifact("step_10_A_star_10_TEMPLATE.json")
        write_json(template_path, template)
        pbar.update(2)
        pbar.close()
        raise RuntimeError(
            "Missing expert artifact step_10_A_star_10.json. "
            f"Template created at {template_path}."
        )

    payload = read_json(required)
    pbar.update(1)
    answers = payload.get("answers", [])
    if len(answers) != 10:
        pbar.close()
        raise ValueError("step_10_A_star_10.json must contain exactly 10 answers.")
    missing = [a.get("id") for a in answers if not (a.get("answer") or "").strip()]
    if missing:
        pbar.close()
        raise ValueError(f"Expert answers are blank for ids: {missing}")
    pbar.update(1)
    pbar.close()
    return {"artifact": str(required), "validated": True, "answer_count": 10}


if __name__ == "__main__":
    run_from_cli("step10", "Step 10: Expert annotation")

