from __future__ import annotations

import json

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from prompt_dev.io import now_utc_iso, read_json, write_json
from prompt_dev.orchestrator import extract_last_json_object
from prompt_dev.prompts import p_question_gen
from prompt_dev.schemas import QuestionRecord
from prompt_dev.steps._step_common import run_from_cli
from tqdm.auto import tqdm


def run(ctx) -> dict:
    k_star = read_json(ctx.artifact("step_06_K_star.json"))
    k_star_text = json.dumps(k_star, ensure_ascii=True)
    sampling = ctx.cfg["sampling"]
    client = ctx.get_client()
    pbar = tqdm(total=3, desc="step07 generate questions", unit="task")
    out = client.chat(
        p_question_gen(k_star_text, theory=ctx.theory),
        temperature=ctx.temperature(),
        top_p=float(sampling["default_top_p"]),
        max_tokens=int(sampling["default_max_tokens"]),
    )
    pbar.update(1)
    parsed = extract_last_json_object(out["text"]) or {}
    questions = parsed.get("questions", [])
    if not isinstance(questions, list):
        questions = []

    validated: list[dict] = []
    for i, q in enumerate(tqdm(questions[:50], desc="step07 validate questions", unit="question"), start=1):
        if isinstance(q, dict):
            q.setdefault("id", f"q{i}")
        try:
            qr = QuestionRecord.model_validate(q)
            validated.append(qr.model_dump())
        except Exception:
            continue
    pbar.update(1)
    if len(validated) < 10:
        raise ValueError("Step 07 produced too few valid questions; expected >=10.")

    payload = {
        "theory": ctx.theory,
        "generated_at": now_utc_iso(),
        "questions": validated[:50],
        "raw_generation": out["text"],
    }
    path = ctx.artifact("step_07_question_pool_50.json")
    write_json(path, payload)
    pbar.update(1)
    pbar.close()
    return {"artifact": str(path), "question_count": len(payload["questions"])}


if __name__ == "__main__":
    run_from_cli("step07", "Step 07: Generate question pool")

