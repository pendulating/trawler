from __future__ import annotations

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from prompt_dev.io import now_utc_iso, read_json, write_json
from prompt_dev.schemas import QuestionRecord, SampledAnswer
from prompt_dev.steps._step_common import run_from_cli
from prompt_dev.uncertainty import entropy_from_answers
from tqdm.auto import tqdm


def run(ctx) -> dict:
    source = read_json(ctx.artifact("step_07_question_pool_50.json"))
    questions = source["questions"]
    sampling = ctx.cfg["sampling"]
    k = int(sampling["active_prompt_k"])
    temp = ctx.temperature(fallback=float(sampling["active_prompt_temperature"]))
    client = ctx.get_client()

    ranked: list[dict] = []
    total_samples = len(questions) * max(k, 1)
    pbar = tqdm(total=total_samples + 1, desc="step08 uncertainty sampling", unit="sample")
    for q in questions:
        question = QuestionRecord.model_validate(q)
        samples: list[str] = []
        sample_records: list[dict] = []
        for i in range(k):
            out = client.chat(
                f"Answer this question with careful reasoning:\n\n{question.question}",
                temperature=temp,
                top_p=float(sampling["default_top_p"]),
                max_tokens=int(sampling["default_max_tokens"]),
            )
            ans = out["text"]
            samples.append(ans)
            sample_records.append(SampledAnswer(sample_id=i + 1, answer=ans).model_dump())
            pbar.update(1)
        uncertainty = entropy_from_answers(samples)
        ranked.append(
            {
                "id": question.id,
                "question": question.question,
                "rationale": question.rationale,
                "uncertainty": uncertainty,
                "samples": sample_records,
            }
        )

    ranked.sort(key=lambda r: r["uncertainty"], reverse=True)
    payload = {
        "theory": ctx.theory,
        "generated_at": now_utc_iso(),
        "k_samples": k,
        "temperature": temp,
        "ranked_questions": ranked,
    }
    path = ctx.artifact("step_08_uncertainty_ranked.json")
    write_json(path, payload)
    pbar.update(1)
    pbar.close()
    return {"artifact": str(path), "question_count": len(ranked), "k_samples": k}


if __name__ == "__main__":
    run_from_cli("step08", "Step 08: Uncertainty estimation")

