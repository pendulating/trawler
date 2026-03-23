from __future__ import annotations

import re
from collections import Counter

if __package__ in {None, ""}:
    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from prompt_dev.io import now_utc_iso, read_json, write_json
from prompt_dev.schemas import AlignmentRecord
from prompt_dev.steps._step_common import run_from_cli
from tqdm.auto import tqdm


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (text or "").lower())


def _jaccard_like(a: str, b: str) -> float:
    tok_a = Counter(_tokenize(a))
    tok_b = Counter(_tokenize(b))
    if not tok_a or not tok_b:
        return 0.0
    overlap = 0
    for token, c in tok_a.items():
        overlap += min(c, tok_b.get(token, 0))
    total = sum(tok_a.values()) + sum(tok_b.values()) - overlap
    if total <= 0:
        return 0.0
    return overlap / total


def run(ctx) -> dict:
    hat = read_json(ctx.artifact("step_11_grounded_answers_hatA.json"))["answers_hat"]
    gold = read_json(ctx.artifact("step_10_A_star_10.json"))["answers"]
    k_star = read_json(ctx.artifact("step_06_K_star.json"))
    gold_by_id = {x["id"]: x for x in gold}

    records: list[dict] = []
    caution_notes: list[dict] = []
    pbar = tqdm(total=len(hat) + 3, desc="step12 alignment assessment", unit="task")
    for h in hat:
        qid = h["question_id"]
        g = gold_by_id.get(qid)
        if not g:
            pbar.update(1)
            continue
        score = float(_jaccard_like(h["answer_hat"], g["answer"]))
        note = (
            "High alignment with expert answer."
            if score >= 0.45
            else "Potential divergence; inspect conceptual mapping and definitions."
        )
        record = AlignmentRecord(question_id=qid, score=score, notes=note)
        records.append(record.model_dump())
        if score < 0.45:
            caution_notes.append(
                {
                    "question_id": qid,
                    "question": h["question"],
                    "model_answer": h["answer_hat"],
                    "expert_answer": g["answer"],
                    "note": "Potential mismatch between generated reasoning and expert consensus.",
                }
            )
        pbar.update(1)

    report = {
        "theory": ctx.theory,
        "generated_at": now_utc_iso(),
        "alignment": records,
        "mean_score": (sum(r["score"] for r in records) / len(records)) if records else 0.0,
    }
    notes_payload = {
        "theory": ctx.theory,
        "generated_at": now_utc_iso(),
        "caution_notes": caution_notes,
    }
    updated = dict(k_star)
    updated["caution_notes"] = caution_notes
    updated["updated_at"] = now_utc_iso()

    report_path = ctx.artifact("step_12_alignment_report.json")
    notes_path = ctx.artifact("step_12_caution_notes_candidates.json")
    updated_path = ctx.artifact("K_star_updated.json")
    write_json(report_path, report)
    pbar.update(1)
    write_json(notes_path, notes_payload)
    pbar.update(1)
    write_json(updated_path, updated)
    pbar.update(1)
    pbar.close()
    return {
        "alignment_report": str(report_path),
        "caution_notes": str(notes_path),
        "k_star_updated": str(updated_path),
        "records": len(records),
    }


if __name__ == "__main__":
    run_from_cli("step12", "Step 12: Alignment assessment")

