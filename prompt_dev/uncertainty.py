from __future__ import annotations

import math
import re
from collections import Counter


def canonicalize_answer(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[^\w\s]", "", t)
    return t


def entropy_from_answers(answers: list[str]) -> float:
    if not answers:
        return 0.0
    canon = [canonicalize_answer(a) for a in answers if (a or "").strip()]
    if not canon:
        return 0.0
    counts = Counter(canon)
    total = sum(counts.values())
    entropy = 0.0
    for count in counts.values():
        p = count / total
        entropy -= p * math.log(p)
    return float(entropy)

