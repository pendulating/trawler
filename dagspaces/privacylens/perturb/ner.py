"""Transformer named-entity detection for cultural perturbation.

Uses a RoBERTa-large token-classification NER model
(``Jean-Baptiste/roberta-large-ner-english``) run through the project's own
``transformers`` stack on GPU when available (falls back to CPU). This avoids
``spacy-transformers``, which pins ``transformers<4.54`` and is incompatible
with the vLLM/training stack (``transformers>=4.56``).

Returns PERSON / GPE / LOC / ORG spans. ORG is returned so callers can *protect*
org/tool names, not perturb them.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass

_MODEL = "Jean-Baptiste/roberta-large-ner-english"

# CoNLL-style labels emitted by the model -> our internal labels. MISC is
# dropped (too noisy to swap). PER/ORG/LOC are the ones we act on.
_LABEL_MAP = {
    "PER": "PERSON",
    "PERSON": "PERSON",
    "ORG": "ORG",
    "LOC": "LOC",
    "GPE": "GPE",
    "MISC": None,
}


@dataclass(frozen=True)
class Entity:
    text: str
    label: str
    start: int
    end: int


@functools.lru_cache(maxsize=1)
def get_nlp():
    """Return a cached HF token-classification NER pipeline (GPU if available).

    Raises a clear, actionable error if transformers/torch or the model weights
    cannot be loaded.
    """
    try:
        import torch
        from transformers import pipeline
    except ImportError as exc:  # pragma: no cover - import guard
        raise ImportError(
            "transformers + torch are required for the cultural-perturbation NER "
            "stage (they ship with the project)."
        ) from exc

    device = 0 if torch.cuda.is_available() else -1
    try:
        return pipeline(
            "token-classification",
            model=_MODEL,
            aggregation_strategy="simple",
            device=device,
        )
    except Exception as exc:
        raise OSError(
            f"Failed to load NER model {_MODEL!r}. The first run downloads the "
            "weights from the HuggingFace hub — ensure network/cache access."
        ) from exc


def detect_entities(text: str, nlp=None) -> list[Entity]:
    """Return mapped entities found in ``text`` (empty for blank input).

    ``nlp`` is a callable returning a list of dicts with ``entity_group`` (or
    ``entity``), ``word``, ``start``, ``end`` — i.e. an HF token-classification
    pipeline (or a compatible fake in tests).
    """
    if not text or not str(text).strip():
        return []
    if nlp is None:
        nlp = get_nlp()
    out: list[Entity] = []
    for r in nlp(str(text)):
        group = str(r.get("entity_group") or r.get("entity") or "").upper()
        label = _LABEL_MAP.get(group)
        if label is None:
            continue
        word = str(r.get("word") or "").strip()
        if not word:
            continue
        out.append(Entity(word, label, int(r.get("start") or 0), int(r.get("end") or 0)))
    return out
