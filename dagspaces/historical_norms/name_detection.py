"""Layered person-name detection for norm / flow extraction QA.

Replaces the manually curated per-book character blocklist as the *primary*
detector (2026-06-09). The hand-list in ``norm_extraction.py`` covered only
the 10-novel COLM corpus; on a larger corpus (e.g. the 100-novel run) QA
coverage would have silently degraded to the title-pattern regex for every
new book. Three layers, cheapest first:

1. **blocklist** — explicit names from ``cfg.norm_quality.character_blocklist``
   plus the built-in 10-novel list. Kept for aliases and fictional places
   NER cannot see ("big brother", "monte cristo", "pemberley").
2. **titled** — ``Mr./Mrs./Lady/... + ProperNoun`` regex.
3. **person_entity** — spaCy ``en_core_web_sm`` PERSON entities. Corpus-
   agnostic: scales to arbitrary novels with zero manual curation.

Only PERSON entities block: real-world places/orgs in a norm ("England",
"the Church") are often legitimate societal context, so GPE/ORG/LOC are
deliberately not flagged. If spaCy or its model is unavailable the detector
degrades to layers 1–2 with one loud warning — QA must never crash a stage.
"""

from __future__ import annotations

import re
from typing import Dict, List, Optional, Set

TITLE_PATTERN = re.compile(
    r"\b(?:Mr\.|Mrs\.|Miss|Ms\.|Lady|Lord|Sir|Reverend|Rev\.|Colonel|Col\.)"
    r"\s+[A-Z][a-z]+"
)

_NLP = None
_NER_UNAVAILABLE = False


def _get_nlp():
    """Lazy spaCy singleton (NER components only); None when unavailable."""
    global _NLP, _NER_UNAVAILABLE
    if _NLP is None and not _NER_UNAVAILABLE:
        try:
            import spacy
            _NLP = spacy.load(
                "en_core_web_sm",
                disable=["tagger", "parser", "attribute_ruler", "lemmatizer"],
            )
        except Exception as e:
            _NER_UNAVAILABLE = True
            print(
                f"[name_detection] WARNING: spaCy NER unavailable ({e}). "
                f"Falling back to blocklist + title-pattern checks only — "
                f"unlisted character names will NOT be flagged."
            )
    return _NLP


def _plausible_person(ent, text: str) -> bool:
    """Precision filter for en_core_web_sm PERSON entities on norm text.

    Measured on the fiction10 abstracted norms, the raw small model tags
    sentence-initial capitalized common nouns ("Law", "Citizens",
    "Children") and stray lowercase words ("inn") as PERSON. Filter:
    drop lowercase entities, and drop single-token entities in
    sentence-initial position. Trade-off (documented, accepted): a real
    single name that ONLY ever appears sentence-initially is missed —
    multi-token names ("Martin Verga") are always kept, and recurring
    names also appear mid-sentence in other norms of the same book.
    """
    t = ent.text.strip()
    if not t or not t[0].isupper():
        return False
    if " " not in t:
        prefix = text[: ent.start_char].rstrip()
        if not prefix or prefix.endswith((".", "!", "?", "|", ":", ";")):
            return False
    return True


class PersonNameDetector:
    """Detect references to specific named persons in short norm/flow text."""

    def __init__(
        self,
        blocklist: Optional[Set[str]] = None,
        use_ner: bool = True,
    ):
        names = sorted({n.lower() for n in (blocklist or set())})
        self._block_patterns = [
            (n, re.compile(r"\b" + re.escape(n) + r"\b")) for n in names
        ]
        self.use_ner = use_ner

    def detect(self, text: Optional[str]) -> Dict[str, List[str]]:
        """Return matches per layer: ``{"blocklist", "titled", "person_entity"}``.

        Empty lists everywhere means the text is person-free as far as the
        active layers can tell.
        """
        out: Dict[str, List[str]] = {
            "blocklist": [], "titled": [], "person_entity": [],
        }
        if not text:
            return out

        lowered = text.lower()
        for name, pattern in self._block_patterns:
            if pattern.search(lowered):
                out["blocklist"].append(name)

        for match in TITLE_PATTERN.finditer(text):
            out["titled"].append(match.group(0))

        if self.use_ner:
            nlp = _get_nlp()
            if nlp is not None:
                for ent in nlp(text).ents:
                    if ent.label_ == "PERSON" and _plausible_person(ent, text):
                        out["person_entity"].append(ent.text)
        return out

    def field_flags(self, field_name: str, text: Optional[str]) -> List[str]:
        """Flags for one field, in the established ``norm_quality_flags`` format.

        ``named_char_in_<field>:<name>`` (blocklist),
        ``titled_name_in_<field>`` (regex),
        ``person_entity_in_<field>:<name>`` (NER).
        """
        found = self.detect(text)
        flags = [f"named_char_in_{field_name}:{n}" for n in found["blocklist"]]
        if found["titled"]:
            flags.append(f"titled_name_in_{field_name}")
        flags.extend(
            f"person_entity_in_{field_name}:{n}" for n in found["person_entity"]
        )
        return flags
