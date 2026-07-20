"""Layered person-name detection for norm / flow extraction QA.

Replaces the manually curated per-book character blocklist as the *primary*
detector (2026-06-09). The hand-list in ``norm_extraction.py`` covered only
the 10-novel COLM corpus; on a larger corpus (e.g. the 100-novel run) QA
coverage would have silently degraded to the title-pattern regex for every
new book. Three layers, cheapest first:

1. **blocklist** — explicit names from ``cfg.norm_quality.character_blocklist``
   plus the built-in 10-novel list. Kept for aliases and fictional places
   NER cannot see ("big brother", "monte cristo", "pemberley"). Matched
   case-insensitively, *except* for entries in ``AMBIGUOUS_NAMES`` (names that
   are also ordinary English words — "May", "Will"), which require the
   capitalized form. See that constant for why.
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

TITLE_PATTERN = re.compile(
    r"\b(?:Mr\.|Mrs\.|Miss|Ms\.|Lady|Lord|Sir|Reverend|Rev\.|Colonel|Col\.)"
    r"\s+[A-Z][a-z]+"
)

# Blocklist entries that are also ordinary English words. These are matched
# CASE-SENSITIVELY (the capitalized proper-noun form only); every other entry
# stays case-insensitive.
#
# Why (2026-07-13): the built-in blocklist carries "will" (Will Ladislaw,
# *Middlemarch*) and "may" (May Welland, *The Age of Innocence*), and it is a
# GLOBAL list applied to all books with a case-insensitive match. So the modal
# verbs in "a servant may not disclose..." and "the heir will inherit..." tripped
# the gate in every novel. On fiction10 that was 373 of 440 flagged norms — 85%
# of `norm_quality_passed == False` was modal verbs, making true leakage look
# like 4.39% when it was 0.32%.
#
# Requiring the capitalized form fixes it: "a servant may not" is clean, while
# "May refuses Archer" still flags. Sentence-initial "May..." can still false-
# positive, but norm text is declarative and this is an advisory flag that drops
# no rows. Entries listed here take effect only if they are actually in a
# blocklist, so seeding likely collisions is free insurance for larger corpora.
AMBIGUOUS_NAMES = frozenset({
    "may", "will", "grace", "hope", "faith", "rose", "mark", "frank", "rich",
    "bill", "art", "dawn", "joy", "prudence", "patience", "charity", "constance",
    "victor", "earnest", "pip", "sue", "wilder",
})

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
        blocklist: set[str] | None = None,
        use_ner: bool = True,
    ):
        names = sorted({n.lower() for n in (blocklist or set())})
        # (canonical_name, pattern, case_sensitive). Ambiguous entries compile
        # against the Title-Cased form and are searched in the ORIGINAL text, so
        # the proper noun matches and the common word does not.
        self._block_patterns = [
            (
                n,
                re.compile(r"\b" + re.escape(n.title() if n in AMBIGUOUS_NAMES else n) + r"\b"),
                n in AMBIGUOUS_NAMES,
            )
            for n in names
        ]
        self.use_ner = use_ner

    def detect(self, text: str | None) -> dict[str, list[str]]:
        """Return matches per layer: ``{"blocklist", "titled", "person_entity"}``.

        Empty lists everywhere means the text is person-free as far as the
        active layers can tell.
        """
        out: dict[str, list[str]] = {
            "blocklist": [], "titled": [], "person_entity": [],
        }
        if not text:
            return out

        lowered = text.lower()
        for name, pattern, case_sensitive in self._block_patterns:
            if pattern.search(text if case_sensitive else lowered):
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

    def field_flags(self, field_name: str, text: str | None) -> list[str]:
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
