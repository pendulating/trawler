"""Alias-set construction and safe, consistent name/location substitution.

The hard part of the cultural perturbation. Given person strings (from the NER model +
structured name slots) and location spans, build one per-record replacement map
and apply it across every field with a **single-pass**, longest-match-first,
word-boundary regex that skips protected tokens. Single-pass substitution
prevents A→B→C cascade bugs (a replacement that happens to contain an original
token is never re-substituted).
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Optional

from .name_bank import CultureBank, ReplacementIdentity, infer_gender

_HONORIFICS = frozenset(
    {
        "mr", "mrs", "ms", "miss", "mx", "dr", "prof", "professor",
        "sir", "madam", "rev", "reverend", "fr", "capt", "lt", "sgt",
        "gen", "col",
        # ecclesiastical / familial titles that precede a given name
        "father", "mother", "sister", "brother", "rabbi", "imam",
        "pastor", "bishop", "deacon", "elder", "aunt", "uncle",
    }
)

# Token must look like a name component: starts with an uppercase letter,
# rest are letters/apostrophe/hyphen. Filters role words ("the", "defendant").
_NAME_TOKEN = re.compile(r"^[A-Z][A-Za-z'\-]*$")

# Capitalized words the NER model can mis-tag as PERSON in this agentic dataset
# (platform/app names, imperative verbs, email boilerplate). A candidate's name
# run stops at any of these so they are never minted as names or substituted.
_STOP_TOKENS = frozenset(
    {
        # platforms / apps / channels
        "messenger", "slack", "facebook", "notion", "gmail", "outlook",
        "whatsapp", "twitter", "instagram", "linkedin", "zoom", "reddit",
        "telegram", "discord", "venmo", "paypal", "messages", "message",
        "email", "inbox", "channel",
        # imperative verbs / agentic + email boilerplate often title-cased
        "respond", "reply", "send", "search", "create", "read", "post",
        "share", "note", "notes", "hello", "hi", "dear", "best", "regards",
        "thanks", "thank", "subject", "action", "observation", "input",
        "confession", "re", "fwd",
    }
)


@dataclass
class PersonIdentity:
    """A clustered person: canonical first/last name + the case-folded tokens."""

    canonical_first: Optional[str] = None
    canonical_last: Optional[str] = None
    honorific: Optional[str] = None
    token_set: set[str] = field(default_factory=set)


def _normalize_candidate(raw: str) -> tuple[Optional[str], list[str]]:
    """Split a raw person string into (honorific, name-tokens).

    Takes the maximal *leading* run of name-like tokens, stopping at the first
    non-name token or stopword. This collapses noisy NER spans like
    "Mary on Messenger" → ["Mary"] and drops false positives like "Respond".
    """
    s = str(raw).strip()
    if not s:
        return None, []
    honorific = None
    m = re.match(r"^([A-Za-z]+)\.?\s+(.*)$", s)
    if m and m.group(1).lower() in _HONORIFICS:
        honorific = m.group(1)
        s = m.group(2).strip()
    for poss in ("'s", "’s"):
        if s.endswith(poss):
            s = s[: -len(poss)]
            break
    s = s.strip(" .,:;\"'()[]’")
    tokens: list[str] = []
    for tok in re.split(r"\s+", s):
        if not _NAME_TOKEN.match(tok) or tok.lower() in _STOP_TOKENS:
            break
        tokens.append(tok)
    return honorific, tokens


def looks_like_name(s: str) -> bool:
    """True if ``s`` is a plausible 1-3 token Title-cased personal name.

    Used to gate structured slots (e.g. ``data_subject_concrete`` is sometimes
    a role like "the defendant", not a name).
    """
    if not s:
        return False
    _, tokens = _normalize_candidate(s)
    return 1 <= len(tokens) <= 3 and len(tokens) == len(str(s).strip().split())


def build_person_identities(person_strings: list[str]) -> list[PersonIdentity]:
    """Cluster raw person strings into identities sharing a name token.

    Clusters by shared (case-folded) token, then assigns canonical first/last:
    a multi-token candidate defines them directly; a single-token cluster is
    typed as a given name when the table knows it, as a surname when an
    honorific is present, else defaults to a given name (narrative single-name
    references are usually given names).
    """
    parsed: list[tuple[Optional[str], list[str]]] = []
    for raw in person_strings:
        hon, toks = _normalize_candidate(raw)
        if toks:
            parsed.append((hon, toks))
    parsed.sort(key=lambda x: -len(x[1]))

    clusters: list[dict] = []
    for hon, toks in parsed:
        low = {t.lower() for t in toks}
        cluster = next((c for c in clusters if low & c["token_set"]), None)
        if cluster is None:
            cluster = {"token_set": set(), "honorific": None, "multi": [], "singles": []}
            clusters.append(cluster)
        cluster["token_set"] |= low
        if hon and not cluster["honorific"]:
            cluster["honorific"] = hon
        if len(toks) >= 2:
            cluster["multi"].append(tuple(toks))
        else:
            cluster["singles"].append(toks[0])

    identities: list[PersonIdentity] = []
    for cluster in clusters:
        first = last = None
        if cluster["multi"]:
            best = max(cluster["multi"], key=len)
            first, last = best[0], best[-1]
        elif cluster["singles"]:
            tok = cluster["singles"][0]
            if infer_gender(tok) in ("m", "f"):
                first = tok
            elif cluster["honorific"]:
                last = tok
            else:
                first = tok
        identities.append(
            PersonIdentity(
                canonical_first=first,
                canonical_last=last,
                honorific=cluster["honorific"],
                token_set=set(cluster["token_set"]),
            )
        )
    return identities


def choose_replacement_identity(
    identity: PersonIdentity,
    bank: CultureBank,
    record_id: str,
    culture: str,
    taken_first: set[str],
    taken_last: set[str],
) -> ReplacementIdentity:
    """Pick a gender-preserving, collision-free replacement, seeded by record."""
    if bank.is_identity():
        return ReplacementIdentity(identity.canonical_first, identity.canonical_last)
    key = (identity.canonical_last or identity.canonical_first or "").lower()
    rng = random.Random(f"{record_id}|{culture}|{key}")
    gender = infer_gender(identity.canonical_first or "")
    first = (
        bank.pick_first(gender, rng, taken_first)
        if identity.canonical_first
        else None
    )
    last = bank.pick_surname(rng, taken_last) if identity.canonical_last else None
    return ReplacementIdentity(first=first, last=last)


def _expand_person_aliases(
    identity: PersonIdentity, repl: ReplacementIdentity
) -> list[tuple[str, str]]:
    """Generate (source, target) surface-form pairs for one person."""
    of, ol = identity.canonical_first, identity.canonical_last
    rf, rl = repl.first, repl.last
    hon = identity.honorific
    pairs: list[tuple[str, str]] = []

    def add(s: Optional[str], t: Optional[str]) -> None:
        if s and t:
            pairs.append((s, t))

    if of and ol and rf and rl:
        add(f"{of} {ol}", f"{rf} {rl}")
        add(f"{of} {ol}'s", f"{rf} {rl}'s")
    if of and rf:
        add(of, rf)
        add(f"{of}'s", f"{rf}'s")
    if ol and rl:
        add(ol, rl)
        add(f"{ol}'s", f"{rl}'s")
        if hon:
            hb = hon.rstrip(".")
            add(f"{hb} {ol}", f"{hb} {rl}")
            add(f"{hb}. {ol}", f"{hb}. {rl}")
    # Email/handle local parts, lowercased, derived from the chosen name so
    # they stay consistent with the visible name swap.
    if of and ol and rf and rl:
        ofl, oll, rfl, rll = of.lower(), ol.lower(), rf.lower(), rl.lower()
        for sep in (".", "_", ""):
            add(f"{ofl}{sep}{oll}", f"{rfl}{sep}{rll}")
        add(f"@{ofl}{oll}", f"@{rfl}{rll}")
    elif of and rf:
        add(of.lower(), rf.lower())
    elif ol and rl:
        add(ol.lower(), rl.lower())
    return pairs


def build_replacement_map(
    person_pairs: list[tuple[PersonIdentity, ReplacementIdentity]],
    location_pairs: list[tuple[str, str]],
    protected: set[str],
) -> list[tuple[str, str]]:
    """Merge person + location substitutions into a longest-source-first list.

    Protected strings are never used as keys. Identity (no-op) and empty entries
    are dropped so the compiled regex only carries effective substitutions.
    """
    pairs: dict[str, str] = {}
    for identity, repl in person_pairs:
        for src, tgt in _expand_person_aliases(identity, repl):
            if not src or src == tgt or src in protected:
                continue
            pairs.setdefault(src, tgt)
    for src, tgt in location_pairs:
        if not src or src == tgt or src in protected:
            continue
        pairs.setdefault(src, tgt)
    return sorted(pairs.items(), key=lambda kv: -len(kv[0]))


def compile_map(ordered_map: list[tuple[str, str]]):
    """Compile an ordered map into a (pattern, lookup) pair (or (None, {}))."""
    if not ordered_map:
        return None, {}
    lookup = {src: tgt for src, tgt in ordered_map}
    alternation = "|".join(re.escape(src) for src, _ in ordered_map)
    pattern = re.compile(
        rf"(?<![A-Za-z0-9_])(?:{alternation})(?![A-Za-z0-9_])"
    )
    return pattern, lookup


def apply_map(text, pattern, lookup) -> tuple[str, int]:
    """Single-pass substitute ``text``; return (new_text, n_substitutions)."""
    if text is None or pattern is None:
        return text, 0
    count = 0

    def _repl(m: re.Match) -> str:
        nonlocal count
        s = m.group(0)
        t = lookup.get(s)
        if t is None or t == s:
            return s
        count += 1
        return t

    return pattern.sub(_repl, str(text)), count
