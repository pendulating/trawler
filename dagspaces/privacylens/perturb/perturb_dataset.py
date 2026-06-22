"""Stage entrypoint: produce a culturally-perturbed copy of a PrivacyLens frame.

Reads the normalized PrivacyLens parquet (columns ``S`` (seed JSON), ``V``
(vignette JSON), ``T`` (trajectory dict), ``record_id``), detects person and
location entities, and applies one deterministic per-record replacement map
across every text field so the vignette stays internally coherent. ``western``
is an identity passthrough (byte-identical output).

Adds audit columns: ``culture``, ``n_persons_swapped``, ``n_locations_swapped``,
``perturb_map_json``.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pandas as pd

from .name_bank import get_bank
from .ner import Entity, detect_entities
from .substitute import (
    _HONORIFICS,
    PersonIdentity,
    apply_map,
    build_person_identities,
    build_replacement_map,
    choose_replacement_identity,
    compile_map,
    looks_like_name,
)

_HONORIFIC_BEFORE = re.compile(r"([A-Za-z]{2,})\.?\s+$")


def _with_recovered_honorific(text: str, ent: Entity, source: str) -> str:
    """Prepend an honorific to ``ent.text`` if one immediately precedes it.

    The NER model strips the honorific from the span ("Mr. Smithson" →
    "Smithson"); recovering it lets the surname be typed correctly.
    """
    pre = source[max(0, ent.start - 14): ent.start]
    m = _HONORIFIC_BEFORE.search(pre)
    if m and m.group(1).lower() in _HONORIFICS:
        return f"{m.group(1)}. {text}"
    return text

# JSON keys embedded in trajectory Action Input / Observation blocks that must
# never be treated as substitution sources (belt-and-suspenders; they are
# lowercase and would not be detected as PERSON/GPE anyway).
_PROTECTED_JSON_KEYS = frozenset(
    {
        "to", "cc", "bcc", "subject", "body", "message", "keywords", "results",
        "recipient_id", "content", "query", "title", "name", "id", "from",
        "date", "time", "status", "text", "url", "attachments",
    }
)

# Scalar S/V fields that may hold a person name and should seed identities even
# if the NER model misses them in free text.
_NAME_SLOTS = (
    ("S", "data_sender_name"),
    ("S", "data_subject"),
    ("S", "data_sender"),
    ("V", "data_sender_concrete"),
    ("V", "data_subject_concrete"),
    ("V", "data_recipient_concrete"),
    ("T", "user_name"),
)

# T subfields that receive the map. final_action (a tool name) and toolkits are
# protected and left untouched.
_T_TEXT_FIELDS = ("user_name", "user_email", "user_instruction", "executable_trajectory")


def _load_obj(raw: Any) -> tuple[dict, str]:
    """Parse an S/V cell into (dict, kind) where kind records the encoding."""
    if isinstance(raw, dict):
        return dict(raw), "dict"
    if raw is None:
        return {}, "none"
    text = str(raw).strip()
    if not text:
        return {}, "empty"
    try:
        obj = json.loads(text)
        return (obj if isinstance(obj, dict) else {}), "json"
    except Exception:
        return {}, "raw"


def _dump_obj(obj: dict, kind: str, original: Any) -> Any:
    """Re-encode an S/V dict back to its original representation."""
    if kind == "json":
        return json.dumps(obj, ensure_ascii=False)
    if kind == "dict":
        return obj
    return original


def _as_str_list(items: Any) -> list[str]:
    """Normalize a list-ish cell (list / numpy array / scalar) to list[str]."""
    if items is None:
        return []
    try:
        seq = list(items)
    except TypeError:
        s = str(items)
        return [s] if s.strip() else []
    return [str(x) for x in seq]


def _extract_t(raw: Any) -> dict:
    if isinstance(raw, dict):
        return dict(raw)
    obj, _ = _load_obj(raw)
    return obj


def perturb_row(
    row: dict, culture: str, bank, nlp
) -> tuple[Any, Any, dict, int, int, dict]:
    """Perturb one record. Returns (S, V, T, n_persons, n_locations, map)."""
    s_obj, s_kind = _load_obj(row.get("S"))
    v_obj, v_kind = _load_obj(row.get("V"))
    t_obj = _extract_t(row.get("T"))
    record_id = str(row.get("record_id", ""))

    objs = {"S": s_obj, "V": v_obj, "T": t_obj}

    # --- gather text for NER + protected tokens -----------------------------
    sensitive = _as_str_list(t_obj.get("sensitive_info_items"))
    toolkits = _as_str_list(t_obj.get("toolkits"))
    final_action = str(t_obj.get("final_action") or "")

    # NER runs over clean prose only (story / instruction / each secret) — NOT
    # the executable_trajectory, whose tool outputs and button labels spawn
    # false-positive PERSON spans. Crucially, each field is processed
    # SEPARATELY: running the NER model on a concatenation lets it mint spans
    # that straddle field boundaries (e.g. "Mary\nJohn Doe"). The resulting map
    # is still applied to every field, so trajectory names are swapped too.
    ner_sources = [
        str(v_obj.get("story") or ""),
        str(t_obj.get("user_instruction") or ""),
    ]
    ner_sources.extend(sensitive)

    person_strings: list[str] = []
    org_texts: list[str] = []
    loc_texts: list[str] = []
    seen_loc: set[str] = set()
    if nlp is not None:
        for src in ner_sources:
            for e in detect_entities(src, nlp):
                if e.label == "PERSON":
                    person_strings.append(_with_recovered_honorific(e.text, e, src))
                elif e.label == "ORG":
                    org_texts.append(e.text)
                elif e.label in ("GPE", "LOC") and e.text not in seen_loc:
                    seen_loc.add(e.text)
                    loc_texts.append(e.text)

    # --- protected set ------------------------------------------------------
    protected: set[str] = set(_PROTECTED_JSON_KEYS)
    protected.update(toolkits)
    if final_action:
        protected.add(final_action)
    for org in org_texts:
        protected.add(org)
        protected.update(org.split())

    # --- person identities (NER + structured name slots, added directly so
    #     they cannot cross-contaminate via NER) ------------------------------
    for col, key in _NAME_SLOTS:
        val = objs[col].get(key)
        if val and looks_like_name(str(val)):
            person_strings.append(str(val))
    identities = build_person_identities(person_strings)

    taken_first: set[str] = set()
    taken_last: set[str] = set()
    person_pairs: list[tuple[PersonIdentity, Any]] = []
    n_persons = 0
    for identity in identities:
        repl = choose_replacement_identity(
            identity, bank, record_id, culture, taken_first, taken_last
        )
        person_pairs.append((identity, repl))
        if (repl.first and repl.first != identity.canonical_first) or (
            repl.last and repl.last != identity.canonical_last
        ):
            n_persons += 1

    # --- locations (GPE/LOC detected above; skip protected org/tool names) ---
    location_pairs: list[tuple[str, str]] = []
    taken_loc: set[str] = set()
    n_locations = 0
    if not bank.is_identity():
        import random

        for loc in loc_texts:
            if loc in protected:
                continue
            rng = random.Random(f"{record_id}|{culture}|loc|{loc.lower()}")
            new_loc = bank.pick_location(rng, taken_loc)
            if new_loc and new_loc != loc:
                location_pairs.append((loc, new_loc))
                n_locations += 1

    # --- build + apply map --------------------------------------------------
    ordered = build_replacement_map(person_pairs, location_pairs, protected)
    pattern, lookup = compile_map(ordered)
    if pattern is None:
        return (
            _dump_obj(s_obj, s_kind, row.get("S")),
            _dump_obj(v_obj, v_kind, row.get("V")),
            t_obj,
            0,
            0,
            {},
        )

    for obj in (s_obj, v_obj):
        for key, val in list(obj.items()):
            if isinstance(val, str):
                obj[key], _ = apply_map(val, pattern, lookup)
    for key in _T_TEXT_FIELDS:
        if isinstance(t_obj.get(key), str):
            t_obj[key], _ = apply_map(t_obj[key], pattern, lookup)
    if sensitive:
        t_obj["sensitive_info_items"] = [
            apply_map(item, pattern, lookup)[0] for item in sensitive
        ]

    return (
        _dump_obj(s_obj, s_kind, row.get("S")),
        _dump_obj(v_obj, v_kind, row.get("V")),
        t_obj,
        n_persons,
        n_locations,
        dict(ordered),
    )


def perturb_dataset(
    df: pd.DataFrame, culture: str, *, seed_namespace: str = "privacylens"
) -> pd.DataFrame:
    """Return a copy of ``df`` with names/locations swapped for ``culture``.

    ``culture='western'`` is an identity passthrough: the frame is returned with
    only the audit columns added, byte-identical in S/V/T (no NER model needed).
    """
    out = df.copy()
    bank = get_bank(culture)

    if bank.is_identity():
        out["culture"] = culture
        out["n_persons_swapped"] = 0
        out["n_locations_swapped"] = 0
        out["perturb_map_json"] = "{}"
        return out

    # Lazy NER-model load only when actually perturbing.
    from .ner import get_nlp

    nlp = get_nlp()

    s_col: list[Any] = []
    v_col: list[Any] = []
    t_col: list[Any] = []
    n_persons_col: list[int] = []
    n_locs_col: list[int] = []
    map_col: list[str] = []

    for _, row in out.iterrows():
        s_new, v_new, t_new, n_p, n_l, rmap = perturb_row(
            row.to_dict(), culture, bank, nlp
        )
        s_col.append(s_new)
        v_col.append(v_new)
        t_col.append(t_new)
        n_persons_col.append(n_p)
        n_locs_col.append(n_l)
        map_col.append(json.dumps(rmap, ensure_ascii=False))

    out["S"] = s_col
    out["V"] = v_col
    out["T"] = t_col
    out["culture"] = culture
    out["n_persons_swapped"] = n_persons_col
    out["n_locations_swapped"] = n_locs_col
    out["perturb_map_json"] = map_col
    return out
