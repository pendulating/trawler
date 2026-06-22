"""Culture name/location banks, gender inference, and seeded replacement choice.

Loads the bundled JSON resources (``resources/name_banks.json``,
``resources/first_name_gender.json``) and exposes a :class:`CultureBank` that
picks gender-preserving, collision-free replacement names/locations. The
``western`` culture resolves to an identity bank (no-op).
"""

from __future__ import annotations

import functools
import json
import os
import random
from dataclasses import dataclass
from typing import Optional

_RES_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "resources")
_NAME_BANKS_PATH = os.path.join(_RES_DIR, "name_banks.json")
_GENDER_PATH = os.path.join(_RES_DIR, "first_name_gender.json")


@functools.lru_cache(maxsize=1)
def _load_name_banks() -> dict:
    with open(_NAME_BANKS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


@functools.lru_cache(maxsize=1)
def _load_gender_table() -> dict:
    with open(_GENDER_PATH, "r", encoding="utf-8") as f:
        return dict(json.load(f).get("names", {}))


def available_cultures() -> list[str]:
    """Return all configured culture keys (including ``western``)."""
    return sorted(_load_name_banks().get("cultures", {}).keys())


def infer_gender(first_name: str) -> str:
    """Infer gender ('m' | 'f' | 'u') of a first name from the vendored table.

    Strips a trailing possessive. Unknown names return ``'u'`` so the caller
    draws from the combined male+female pool.
    """
    if not first_name:
        return "u"
    key = str(first_name).strip().lower()
    for poss in ("'s", "’s"):
        if key.endswith(poss):
            key = key[: -len(poss)]
            break
    return _load_gender_table().get(key, "u")


@dataclass(frozen=True)
class ReplacementIdentity:
    """A chosen replacement person identity (either component may be None)."""

    first: Optional[str]
    last: Optional[str]


class CultureBank:
    """First-name/surname/location pools for one culture.

    An ``is_identity()`` bank (``western`` → ``null`` in the JSON) echoes the
    original tokens, so the control variant traverses the same code path with a
    no-op effect.
    """

    def __init__(self, culture: str, data: Optional[dict]):
        self.culture = culture
        self._data = data
        if data is not None:
            self._first = data.get("first_names", {}) or {}
            self._surnames = list(data.get("surnames", []) or [])
            self._locations = list(data.get("locations", []) or [])
        else:
            self._first = {}
            self._surnames = []
            self._locations = []

    def is_identity(self) -> bool:
        return self._data is None

    @staticmethod
    def _pick(pool: list[str], rng: random.Random, exclude: set[str]) -> Optional[str]:
        if not pool:
            return None
        candidates = [x for x in pool if x not in exclude]
        if not candidates:
            candidates = list(pool)
        choice = rng.choice(candidates)
        exclude.add(choice)
        return choice

    def pick_first(self, gender: str, rng: random.Random, exclude: set[str]) -> Optional[str]:
        if gender in ("m", "f") and self._first.get(gender):
            pool = list(self._first[gender])
        else:
            pool = list(self._first.get("m", [])) + list(self._first.get("f", []))
            pool += list(self._first.get("u", []))
        return self._pick(pool, rng, exclude)

    def pick_surname(self, rng: random.Random, exclude: set[str]) -> Optional[str]:
        return self._pick(self._surnames, rng, exclude)

    def pick_location(self, rng: random.Random, exclude: set[str]) -> Optional[str]:
        return self._pick(self._locations, rng, exclude)


def get_bank(culture: str) -> CultureBank:
    """Return the :class:`CultureBank` for ``culture``.

    Raises ``KeyError`` (with the valid options) for an unknown culture so a
    typo fails fast instead of silently producing the control.
    """
    cultures = _load_name_banks().get("cultures", {})
    if culture not in cultures:
        raise KeyError(
            f"Unknown culture {culture!r}. Available: {sorted(cultures.keys())}"
        )
    return CultureBank(culture, cultures[culture])
