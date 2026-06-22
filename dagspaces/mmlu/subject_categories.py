"""MMLU subject → 4-category mapping (STEM / humanities / social_sciences / other).

Mirrors the canonical grouping in the original Hendrycks et al. (2021) MMLU
paper and OpenAI's simple-evals adaptation. Used by compute_metrics to
break down accuracy into the four standard category buckets that
appear in every published MMLU leaderboard.

57 subjects total: 19 STEM + 13 humanities + 12 social_sciences + 13 other.
"""

from __future__ import annotations

from typing import Dict

#: subject string (as it appears on cais/mmlu) → canonical category name.
SUBJECT_CATEGORY: Dict[str, str] = {
    # ── STEM (19) ─────────────────────────────────────────────────────
    "abstract_algebra": "STEM",
    "anatomy": "STEM",
    "astronomy": "STEM",
    "college_biology": "STEM",
    "college_chemistry": "STEM",
    "college_computer_science": "STEM",
    "college_mathematics": "STEM",
    "college_physics": "STEM",
    "computer_security": "STEM",
    "conceptual_physics": "STEM",
    "electrical_engineering": "STEM",
    "elementary_mathematics": "STEM",
    "high_school_biology": "STEM",
    "high_school_chemistry": "STEM",
    "high_school_computer_science": "STEM",
    "high_school_mathematics": "STEM",
    "high_school_physics": "STEM",
    "high_school_statistics": "STEM",
    "machine_learning": "STEM",
    # ── Humanities (13) ───────────────────────────────────────────────
    "formal_logic": "humanities",
    "high_school_european_history": "humanities",
    "high_school_us_history": "humanities",
    "high_school_world_history": "humanities",
    "international_law": "humanities",
    "jurisprudence": "humanities",
    "logical_fallacies": "humanities",
    "moral_disputes": "humanities",
    "moral_scenarios": "humanities",
    "philosophy": "humanities",
    "prehistory": "humanities",
    "professional_law": "humanities",
    "world_religions": "humanities",
    # ── Social sciences (12) ──────────────────────────────────────────
    "econometrics": "social_sciences",
    "high_school_geography": "social_sciences",
    "high_school_government_and_politics": "social_sciences",
    "high_school_macroeconomics": "social_sciences",
    "high_school_microeconomics": "social_sciences",
    "high_school_psychology": "social_sciences",
    "human_sexuality": "social_sciences",
    "professional_psychology": "social_sciences",
    "public_relations": "social_sciences",
    "security_studies": "social_sciences",
    "sociology": "social_sciences",
    "us_foreign_policy": "social_sciences",
    # ── Other (13) ────────────────────────────────────────────────────
    "business_ethics": "other",
    "clinical_knowledge": "other",
    "college_medicine": "other",
    "global_facts": "other",
    "human_aging": "other",
    "management": "other",
    "marketing": "other",
    "medical_genetics": "other",
    "miscellaneous": "other",
    "nutrition": "other",
    "professional_accounting": "other",
    "professional_medicine": "other",
    "virology": "other",
}


CATEGORIES = ("STEM", "humanities", "social_sciences", "other")


def category_for(subject: str) -> str:
    """Return the category for ``subject`` or ``"other"`` if unknown.

    Unknown subjects bucket into ``other`` rather than raising so a HF
    schema drift (new subject added) doesn't break the run — the new
    subject still shows up in per_subject metrics; only the category
    rollup is affected.
    """
    return SUBJECT_CATEGORY.get(str(subject), "other")
