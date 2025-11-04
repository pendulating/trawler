from __future__ import annotations

import os
from typing import Any

import pandas as pd
from omegaconf import OmegaConf


def ensure_profile(cfg, profile: str) -> None:
    try:
        OmegaConf.update(cfg, "runtime.classification_profile", str(profile), merge=True)
        # Prefer LLM for EU/RB; relevance can be heuristic or LLM via cfg
        if profile in ("eu_ai_act", "risks_and_benefits"):
            OmegaConf.update(cfg, "runtime.use_llm_classify", True, merge=True)
    except Exception:
        pass


def inject_prompt_from_file(cfg, prompt_filename: str) -> None:
    try:
        base_dir = os.path.dirname(os.path.dirname(__file__))  # dagspaces/uair
        prompt_path = os.path.join(base_dir, "conf", "prompt", prompt_filename)
        if os.path.exists(prompt_path):
            prompt_cfg = OmegaConf.load(prompt_path)
            sys_p = prompt_cfg.get("system_prompt")
            usr_p = prompt_cfg.get("prompt_template")
            if sys_p:
                OmegaConf.update(cfg, "prompt.system_prompt", sys_p, merge=True)
            if usr_p:
                OmegaConf.update(cfg, "prompt.prompt_template", usr_p, merge=True)
    except Exception:
        pass


def dedupe_by_article_id(df: pd.DataFrame) -> pd.DataFrame:
    try:
        if "article_id" in df.columns and len(df):
            return df.sort_values(by=["article_id"]).drop_duplicates(subset=["article_id"], keep="first")
    except Exception:
        pass
    return df


