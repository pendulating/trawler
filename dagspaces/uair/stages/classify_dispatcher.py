"""
Classification stage dispatcher.

This module routes classification requests to the appropriate profile-specific implementation
based on the configuration's classification_profile setting.

Usage:
    from dagspaces.uair.stages.classify_dispatcher import run_classification_stage
    
    result = run_classification_stage(df, cfg)
"""

import warnings
from typing import Any

import pandas as pd

# Deprecation warning flag
_DEPRECATION_WARNING_SHOWN = False


def run_classification_stage(df: pd.DataFrame, cfg) -> Any:
    """
    Classification stage dispatcher - delegates to profile-specific implementations.
    
    This function routes classification requests to the appropriate profile-specific
    implementation based on cfg.runtime.classification_profile:
    - "relevance" -> classify_relevance.run_classification_relevance()
    - "eu_ai_act" -> classify_eu_act.run_classification_eu_act()
    - "risks_and_benefits" -> classify_risks_benefits.run_classification_risks_benefits()
    
    Args:
        df: Input DataFrame or Ray Dataset
        cfg: Configuration object with runtime.classification_profile setting
        
    Returns:
        Classification results DataFrame
        
    Raises:
        ValueError: If classification_profile is invalid or profile-specific module fails
    """
    global _DEPRECATION_WARNING_SHOWN
    
    # Determine classification profile
    try:
        classification_profile = str(
            getattr(cfg.runtime, "classification_profile", "relevance") or "relevance"
        ).strip().lower()
    except Exception:
        classification_profile = "relevance"
    
    # Dispatch to profile-specific implementation
    # Each profile module is a complete standalone implementation following decompose_nbl.py pattern
    # Lazy imports to avoid circular dependencies
    if classification_profile == "eu_ai_act":
        from .classify_eu_act import run_classification_eu_act
        return run_classification_eu_act(df, cfg)
    elif classification_profile == "risks_and_benefits":
        from .classify_risks_benefits import run_classification_risks_benefits
        return run_classification_risks_benefits(df, cfg)
    elif classification_profile == "relevance":
        from .classify_relevance import run_classification_relevance
        return run_classification_relevance(df, cfg)
    else:
        # Unknown profile - default to relevance with warning
        if not _DEPRECATION_WARNING_SHOWN:
            warnings.warn(
                f"Unknown classification_profile '{classification_profile}'. "
                "Defaulting to 'relevance'.",
                UserWarning,
                stacklevel=2,
            )
            _DEPRECATION_WARNING_SHOWN = True
        from .classify_relevance import run_classification_relevance
        return run_classification_relevance(df, cfg)

