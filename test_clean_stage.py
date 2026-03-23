#!/usr/bin/env python
"""Direct test of the norm_reasoning stage with existing data."""

import sys
sys.path.insert(0, "/share/pierson/matt/UAIR")

import pandas as pd
from omegaconf import OmegaConf

# Load existing test data
df = pd.read_parquet("/share/pierson/matt/UAIR/outputs/2026-01-31/15-38-01/norm_extraction/outputs/fetch/chunks.parquet")
print(f"Loaded {len(df)} rows")
print(f"Columns: {list(df.columns)}")

# Take just 2 rows for quick test
df = df.head(2)
print(f"Testing with {len(df)} rows")

# Create minimal config
cfg = OmegaConf.create({
    "model": {
        "model_source": "/share/pierson/matt/zoo/models/Qwen2.5-72B-Instruct-AWQ",
        "engine_kwargs": {
            "tensor_parallel_size": 2,
            "quantization": "awq",
        },
        "batch_size": 2,
    },
    "prompt_reasoning": {
        "system_prompt": "You are an expert in social and historical norms.",
        "prompt_template": "Text:\n{{article_text}}\n\nIdentify norms in this text:",
    }
})

# Import and run the standard stage
from dagspaces.historical_norms.stages.norm_reasoning import run_norm_reasoning_stage

print("Running norm_reasoning stage...")
result = run_norm_reasoning_stage(df, cfg)
print(f"Result shape: {result.shape if hasattr(result, 'shape') else 'N/A'}")
print(f"Result columns: {list(result.columns) if hasattr(result, 'columns') else 'N/A'}")
print("Done!")
