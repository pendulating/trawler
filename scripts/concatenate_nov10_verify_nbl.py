#!/usr/bin/env python3
"""Concatenate global and US nov10_workshop verify_nbl outputs into a joint dataset."""

import os
import pandas as pd
from pathlib import Path

# Define paths
GLOBAL_PATH = "/share/pierson/matt/UAIR/outputs/for_nov10workshop_global_results/verify_nbl/verify_nbl_results.parquet"
US_PATH = "/share/pierson/matt/UAIR/outputs/for_nov10workshop_us_results/verify_nbl/verify_nbl_results.parquet"
OUTPUT_DIR = "/share/pierson/matt/UAIR/outputs/for_nov10workshop_joint_results/verify_nbl"
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "verify_nbl_results.parquet")

def main():
    print(f"Loading global dataset from: {GLOBAL_PATH}")
    df_global = pd.read_parquet(GLOBAL_PATH)
    print(f"  Global dataset: {len(df_global)} rows, {len(df_global.columns)} columns")
    
    print(f"\nLoading US dataset from: {US_PATH}")
    df_us = pd.read_parquet(US_PATH)
    print(f"  US dataset: {len(df_us)} rows, {len(df_us.columns)} columns")
    
    # Check for column mismatches
    global_cols = set(df_global.columns)
    us_cols = set(df_us.columns)
    if global_cols != us_cols:
        print(f"\nWarning: Column mismatch detected!")
        print(f"  Global-only columns: {global_cols - us_cols}")
        print(f"  US-only columns: {us_cols - global_cols}")
        print(f"  Common columns: {len(global_cols & us_cols)}")
        # Use common columns
        common_cols = sorted(list(global_cols & us_cols))
        df_global = df_global[common_cols]
        df_us = df_us[common_cols]
        print(f"  Using {len(common_cols)} common columns")
    
    print(f"\nConcatenating datasets...")
    df_combined = pd.concat([df_global, df_us], ignore_index=True)
    print(f"  Combined dataset: {len(df_combined)} rows, {len(df_combined.columns)} columns")
    
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\nSaving combined dataset to: {OUTPUT_PATH}")
    df_combined.to_parquet(OUTPUT_PATH, index=False)
    print(f"  Successfully saved {len(df_combined)} rows to {OUTPUT_PATH}")
    
    # Print summary statistics
    print(f"\nSummary:")
    print(f"  Total rows: {len(df_combined)}")
    if 'article_id' in df_combined.columns:
        unique_articles = df_combined['article_id'].nunique()
        print(f"  Unique articles: {unique_articles}")

if __name__ == "__main__":
    main()

