"""Load and join VLM-GeoPrivacyBench annotations, metadata, and available images."""

from __future__ import annotations

import logging
import os
import re
from pathlib import Path
from typing import List

import pandas as pd

logger = logging.getLogger(__name__)


def load_dataset(
    annotations_path: str,
    metadata_path: str,
    image_dir: str,
    exclude_sources: List[str] | None = None,
    sample_n: int | None = None,
) -> pd.DataFrame:
    """Load VLM-GeoPrivacyBench dataset.

    1. Read annotation_labels.csv (id, Q1..Q7)
    2. Read images_metadata.csv (image_id, image_source, true_coordinate, sharing_intent)
    3. Extract numeric suffix from image_id to match annotation id
    4. Scan image_dir for available .jpg files
    5. Filter to rows with available images
    6. Optionally exclude sources (e.g. Flickr-yfcc26k)

    Returns:
        DataFrame with columns: image_id, numeric_id, image_path, image_source,
        true_coordinate, sharing_intent, Q1_true..Q7_true
    """
    # Load annotations
    df_annot = pd.read_csv(annotations_path)
    df_annot["id"] = df_annot["id"].astype(str)
    # Rename Q columns to Q*_true
    rename_map = {f"Q{i}": f"Q{i}_true" for i in range(1, 8)}
    df_annot = df_annot.rename(columns=rename_map)

    # Load metadata
    df_meta = pd.read_csv(metadata_path)
    # Extract numeric id from image_id (e.g. "yfcc-1008954785" -> "1008954785")
    df_meta["numeric_id"] = df_meta["image_id"].apply(
        lambda x: re.sub(r"^[a-zA-Z]+-", "", str(x))
    )

    # Exclude sources
    if exclude_sources:
        before = len(df_meta)
        df_meta = df_meta[~df_meta["image_source"].isin(exclude_sources)]
        excluded = before - len(df_meta)
        if excluded > 0:
            logger.info(f"Excluded {excluded} rows from sources: {exclude_sources}")

    # Join on numeric id
    df = pd.merge(
        df_meta,
        df_annot,
        left_on="numeric_id",
        right_on="id",
        how="inner",
    )
    logger.info(f"Joined annotations+metadata: {len(df)} rows")

    # Scan available images
    image_dir_path = Path(image_dir)
    available_images = {}
    if image_dir_path.exists():
        for img_file in image_dir_path.glob("*.jpg"):
            # Extract numeric id from filename (e.g. "gptgeochat-109970657.jpg" -> "109970657")
            stem = img_file.stem
            numeric = re.sub(r"^[a-zA-Z]+-", "", stem)
            available_images[numeric] = str(img_file)

    logger.info(f"Found {len(available_images)} images in {image_dir}")

    # Filter to rows with available images
    df["image_path"] = df["numeric_id"].map(available_images)
    df = df.dropna(subset=["image_path"]).reset_index(drop=True)
    logger.info(f"Rows with available images: {len(df)}")

    # Sample if requested
    if sample_n and sample_n < len(df):
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        logger.info(f"Sampled {sample_n} rows")

    # Select and order columns
    keep_cols = [
        "image_id", "numeric_id", "image_path", "image_source",
        "true_coordinate", "sharing_intent",
    ] + [f"Q{i}_true" for i in range(1, 8)]
    keep_cols = [c for c in keep_cols if c in df.columns]

    return df[keep_cols]
