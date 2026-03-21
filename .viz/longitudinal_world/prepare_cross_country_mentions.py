#!/usr/bin/env python3
"""
Build cross-country mention datasets for the longitudinal world visualization.

This script scans the global article corpus, counts how often articles from one
country mention other countries (by name, abbreviation, or alias), aggregates
per-country totals, and exports both parquet and JSON artifacts used by the
Deck.gl globe.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
import argparse
import json
import re
import sys
import unicodedata
from typing import Iterable

import pandas as pd
import pyarrow.parquet as pq
import pycountry
from flashtext import KeywordProcessor
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = PROJECT_ROOT / "outputs/for_nov10workshop_global_results/classify/classify_all.parquet"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/derived"
DEFAULT_JSON_PATH = Path(__file__).resolve().parent / "public" / "cross_country_mentions_by_year.json"


MANUAL_SYNONYMS: dict[str, list[str]] = {
    "CZ": ["czech republic"],
    "HK": ["hong kong", "hongkong"],
    "IR": ["iran", "islamic republic of iran"],
    "KR": ["south korea", "republic of korea"],
    "NL": ["holland", "the netherlands"],
    "SA": ["saudi arabia", "kingdom of saudi arabia", "ksa"],
    "SV": ["el salvador"],
    "TR": ["turkey", "republic of turkey", "republic of turkiye"],
    "TT": ["trinidad and tobago", "trinidad & tobago"],
    "TW": ["taiwan", "republic of china", "taiwan roc"],
    "TZ": ["tanzania", "united republic of tanzania"],
    "VE": ["venezuela", "bolivarian republic of venezuela"],
    "VN": ["vietnam", "viet nam"],
}


@dataclass(frozen=True)
class CountryAlias:
    country_code: str
    raw_alias: str
    normalized_alias: str
    kind: str


def normalize_text(value: str | None) -> str:
    """Lower-case text, remove accents, and collapse whitespace."""

    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKD", value)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = text.lower()
    text = text.replace("-", " ")
    text = re.sub(r"[^a-z0-9\.\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def parse_year(value) -> int | None:
    """Coerce the provided value into a reasonable calendar year."""

    if value is None:
        return None
    try:
        year_int = int(str(value).strip())
    except (TypeError, ValueError):
        return None
    return year_int if 1900 <= year_int <= 2100 else None


def build_alias_records(country_codes: Iterable[str]) -> list[CountryAlias]:
    aliases: list[CountryAlias] = []
    for code in country_codes:
        country = pycountry.countries.get(alpha_2=code)
        if country is None:
            continue
        candidates: list[tuple[str, str]] = [
            ("name", country.name),
            ("official", getattr(country, "official_name", "")),
            ("common", getattr(country, "common_name", "")),
            ("alpha3", country.alpha_3),
        ]
        candidates.extend(("manual", term) for term in MANUAL_SYNONYMS.get(code, []))
        for kind, term in candidates:
            if not term:
                continue
            normalized = normalize_text(term)
            if not normalized:
                continue
            aliases.append(CountryAlias(code, term, normalized, kind))
    seen = set()
    unique_aliases: list[CountryAlias] = []
    for alias in aliases:
        key = (alias.country_code, alias.normalized_alias)
        if key in seen:
            continue
        seen.add(key)
        unique_aliases.append(alias)
    return unique_aliases


def build_keyword_processor(alias_table: pd.DataFrame) -> KeywordProcessor:
    processor = KeywordProcessor(case_sensitive=False)
    for row in alias_table.itertuples():
        processor.add_keyword(row.normalized_alias, row.country_code)
    return processor


def count_country_mentions(normalized_text: str, keyword_processor: KeywordProcessor) -> Counter[str]:
    if not normalized_text:
        return Counter()
    matches = keyword_processor.extract_keywords(normalized_text)
    return Counter(matches)


def aggregate_mentions(
    data_path: Path,
    keyword_processor: KeywordProcessor,
    country_names: dict[str, str],
    debug_fraction: float | None = None,
    random_seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    pf = pq.ParquetFile(data_path)
    country_series = (
        pq.read_table(data_path, columns=["country"]).to_pandas()["country"].dropna().str.upper()
    )
    country_codes = sorted(country_series.unique().tolist())
    country_set = set(country_codes)

    occurrence_counts: dict[str, Counter[str]] = defaultdict(Counter)
    article_hit_counts: dict[str, Counter[str]] = defaultdict(Counter)
    occurrence_counts_by_year: dict[str, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    article_hit_counts_by_year: dict[str, dict[int, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    articles_seen: Counter[str] = Counter()
    articles_with_foreign_mentions: Counter[str] = Counter()
    articles_seen_by_year: dict[str, Counter[int]] = defaultdict(Counter)
    articles_with_foreign_mentions_by_year: dict[str, Counter[int]] = defaultdict(Counter)

    if debug_fraction is not None and not (0 < debug_fraction <= 1):
        raise ValueError("debug_fraction must be in (0, 1].")

    progress = tqdm(total=pf.metadata.num_rows, desc="Scanning articles", unit="articles")
    for batch in pf.iter_batches(columns=["country", "article_text", "year"], batch_size=1000):
        df_batch = batch.to_pandas().dropna(subset=["country", "article_text"])
        if df_batch.empty:
            continue
        df_batch["country"] = df_batch["country"].str.upper()
        df_batch = df_batch[df_batch["country"].isin(country_set)]
        if df_batch.empty:
            continue
        raw_batch_len = len(df_batch)
        if debug_fraction is not None:
            df_batch = df_batch.sample(frac=debug_fraction, random_state=random_seed)
            if df_batch.empty:
                progress.update(raw_batch_len)
                continue
        progress.update(raw_batch_len)
        for row in df_batch.itertuples(index=False):
            source = row.country
            articles_seen[source] += 1
            year_value = parse_year(getattr(row, "year", None))
            if year_value is not None:
                articles_seen_by_year[source][year_value] += 1

            normalized = normalize_text(row.article_text)
            mention_counts = count_country_mentions(normalized, keyword_processor)
            mention_counts.pop(source, None)
            if not mention_counts:
                continue
            articles_with_foreign_mentions[source] += 1
            if year_value is not None:
                articles_with_foreign_mentions_by_year[source][year_value] += 1
            for target, count in mention_counts.items():
                occurrence_counts[source][target] += int(count)
                article_hit_counts[source][target] += 1
                if year_value is not None:
                    occurrence_counts_by_year[source][year_value][target] += int(count)
                    article_hit_counts_by_year[source][year_value][target] += 1
    progress.close()

    def build_mentions_df() -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for source, target_counts in occurrence_counts.items():
            for target, occurrences in target_counts.items():
                records.append(
                    {
                        "source_country": source,
                        "mentioned_country": target,
                        "mention_occurrences": int(occurrences),
                        "article_hits": int(article_hit_counts[source][target]),
                    }
                )
        df = pd.DataFrame(records)
        if df.empty:
            raise RuntimeError("No cross-country mentions detected.")
        df["source_total_mentions"] = df.groupby("source_country")["mention_occurrences"].transform("sum")
        df["mention_share"] = df["mention_occurrences"] / df["source_total_mentions"].replace({0: pd.NA})
        df["source_articles_seen"] = df["source_country"].map(articles_seen)
        df["source_articles_with_foreign_mentions"] = df["source_country"].map(articles_with_foreign_mentions)
        df["article_hit_share"] = df["article_hits"] / df["source_articles_seen"].replace({0: pd.NA})
        df["source_country_name"] = df["source_country"].map(country_names)
        df["mentioned_country_name"] = df["mentioned_country"].map(country_names)
        return df.sort_values(["source_country", "mention_occurrences"], ascending=[True, False]).reset_index(drop=True)

    def build_mentions_by_year_df() -> pd.DataFrame:
        records: list[dict[str, object]] = []
        for source, year_counts in occurrence_counts_by_year.items():
            for year_value, target_counts in year_counts.items():
                for target, occurrences in target_counts.items():
                    records.append(
                        {
                            "year": year_value,
                            "source_country": source,
                            "mentioned_country": target,
                            "mention_occurrences": int(occurrences),
                            "article_hits": int(article_hit_counts_by_year[source][year_value][target]),
                        }
                    )
        df = pd.DataFrame(records)
        if df.empty:
            raise RuntimeError("No year-level cross-country mentions detected.")
        df["source_year_total_mentions"] = (
            df.groupby(["year", "source_country"])["mention_occurrences"].transform("sum")
        )
        df["mention_share"] = df["mention_occurrences"] / df["source_year_total_mentions"].replace({0: pd.NA})
        df["source_year_articles_seen"] = [
            articles_seen_by_year[source].get(year, 0)
            for source, year in zip(df["source_country"], df["year"])
        ]
        df["source_year_articles_with_foreign_mentions"] = [
            articles_with_foreign_mentions_by_year[source].get(year, 0)
            for source, year in zip(df["source_country"], df["year"])
        ]
        df["article_hit_share"] = df["article_hits"] / df["source_year_articles_seen"].replace({0: pd.NA})
        df["source_country_name"] = df["source_country"].map(country_names)
        df["mentioned_country_name"] = df["mentioned_country"].map(country_names)
        return df.sort_values(["year", "source_country", "mention_occurrences"], ascending=[True, True, False]).reset_index(
            drop=True
        )

    return build_mentions_df(), build_mentions_by_year_df()


def export_arc_json(mentions_by_year_df: pd.DataFrame, json_path: Path) -> None:
    arc_columns = [
        "year",
        "source_country",
        "source_country_name",
        "mentioned_country",
        "mentioned_country_name",
        "mention_occurrences",
        "mention_share",
        "article_hits",
        "article_hit_share",
        "source_year_total_mentions",
        "source_year_articles_seen",
    ]
    json_path.parent.mkdir(parents=True, exist_ok=True)
    mentions_by_year_df[arc_columns].to_json(json_path, orient="records")


def prepare_data(
    input_path: Path,
    output_dir: Path,
    json_path: Path,
    debug_fraction: float | None = None,
    random_seed: int = 42,
) -> None:
    print(f"Loading dataset metadata from {input_path}...")
    country_series = pq.read_table(input_path, columns=["country"]).to_pandas()["country"].dropna().str.upper()
    country_codes = sorted(country_series.unique().tolist())
    country_names = {code: pycountry.countries.get(alpha_2=code).name for code in country_codes}

    alias_df = pd.DataFrame(
        [
            {"country_code": alias.country_code, "raw_alias": alias.raw_alias, "normalized_alias": alias.normalized_alias}
            for alias in build_alias_records(country_codes)
        ]
    )
    print(f"Prepared {len(alias_df)} aliases across {len(country_codes)} source countries.")

    keyword_processor = build_keyword_processor(alias_df)

    mentions_df, mentions_by_year_df = aggregate_mentions(
        data_path=input_path,
        keyword_processor=keyword_processor,
        country_names=country_names,
        debug_fraction=debug_fraction,
        random_seed=random_seed,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_parquet = output_dir / "cross_country_mentions.parquet"
    output_by_year_parquet = output_dir / "cross_country_mentions_by_year.parquet"

    mentions_df.to_parquet(output_parquet, index=False)
    mentions_by_year_df.to_parquet(output_by_year_parquet, index=False)
    export_arc_json(mentions_by_year_df, json_path)

    print(f"✓ Saved aggregate mentions to {output_parquet}")
    print(f"✓ Saved year-level mentions to {output_by_year_parquet}")
    print(f"✓ Exported arc JSON to {json_path} ({json_path.stat().st_size / 1024:.1f} KB)")
    print("Top source-target pairs:")
    print(
        mentions_df.sort_values("mention_occurrences", ascending=False)
        .head(10)[["source_country", "mentioned_country", "mention_occurrences", "mention_share"]]
        .to_string(index=False)
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare cross-country mention datasets for the UAIR globe.")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help=f"Path to the global article parquet file (default: {DEFAULT_INPUT})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for parquet outputs (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--json-path",
        type=Path,
        default=DEFAULT_JSON_PATH,
        help=f"Path for the arc JSON consumed by the visualization (default: {DEFAULT_JSON_PATH})",
    )
    parser.add_argument(
        "--debug-fraction",
        type=float,
        default=None,
        help="Optional sampling fraction (0-1] to speed up local debugging.",
    )
    parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed used when debug sampling is enabled.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        prepare_data(
            input_path=args.input,
            output_dir=args.output_dir,
            json_path=args.json_path,
            debug_fraction=args.debug_fraction,
            random_seed=args.random_seed,
        )
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()




