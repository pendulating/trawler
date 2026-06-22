"""Per-stage data-quality scalars (2026-06-09 logging review, item D)."""

import pandas as pd
import pytest

from dagspaces.historical_norms.stage_metrics import compute_stage_quality_metrics


class TestBaseCounts:
    def test_empty_df(self):
        m = compute_stage_quality_metrics("ci_reasoning", pd.DataFrame())
        assert m == {"data_quality/rows": 0}

    def test_unknown_stage_still_counts(self):
        df = pd.DataFrame({"gutenberg_id": ["11", "11", "1342"]})
        m = compute_stage_quality_metrics("mystery_stage", df)
        assert m["data_quality/rows"] == 3
        assert m["data_quality/books"] == 2
        assert m["data_quality/rows_per_book_min"] == 1
        assert m["data_quality/rows_per_book_max"] == 2


class TestFetchGutenberg:
    def test_chunk_length_stats(self):
        df = pd.DataFrame({
            "gutenberg_id": ["11"] * 3,
            "article_text": ["a" * 100, "b" * 200, "c" * 300],
        })
        m = compute_stage_quality_metrics("fetch_gutenberg", df)
        assert m["data_quality/chunk_len_mean"] == 200.0
        assert m["data_quality/chunk_len_max"] == 300.0
        assert m["data_quality/chunk_len_p50"] == 200.0


class TestCIReasoning:
    def test_parse_error_and_flow_stats(self):
        df = pd.DataFrame({
            "gutenberg_id": ["11"] * 4,
            "ci_reasoning_parse_error": [None, "boom", None, None],
            "has_information_exchange": [True, False, True, None],
            "ci_flow_count": [2, 0, 1, 0],
        })
        m = compute_stage_quality_metrics("ci_reasoning", df)
        assert m["data_quality/parse_error_rate"] == 0.25
        assert m["data_quality/has_exchange_rate"] == pytest.approx(2 / 3, abs=1e-4)
        assert m["data_quality/flows_per_chunk_mean"] == 0.75
        assert m["data_quality/flows_per_chunk_max"] == 2.0
        assert m["data_quality/zero_flow_frac"] == 0.5


class TestCIExtraction:
    def test_appropriateness_distribution(self):
        df = pd.DataFrame({
            "ci_appropriateness": ["appropriate", "Inappropriate",
                                   "ambiguous", "appropriate"],
            "extraction_error": [None, None, None, "parse fail"],
        })
        m = compute_stage_quality_metrics("ci_extraction", df)
        assert m["data_quality/ci_appropriateness_appropriate_frac"] == 0.5
        assert m["data_quality/ci_appropriateness_inappropriate_frac"] == 0.25
        assert m["data_quality/ci_appropriateness_ambiguous_frac"] == 0.25
        assert m["data_quality/extraction_error_rate"] == 0.25


class TestNormStages:
    def test_norm_reasoning(self):
        df = pd.DataFrame({
            "reasoning_error": [None, None],
            "has_prescriptive_content": [True, False],
            "norm_count": [3, 1],
        })
        m = compute_stage_quality_metrics("norm_reasoning", df)
        assert m["data_quality/parse_error_rate"] == 0.0
        assert m["data_quality/prescriptive_content_rate"] == 0.5
        assert m["data_quality/norms_per_chunk_mean"] == 2.0

    def test_norm_extraction_distributions(self):
        df = pd.DataFrame({
            "extraction_failed": [False, False, True, False],
            "norm_quality_passed": [True, True, None, False],
            "raz_governs_info_flow": [True, False, True, True],
            "raz_normative_force": ["obligatory", "prohibited",
                                    "obligatory", "recommended"],
            "raz_confidence_quant": [0.9, 0.7, None, 0.8],
        })
        m = compute_stage_quality_metrics("norm_extraction", df)
        assert m["data_quality/extraction_error_rate"] == 0.25
        assert m["data_quality/norm_quality_passed_rate"] == pytest.approx(2 / 3, abs=1e-4)
        assert m["data_quality/governs_info_flow_rate"] == 0.75
        assert m["data_quality/raz_normative_force_obligatory_frac"] == 0.5
        assert m["data_quality/raz_normative_force_permitted_frac"] == 0.0
        assert m["data_quality/confidence_mean"] == pytest.approx(0.8)

    def test_role_abstraction_failure_rate(self):
        df = pd.DataFrame({"role_abstraction_failed": [False, True, False, False]})
        m = compute_stage_quality_metrics("norm_role_abstraction", df)
        assert m["data_quality/role_abstraction_error_rate"] == 0.25

    def test_missing_columns_skipped_not_raised(self):
        df = pd.DataFrame({"unrelated": [1, 2]})
        m = compute_stage_quality_metrics("norm_extraction", df)
        assert m["data_quality/rows"] == 2
        assert not any(k.endswith("_rate") for k in m)
