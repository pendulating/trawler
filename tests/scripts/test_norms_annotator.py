"""Tests for the norms_annotator export → annotate → merge round-trip.

The annotator is the tool used to build human-labeled ground truth for the
normative-simulacra norm-extraction stage. Two failure modes would silently
corrupt that ground truth:

1. **HTML injection from book text.** The manifest is embedded inline in a
   `<script>` block; raw `</script>`, `<!--`, or `-->` in any chunk's
   article_text or annotator-supplied string would terminate the script and
   produce a blank-page tool — the annotator wouldn't notice until they tried
   to annotate.
2. **Silent merge misses from dtype drift.** Parquet round-trips can return
   `chunk_id` as `int32`/`int64`/`object`; if the merge equality check fails,
   the row falls back to the manifest-embedded fields and book_summary /
   prescriptive context drop out without any error.

Coverage:

- ``_build_manifest`` is deterministic under seed, produces correct stats
  (has_norms counts, total model norms), and falls back gracefully when N
  exceeds available chunks.
- ``_build_html`` escapes the closing-script and HTML-comment sequences in
  the embedded JSON payload.
- ``cmd_merge`` correctly fans out each of the five decision types
  (confirm / modify / reject / uncertain / add) and produces the expected
  ``gold_raz_*`` columns.
- ``cmd_merge`` survives chunk_id dtype mismatch (Int64 vs Python int)
  without silently dropping the structured-extraction context columns.
- Hash mismatch surfaces a stderr warning rather than a silent join.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import pytest

import scripts.norms_annotator as na


# ── fixtures ──────────────────────────────────────────────────────────────

def _make_reasoning_df():
    """3 books × 3 chunks each = 9 unique chunks; 2 with has_norms=True."""
    rows = []
    for gb_id in ["100", "200", "300"]:
        for chunk_id in [0, 1, 2]:
            has_norms = (gb_id, chunk_id) in [("100", 0), ("200", 1)]
            rows.append({
                "gutenberg_id": gb_id,
                "chunk_id": chunk_id,
                "article_text": f"Text of {gb_id}:{chunk_id}",
                "chunk_size": len(f"Text of {gb_id}:{chunk_id}"),
                "book_title": f"Book {gb_id}",
                "book_author": f"Author {gb_id}",
                "has_norms": has_norms,
                "generated_text": "{\"norms\": []}" if not has_norms else "{\"norms\": [{}]}",
            })
    return pd.DataFrame(rows)


def _make_extraction_df():
    """Two chunks have norms: (100,0) has 2 norms, (200,1) has 1 norm.

    Includes orig_raz_* (pre-role-abstraction) and role_rationale /
    role_abstraction_failed so the abstraction-view fields are exercised.
    """
    return pd.DataFrame([
        {
            "gutenberg_id": "100", "chunk_id": 0, "norm_index": 0,
            "raz_norm_subject": "a friend with a confidence to share",  # abstracted
            "orig_raz_norm_subject": "Alice",                            # pre-abstraction
            "raz_norm_act": "tell",
            "orig_raz_norm_act": "tell",
            "raz_condition_of_application": "in private",
            "orig_raz_condition_of_application": "in private",
            "raz_normative_force": "obligation",
            "raz_norm_articulation": "A confidant should share private information only in private.",
            "orig_raz_norm_articulation": "Alice should tell Bob in private.",
            "raz_governs_info_flow": True,
            "raz_info_flow_note": "info to Bob only",
            "raz_prescriptive_element": "should", "raz_context": "family",
            "raz_norm_source": "narrator", "raz_confidence_qual": "high",
            "raz_confidence_quant": 0.9,
            "role_rationale": "Alice is in the role of a confidant.",
            "role_abstraction_failed": False,
            "article_text": "Text of 100:0", "chunk_size": 13,
            "book_title": "Book 100", "book_author": "Author 100",
            "book_summary": "Summary of book 100",
        },
        {
            "gutenberg_id": "100", "chunk_id": 0, "norm_index": 1,
            "raz_norm_subject": "Bob", "orig_raz_norm_subject": "Bob",  # no abstraction change
            "raz_norm_act": "keep secret", "orig_raz_norm_act": "keep secret",
            "raz_condition_of_application": "always",
            "orig_raz_condition_of_application": "always",
            "raz_normative_force": "obligation",
            "raz_norm_articulation": "Bob should keep the secret.",
            "orig_raz_norm_articulation": "Bob should keep the secret.",
            "raz_governs_info_flow": True,
            "raz_info_flow_note": "no further sharing",
            "raz_prescriptive_element": "should", "raz_context": "family",
            "raz_norm_source": "narrator", "raz_confidence_qual": "high",
            "raz_confidence_quant": 0.85,
            "role_rationale": "Bob is the named individual.",
            "role_abstraction_failed": False,
            "article_text": "Text of 100:0", "chunk_size": 13,
            "book_title": "Book 100", "book_author": "Author 100",
            "book_summary": "Summary of book 100",
        },
        {
            "gutenberg_id": "200", "chunk_id": 1, "norm_index": 0,
            "raz_norm_subject": "a member of a social class greeting peers",
            "orig_raz_norm_subject": "Carol",
            "raz_norm_act": "greet",
            "orig_raz_norm_act": "greet",
            "raz_condition_of_application": "in public",
            "orig_raz_condition_of_application": "in public",
            "raz_normative_force": "convention",
            "raz_norm_articulation": "Carol should greet acquaintances.",
            "orig_raz_norm_articulation": "Carol should greet acquaintances.",
            "raz_governs_info_flow": False,
            "raz_info_flow_note": None,
            "raz_prescriptive_element": "should", "raz_context": "social",
            "raz_norm_source": "narrator", "raz_confidence_qual": "medium",
            "raz_confidence_quant": 0.7,
            "role_rationale": "Carol is in the role of a social peer.",
            "role_abstraction_failed": False,
            "article_text": "Text of 200:1", "chunk_size": 13,
            "book_title": "Book 200", "book_author": "Author 200",
            "book_summary": "Summary of book 200",
        },
    ])


@pytest.fixture
def parquet_pair(tmp_path):
    rdir = tmp_path / "reasoning"
    rdir.mkdir()
    rpath = rdir / "reasoning.parquet"
    _make_reasoning_df().to_parquet(rpath)

    edir = tmp_path / "extraction"
    edir.mkdir()
    epath = edir / "structured_norms.parquet"
    _make_extraction_df().to_parquet(epath)
    return rpath, epath


# ── unit: _build_manifest ─────────────────────────────────────────────────

def test_build_manifest_deterministic(parquet_pair):
    rpath, epath = parquet_pair
    a = na._build_manifest(rpath, epath, n=5, seed=42, books=None)
    b = na._build_manifest(rpath, epath, n=5, seed=42, books=None)
    keys_a = [c["key"] for c in a["chunks"]]
    keys_b = [c["key"] for c in b["chunks"]]
    assert keys_a == keys_b
    assert a["manifest_hash"] == b["manifest_hash"]


def test_build_manifest_seed_changes_sample(parquet_pair):
    rpath, epath = parquet_pair
    a = na._build_manifest(rpath, epath, n=5, seed=0, books=None)
    b = na._build_manifest(rpath, epath, n=5, seed=1, books=None)
    # 5 of 9 chunks — almost-certainly different draws.
    assert [c["key"] for c in a["chunks"]] != [c["key"] for c in b["chunks"]]


def test_build_manifest_stats(parquet_pair):
    rpath, epath = parquet_pair
    # Sample all 9 chunks; stats should be exact.
    m = na._build_manifest(rpath, epath, n=9, seed=0, books=None)
    assert m["n_sampled"] == 9
    assert m["stats"]["chunks_with_norms"] == 2
    assert m["stats"]["chunks_without_norms"] == 7
    assert m["stats"]["total_model_norms"] == 3  # 2 from (100,0) + 1 from (200,1)


def test_build_manifest_oversample_clamps(parquet_pair, capsys):
    rpath, epath = parquet_pair
    m = na._build_manifest(rpath, epath, n=999, seed=0, books=None)
    assert m["n_sampled"] == 9
    captured = capsys.readouterr()
    assert "available chunks" in captured.err


def test_build_manifest_books_filter(parquet_pair):
    rpath, epath = parquet_pair
    m = na._build_manifest(rpath, epath, n=5, seed=0, books=["100"])
    assert {c["gutenberg_id"] for c in m["chunks"]} == {"100"}
    assert m["books_filter"] == ["100"]


def test_build_manifest_books_filter_empty_raises(parquet_pair):
    rpath, epath = parquet_pair
    with pytest.raises(SystemExit):
        na._build_manifest(rpath, epath, n=5, seed=0, books=["999"])


# ── unit: _build_html injection safety ────────────────────────────────────

def test_build_html_escapes_closing_script_tag():
    # Use a real manifest skeleton; only the chunk text differs.
    manifest = {
        "schema_version": 1,
        "manifest_hash": "sha256:test",
        "editable_fields": [list(t) for t in na.EDITABLE_FIELDS],
        "readonly_fields": [list(t) for t in na.READONLY_FIELDS],
        "chunks": [{
            "key": "100:0", "gutenberg_id": "100", "chunk_id": 0,
            "book_title": "B", "book_author": "A",
            "article_text": "evil </script><img src=x onerror=alert(1)>",
            "chunk_size": 10, "has_norms": False, "model_norms": [],
        }],
    }
    html = na._build_html(manifest)
    # Raw </script> must not survive inside the JSON literal.
    # (There's exactly one legitimate </script> at end of the inline script block.)
    assert html.count("</script>") == 1
    assert "<\\/script>" in html


def test_build_html_escapes_html_comments():
    manifest = {
        "schema_version": 1, "manifest_hash": "sha256:test",
        "editable_fields": [], "readonly_fields": [],
        "chunks": [{
            "key": "1:0", "gutenberg_id": "1", "chunk_id": 0,
            "book_title": "B", "book_author": "A",
            "article_text": "<!-- hidden --> and -->",
            "chunk_size": 0, "has_norms": False, "model_norms": [],
        }],
    }
    html = na._build_html(manifest)
    # Both opening and closing comment markers from CHUNK CONTENT should be neutralized.
    # The static template may legitimately contain neither, but if either did,
    # the count here would have to relax. As of writing, the template has none.
    assert "<!--" not in html
    # `-->` could appear in CSS/JS comment terminators in the template; the check
    # we care about is that the chunk's `-->` is escaped, so look inside the JSON.
    script_start = html.index("const MANIFEST = ")
    script_end = html.index(";", script_start)
    payload = html[script_start:script_end]
    assert "-->" not in payload
    assert "--\\>" in payload


# ── integration: cmd_merge round-trip with all decision types ─────────────

def _write_annotations(path: Path, manifest: dict, *, decisions: dict, added=None, chunk_notes=None):
    """decisions: {chunk_key: {norm_index_str: {decision: str, modified_fields?: dict, notes?: str}}}"""
    all_keys = set(decisions or {}) | set(added or {}) | set(chunk_notes or {})
    chunks_payload = {}
    for key in all_keys:
        chunks_payload[key] = {
            "model_norms": (decisions or {}).get(key, {}),
            "added_norms": (added or {}).get(key, []),
            "chunk_notes": (chunk_notes or {}).get(key, ""),
        }
    payload = {
        "schema_version": manifest["schema_version"],
        "sample_manifest_hash": manifest["manifest_hash"],
        "annotator": "tester",
        "annotated_at_utc": "2026-05-31T00:00:00Z",
        "n_chunks": len(manifest["chunks"]),
        "chunks": chunks_payload,
    }
    path.write_text(json.dumps(payload))


def test_merge_round_trip_all_decision_types(parquet_pair, tmp_path, capsys):
    rpath, epath = parquet_pair
    # Sample all 9 chunks so we definitely include (100,0) [2 norms] and (200,1) [1 norm].
    manifest = na._build_manifest(rpath, epath, n=9, seed=0, books=None)
    manifest_path = tmp_path / "m.manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    # confirm norm (100:0 #0), modify norm (100:0 #1), reject norm (200:1 #0)
    decisions = {
        "100:0": {
            "0": {"decision": "confirm"},
            "1": {"decision": "modify",
                  "modified_fields": {"raz_norm_subject": "Bob (corrected)",
                                      "raz_governs_info_flow": False},
                  "notes": "subject should be more specific"},
        },
        "200:1": {
            "0": {"decision": "reject", "notes": "not actually normative"},
        },
    }
    # Add an annotator-added norm on a chunk the model got nothing from
    added = {
        "300:0": [{
            "raz_norm_subject": "Dave", "raz_norm_act": "warn",
            "raz_condition_of_application": "before danger",
            "raz_normative_force": "obligation",
            "raz_norm_articulation": "Dave must warn the others.",
            "raz_governs_info_flow": True,
            "raz_info_flow_note": "to all present",
            "notes": "model missed this entirely",
        }],
    }
    chunk_notes = {"100:0": "chunk-level observation"}

    ann_path = tmp_path / "ann.json"
    _write_annotations(ann_path, manifest, decisions=decisions, added=added, chunk_notes=chunk_notes)

    out_path = tmp_path / "gold.parquet"
    rc = na.cmd_merge(argparse.Namespace(
        manifest=str(manifest_path), annotations=str(ann_path), output=str(out_path),
    ))
    assert rc == 0
    assert out_path.exists()

    df = pd.read_parquet(out_path)
    # Expect 3 model-norm rows (2 from 100:0, 1 from 200:1) + 1 added row.
    assert len(df) == 4
    by_dec = df["annot_decision"].value_counts().to_dict()
    assert by_dec.get("confirm") == 1
    assert by_dec.get("modify") == 1
    assert by_dec.get("reject") == 1
    assert by_dec.get("add") == 1

    # CONFIRM: gold should mirror base values exactly
    confirm_row = df[df["annot_decision"] == "confirm"].iloc[0]
    assert confirm_row["gold_raz_norm_subject"] == "a friend with a confidence to share"
    assert confirm_row["gold_raz_norm_act"] == "tell"

    # MODIFY: gold takes the annotator overrides; unmodified fields pass through base
    modify_row = df[df["annot_decision"] == "modify"].iloc[0]
    assert modify_row["gold_raz_norm_subject"] == "Bob (corrected)"
    assert modify_row["gold_raz_governs_info_flow"] is False or modify_row["gold_raz_governs_info_flow"] == 0
    # Untouched editable field falls through to base
    assert modify_row["gold_raz_norm_act"] == "keep secret"
    assert modify_row["annot_notes"] == "subject should be more specific"

    # REJECT: every gold_* field should be null
    reject_row = df[df["annot_decision"] == "reject"].iloc[0]
    for col, _, _ in na.EDITABLE_FIELDS:
        assert pd.isna(reject_row[f"gold_{col}"]) or reject_row[f"gold_{col}"] is None

    # ADD: gold fields populated from annotator, model fields null
    add_row = df[df["annot_decision"] == "add"].iloc[0]
    assert add_row["gold_raz_norm_subject"] == "Dave"
    assert add_row["gold_raz_norm_act"] == "warn"
    # The model-side raz_norm_subject (un-prefixed) should be null for an added row
    assert pd.isna(add_row["raz_norm_subject"]) or add_row["raz_norm_subject"] is None
    assert add_row["_annotation_source"] == "annotator_added"

    # Chunk notes carried through to every row of that chunk
    chunk_100_rows = df[(df["gutenberg_id"] == "100") & (df["chunk_id"] == 0)]
    assert (chunk_100_rows["chunk_notes"] == "chunk-level observation").all()

    # Verify the merge picked up the structured context column (book_summary)
    # — this is the column that silently vanishes if chunk_id dtype mismatches
    # and the lookup falls back to the manifest-embedded fields.
    confirm_row = df[df["annot_decision"] == "confirm"].iloc[0]
    assert confirm_row["book_summary"] == "Summary of book 100"


def test_merge_handles_uncertain_decision(parquet_pair, tmp_path):
    rpath, epath = parquet_pair
    manifest = na._build_manifest(rpath, epath, n=9, seed=0, books=None)
    manifest_path = tmp_path / "m.manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    decisions = {"100:0": {"0": {"decision": "uncertain", "notes": "ambiguous"}}}
    ann_path = tmp_path / "ann.json"
    _write_annotations(ann_path, manifest, decisions=decisions)

    out_path = tmp_path / "gold.parquet"
    rc = na.cmd_merge(argparse.Namespace(
        manifest=str(manifest_path), annotations=str(ann_path), output=str(out_path),
    ))
    assert rc == 0
    df = pd.read_parquet(out_path)
    uncertain_rows = df[df["annot_decision"] == "uncertain"]
    assert len(uncertain_rows) == 1
    # Like reject: gold fields all null
    for col, _, _ in na.EDITABLE_FIELDS:
        assert pd.isna(uncertain_rows.iloc[0][f"gold_{col}"]) or uncertain_rows.iloc[0][f"gold_{col}"] is None


def test_merge_survives_chunk_id_dtype_drift(parquet_pair, tmp_path):
    """The merge must work even if extraction.chunk_id round-trips as object/string."""
    rpath, epath = parquet_pair
    # Rewrite extraction with chunk_id as string to mimic a parquet dtype drift.
    ext = pd.read_parquet(epath)
    ext["chunk_id"] = ext["chunk_id"].astype(str)
    ext.to_parquet(epath)

    manifest = na._build_manifest(rpath, epath, n=9, seed=0, books=None)
    manifest_path = tmp_path / "m.manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    decisions = {"100:0": {"0": {"decision": "confirm"}}}
    ann_path = tmp_path / "ann.json"
    _write_annotations(ann_path, manifest, decisions=decisions)

    out_path = tmp_path / "gold.parquet"
    rc = na.cmd_merge(argparse.Namespace(
        manifest=str(manifest_path), annotations=str(ann_path), output=str(out_path),
    ))
    assert rc == 0
    df = pd.read_parquet(out_path)
    # The confirm row should still find its extraction row and carry book_summary,
    # which only lives on the extraction side.
    confirm_row = df[df["annot_decision"] == "confirm"].iloc[0]
    assert confirm_row["book_summary"] == "Summary of book 100", \
        "extraction lookup silently failed — book_summary missing means chunk_id dtype drift broke the merge"


def test_merge_warns_on_hash_mismatch(parquet_pair, tmp_path, capsys):
    rpath, epath = parquet_pair
    manifest = na._build_manifest(rpath, epath, n=5, seed=0, books=None)
    manifest_path = tmp_path / "m.manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    # Annotations from a "different" sample (deliberately wrong hash).
    payload = {
        "schema_version": 1,
        "sample_manifest_hash": "sha256:wrong",
        "annotator": "tester",
        "annotated_at_utc": "2026-05-31T00:00:00Z",
        "n_chunks": 0,
        "chunks": {"100:0": {"model_norms": {"0": {"decision": "confirm"}},
                              "added_norms": [], "chunk_notes": ""}},
    }
    ann_path = tmp_path / "ann.json"
    ann_path.write_text(json.dumps(payload))

    out_path = tmp_path / "gold.parquet"
    rc = na.cmd_merge(argparse.Namespace(
        manifest=str(manifest_path), annotations=str(ann_path), output=str(out_path),
    ))
    # Either succeeds with a warning (most likely — keys may still intersect)
    # or returns 1 (no rows merged); in either case stderr must carry the warning.
    captured = capsys.readouterr()
    assert "hash mismatch" in captured.err
    assert rc in (0, 1)


# ── CLI smoke ─────────────────────────────────────────────────────────────

def test_export_writes_html_and_manifest(parquet_pair, tmp_path):
    rpath, epath = parquet_pair
    out_html = tmp_path / "out.html"
    rc = na.cmd_export(argparse.Namespace(
        reasoning=str(rpath), extraction=str(epath),
        n=5, seed=42, books=None, output=str(out_html), schema="norms",
    ))
    assert rc == 0
    assert out_html.exists()
    manifest_path = out_html.with_suffix(".manifest.json")
    assert manifest_path.exists()

    html = out_html.read_text()
    assert "const MANIFEST = " in html
    assert "norms_annotator:" in html  # storage key prefix

    manifest = json.loads(manifest_path.read_text())
    assert manifest["seed"] == 42
    assert "manifest_hash" in manifest
    assert manifest["schema_key"] == "norms"
    assert manifest["item_label_singular"] == "Norm"


# ── pre-abstraction view + abstraction-bad flag ──────────────────────────

def test_build_manifest_loads_orig_columns_and_propagates_before_fields(parquet_pair):
    rpath, epath = parquet_pair
    m = na._build_manifest(rpath, epath, n=9, seed=0, books=None, schema_key="norms")
    # before_fields propagated through the manifest top-level
    assert "before_fields" in m
    assert m["before_fields"]["raz_norm_subject"] == "orig_raz_norm_subject"
    # orig_raz_* values land on each model_norm dict so the JS can show them
    sample_norm = None
    for c in m["chunks"]:
        for mn in c["model_norms"]:
            if mn.get("orig_raz_norm_subject") == "Alice":
                sample_norm = mn
    assert sample_norm is not None, "orig_raz_norm_subject not loaded into manifest model_norms"
    # role_rationale (readonly with textarea widget) also makes it through
    assert sample_norm.get("role_rationale", "").startswith("Alice is in the role")


def test_html_renders_pre_abstraction_block_and_flag_toggle(parquet_pair, tmp_path):
    rpath, epath = parquet_pair
    out_html = tmp_path / "out.html"
    na.cmd_export(argparse.Namespace(
        reasoning=str(rpath), extraction=str(epath),
        n=9, per_book=None, seed=0, books=None, output=str(out_html), schema="norms",
    ))
    html = out_html.read_text()
    # The CSS hooks for the pre-abstraction view must be present.
    assert ".field-input .before" in html
    assert ".flagged-bad" in html
    # JS pulls BEFORE_FIELDS from the manifest constant.
    assert "BEFORE_FIELDS" in html
    # The literal user-facing string for the toggle must reach the page.
    assert "flag this abstraction as bad" in html


def test_merge_emits_abstraction_flagged_columns(parquet_pair, tmp_path):
    """Annotator flags subject + articulation as bad on one norm.

    The merged parquet must carry boolean abstraction_flagged_<col> columns
    for every field with a before-mapping (defaults False; True when flagged).
    """
    rpath, epath = parquet_pair
    manifest = na._build_manifest(rpath, epath, n=9, seed=0, books=None, schema_key="norms")
    manifest_path = tmp_path / "m.manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    decisions = {
        "100:0": {
            "0": {  # the Alice norm — flag subject as bad, leave act untouched
                "decision": "modify",
                "modified_fields": {"raz_norm_subject": "a friend with a confidence (corrected)"},
                "abstraction_flags": {
                    "raz_norm_subject": True,
                    "raz_norm_articulation": True,
                },
                "notes": "subject abstraction loses the named character",
            },
            "1": {"decision": "confirm"},  # Bob — no flags, no corrections
        },
    }
    ann_path = tmp_path / "ann.json"
    _write_annotations(ann_path, manifest, decisions=decisions)

    out_path = tmp_path / "gold.parquet"
    rc = na.cmd_merge(argparse.Namespace(
        manifest=str(manifest_path), annotations=str(ann_path), output=str(out_path),
    ))
    assert rc == 0
    df = pd.read_parquet(out_path)

    # The flagged columns exist for every before-mapped editable field.
    expected = {f"abstraction_flagged_{c}" for c in manifest["before_fields"]}
    assert expected.issubset(df.columns), f"missing flag columns: {expected - set(df.columns)}"

    alice = df[(df["gutenberg_id"] == "100") & (df["norm_index"] == 0)].iloc[0]
    assert alice["abstraction_flagged_raz_norm_subject"] is True or alice["abstraction_flagged_raz_norm_subject"] == 1
    assert alice["abstraction_flagged_raz_norm_articulation"] is True or alice["abstraction_flagged_raz_norm_articulation"] == 1
    # Unflagged fields default to False
    assert alice["abstraction_flagged_raz_norm_act"] in (False, 0)
    # The correction landed in gold_*
    assert alice["gold_raz_norm_subject"] == "a friend with a confidence (corrected)"

    bob = df[(df["gutenberg_id"] == "100") & (df["norm_index"] == 1)].iloc[0]
    # No flags on Bob's norm
    for c in manifest["before_fields"]:
        assert bob[f"abstraction_flagged_{c}"] in (False, 0)


def test_flows_schema_has_no_before_fields(flows_parquet_pair):
    """Flows have no role-abstraction step → before_fields should be empty,
    and the merged parquet must NOT carry abstraction_flagged_* columns."""
    rpath, epath = flows_parquet_pair
    m = na._build_manifest(rpath, epath, n=5, seed=0, books=None, schema_key="flows")
    assert m["before_fields"] == {}


# ── stratified-by-book sampling ──────────────────────────────────────────

def test_build_manifest_per_book_even_distribution(parquet_pair):
    """3 books × 3 chunks → per_book=2 → 2 chunks per book, 6 total."""
    rpath, epath = parquet_pair
    m = na._build_manifest(rpath, epath, n=None, seed=0, books=None,
                            per_book=2, schema_key="norms")
    assert m["sampling_strategy"] == "stratified_by_book"
    assert m["per_book"] == 2
    assert m["n_sampled"] == 6
    by_book = {}
    for c in m["chunks"]:
        by_book[c["gutenberg_id"]] = by_book.get(c["gutenberg_id"], 0) + 1
    assert by_book == {"100": 2, "200": 2, "300": 2}


def test_build_manifest_per_book_clamps_undersized_books(parquet_pair, capsys):
    """per_book=10 against fixture with only 3 chunks per book → take all 9 + warnings."""
    rpath, epath = parquet_pair
    m = na._build_manifest(rpath, epath, n=None, seed=0, books=None,
                            per_book=10, schema_key="norms")
    assert m["n_sampled"] == 9
    by_book = {}
    for c in m["chunks"]:
        by_book[c["gutenberg_id"]] = by_book.get(c["gutenberg_id"], 0) + 1
    assert by_book == {"100": 3, "200": 3, "300": 3}
    captured = capsys.readouterr()
    # Each undersized book emits a warning.
    assert captured.err.count("only 3 chunks") == 3
    # per_book_actual records the per-book truth.
    assert m["per_book_actual"] == {"100": 3, "200": 3, "300": 3}


def test_build_manifest_per_book_deterministic_under_seed(parquet_pair):
    rpath, epath = parquet_pair
    a = na._build_manifest(rpath, epath, n=None, seed=7, books=None, per_book=2)
    b = na._build_manifest(rpath, epath, n=None, seed=7, books=None, per_book=2)
    assert [c["key"] for c in a["chunks"]] == [c["key"] for c in b["chunks"]]
    assert a["manifest_hash"] == b["manifest_hash"]


def test_build_manifest_rejects_both_n_and_per_book(parquet_pair):
    rpath, epath = parquet_pair
    with pytest.raises(SystemExit, match="exactly one of"):
        na._build_manifest(rpath, epath, n=5, seed=0, books=None, per_book=2)


def test_build_manifest_rejects_neither_n_nor_per_book(parquet_pair):
    rpath, epath = parquet_pair
    with pytest.raises(SystemExit, match="exactly one of"):
        na._build_manifest(rpath, epath, n=None, seed=0, books=None, per_book=None)


def test_cmd_export_per_book_writes_stratified_manifest(parquet_pair, tmp_path):
    rpath, epath = parquet_pair
    out_html = tmp_path / "out.html"
    rc = na.cmd_export(argparse.Namespace(
        reasoning=str(rpath), extraction=str(epath),
        n=None, per_book=2, seed=42, books=None, output=str(out_html), schema="norms",
    ))
    assert rc == 0
    manifest = json.loads(out_html.with_suffix(".manifest.json").read_text())
    assert manifest["sampling_strategy"] == "stratified_by_book"
    assert manifest["per_book"] == 2
    assert manifest["n_sampled"] == 6


def test_cmd_export_rejects_both_n_and_per_book(parquet_pair, tmp_path, capsys):
    rpath, epath = parquet_pair
    rc = na.cmd_export(argparse.Namespace(
        reasoning=str(rpath), extraction=str(epath),
        n=5, per_book=2, seed=42, books=None,
        output=str(tmp_path / "out.html"), schema="norms",
    ))
    assert rc == 2
    assert "exactly one of" in capsys.readouterr().err


# ── flows-schema coverage ────────────────────────────────────────────────

def _make_flows_reasoning_df():
    """5 chunks; 3 have has_information_exchange=True, 2 don't."""
    rows = []
    for gb_id, ch_id, gate in [
        ("11", 0, True), ("11", 1, False), ("11", 2, True),
        ("135", 0, True), ("135", 1, False),
    ]:
        rows.append({
            "gutenberg_id": gb_id, "chunk_id": ch_id,
            "article_text": f"Text of {gb_id}:{ch_id}",
            "chunk_size": 13,
            "book_title": f"Book {gb_id}", "book_author": f"Author {gb_id}",
            "book_summary": f"Summary {gb_id}",
            "has_information_exchange": gate,
            "ci_flow_count": 1 if gate else 0,
            "generated_text": "{}",
        })
    return pd.DataFrame(rows)


def _make_flows_extraction_df():
    """One flow per has_information_exchange=True chunk."""
    rows = []
    for gb_id, ch_id, sender in [("11", 0, "Pip"), ("11", 2, "Estella"), ("135", 0, "Valjean")]:
        rows.append({
            "gutenberg_id": gb_id, "chunk_id": ch_id, "ci_flow_index": 0,
            "article_text": f"Text of {gb_id}:{ch_id}", "chunk_size": 13,
            "book_title": f"Book {gb_id}", "book_author": f"Author {gb_id}",
            "book_summary": f"Summary {gb_id}",
            "ci_subject": sender, "ci_sender": sender, "ci_recipient": "Other",
            "ci_information_type": "secret", "ci_transmission_principle": "discretion",
            "ci_appropriateness": "appropriate", "ci_flow_direction": "outgoing",
            "ci_context": "private conversation",
            "ci_norms_invoked": "confidentiality",
            "ci_flow_snippet": f"snippet for {gb_id}:{ch_id}",
            "ci_norm_source": "narrator", "ci_is_new_flow": True,
            "ci_confidence_qual": "high", "ci_confidence_quant": 0.9,
        })
    return pd.DataFrame(rows)


@pytest.fixture
def flows_parquet_pair(tmp_path):
    rdir = tmp_path / "flows_reasoning"; rdir.mkdir()
    rpath = rdir / "ci_reasoning.parquet"
    _make_flows_reasoning_df().to_parquet(rpath)

    edir = tmp_path / "flows_extraction"; edir.mkdir()
    epath = edir / "ci_flows.parquet"
    _make_flows_extraction_df().to_parquet(epath)
    return rpath, epath


def test_build_manifest_flows_schema_uses_correct_gate_and_index(flows_parquet_pair):
    rpath, epath = flows_parquet_pair
    m = na._build_manifest(rpath, epath, n=5, seed=0, books=None, schema_key="flows")
    assert m["schema_key"] == "flows"
    assert m["gate_column"] == "has_information_exchange"
    assert m["item_index_column"] == "ci_flow_index"
    assert m["item_label_singular"] == "Flow"
    # 5 chunks total; 3 gated-in, 2 gated-out.
    assert m["stats"]["chunks_with_norms"] == 3
    assert m["stats"]["chunks_without_norms"] == 2
    assert m["stats"]["total_model_norms"] == 3
    # editable_fields list must come from the flows schema
    cols = [t[0] for t in m["editable_fields"]]
    assert "ci_sender" in cols
    assert "ci_recipient" in cols
    assert "raz_norm_subject" not in cols


def test_merge_flows_schema_round_trip(flows_parquet_pair, tmp_path):
    rpath, epath = flows_parquet_pair
    manifest = na._build_manifest(rpath, epath, n=5, seed=0, books=None, schema_key="flows")
    manifest_path = tmp_path / "m.manifest.json"
    manifest_path.write_text(json.dumps(manifest))

    # Confirm one flow, modify one, reject one.
    decisions = {
        "11:0": {"0": {"decision": "confirm"}},
        "11:2": {"0": {"decision": "modify",
                         "modified_fields": {"ci_recipient": "Pip (corrected)"},
                         "notes": "recipient was actually Pip"}},
        "135:0": {"0": {"decision": "reject", "notes": "not a real flow"}},
    }
    ann_path = tmp_path / "ann.json"
    _write_annotations(ann_path, manifest, decisions=decisions)

    out_path = tmp_path / "gold.parquet"
    rc = na.cmd_merge(argparse.Namespace(
        manifest=str(manifest_path), annotations=str(ann_path), output=str(out_path),
    ))
    assert rc == 0
    df = pd.read_parquet(out_path)
    # 3 model flows annotated.
    assert len(df) == 3
    modify_row = df[df["annot_decision"] == "modify"].iloc[0]
    assert modify_row["gold_ci_recipient"] == "Pip (corrected)"
    # Unmodified flow fields fall through from base
    assert modify_row["gold_ci_sender"] == "Estella"
    # Confirm carries the original
    confirm_row = df[df["annot_decision"] == "confirm"].iloc[0]
    assert confirm_row["gold_ci_sender"] == "Pip"
    # Reject nulls out gold fields
    reject_row = df[df["annot_decision"] == "reject"].iloc[0]
    assert pd.isna(reject_row["gold_ci_sender"]) or reject_row["gold_ci_sender"] is None
    # book_summary survives the merge (regression guard for chunk_id dtype path)
    assert confirm_row["book_summary"] == "Summary 11"


def test_export_flows_schema_html_carries_flow_labels(flows_parquet_pair, tmp_path):
    rpath, epath = flows_parquet_pair
    out_html = tmp_path / "flows.html"
    rc = na.cmd_export(argparse.Namespace(
        reasoning=str(rpath), extraction=str(epath),
        n=5, seed=42, books=None, output=str(out_html), schema="flows",
    ))
    assert rc == 0
    html = out_html.read_text()
    # The schema-driven labels should land in the manifest section of the page.
    assert '"schema_key": "flows"' in html or '"schema_key":"flows"' in html
    assert "Flow" in html  # singular label
    manifest_path = out_html.with_suffix(".manifest.json")
    manifest = json.loads(manifest_path.read_text())
    assert manifest["schema_key"] == "flows"
    assert manifest["gate_column"] == "has_information_exchange"
