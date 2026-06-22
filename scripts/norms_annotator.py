#!/usr/bin/env python3
"""Norms Annotator — sample chunks for ground-truth annotation and merge results.

Two subcommands:

    export   build a self-contained HTML annotation tool + a manifest JSON
             from a uniform random sample of N chunks across the reasoning
             parquet (all chunks, regardless of has_norms gate).

    merge    join a downloaded annotations JSON back against the original
             structured_norms parquet to produce a gold_labels parquet with
             per-norm decisions (confirm / modify / reject / uncertain) and
             annotator-added norms appended.

Annotation actions captured per model norm:
    - confirm   norm is correct as-is
    - modify    norm partially correct; annotator's edits overlaid
    - reject    not actually a norm
    - uncertain unclear / skip

Plus annotators can add norms the model missed (one or more per chunk).

Usage:

    python -m scripts.norms_annotator export \\
        --reasoning  /path/to/reasoning.parquet \\
        --extraction /path/to/structured_norms.parquet \\
        -n 50 --seed 42 \\
        -o /abs/path/qwen36_n50.html

    # → writes qwen36_n50.html (open in browser, annotate, click "Download annotations")
    # → also writes qwen36_n50.manifest.json beside it

    python -m scripts.norms_annotator merge \\
        --manifest    /abs/path/qwen36_n50.manifest.json \\
        --annotations ~/Downloads/qwen36_n50.annotations.json \\
        -o /abs/path/qwen36_n50.gold.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


SCHEMA_VERSION = 1

# Per-schema column conventions. Internally the manifest always uses
# `has_norms` (gate) and `norm_index` (per-item index) — the loaders below
# alias the schema's actual column names onto those keys so every downstream
# code path is schema-agnostic.
SCHEMAS: dict[str, dict[str, Any]] = {
    "norms": {
        "gate_column": "has_norms",
        "item_index_column": "norm_index",
        "item_label_singular": "Norm",
        "item_label_plural": "norms",
        # (column, label, widget)  widget ∈ {input, textarea, checkbox}
        "editable_fields": [
            ("raz_norm_subject",             "Subject",            "input"),
            ("raz_norm_act",                 "Act",                "input"),
            ("raz_condition_of_application", "Condition",          "input"),
            ("raz_normative_force",          "Normative force",    "input"),
            ("raz_norm_articulation",        "Articulation",       "textarea"),
            ("raz_governs_info_flow",        "Governs info flow",  "checkbox"),
            ("raz_info_flow_note",           "Info-flow note",     "input"),
        ],
        # Optional 3rd tuple element = widget hint ("input"|"textarea"); defaults to "input".
        "readonly_fields": [
            ("raz_prescriptive_element", "Prescriptive element"),
            ("raz_context",              "Context"),
            ("raz_norm_source",          "Norm source"),
            ("raz_confidence_qual",      "Confidence (qual)"),
            ("raz_confidence_quant",     "Confidence (quant)"),
            ("role_rationale",           "Role-abstraction rationale", "textarea"),
            ("role_abstraction_failed",  "Role-abstraction failed"),
        ],
        # Per-editable-field map to the pre-role-abstraction column. When the
        # abstracted value differs from the original, the UI renders a
        # "pre-abstraction: …" subtitle so the annotator can validate the rewrite.
        "before_fields": {
            "raz_norm_subject":             "orig_raz_norm_subject",
            "raz_norm_act":                 "orig_raz_norm_act",
            "raz_condition_of_application": "orig_raz_condition_of_application",
            "raz_norm_articulation":        "orig_raz_norm_articulation",
        },
    },
    "flows": {
        "gate_column": "has_information_exchange",
        "item_index_column": "ci_flow_index",
        "item_label_singular": "Flow",
        "item_label_plural": "flows",
        "editable_fields": [
            ("ci_subject",                 "Subject",                "input"),
            ("ci_sender",                  "Sender",                 "input"),
            ("ci_recipient",               "Recipient",              "input"),
            ("ci_information_type",        "Information type",       "input"),
            ("ci_transmission_principle",  "Transmission principle", "input"),
            ("ci_appropriateness",         "Appropriateness",        "input"),
            ("ci_flow_direction",          "Direction",              "input"),
            ("ci_context",                 "Context",                "textarea"),
            ("ci_norms_invoked",           "Norms invoked",          "textarea"),
            ("ci_flow_snippet",            "Flow snippet",           "textarea"),
        ],
        "readonly_fields": [
            ("ci_norm_source",       "Norm source"),
            ("ci_is_new_flow",       "Is new flow"),
            ("ci_confidence_qual",   "Confidence (qual)"),
            ("ci_confidence_quant",  "Confidence (quant)"),
        ],
        "before_fields": {},
    },
}

# Back-compat aliases used by tests written against the pre-multi-schema API.
EDITABLE_FIELDS: list[tuple[str, str, str]] = SCHEMAS["norms"]["editable_fields"]
READONLY_FIELDS: list[tuple[str, str]] = SCHEMAS["norms"]["readonly_fields"]


def _serialize(v: Any) -> Any:
    """Make a value JSON-serializable. Lifted from norms_inspector."""
    if v is None:
        return None
    if isinstance(v, float) and (np.isnan(v) or np.isinf(v)):
        return None
    if isinstance(v, np.ndarray):
        return [_serialize(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return float(v)
    if isinstance(v, (np.bool_,)):
        return bool(v)
    if isinstance(v, dict):
        return {k: _serialize(val) for k, val in v.items()}
    if isinstance(v, (list, tuple)):
        return [_serialize(x) for x in v]
    if isinstance(v, bytes):
        return v.decode("utf-8", errors="replace")
    return v


# ── EXPORT ───────────────────────────────────────────────────────────────

def _build_manifest(
    reasoning_path: Path,
    extraction_path: Path,
    n: int | None,
    seed: int,
    books: list[str] | None,
    schema_key: str = "norms",
    per_book: int | None = None,
) -> dict:
    if (n is None) == (per_book is None):
        raise SystemExit("exactly one of n / per_book must be provided")
    if schema_key not in SCHEMAS:
        raise SystemExit(f"unknown schema {schema_key!r}; choose from {sorted(SCHEMAS)}")
    schema = SCHEMAS[schema_key]
    editable_fields = schema["editable_fields"]
    readonly_fields = schema["readonly_fields"]
    gate_col = schema["gate_column"]
    item_index_col = schema["item_index_column"]
    before_fields = dict(schema.get("before_fields") or {})

    reasoning = pd.read_parquet(reasoning_path)
    reasoning["gutenberg_id"] = reasoning["gutenberg_id"].astype(str)
    reasoning["chunk_id"] = pd.to_numeric(reasoning["chunk_id"], errors="coerce").astype("Int64")
    # Alias the schema's gate column onto the internal name `has_norms` so the
    # downstream code stays schema-agnostic. If the gate column is missing,
    # default to True (treat every chunk as gated-in).
    if gate_col in reasoning.columns:
        reasoning["has_norms"] = reasoning[gate_col].astype(bool)
    elif "has_norms" not in reasoning.columns:
        reasoning["has_norms"] = True

    extraction = pd.read_parquet(extraction_path)
    extraction["gutenberg_id"] = extraction["gutenberg_id"].astype(str)
    # Normalize chunk_id / item-index to plain int so groupby/lookup keys align
    # with the int-keyed chunk_meta index built from reasoning — otherwise a
    # parquet dtype drift silently drops every chunk's model_norms.
    extraction["chunk_id"] = pd.to_numeric(extraction["chunk_id"], errors="coerce").astype("Int64")
    if item_index_col in extraction.columns:
        extraction[item_index_col] = pd.to_numeric(extraction[item_index_col], errors="coerce").astype("Int64")
        # Alias the schema's item-index column onto the internal name `norm_index`.
        if item_index_col != "norm_index":
            extraction["norm_index"] = extraction[item_index_col]

    # Unique chunks (reasoning is exploded per-norm-snippet, so dedupe)
    chunk_keys = (
        reasoning[["gutenberg_id", "chunk_id"]]
        .drop_duplicates()
        .sort_values(["gutenberg_id", "chunk_id"])
        .reset_index(drop=True)
    )

    if books:
        book_set = set(books)
        chunk_keys = chunk_keys[chunk_keys["gutenberg_id"].isin(book_set)].reset_index(drop=True)
        if chunk_keys.empty:
            raise SystemExit(f"No chunks found for books: {books}")

    n_avail = len(chunk_keys)
    rng = np.random.default_rng(seed)

    if per_book is not None:
        # Stratified: take per_book chunks from each gutenberg_id, deterministic
        # under seed by iterating books in sorted order.
        sampled_parts = []
        per_book_actual = {}
        for gb_id in sorted(chunk_keys["gutenberg_id"].unique()):
            group = chunk_keys[chunk_keys["gutenberg_id"] == gb_id]
            n_take = min(per_book, len(group))
            if n_take < per_book:
                print(f"[warn] book {gb_id} has only {len(group)} chunks; taking all {n_take}",
                      file=sys.stderr)
            local_idx = rng.choice(len(group), size=n_take, replace=False)
            local_idx.sort()
            sampled_parts.append(group.iloc[local_idx])
            per_book_actual[gb_id] = int(n_take)
        sampled = (
            pd.concat(sampled_parts)
            .sort_values(["gutenberg_id", "chunk_id"])
            .reset_index(drop=True)
        )
        sampling_strategy = "stratified_by_book"
        n_effective = len(sampled)
    else:
        if n > n_avail:
            print(f"[warn] requested n={n} > available chunks {n_avail}; using all", file=sys.stderr)
            n = n_avail
        sampled_idx = rng.choice(n_avail, size=n, replace=False)
        sampled_idx.sort()  # stable display order
        sampled = chunk_keys.iloc[sampled_idx].reset_index(drop=True)
        sampling_strategy = "uniform"
        per_book_actual = None
        n_effective = n

    # One canonical chunk row from reasoning (first occurrence carries chunk metadata)
    chunk_meta = (
        reasoning.drop_duplicates(subset=["gutenberg_id", "chunk_id"], keep="first")
        .set_index(["gutenberg_id", "chunk_id"])
    )

    # Per-chunk has_norms flag: True if ANY reasoning row for that chunk says has_norms
    has_norms_by_chunk = (
        reasoning.groupby(["gutenberg_id", "chunk_id"])["has_norms"].any()
    )

    # Model norms (post-extraction, structured): group by chunk
    model_norms_by_chunk: dict[tuple[str, int], list[dict]] = {}
    editable_cols = {t[0] for t in editable_fields}
    readonly_cols = {t[0] for t in readonly_fields}
    before_cols = set(before_fields.values())
    norm_cols = list(editable_cols | readonly_cols | before_cols | {"norm_index"})
    for key, sub in extraction.groupby(["gutenberg_id", "chunk_id"]):
        items = []
        for _, row in sub.iterrows():
            d = {c: _serialize(row.get(c)) for c in norm_cols if c in row.index}
            items.append(d)
        items.sort(key=lambda d: (d.get("norm_index") if d.get("norm_index") is not None else 0))
        model_norms_by_chunk[key] = items

    chunks_out: list[dict] = []
    n_with_norms = 0
    n_without_norms = 0
    for _, r in sampled.iterrows():
        key = (r["gutenberg_id"], int(r["chunk_id"]))
        if key not in chunk_meta.index:
            continue
        meta = chunk_meta.loc[key]
        # If duplicate index (rare) take first
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]

        model_norms = model_norms_by_chunk.get(key, [])
        has_norms = bool(has_norms_by_chunk.get(key, False))
        n_with_norms += int(has_norms)
        n_without_norms += int(not has_norms)

        chunks_out.append({
            "key": f"{key[0]}:{key[1]}",
            "gutenberg_id": key[0],
            "chunk_id": key[1],
            "book_title": _serialize(meta.get("book_title")),
            "book_author": _serialize(meta.get("book_author")),
            "article_text": _serialize(meta.get("article_text")),
            "chunk_size": _serialize(meta.get("chunk_size")),
            "has_norms": has_norms,
            "model_norms": model_norms,
        })

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "created_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "schema_key": schema_key,
        "gate_column": gate_col,
        "item_index_column": item_index_col,
        "item_label_singular": schema["item_label_singular"],
        "item_label_plural": schema["item_label_plural"],
        "source_reasoning": str(reasoning_path.resolve()),
        "source_extraction": str(extraction_path.resolve()),
        "seed": seed,
        "sampling_strategy": sampling_strategy,
        "per_book": per_book,
        "per_book_actual": per_book_actual,
        "n_requested": n_effective,
        "n_sampled": len(chunks_out),
        "books_filter": books,
        "stats": {
            "chunks_with_norms": n_with_norms,
            "chunks_without_norms": n_without_norms,
            "total_model_norms": sum(len(c["model_norms"]) for c in chunks_out),
        },
        "editable_fields": [list(t) for t in editable_fields],
        "readonly_fields": [list(t) for t in readonly_fields],
        "before_fields": before_fields,
        "chunks": chunks_out,
    }

    # Hash for round-trip sanity-checking (compute over canonical *content*).
    # Exclude manifest_hash itself and created_at_utc: the timestamp is not
    # content, and including it made the hash nondeterministic for identical
    # inputs (same sample re-exported → different storage key), defeating the
    # hash's purpose as a stable storage key / sample-mismatch detector.
    _non_content = {"manifest_hash", "created_at_utc"}
    payload = {k: v for k, v in manifest.items() if k not in _non_content}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    manifest["manifest_hash"] = "sha256:" + hashlib.sha256(blob).hexdigest()
    return manifest


def cmd_export(args: argparse.Namespace) -> int:
    out_html = Path(args.output).resolve()
    if not out_html.parent.exists():
        print(f"[error] parent dir does not exist: {out_html.parent}", file=sys.stderr)
        return 2

    schema_key = getattr(args, "schema", "norms") or "norms"
    per_book = getattr(args, "per_book", None)
    n_arg = getattr(args, "n", None)
    if (n_arg is None) == (per_book is None):
        print("[error] exactly one of -n / --per-book must be provided", file=sys.stderr)
        return 2
    manifest = _build_manifest(
        reasoning_path=Path(args.reasoning),
        extraction_path=Path(args.extraction),
        n=n_arg,
        seed=args.seed,
        books=args.books.split(",") if args.books else None,
        schema_key=schema_key,
        per_book=per_book,
    )

    manifest_path = out_html.with_suffix(".manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))

    html = _build_html(manifest)
    out_html.write_text(html)

    s = manifest["stats"]
    gate_col = manifest.get("gate_column", "has_norms")
    plural = manifest.get("item_label_plural", "norms")
    strategy = manifest.get("sampling_strategy", "uniform")
    print(f"[export] wrote {out_html}")
    print(f"[export] wrote {manifest_path}")
    print(f"[export] schema={schema_key} sampling={strategy}")
    if strategy == "stratified_by_book":
        print(f"[export] sampled n={manifest['n_sampled']} chunks "
              f"(per_book={per_book}, seed={args.seed})")
        actual = manifest.get("per_book_actual") or {}
        short_books = [gb for gb, k in actual.items() if k < (per_book or 0)]
        if short_books:
            print(f"         {len(short_books)} book(s) under quota: {short_books}")
    else:
        print(f"[export] sampled n={manifest['n_sampled']} chunks (seed={args.seed})")
    print(f"         {gate_col}=True:  {s['chunks_with_norms']}")
    print(f"         {gate_col}=False: {s['chunks_without_norms']}")
    print(f"         total model {plural} to annotate: {s['total_model_norms']}")
    print(f"         manifest_hash: {manifest['manifest_hash']}")
    return 0


# ── MERGE ────────────────────────────────────────────────────────────────

def cmd_merge(args: argparse.Namespace) -> int:
    manifest = json.loads(Path(args.manifest).read_text())
    annotations = json.loads(Path(args.annotations).read_text())

    expected_hash = manifest.get("manifest_hash")
    seen_hash = annotations.get("sample_manifest_hash")
    if expected_hash and seen_hash and expected_hash != seen_hash:
        print(
            f"[warn] manifest hash mismatch — annotations may be from a different "
            f"sample.\n  manifest:   {expected_hash}\n  annotation: {seen_hash}",
            file=sys.stderr,
        )

    schema_key = manifest.get("schema_key", "norms")
    item_index_col = manifest.get("item_index_column", "norm_index")
    # Prefer the per-manifest editable_fields list (round-trip-correct);
    # fall back to the in-process schema if the manifest pre-dates the field.
    editable_fields = manifest.get("editable_fields") or [list(t) for t in SCHEMAS[schema_key]["editable_fields"]]
    editable_cols = [t[0] for t in editable_fields]
    before_fields_map = dict(manifest.get("before_fields") or {})
    before_cols = sorted(before_fields_map)  # raz_* cols that have a pre-abstraction column

    extraction_path = Path(manifest["source_extraction"])
    extraction = pd.read_parquet(extraction_path)
    extraction["gutenberg_id"] = extraction["gutenberg_id"].astype(str)
    # Normalize chunk_id / item-index to plain int so equality holds against the
    # int values carried in the manifest, regardless of parquet round-trip dtype.
    if "chunk_id" in extraction.columns:
        extraction["chunk_id"] = pd.to_numeric(extraction["chunk_id"], errors="coerce").astype("Int64")
    if item_index_col in extraction.columns:
        extraction[item_index_col] = pd.to_numeric(extraction[item_index_col], errors="coerce").astype("Int64")
        if item_index_col != "norm_index":
            extraction["norm_index"] = extraction[item_index_col]

    chunk_meta_cols = [
        "gutenberg_id", "chunk_id", "article_text", "chunk_size",
        "book_title", "book_author", "book_summary",
    ]
    chunk_meta_cols = [c for c in chunk_meta_cols if c in extraction.columns]

    annotator = annotations.get("annotator")
    annotated_at = annotations.get("annotated_at_utc")
    chunks_ann = annotations.get("chunks", {})

    rows: list[dict] = []
    for chunk in manifest["chunks"]:
        key = chunk["key"]
        gb_id = chunk["gutenberg_id"]
        ch_id = chunk["chunk_id"]
        ann = chunks_ann.get(key, {})
        model_anns: dict[str, dict] = ann.get("model_norms", {})
        added_anns: list[dict] = ann.get("added_norms", [])
        chunk_notes = ann.get("chunk_notes") or None

        # Model norms — pull the corresponding extraction row for each
        for mn in chunk["model_norms"]:
            ni = mn.get("norm_index")
            ann_key = str(ni) if ni is not None else ""
            sel = extraction[
                (extraction["gutenberg_id"] == gb_id)
                & (extraction["chunk_id"] == ch_id)
                & (extraction.get("norm_index") == ni)
            ]
            if sel.empty:
                # Fall back to manifest-embedded fields
                base = {k: mn.get(k) for k in extraction.columns if k in mn}
                base.update({"gutenberg_id": gb_id, "chunk_id": ch_id, "norm_index": ni})
            else:
                base = sel.iloc[0].to_dict()

            ann_norm = model_anns.get(ann_key, {})
            decision = ann_norm.get("decision")  # may be None (unannotated)
            modified = ann_norm.get("modified_fields") or {}
            abstraction_flags = ann_norm.get("abstraction_flags") or {}

            gold = {}
            if decision == "confirm":
                for c in editable_cols:
                    gold[f"gold_{c}"] = base.get(c)
            elif decision == "modify":
                for c in editable_cols:
                    gold[f"gold_{c}"] = modified.get(c, base.get(c))
            elif decision in ("reject", "uncertain"):
                for c in editable_cols:
                    gold[f"gold_{c}"] = None
            else:
                for c in editable_cols:
                    gold[f"gold_{c}"] = None  # unannotated → null gold

            # Per-field abstraction flag: True = annotator marked the role
            # abstraction as bad for this column. Default False (not flagged).
            # Only emit columns for fields that actually have a pre-abstraction
            # counterpart (per the manifest's before_fields).
            for c in before_cols:
                gold[f"abstraction_flagged_{c}"] = bool(abstraction_flags.get(c, False))

            base.update(gold)
            base["annot_decision"] = decision
            base["annot_notes"] = ann_norm.get("notes") or None
            base["annotator"] = annotator
            base["annotated_at_utc"] = annotated_at
            base["_annotation_source"] = "model_extraction"
            base["chunk_notes"] = chunk_notes
            rows.append(base)

        # Added norms — synthesize a new row with model fields null
        for i, added in enumerate(added_anns):
            base: dict[str, Any] = {col: None for col in extraction.columns}
            base["gutenberg_id"] = gb_id
            base["chunk_id"] = ch_id
            base["article_text"] = chunk.get("article_text")
            base["chunk_size"] = chunk.get("chunk_size")
            base["book_title"] = chunk.get("book_title")
            base["book_author"] = chunk.get("book_author")
            for c in editable_cols:
                base[f"gold_{c}"] = added.get(c)
            base["annot_decision"] = "add"
            base["annot_notes"] = added.get("notes") or None
            base["annotator"] = annotator
            base["annotated_at_utc"] = annotated_at
            base["_annotation_source"] = "annotator_added"
            base["_added_index"] = i
            base["chunk_notes"] = chunk_notes
            rows.append(base)

    if not rows:
        print("[merge] no rows to write — empty annotations?", file=sys.stderr)
        return 1

    out_df = pd.DataFrame(rows)

    # Stable column ordering: model fields first, then gold_*, then annot_*
    front = chunk_meta_cols + ["norm_index"] + [c for c in editable_cols if c in out_df.columns]
    rest = [c for c in out_df.columns if c not in front]
    out_df = out_df[[c for c in front if c in out_df.columns] + rest]

    out_path = Path(args.output).resolve()
    if not out_path.parent.exists():
        print(f"[error] parent dir does not exist: {out_path.parent}", file=sys.stderr)
        return 2
    out_df.to_parquet(out_path, index=False)

    # Print a tidy summary
    by_dec = out_df["annot_decision"].fillna("(unannotated)").value_counts()
    print(f"[merge] wrote {out_path}")
    print(f"[merge] rows: {len(out_df)}")
    print("[merge] decision counts:")
    for k, v in by_dec.items():
        print(f"           {k:>14}  {v}")
    return 0


# ── HTML BUILDER ─────────────────────────────────────────────────────────

def _build_html(manifest: dict) -> str:
    # JSON does not escape </script> or <!-- — book text or annotator-supplied
    # strings containing those sequences would terminate the inline <script>
    # block and silently break the page. Neutralize them inside the JSON literal.
    payload = (
        json.dumps(manifest)
        .replace("</", "<\\/")
        .replace("<!--", "<\\!--")
        .replace("-->", "--\\>")
    )
    return _HTML_TEMPLATE.replace("__MANIFEST_PLACEHOLDER__", payload)


_HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Annotator</title>
<style>
:root {
  --bg:#fafafa; --fg:#1a1a1a; --border:#e0e0e0; --accent:#6a1b9a;
  --accent-light:#f3e5f5;
  --green:#2e7d32; --green-bg:#e8f5e9;
  --amber:#f9a825; --amber-bg:#fff8e1;
  --red:#c62828; --red-bg:#ffebee;
  --blue:#1565c0; --blue-bg:#e3f2fd;
  --grey:#757575; --grey-bg:#f0f0f0;
  --mono:'SF Mono','Cascadia Code','Fira Code',Consolas,monospace;
  --sans:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;
}
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:var(--sans);background:var(--bg);color:var(--fg);font-size:14px;padding-bottom:60px}
.topbar{
  position:sticky;top:0;z-index:100;background:#fff;border-bottom:1px solid var(--border);
  padding:8px 16px;display:flex;flex-wrap:wrap;gap:8px;align-items:center;
  box-shadow:0 1px 3px rgba(0,0,0,0.06);
}
.topbar button{
  cursor:pointer;background:var(--accent);color:#fff;border:none;padding:5px 12px;
  border-radius:4px;font-weight:500;font-size:13px;
}
.topbar button.secondary{background:var(--grey)}
.topbar button:hover{opacity:.9}
.topbar input[type=text]{font-size:13px;padding:4px 8px;border:1px solid var(--border);border-radius:4px;width:180px}
.topbar .sep{width:1px;height:24px;background:var(--border)}
.topbar .info{font-size:12px;color:#555}
.topbar .kbd{display:inline-block;background:#eee;border:1px solid #ccc;border-radius:3px;padding:1px 5px;font-size:11px;font-family:var(--mono);color:#555}
.topbar .save-state{font-size:12px;color:var(--green);padding:0 8px}
.topbar .save-state.dirty{color:var(--amber)}

.container{max-width:1100px;margin:0 auto;padding:14px 16px}

.chunk{
  background:#fff;border:1px solid var(--border);border-radius:8px;
  margin-bottom:18px;overflow:hidden;
}
.chunk-header{
  display:flex;align-items:center;gap:10px;padding:10px 14px;background:#f5f5f5;
  border-bottom:1px solid var(--border);cursor:pointer;user-select:none;
}
.chunk-header:hover{background:#eee}
.chunk-num{font-weight:700;color:var(--accent);min-width:60px;font-size:13px}
.chunk-ids{flex:1;font-size:13px}
.chunk-ids b{color:#333}
.chunk-ids .book{color:#444;margin-left:8px;font-style:italic}
.chunk-badges{display:flex;gap:6px}
.badge{font-size:11px;padding:2px 8px;border-radius:10px;font-weight:600;line-height:1.4}
.badge.has-norms{background:var(--blue-bg);color:var(--blue)}
.badge.no-norms{background:var(--grey-bg);color:var(--grey)}
.badge.progress{background:var(--green-bg);color:var(--green)}
.badge.progress.partial{background:var(--amber-bg);color:var(--amber)}
.badge.progress.empty{background:var(--grey-bg);color:var(--grey)}

.chunk-body{display:none;padding:14px}
.chunk.expanded .chunk-body{display:block}

.source-text{
  background:#f8f9fa;border:1px solid #e9ecef;border-radius:6px;padding:10px 14px;
  margin-bottom:14px;font-size:13px;
}
.source-text summary{cursor:pointer;font-weight:600;color:#555;font-size:12px}
.source-text pre{white-space:pre-wrap;word-break:break-word;margin:8px 0 0;font-family:var(--mono);font-size:12px;max-height:340px;overflow-y:auto}

.norm-card{
  border:1px solid var(--border);border-radius:6px;margin-bottom:12px;
}
.norm-card.confirm{border-left:4px solid var(--green)}
.norm-card.modify{border-left:4px solid var(--amber)}
.norm-card.reject{border-left:4px solid var(--red)}
.norm-card.uncertain{border-left:4px solid var(--grey)}
.norm-card.added{border-left:4px solid var(--blue);background:#f7faff}

.norm-header{
  display:flex;align-items:center;gap:10px;padding:8px 12px;background:#fafafa;
  border-bottom:1px solid var(--border);
}
.norm-header .label{font-weight:600;font-size:12px;color:#555;min-width:120px}
.decision-radios{display:flex;gap:10px;flex:1;flex-wrap:wrap}
.decision-radios label{display:inline-flex;align-items:center;gap:4px;font-size:12px;cursor:pointer;padding:2px 8px;border-radius:4px}
.decision-radios label:has(input:checked).d-confirm{background:var(--green-bg);color:var(--green);font-weight:600}
.decision-radios label:has(input:checked).d-modify{background:var(--amber-bg);color:var(--amber);font-weight:600}
.decision-radios label:has(input:checked).d-reject{background:var(--red-bg);color:var(--red);font-weight:600}
.decision-radios label:has(input:checked).d-uncertain{background:var(--grey-bg);color:#555;font-weight:600}
.btn-remove{cursor:pointer;background:transparent;border:1px solid var(--border);color:var(--red);padding:2px 8px;border-radius:4px;font-size:11px}
.btn-remove:hover{background:var(--red-bg)}

.norm-body{padding:8px 12px}
.field-row{display:flex;margin-bottom:6px;gap:8px}
.field-row label{width:140px;min-width:140px;font-weight:500;color:#555;font-size:12px;padding-top:4px}
.field-row .field-input{flex:1}
.field-input input[type=text],.field-input textarea{
  width:100%;font-family:var(--mono);font-size:12px;padding:4px 6px;
  border:1px solid var(--border);border-radius:4px;background:#fff;
}
.field-input textarea{min-height:50px;resize:vertical;font-family:var(--mono)}
.field-input.changed input,.field-input.changed textarea{border-color:var(--amber);background:#fffbeb}
.field-input .orig{font-size:11px;color:#888;font-family:var(--mono);margin-top:2px;padding:2px 6px;background:#f5f5f5;border-radius:3px;display:none}
.field-input.changed .orig{display:block}
.field-input .before{font-size:11px;color:var(--blue);font-family:var(--mono);margin-top:2px;padding:2px 6px;background:var(--blue-bg);border-radius:3px;white-space:pre-wrap;display:flex;flex-direction:column;gap:4px}
.field-input .before .before-label{font-weight:600;text-transform:uppercase;font-size:10px;letter-spacing:0.5px;margin-right:6px;color:var(--blue)}
.field-input .before-toggle{display:inline-flex;align-items:center;gap:5px;font-size:11px;color:#555;cursor:pointer;font-family:var(--sans);user-select:none;align-self:flex-start;padding:2px 6px;border-radius:3px;background:transparent;border:1px solid transparent}
.field-input .before-toggle:hover{background:#fff}
.field-input .before-toggle input{margin:0}
.field-input.flagged-bad{background:var(--red-bg);border-radius:4px;padding:4px;margin:-4px}
.field-input.flagged-bad input[type=text],.field-input.flagged-bad textarea{border-color:var(--red);background:#fff5f5}
.field-input.flagged-bad .before{background:#fff;border:1px solid var(--red);color:var(--red)}
.field-input.flagged-bad .before-label{color:var(--red)}
.field-input.flagged-bad .before-toggle{color:var(--red);font-weight:600;background:var(--red-bg)}
.field-input.readonly input[disabled],.field-input.readonly textarea[disabled]{background:#f5f5f5;color:#777;cursor:not-allowed}
.field-input.readonly textarea[disabled]{font-family:var(--sans);font-size:12px;width:100%;padding:4px 6px;border:1px solid var(--border);border-radius:4px;min-height:60px;resize:vertical}

.notes-row textarea{width:100%;font-family:var(--sans);font-size:12px;padding:4px 6px;border:1px solid var(--border);border-radius:4px;min-height:36px;resize:vertical;background:#fffdf2}
.notes-row label{font-size:11px;color:#888;font-weight:500}

.section-divider{margin:14px 0 8px;padding:6px 10px;font-size:12px;font-weight:600;color:#555;text-transform:uppercase;letter-spacing:0.5px;border-top:1px solid var(--border)}
.add-btn{background:var(--blue);color:#fff;border:none;padding:6px 12px;border-radius:4px;cursor:pointer;font-size:12px;font-weight:500}
.add-btn:hover{opacity:.9}

.chunk-notes-row{margin-top:14px}
.chunk-notes-row label{font-size:12px;font-weight:600;color:#555;display:block;margin-bottom:4px}
.chunk-notes-row textarea{width:100%;min-height:44px;font-family:var(--sans);font-size:12px;padding:6px 8px;border:1px solid var(--border);border-radius:4px;background:#fffdf2}

.empty-note{color:#888;font-size:13px;font-style:italic;padding:8px 0}
</style>
</head>
<body>

<div class="topbar">
  <button onclick="prevChunk()" class="secondary">◀</button>
  <span class="info" id="progress-info">…</span>
  <button onclick="nextChunk()" class="secondary">▶</button>
  <div class="sep"></div>
  <input type="text" id="annotator-input" placeholder="annotator id">
  <div class="sep"></div>
  <button onclick="downloadAnnotations()">⬇ Download annotations</button>
  <input type="file" id="upload-input" accept=".json" style="display:none" onchange="uploadAnnotations(event)">
  <button onclick="document.getElementById('upload-input').click()" class="secondary">⬆ Load annotations</button>
  <div class="sep"></div>
  <span class="save-state" id="save-state">saved</span>
  <div class="info">
    <span class="kbd">j</span>/<span class="kbd">k</span> next/prev chunk
  </div>
</div>

<div class="container" id="container"></div>

<script>
const MANIFEST = __MANIFEST_PLACEHOLDER__;
const STORAGE_KEY = "norms_annotator:" + MANIFEST.manifest_hash;
const EDITABLE_FIELDS = MANIFEST.editable_fields;  // [[col,label,widget], ...]
const READONLY_FIELDS = MANIFEST.readonly_fields;  // [[col,label,widget?], ...]
const BEFORE_FIELDS = MANIFEST.before_fields || {};  // {raz_col: orig_col}
const ITEM_LABEL_SINGULAR = MANIFEST.item_label_singular || "Norm";
const ITEM_LABEL_PLURAL = MANIFEST.item_label_plural || "norms";
const GATE_COLUMN = MANIFEST.gate_column || "has_norms";
document.title = ITEM_LABEL_SINGULAR + "s Annotator";

let STATE = {
  annotator: "",
  chunks: {},   // key -> {model_norms: {norm_idx: {decision,modified_fields,notes}}, added_norms: [], chunk_notes:""}
};

let currentChunkIdx = 0;
let saveTimer = null;
let dirty = false;

// ── State persistence ─────────────────────────────────────────────────
function loadFromStorage() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const s = JSON.parse(raw);
      if (s && s.chunks) STATE = s;
    }
  } catch (e) { console.warn("loadFromStorage failed", e); }
  if (!STATE.annotator) STATE.annotator = "";
  if (!STATE.chunks) STATE.chunks = {};
  document.getElementById("annotator-input").value = STATE.annotator || "";
}

function markDirty() {
  dirty = true;
  const el = document.getElementById("save-state");
  el.textContent = "saving…";
  el.classList.add("dirty");
  if (saveTimer) clearTimeout(saveTimer);
  saveTimer = setTimeout(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(STATE));
      el.textContent = "saved ✓";
      el.classList.remove("dirty");
      dirty = false;
      updateProgress();
    } catch (e) {
      el.textContent = "save failed!";
      console.error(e);
    }
  }, 350);
}

window.addEventListener("beforeunload", (e) => {
  if (dirty) {
    e.preventDefault();
    return (e.returnValue = "Unsaved annotations — download before leaving?");
  }
});

// ── Per-chunk state accessors ─────────────────────────────────────────
function chunkState(key) {
  if (!STATE.chunks[key]) {
    STATE.chunks[key] = {model_norms: {}, added_norms: [], chunk_notes: ""};
  }
  return STATE.chunks[key];
}

// ── Render ────────────────────────────────────────────────────────────
function render() {
  const ct = document.getElementById("container");
  ct.innerHTML = "";
  MANIFEST.chunks.forEach((chunk, i) => {
    ct.appendChild(buildChunkCard(chunk, i));
  });
  expandOnly(currentChunkIdx);
  updateProgress();
}

function buildChunkCard(chunk, idx) {
  const cs = chunkState(chunk.key);
  const wrap = document.createElement("div");
  wrap.className = "chunk";
  wrap.dataset.chunkKey = chunk.key;
  wrap.dataset.chunkIdx = idx;

  // ── Header ────────────────────────────────────────────────
  const hdr = document.createElement("div");
  hdr.className = "chunk-header";
  hdr.addEventListener("click", () => {
    if (idx === currentChunkIdx) return;
    currentChunkIdx = idx;
    expandOnly(idx);
  });

  const num = document.createElement("span");
  num.className = "chunk-num";
  num.textContent = `${idx+1}/${MANIFEST.chunks.length}`;
  hdr.appendChild(num);

  const ids = document.createElement("span");
  ids.className = "chunk-ids";
  ids.innerHTML = `<b>${esc(chunk.key)}</b> <span class="book">${esc(chunk.book_title || "?")} — ${esc(chunk.book_author || "?")}</span>`;
  hdr.appendChild(ids);

  const badges = document.createElement("span");
  badges.className = "chunk-badges";
  const gateOn = `${GATE_COLUMN}=true`;
  const gateOff = `${GATE_COLUMN}=false`;
  badges.innerHTML = `
    <span class="badge ${chunk.has_norms ? "has-norms" : "no-norms"}">
      ${chunk.has_norms ? gateOn : gateOff} · ${chunk.model_norms.length} model
    </span>
    <span class="badge progress" data-progress-key="${esc(chunk.key)}">…</span>
  `;
  hdr.appendChild(badges);
  wrap.appendChild(hdr);

  // ── Body ──────────────────────────────────────────────────
  const body = document.createElement("div");
  body.className = "chunk-body";

  // Source text (collapsible)
  const txt = document.createElement("details");
  txt.className = "source-text";
  txt.open = false;
  const srcText = chunk.article_text || "";
  txt.innerHTML = `<summary>Source text (${srcText.length.toLocaleString()} chars)</summary><pre>${esc(srcText)}</pre>`;
  body.appendChild(txt);

  // Model norms section
  if (chunk.model_norms.length === 0) {
    const p = document.createElement("p");
    p.className = "empty-note";
    p.textContent = `Model extracted no ${ITEM_LABEL_PLURAL} from this chunk. Use 'Add missed ${ITEM_LABEL_SINGULAR.toLowerCase()}' below if you find any.`;
    body.appendChild(p);
  } else {
    const divider = document.createElement("div");
    divider.className = "section-divider";
    divider.textContent = `Model-extracted ${ITEM_LABEL_PLURAL} (${chunk.model_norms.length})`;
    body.appendChild(divider);
    chunk.model_norms.forEach(mn => {
      body.appendChild(buildModelNormCard(chunk.key, mn));
    });
  }

  // Added norms section
  const addedDivider = document.createElement("div");
  addedDivider.className = "section-divider";
  addedDivider.innerHTML = `Annotator-added ${ITEM_LABEL_PLURAL} <button class="add-btn" data-add-for="${esc(chunk.key)}">+ Add missed ${ITEM_LABEL_SINGULAR.toLowerCase()}</button>`;
  body.appendChild(addedDivider);

  const addedContainer = document.createElement("div");
  addedContainer.dataset.addedContainer = chunk.key;
  body.appendChild(addedContainer);

  // Render any already-added norms from state
  cs.added_norms.forEach((_, ai) => {
    addedContainer.appendChild(buildAddedNormCard(chunk.key, ai));
  });

  addedDivider.querySelector(".add-btn").addEventListener("click", (e) => {
    e.stopPropagation();
    cs.added_norms.push({});
    const newIdx = cs.added_norms.length - 1;
    addedContainer.appendChild(buildAddedNormCard(chunk.key, newIdx));
    markDirty();
  });

  // Chunk-level notes
  const cnRow = document.createElement("div");
  cnRow.className = "chunk-notes-row";
  cnRow.innerHTML = `<label>Chunk-level notes (whole-chunk observations)</label><textarea data-chunk-notes="${esc(chunk.key)}"></textarea>`;
  body.appendChild(cnRow);
  const cnTa = cnRow.querySelector("textarea");
  cnTa.value = cs.chunk_notes || "";
  cnTa.addEventListener("input", () => {
    cs.chunk_notes = cnTa.value;
    markDirty();
  });

  wrap.appendChild(body);
  return wrap;
}

function buildModelNormCard(chunkKey, mn) {
  const cs = chunkState(chunkKey);
  const ni = mn.norm_index !== undefined && mn.norm_index !== null ? String(mn.norm_index) : "";
  if (!cs.model_norms[ni]) cs.model_norms[ni] = {decision: null, modified_fields: {}, notes: ""};
  const annot = cs.model_norms[ni];

  const card = document.createElement("div");
  card.className = "norm-card";
  card.dataset.chunkKey = chunkKey;
  card.dataset.normIdx = ni;
  applyDecisionClass(card, annot.decision);

  // Header with decision radios
  const hdr = document.createElement("div");
  hdr.className = "norm-header";
  hdr.innerHTML = `<span class="label">${ITEM_LABEL_SINGULAR} #${esc(ni)}</span>`;
  const rdiv = document.createElement("div");
  rdiv.className = "decision-radios";
  ["confirm","modify","reject","uncertain"].forEach(opt => {
    const id = `dec-${chunkKey}-${ni}-${opt}`;
    const lbl = document.createElement("label");
    lbl.className = `d-${opt}`;
    lbl.innerHTML = `<input type="radio" name="dec-${chunkKey}-${ni}" value="${opt}" id="${id}"><span>${opt}</span>`;
    const inp = lbl.querySelector("input");
    if (annot.decision === opt) inp.checked = true;
    inp.addEventListener("change", () => {
      annot.decision = opt;
      applyDecisionClass(card, opt);
      markDirty();
    });
    rdiv.appendChild(lbl);
  });
  hdr.appendChild(rdiv);
  card.appendChild(hdr);

  // Body: editable fields
  const body = document.createElement("div");
  body.className = "norm-body";

  EDITABLE_FIELDS.forEach(([col, label, widget]) => {
    const row = document.createElement("div");
    row.className = "field-row";
    const lbl = document.createElement("label");
    lbl.textContent = label;
    row.appendChild(lbl);

    const wrap = document.createElement("div");
    wrap.className = "field-input";
    const orig = mn[col];
    const current = annot.modified_fields && col in annot.modified_fields ? annot.modified_fields[col] : orig;
    let ctl;
    if (widget === "checkbox") {
      ctl = document.createElement("input");
      ctl.type = "checkbox";
      ctl.checked = !!current;
    } else if (widget === "textarea") {
      ctl = document.createElement("textarea");
      ctl.value = current ?? "";
    } else {
      ctl = document.createElement("input");
      ctl.type = "text";
      ctl.value = current ?? "";
    }
    wrap.appendChild(ctl);

    const origDisp = document.createElement("div");
    origDisp.className = "orig";
    origDisp.textContent = "original: " + formatVal(orig);
    wrap.appendChild(origDisp);

    // Pre-role-abstraction value (when one exists and differs from the abstracted form).
    // Also surface a "flag this abstraction as bad" toggle here, with corrections
    // captured via the same modify path as everything else.
    const beforeCol = BEFORE_FIELDS[col];
    if (beforeCol && mn[beforeCol] !== undefined && mn[beforeCol] !== null) {
      const beforeVal = mn[beforeCol];
      if (formatVal(beforeVal) !== formatVal(orig)) {
        if (!annot.abstraction_flags) annot.abstraction_flags = {};
        const isFlagged = !!annot.abstraction_flags[col];
        if (isFlagged) wrap.classList.add("flagged-bad");

        const beforeDisp = document.createElement("div");
        beforeDisp.className = "before";
        const valLine = document.createElement("div");
        valLine.innerHTML = `<span class="before-label">pre-abstraction</span>${esc(formatVal(beforeVal))}`;
        beforeDisp.appendChild(valLine);

        const toggle = document.createElement("label");
        toggle.className = "before-toggle";
        toggle.title = "Mark this role abstraction as unfaithful / lossy. " +
                       "Use the field above to type a corrected abstraction (optional).";
        toggle.innerHTML = `<input type="checkbox" ${isFlagged ? "checked" : ""}> ` +
                          `🚩 flag this abstraction as bad`;
        const cb = toggle.querySelector("input");
        cb.addEventListener("change", () => {
          if (!annot.abstraction_flags) annot.abstraction_flags = {};
          annot.abstraction_flags[col] = cb.checked;
          wrap.classList.toggle("flagged-bad", cb.checked);
          markDirty();
        });
        beforeDisp.appendChild(toggle);
        wrap.appendChild(beforeDisp);
      }
    }

    const updateChangedFlag = () => {
      const newVal = widget === "checkbox" ? ctl.checked : ctl.value;
      const changed = !valuesEqual(newVal, orig, widget);
      wrap.classList.toggle("changed", changed);
      if (changed) {
        if (!annot.modified_fields) annot.modified_fields = {};
        annot.modified_fields[col] = newVal;
        // auto-flip to 'modify' unless explicitly reject/uncertain
        if (annot.decision !== "reject" && annot.decision !== "uncertain") {
          annot.decision = "modify";
          const r = card.querySelector(`input[type=radio][value=modify]`);
          if (r) r.checked = true;
          applyDecisionClass(card, "modify");
        }
      } else {
        if (annot.modified_fields) delete annot.modified_fields[col];
        // if no modifications remain and decision was 'modify', revert to confirm
        if ((!annot.modified_fields || Object.keys(annot.modified_fields).length === 0)
            && annot.decision === "modify") {
          annot.decision = "confirm";
          const r = card.querySelector(`input[type=radio][value=confirm]`);
          if (r) r.checked = true;
          applyDecisionClass(card, "confirm");
        }
      }
      markDirty();
    };
    ctl.addEventListener("input", updateChangedFlag);
    if (widget === "checkbox") ctl.addEventListener("change", updateChangedFlag);
    // Set initial 'changed' flag if state already has a modification
    if (annot.modified_fields && col in annot.modified_fields) wrap.classList.add("changed");

    row.appendChild(wrap);
    body.appendChild(row);
  });

  // Read-only context fields
  READONLY_FIELDS.forEach((tuple) => {
    const col = tuple[0], label = tuple[1], widget = tuple[2] || "input";
    const v = mn[col];
    if (v === undefined || v === null || v === "") return;
    const row = document.createElement("div");
    row.className = "field-row";
    const ctrl = widget === "textarea"
      ? `<textarea disabled>${esc(formatVal(v))}</textarea>`
      : `<input type="text" disabled value="${esc(formatVal(v))}">`;
    row.innerHTML = `<label>${esc(label)}</label>
      <div class="field-input readonly">${ctrl}</div>`;
    body.appendChild(row);
  });

  // Notes
  const notes = document.createElement("div");
  notes.className = "field-row notes-row";
  notes.innerHTML = `<label>Notes</label><div class="field-input"><textarea placeholder="Why confirm/modify/reject? Edge case?"></textarea></div>`;
  const ta = notes.querySelector("textarea");
  ta.value = annot.notes || "";
  ta.addEventListener("input", () => { annot.notes = ta.value; markDirty(); });
  body.appendChild(notes);

  card.appendChild(body);
  return card;
}

function buildAddedNormCard(chunkKey, addedIdx) {
  const cs = chunkState(chunkKey);
  const ann = cs.added_norms[addedIdx];

  const card = document.createElement("div");
  card.className = "norm-card added";
  card.dataset.addedIdx = addedIdx;

  const hdr = document.createElement("div");
  hdr.className = "norm-header";
  hdr.innerHTML = `<span class="label">Added ${ITEM_LABEL_SINGULAR.toLowerCase()} #${addedIdx+1}</span>
    <div style="flex:1;font-size:12px;color:#555">(annotator-added — model missed this)</div>`;
  const rm = document.createElement("button");
  rm.className = "btn-remove";
  rm.textContent = "Remove";
  rm.addEventListener("click", (e) => {
    e.stopPropagation();
    cs.added_norms.splice(addedIdx, 1);
    // Re-render this chunk's added container
    const container = document.querySelector(`[data-added-container="${cssEsc(chunkKey)}"]`);
    if (container) {
      container.innerHTML = "";
      cs.added_norms.forEach((_, ai) => container.appendChild(buildAddedNormCard(chunkKey, ai)));
    }
    markDirty();
  });
  hdr.appendChild(rm);
  card.appendChild(hdr);

  const body = document.createElement("div");
  body.className = "norm-body";
  EDITABLE_FIELDS.forEach(([col, label, widget]) => {
    const row = document.createElement("div");
    row.className = "field-row";
    row.innerHTML = `<label>${esc(label)}</label>`;
    const wrap = document.createElement("div");
    wrap.className = "field-input";
    let ctl;
    if (widget === "checkbox") {
      ctl = document.createElement("input");
      ctl.type = "checkbox";
      ctl.checked = !!ann[col];
    } else if (widget === "textarea") {
      ctl = document.createElement("textarea");
      ctl.value = ann[col] || "";
    } else {
      ctl = document.createElement("input");
      ctl.type = "text";
      ctl.value = ann[col] || "";
    }
    ctl.addEventListener("input", () => {
      ann[col] = widget === "checkbox" ? ctl.checked : ctl.value;
      markDirty();
    });
    if (widget === "checkbox") ctl.addEventListener("change", () => {
      ann[col] = ctl.checked; markDirty();
    });
    wrap.appendChild(ctl);
    row.appendChild(wrap);
    body.appendChild(row);
  });

  // Notes
  const notes = document.createElement("div");
  notes.className = "field-row notes-row";
  notes.innerHTML = `<label>Notes</label><div class="field-input"><textarea placeholder="Why did the model miss this?"></textarea></div>`;
  const ta = notes.querySelector("textarea");
  ta.value = ann.notes || "";
  ta.addEventListener("input", () => { ann.notes = ta.value; markDirty(); });
  body.appendChild(notes);

  card.appendChild(body);
  return card;
}

function applyDecisionClass(card, decision) {
  card.classList.remove("confirm","modify","reject","uncertain");
  if (decision) card.classList.add(decision);
}

// ── Navigation ────────────────────────────────────────────────────────
function expandOnly(idx) {
  document.querySelectorAll(".chunk").forEach((el, i) => {
    el.classList.toggle("expanded", i === idx);
  });
  const el = document.querySelector(`.chunk[data-chunk-idx="${idx}"]`);
  if (el) {
    const topbar = document.querySelector(".topbar");
    const offset = (topbar ? topbar.getBoundingClientRect().height : 50) + 8;
    const rect = el.getBoundingClientRect();
    window.scrollTo({top: window.scrollY + rect.top - offset, behavior: "smooth"});
  }
}
function nextChunk() {
  if (currentChunkIdx < MANIFEST.chunks.length - 1) {
    currentChunkIdx++;
    expandOnly(currentChunkIdx);
  }
}
function prevChunk() {
  if (currentChunkIdx > 0) {
    currentChunkIdx--;
    expandOnly(currentChunkIdx);
  }
}
document.addEventListener("keydown", (e) => {
  if (["INPUT","TEXTAREA"].includes(document.activeElement.tagName)) return;
  if (e.key === "j") nextChunk();
  else if (e.key === "k") prevChunk();
});

// ── Progress ──────────────────────────────────────────────────────────
function updateProgress() {
  let done = 0;
  MANIFEST.chunks.forEach(chunk => {
    const cs = chunkState(chunk.key);
    const totalModel = chunk.model_norms.length;
    const decided = Object.values(cs.model_norms || {}).filter(a => a.decision).length;
    const fullyDone = totalModel === 0
      ? (cs.added_norms.length > 0 || (cs.chunk_notes && cs.chunk_notes.length > 0))
      : (decided === totalModel);
    if (fullyDone) done++;
    // Per-chunk badge
    const badge = document.querySelector(`[data-progress-key="${cssEsc(chunk.key)}"]`);
    if (badge) {
      badge.classList.remove("partial","empty");
      if (totalModel === 0) {
        if (cs.added_norms.length > 0) {
          badge.textContent = `+${cs.added_norms.length} added`;
          badge.classList.remove("empty");
        } else {
          badge.textContent = "no annotations";
          badge.classList.add("empty");
        }
      } else if (decided === totalModel) {
        badge.textContent = `${decided}/${totalModel} ✓`;
      } else if (decided > 0) {
        badge.textContent = `${decided}/${totalModel}`;
        badge.classList.add("partial");
      } else {
        badge.textContent = `0/${totalModel}`;
        badge.classList.add("empty");
      }
    }
  });
  const pi = document.getElementById("progress-info");
  pi.textContent = `${done} / ${MANIFEST.chunks.length} chunks fully annotated`;
}

// ── Download / upload ─────────────────────────────────────────────────
function downloadAnnotations() {
  STATE.annotator = document.getElementById("annotator-input").value || null;
  const payload = {
    schema_version: MANIFEST.schema_version,
    sample_manifest_hash: MANIFEST.manifest_hash,
    annotator: STATE.annotator,
    annotated_at_utc: new Date().toISOString(),
    n_chunks: MANIFEST.chunks.length,
    chunks: prepareForDownload(STATE.chunks),
  };
  const blob = new Blob([JSON.stringify(payload, null, 2)], {type:"application/json"});
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  const ts = new Date().toISOString().replace(/[:.]/g,"-").slice(0,19);
  a.download = `norms_annotations_${ts}.json`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
}

// Strip stale modified_fields entries that got reverted, and drop empty chunks
function prepareForDownload(chunks) {
  const out = {};
  for (const [key, cs] of Object.entries(chunks)) {
    const mn = {};
    for (const [ni, a] of Object.entries(cs.model_norms || {})) {
      const flagsActive = a.abstraction_flags
        ? Object.values(a.abstraction_flags).some(Boolean)
        : false;
      if (!a.decision
          && !(a.notes && a.notes.length)
          && !(a.modified_fields && Object.keys(a.modified_fields).length)
          && !flagsActive) continue;
      const rec = {decision: a.decision || null};
      if (a.modified_fields && Object.keys(a.modified_fields).length) rec.modified_fields = a.modified_fields;
      if (a.notes) rec.notes = a.notes;
      if (flagsActive) {
        // Strip false entries so the JSON only carries the active flags.
        const flags = {};
        for (const [c, v] of Object.entries(a.abstraction_flags)) if (v) flags[c] = true;
        rec.abstraction_flags = flags;
      }
      mn[ni] = rec;
    }
    const added = (cs.added_norms || []).filter(x => Object.keys(x).some(k => x[k] != null && x[k] !== ""));
    const cn = cs.chunk_notes || "";
    if (Object.keys(mn).length === 0 && added.length === 0 && !cn) continue;
    out[key] = {model_norms: mn, added_norms: added, chunk_notes: cn};
  }
  return out;
}

function uploadAnnotations(evt) {
  const file = evt.target.files[0];
  if (!file) return;
  const reader = new FileReader();
  reader.onload = (e) => {
    try {
      const data = JSON.parse(e.target.result);
      if (data.sample_manifest_hash && data.sample_manifest_hash !== MANIFEST.manifest_hash) {
        if (!confirm("Annotation hash does not match this sample. Load anyway?")) return;
      }
      // Map into STATE.chunks
      STATE.annotator = data.annotator || STATE.annotator || "";
      document.getElementById("annotator-input").value = STATE.annotator;
      STATE.chunks = {};
      for (const [key, cs] of Object.entries(data.chunks || {})) {
        STATE.chunks[key] = {
          model_norms: cs.model_norms || {},
          added_norms: cs.added_norms || [],
          chunk_notes: cs.chunk_notes || "",
        };
      }
      // Backfill in-memory shape (each model_norm gets {decision,modified_fields,notes,abstraction_flags})
      for (const cs of Object.values(STATE.chunks)) {
        for (const [ni, a] of Object.entries(cs.model_norms)) {
          if (!a.modified_fields) a.modified_fields = {};
          if (!("notes" in a)) a.notes = "";
          if (!a.abstraction_flags) a.abstraction_flags = {};
        }
      }
      markDirty();
      render();
      alert("Loaded " + Object.keys(STATE.chunks).length + " chunk annotations.");
    } catch (err) {
      alert("Failed to parse JSON: " + err.message);
    }
  };
  reader.readAsText(file);
  evt.target.value = "";  // allow re-uploading same file
}

// ── Annotator id ──────────────────────────────────────────────────────
document.getElementById("annotator-input").addEventListener("input", (e) => {
  STATE.annotator = e.target.value;
  markDirty();
});

// ── Helpers ───────────────────────────────────────────────────────────
function esc(s) {
  if (s == null) return "";
  const d = document.createElement("div");
  d.textContent = String(s);
  return d.innerHTML;
}
function cssEsc(s) { return String(s).replace(/"/g, '\\"'); }
function formatVal(v) {
  if (v === null || v === undefined) return "—";
  if (typeof v === "boolean") return v ? "true" : "false";
  return String(v);
}
function valuesEqual(a, b, widget) {
  if (widget === "checkbox") return !!a === !!b;
  const sa = a == null ? "" : String(a);
  const sb = b == null ? "" : String(b);
  return sa === sb;
}

// ── Boot ──────────────────────────────────────────────────────────────
loadFromStorage();
render();
</script>
</body>
</html>
"""


# ── ENTRY ────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="norms_annotator", description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_export = sub.add_parser("export", help="build annotation sample HTML + manifest")
    p_export.add_argument("--reasoning", required=True, help="path to reasoning.parquet (chunk universe)")
    p_export.add_argument("--extraction", required=True, help="path to structured_norms / ci_flows parquet")
    p_export.add_argument("-n", type=int, default=None,
                          help="total number of chunks to sample (uniform). Mutex with --per-book.")
    p_export.add_argument("--per-book", dest="per_book", type=int, default=None,
                          help="chunks per gutenberg_id (stratified). Mutex with -n.")
    p_export.add_argument("--seed", type=int, default=42, help="rng seed (default 42)")
    p_export.add_argument("--books", default=None, help="comma-separated gutenberg_ids to restrict to")
    p_export.add_argument("--schema", choices=sorted(SCHEMAS), default="norms",
                          help=f"annotation schema (default: norms; choices: {sorted(SCHEMAS)})")
    p_export.add_argument("-o", "--output", required=True, help="output HTML path (manifest written beside it)")
    p_export.set_defaults(func=cmd_export)

    p_merge = sub.add_parser("merge", help="merge annotations JSON → gold_labels parquet")
    p_merge.add_argument("--manifest", required=True, help="manifest JSON written by export")
    p_merge.add_argument("--annotations", required=True, help="annotations JSON downloaded from the HTML tool")
    p_merge.add_argument("-o", "--output", required=True, help="output gold_labels parquet path")
    p_merge.set_defaults(func=cmd_merge)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
