# Norm extraction stage using the shared direct vLLM helper.

import re
import pandas as pd
import json
from omegaconf import OmegaConf
from typing import Any, Dict, List

from dagspaces.common.vllm_inference import run_vllm_inference
from ..ci_schema import PrescriptiveNormExtractionResult


# All raz_* columns that the extraction stage produces.  Used to guarantee
# every output row has the same schema regardless of parse success/failure.
_RAZ_COLUMNS = [
    "raz_prescriptive_element", "raz_norm_subject", "raz_norm_act",
    "raz_condition_of_application", "raz_normative_force",
    "raz_norm_articulation", "raz_norm_source", "raz_governs_info_flow",
    "raz_info_flow_note", "raz_confidence_qual", "raz_confidence_quant",
    "raz_context",
]

# ---------------------------------------------------------------------------
# Post-hoc norm quality validation
# ---------------------------------------------------------------------------

# Pattern: "Mr./Mrs./Miss/Lady/Sir/Lord/Colonel/Rev. + ProperNoun"
_TITLE_PATTERN = re.compile(
    r"\b(?:Mr\.|Mrs\.|Miss|Ms\.|Lady|Lord|Sir|Reverend|Rev\.|Colonel|Col\.)"
    r"\s+[A-Z][a-z]+",
)


def _get_character_blocklist(cfg: Any) -> set[str]:
    """Return lowercased character names to flag in norm fields.

    Checks ``cfg.norm_quality.character_blocklist`` (a list of strings).
    Falls back to a built-in set covering the 10-novel fiction corpus.
    """
    custom = None
    try:
        custom = OmegaConf.select(cfg, "norm_quality.character_blocklist")
    except Exception:
        pass
    if custom:
        return {n.lower() for n in custom}

    # Default blocklist — named characters and places across the 10-novel
    # fiction corpus.  Kept intentionally broad; false positives are cheap
    # (just a flag, no rows dropped).
    return {
        # Pride and Prejudice
        "elizabeth", "darcy", "bennet", "bingley", "wickham", "collins",
        "lydia", "jane", "jane bennet", "kitty", "mary", "mary bennet",
        "charlotte", "lucas", "gardiner", "fitzwilliam", "georgiana",
        "lady catherine", "de bourgh", "longbourn", "netherfield",
        "pemberley",
        # Middlemarch
        "dorothea", "casaubon", "lydgate", "rosamond", "bulstrode",
        "ladislaw", "brooke", "celia", "garth", "vincy", "farebrother",
        "caleb", "fred", "will", "sir james",
        # The Age of Innocence
        "newland", "archer", "ellen olenska", "olenska", "may welland",
        "welland", "mingott", "beaufort", "may",
        # Les Misérables
        "valjean", "jean valjean", "marius", "cosette", "javert",
        "fantine", "fauchelevent", "father fauvent", "gavroche",
        "thénardier", "thenardier", "enjolras", "madeleine",
        "gillenormand", "jondrette", "mabeuf", "montparnasse",
        "grantaire", "joly", "théodule", "éponine", "eponine",
        "leblanc",
        # Anna Karenina
        "levin", "anna", "vronsky", "karenin", "alexey alexandrovitch",
        "kitty", "dolly", "oblonsky", "stepan arkadyevitch",
        "seryozha", "sergey ivanovitch", "varenka", "kostya",
        "darya alexandrovna", "lidia ivanovna", "veslovsky",
        # Nineteen Eighty-Four
        "winston", "julia", "o'brien", "syme", "parsons",
        "ampleforth", "charrington", "goldstein", "big brother",
        # Alice's Adventures in Wonderland
        "alice", "mary ann", "pat", "dinah",
        # The Count of Monte Cristo
        "dantès", "dantes", "edmond", "edward", "monte cristo",
        "valentine", "caderousse", "villefort", "morrel",
        "maximilian", "albert", "franz", "mercédès", "mercedes",
        "danglars", "bertuccio", "andrea", "haydée", "haydee",
        "beauchamp", "fernand", "baptistin", "eugénie", "eugenie",
        "noirtier", "julie", "la carconte", "luigi vampa", "vampa",
        "barrois", "ali", "morcerf",
        # Bleak House
        "esther", "summerson", "richard", "george", "guppy",
        "snagsby", "vholes", "bucket", "woodcourt", "charley",
        "dedlock", "caddy", "skimpole", "jarndyce", "tulkinghorn",
        "bagnet", "rouncewell", "smallweed", "hortense", "ada",
        "jo", "nemo", "gridley", "flite", "rosa", "volumnia",
        "jellyby", "tom",
        # The Picture of Dorian Gray
        "dorian", "dorian gray", "basil", "hallward",
        "lord henry", "sibyl", "sibyl vane", "alan campbell",
        "alan", "harry",
    }


def _validate_norm_quality(flat: Dict[str, Any],
                           blocklist: set[str]) -> Dict[str, Any]:
    """Flag norms that contain named characters or plot-specific details.

    Adds two columns:
      - norm_quality_flags: semicolon-separated list of issues (or None)
      - norm_quality_passed: bool
    Does NOT drop any rows — downstream consumers decide filtering policy.
    """
    subject = (flat.get("raz_norm_subject") or "").lower()
    act = (flat.get("raz_norm_act") or "").lower()
    condition = (flat.get("raz_condition_of_application") or "").lower()
    articulation = (flat.get("raz_norm_articulation") or "").lower()

    flags: list[str] = []

    # Check for named characters in each Raz component.
    # Use word-boundary matching to avoid substring false positives
    # (e.g., "pearl" matching in "a pearl of great price").
    for name in blocklist:
        pattern = re.compile(r"\b" + re.escape(name) + r"\b")
        if pattern.search(subject):
            flags.append(f"named_char_in_subject:{name}")
        if pattern.search(act):
            flags.append(f"named_char_in_act:{name}")
        if pattern.search(condition):
            flags.append(f"named_char_in_condition:{name}")
        if pattern.search(articulation):
            flags.append(f"named_char_in_articulation:{name}")

    # Check for titled names (Mr./Mrs./Miss/etc. + ProperNoun)
    for field_name, field_val in [
        ("subject", flat.get("raz_norm_subject") or ""),
        ("act", flat.get("raz_norm_act") or ""),
        ("condition", flat.get("raz_condition_of_application") or ""),
        ("articulation", flat.get("raz_norm_articulation") or ""),
    ]:
        if _TITLE_PATTERN.search(field_val):
            flags.append(f"titled_name_in_{field_name}")

    # Deduplicate flags (a name might match both blocklist and title pattern)
    seen = set()
    unique_flags = []
    for f in flags:
        if f not in seen:
            seen.add(f)
            unique_flags.append(f)

    flat["norm_quality_flags"] = "; ".join(unique_flags) if unique_flags else None
    flat["norm_quality_passed"] = len(unique_flags) == 0
    return flat


def _clean_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Clean DataFrame to avoid PyArrow serialization issues.

    Removes or converts columns that cause parquet write errors:
    - Empty struct columns (e.g., 'metadata' with {})
    - Complex nested types that Arrow can't handle
    """
    # Columns that commonly cause issues - drop them
    problematic_cols = [
        "metadata", "reasoning_data", "raz_norms_raw",
        "__inference_error__", "embeddings"
    ]

    for col in problematic_cols:
        if col in df.columns:
            # Check if column contains empty dicts/structs
            try:
                sample = df[col].dropna().head(1)
                if len(sample) > 0:
                    val = sample.iloc[0]
                    # Drop if it's an empty dict or list of empty dicts
                    if val == {} or val == [] or (isinstance(val, list) and all(v == {} for v in val)):
                        df = df.drop(columns=[col])
                        print(f"[norm_extraction] Dropped empty struct column: {col}")
                        continue
            except Exception:
                pass

            # Convert complex objects to JSON strings for safe parquet storage
            try:
                df[col] = df[col].apply(lambda x: json.dumps(x) if isinstance(x, (dict, list)) else x)
            except Exception:
                df = df.drop(columns=[col])
                print(f"[norm_extraction] Dropped problematic column: {col}")

    return df

try:
    from json_repair import repair_json
    _JSON_REPAIR_OK = True
except ImportError:
    _JSON_REPAIR_OK = False


def _to_str(v):
    """Convert a value to a string suitable for a flat DataFrame column."""
    if v is None:
        return None
    if isinstance(v, list):
        return "; ".join(str(x) for x in v)
    return str(v)


def _flatten_norm(norm_entry: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a single PrescriptiveNormExtraction dict into raz_* columns."""
    flat: Dict[str, Any] = {}
    norm_tuple = norm_entry.get("norm", {})
    for k, v in norm_tuple.items():
        flat[f"raz_{k}"] = _to_str(v)
    flat["raz_normative_force"] = _to_str(norm_entry.get("normative_force"))
    flat["raz_norm_articulation"] = _to_str(norm_entry.get("norm_articulation"))
    flat["raz_norm_source"] = _to_str(norm_entry.get("norm_source"))
    flat["raz_governs_info_flow"] = norm_entry.get("governs_information_flow")
    flat["raz_info_flow_note"] = _to_str(norm_entry.get("information_flow_note"))
    flat["raz_confidence_qual"] = _to_str(norm_entry.get("confidence_qual"))
    flat["raz_confidence_quant"] = norm_entry.get("confidence_quant")
    flat["raz_context"] = _to_str(norm_entry.get("context"))
    return flat


def _null_raz_columns() -> Dict[str, Any]:
    """Return a dict with all raz_* columns set to None."""
    return {col: None for col in _RAZ_COLUMNS}


def run_norm_extraction_stage(df, cfg: Any) -> pd.DataFrame:
    """
    Stage 2 of Norm Extraction: Raz Norm Tuple Extraction.
    Uses constrained decoding to map reasoning traces to structured Raz norm tuples.

    Args:
        df: Input pandas DataFrame
        cfg: Configuration object
    """
    # Validate required input columns
    required = ["reasoning_trace"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"[norm_extraction] Missing required input columns: {missing}. "
            f"Available: {list(df.columns)}"
        )

    # Filter to rows that have a reasoning trace
    pre_filter = len(df)
    df = df[df["reasoning_trace"].notna() & (df["reasoning_trace"] != "")]
    post_filter = len(df)
    if pre_filter != post_filter:
        print(f"[norm_extraction] Filtered {pre_filter - post_filter} rows "
              f"with empty/null reasoning_trace ({post_filter} remaining)")
    if post_filter == 0:
        print("[norm_extraction] No rows with reasoning traces, returning empty")
        return df

    prompt_cfg = OmegaConf.select(cfg, "prompt_extraction") or OmegaConf.select(cfg, "prompt")
    if prompt_cfg is None:
        raise RuntimeError(
            "[norm_extraction] No prompt config found at 'prompt_extraction' or 'prompt'. "
            "Check config.yaml defaults and pipeline overrides."
        )

    system_prompt = OmegaConf.select(prompt_cfg, "system_prompt")
    prompt_template = OmegaConf.select(prompt_cfg, "prompt_template")
    if not system_prompt or not prompt_template:
        raise RuntimeError(
            f"[norm_extraction] Prompt config is missing required fields. "
            f"system_prompt={'present' if system_prompt else 'MISSING'}, "
            f"prompt_template={'present' if prompt_template else 'MISSING'}"
        )
    system_prompt = str(system_prompt)
    prompt_template = str(prompt_template)
    print(f"[norm_extraction] Loaded prompt from config "
          f"(system_prompt: {len(system_prompt)} chars, prompt_template: {len(prompt_template)} chars)",
          flush=True)

    def _format_prompt(row: Dict[str, Any]) -> str:
        text = str(row.get("norm_snippet") or row.get("article_text") or "")
        reasoning = str(row.get("reasoning_trace", ""))
        book_context = ""
        title = row.get("book_title", "")
        author = row.get("book_author", "")
        summary = row.get("book_summary", "")
        if title:
            book_context = f'Novel Context:\nThe source text below is a short excerpt from the novel "{title}"'
            if author:
                book_context += f" by {author}"
            book_context += (
                ". It is one of many consecutive chunks extracted from the full novel. "
                "Use the summary below to understand the broader societal context.\n"
            )
            if summary:
                book_context += f"\nNovel summary: {summary}\n\n---\n\n"
            else:
                book_context += "\n---\n\n"
        return (prompt_template
                .replace("{{book_context}}", book_context)
                .replace("{{article_text}}", text)
                .replace("{{reasoning_trace}}", reasoning))

    sampling_params = dict(
        OmegaConf.to_container(
            OmegaConf.select(cfg, "sampling_params"),
            resolve=True,
        ) or {}
    )
    sampling_params.setdefault("temperature", 0.0)
    sampling_params.setdefault("max_tokens", 2048)
    json_schema = PrescriptiveNormExtractionResult.model_json_schema()
    sampling_params["guided_decoding"] = {"json": json_schema}

    def _extract_json(gen_text: str) -> tuple[dict | None, str | None]:
        obj = None
        parse_error = None
        json_text = gen_text

        if "{" in gen_text:
            start = gen_text.find("{")
            end = gen_text.rfind("}") + 1
            if start < end:
                json_text = gen_text[start:end]

        try:
            obj = json.loads(json_text)
        except json.JSONDecodeError as e:
            parse_error = e
            if _JSON_REPAIR_OK:
                try:
                    repaired = repair_json(json_text, return_objects=True)
                    if isinstance(repaired, dict):
                        obj = repaired
                except Exception as repair_err:
                    parse_error = f"JSON repair failed: {repair_err}"
        return obj, parse_error

    def _preprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        result_row = dict(row)
        user_prompt = _format_prompt(result_row)
        result_row["messages"] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        result_row["sampling_params"] = sampling_params
        return result_row

    def _postprocess(row: Dict[str, Any]) -> Dict[str, Any]:
        result_row = dict(row)
        result_row.pop("messages", None)
        result_row.pop("sampling_params", None)
        result_row.pop("usage", None)
        gen_text = result_row.get("generated_text", "{}")
        obj, parse_error = _extract_json(gen_text)
        if obj is not None:
            norms = obj.get("norms", [])
            result_row["raz_norms_raw"] = norms
            result_row["raz_norm_count"] = len(norms)
            result_row["extraction_failed"] = False

            if norms:
                # Flatten norms[0] as default columns (will be overwritten
                # by explosion below if multiple norms exist)
                result_row.update(_flatten_norm(norms[0]))
            else:
                result_row.update(_null_raz_columns())
        else:
            print(f"[norm_extraction] JSON parse error: {parse_error}")
            result_row["extraction_error"] = str(parse_error)
            result_row["extraction_failed"] = True
            result_row["raz_norms_raw"] = []
            result_row["raz_norm_count"] = 0
            result_row.update(_null_raz_columns())
        return result_row

    result_df = run_vllm_inference(
        df=df,
        cfg=cfg,
        preprocess=_preprocess,
        postprocess=_postprocess,
        stage_name="norm_extraction",
    )

    print(f"[norm_extraction] Completed inference, {len(result_df)} results")

    # Explode rows so each extracted norm gets its own row.
    # Must happen before _clean_for_parquet which drops raz_norms_raw.
    char_blocklist = _get_character_blocklist(cfg)

    if "raz_norms_raw" in result_df.columns:
        rows: List[Dict[str, Any]] = []
        for _, row in result_df.iterrows():
            norms_raw = row.get("raz_norms_raw")
            if isinstance(norms_raw, list) and len(norms_raw) > 0:
                added = 0
                for i, norm_entry in enumerate(norms_raw):
                    if not isinstance(norm_entry, dict):
                        print(f"[norm_extraction] Skipping non-dict norm entry "
                              f"at index {i}: {type(norm_entry)}")
                        continue
                    new_row = row.to_dict()
                    new_row["raz_norm_index"] = i
                    new_row["raz_norm_count"] = len(norms_raw)
                    new_row.update(_flatten_norm(norm_entry))
                    new_row = _validate_norm_quality(new_row, char_blocklist)
                    rows.append(new_row)
                    added += 1
                if added == 0:
                    # All entries were non-dict; keep row with nulls
                    new_row = row.to_dict()
                    new_row["raz_norm_index"] = None
                    new_row.update(_null_raz_columns())
                    new_row["norm_quality_flags"] = None
                    new_row["norm_quality_passed"] = None
                    rows.append(new_row)
            else:
                new_row = row.to_dict()
                new_row["raz_norm_index"] = None
                new_row["norm_quality_flags"] = None
                new_row["norm_quality_passed"] = None
                rows.append(new_row)
        pre_count = len(result_df)
        result_df = pd.DataFrame(rows)
        print(f"[norm_extraction] Exploded {pre_count} rows -> "
              f"{len(result_df)} rows (one per extracted norm)")

    # Report quality validation stats
    if "norm_quality_passed" in result_df.columns:
        valid_mask = result_df["norm_quality_passed"] == True  # noqa: E712
        total = valid_mask.notna().sum()
        passed = valid_mask.sum()
        flagged = total - passed
        print(f"[norm_extraction] Quality validation: {passed}/{total} norms "
              f"passed ({flagged} flagged with named characters or plot-specificity)")

    result_df = _clean_for_parquet(result_df)
    return result_df
