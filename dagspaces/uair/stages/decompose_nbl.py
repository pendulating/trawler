from typing import Any, Dict, Optional
import pandas as pd
import json
import os
from omegaconf import OmegaConf
from enum import Enum
try:
    from pydantic import BaseModel, Field  # type: ignore
    try:
        from pydantic import ConfigDict  # type: ignore
    except Exception:
        ConfigDict = None  # type: ignore
except Exception:
    BaseModel = None  # type: ignore
    Field = None  # type: ignore
    ConfigDict = None  # type: ignore
from dagspaces.uair.schema_builders import (
    object_schema,
    nullable_string_enum,
    string_or_null,
    array_of_strings,
)

from dagspaces.common.vllm_inference import run_vllm_inference
from dagspaces.common.stage_utils import (
    maybe_silence_vllm_logs,
    to_json_str,
    serialize_arrow_unfriendly_in_row,
    extract_last_json,
    sanitize_for_json,
)

_SCHEMA_WRITTEN = False  # one-time per-process schema dump guard
_SAMPLING_PARAMS_WRITTEN = False  # one-time per-process sampling params dump

## Pydantic model for structured outputs (prefer over raw JSON Schema)
## Falls back to legacy dict-based schema if Pydantic is unavailable.
if BaseModel is not None:
    class DeploymentDomain(str, Enum):
        BIOMETRIC_IDENTIFICATION_AND_CATEGORIZATION_OF_NATURAL_PERSONS = "Biometric identification and categorization of natural persons"
        FAMILY = "Family"
        ROMANTIC_RELATIONSHIPS_AND_FRIENDSHIPS = "Romantic relationships and friendships"
        HEALTH_AND_HEALTHCARE = "Health and Healthcare"
        WELL_BEING = "Well-being"
        HUMAN_COMPUTER_INTERACTION = "Human-Computer Interaction"
        FINANCE_AND_INVESTMENT = "Finance and Investment"
        EDUCATION_AND_VOCATIONAL_TRAINING = "Education and vocational training"
        EMPLOYMENT_WORKERS_MANAGEMENT_AND_ACCESS_TO_SELF_EMPLOYMENT = "Employment, workers management and access to self-employment"
        ESSENTIAL_PRIVATE_SERVICES_AND_PUBLIC_SERVICES_AND_BENEFITS = "Essential private services and public services and benefits"
        RECOMMENDER_SYSTEMS_AND_PERSONALIZATION = "Recommender Systems and Personalization"
        SOCIAL_MEDIA = "Social Media"
        SPORTS_AND_RECREATION = "Sports and Recreation"
        ARTS_AND_ENTERTAINMENT = "Arts and Entertainment"
        SECURITY_AND_CYBERSECURITY = "Security and Cybersecurity"
        MARKETING_AND_ADVERTISING = "Marketing and Advertising"
        AGRICULTURE_AND_FARMING = "Agriculture and Farming"
        ENTREPRENEURSHIP = "Entrepreneurship"
        AUTONOMOUS_ROBOTS_AND_ROBOTICS = "Autonomous Robots and Robotics"
        INNOVATION_AND_RESEARCH = "Innovation and Research"
        MANAGEMENT_AND_OPERATION_OF_CRITICAL_INFRASTRUCTURE = "Management and Operation of critical infrastructure"
        LAW_ENFORCEMENT = "Law enforcement"
        MIGRATION_ASYLUM_AND_BORDER_CONTROL_MANAGEMENT = "Migration, Asylum and Border control management"
        DEMOCRACY = "Democracy"
        MEDIA_AND_COMMUNICATION = "Media and Communication"
        ACCESSIBILITY_AND_INCLUSION = "Accessibility and Inclusion"
        ENERGY = "Energy"
        MILITARY_AND_DEFENSE = "Military and Defense"
        ADMINISTRATION_OF_JUSTICE_AND_DEMOCRATIC_PROCESSES = "Administration of justice and democratic processes"
        GOVERNMENT_SERVICES_AND_ADMINISTRATION = "Government Services and Administration"
        DIPLOMACY_AND_FOREIGN_POLICY = "Diplomacy and Foreign Policy"
        FOOD_SAFETY_AND_REGULATION = "Food Safety and Regulation"
        CRISIS_MANAGEMENT_AND_EMERGENCY_RESPONSE = "Crisis Management and Emergency Response"
        HUMANITARIAN_AID = "Humanitarian Aid"
        TRANSPORT_AND_LOGISTICS = "Transport and Logistics"
        URBAN_PLANNING = "Urban Planning"
        COUNTERTERRORISM = "Counterterrorism"
        ENVIRONMENT_AND_SUSTAINABILITY = "Environment and Sustainability"
        INTERNATIONAL_LAW_ENFORCEMENT_AND_COOPERATION = "International Law Enforcement and Cooperation"
        CLIMATE_CHANGE_MITIGATION_AND_ADAPTATION = "Climate Change Mitigation and Adaptation"
        GAMING_AND_INTERACTIVE_EXPERIENCES = "Gaming and interactive experiences"
        HOBBIES = "Hobbies"
        SMART_HOME = "Smart home"
        SOCIAL_AND_COMMUNITY_SERVICES = "Social and Community Services"
        PUBLIC_AND_PRIVATE_TRANSPORTATION = "Public and private transportation"
        INTERPERSONAL_COMMUNICATION = "Interpersonal Communication"

    class DeploymentSpace(str, Enum):
        ONLINE_SPACE = "Online space"
        PUBLICLY_ACCESSIBLE_SPACE = "Publicly accessible space"
        NOT_PUBLICLY_ACCESSIBLE_SPACE = "Not publicly accessible space"

    class NBLDecomposition(BaseModel):
        # Pydantic v2 vs v1 configuration for forbidding extra keys
        if ConfigDict is not None:  # pydantic v2
            model_config = ConfigDict(extra="forbid")
        else:  # pydantic v1
            class Config:
                extra = "forbid"

        deployment_domain: Optional[DeploymentDomain] = None
        deployment_space: Optional[DeploymentSpace] = None
        purpose: Optional[str] = None
        capability: Optional[str] = None
        ai_deployer: Optional[str] = None
        location_of_ai_deployer: Optional[str] = None
        ai_subject: Optional[str] = None
        location_of_ai_subject: Optional[str] = None
        ai_developer: Optional[str] = None
        date___time_of_event: Optional[str] = None
        date_and_time_of_event: Optional[str] = None
        list_of_harms_that_occurred: list[str]
        list_of_risks_that_occurred: list[str]
        list_of_benefits_that_occurred: list[str]
        missing: list[str]



def _inline_json_schema_refs(schema: Dict[str, Any]) -> Dict[str, Any]:
    """Inline simple JSON Schema $ref that point into $defs/definitions.

    - Supports refs like "#/$defs/Name" and "#/definitions/Name".
    - Removes $defs/definitions and metadata like title/default.
    - Best-effort; returns original schema on failure.
    """
    try:
        if not isinstance(schema, dict):
            return schema
        defs = schema.get("$defs") or schema.get("definitions") or {}

        def resolve_ref(ref_value: Any) -> Optional[Dict[str, Any]]:
            try:
                if not isinstance(ref_value, str):
                    return None
                if ref_value.startswith("#/"):
                    parts = ref_value[2:].split("/")
                    if len(parts) == 2 and parts[0] in ("$defs", "definitions"):
                        return defs.get(parts[1])
            except Exception:
                pass
            return None

        def rewrite(node: Any) -> Any:
            if isinstance(node, dict):
                if "$ref" in node:
                    target = resolve_ref(node.get("$ref"))
                    if isinstance(target, dict):
                        return rewrite(target)
                    return node
                out: Dict[str, Any] = {}
                for k, v in node.items():
                    if k in ("$defs", "definitions", "title", "default"):
                        continue
                    out[k] = rewrite(v)
                return out
            if isinstance(node, list):
                return [rewrite(x) for x in node]
            return node

        inlined = rewrite(schema)
        try:
            inlined.pop("$defs", None)
            inlined.pop("definitions", None)
        except Exception:
            pass
        return inlined
    except Exception:
        return schema



def _get_tokenizer(model_source: str):
    try:
        from transformers import AutoTokenizer  # type: ignore
        return AutoTokenizer.from_pretrained(model_source, trust_remote_code=True, use_fast=True)
    except Exception:
        return None

_TOK_CACHED = None  # lazy per-process tokenizer cache
def _get_tokenizer_cached(model_source: str):
    global _TOK_CACHED
    try:
        if _TOK_CACHED is None:
            _TOK_CACHED = _get_tokenizer(model_source)
        return _TOK_CACHED
    except Exception:
        return None

def _get_max_user_input_tokens(tokenizer, system_text: str, sampling_params: Dict[str, Any], cfg) -> int:
    try:
        system_tokens = len(tokenizer.encode(system_text, add_special_tokens=False)) if tokenizer else 0
        max_model_len = int(getattr(cfg.model, "engine_kwargs", {}).get("max_model_len", 4096))
        try:
            if hasattr(sampling_params, "get"):
                max_output = int(sampling_params.get("max_tokens", 1024) or 1024)
            else:
                max_output = 1024
        except Exception:
            max_output = 1024
        safety = 512
        return max(512, max_model_len - max_output - system_tokens - safety)
    except Exception:
        return 2048

def _trim_text_to_token_budget(text: str, tokenizer, token_budget: int) -> str:
    if token_budget is None or token_budget <= 0:
        return ""
    if tokenizer is None:
        # Fallback: approx 4 chars per token
        try:
            max_chars = int(token_budget) * 4
            return text if len(text or "") <= max_chars else str(text or "")[:max_chars]
        except Exception:
            return text
    try:
        ids = tokenizer.encode(text or "", add_special_tokens=False)
        if len(ids) <= token_budget:
            return text
        ids = ids[:token_budget]
        return tokenizer.decode(ids, skip_special_tokens=True)
    except Exception:
        # Last-resort char fallback
        try:
            max_chars = int(token_budget) * 4
            return text if len(text or "") <= max_chars else str(text or "")[:max_chars]
        except Exception:
            return text

def run_decomposition_stage_nbl(df: pd.DataFrame, cfg):
    """Urban AI risk decomposition stage (NBL variant).

    Uses the longer prompt in prompt/decompose_nbl_prompt.yaml and extracts an
    expanded schema, including deployment space, harms list.
    """
    base_cols = [
        "deployment_domain",
        "deployment_purpose",
        "deployment_capability",
        "identity_of_ai_deployer",
        "identity_of_ai_subject",
        "identity_of_ai_developer",
        "location_of_ai_deployer",
        "location_of_ai_subject",
        "date_and_time_of_event",
        "deployment_space",
        "list_of_harms_that_occurred",
        "list_of_risks_that_occurred",
        "list_of_benefits_that_occurred",
        "missing",
    ]
    if hasattr(df, "to_pandas"):
        df = df.to_pandas()
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["article_text"] + base_cols)
    out = df.copy()
    use_llm = bool(getattr(cfg.runtime, "use_llm_decompose", False))
    # Whether to JSON-serialize nested columns for Arrow/Parquet friendliness
    try:
        serialize_nested = bool(getattr(cfg.runtime, "serialize_nested_json", True))
    except Exception:
        serialize_nested = True
    if not use_llm:
        missing_keys = [
            "deployment_domain",
            "deployment_purpose",
            "deployment_capability",
            "identity_of_ai_deployer",
            "identity_of_ai_subject",
            "identity_of_ai_developer",
            "location_of_ai_deployer",
            "location_of_ai_subject",
            "date_and_time_of_event",
            "deployment_space",
            "list_of_harms_that_occurred",
            "list_of_risks_that_occurred",
            "list_of_benefits_that_occurred",
        ]
        for c in base_cols:
            out[c] = None
        out["missing"] = out.apply(lambda r: list(missing_keys), axis=1)
        if serialize_nested:
            out["missing"] = out["missing"].map(to_json_str)
        return out

    # Prefer the longer NBL prompt; fall back to decompose or generic prompt
    # Support prompt subdirectory override (e.g., 'general_ai' for general_ai/decompose_nbl_prompt.yaml)
    prompt_subdir = getattr(cfg.runtime, "prompt_subdirectory", None)
    if prompt_subdir:
        # Load prompt from subdirectory file
        try:
            from .classify_shared import inject_prompt_from_file
            inject_prompt_from_file(cfg, f"{prompt_subdir}/decompose_nbl_prompt.yaml")
            # After injection, the prompt will be in cfg.prompt, so update cfg.prompt_decompose_nbl
            if OmegaConf.select(cfg, "prompt.system_prompt"):
                OmegaConf.update(cfg, "prompt_decompose_nbl.system_prompt", OmegaConf.select(cfg, "prompt.system_prompt"), merge=True)
            if OmegaConf.select(cfg, "prompt.prompt_template"):
                OmegaConf.update(cfg, "prompt_decompose_nbl.prompt_template", OmegaConf.select(cfg, "prompt.prompt_template"), merge=True)
        except Exception:
            pass  # Fall back to config-based loading
    
    try:
        system_prompt = str(
            OmegaConf.select(cfg, "prompt_decompose_nbl.system_prompt")
            or OmegaConf.select(cfg, "prompt_decompose.system_prompt")
            or OmegaConf.select(cfg, "prompt.system_prompt")
            or ""
        )
    except Exception:
        system_prompt = ""
    try:
        prompt_template = str(
            OmegaConf.select(cfg, "prompt_decompose_nbl.prompt_template")
            or OmegaConf.select(cfg, "prompt_decompose.prompt_template")
            or OmegaConf.select(cfg, "prompt.prompt_template")
            or ""
        )
    except Exception:
        prompt_template = ""

    def _format_prompt(article_text: str) -> str:
        return (
            prompt_template
            .replace("{{rule_text}}", str(article_text or ""))
            .replace("{{article_text}}", str(article_text or ""))
        )

    # Prefer stage-specific sampling params when present; convert nested DictConfig -> dict
    try:
        sp_src = getattr(cfg, "sampling_params_decompose_nbl", getattr(cfg, "sampling_params_decompose", getattr(cfg, "sampling_params", {})))
        sampling_params = OmegaConf.to_container(sp_src, resolve=True) if isinstance(sp_src, (dict,)) or hasattr(sp_src, "_get_node") else dict(sp_src)
    except Exception:
        sampling_params = dict(getattr(cfg, "sampling_params", {}))

    def _pre(row: Dict[str, Any]) -> Dict[str, Any]:
        global _SCHEMA_WRITTEN, _SAMPLING_PARAMS_WRITTEN
        maybe_silence_vllm_logs()
        raw_article = str(row.get("article_text") or "")
        sp = dict(sampling_params)
        # Longer outputs for NBL prompt
        sp.setdefault("max_tokens", 1024)
        sp.setdefault("detokenize", False)
        # Optional guided decoding hook
        try:
            if bool(getattr(cfg.runtime, "guided_decoding_decompose", False)):
                # Prefer Pydantic-generated JSON Schema for compatibility with vLLM structured outputs
                schema = None
                if BaseModel is not None:
                    try:
                        # Pydantic v2
                        if hasattr(NBLDecomposition, "model_json_schema"):
                            schema = NBLDecomposition.model_json_schema()
                            try:
                                schema = _inline_json_schema_refs(schema)  # inline $ref/$defs for xgrammar
                            except Exception:
                                pass
                        # Pydantic v1 fallback
                        elif hasattr(NBLDecomposition, "schema"):
                            schema = NBLDecomposition.schema()  # type: ignore
                            try:
                                schema = _inline_json_schema_refs(schema)
                            except Exception:
                                pass
                    except Exception:
                        schema = None
                if not schema:
                    # Fallback to legacy builder if Pydantic unavailable
                    schema = object_schema(
                        properties={
                            "deployment_domain": string_or_null(),
                            "deployment_space": nullable_string_enum([
                                "Online space",
                                "Publicly accessible space",
                                "Not publicly accessible space",
                            ]),
                            "purpose": string_or_null(),
                            "capability": string_or_null(),
                            "ai_deployer": string_or_null(),
                            "location_of_ai_deployer": string_or_null(),
                            "ai_subject": string_or_null(),
                            "location_of_ai_subject": string_or_null(),
                            "ai_developer": string_or_null(),
                            "date___time_of_event": string_or_null(),
                            "date_and_time_of_event": string_or_null(),
                            "list_of_harms_that_occurred": array_of_strings(),
                            "list_of_risks_that_occurred": array_of_strings(),
                            "list_of_benefits_that_occurred": array_of_strings(),
                            "missing": array_of_strings(),
                        },
                        required=[
                            "missing",
                            "list_of_harms_that_occurred",
                            "list_of_risks_that_occurred",
                            "list_of_benefits_that_occurred",
                        ],
                        additional_properties=False,
                    )
                try:
                    schema_json_str = json.dumps(schema, ensure_ascii=False)
                except Exception:
                    schema_json_str = json.dumps(sanitize_for_json(schema), ensure_ascii=False)
                sp["guided_decoding"] = {"json": schema_json_str}
        except Exception:
            pass
        # Token-aware trimming to respect model context window
        try:
            tokenizer = _get_tokenizer_cached(str(getattr(cfg.model, "model_source", "")))
        except Exception:
            tokenizer = None
        try:
            max_user = _get_max_user_input_tokens(tokenizer, system_prompt, sp, cfg)
        except Exception:
            max_user = 2048
        # Estimate static tokens from the prompt template itself (without article text)
        try:
            static_user = _format_prompt("")
            static_tokens = len(tokenizer.encode(static_user, add_special_tokens=False)) if tokenizer else int(len(static_user or "") / 4)
        except Exception:
            static_tokens = 0
        try:
            article_budget = max(64, int(max_user) - int(static_tokens))
        except Exception:
            article_budget = 512
        trimmed_article = _trim_text_to_token_budget(raw_article, tokenizer, article_budget)
        try:
            was_trimmed = (trimmed_article != raw_article)
        except Exception:
            was_trimmed = False
        if was_trimmed:
            try:
                aid = row.get("article_id") or row.get("name") or "unknown"
            except Exception:
                aid = "unknown"
            try:
                orig_tokens = len(tokenizer.encode(raw_article, add_special_tokens=False)) if tokenizer else int(len(raw_article or "") / 4)
            except Exception:
                orig_tokens = None
            try:
                new_tokens = len(tokenizer.encode(trimmed_article, add_special_tokens=False)) if tokenizer else int(len(trimmed_article or "") / 4)
            except Exception:
                new_tokens = None
            try:
                max_out = int(sp.get("max_tokens", 1024) or 1024)
            except Exception:
                max_out = 1024
            try:
                print(f"[decompose_nbl] Trimmed article_id={aid} tokens {orig_tokens}->{new_tokens} (user_budget={article_budget}, static_tokens={static_tokens}, max_output={max_out}); chars {len(raw_article)}->{len(trimmed_article)}", flush=True)
            except Exception:
                pass
        user = _format_prompt(trimmed_article)
        # Remove artifacts that might override our messages/params or collide with schema keys
        base = {k: v for k, v in row.items() if k not in {"messages", "sampling_params", "generated_text", "llm_output", "json", "guided_decoding", "response_format", "structured_output"}}
        # Sanitize sampling params to ensure JSON-serializable (convert ndarrays -> lists)
        sp_sanitized = sanitize_for_json(sp)
        # Remove conflicting/unsupported structured output keys at top-level
        try:
            if isinstance(sp_sanitized, dict):
                for bad in ("json", "structured_output", "response_format"):
                    sp_sanitized.pop(bad, None)
        except Exception:
            pass
        # Do NOT promote to top-level 'json'; vLLM SamplingParams does not accept it

        # Debug: log the exact schema that will be consumed by xgrammar/guidance
        try:
            debug_schema = None
            if isinstance(sp_sanitized, dict):
                debug_schema = sp_sanitized.get("json")
                if debug_schema is None:
                    gd2 = sp_sanitized.get("guided_decoding", {})
                    if isinstance(gd2, dict):
                        debug_schema = gd2.get("json")
            if debug_schema is not None:
                try:
                    dbg_str = json.dumps(debug_schema, ensure_ascii=False)

                except Exception as e:
                    try:
                        dbg_str = json.dumps(sanitize_for_json(debug_schema), ensure_ascii=False)
                    except Exception:
                        dbg_str = f"<unserializable type={type(debug_schema)}>"
                    print(f"[decompose_nbl] structured_output schema not serializable: {e}; sanitized={dbg_str}", flush=True)
                # Also write schema to file once per process to avoid terminal truncation
                try:
                    if not _SCHEMA_WRITTEN:
                        try:
                            base_dir = (
                                str(OmegaConf.select(cfg, "pipeline.output_root")
                                    or OmegaConf.select(cfg, "runtime.output_root")
                                    or os.environ.get("SLURM_SUBMIT_DIR")
                                    or os.getcwd())
                            )
                        except Exception:
                            base_dir = os.environ.get("SLURM_SUBMIT_DIR") or os.getcwd()
                        debug_dir = os.path.join(base_dir, "debug", "decompose")
                        os.makedirs(debug_dir, exist_ok=True)
                        out_path = os.path.join(debug_dir, f"structured_schema_pid{os.getpid()}.json")
                        with open(out_path, "w", encoding="utf-8") as f:
                            json.dump(sanitize_for_json(debug_schema), f, ensure_ascii=False, indent=2)
                        print(f"[decompose_nbl] wrote structured_output schema to {out_path}", flush=True)
                        _SCHEMA_WRITTEN = True
                except Exception:
                    pass
        except Exception:
            pass
        base.update({
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user},
            ],
            "sampling_params": sp_sanitized,
        })
        # Debug: persist full sanitized sampling params to file for postmortem
        try:
            if not _SAMPLING_PARAMS_WRITTEN:
                try:
                    base_dir = (
                        str(OmegaConf.select(cfg, "pipeline.output_root")
                            or OmegaConf.select(cfg, "runtime.output_root")
                            or os.environ.get("SLURM_SUBMIT_DIR")
                            or os.getcwd())
                    )
                except Exception:
                    base_dir = os.environ.get("SLURM_SUBMIT_DIR") or os.getcwd()
                debug_dir = os.path.join(base_dir, "debug", "decompose")
                os.makedirs(debug_dir, exist_ok=True)
                out_path_sp = os.path.join(debug_dir, f"sampling_params_pid{os.getpid()}.json")
                with open(out_path_sp, "w", encoding="utf-8") as f:
                    json.dump(sp_sanitized, f, ensure_ascii=False, indent=2)
                print(f"[decompose_nbl] wrote sanitized sampling_params to {out_path_sp}", flush=True)
                _SAMPLING_PARAMS_WRITTEN = True
        except Exception:
            pass
        return base

    def _post(row: Dict[str, Any]) -> Dict[str, Any]:
        txt = row.get("generated_text")
        obj = extract_last_json(txt if isinstance(txt, str) else "") or {}

        # Strongly-typed parse using Pydantic model when available to ensure
        # canonical schema keys are mapped consistently to output columns.
        parsed = None
        parsed_dict: Optional[Dict[str, Any]] = None
        if BaseModel is not None and isinstance(txt, str) and txt.strip():
            try:
                # Pydantic v2
                if hasattr(NBLDecomposition, "model_validate_json"):
                    parsed = NBLDecomposition.model_validate_json(txt)  # type: ignore
                    if hasattr(parsed, "model_dump"):
                        parsed_dict = parsed.model_dump()  # type: ignore
                # Pydantic v1 fallback
                elif hasattr(NBLDecomposition, "parse_raw"):
                    parsed = NBLDecomposition.parse_raw(txt)  # type: ignore
                    if hasattr(parsed, "dict"):
                        parsed_dict = parsed.dict()  # type: ignore
            except Exception:
                parsed = None
                parsed_dict = None

        # Robust key normalization and synonyms mapping
        def _norm_key(k: Any) -> str:
            s = str(k).strip().lower()
            out = []
            for ch in s:
                out.append(ch if (ch.isalnum() or ch == "_") else "_")
            return "".join(out)

        norm_obj: Dict[str, Any] = {}
        try:
            for k, v in (obj.items() if isinstance(obj, dict) else []):
                norm_obj[_norm_key(k)] = v
        except Exception:
            pass

        def _first_key(keys):
            for k in keys:
                if k in norm_obj:
                    return norm_obj.get(k)
            return None

        # Normalize 'missing' items to canonical output column names
        def _normalize_missing(items: Any) -> list[str]:
            try:
                # Accepted set of canonical output columns
                canonical_set = {
                    "deployment_domain",
                    "deployment_purpose",
                    "deployment_capability",
                    "identity_of_ai_deployer",
                    "identity_of_ai_subject",
                    "identity_of_ai_developer",
                    "location_of_ai_deployer",
                    "location_of_ai_subject",
                    "date_and_time_of_event",
                    "deployment_space",
                    "list_of_harms_that_occurred",
                    "list_of_risks_that_occurred",
                    "list_of_benefits_that_occurred",
                }
                alias_to_canonical = {
                    # domain
                    "domain": "deployment_domain",
                    "deployment_domain": "deployment_domain",
                    # purpose
                    "purpose": "deployment_purpose",
                    "deployment_purpose": "deployment_purpose",
                    # capability
                    "capability": "deployment_capability",
                    "deployment_capability": "deployment_capability",
                    # space
                    "space": "deployment_space",
                    "deployment_space": "deployment_space",
                    # identities
                    "ai_deployer": "identity_of_ai_deployer",
                    "identity_of_ai_deployer": "identity_of_ai_deployer",
                    "ai_subject": "identity_of_ai_subject",
                    "identity_of_ai_subject": "identity_of_ai_subject",
                    "ai_developer": "identity_of_ai_developer",
                    "identity_of_ai_developer": "identity_of_ai_developer",
                    # locations
                    "location_of_ai_deployer": "location_of_ai_deployer",
                    "location_of_ai_subject": "location_of_ai_subject",
                    # date/time
                    "date_and_time_of_event": "date_and_time_of_event",
                    "date___time_of_event": "date_and_time_of_event",
                    "date_time": "date_and_time_of_event",
                    "datetime": "date_and_time_of_event",
                    "date": "date_and_time_of_event",
                    "time": "date_and_time_of_event",
                    # lists
                    "list_of_harms_that_occurred": "list_of_harms_that_occurred",
                    "harms": "list_of_harms_that_occurred",
                    "list_of_risks_that_occurred": "list_of_risks_that_occurred",
                    "risks": "list_of_risks_that_occurred",
                    "list_of_benefits_that_occurred": "list_of_benefits_that_occurred",
                    "benefits": "list_of_benefits_that_occurred",
                }
                out: list[str] = []
                seen = set()
                seq = items if isinstance(items, (list, tuple, set)) else [items]
                for it in seq:
                    try:
                        nk = _norm_key(it)
                        canonical = alias_to_canonical.get(nk)
                        if canonical and canonical in canonical_set and canonical not in seen:
                            out.append(canonical)
                            seen.add(canonical)
                    except Exception:
                        continue
                return out
            except Exception:
                return []

        # Map NBL fields to canonical outputs (prefer typed model; fallback to synonyms)
        if isinstance(parsed_dict, dict) and parsed_dict:
            deployment_domain = parsed_dict.get("deployment_domain")
            deployment_purpose = parsed_dict.get("purpose")
            deployment_capability = parsed_dict.get("capability")
            identity_of_ai_deployer = parsed_dict.get("ai_deployer")
            identity_of_ai_subject = parsed_dict.get("ai_subject")
            identity_of_ai_developer = parsed_dict.get("ai_developer")
            location_of_ai_deployer = parsed_dict.get("location_of_ai_deployer")
            location_of_ai_subject = parsed_dict.get("location_of_ai_subject")
            date_and_time_of_event = (
                parsed_dict.get("date_and_time_of_event")
                or parsed_dict.get("date___time_of_event")
            )
            deployment_space = parsed_dict.get("deployment_space")
            harms_list = parsed_dict.get("list_of_harms_that_occurred")
            risks_list = parsed_dict.get("list_of_risks_that_occurred")
            benefits_list = parsed_dict.get("list_of_benefits_that_occurred")
        else:
            deployment_domain = _first_key(["deployment_domain", "domain", "use_domain"]) 
            deployment_purpose = _first_key(["deployment_purpose", "purpose", "goal", "objective"]) 
            deployment_capability = _first_key(["deployment_capability", "capability", "capabilities", "function", "ability"]) 
            identity_of_ai_deployer = _first_key(["identity_of_ai_deployer", "ai_deployer", "deployer", "operator", "implementer", "user", "agency", "organization_deployer"]) 
            identity_of_ai_subject = _first_key(["identity_of_ai_subject", "ai_subject", "subject", "data_subject", "affected_party", "individual", "group"]) 
            identity_of_ai_developer = _first_key(["identity_of_ai_developer", "ai_developer", "developer", "vendor", "builder", "provider", "manufacturer"]) 
            location_of_ai_deployer = _first_key(["location_of_ai_deployer", "deployer_location", "operator_location", "location_deployer"]) 
            location_of_ai_subject = _first_key(["location_of_ai_subject", "subject_location", "location_subject", "where"]) 
            # Note: handle both date_and_time_of_event and date___time_of_event (normalized from 'Date & Time of Event')
            date_and_time_of_event = _first_key(["date_and_time_of_event", "date___time_of_event", "datetime", "date_time", "date", "time", "event_time", "when"]) 
            deployment_space = _first_key(["deployment_space"]) 
            harms_list = _first_key(["list_of_harms_that_occurred", "harms", "list_of_harms"]) 
            risks_list = _first_key(["list_of_risks_that_occurred", "risks", "list_of_risks"]) 
            benefits_list = _first_key(["list_of_benefits_that_occurred", "benefits", "list_of_benefits"]) 
        missing_raw = _first_key(["missing", "missing_elements", "missing_fields"]) or []
        if not isinstance(missing_raw, list):
            try:
                missing_raw = list(missing_raw) if missing_raw is not None else []
            except Exception:
                missing_raw = []
        missing_norm = _normalize_missing(missing_raw)

        # Optionally serialize nested columns for Arrow/Parquet compatibility
        if serialize_nested:
            serialize_arrow_unfriendly_in_row(row, [
                "messages",
                "sampling_params",
                "usage",
                "token_counts",
            ])
            missing_out = to_json_str(missing_norm)
            # Persist typed/normalized JSON for traceability when available
            if isinstance(parsed_dict, dict):
                row["llm_json"] = to_json_str(parsed_dict)
            else:
                row["llm_json"] = to_json_str(obj)
        else:
            missing_out = missing_norm
            row["llm_json"] = parsed_dict if isinstance(parsed_dict, dict) else obj

        # Ensure scalar columns are Arrow-friendly (no lists/dicts in columns)
        def _norm_ci_value(v: Any) -> Any:
            if isinstance(v, (list, tuple)):
                return to_json_str(v) if serialize_nested else ", ".join([str(x) for x in v if x is not None])
            if isinstance(v, dict):
                return to_json_str(v) if serialize_nested else str(v)
            return v

        deployment_domain_out = _norm_ci_value(deployment_domain)
        deployment_purpose_out = _norm_ci_value(deployment_purpose)
        deployment_capability_out = _norm_ci_value(deployment_capability)
        identity_of_ai_deployer_out = _norm_ci_value(identity_of_ai_deployer)
        identity_of_ai_subject_out = _norm_ci_value(identity_of_ai_subject)
        identity_of_ai_developer_out = _norm_ci_value(identity_of_ai_developer)
        location_of_ai_deployer_out = _norm_ci_value(location_of_ai_deployer)
        location_of_ai_subject_out = _norm_ci_value(location_of_ai_subject)
        date_and_time_of_event_out = _norm_ci_value(date_and_time_of_event)
        deployment_space_out = _norm_ci_value(deployment_space)
        harms_list_out = _norm_ci_value(harms_list)
        risks_list_out = _norm_ci_value(risks_list)
        benefits_list_out = _norm_ci_value(benefits_list)

        return {
            **row,
            "deployment_domain": deployment_domain_out,
            "deployment_purpose": deployment_purpose_out,  # maps from schema field 'purpose'
            "deployment_capability": deployment_capability_out,  # maps from schema field 'capability'
            "identity_of_ai_deployer": identity_of_ai_deployer_out,
            "identity_of_ai_subject": identity_of_ai_subject_out,
            "identity_of_ai_developer": identity_of_ai_developer_out,
            "location_of_ai_deployer": location_of_ai_deployer_out,
            "location_of_ai_subject": location_of_ai_subject_out,
            "date_and_time_of_event": date_and_time_of_event_out,
            "deployment_space": deployment_space_out,
            "list_of_harms_that_occurred": harms_list_out,
            "list_of_risks_that_occurred": risks_list_out,
            "list_of_benefits_that_occurred": benefits_list_out,
            "missing": missing_out,
            "llm_output": row.get("generated_text"),
            "_progress_row": 1,
        }

    out_df = run_vllm_inference(out, cfg, _pre, _post, "decompose_nbl")
    for c in base_cols:
        if c not in out_df.columns:
            out_df[c] = None
    # Stage-scoped logging is handled by the orchestrator
    return out_df


