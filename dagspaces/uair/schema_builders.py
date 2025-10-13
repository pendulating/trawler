from typing import Any, Dict, List


def _as_list(values: Any) -> List[str]:
    try:
        # Ensure plain Python list of strings
        return [str(v) for v in list(values)]
    except Exception:
        return [str(values)]


def string_enum(values: List[str]) -> Dict[str, Any]:
    return {"type": "string", "enum": _as_list(values)}


def nullable_string_enum(values: List[str]) -> Dict[str, Any]:
    return {"anyOf": [string_enum(values), {"type": "null"}]}


def string_or_null() -> Dict[str, Any]:
    return {"anyOf": [{"type": "string"}, {"type": "null"}]}


def array_of_strings() -> Dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def object_schema(properties: Dict[str, Any], required: List[str], additional_properties: bool = False) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": bool(additional_properties),
    }


