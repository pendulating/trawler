from typing import Any


def _as_list(values: Any) -> list[str]:
    try:
        # Ensure plain Python list of strings
        return [str(v) for v in list(values)]
    except Exception:
        return [str(values)]


def string_enum(values: list[str]) -> dict[str, Any]:
    return {"type": "string", "enum": _as_list(values)}


def nullable_string_enum(values: list[str]) -> dict[str, Any]:
    return {"anyOf": [string_enum(values), {"type": "null"}]}


def string_or_null() -> dict[str, Any]:
    return {"anyOf": [{"type": "string"}, {"type": "null"}]}


def array_of_strings() -> dict[str, Any]:
    return {"type": "array", "items": {"type": "string"}}


def object_schema(properties: dict[str, Any], required: list[str], additional_properties: bool = False) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(required),
        "additionalProperties": bool(additional_properties),
    }


