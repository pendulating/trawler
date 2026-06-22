# Vendored from CI-RL (github.com/EricGLan/CI-RL) - Apache 2.0 License
import json
import random
import re

from langchain.tools import BaseTool

from .my_typing import *

PRIMITIVE_TYPES = {
    "integer": int,
    "number": float,
    "string": str,
    "boolean": bool,
    "array": List,
    "object": Dict,
    "null": None,
}


def insert_indent(s: str, indent: str = "\t", insert_first=True) -> str:
    prefix = indent if insert_first else ""
    return prefix + s.rstrip("\n ").replace("\n", "\n" + indent) + "\n"


def create_str(data, include_desc=True, indent="\t"):
    name = data["name"]
    prefix = f"{indent}- " if include_desc else ""
    desc = f": {data['description']}" if include_desc else ""
    type = ""
    if "type" in data:
        type = data["type"]
        if not data.get("required", True):
            type += ", optional"
        type = f" ({type})"
    return f"{prefix}{name}{type}{desc}"


class ArgParameter(TypedDict):
    name: str
    type: str
    description: str
    required: bool


class ArgReturn(TypedDict):
    name: str
    type: str
    description: str


class ArgException(TypedDict):
    name: str
    description: str


class DummyToolWithMessage(BaseTool):
    name: str = "dummy_tool_with_message"
    description: str = "Called when tool is not actually run."
    def _get_message(self, msg):
        return msg
    def _run(self, msg):
        return self._get_message(msg)
    async def _arun(self, msg):
        return self._get_message(msg)


class InvalidTool(DummyToolWithMessage):
    name: str = "invalid_tool"
    description: str = "Called when tool name is invalid."
    available_tools: List[str] = []
    def _get_message(self, tool_name):
        availabel_tool_string = ", ".join(self.available_tools)
        return f'{{"error": "InvalidRequestException: {tool_name} is not a valid tool, try another one. You should use available tools in [{availabel_tool_string}] only."}}'


def get_first_json_object_str(s, enable_check=True, *args, **kwargs):
    regex = r"```(?:json)?\s*(\{.*?\})\s*```"
    match = re.search(regex, s, flags=re.DOTALL)
    if match:
        s = match.group(1)
    s = s.lstrip()
    ret = s
    while True:
        try:
            _ = json.loads(s, *args, **kwargs)
            break
        except json.JSONDecodeError as e:
            fix_res = special_fix_for_json_obj(s, e.pos)
            if fix_res is not None:
                ret = s = fix_res
            else:
                s = s[:e.pos].rstrip()
                break
    try:
        _ = json.loads(s, *args, **kwargs)
        ret = s
    except Exception as e:
        if enable_check:
            raise e
    return ret


def special_fix_for_json_obj(s, pos):
    if pos >= len(s):
        return None
    if s[pos] == "\\":
        broken = s[pos:pos+2]
        return s.replace(broken, broken[1])
    elif s[pos:pos+4] == "True":
        return s.replace("True", "true")
    elif s[pos:pos+5] == "False":
        return s.replace("False", "false")
    return None


def load_dict(tool_input):
    try:
        params = json.loads(tool_input, strict=False)
        return params
    except Exception as e:
        msg = str(e)
        if msg.startswith("Invalid \\escape"):
            raise ValueError(f"Invalid syntax for \\\\', you may need to use \\\\\\\\' instead")
        raise ValueError(f"Invalid format: {e}. Please format it as a JSON object.")


def match_type(value, expected_type):
    if isinstance(value, expected_type):
        return True
    return isinstance(value, int) and expected_type == float


def check_type(name, value, expected_type):
    if not match_type(value, expected_type):
        raise ValueError(f"Invalid type for {name}: {type(value).__name__}, expected {expected_type.__name__}.")


def validate_inputs(parameters, inputs):
    remains = set(inputs.keys())
    for param in parameters:
        name = param["name"]
        if name in remains:
            remains.remove(name)
        if name in inputs:
            expected_type = PRIMITIVE_TYPES[param["type"]]
            check_type(name, inputs[name], expected_type)
        elif param.get("required", False):
            raise ValueError(f"Missing required parameter: {name}.")
    if remains:
        raise ValueError(f"Unexpected parameters: {remains}.")


def validate_outputs(returns, outputs):
    if not isinstance(outputs, dict):
        raise ValueError(f"Invalid type for outputs: {type(outputs).__name__}, expected dict.")
    if list(outputs.keys()) == ["error"]:
        return
    remains = set(outputs.keys())
    for ret in returns:
        name = ret["name"]
        if name in remains:
            remains.remove(name)
        if name not in outputs:
            raise ValueError(f"Missing return: {name}.")
        expected_type = PRIMITIVE_TYPES[ret["type"]]
        check_type(name, outputs[name], expected_type)
    if remains:
        raise ValueError(f"Unexpected returns: {list(remains)}.")


def run_with_input_validation(run_func, inputs, tool, raw_inputs, **kwargs):
    try:
        params = load_dict(raw_inputs)
        validate_inputs(tool.parameters, params)
    except Exception as e:
        return DummyToolWithMessage().run(f'{{"error": "InvalidRequestException: {e}"}}', **kwargs)
    return run_func(inputs, **kwargs)


def find_toolkit_spec(name, toolkits):
    for toolkit in toolkits:
        if toolkit["toolkit"] == name:
            return toolkit
    raise ValueError(f"Toolkit not found: {name}.")


def format_toolkit_dict(toolkit, namekey, add_risks=True, indent="\t", use_simple_tool_desc=False):
    toolkit_name = toolkit[namekey]
    desc = f"<{toolkit_name}> toolkit with following tool APIs:\n"
    for tool in toolkit["tools"]:
        if use_simple_tool_desc:
            summary = tool["summary"].rstrip("\n ")
            desc += f"{indent}* {tool['name']}: {summary}\n"
        else:
            summary = tool["summary"].rstrip("\n ")
            desc += f"{indent}* {tool['name']}"
            params = [p['name'] for p in tool["parameters"]]
            desc += f"({', '.join(params)})"
            returns = [r['name'] for r in tool["returns"]]
            desc += f" -> {', '.join(returns)}"
            desc += f" : {summary}\n"
    if add_risks and "risks" in toolkit:
        desc += "Potential Risky Outcomes:\n"
        risks = list(toolkit["risks"])
        random.shuffle(risks)
        for risk in risks:
            desc += f"{indent}* {risk}\n"
    return desc
