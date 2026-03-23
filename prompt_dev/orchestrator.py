from __future__ import annotations

import importlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .io import (
    artifact_path,
    default_run_id,
    latest_run_id,
    sanitize_theory,
    set_step_status,
)
from .llm_client import LLMClientConfig, VLLMClient
from .server_manager import ServerConfig, ensure_server


STEP_MODULES: dict[str, str] = {
    "step01": "prompt_dev.steps.step_01_initial_elicitation",
    "step02": "prompt_dev.steps.step_02_self_refine_overview",
    "step03": "prompt_dev.steps.step_03_subcomponent_decomposition",
    "step04": "prompt_dev.steps.step_04_recursive_component_definition",
    "step05": "prompt_dev.steps.step_05_self_refine_component_definitions",
    "step06": "prompt_dev.steps.step_06_expert_revision",
    "step07": "prompt_dev.steps.step_07_question_generation",
    "step08": "prompt_dev.steps.step_08_uncertainty_estimation",
    "step09": "prompt_dev.steps.step_09_expert_revision_high_uncertainty",
    "step10": "prompt_dev.steps.step_10_expert_annotation",
    "step11": "prompt_dev.steps.step_11_grounded_inference",
    "step12": "prompt_dev.steps.step_12_alignment_assessment",
}

MODEL_REQUIRED_STEPS = {"step01", "step02", "step03", "step04", "step05", "step07", "step08", "step11"}


@dataclass
class RunContext:
    cfg: dict[str, Any]
    theory: str
    run_id: str
    step_key: str

    @property
    def base_output_dir(self) -> str:
        return str(self.cfg["io"]["base_output_dir"])

    @property
    def endpoint(self) -> str:
        return str(self.cfg["server"]["endpoint"])

    @property
    def model_name(self) -> str:
        return str(self.cfg["model"]["name"])

    def artifact(self, filename: str) -> Path:
        return artifact_path(self.base_output_dir, self.theory, self.run_id, filename)

    def get_client(self) -> VLLMClient:
        llm_cfg = LLMClientConfig(
            endpoint=self.endpoint,
            model=self.model_name,
            timeout_sec=int(self.cfg["model"].get("timeout_sec", 180)),
            max_retries=int(self.cfg["model"].get("max_retries", 3)),
            retry_backoff_sec=float(self.cfg["model"].get("retry_backoff_sec", 2.0)),
        )
        return VLLMClient(llm_cfg)

    def temperature(self, fallback: float | None = None) -> float:
        sampling = self.cfg.get("sampling", {})
        per_step = sampling.get("step_temperatures", {}) or {}
        if self.step_key in per_step:
            return float(per_step[self.step_key])
        if fallback is not None:
            return float(fallback)
        return float(sampling.get("default_temperature", 0.2))


def load_config(config_path: str) -> dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    # Default shape
    data.setdefault("io", {})
    data["io"].setdefault("base_output_dir", "outputs/prompt_dev")
    data.setdefault("server", {})
    data["server"].setdefault("mode", "external")
    data["server"].setdefault("endpoint", "http://127.0.0.1:8924")
    data["server"].setdefault("slurm_script", "scripts/run_qwen25_72b_llm.sub")
    data["server"].setdefault("startup_timeout_sec", 180)
    data["server"].setdefault("poll_interval_sec", 5)
    data["server"].setdefault("skip_submission_if_healthy", True)
    data.setdefault("model", {})
    data["model"].setdefault("name", "qwen25-72b-llm")
    data["model"].setdefault("timeout_sec", 180)
    data["model"].setdefault("max_retries", 3)
    data["model"].setdefault("retry_backoff_sec", 2.0)
    data.setdefault("sampling", {})
    data["sampling"].setdefault("default_temperature", 0.2)
    data["sampling"].setdefault("default_top_p", 0.95)
    data["sampling"].setdefault("default_max_tokens", 2048)
    data["sampling"].setdefault("active_prompt_temperature", 0.7)
    data["sampling"].setdefault("active_prompt_k", 5)
    data["sampling"].setdefault(
        "step_temperatures",
        {
            "step01": 0.3,
            "step02": 0.1,
            "step03": 0.1,
            "step04": 0.2,
            "step05": 0.1,
            "step07": 0.4,
            "step08": 0.7,
            "step11": 0.1,
        },
    )
    data.setdefault("iterations", {})
    data["iterations"].setdefault("overview_refine_t", 2)
    data["iterations"].setdefault("definition_refine_t", 2)
    return data


def resolve_run_id(cfg: dict[str, Any], theory: str, run_id: str | None, create_if_missing: bool) -> str:
    if run_id:
        return run_id
    if create_if_missing:
        return default_run_id()
    return latest_run_id(str(cfg["io"]["base_output_dir"]), theory)


def maybe_ensure_server(ctx: RunContext) -> dict[str, Any] | None:
    if ctx.step_key not in MODEL_REQUIRED_STEPS:
        return None
    server_cfg = ServerConfig(
        mode=str(ctx.cfg["server"]["mode"]),
        endpoint=str(ctx.cfg["server"]["endpoint"]),
        slurm_script=str(ctx.cfg["server"]["slurm_script"]),
        startup_timeout_sec=int(ctx.cfg["server"]["startup_timeout_sec"]),
        poll_interval_sec=int(ctx.cfg["server"]["poll_interval_sec"]),
        skip_submission_if_healthy=bool(ctx.cfg["server"]["skip_submission_if_healthy"]),
    )
    return ensure_server(server_cfg)


def run_step(
    *,
    step_key: str,
    theory: str,
    config_path: str,
    run_id: str | None,
) -> dict[str, Any]:
    if step_key not in STEP_MODULES:
        raise ValueError(f"Unknown step key: {step_key}")
    clean_theory = sanitize_theory(theory)
    cfg = load_config(config_path)
    resolved_run_id = resolve_run_id(cfg, clean_theory, run_id, create_if_missing=(step_key == "step01"))
    ctx = RunContext(cfg=cfg, theory=clean_theory, run_id=resolved_run_id, step_key=step_key)
    set_step_status(ctx.base_output_dir, ctx.theory, ctx.run_id, step_key, "running")
    try:
        server_meta = maybe_ensure_server(ctx)
        module = importlib.import_module(STEP_MODULES[step_key])
        result: dict[str, Any] = module.run(ctx)  # type: ignore[attr-defined]
        if server_meta:
            result["server"] = server_meta
        set_step_status(ctx.base_output_dir, ctx.theory, ctx.run_id, step_key, "completed", result)
        return {"run_id": resolved_run_id, "step": step_key, "result": result}
    except Exception as exc:
        set_step_status(
            ctx.base_output_dir,
            ctx.theory,
            ctx.run_id,
            step_key,
            "failed",
            {"error": str(exc)},
        )
        raise


def extract_last_json_object(text: str) -> dict[str, Any] | None:
    if not isinstance(text, str) or not text.strip():
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    snippets = re.findall(r"\{[\s\S]*\}", text)
    for snip in reversed(snippets):
        try:
            obj = json.loads(snip)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None

