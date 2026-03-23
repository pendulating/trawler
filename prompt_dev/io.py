from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_theory(theory: str) -> str:
    value = (theory or "").strip().upper()
    if value not in {"CI", "IG"}:
        raise ValueError(f"Unsupported theory '{theory}'. Expected one of: CI, IG.")
    return value


def default_run_id() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def run_dir(base_output_dir: str, theory: str, run_id: str) -> Path:
    safe_theory = sanitize_theory(theory)
    return ensure_dir(Path(base_output_dir) / safe_theory / run_id)


def latest_run_id(base_output_dir: str, theory: str) -> str:
    theory_root = Path(base_output_dir) / sanitize_theory(theory)
    if not theory_root.exists():
        raise FileNotFoundError(f"No run directory exists yet for theory={theory}.")
    run_dirs = sorted([p for p in theory_root.iterdir() if p.is_dir()], key=lambda p: p.name)
    if not run_dirs:
        raise FileNotFoundError(f"No run directory exists yet for theory={theory}.")
    return run_dirs[-1].name


def artifact_path(base_output_dir: str, theory: str, run_id: str, filename: str) -> Path:
    return run_dir(base_output_dir, theory, run_id) / filename


def read_json(path: Path) -> Any:
    if not path.exists():
        raise FileNotFoundError(f"Required artifact not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=True, sort_keys=False)
        f.write("\n")


def read_text(path: Path) -> str:
    if not path.exists():
        raise FileNotFoundError(f"Required file not found: {path}")
    return path.read_text(encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text, encoding="utf-8")


def load_manifest(base_output_dir: str, theory: str, run_id: str) -> dict[str, Any]:
    path = artifact_path(base_output_dir, theory, run_id, "run_manifest.json")
    if not path.exists():
        return {
            "theory": sanitize_theory(theory),
            "run_id": run_id,
            "created_at": now_utc_iso(),
            "updated_at": now_utc_iso(),
            "steps": {},
        }
    return read_json(path)


def set_step_status(
    base_output_dir: str,
    theory: str,
    run_id: str,
    step_key: str,
    status: str,
    details: dict[str, Any] | None = None,
) -> None:
    manifest = load_manifest(base_output_dir, theory, run_id)
    manifest["updated_at"] = now_utc_iso()
    steps = manifest.setdefault("steps", {})
    steps[step_key] = {
        "status": status,
        "updated_at": now_utc_iso(),
        "details": details or {},
    }
    path = artifact_path(base_output_dir, theory, run_id, "run_manifest.json")
    write_json(path, manifest)

