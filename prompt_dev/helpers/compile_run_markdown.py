from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STEP_TITLES: dict[str, str] = {
    "step01": "Step 1 - Initial Elicitation",
    "step02": "Step 2 - Self-Refinement of Theoretical Overview",
    "step03": "Step 3 - Sub-Component Decomposition",
    "step04": "Step 4 - Recursive Component Definition",
    "step05": "Step 5 - Self-Refinement of Component Definitions",
    "step06": "Step 6 - Expert-in-the-Loop Revision",
    "step07": "Step 7 - Question Generation",
    "step08": "Step 8 - Uncertainty Estimation (Active Prompting)",
    "step09": "Step 9 - Expert Revision of High-Uncertainty Questions",
    "step10": "Step 10 - Expert Annotation",
    "step11": "Step 11 - Grounded Inference via CoT",
    "step12": "Step 12 - Alignment Assessment",
}


ARTIFACT_FILES: dict[str, str] = {
    "step01": "step_01_overview_y0.json",
    "step02": "step_02_overview_yT.json",
    "step03": "step_03_components_C.json",
    "step04": "step_04_definitions_d0.json",
    "step05": "step_05_definitions_dT.json",
    "step06": "step_06_K_star.json",
    "step07": "step_07_question_pool_50.json",
    "step08": "step_08_uncertainty_ranked.json",
    "step09": "step_09_Q_star_10.json",
    "step10": "step_10_A_star_10.json",
    "step11": "step_11_grounded_answers_hatA.json",
    "step12": "step_12_alignment_report.json",
}


def theory_full_name(theory: str) -> str:
    up = (theory or "").strip().upper()
    if up == "CI":
        return "Contextual Integrity"
    if up == "IG":
        return "Institutional Grammar"
    return theory


def read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def to_pretty_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=True, sort_keys=False)


def discover_run_dir(base_output_dir: str, run_id: str, theory: str | None) -> tuple[Path, str]:
    base = Path(base_output_dir)
    if theory:
        run_dir = base / theory.upper() / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory not found: {run_dir}")
        return run_dir, theory.upper()

    candidates = [p for p in base.glob(f"*/{run_id}") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No run found for run_id='{run_id}' under {base}")
    if len(candidates) > 1:
        paths = ", ".join(str(p) for p in candidates)
        raise RuntimeError(
            f"Multiple runs found for run_id='{run_id}'. Pass --theory explicitly. Candidates: {paths}"
        )
    run_dir = candidates[0]
    detected_theory = run_dir.parent.name.upper()
    return run_dir, detected_theory


def summarize_step(step_key: str, payload: dict[str, Any]) -> str:
    if step_key == "step01":
        return f"- Prompt: `{payload.get('prompt', '')}`\n- Initial overview captured."
    if step_key == "step02":
        iterations = payload.get("iterations", [])
        return f"- Refinement iterations: `{len(iterations)}`\n- Refined overview captured."
    if step_key == "step03":
        components = payload.get("components", [])
        names = [c.get("name", "") for c in components][:15]
        bullets = "\n".join(f"  - {name}" for name in names if name)
        return f"- Extracted components: `{len(components)}`\n- Component names:\n{bullets or '  - (none)'}"
    if step_key in {"step04", "step05"}:
        defs = payload.get("definitions", [])
        sample = defs[0] if defs else {}
        return f"- Definitions count: `{len(defs)}`\n- First component: `{sample.get('component', 'N/A')}`"
    if step_key == "step06":
        defs = payload.get("definitions", [])
        return "- Expert artifact detected (`K*`).\n" f"- Definitions in K*: `{len(defs)}`"
    if step_key == "step07":
        questions = payload.get("questions", [])
        first = questions[0] if questions else {}
        return (
            f"- Generated questions: `{len(questions)}`\n"
            f"- First question: `{first.get('question', 'N/A')}`"
        )
    if step_key == "step08":
        ranked = payload.get("ranked_questions", [])
        top = ranked[0] if ranked else {}
        return (
            f"- Ranked questions: `{len(ranked)}`\n"
            f"- K samples/question: `{payload.get('k_samples', 'N/A')}`\n"
            f"- Top-uncertainty question: `{top.get('question', 'N/A')}`\n"
            f"- Top uncertainty score: `{top.get('uncertainty', 'N/A')}`"
        )
    if step_key == "step09":
        questions = payload.get("questions", [])
        return f"- Expert-curated final questions (`Q*`): `{len(questions)}`"
    if step_key == "step10":
        answers = payload.get("answers", [])
        return f"- Expert-annotated answers (`A*`): `{len(answers)}`"
    if step_key == "step11":
        answers = payload.get("answers_hat", [])
        sample = answers[0] if answers else {}
        return (
            f"- Grounded model answers: `{len(answers)}`\n"
            f"- Sample question id: `{sample.get('question_id', 'N/A')}`"
        )
    if step_key == "step12":
        alignment = payload.get("alignment", [])
        mean_score = payload.get("mean_score", "N/A")
        return f"- Alignment records: `{len(alignment)}`\n- Mean alignment score: `{mean_score}`"
    return f"- Keys: {', '.join(payload.keys())}"


def build_markdown(run_dir: Path, theory: str) -> str:
    manifest_path = run_dir / "run_manifest.json"
    manifest = read_json(manifest_path) if manifest_path.exists() else {}
    steps_status = manifest.get("steps", {})
    generated_at = datetime.now(timezone.utc).isoformat()

    lines: list[str] = []
    lines.append(f"# Prompt Development Inspection Report - {run_dir.name}")
    lines.append("")
    lines.append("## Run Metadata")
    lines.append("")
    lines.append(f"- Theory: `{theory}` ({theory_full_name(theory)})")
    lines.append(f"- Run ID: `{run_dir.name}`")
    lines.append(f"- Run Directory: `{run_dir}`")
    if manifest:
        lines.append(f"- Created At: `{manifest.get('created_at', 'N/A')}`")
        lines.append(f"- Last Updated: `{manifest.get('updated_at', 'N/A')}`")
    lines.append(f"- Report Generated At: `{generated_at}`")
    lines.append("")

    lines.append("## Step Status Overview")
    lines.append("")
    lines.append("| Step | Title | Status | Artifact |")
    lines.append("|---|---|---|---|")
    for step_num in range(1, 13):
        step_key = f"step{step_num:02d}"
        status = steps_status.get(step_key, {}).get("status", "not_run")
        artifact_file = ARTIFACT_FILES[step_key]
        artifact_path = run_dir / artifact_file
        artifact_cell = f"`{artifact_file}`" if artifact_path.exists() else "`(missing)`"
        lines.append(f"| `{step_key}` | {STEP_TITLES[step_key]} | `{status}` | {artifact_cell} |")
    lines.append("")

    lines.append("## Detailed Step Outputs")
    lines.append("")
    for step_num in range(1, 13):
        step_key = f"step{step_num:02d}"
        title = STEP_TITLES[step_key]
        artifact = run_dir / ARTIFACT_FILES[step_key]
        lines.append(f"### {title}")
        lines.append("")
        if not artifact.exists():
            lines.append("- Artifact not present for this run.")
            lines.append("")
            continue
        try:
            payload = read_json(artifact)
            lines.append(summarize_step(step_key, payload))
            lines.append("")
            lines.append("#### Full Artifact")
            lines.append("")
            lines.append("```json")
            lines.append(to_pretty_json(payload))
            lines.append("```")
        except Exception as exc:
            lines.append(f"- Failed to parse artifact `{artifact.name}`: `{exc}`")
        lines.append("")

    lines.append("## Files Present in Run Directory")
    lines.append("")
    for p in sorted(run_dir.iterdir(), key=lambda x: x.name):
        if p.is_file():
            lines.append(f"- `{p.name}`")
    lines.append("")

    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compile available prompt_dev artifacts for a run_id into a polished Markdown "
            "inspection report."
        )
    )
    parser.add_argument("--run-id", required=True, help="Run ID under outputs/prompt_dev/<THEORY>/<RUN_ID>.")
    parser.add_argument("--theory", default=None, help="Optional CI or IG. If omitted, auto-detect.")
    parser.add_argument("--base-output-dir", default="prompt_dev/outputs/prompt_dev")
    parser.add_argument("--output", default=None, help="Optional output .md path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir, detected_theory = discover_run_dir(args.base_output_dir, args.run_id, args.theory)
    md = build_markdown(run_dir, detected_theory)
    output_path = Path(args.output) if args.output else (run_dir / f"inspection_{args.run_id}.md")
    output_path.write_text(md, encoding="utf-8")
    print(f"Wrote inspection report: {output_path}")


if __name__ == "__main__":
    main()

