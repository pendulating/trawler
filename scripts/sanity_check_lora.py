#!/usr/bin/env python3
"""Sanity-check LoRA adapter weights for all model configs with a lora_path.

For each adapter found, reports:
  - Weight counts and shapes
  - lora_A / lora_B Frobenius norms (min, mean, max)
  - Whether lora_B is effectively zero (dead adapter)
  - adapter_config.json base_model_name_or_path validity
  - Key prefix compatibility with the base model (VLM language_model.* check)

Usage:
    python scripts/sanity_check_lora.py
    python scripts/sanity_check_lora.py --model-dir dagspaces/common/conf/model
    python scripts/sanity_check_lora.py --adapter /path/to/adapter/dir
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path


def check_adapter(adapter_path: str, model_source: str | None = None, label: str = "") -> dict:
    """Check a single LoRA adapter directory. Returns a results dict."""
    from safetensors import safe_open

    results = {"label": label, "adapter_path": adapter_path, "issues": []}

    # --- adapter_config.json ---
    config_path = os.path.join(adapter_path, "adapter_config.json")
    if os.path.exists(config_path):
        with open(config_path) as f:
            acfg = json.load(f)
        base_model = acfg.get("base_model_name_or_path", "")
        results["base_model_name_or_path"] = base_model
        results["rank"] = acfg.get("r")
        results["target_modules"] = acfg.get("target_modules", [])
        if base_model and not os.path.exists(base_model):
            results["issues"].append(f"base_model_name_or_path does not exist: {base_model}")
    else:
        results["issues"].append("adapter_config.json not found")

    # --- adapter_model.safetensors ---
    sf_path = os.path.join(adapter_path, "adapter_model.safetensors")
    if not os.path.exists(sf_path):
        results["issues"].append("adapter_model.safetensors not found")
        return results

    with safe_open(sf_path, framework="pt") as f:
        keys = list(f.keys())
        norms_a, norms_b = [], []
        shapes_a, shapes_b = [], []
        for k in keys:
            t = f.get_tensor(k)
            n = t.float().norm().item()
            if "lora_A" in k:
                norms_a.append(n)
                shapes_a.append(tuple(t.shape))
            elif "lora_B" in k:
                norms_b.append(n)
                shapes_b.append(tuple(t.shape))

    results["total_keys"] = len(keys)
    results["lora_a_count"] = len(norms_a)
    results["lora_b_count"] = len(norms_b)
    results["file_size_mb"] = os.path.getsize(sf_path) / 1024 / 1024

    if norms_a:
        results["lora_a_norms"] = {
            "min": min(norms_a), "mean": sum(norms_a) / len(norms_a), "max": max(norms_a),
        }
    if norms_b:
        results["lora_b_norms"] = {
            "min": min(norms_b), "mean": sum(norms_b) / len(norms_b), "max": max(norms_b),
        }
        if max(norms_b) < 1e-6:
            results["issues"].append("DEAD ADAPTER: all lora_B weights are zero")
        elif sum(1 for n in norms_b if n < 1e-6) > len(norms_b) * 0.5:
            n_zero = sum(1 for n in norms_b if n < 1e-6)
            results["issues"].append(
                f"PARTIALLY DEAD: {n_zero}/{len(norms_b)} lora_B weights are zero"
            )

    # --- Key prefix check against base model ---
    if model_source and os.path.exists(model_source):
        base_sf_files = sorted(glob.glob(os.path.join(model_source, "*.safetensors")))
        if base_sf_files:
            with safe_open(base_sf_files[0], framework="pt") as bf:
                base_keys = list(bf.keys())
            has_lm_prefix = any("language_model.layers." in k for k in base_keys)

            from vllm.lora.utils import parse_fine_tuned_lora_name
            adapter_module_names = set()
            for k in keys:
                try:
                    mod, _ = parse_fine_tuned_lora_name(k)
                    adapter_module_names.add(mod)
                except Exception:
                    pass

            adapter_has_lm = any("language_model." in m for m in adapter_module_names)
            if has_lm_prefix and not adapter_has_lm:
                results["issues"].append(
                    "KEY MISMATCH: base model uses language_model.* prefix "
                    "but adapter keys do not — LoRA will silently fail"
                )
                # Check if _vlm_remapped exists
                remapped = os.path.join(adapter_path, "_vlm_remapped")
                if os.path.exists(remapped):
                    results["vlm_remapped"] = True
                else:
                    results["vlm_remapped"] = False
                    results["issues"].append(
                        "No _vlm_remapped/ cache — run an eval to auto-generate it"
                    )

    return results


def format_results(results: dict) -> str:
    """Pretty-print a single adapter check result."""
    lines = []
    label = results.get("label", results["adapter_path"])
    lines.append(f"\n{'=' * 70}")
    lines.append(f"  {label}")
    lines.append(f"{'=' * 70}")
    lines.append(f"  Adapter: {results['adapter_path']}")
    if "base_model_name_or_path" in results:
        lines.append(f"  Base model: {results['base_model_name_or_path']}")
    if "rank" in results:
        lines.append(f"  Rank: {results['rank']}  Targets: {results.get('target_modules', [])}")
    if "total_keys" in results:
        lines.append(f"  Weights: {results['total_keys']} tensors "
                     f"({results['lora_a_count']} A + {results['lora_b_count']} B), "
                     f"{results.get('file_size_mb', 0):.1f} MB")
    if "lora_a_norms" in results:
        n = results["lora_a_norms"]
        lines.append(f"  lora_A norms:  min={n['min']:.4f}  mean={n['mean']:.4f}  max={n['max']:.4f}")
    if "lora_b_norms" in results:
        n = results["lora_b_norms"]
        lines.append(f"  lora_B norms:  min={n['min']:.4f}  mean={n['mean']:.4f}  max={n['max']:.4f}")

    issues = results.get("issues", [])
    if issues:
        lines.append("")
        for issue in issues:
            lines.append(f"  ⚠  {issue}")
    else:
        lines.append(f"\n  ✓  No issues detected")

    return "\n".join(lines)


def scan_model_configs(model_dir: str) -> list[tuple[str, str, str]]:
    """Scan YAML configs for lora_path entries. Returns (label, lora_path, model_source)."""
    import yaml

    results = []
    for yaml_path in sorted(glob.glob(os.path.join(model_dir, "*.yaml"))):
        with open(yaml_path) as f:
            try:
                cfg = yaml.safe_load(f)
            except Exception:
                continue
        model_cfg = cfg.get("model", {})
        lora_path = model_cfg.get("lora_path")
        if not lora_path:
            continue
        model_source = model_cfg.get("model_source", "")
        label = Path(yaml_path).stem
        results.append((label, str(lora_path), str(model_source)))
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Sanity-check LoRA adapter weights.")
    parser.add_argument("--model-dir", default="dagspaces/common/conf/model",
                        help="Directory containing model YAML configs")
    parser.add_argument("--adapter", help="Check a single adapter directory instead of scanning configs")
    parser.add_argument("--model-source", help="Base model path (used with --adapter)")
    parser.add_argument("--json", action="store_true", help="Output results as JSON")
    args = parser.parse_args()

    all_results = []

    if args.adapter:
        r = check_adapter(args.adapter, model_source=args.model_source, label=args.adapter)
        all_results.append(r)
    else:
        configs = scan_model_configs(args.model_dir)
        if not configs:
            print(f"No model configs with lora_path found in {args.model_dir}")
            return 1
        print(f"Found {len(configs)} model configs with LoRA adapters\n")
        for label, lora_path, model_source in configs:
            r = check_adapter(lora_path, model_source=model_source, label=label)
            all_results.append(r)

    if args.json:
        print(json.dumps(all_results, indent=2, default=str))
    else:
        n_issues = 0
        for r in all_results:
            print(format_results(r))
            n_issues += len(r.get("issues", []))

        print(f"\n{'─' * 70}")
        print(f"Summary: {len(all_results)} adapters checked, "
              f"{n_issues} issue(s) found")
        if n_issues:
            print("Run with --json for machine-readable output.")

    return 1 if any(r.get("issues") for r in all_results) else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
