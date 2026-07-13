#!/usr/bin/env python3
"""Audit the complete Experiment 4 artifact tree and emit a machine-readable report."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def read_json(path: Path, errors: list[str]) -> Any | None:
    if not path.is_file():
        errors.append(f"missing file: {path}")
        return None
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        errors.append(f"invalid JSON: {path}: {error}")
        return None


def expected_models(config: dict, stage: str) -> list[str]:
    if stage == "screen":
        weights = config["training"]["decoder_weight_multipliers"]
        return ["mse_s0", "whitening_s0", *(f"dpsae_w{weight:g}_s0" for weight in weights)]
    if stage == "confirmation":
        return [
            name
            for seed in config["training"]["confirmation_seeds"]
            for name in (f"mse_s{seed}", f"dpsae_s{seed}", f"whitening_s{seed}")
        ]
    k = int(stage.removeprefix("robustness"))
    return [
        name
        for seed in config["training"]["robustness_seeds"]
        for name in (f"mse_k{k}_s{seed}", f"dpsae_k{k}_s{seed}")
    ]


def method_name(model_name: str) -> str:
    if model_name.startswith("dpsae"):
        return "dpsae"
    if model_name.startswith("whitening"):
        return "whitening"
    return "mse"


def finite_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(value)


def audit_stage(
    artifact_dir: Path,
    config: dict,
    stage: str,
    token_budget: int,
    errors: list[str],
) -> dict:
    stage_dir = artifact_dir / stage
    done = read_json(stage_dir / "done.json", errors)
    validation = read_json(stage_dir / "validation.json", errors)
    model_path = stage_dir / "models.pt"
    if not model_path.is_file() or model_path.stat().st_size == 0:
        errors.append(f"missing or empty model fleet: {model_path}")
    expected = expected_models(config, stage)
    if done is not None:
        if done.get("stage") != stage:
            errors.append(f"{stage}: wrong done marker stage {done.get('stage')!r}")
        if done.get("tokens_seen", 0) < token_budget:
            errors.append(
                f"{stage}: only {done.get('tokens_seen', 0):,} of {token_budget:,} tokens"
            )
    summary: dict[str, dict[str, float | int]] = {}
    if validation is None:
        return summary
    missing = sorted(set(expected) - set(validation))
    extra = sorted(set(validation) - set(expected))
    if missing:
        errors.append(f"{stage}: missing validation models: {missing}")
    if extra:
        errors.append(f"{stage}: unexpected validation models: {extra}")
    for name in expected:
        row = validation.get(name)
        if not isinstance(row, dict):
            continue
        for metric in ("nmse", "decoder", "l0"):
            if not finite_number(row.get(metric)):
                errors.append(f"{stage}/{name}: invalid {metric}: {row.get(metric)!r}")
        if not isinstance(row.get("dead"), int) or row["dead"] < 0:
            errors.append(f"{stage}/{name}: invalid dead-feature count: {row.get('dead')!r}")
    for method in sorted({method_name(name) for name in expected}):
        rows = [
            validation[name]
            for name in expected
            if name in validation and method_name(name) == method
        ]
        if not rows:
            continue
        summary[method] = {
            "models": len(rows),
            "median_nmse": statistics.median(row["nmse"] for row in rows),
            "median_decoder": statistics.median(row["decoder"] for row in rows),
            "median_l0": statistics.median(row["l0"] for row in rows),
            "dead_total": sum(row["dead"] for row in rows),
        }
    return summary


def check_probe_metrics(row: Any, label: str, errors: list[str]) -> None:
    if not isinstance(row, dict):
        errors.append(f"{label}: missing metric object")
        return
    for metric in ("accuracy", "auc"):
        value = row.get(metric)
        if not finite_number(value) or not 0 <= value <= 1:
            errors.append(f"{label}: invalid {metric}: {value!r}")


def feature_axis(rows: Any) -> list[Any] | None:
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        return None
    return [row.get("features") for row in rows]


def audit_analysis(artifact_dir: Path, config: dict, errors: list[str]) -> dict:
    analysis = read_json(artifact_dir / "analysis.json", errors)
    if analysis is None:
        return {}
    feature_counts = config["ioi"]["feature_counts"]
    summary = {}
    for stage in ("confirmation", "robustness16", "robustness64"):
        expected = expected_models(config, stage)
        stage_result = analysis.get(stage)
        if not isinstance(stage_result, dict):
            errors.append(f"analysis: missing stage {stage}")
            continue
        if set(stage_result) != set(expected):
            errors.append(
                f"analysis/{stage}: model set mismatch; "
                f"missing={sorted(set(expected) - set(stage_result))}, "
                f"extra={sorted(set(stage_result) - set(expected))}"
            )
        threshold_values: dict[str, list[int]] = {}
        for name in expected:
            row = stage_result.get(name)
            if not isinstance(row, dict):
                continue
            curve = row.get("sparse_probe_curve")
            if feature_axis(curve) != feature_counts:
                errors.append(f"analysis/{stage}/{name}: malformed sparse probe curve")
            else:
                for index, item in enumerate(curve):
                    check_probe_metrics(item, f"analysis/{stage}/{name}/curve/{index}", errors)
            check_probe_metrics(
                row.get("original_dense_probe"),
                f"analysis/{stage}/{name}/original_dense_probe",
                errors,
            )
            check_probe_metrics(
                row.get("reconstruction_dense_probe"),
                f"analysis/{stage}/{name}/reconstruction_dense_probe",
                errors,
            )
            threshold = row.get("features_to_80pct_dense")
            if threshold is not None and threshold not in feature_counts:
                errors.append(f"analysis/{stage}/{name}: invalid 80% threshold {threshold!r}")
            if threshold is not None:
                threshold_values.setdefault(method_name(name), []).append(threshold)
            if stage == "confirmation":
                causal = row.get("causal_frontier")
                collateral = row.get("collateral_frontier")
                if feature_axis(causal) != feature_counts:
                    errors.append(f"analysis/{stage}/{name}: malformed causal frontier")
                if feature_axis(collateral) != feature_counts:
                    errors.append(f"analysis/{stage}/{name}: malformed collateral frontier")
        summary[stage] = {
            method: statistics.median(values)
            for method, values in threshold_values.items()
        }
    for path in (
        artifact_dir / "ioi_state_activations.pt",
        artifact_dir / "figures" / "exp04_headline.pdf",
        artifact_dir / "figures" / "exp04_headline.png",
    ):
        if not path.is_file() or path.stat().st_size == 0:
            errors.append(f"missing or empty final artifact: {path}")
    return summary


def run_audit(artifact_dir: Path) -> dict:
    errors: list[str] = []
    config = read_json(artifact_dir / "resolved_config.json", errors)
    if config is None:
        return {"status": "failed", "errors": errors}
    selection = read_json(artifact_dir / "screening_selection.json", errors)
    if selection is not None and not finite_number(selection.get("selected_decoder_weight")):
        errors.append("screening selection has no finite decoder weight")
    token_budgets = {
        "screen": config["training"]["screen_tokens"],
        "confirmation": config["training"]["confirmation_tokens"],
        "robustness16": config["training"]["robustness_tokens"],
        "robustness64": config["training"]["robustness_tokens"],
    }
    validation_summary = {
        stage: audit_stage(artifact_dir, config, stage, budget, errors)
        for stage, budget in token_budgets.items()
    }
    analysis_summary = audit_analysis(artifact_dir, config, errors)
    return {
        "status": "passed" if not errors else "failed",
        "errors": errors,
        "validation_summary": validation_summary,
        "feature_threshold_summary": analysis_summary,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "artifact_dir",
        nargs="?",
        type=Path,
        default=ROOT / "artifacts" / "exp04_ioi_mechanism",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = run_audit(args.artifact_dir)
    output = args.output or args.artifact_dir / "completion_audit.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "passed" else 1)


if __name__ == "__main__":
    main()
