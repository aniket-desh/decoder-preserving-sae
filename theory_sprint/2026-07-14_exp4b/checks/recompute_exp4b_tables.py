#!/usr/bin/env python3
"""Recompute Experiment 4b natural-text headline tables from raw JSON.

This checks the claim that reported paired reductions equal ratios of the raw
per-group exact numerators. A mismatch above 1e-10 falsifies the audit.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def ratio_sum(values: list[float]) -> float:
    return sum(float(value) for value in values)


def recompute(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not payload.get("complete"):
        raise RuntimeError(f"incomplete evaluation artifact: {path}")

    models = payload["models"]
    protocol = payload["protocol"]
    base = (
        float(protocol["base_ridge"]),
        int(protocol["base_group_size"]),
        "contiguous",
    )
    exact = {
        (row["model"], float(row["ridge"]), int(row["group_size"]), row["grouping"]): row
        for row in payload["exact_identity_audit"]
    }
    reported = {
        (row["candidate"], row["baseline"]): row for row in payload["paired_reductions"]
    }

    mse_by_key = {
        (int(value["spec"]["seed"]), int(value["spec"]["k"])): name
        for name, value in models.items()
        if value["spec"]["method"] == "mse"
    }
    headline = []
    for name, value in models.items():
        spec = value["spec"]
        key = (int(spec["seed"]), int(spec["k"]))
        if spec["method"] == "mse" or key not in mse_by_key:
            continue
        baseline = mse_by_key[key]
        baseline_row = exact[(baseline, *base)]
        candidate_row = exact[(name, *base)]
        reduction = 1 - ratio_sum(candidate_row["numerator_by_group"]) / ratio_sum(
            baseline_row["numerator_by_group"]
        )
        nmse_change = models[name]["sampled_primary"]["nmse"] / models[baseline][
            "sampled_primary"
        ]["nmse"] - 1
        reported_value = reported[(name, baseline)]["exact_identity_reduction"]["estimate"]
        if abs(reduction - reported_value) > 1e-10:
            raise AssertionError((name, reduction, reported_value))
        headline.append(
            {
                "method": spec["method"],
                "seed": spec["seed"],
                "k": spec["k"],
                "exact_reduction": reduction,
                "nmse_change": nmse_change,
            }
        )

    rows_by_setting: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for row in payload["exact_identity_audit"]:
        model = models[row["model"]]
        setting = (
            row["audit_axis"],
            float(row["ridge"]),
            int(row["group_size"]),
            row["grouping"],
            int(model["spec"]["seed"]),
            int(model["spec"]["k"]),
        )
        rows_by_setting[setting][model["spec"]["method"]] = row

    robustness = []
    for setting, methods in rows_by_setting.items():
        if "mse" not in methods or "dpsae" not in methods:
            continue
        reduction = 1 - ratio_sum(methods["dpsae"]["numerator_by_group"]) / ratio_sum(
            methods["mse"]["numerator_by_group"]
        )
        axis, ridge, group_size, grouping, seed, k = setting
        robustness.append(
            {
                "axis": axis,
                "ridge": ridge,
                "group_size": group_size,
                "grouping": grouping,
                "seed": seed,
                "k": k,
                "exact_reduction": reduction,
            }
        )
    return {"artifact": str(path), "headline": headline, "robustness": robustness}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", nargs="+", type=Path)
    args = parser.parse_args()
    print(json.dumps([recompute(path) for path in args.artifacts], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
