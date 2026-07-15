#!/usr/bin/env python3
"""Merge independently run seeds of the empirical structured-prior sweep."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from typing import Any


TABLES = (
    "metrics",
    "group_metrics",
    "training_curves",
    "calibration",
    "paired_metrics",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty merged table: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def merge_seed_directories(
    input_dir: Path, output_dir: Path, seeds: list[int]
) -> dict[str, Any]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be nonempty and unique")
    tables = {name: [] for name in TABLES}
    metadata = []
    inputs = {}
    for seed in seeds:
        directory = input_dir / f"seed{seed}"
        meta_path = directory / "metadata.json"
        meta = json.loads(meta_path.read_text())
        if not meta.get("complete") or meta["config"]["seeds"] != [seed]:
            raise ValueError(f"seed directory has incompatible metadata: {directory}")
        metadata.append(meta)
        inputs[str(seed)] = {
            "metadata": sha256_file(meta_path),
            "tables": {},
        }
        for name in TABLES:
            path = directory / f"{name}.csv"
            rows = read_csv(path)
            if any(int(row["seed"]) != seed for row in rows):
                raise ValueError(f"table contains another seed: {path}")
            tables[name].extend(rows)
            inputs[str(seed)]["tables"][name] = sha256_file(path)
    revisions = {meta["git_revision"] for meta in metadata}
    if len(revisions) != 1:
        raise ValueError(f"seed runs used multiple revisions: {sorted(revisions)}")
    normalized_configs = []
    for meta in metadata:
        normalized = deepcopy(meta["config"])
        normalized["seeds"] = []
        normalized_configs.append(normalized)
    if any(config != normalized_configs[0] for config in normalized_configs[1:]):
        raise ValueError("seed runs used different normalized configurations")
    if len({float(meta["crossover_weight"]) for meta in metadata}) != 1:
        raise ValueError("seed runs used different two-direction reference weights")
    parameterizations = [meta.get("weight_parameterization") for meta in metadata]
    if any(value != parameterizations[0] for value in parameterizations[1:]):
        raise ValueError("seed runs used different task-weight parameterizations")

    output_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        write_csv(output_dir / f"{name}.csv", rows)
    config = deepcopy(metadata[0]["config"])
    config["seeds"] = seeds
    merged = {
        "complete": True,
        "experiment": "exp02_prior_weight_sweep",
        "config": config,
        "git_revision": revisions.pop(),
        "crossover_weight": metadata[0]["crossover_weight"],
        "weight_parameterization": parameterizations[0],
        "seed_count": len(seeds),
        "paired_rows": len(tables["paired_metrics"]),
        "sum_seed_elapsed_seconds": sum(meta["elapsed_seconds"] for meta in metadata),
        "maximum_seed_elapsed_seconds": max(
            meta["elapsed_seconds"] for meta in metadata
        ),
        "seed_inputs": inputs,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(merged, indent=2, sort_keys=True) + "\n"
    )
    return merged


def main() -> None:
    args = parse_args()
    merged = merge_seed_directories(args.input_dir, args.output_dir, args.seeds)
    print(json.dumps(merged, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
