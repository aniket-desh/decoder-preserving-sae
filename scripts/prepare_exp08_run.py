#!/usr/bin/env python3
"""Create or validate the immutable contract for the Exp08 closure queue."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
CODE_PATHS = (
    "configs/exp02_structured_prior.json",
    "configs/exp04b_confirmatory.json",
    "configs/paper_closure.json",
    "configs/exp08_task_spectrum.json",
    "experiments/exp02_structured_prior.py",
    "experiments/exp02_prior_weight_sweep.py",
    "experiments/exp04b_confirmatory.py",
    "experiments/exp08_language_evidence.py",
    "experiments/paper_closure.py",
    "scripts/launch_exp08_runpod.sh",
    "scripts/finalize_exp08_candidates.sh",
    "scripts/merge_exp02_prior_sweep.py",
    "scripts/plot_exp08_candidates.py",
    "scripts/prepare_exp08_run.py",
    "scripts/run_exp08_gpu_runpod.sh",
    "scripts/run_exp08_synthetic_runpod.sh",
    "scripts/status_exp08_runpod.sh",
    "scripts/summarize_exp08_compute.py",
    "scripts/summarize_exp08_confirmation.py",
    "src/dpsae/language_training.py",
    "src/dpsae/plot_style.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def repository_state(path: Path) -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=path, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=path, text=True
    ).splitlines()
    return {"revision": revision, "dirty": bool(status), "status": status}


def external_paths(source_root: Path) -> dict[str, Path]:
    return {
        "source_tokens": source_root
        / "artifacts/exp04_ioi_mechanism/fineweb_gpt2_tokens.bin",
        "activation_calibration": source_root
        / "artifacts/exp04_ioi_mechanism/calibration.pt",
        "tail_180m_tokens": source_root
        / "artifacts/exp04b_confirmatory/fineweb_gpt2_tail_tokens.bin",
        "tail_180m_metadata": source_root
        / "artifacts/exp04b_confirmatory/fineweb_gpt2_tail_tokens.bin.json",
        "tail_190m_tokens": source_root
        / "artifacts/paper_closure/fineweb_gpt2_tail_tokens.bin",
        "tail_190m_metadata": source_root
        / "artifacts/paper_closure/fineweb_gpt2_tail_tokens.bin.json",
        "static_calibration": source_root
        / "artifacts/exp04b_confirmatory/static_calibration.pt",
        "static_baseline_evaluation": source_root
        / "artifacts/exp04b_confirmatory/natural_evaluation_baseline.json",
        "structured_baseline_metrics": source_root
        / "experiments/outputs/exp02_structured_prior/metrics.csv",
        "structured_baseline_group_metrics": source_root
        / "experiments/outputs/exp02_structured_prior/group_metrics.csv",
        "structured_baseline_metadata": source_root
        / "experiments/outputs/exp02_structured_prior/metadata.json",
        "structured_baseline_crossover": source_root
        / "experiments/outputs/exp02_structured_prior/crossover.csv",
    }


def build_contract(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    if source_root == ROOT.resolve():
        raise ValueError("source artifact tree must differ from the clean worktree")
    repository = repository_state(ROOT)
    if repository["dirty"]:
        raise RuntimeError(f"Exp08 requires a clean repository: {repository['status']}")
    code = {relative: file_record(ROOT / relative) for relative in CODE_PATHS}
    external = {
        name: file_record(path) for name, path in external_paths(source_root).items()
    }
    contract = {
        "experiment": "exp08_experiment_figure_closure",
        "repository": repository,
        "source_artifact_tree": {
            "root": str(source_root),
            "repository": repository_state(source_root),
            "used_for_code": False,
        },
        "code": code,
        "external_inputs": external,
        "protocol_intervals": {
            "gamma_training": [10_000_000, 50_000_000],
            "confirmation_training": [50_000_000, 120_000_000],
            "gamma_selection": [180_000_000, 185_000_000],
            "confirmation_evaluation": [190_000_000, 195_000_000],
            "frozen_evaluation": [195_000_000, 200_000_000],
        },
        "platform": platform.platform(),
    }
    encoded = json.dumps(contract, sort_keys=True, separators=(",", ":")).encode()
    return {**contract, "contract_sha256": hashlib.sha256(encoded).hexdigest()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    contract = build_contract(args.source_root)
    if args.output.exists():
        existing = json.loads(args.output.read_text())
        comparable = {
            key: value
            for key, value in existing.items()
            if key not in {"created_unix", "complete"}
        }
        if comparable != contract:
            raise RuntimeError(
                "existing Exp08 run manifest differs from the current contract"
            )
        print(args.output)
        return
    payload = {"complete": True, "created_unix": time.time(), **contract}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
