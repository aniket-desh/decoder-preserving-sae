#!/usr/bin/env python3
"""Write a hash-complete environment and artifact manifest for paper closure."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import socket
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import torch


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOTS = (
    "artifacts/paper_closure",
    "artifacts/exp04b_mechanism_attribution",
    "artifacts/exp05_decoder_advantage_discovery",
    "artifacts/exp06_generality",
)
CODE_PATHS = (
    "configs/paper_closure.json",
    "configs/exp04b_confirmatory.json",
    "experiments/exp04b_confirmatory.py",
    "experiments/exp04b_mechanism_attribution.py",
    "experiments/exp05_decoder_advantage_discovery.py",
    "experiments/exp05_semantic_recurrence.py",
    "experiments/exp05_finalize_semantic_review.py",
    "experiments/exp06_generality.py",
    "experiments/exp07_jumprelu_calibration.py",
    "experiments/paper_closure.py",
    "scripts/finalize_paper_closure.py",
    "src/dpsae/language_model.py",
    "src/dpsae/language_sae.py",
    "src/dpsae/language_training.py",
    "src/dpsae/mech_analysis.py",
)
EXCLUDED_NAMES = {"checkpoint.pt"}


def sha256_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def artifact_files(paths: Iterable[Path]) -> list[Path]:
    files = []
    for root in paths:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if (
                path.is_file()
                and path.name not in EXCLUDED_NAMES
                and not path.name.endswith(".tmp")
                and ".hf_backup" not in path.parts
            ):
                files.append(path)
    return sorted(set(files))


def write_code_bundle(root: Path, output: Path) -> None:
    """Persist the exact dirty-tree code inputs alongside the result artifacts."""

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")

    def normalized(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.mtime = 0
        info.uid = 0
        info.gid = 0
        info.uname = ""
        info.gname = ""
        return info

    with tarfile.open(temporary, "w") as archive:
        for relative in CODE_PATHS:
            path = root / relative
            if not path.is_file():
                raise FileNotFoundError(path)
            archive.add(path, arcname=relative, filter=normalized)
        lock = root / "uv.lock"
        if lock.is_file():
            archive.add(lock, arcname="uv.lock", filter=normalized)
    temporary.replace(output)


def _command(args: list[str], *, cwd: Path) -> str | None:
    try:
        return subprocess.check_output(args, cwd=cwd, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def repository_record(root: Path) -> dict[str, Any]:
    status = _command(["git", "status", "--porcelain"], cwd=root)
    return {
        "revision": _command(["git", "rev-parse", "HEAD"], cwd=root),
        "status": [] if not status else status.splitlines(),
        "dirty": bool(status),
    }


def environment_record() -> dict[str, Any]:
    result: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": importlib.metadata.version("transformers"),
        "datasets": importlib.metadata.version("datasets"),
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "argv": sys.argv,
        "safe_environment": {
            key: os.environ.get(key)
            for key in (
                "HF_HOME",
                "PYTHONPATH",
                "TOKENIZERS_PARALLELISM",
                "PYTHONDONTWRITEBYTECODE",
            )
        },
    }
    driver = _command(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        cwd=ROOT,
    )
    result["nvidia_driver"] = driver
    if torch.cuda.is_available():
        devices = []
        for index in range(torch.cuda.device_count()):
            properties = torch.cuda.get_device_properties(index)
            devices.append(
                {
                    "index": index,
                    "name": properties.name,
                    "total_memory_bytes": properties.total_memory,
                    "compute_capability": [properties.major, properties.minor],
                }
            )
        result["cuda_devices"] = devices
    else:
        result["cuda_devices"] = []
    return result


def build_manifest(root: Path, artifact_roots: Iterable[Path]) -> dict[str, Any]:
    code = []
    for relative in CODE_PATHS:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        code.append(file_record(path, root))
    lock = root / "uv.lock"
    return {
        "schema_version": 1,
        "experiment": "paper_closure",
        "complete": True,
        "repository": repository_record(root),
        "environment": environment_record(),
        "dependency_lock": file_record(lock, root) if lock.is_file() else None,
        "code": code,
        "artifacts": [
            file_record(path, root) for path in artifact_files(artifact_roots)
        ],
        "excluded": {
            "names": sorted(EXCLUDED_NAMES),
            "reason": "rolling optimizer checkpoints are not release artifacts",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/paper_closure/reproducibility_manifest.json",
    )
    parser.add_argument("--artifact-root", action="append", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    roots = (
        [path.resolve() for path in args.artifact_root]
        if args.artifact_root
        else [root / relative for relative in ARTIFACT_ROOTS]
    )
    write_code_bundle(root, args.output.parent / "code_bundle.tar")
    manifest = build_manifest(root, roots)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
