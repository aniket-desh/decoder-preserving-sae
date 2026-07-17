#!/usr/bin/env python3
"""Build or audit the result-blind arXiv experiment release manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dpsae.release_manifest import (  # noqa: E402
    atomic_json,
    audit_manifest,
    build_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("build", "audit"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument(
            "--policy", type=Path, default=ROOT / "configs/arxiv_release_closure.json"
        )
        subparser.add_argument("--repository-root", type=Path, default=ROOT)
        subparser.add_argument("--run-root", type=Path, required=True)
        subparser.add_argument("--manifest", type=Path, required=True)
    subparsers.choices["build"].add_argument(
        "--allow-dirty-repository", action="store_true"
    )
    subparsers.choices["audit"].add_argument("--report", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_path = args.manifest.resolve()
    if args.command == "build":
        manifest = build_manifest(
            policy_path=args.policy,
            repository_root=args.repository_root,
            run_root=args.run_root,
            require_clean_repository=not args.allow_dirty_repository,
        )
        for group in manifest["artifact_groups"]:
            if group["present"]:
                root = (
                    args.repository_root
                    if group["anchor"] == "repository"
                    else args.run_root
                ) / group["configured_path"]
                try:
                    manifest_path.relative_to(root.resolve())
                except ValueError:
                    continue
                raise ValueError("release manifest may not be written inside a scanned root")
        atomic_json(manifest_path, manifest)
        print(manifest_path)
        return

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = audit_manifest(
        manifest,
        policy_path=args.policy,
        repository_root=args.repository_root,
        run_root=args.run_root,
    )
    if args.report:
        atomic_json(args.report.resolve(), report)
        print(args.report.resolve())
    else:
        print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
