#!/usr/bin/env python3
"""Audit the exact Exp13 worker matrix and retained scientific artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from experiments import exp13_concept_confirmation as exp13


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=exp13.DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=exp13.DEFAULT_OUTPUT)
    parser.add_argument(
        "--phase", choices=("pre-aggregate", "final"), required=True
    )
    parser.add_argument("--wait-seconds", type=float, default=0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config = exp13.load_config(args.config.expanduser().resolve())
    report = exp13.audit_artifacts(
        config=config,
        output_root=args.output_root.expanduser().resolve(),
        phase=args.phase,
        wait_seconds=args.wait_seconds,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
