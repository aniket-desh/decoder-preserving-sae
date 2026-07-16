#!/usr/bin/env python3
"""Submit one preflighted exp10 Batch job without duplicate resubmission."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dpsae.exp10_autointerp import submit_batch  # noqa: E402
from experiments.exp10_concept_discovery import DEFAULT_CONFIG, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    args = parser.parse_args()
    result = submit_batch(
        config=load_config(args.config),
        manifest_path=args.manifest,
        request_path=args.requests,
        state_path=args.state,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
