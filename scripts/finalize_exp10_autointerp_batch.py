#!/usr/bin/env python3
"""Validate and merge downloaded exp10 Batch output JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dpsae.exp10_autointerp import finalize_batch  # noqa: E402
from experiments.exp10_concept_discovery import DEFAULT_CONFIG, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--batch-output", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = finalize_batch(
        config=load_config(args.config),
        manifest_path=args.manifest,
        request_path=args.requests,
        mapping_path=args.mapping,
        batch_output_path=args.batch_output,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
