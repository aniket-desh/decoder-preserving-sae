#!/usr/bin/env python3
"""Poll one submitted exp10 Batch and durably validate its downloaded output."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dpsae.exp10_autointerp import poll_and_download_batch  # noqa: E402
from experiments.exp10_concept_discovery import DEFAULT_CONFIG, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--requests", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--batch-output", type=Path, required=True)
    parser.add_argument("--poll-seconds", type=float, default=60)
    parser.add_argument("--timeout-hours", type=float, default=26)
    parser.add_argument("--max-consecutive-poll-errors", type=int, default=5)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    result = poll_and_download_batch(
        config=load_config(args.config),
        manifest_path=args.manifest,
        request_path=args.requests,
        mapping_path=args.mapping,
        state_path=args.state,
        batch_output_path=args.batch_output,
        poll_seconds=args.poll_seconds,
        timeout_seconds=args.timeout_hours * 3600,
        max_consecutive_poll_errors=args.max_consecutive_poll_errors,
        wait=not args.once,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
