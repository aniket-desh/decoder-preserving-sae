#!/usr/bin/env python3
"""Prepare a confirmation-gated exp10 OpenAI Batch JSONL file."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dpsae.exp10_autointerp import prepare_batch  # noqa: E402
from experiments.exp10_concept_discovery import DEFAULT_CONFIG, load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-manifest", type=Path, required=True)
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--context-manifest", type=Path, required=True)
    parser.add_argument("--contexts", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_batch(
        config=load_config(args.config),
        candidate_manifest_path=args.candidate_manifest,
        candidates_path=args.candidates,
        context_manifest_path=args.context_manifest,
        contexts_path=args.contexts,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
