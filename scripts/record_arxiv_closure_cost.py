#!/usr/bin/env python3
"""Write a conservative, machine-readable cost ledger for the closure pod."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path


def _utc(value: float) -> str:
    return datetime.fromtimestamp(value, tz=timezone.utc).isoformat()


def build_ledger(
    *,
    retained_start: Path,
    hourly_rate_usd: float,
    gpu_count: int,
    api_spend_usd: float,
    end_timestamp: float | None = None,
) -> dict:
    start_timestamp = retained_start.stat().st_mtime
    end_timestamp = datetime.now(timezone.utc).timestamp() if end_timestamp is None else end_timestamp
    if end_timestamp < start_timestamp:
        raise ValueError("cost-ledger end precedes retained start")
    if hourly_rate_usd < 0 or gpu_count <= 0 or api_spend_usd < 0:
        raise ValueError("cost-ledger rates and counts must be nonnegative")
    elapsed_hours = (end_timestamp - start_timestamp) / 3600.0
    pod_charge = elapsed_hours * hourly_rate_usd
    return {
        "schema_version": 1,
        "complete": True,
        "estimation_kind": "retained-window lower bound",
        "retained_start_source": str(retained_start.resolve()),
        "retained_start_utc": _utc(start_timestamp),
        "ledger_end_utc": _utc(end_timestamp),
        "elapsed_pod_hours": elapsed_hours,
        "allocated_gpu_count": gpu_count,
        "allocated_a40_gpu_hours": elapsed_hours * gpu_count,
        "pod_hourly_rate_usd": hourly_rate_usd,
        "estimated_pod_charge_usd": pod_charge,
        "openai_api_spend_usd": api_spend_usd,
        "estimated_total_excluding_storage_and_tax_usd": pod_charge + api_spend_usd,
        "exclusions": [
            "pod time before the earliest retained setup record",
            "network-volume storage charges",
            "taxes and provider-side billing adjustments",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retained-start", type=Path, required=True)
    parser.add_argument("--hourly-rate-usd", type=float, required=True)
    parser.add_argument("--gpu-count", type=int, required=True)
    parser.add_argument("--api-spend-usd", type=float, default=0.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    ledger = build_ledger(
        retained_start=args.retained_start,
        hourly_rate_usd=args.hourly_rate_usd,
        gpu_count=args.gpu_count,
        api_spend_usd=args.api_spend_usd,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(ledger, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, args.output)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
