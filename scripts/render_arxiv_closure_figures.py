#!/usr/bin/env python3
"""Render only post-audit closure payloads, or print the result-free contract."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for path in (ROOT, ROOT / "src"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dpsae.closure_plots import (  # noqa: E402
    PLOT_PAYLOAD_CONTRACT,
    plot_concept_ladder,
    plot_frozen_network_noninferiority,
    plot_static_nmse_control,
    validate_payload,
    write_summary_table,
)
from dpsae.release_manifest import (  # noqa: E402
    atomic_json,
    canonical_digest,
    file_record,
    sha256_stable_file,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--write-contract", type=Path)
    parser.add_argument("--payload", type=Path)
    parser.add_argument("--payload-manifest", type=Path)
    parser.add_argument("--release-manifest", type=Path)
    parser.add_argument("--output-dir", type=Path)
    return parser.parse_args()


def _validate_release_manifest(path: Path) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("inventory_complete") is not True:
        raise ValueError("closure figures require a complete release inventory")
    digest = manifest.get("manifest_sha256")
    observed = canonical_digest(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )
    if digest != observed:
        raise ValueError("closure release-manifest self digest mismatch")
    return manifest


def _validate_payload_manifest(path: Path, payload_path: Path, release: dict) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 1 or manifest.get("complete") is not True:
        raise ValueError("closure payload build manifest is incomplete")
    digest = manifest.get("build_manifest_sha256")
    observed = canonical_digest(
        {key: value for key, value in manifest.items() if key != "build_manifest_sha256"}
    )
    if digest != observed:
        raise ValueError("closure payload build-manifest self digest mismatch")
    if manifest.get("core_release_manifest_sha256") != release.get("manifest_sha256"):
        raise ValueError("closure payload build manifest names another core release")
    outputs = manifest.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError("closure payload build manifest has no output records")
    records = [row for row in outputs if row.get("role") == "closure_payload"]
    if len(records) != 1:
        raise ValueError("closure payload build manifest must identify one payload output")
    record = records[0]
    if Path(record.get("path", "")).resolve() != payload_path:
        raise ValueError("closure payload path disagrees with its build manifest")
    if (
        record.get("bytes") != payload_path.stat().st_size
        or record.get("sha256") != sha256_stable_file(payload_path)
    ):
        raise ValueError("closure payload changed after its build manifest was written")
    return manifest


def main() -> None:
    args = parse_args()
    if args.write_contract:
        if any(
            (args.payload, args.payload_manifest, args.release_manifest, args.output_dir)
        ):
            raise ValueError("--write-contract cannot be combined with result inputs")
        atomic_json(args.write_contract.resolve(), PLOT_PAYLOAD_CONTRACT)
        print(args.write_contract.resolve())
        return
    if not all(
        (args.payload, args.payload_manifest, args.release_manifest, args.output_dir)
    ):
        raise ValueError(
            "rendering requires payload, payload manifest, release manifest, and output directory"
        )

    payload_path = args.payload.resolve()
    payload_manifest_path = args.payload_manifest.resolve()
    release_path = args.release_manifest.resolve()
    output_dir = args.output_dir.resolve()
    payload = json.loads(payload_path.read_text(encoding="utf-8"))
    validate_payload(payload)
    release = _validate_release_manifest(release_path)
    _validate_payload_manifest(payload_manifest_path, payload_path, release)
    if payload["release_manifest_sha256"] != release["manifest_sha256"]:
        raise ValueError("plotting payload names another release manifest")
    output_dir.mkdir(parents=True, exist_ok=True)

    renderers = {
        "concept_ladder": plot_concept_ladder,
        "frozen_network_noninferiority": plot_frozen_network_noninferiority,
        "static_nmse_control": plot_static_nmse_control,
    }
    figures = {}
    omitted = []
    for name, renderer in renderers.items():
        block = payload.get("figures", {}).get(name, {"available": False})
        if block.get("available") is not True:
            omitted.append(name)
            continue
        pdf, png = renderer(block, output_dir / name)
        figures[name] = {
            "pdf": file_record(pdf, output_dir),
            "png": file_record(png, output_dir),
        }
    table = payload["summary_table"]
    table_record = None
    if table["available"]:
        table_path = write_summary_table(table, output_dir / "closure_summary.csv")
        table_record = file_record(table_path, output_dir)

    manifest = {
        "schema_version": 1,
        "experiment": "arxiv_closure_candidate_outputs",
        "complete": True,
        "release_manifest": {
            "path": str(release_path),
            "sha256": sha256_stable_file(release_path),
            "manifest_sha256": release["manifest_sha256"],
        },
        "plot_payload": {
            "path": str(payload_path),
            "sha256": sha256_stable_file(payload_path),
        },
        "plot_payload_manifest": file_record(payload_manifest_path, payload_manifest_path.parent),
        "renderer": file_record(ROOT / "src/dpsae/closure_plots.py", ROOT),
        "plot_style": file_record(ROOT / "src/dpsae/plot_style.py", ROOT),
        "figures": figures,
        "omitted_unavailable_figures": omitted,
        "summary_table": table_record,
    }
    atomic_json(output_dir / "candidate_output_manifest.json", manifest)
    print(output_dir / "candidate_output_manifest.json")


if __name__ == "__main__":
    main()
