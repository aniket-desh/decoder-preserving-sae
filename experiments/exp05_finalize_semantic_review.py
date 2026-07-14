#!/usr/bin/env python3
"""Bind a completed zero-hypothesis semantic review and freeze its registry.

This command reads only the open semantic-recurrence artifact and search log.
It does not load a natural-text cache or access the sealed final range.
"""

from __future__ import annotations

import argparse
import copy
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from experiments import exp05_decoder_advantage_discovery as discovery


ROOT = Path(__file__).resolve().parents[1]
AUDIT_SCHEMA_VERSION = 1


def _cluster_index(artifact: Mapping[str, Any]) -> dict[tuple[str, str], dict]:
    return {
        (row["side"], row["feature"]): dict(row)
        for row in artifact["all_feature_clusters"]
    }


def _rejection_note(
    mode: Mapping[str, Any],
    clusters: Mapping[tuple[str, str], Mapping[str, Any]],
) -> str:
    prefix = (
        "Reviewed positive/negative extreme tokens; full contexts were inspected "
        "for every lexical outlier and every mode beating the row-shuffle control. "
    )
    qualifying = mode["qualifying_features"]
    if qualifying:
        details = []
        alignments = {row["feature"]: row for row in mode["top_alignments"]}
        for feature in qualifying:
            alignment = alignments[feature]
            cluster = clusters[(mode["side"], feature)]
            details.append(
                f"{feature} cosine={alignment['cosine']:.3f}, "
                f"familywise-p={alignment['familywise_permutation_p']:.3f}, "
                f"seeds={len(cluster['seeds'])}, groups={len(cluster['group_positions'])}"
            )
        evidence = "Only singleton syntactic/positional alignment: " + "; ".join(details)
    else:
        best = mode["top_alignments"][0]
        evidence = (
            f"No feature survived the within-mode max-statistic; best={best['feature']} "
            f"cosine={best['cosine']:.3f}, "
            f"familywise-p={best['familywise_permutation_p']:.3f}"
        )
    return (
        prefix
        + evidence
        + f". Row-shuffle absolute ratio={mode['row_shuffle_absolute_ratio']:.3f}. "
        + "No same-side interpretation recurred across 2 seeds and 4 groups; rejected."
    )


def finalize_zero_hypothesis_review(
    *,
    recurrence_path: Path,
    registry_path: Path,
    audit_path: Path,
) -> dict[str, Any]:
    artifact = json.loads(recurrence_path.read_text())
    registry = json.loads(registry_path.read_text())
    if registry.get("status") != "open":
        raise RuntimeError("zero-hypothesis review requires an open registry")
    summary = artifact.get("summary", {})
    if summary.get("modes_reviewed") != discovery.EXPECTED_MODE_COUNT:
        raise RuntimeError("semantic recurrence artifact did not review every mode")
    if summary.get("qualified_recurrence_clusters") != 0:
        raise RuntimeError("a recurring semantic candidate prevents zero-hypothesis freeze")
    modes = artifact.get("modes", [])
    if any(mode.get("recurring_features") for mode in modes):
        raise RuntimeError("a mode retains a recurring semantic feature")
    mode_ids = {mode["mode_id"] for mode in modes}
    if set(registry.get("mode_dispositions", {})) != mode_ids:
        raise RuntimeError("semantic audit and registry mode IDs differ")
    source_registry = artifact.get("source_artifacts", {}).get("registry", {})
    if source_registry.get("sha256") != discovery.file_sha256(registry_path):
        raise RuntimeError("registry changed after semantic recurrence analysis")

    clusters = _cluster_index(artifact)
    full_context_ids = sorted(
        mode["mode_id"]
        for mode in modes
        if mode["qualifying_features"] or mode["row_shuffle_absolute_ratio"] < 1
    )
    reviews = {
        mode["mode_id"]: {
            "status": "rejected",
            "note": _rejection_note(mode, clusters),
            "compact_extremes_inspected": True,
            "full_context_inspected": mode["mode_id"] in full_context_ids,
        }
        for mode in modes
    }
    audit = {
        "schema_version": AUDIT_SCHEMA_VERSION,
        "reviewed_at_utc": datetime.now(timezone.utc).isoformat(),
        "verdict": "freeze_zero_hypotheses",
        "source_semantic_recurrence": {
            "path": str(recurrence_path.resolve()),
            "sha256": discovery.file_sha256(recurrence_path),
        },
        "method": {
            "all_extreme_token_sets_manually_inspected": True,
            "mode_count": len(modes),
            "full_context_selection_rule": (
                "all familywise lexical outliers plus every mode with "
                "row-shuffle absolute ratio below one"
            ),
            "full_context_mode_ids": full_context_ids,
            "full_context_mode_count": len(full_context_ids),
        },
        "summary": copy.deepcopy(summary),
        "mode_reviews": reviews,
    }
    discovery.atomic_json(audit_path, audit)

    updated = copy.deepcopy(registry)
    updated["hypotheses"] = []
    updated["semantic_review"] = {
        "path": str(audit_path.relative_to(registry_path.parent)),
        "sha256": discovery.file_sha256(audit_path),
        "verdict": audit["verdict"],
    }
    for mode_id, review in reviews.items():
        updated["mode_dispositions"][mode_id] = {
            "status": "rejected",
            "hypothesis_id": None,
            "note": review["note"],
        }
    discovery.atomic_json(registry_path, updated)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    output = ROOT / "artifacts/exp05_decoder_advantage_discovery"
    parser.add_argument(
        "--recurrence",
        type=Path,
        default=output / "semantic_recurrence.json",
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=output / "hypothesis_registry.json",
    )
    parser.add_argument(
        "--audit",
        type=Path,
        default=output / "semantic_manual_audit.json",
    )
    parser.add_argument(
        "--confirm-all-192-extreme-sets-reviewed",
        action="store_true",
        help="required acknowledgement of the manual review completed outside this script",
    )
    args = parser.parse_args()
    if not args.confirm_all_192_extreme_sets_reviewed:
        raise SystemExit("refusing to finalize without explicit manual-review confirmation")
    audit = finalize_zero_hypothesis_review(
        recurrence_path=args.recurrence,
        registry_path=args.registry,
        audit_path=args.audit,
    )
    frozen = discovery.freeze_hypothesis_registry(registry_path=args.registry)
    print(
        json.dumps(
            {
                "verdict": audit["verdict"],
                "mode_count": audit["method"]["mode_count"],
                "full_context_mode_count": audit["method"]["full_context_mode_count"],
                "registry_status": frozen["status"],
                "hypothesis_count": len(frozen["hypotheses"]),
                "frozen_digest": frozen["frozen_digest"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
