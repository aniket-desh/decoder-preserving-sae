#!/usr/bin/env python3
"""Deterministically mine blinded discovery and held-out feature contexts.

The input is a versioned JSONL activation scan. Each row represents one
candidate/context pair and contains only text, split, and the candidate's
maximum activation in that context. Producing the activation scan is a pure
model-inference stage; this selector is deliberately independent of labels and
benchmark outcomes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.exp10_concept_discovery import (  # noqa: E402
    DEFAULT_CONFIG,
    atomic_json,
    atomic_jsonl,
    canonical_digest,
    file_sha256,
    load_config,
    read_json,
)


WORD = re.compile(r"[A-Za-z0-9_]+")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        values.append(value)
    return values


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def tokens(record: Mapping[str, Any]) -> set[str]:
    explicit = record.get("tokens")
    if isinstance(explicit, list) and all(isinstance(value, str) for value in explicit):
        return {value.casefold() for value in explicit if value.strip()}
    return {value.casefold() for value in WORD.findall(record["text"])}


def _context_view(record: Mapping[str, Any]) -> dict[str, Any]:
    value = {
        "context_id": record["context_id"],
        "text": record["text"],
        "text_sha256": text_hash(record["text"]),
        "activation": float(record["activation"]),
    }
    for key in ("tokens", "active_token_index", "active_token"):
        if key in record:
            value[key] = record[key]
    return value


def _take_evenly(records: list[dict[str, Any]], count: int) -> list[dict[str, Any]]:
    if count == 0:
        return []
    if len(records) < count:
        raise ValueError(f"need {count} contexts, observed {len(records)}")
    if count == 1:
        return [records[0]]
    indices = []
    used = set()
    for raw in (round(index * (len(records) - 1) / (count - 1)) for index in range(count)):
        candidate = int(raw)
        while candidate in used and candidate + 1 < len(records):
            candidate += 1
        while candidate in used and candidate > 0:
            candidate -= 1
        if candidate in used:
            raise ValueError("could not select unique stratified contexts")
        used.add(candidate)
        indices.append(candidate)
    return [records[index] for index in indices]


def _take_two_disjoint(
    records: list[dict[str, Any]], count: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    primary = _take_evenly(records, count)
    selected = {record["context_id"] for record in primary}
    remaining = [record for record in records if record["context_id"] not in selected]
    relabel = _take_evenly(remaining, count)
    return primary, relabel


def _near_miss_order(
    negatives: list[dict[str, Any]], positives: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    positive_tokens = set().union(*(tokens(record) for record in positives))

    def score(record: Mapping[str, Any]):
        current = tokens(record)
        overlap = len(current.intersection(positive_tokens)) / max(
            1, len(current.union(positive_tokens))
        )
        return (-overlap, -float(record["activation"]), str(record["context_id"]))

    return sorted(negatives, key=score)


def select_candidate_contexts(
    records: list[dict[str, Any]], config: Mapping[str, Any]
) -> dict[str, Any]:
    selection = config["context_mining"]
    discovery_name = selection["discovery_split"]
    heldout_name = selection["heldout_split"]
    threshold = float(selection["positive_activation_threshold"])
    discovery = [record for record in records if record["split"] == discovery_name]
    heldout = [record for record in records if record["split"] == heldout_name]
    positives = sorted(
        [record for record in discovery if float(record["activation"]) > threshold],
        key=lambda record: (-float(record["activation"]), str(record["context_id"])),
    )
    negatives = [record for record in discovery if float(record["activation"]) <= threshold]
    if not positives or not negatives:
        raise ValueError("discovery pool must contain active and inactive contexts")
    high_cut = max(1, math.ceil(0.4 * len(positives)))
    high_pool = positives[:high_cut]
    middle_pool = positives[high_cut:]
    high_primary, high_relabel = _take_two_disjoint(
        high_pool, int(selection["discovery_high"])
    )
    middle_primary, middle_relabel = _take_two_disjoint(
        middle_pool, int(selection["discovery_middle"])
    )
    ordered_negatives = _near_miss_order(negatives, high_primary + middle_primary)
    negative_primary, negative_relabel = _take_two_disjoint(
        ordered_negatives, int(selection["discovery_near_miss_negative"])
    )

    heldout_positive_pool = sorted(
        [record for record in heldout if float(record["activation"]) > threshold],
        key=lambda record: (-float(record["activation"]), str(record["context_id"])),
    )
    heldout_negative_pool = _near_miss_order(
        [record for record in heldout if float(record["activation"]) <= threshold],
        high_primary + middle_primary,
    )
    heldout_positive = _take_evenly(
        heldout_positive_pool, int(selection["heldout_positive"])
    )
    heldout_negative = _take_evenly(
        heldout_negative_pool, int(selection["heldout_negative"])
    )
    groups = {
        "discovery_primary": {
            "high": high_primary,
            "middle": middle_primary,
            "near_miss_negative": negative_primary,
        },
        "discovery_relabel": {
            "high": high_relabel,
            "middle": middle_relabel,
            "near_miss_negative": negative_relabel,
        },
        "heldout": {
            "positive": heldout_positive,
            "negative": heldout_negative,
        },
    }
    flat_ids = [
        record["context_id"]
        for group in groups.values()
        for records_in_kind in group.values()
        for record in records_in_kind
    ]
    if len(flat_ids) != len(set(flat_ids)):
        raise RuntimeError("selected context groups are not disjoint")
    return {
        name: {kind: [_context_view(record) for record in values] for kind, values in group.items()}
        for name, group in groups.items()
    }


def mine_contexts(
    *,
    config: Mapping[str, Any],
    candidate_manifest_path: Path,
    candidates_path: Path,
    activation_jsonl: Path,
    output_root: Path,
) -> dict[str, Any]:
    manifest = read_json(candidate_manifest_path)
    confirmation = manifest.get("confirmation_gate", {})
    if confirmation.get("passed") is not True or manifest.get("autointerp_eligible") is not True:
        raise RuntimeError("context mining requires a passed fresh-confirmation gate")
    if manifest.get("candidate_jsonl_sha256") != file_sha256(candidates_path):
        raise RuntimeError("candidate JSONL changed after confirmation")
    candidates = read_jsonl(candidates_path)
    candidate_ids = [record.get("candidate_id") for record in candidates]
    if any(not isinstance(value, str) or not value for value in candidate_ids):
        raise ValueError("every candidate requires a stable candidate_id")
    if len(candidate_ids) != len(set(candidate_ids)):
        raise ValueError("candidate IDs are duplicated")
    if any(record.get("autointerp_eligible") is not True for record in candidates):
        raise RuntimeError("candidate file includes non-confirmed associations")

    activation_rows = read_jsonl(activation_jsonl)
    grouped: dict[str, list[dict[str, Any]]] = {value: [] for value in candidate_ids}
    seen_pairs = set()
    split_by_context: dict[str, str] = {}
    split_by_text_hash: dict[str, str] = {}
    allowed_splits = {
        config["context_mining"]["discovery_split"],
        config["context_mining"]["heldout_split"],
    }
    for row in activation_rows:
        required = {"candidate_id", "context_id", "split", "text", "activation"}
        if not required.issubset(row):
            raise ValueError(f"activation row missing fields: {sorted(required.difference(row))}")
        candidate_id = row["candidate_id"]
        if candidate_id not in grouped:
            raise ValueError(f"activation scan has unknown candidate {candidate_id}")
        if row["split"] not in allowed_splits:
            raise ValueError(f"unknown context split {row['split']!r}")
        if not isinstance(row["text"], str) or not row["text"].strip():
            raise ValueError("context text must be a nonempty string")
        if not math.isfinite(float(row["activation"])):
            raise ValueError("context activation must be finite")
        pair = (candidate_id, row["context_id"])
        if pair in seen_pairs:
            raise ValueError(f"duplicate candidate/context pair: {pair}")
        seen_pairs.add(pair)
        context_id = str(row["context_id"])
        prior_split = split_by_context.setdefault(context_id, row["split"])
        if prior_split != row["split"]:
            raise RuntimeError(f"context ID {context_id} crosses discovery/heldout splits")
        digest = text_hash(row["text"])
        prior_text_split = split_by_text_hash.setdefault(digest, row["split"])
        if prior_text_split != row["split"]:
            raise RuntimeError("identical context text crosses discovery/heldout splits")
        grouped[candidate_id].append(row)

    selected = []
    for candidate_id in candidate_ids:
        if not grouped[candidate_id]:
            raise RuntimeError(f"activation scan has no rows for candidate {candidate_id}")
        selected.append(
            {
                "schema_version": 1,
                "candidate_id": candidate_id,
                "contexts": select_candidate_contexts(grouped[candidate_id], config),
            }
        )
    contexts_path = output_root / "candidate_contexts.jsonl"
    atomic_jsonl(contexts_path, selected)
    result = {
        "schema_version": 1,
        "complete": True,
        "config_digest": canonical_digest(config),
        "confirmation_gate": confirmation,
        "candidate_count": len(selected),
        "candidate_manifest_sha256": file_sha256(candidate_manifest_path),
        "candidate_jsonl_sha256": file_sha256(candidates_path),
        "activation_jsonl_sha256": file_sha256(activation_jsonl),
        "contexts_jsonl_sha256": file_sha256(contexts_path),
        "context_mining_source_sha256": file_sha256(Path(__file__).resolve()),
        "selection": dict(config["context_mining"]),
    }
    atomic_json(output_root / "context_manifest.json", result)
    return result


def _path(value: str) -> Path:
    return Path(value).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=_path, default=DEFAULT_CONFIG)
    parser.add_argument("--candidate-manifest", type=_path, required=True)
    parser.add_argument("--candidates", type=_path, required=True)
    parser.add_argument("--activation-jsonl", type=_path, required=True)
    parser.add_argument("--output-root", type=_path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    result = mine_contexts(
        config=config,
        candidate_manifest_path=args.candidate_manifest,
        candidates_path=args.candidates,
        activation_jsonl=args.activation_jsonl,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
