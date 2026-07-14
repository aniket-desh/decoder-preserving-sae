#!/usr/bin/env python3
"""Lexical and context recurrence audit for the open Experiment 5 modes.

This analysis reads only the registered 180M--185M natural-selection cache.
It does not know a final-cache path and never changes the hypothesis registry.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import torch
from torch import Tensor

from experiments import exp05_decoder_advantage_discovery as discovery


ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ReviewProtocol:
    permutation_samples: int = 256
    minimum_feature_support: int = 4
    maximum_feature_support: int = 124
    minimum_absolute_cosine: float = 0.25
    familywise_alpha: float = 0.05
    minimum_recurrence_seeds: int = 2
    minimum_recurrence_groups: int = 4
    context_radius: int = 6
    reported_alignments: int = 5


DEFAULT_REVIEW = ReviewProtocol()
TOKENIZER_NAME = "openai-community/gpt2"


def _stable_seed(*parts: str) -> int:
    payload = ":".join(parts).encode()
    return int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "big") % (
        2**31 - 1
    )


def _clean(value: str) -> str:
    return " ".join(value.replace("\x00", " ").split())


def _surface(tokenizer, token_id: int) -> str:
    return tokenizer.decode([token_id], clean_up_tokenization_spaces=False)


def _normalized_token(surface: str) -> str:
    return surface.strip().casefold()


def _token_record(
    tokenizer,
    sequence: Tensor,
    *,
    position: int,
    absolute_offset: int,
    radius: int,
) -> dict[str, Any]:
    start = max(0, position - radius)
    stop = min(len(sequence), position + radius + 1)
    token_id = int(sequence[position])
    surface = _surface(tokenizer, token_id)
    context = tokenizer.decode(
        sequence[start:stop].tolist(), clean_up_tokenization_spaces=False
    )
    previous = "" if position == 0 else _surface(tokenizer, int(sequence[position - 1]))
    following = (
        "" if position + 1 == len(sequence) else _surface(tokenizer, int(sequence[position + 1]))
    )
    return {
        "token_id": token_id,
        "surface": surface,
        "normalized": _normalized_token(surface),
        "previous_normalized": _normalized_token(previous),
        "next_normalized": _normalized_token(following),
        "position": position,
        "absolute_offset": absolute_offset,
        "context": _clean(context),
    }


def _structural_features(record: Mapping[str, Any]) -> set[str]:
    surface = str(record["surface"])
    stripped = surface.strip()
    context = str(record["context"])
    features = set()
    if surface[:1].isspace():
        features.add("structure:leading_whitespace")
    if not stripped:
        features.add("structure:whitespace_only")
    if "\n" in surface:
        features.add("structure:newline")
    if stripped and all(not character.isalnum() for character in stripped):
        features.add("structure:punctuation")
    if any(character.isdigit() for character in stripped):
        features.add("structure:contains_digit")
    if stripped.isalpha():
        features.add("structure:alphabetic")
    if stripped[:1].isupper():
        features.add("structure:uppercase_initial")
    if any(ord(character) > 127 for character in surface):
        features.add("structure:non_ascii")
    if re.search(r"https?://|www\.|\.com\b", context, flags=re.IGNORECASE):
        features.add("context:url")
    if re.search(r"[{};]|\b(?:def|class|function|return|import)\b", context):
        features.add("context:code_like")
    if re.search(r"\d", context):
        features.add("context:numeric")
    if re.search(r"['\"“”‘’]", context):
        features.add("context:quotation")
    position = int(record["position"])
    if position < 8:
        features.add("position:first_8")
    if position >= 248:
        features.add("position:last_8")
    features.add(f"position:quartile_{min(position // 64, 3) + 1}")
    return features


def feature_sets(records: list[Mapping[str, Any]]) -> dict[str, set[int]]:
    values: dict[str, set[int]] = defaultdict(set)
    for index, record in enumerate(records):
        for feature in _structural_features(record):
            values[feature].add(index)
        for prefix, field in (
            ("token", "normalized"),
            ("previous", "previous_normalized"),
            ("next", "next_normalized"),
        ):
            normalized = str(record[field])
            if normalized:
                values[f"{prefix}:{normalized}"].add(index)
    return dict(values)


def _unit_feature(indices: set[int], size: int) -> Tensor:
    feature = torch.zeros(size, dtype=torch.float64)
    feature[list(indices)] = 1
    feature -= feature.mean()
    return feature / feature.norm().clamp_min(1e-12)


def mode_alignments(
    vector: Tensor,
    records: list[Mapping[str, Any]],
    *,
    mode_id: str,
    protocol: ReviewProtocol = DEFAULT_REVIEW,
) -> list[dict[str, Any]]:
    vectors, names, supports = [], [], []
    for name, indices in sorted(feature_sets(records).items()):
        support = len(indices)
        if not protocol.minimum_feature_support <= support <= protocol.maximum_feature_support:
            continue
        names.append(name)
        supports.append(support)
        vectors.append(_unit_feature(indices, len(records)))
    if not vectors:
        return []
    features = torch.stack(vectors)
    vector = vector.detach().cpu().double()
    observed = features @ vector
    generator = torch.Generator().manual_seed(_stable_seed("semantic", mode_id))
    permutations = torch.stack(
        [
            vector[torch.randperm(len(vector), generator=generator)]
            for _ in range(protocol.permutation_samples)
        ]
    )
    null_maximum = (permutations @ features.mT).abs().max(dim=1).values
    result = []
    for name, support, score in zip(names, supports, observed.tolist()):
        probability = float(
            (1 + (null_maximum >= abs(score)).sum())
            / (protocol.permutation_samples + 1)
        )
        result.append(
            {
                "feature": name,
                "support": support,
                "cosine": score,
                "absolute_cosine": abs(score),
                "familywise_permutation_p": probability,
                "qualifies": (
                    abs(score) >= protocol.minimum_absolute_cosine
                    and probability <= protocol.familywise_alpha
                ),
            }
        )
    return sorted(
        result,
        key=lambda row: (
            row["familywise_permutation_p"],
            -row["absolute_cosine"],
            row["feature"],
        ),
    )


def _records_for_group(
    tokenizer,
    selected_ids: Tensor,
    selected_starts: Tensor,
    group_indices: Tensor,
    *,
    radius: int,
) -> list[dict[str, Any]]:
    sequence_length = selected_ids.shape[1]
    records = []
    for flat_index in group_indices.tolist():
        row, position = divmod(int(flat_index), sequence_length)
        records.append(
            _token_record(
                tokenizer,
                selected_ids[row],
                position=position,
                absolute_offset=int(selected_starts[row]) + position,
                radius=radius,
            )
        )
    return records


def _extreme_records(
    records: list[Mapping[str, Any]], offsets: list[int]
) -> list[dict[str, Any]]:
    by_offset = {int(record["absolute_offset"]): record for record in records}
    if not set(offsets) <= set(by_offset):
        raise RuntimeError("searched extreme offset is absent from its registered group")
    return [dict(by_offset[offset]) for offset in offsets]


def _recurrence_records(
    tokenizer,
    cache: Mapping[str, Any],
    *,
    token_range: tuple[int, int],
    radius: int,
) -> list[dict[str, Any]]:
    records = []
    for sequence, start in zip(cache["input_ids"], cache["starts"]):
        start = int(start)
        if start < token_range[0] or start + len(sequence) > token_range[1]:
            continue
        for position in range(len(sequence)):
            records.append(
                _token_record(
                    tokenizer,
                    sequence,
                    position=position,
                    absolute_offset=start + position,
                    radius=radius,
                )
            )
    return records


def _feature_predicate(feature: str) -> Callable[[Mapping[str, Any]], bool]:
    if feature.startswith("token:"):
        value = feature.removeprefix("token:")
        return lambda record: record["normalized"] == value
    if feature.startswith("previous:"):
        value = feature.removeprefix("previous:")
        return lambda record: record["previous_normalized"] == value
    if feature.startswith("next:"):
        value = feature.removeprefix("next:")
        return lambda record: record["next_normalized"] == value
    return lambda record: feature in _structural_features(record)


def analyze(
    *,
    natural_selection: Path,
    manifest_path: Path,
    search_path: Path,
    registry_path: Path,
    tokenizer,
    output_path: Path,
    review_path: Path,
    protocol: ReviewProtocol = DEFAULT_REVIEW,
) -> dict[str, Any]:
    manifest = json.loads(manifest_path.read_text())
    search = json.loads(search_path.read_text())
    registry = json.loads(registry_path.read_text())
    discovery.validate_discovery_manifest(manifest)
    discovery.validate_search_log(search)
    if registry.get("status") != "open":
        raise RuntimeError("semantic recurrence analysis requires an open registry")
    if registry.get("search_log", {}).get("sha256") != discovery.file_sha256(search_path):
        raise RuntimeError("open registry is bound to another searched-mode log")
    cache = discovery.guarded_load_natural_cache(
        natural_selection,
        requested_range=discovery.DEFAULT_PROTOCOL.source_selection_range,
        protocol=discovery.DEFAULT_PROTOCOL,
    )
    if discovery.file_sha256(natural_selection) != manifest["source_cache"]["sha256"]:
        raise RuntimeError("natural-selection cache changed after discovery registration")

    rows = torch.tensor(manifest["selected_sequence_rows"], dtype=torch.long)
    selected_ids = cache["input_ids"][rows]
    selected_starts = cache["starts"][rows].long()
    group_indices = torch.tensor(manifest["group_indices"], dtype=torch.long)
    group_records = [
        _records_for_group(
            tokenizer,
            selected_ids,
            selected_starts,
            indices,
            radius=protocol.context_radius,
        )
        for indices in group_indices
    ]
    mode_rows = []
    for mode in search["modes"]:
        records = group_records[int(mode["group_slot"])]
        alignments = mode_alignments(
            torch.tensor(mode["eigentask"]),
            records,
            mode_id=mode["mode_id"],
            protocol=protocol,
        )
        eigenvalue = float(mode["eigenvalue"])
        row_shuffle = float(mode["controls"]["row_shuffle_rayleigh"])
        mode_rows.append(
            {
                "mode_id": mode["mode_id"],
                "seed": int(mode["seed"]),
                "group_slot": int(mode["group_slot"]),
                "group_position": int(mode["group_position"]),
                "side": mode["side"],
                "rank": int(mode["rank"]),
                "eigenvalue": eigenvalue,
                "row_shuffle_rayleigh": row_shuffle,
                "row_shuffle_absolute_ratio": abs(row_shuffle)
                / max(abs(eigenvalue), 1e-30),
                "random_rayleigh_percentile": float(
                    mode["controls"]["random_rayleigh_percentile"]
                ),
                "top_alignments": alignments[: protocol.reported_alignments],
                "qualifying_features": [
                    row["feature"] for row in alignments if row["qualifies"]
                ],
                "positive_extremes": _extreme_records(
                    records, mode["positive_extreme_absolute_offsets"]
                ),
                "negative_extremes": _extreme_records(
                    records, mode["negative_extreme_absolute_offsets"]
                ),
            }
        )

    clusters: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for mode in mode_rows:
        for feature in mode["qualifying_features"]:
            clusters[(mode["side"], feature)].append(mode)
    recurrence_rows = []
    for (side, feature), evidence in sorted(clusters.items()):
        seeds = sorted({row["seed"] for row in evidence})
        groups = sorted({row["group_position"] for row in evidence})
        qualifies = (
            len(seeds) >= protocol.minimum_recurrence_seeds
            and len(groups) >= protocol.minimum_recurrence_groups
        )
        recurrence_rows.append(
            {
                "side": side,
                "feature": feature,
                "mode_ids": [row["mode_id"] for row in evidence],
                "seeds": seeds,
                "group_positions": groups,
                "qualifies": qualifies,
            }
        )
    qualified = [row for row in recurrence_rows if row["qualifies"]]
    qualified_keys = {(row["side"], row["feature"]) for row in qualified}
    for mode in mode_rows:
        recurring = sorted(
            feature
            for feature in mode["qualifying_features"]
            if (mode["side"], feature) in qualified_keys
        )
        mode["recurring_features"] = recurring
        mode["automated_verdict"] = (
            "manual_recurrence_candidate"
            if recurring
            else "no_cross_seed_group_lexical_recurrence"
        )

    recurrence_records = _recurrence_records(
        tokenizer,
        cache,
        token_range=discovery.DEFAULT_PROTOCOL.recurrence_range,
        radius=protocol.context_radius,
    )
    recurrence_support = []
    for row in qualified:
        predicate = _feature_predicate(row["feature"])
        recurrence_support.append(
            {
                "side": row["side"],
                "feature": row["feature"],
                "token_support": sum(predicate(record) for record in recurrence_records),
                "token_count": len(recurrence_records),
            }
        )

    result = {
        "schema_version": ARTIFACT_SCHEMA_VERSION,
        "review_protocol": asdict(protocol),
        "review_protocol_digest": discovery.canonical_digest(asdict(protocol)),
        "source_artifacts": {
            "natural_selection": {
                "path": str(natural_selection.resolve()),
                "sha256": discovery.file_sha256(natural_selection),
            },
            "manifest": {
                "path": str(manifest_path.resolve()),
                "sha256": discovery.file_sha256(manifest_path),
            },
            "search": {
                "path": str(search_path.resolve()),
                "sha256": discovery.file_sha256(search_path),
            },
            "registry": {
                "path": str(registry_path.resolve()),
                "sha256": discovery.file_sha256(registry_path),
                "status": registry["status"],
            },
        },
        "accessed_ranges": {
            "source_selection": list(discovery.DEFAULT_PROTOCOL.source_selection_range),
            "discovery": list(discovery.DEFAULT_PROTOCOL.discovery_range),
            "recurrence": list(discovery.DEFAULT_PROTOCOL.recurrence_range),
        },
        "tokenizer": {
            "name": TOKENIZER_NAME,
            "resolved_name_or_path": tokenizer.name_or_path,
        },
        "summary": {
            "modes_reviewed": len(mode_rows),
            "controls_reviewed": len(search["controls"]),
            "row_shuffle_dominates_modes": sum(
                row["row_shuffle_absolute_ratio"] >= 1 for row in mode_rows
            ),
            "modes_with_familywise_lexical_alignment": sum(
                bool(row["qualifying_features"]) for row in mode_rows
            ),
            "qualified_recurrence_clusters": len(qualified),
            "modes_in_qualified_recurrence_clusters": sum(
                bool(row["recurring_features"]) for row in mode_rows
            ),
            "recurrence_range_sequences": sum(
                discovery.DEFAULT_PROTOCOL.recurrence_range[0] <= int(start)
                and int(start) + cache["input_ids"].shape[1]
                <= discovery.DEFAULT_PROTOCOL.recurrence_range[1]
                for start in cache["starts"]
            ),
            "recurrence_range_tokens_inspected": len(recurrence_records),
        },
        "qualified_recurrence_clusters": qualified,
        "all_feature_clusters": recurrence_rows,
        "recurrence_range_support": recurrence_support,
        "modes": mode_rows,
    }
    if len(mode_rows) != discovery.EXPECTED_MODE_COUNT:
        raise AssertionError("semantic audit did not cover every searched mode")
    discovery.atomic_json(output_path, result)
    write_review_table(review_path, result)
    return result


def _extreme_summary(rows: list[Mapping[str, Any]]) -> str:
    return " || ".join(
        f"{row['absolute_offset']}:{_clean(str(row['surface']))!r} [{row['context']}]"
        for row in rows
    )


def write_review_table(path: Path, artifact: Mapping[str, Any]) -> None:
    columns = (
        "mode_id",
        "seed",
        "group_position",
        "side",
        "rank",
        "eigenvalue",
        "row_shuffle_absolute_ratio",
        "top_alignments",
        "recurring_features",
        "positive_extremes",
        "negative_extremes",
        "automated_verdict",
    )
    lines = ["\t".join(columns)]
    for row in artifact["modes"]:
        values = {
            **row,
            "top_alignments": "; ".join(
                f"{value['feature']}={value['cosine']:.3f},"
                f"p={value['familywise_permutation_p']:.3f}"
                for value in row["top_alignments"]
            ),
            "recurring_features": "; ".join(row["recurring_features"]),
            "positive_extremes": _extreme_summary(row["positive_extremes"]),
            "negative_extremes": _extreme_summary(row["negative_extremes"]),
        }
        lines.append("\t".join(_clean(str(values[column])) for column in columns))
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    output = ROOT / "artifacts/exp05_decoder_advantage_discovery"
    parser.add_argument(
        "--natural-selection",
        type=Path,
        default=ROOT / "artifacts/exp04b_confirmatory/natural_selection.pt",
    )
    parser.add_argument("--manifest", type=Path, default=output / "discovery_manifest.json")
    parser.add_argument("--search", type=Path, default=output / "searched_modes.json")
    parser.add_argument("--registry", type=Path, default=output / "hypothesis_registry.json")
    parser.add_argument("--output", type=Path, default=output / "semantic_recurrence.json")
    parser.add_argument("--review-table", type=Path, default=output / "semantic_review.tsv")
    parser.add_argument("--allow-tokenizer-download", action="store_true")
    args = parser.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        TOKENIZER_NAME,
        local_files_only=not args.allow_tokenizer_download,
    )
    result = analyze(
        natural_selection=args.natural_selection,
        manifest_path=args.manifest,
        search_path=args.search,
        registry_path=args.registry,
        tokenizer=tokenizer,
        output_path=args.output,
        review_path=args.review_table,
    )
    print(json.dumps(result["summary"], indent=2), flush=True)


if __name__ == "__main__":
    main()
