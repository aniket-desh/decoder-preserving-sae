#!/usr/bin/env python3
"""Sealed, evaluation-only decoder-advantage eigentask discovery.

This runner deliberately stops before semantic interpretation.  It selects a
fixed discovery view from the immutable Experiment 4b natural-selection cache,
reconstructs that view with the paired block-8 MSE/DPSAE checkpoints one model
at a time, and logs every preregistered extreme eigentask and numerical control.

The final natural-text range is sealed behind a signed machine-readable
hypothesis registry.  Cache loading checks the declared range before calling
``torch.load`` so final contexts are never deserialized while the registry is
open or internally inconsistent.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch
from torch import Tensor

from dpsae.decoder_distance import batched_ridge_predict
from dpsae.exp04b_natural_text import apply_geometry_groups, geometry_group_indices
from dpsae.mech_analysis import load_sae


ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = 1
ALLOWED_DISPOSITIONS = {"unreviewed", "rejected", "supports_hypothesis"}
REQUIRED_HYPOTHESIS_FIELDS = {
    "hypothesis_id",
    "semantic_statement",
    "target_constructor",
    "exclusion_rules",
    "feature_ranking_rule",
    "maximum_feature_count",
    "predicted_advantage_sign",
    "evidence_mode_ids",
}


@dataclass(frozen=True)
class SearchProtocol:
    """All choices that determine the numerical discovery search."""

    source_selection_range: tuple[int, int] = (180_000_000, 185_000_000)
    discovery_range: tuple[int, int] = (180_000_000, 182_500_000)
    recurrence_range: tuple[int, int] = (182_500_000, 185_000_000)
    sealed_final_range: tuple[int, int] = (195_000_000, 200_000_000)
    exact_tokens: int = 16_384
    group_size: int = 128
    discovery_groups: int = 16
    extreme_modes_per_side: int = 2
    random_directions_per_group: int = 64
    seeds: tuple[int, ...] = (0, 1, 2)
    k: int = 32
    grouping: str = "document_balanced"
    sequence_selection_seed: int = 2_027_071_401
    grouping_seed: int = 2_027_071_402
    group_selection_seed: int = 2_027_071_403
    control_seed: int = 2_027_071_404

    def __post_init__(self) -> None:
        for name in (
            "source_selection_range",
            "discovery_range",
            "recurrence_range",
            "sealed_final_range",
        ):
            start, stop = getattr(self, name)
            if start < 0 or stop <= start:
                raise ValueError(f"{name} must be a nonempty nonnegative range")
        if not (
            self.source_selection_range[0]
            <= self.discovery_range[0]
            < self.discovery_range[1]
            <= self.recurrence_range[0]
            < self.recurrence_range[1]
            <= self.source_selection_range[1]
        ):
            raise ValueError("discovery and recurrence must be ordered inside selection")
        if self.exact_tokens <= 0 or self.group_size <= 1:
            raise ValueError("exact_tokens and group_size must be positive")
        if self.exact_tokens % self.group_size:
            raise ValueError("exact_tokens must divide into complete geometry groups")
        if not 0 < self.discovery_groups <= self.exact_tokens // self.group_size:
            raise ValueError("discovery_groups exceed the available exact groups")
        if not 0 < self.extreme_modes_per_side <= self.group_size // 2:
            raise ValueError("extreme mode count would overlap top and bottom modes")
        if self.random_directions_per_group < 1 or not self.seeds:
            raise ValueError("controls and seeds must be nonempty")
        if len(set(self.seeds)) != len(self.seeds):
            raise ValueError("seeds must be unique")
        if self.grouping != "document_balanced":
            raise ValueError("Experiment 5 discovery is preregistered as document-balanced")

    @property
    def expected_mode_count(self) -> int:
        return (
            len(self.seeds)
            * self.discovery_groups
            * 2
            * self.extreme_modes_per_side
        )

    @property
    def expected_control_count(self) -> int:
        return len(self.seeds) * self.discovery_groups


DEFAULT_PROTOCOL = SearchProtocol()
EXPECTED_MODE_COUNT = DEFAULT_PROTOCOL.expected_mode_count
EXPECTED_CONTROL_COUNT = DEFAULT_PROTOCOL.expected_control_count


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def file_sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_sha256(value: Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode())
    digest.update(_json_bytes(list(tensor.shape)))
    digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def repository_state(root: Path = ROOT) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", None
    return {"revision": revision, "dirty": dirty}


def ranges_overlap(first: tuple[int, int], second: tuple[int, int]) -> bool:
    return max(first[0], second[0]) < min(first[1], second[1])


def _resolve_recorded_path(recorded: str, registry_path: Path) -> Path:
    path = Path(recorded)
    return path if path.is_absolute() else registry_path.parent / path


def registry_frozen_digest(registry: Mapping[str, Any]) -> str:
    unsigned = dict(registry)
    unsigned.pop("frozen_digest", None)
    return canonical_digest(unsigned)


def validate_frozen_registry(registry: Mapping[str, Any], registry_path: Path) -> None:
    if registry.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("hypothesis registry has an unsupported schema")
    if registry.get("status") != "frozen":
        raise PermissionError("final contexts are sealed until the registry is frozen")
    observed_digest = registry.get("frozen_digest")
    if not isinstance(observed_digest, str) or observed_digest != registry_frozen_digest(registry):
        raise PermissionError("frozen hypothesis registry digest is invalid")
    search = registry.get("search_log")
    if not isinstance(search, Mapping):
        raise PermissionError("frozen registry is missing its search-log binding")
    search_path = _resolve_recorded_path(str(search.get("path", "")), registry_path)
    if not search_path.is_file() or file_sha256(search_path) != search.get("sha256"):
        raise PermissionError("searched-mode log changed after registry freeze")


def guard_range_access(
    requested_range: tuple[int, int],
    *,
    registry_path: Path | None,
    protocol: SearchProtocol = DEFAULT_PROTOCOL,
) -> None:
    """Refuse any access overlapping the sealed range before deserialization."""

    start, stop = requested_range
    if start < 0 or stop <= start:
        raise ValueError("requested range must be nonempty and nonnegative")
    if not ranges_overlap(requested_range, protocol.sealed_final_range):
        return
    if registry_path is None or not registry_path.is_file():
        raise PermissionError("final contexts are sealed and require a frozen registry")
    registry = json.loads(registry_path.read_text())
    validate_frozen_registry(registry, registry_path)
    if tuple(registry.get("sealed_final_range", ())) != protocol.sealed_final_range:
        raise PermissionError("frozen registry authorizes a different final range")
    if registry.get("protocol_digest") != canonical_digest(asdict(protocol)):
        raise PermissionError("frozen registry does not match the active protocol")


def guarded_load_natural_cache(
    path: Path,
    *,
    requested_range: tuple[int, int],
    registry_path: Path | None = None,
    protocol: SearchProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    """Guard the requested range before loading and then validate cache alignment."""

    guard_range_access(requested_range, registry_path=registry_path, protocol=protocol)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"input_ids", "activations", "starts", "eos_token_id"}
    if not isinstance(payload, dict) or not required <= payload.keys():
        raise ValueError(f"natural cache is missing {sorted(required - set(payload))}")
    input_ids, activations, starts = (
        payload["input_ids"],
        payload["activations"],
        payload["starts"],
    )
    if input_ids.ndim != 2 or activations.ndim != 3:
        raise ValueError("natural cache tensors have invalid ranks")
    if activations.shape[:2] != input_ids.shape or starts.shape != (len(input_ids),):
        raise ValueError("natural cache ids, activations, and starts are misaligned")
    sequence_length = input_ids.shape[1]
    if not bool(
        ((starts >= requested_range[0]) & (starts + sequence_length <= requested_range[1])).all()
    ):
        raise ValueError("natural cache contains sequences outside its declared range")
    return payload


def _select_unique_sequence_rows(
    starts: Tensor,
    *,
    sequence_length: int,
    count: int,
    token_range: tuple[int, int],
    seed: int,
) -> Tensor:
    eligible = (
        (starts >= token_range[0])
        & (starts + sequence_length <= token_range[1])
    ).nonzero(as_tuple=False).flatten()
    generator = torch.Generator().manual_seed(seed)
    order = eligible[torch.randperm(len(eligible), generator=generator)]
    selected, seen = [], set()
    for row in order.tolist():
        start = int(starts[row])
        if start not in seen:
            selected.append(row)
            seen.add(start)
        if len(selected) == count:
            break
    if len(selected) != count:
        raise RuntimeError(
            f"discovery range has {len(selected)} unique sequences; {count} are required"
        )
    return torch.tensor(sorted(selected), dtype=torch.long)


def build_discovery_manifest(
    cache: Mapping[str, Any],
    *,
    cache_path: Path,
    protocol: SearchProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    input_ids = cache["input_ids"]
    starts = cache["starts"].long()
    sequence_length = int(input_ids.shape[1])
    if protocol.exact_tokens % sequence_length:
        raise ValueError("exact token budget must contain whole cached sequences")
    sequence_count = protocol.exact_tokens // sequence_length
    rows = _select_unique_sequence_rows(
        starts,
        sequence_length=sequence_length,
        count=sequence_count,
        token_range=protocol.discovery_range,
        seed=protocol.sequence_selection_seed,
    )
    selected_ids = input_ids[rows]
    all_groups = geometry_group_indices(
        selected_ids,
        protocol.group_size,
        protocol.grouping,
        seed=protocol.grouping_seed,
        eos_token_id=int(cache["eos_token_id"]),
    )
    generator = torch.Generator().manual_seed(protocol.group_selection_seed)
    group_positions = torch.randperm(len(all_groups), generator=generator)[
        : protocol.discovery_groups
    ]
    indices = all_groups[group_positions]
    if indices.shape != (protocol.discovery_groups, protocol.group_size):
        raise AssertionError("preregistered group construction produced the wrong shape")
    return {
        "schema_version": SCHEMA_VERSION,
        "protocol": asdict(protocol),
        "protocol_digest": canonical_digest(asdict(protocol)),
        "repository": repository_state(),
        "source_cache": {
            "path": str(cache_path.resolve()),
            "sha256": file_sha256(cache_path),
        },
        "sequence_length": sequence_length,
        "selected_sequence_rows": rows.tolist(),
        "selected_sequence_starts": starts[rows].tolist(),
        "selected_input_ids_sha256": tensor_sha256(selected_ids),
        "selected_activations_sha256": tensor_sha256(cache["activations"][rows]),
        "group_positions": group_positions.tolist(),
        "group_indices": indices.tolist(),
    }


def validate_discovery_manifest(
    manifest: Mapping[str, Any], protocol: SearchProtocol = DEFAULT_PROTOCOL
) -> None:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("discovery manifest schema mismatch")
    if manifest.get("protocol_digest") != canonical_digest(asdict(protocol)):
        raise RuntimeError("discovery manifest protocol changed")
    if len(manifest.get("selected_sequence_rows", [])) == 0:
        raise RuntimeError("discovery manifest has no selected sequences")
    indices = torch.tensor(manifest.get("group_indices", []), dtype=torch.long)
    if indices.shape != (protocol.discovery_groups, protocol.group_size):
        raise RuntimeError("discovery manifest does not contain the preregistered groups")
    positions = manifest.get("group_positions", [])
    if len(positions) != protocol.discovery_groups or len(set(positions)) != len(positions):
        raise RuntimeError("discovery group positions must be unique and complete")


def paired_model_names(
    payloads: Mapping[str, Mapping[str, Any]],
    protocol: SearchProtocol = DEFAULT_PROTOCOL,
) -> dict[int, dict[str, str]]:
    result: dict[int, dict[str, str]] = {}
    for name, payload in payloads.items():
        spec = payload.get("spec", {})
        method = spec.get("method")
        seed = spec.get("seed")
        if method not in {"mse", "dpsae"} or seed not in protocol.seeds:
            continue
        if int(spec.get("k", -1)) != protocol.k:
            continue
        slot = result.setdefault(int(seed), {})
        if method in slot:
            raise ValueError(f"multiple {method} models found for seed {seed}")
        slot[method] = name
    expected = {"mse", "dpsae"}
    if set(result) != set(protocol.seeds) or any(set(pair) != expected for pair in result.values()):
        raise ValueError("source fleet must contain one paired MSE/DPSAE model per seed")
    return result


@torch.inference_mode()
def reconstruct_one(
    payload: Mapping[str, Any],
    activations: Tensor,
    *,
    device: torch.device,
    batch_tokens: int = 4_096,
) -> Tensor:
    if activations.ndim != 3 or batch_tokens < 1:
        raise ValueError("activations must be rank-3 and batch_tokens positive")
    model = load_sae(dict(payload), input_dim=activations.shape[-1], device=device)
    chunks = []
    for batch in activations.flatten(0, 1).split(batch_tokens):
        reconstruction, _ = model(batch.to(device).float(), use_threshold=True)
        chunks.append(reconstruction.cpu().half())
    result = torch.cat(chunks).reshape_as(activations)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _reconstruction_path(output: Path, name: str) -> Path:
    return output / "reconstructions" / f"{name}.pt"


def prepare_reconstructions(
    *,
    natural_selection: Path,
    models_path: Path,
    output: Path,
    device: torch.device,
    batch_tokens: int = 4_096,
    protocol: SearchProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    cache = guarded_load_natural_cache(
        natural_selection,
        requested_range=protocol.source_selection_range,
        protocol=protocol,
    )
    manifest_path = output / "discovery_manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        validate_discovery_manifest(manifest, protocol)
        if manifest["source_cache"]["sha256"] != file_sha256(natural_selection):
            raise RuntimeError("natural-selection cache changed after discovery registration")
    else:
        manifest = build_discovery_manifest(
            cache, cache_path=natural_selection, protocol=protocol
        )
        atomic_json(manifest_path, manifest)
    rows = torch.tensor(manifest["selected_sequence_rows"], dtype=torch.long)
    activations = cache["activations"][rows]
    if activations.numel() // activations.shape[-1] != protocol.exact_tokens:
        raise AssertionError("reconstruction view is not exactly the preregistered token count")

    source_sha = file_sha256(models_path)
    payloads = torch.load(models_path, map_location="cpu", weights_only=False)
    pairs = paired_model_names(payloads, protocol)
    ordered_names = [pairs[seed][method] for seed in protocol.seeds for method in ("mse", "dpsae")]
    selected_payloads = {name: payloads[name] for name in ordered_names}
    del payloads
    for name in ordered_names:
        payload = selected_payloads.pop(name)
        destination = _reconstruction_path(output, name)
        metadata = {
            "schema_version": SCHEMA_VERSION,
            "model_name": name,
            "model_spec": payload["spec"],
            "source_models": {"path": str(models_path.resolve()), "sha256": source_sha},
            "manifest_digest": canonical_digest(manifest),
            "shape": list(activations.shape),
            "dtype": "torch.float16",
        }
        if destination.exists():
            existing = torch.load(destination, map_location="cpu", weights_only=False)
            cached_reconstruction = existing.get("reconstruction")
            if (
                existing.get("metadata") != metadata
                or not isinstance(cached_reconstruction, Tensor)
                or cached_reconstruction.shape != activations.shape
            ):
                raise RuntimeError(f"incompatible reconstruction cache: {destination}")
            continue
        reconstruction = reconstruct_one(
            payload, activations, device=device, batch_tokens=batch_tokens
        )
        atomic_torch(destination, {"metadata": metadata, "reconstruction": reconstruction})
        del reconstruction, payload
    del selected_payloads, activations, cache
    return {"manifest": manifest, "paired_models": pairs}


def _hat_matrices(groups: Tensor, ridge: float) -> Tensor:
    count, samples, _ = groups.shape
    identity = torch.eye(samples, dtype=torch.float32).expand(count, samples, samples)
    return batched_ridge_predict(groups.float(), identity, ridge)


def _canonicalize_mode(vector: Tensor) -> Tensor:
    pivot = int(vector.abs().argmax())
    return -vector if vector[pivot] < 0 else vector


def _group_control_seed(protocol: SearchProtocol, seed: int, group_slot: int) -> int:
    payload = f"{protocol.control_seed}:{seed}:{group_slot}".encode()
    return int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "big") % (2**31 - 1)


def extreme_modes_for_group(
    q: Tensor,
    q_row_shuffle: Tensor,
    *,
    seed: int,
    group_slot: int,
    group_position: int,
    flat_token_indices: Tensor,
    selected_sequence_starts: Tensor,
    sequence_length: int,
    protocol: SearchProtocol = DEFAULT_PROTOCOL,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if q.shape != (protocol.group_size, protocol.group_size) or q_row_shuffle.shape != q.shape:
        raise ValueError("advantage operators must match the preregistered group size")
    if flat_token_indices.shape != (protocol.group_size,):
        raise ValueError("flat token indices do not match the group")
    q = 0.5 * (q.double() + q.double().mT)
    q_row_shuffle = 0.5 * (q_row_shuffle.double() + q_row_shuffle.double().mT)
    eigenvalues, eigenvectors = torch.linalg.eigh(q)

    generator = torch.Generator().manual_seed(_group_control_seed(protocol, seed, group_slot))
    random_tasks = torch.randn(
        protocol.group_size,
        protocol.random_directions_per_group,
        generator=generator,
        dtype=torch.float64,
    )
    random_tasks /= random_tasks.norm(dim=0, keepdim=True).clamp_min(1e-12)
    random_rayleigh = (random_tasks * (q @ random_tasks)).sum(0)
    shuffle_eigenvalues = torch.linalg.eigvalsh(q_row_shuffle)

    mode_rows = []
    for side in ("top", "bottom"):
        for rank in range(1, protocol.extreme_modes_per_side + 1):
            column = -rank if side == "top" else rank - 1
            vector = _canonicalize_mode(eigenvectors[:, column])
            eigenvalue = float(eigenvalues[column])
            row_shuffle_rayleigh = float(vector @ q_row_shuffle @ vector)
            percentile = float((random_rayleigh <= eigenvalue).double().mean())
            extreme_count = min(8, protocol.group_size)
            positive = vector.topk(extreme_count).indices
            negative = (-vector).topk(extreme_count).indices

            def absolute_offsets(local_positions: Tensor) -> list[int]:
                flat = flat_token_indices[local_positions]
                sequence_rows = torch.div(flat, sequence_length, rounding_mode="floor")
                token_positions = flat % sequence_length
                return (selected_sequence_starts[sequence_rows] + token_positions).tolist()

            mode_rows.append(
                {
                    "mode_id": f"s{seed}_g{group_slot:02d}_{side}{rank}",
                    "seed": seed,
                    "group_slot": group_slot,
                    "group_position": group_position,
                    "side": side,
                    "rank": rank,
                    "eigenvalue": eigenvalue,
                    "eigentask": vector.tolist(),
                    "positive_extreme_absolute_offsets": absolute_offsets(positive),
                    "negative_extreme_absolute_offsets": absolute_offsets(negative),
                    "controls": {
                        "row_shuffle_rayleigh": row_shuffle_rayleigh,
                        "random_reference_count": protocol.random_directions_per_group,
                        "random_rayleigh_percentile": percentile,
                    },
                }
            )
    control = {
        "control_id": f"s{seed}_g{group_slot:02d}",
        "seed": seed,
        "group_slot": group_slot,
        "group_position": group_position,
        "row_shuffle": {
            "minimum_eigenvalue": float(shuffle_eigenvalues[0]),
            "maximum_eigenvalue": float(shuffle_eigenvalues[-1]),
        },
        "random_directions": {
            "count": protocol.random_directions_per_group,
            "rayleigh_values": random_rayleigh.tolist(),
        },
        "observed_operator": {
            "trace": float(torch.trace(q)),
            "frobenius_norm": float(torch.linalg.matrix_norm(q)),
            "minimum_eigenvalue": float(eigenvalues[0]),
            "maximum_eigenvalue": float(eigenvalues[-1]),
        },
    }
    return mode_rows, control


def validate_search_log(
    search: Mapping[str, Any], protocol: SearchProtocol = DEFAULT_PROTOCOL
) -> None:
    if search.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError("searched-mode log schema mismatch")
    if search.get("protocol_digest") != canonical_digest(asdict(protocol)):
        raise RuntimeError("searched-mode log protocol mismatch")
    modes, controls = search.get("modes"), search.get("controls")
    if not isinstance(modes, list) or len(modes) != protocol.expected_mode_count:
        raise RuntimeError(
            f"searched-mode log must contain exactly {protocol.expected_mode_count} modes"
        )
    if not isinstance(controls, list) or len(controls) != protocol.expected_control_count:
        raise RuntimeError(
            f"searched-mode log must contain exactly {protocol.expected_control_count} controls"
        )
    ids = [row.get("mode_id") for row in modes]
    if len(set(ids)) != len(ids) or any(not isinstance(value, str) for value in ids):
        raise RuntimeError("searched modes require unique string IDs")
    required_mode = {
        "seed",
        "group_slot",
        "group_position",
        "side",
        "rank",
        "eigenvalue",
        "eigentask",
        "controls",
    }
    if any(not required_mode <= row.keys() for row in modes):
        raise RuntimeError("searched mode is missing numerical evidence or controls")
    if any(len(row["eigentask"]) != protocol.group_size for row in modes):
        raise RuntimeError("searched eigentask has the wrong group dimension")
    control_ids = [row.get("control_id") for row in controls]
    if len(set(control_ids)) != len(control_ids):
        raise RuntimeError("numerical controls require unique IDs")
    if any(
        not {"row_shuffle", "random_directions", "observed_operator"} <= row.keys()
        for row in controls
    ):
        raise RuntimeError("control log is incomplete")


def _load_reconstruction(
    path: Path,
    *,
    expected_manifest_digest: str,
    expected_name: str,
) -> Tensor:
    value = torch.load(path, map_location="cpu", weights_only=False)
    metadata = value.get("metadata", {})
    if metadata.get("model_name") != expected_name:
        raise RuntimeError(f"reconstruction cache has the wrong model: {path}")
    if metadata.get("manifest_digest") != expected_manifest_digest:
        raise RuntimeError(f"reconstruction cache has the wrong discovery manifest: {path}")
    reconstruction = value.get("reconstruction")
    if not isinstance(reconstruction, Tensor) or reconstruction.ndim != 3:
        raise RuntimeError(f"reconstruction cache is malformed: {path}")
    return reconstruction


def _row_shuffled_groups(
    groups: Tensor,
    *,
    seed: int,
    protocol: SearchProtocol,
) -> tuple[Tensor, list[list[int]]]:
    shuffled, permutations = [], []
    for group_slot, group in enumerate(groups):
        generator = torch.Generator().manual_seed(
            _group_control_seed(protocol, seed, group_slot)
        )
        permutation = torch.randperm(protocol.group_size, generator=generator)
        shuffled.append(group[permutation])
        permutations.append(permutation.tolist())
    return torch.stack(shuffled), permutations


def compute_search_log(
    *,
    natural_selection: Path,
    static_calibration: Path,
    output: Path,
    protocol: SearchProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    manifest_path = output / "discovery_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError("prepare reconstructions before computing discovery modes")
    manifest = json.loads(manifest_path.read_text())
    validate_discovery_manifest(manifest, protocol)
    manifest_digest = canonical_digest(manifest)
    cache = guarded_load_natural_cache(
        natural_selection,
        requested_range=protocol.source_selection_range,
        protocol=protocol,
    )
    if file_sha256(natural_selection) != manifest["source_cache"]["sha256"]:
        raise RuntimeError("natural-selection cache changed after reconstruction")
    rows = torch.tensor(manifest["selected_sequence_rows"], dtype=torch.long)
    indices = torch.tensor(manifest["group_indices"], dtype=torch.long)
    selected_starts = torch.tensor(manifest["selected_sequence_starts"], dtype=torch.long)
    selected_activations = cache["activations"][rows]
    if tensor_sha256(selected_activations) != manifest["selected_activations_sha256"]:
        raise RuntimeError("selected activations no longer match the manifest")
    original_groups = apply_geometry_groups(selected_activations, indices).float()
    static = torch.load(static_calibration, map_location="cpu", weights_only=False)
    if (
        "ridge" not in static
        or not math.isfinite(float(static["ridge"]))
        or float(static["ridge"]) <= 0
    ):
        raise ValueError("static calibration requires a finite positive ridge")
    ridge = float(static["ridge"])
    reference = _hat_matrices(original_groups, ridge)
    del static

    modes, controls = [], []
    group_positions = manifest["group_positions"]
    sequence_length = int(manifest["sequence_length"])
    for seed in protocol.seeds:
        mse_name, dpsae_name = f"mse_s{seed}", f"dpsae_s{seed}"
        mse = _load_reconstruction(
            _reconstruction_path(output, mse_name),
            expected_manifest_digest=manifest_digest,
            expected_name=mse_name,
        )
        mse_delta = reference - _hat_matrices(apply_geometry_groups(mse, indices), ridge)
        del mse

        dpsae = _load_reconstruction(
            _reconstruction_path(output, dpsae_name),
            expected_manifest_digest=manifest_digest,
            expected_name=dpsae_name,
        )
        dpsae_groups = apply_geometry_groups(dpsae, indices)
        dpsae_delta = reference - _hat_matrices(dpsae_groups, ridge)
        shuffled_groups, permutations = _row_shuffled_groups(
            dpsae_groups, seed=seed, protocol=protocol
        )
        shuffled_delta = reference - _hat_matrices(shuffled_groups, ridge)
        del dpsae, dpsae_groups, shuffled_groups

        q = mse_delta.mT @ mse_delta - dpsae_delta.mT @ dpsae_delta
        q_shuffle = mse_delta.mT @ mse_delta - shuffled_delta.mT @ shuffled_delta
        for group_slot in range(protocol.discovery_groups):
            group_modes, group_control = extreme_modes_for_group(
                q[group_slot],
                q_shuffle[group_slot],
                seed=seed,
                group_slot=group_slot,
                group_position=int(group_positions[group_slot]),
                flat_token_indices=indices[group_slot],
                selected_sequence_starts=selected_starts,
                sequence_length=sequence_length,
                protocol=protocol,
            )
            group_control["row_shuffle"]["permutation"] = permutations[group_slot]
            modes.extend(group_modes)
            controls.append(group_control)
        del mse_delta, dpsae_delta, shuffled_delta, q, q_shuffle

    result = {
        "schema_version": SCHEMA_VERSION,
        "protocol": asdict(protocol),
        "protocol_digest": canonical_digest(asdict(protocol)),
        "repository": repository_state(),
        "manifest": {"path": str(manifest_path.resolve()), "sha256": file_sha256(manifest_path)},
        "static_calibration": {
            "path": str(static_calibration.resolve()),
            "sha256": file_sha256(static_calibration),
            "ridge": ridge,
        },
        "modes": modes,
        "controls": controls,
    }
    validate_search_log(result, protocol)
    atomic_json(output / "searched_modes.json", result)
    del cache, selected_activations, original_groups, reference
    return result


def initialize_hypothesis_registry(
    *,
    search_log_path: Path,
    registry_path: Path,
    protocol: SearchProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    search = json.loads(search_log_path.read_text())
    validate_search_log(search, protocol)
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
        if registry.get("search_log", {}).get("sha256") != file_sha256(search_log_path):
            raise RuntimeError("existing hypothesis registry refers to another search log")
        return registry
    try:
        recorded_path = str(search_log_path.relative_to(registry_path.parent))
    except ValueError:
        recorded_path = str(search_log_path.resolve())
    registry = {
        "schema_version": SCHEMA_VERSION,
        "status": "open",
        "protocol_digest": canonical_digest(asdict(protocol)),
        "sealed_final_range": list(protocol.sealed_final_range),
        "search_log": {"path": recorded_path, "sha256": file_sha256(search_log_path)},
        "expected_mode_count": protocol.expected_mode_count,
        "mode_dispositions": {
            row["mode_id"]: {
                "status": "unreviewed",
                "hypothesis_id": None,
                "note": "",
            }
            for row in search["modes"]
        },
        "hypotheses": [],
        "opened_at_utc": datetime.now(timezone.utc).isoformat(),
        "frozen_at_utc": None,
        "frozen_digest": None,
    }
    atomic_json(registry_path, registry)
    return registry


def _validate_hypotheses(
    registry: Mapping[str, Any], modes_by_id: Mapping[str, Mapping[str, Any]]
) -> dict[str, Mapping[str, Any]]:
    mode_ids = set(modes_by_id)
    hypotheses = registry.get("hypotheses")
    if not isinstance(hypotheses, list) or len(hypotheses) > 3:
        raise ValueError("registry must contain at most three hypotheses")
    by_id: dict[str, Mapping[str, Any]] = {}
    for hypothesis in hypotheses:
        if (
            not isinstance(hypothesis, Mapping)
            or not REQUIRED_HYPOTHESIS_FIELDS <= hypothesis.keys()
        ):
            raise ValueError("hypothesis entry is missing its frozen target specification")
        hypothesis_id = hypothesis["hypothesis_id"]
        if not isinstance(hypothesis_id, str) or not hypothesis_id or hypothesis_id in by_id:
            raise ValueError("hypothesis IDs must be unique nonempty strings")
        if hypothesis["predicted_advantage_sign"] not in {"positive", "negative"}:
            raise ValueError("predicted advantage sign must be positive or negative")
        if (
            not isinstance(hypothesis["maximum_feature_count"], int)
            or hypothesis["maximum_feature_count"] < 1
        ):
            raise ValueError("maximum feature count must be a positive integer")
        evidence = hypothesis["evidence_mode_ids"]
        if not isinstance(evidence, list) or not evidence or not set(evidence) <= mode_ids:
            raise ValueError("hypothesis evidence must name searched modes")
        evidence_rows = [modes_by_id[mode_id] for mode_id in evidence]
        if len({row["seed"] for row in evidence_rows}) < 2:
            raise ValueError("a promoted hypothesis must recur in at least two seed pairs")
        if len({row["group_position"] for row in evidence_rows}) < 4:
            raise ValueError("a promoted hypothesis must recur in four independent groups")
        expected_side = (
            "top" if hypothesis["predicted_advantage_sign"] == "positive" else "bottom"
        )
        if any(row["side"] != expected_side for row in evidence_rows):
            raise ValueError("hypothesis evidence disagrees with its predicted advantage sign")
        for field in ("semantic_statement", "target_constructor", "feature_ranking_rule"):
            if not isinstance(hypothesis[field], str) or not hypothesis[field].strip():
                raise ValueError(f"hypothesis {field} must be nonempty")
        if not isinstance(hypothesis["exclusion_rules"], list):
            raise ValueError("hypothesis exclusion rules must be a list")
        by_id[hypothesis_id] = hypothesis
    return by_id


def freeze_hypothesis_registry(
    *,
    registry_path: Path,
    protocol: SearchProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    registry = json.loads(registry_path.read_text())
    if registry.get("status") == "frozen":
        validate_frozen_registry(registry, registry_path)
        return registry
    if registry.get("status") != "open":
        raise ValueError("hypothesis registry must be open before it can freeze")
    if registry.get("protocol_digest") != canonical_digest(asdict(protocol)):
        raise ValueError("hypothesis registry protocol changed")
    search_path = _resolve_recorded_path(registry["search_log"]["path"], registry_path)
    if file_sha256(search_path) != registry["search_log"].get("sha256"):
        raise ValueError("searched-mode log changed while registry was open")
    search = json.loads(search_path.read_text())
    validate_search_log(search, protocol)
    modes_by_id = {row["mode_id"]: row for row in search["modes"]}
    mode_ids = set(modes_by_id)
    dispositions = registry.get("mode_dispositions")
    if not isinstance(dispositions, Mapping) or set(dispositions) != mode_ids:
        raise ValueError("registry must disposition every searched mode exactly once")
    hypotheses = _validate_hypotheses(registry, modes_by_id)
    for mode_id, disposition in dispositions.items():
        if (
            not isinstance(disposition, Mapping)
            or disposition.get("status") not in ALLOWED_DISPOSITIONS
        ):
            raise ValueError(f"mode {mode_id} has an invalid disposition")
        if disposition["status"] == "unreviewed":
            raise ValueError("all searched modes must be reviewed before registry freeze")
        hypothesis_id = disposition.get("hypothesis_id")
        if disposition["status"] == "supports_hypothesis":
            if hypothesis_id not in hypotheses:
                raise ValueError(f"mode {mode_id} refers to an unknown hypothesis")
            if mode_id not in hypotheses[hypothesis_id]["evidence_mode_ids"]:
                raise ValueError(f"mode {mode_id} is absent from its hypothesis evidence")
        elif hypothesis_id is not None:
            raise ValueError(f"rejected mode {mode_id} cannot name a hypothesis")
    for hypothesis_id, hypothesis in hypotheses.items():
        supported = {
            mode_id
            for mode_id, disposition in dispositions.items()
            if disposition.get("hypothesis_id") == hypothesis_id
            and disposition.get("status") == "supports_hypothesis"
        }
        if supported != set(hypothesis["evidence_mode_ids"]):
            raise ValueError(f"hypothesis {hypothesis_id} evidence and dispositions disagree")
    frozen = copy.deepcopy(registry)
    frozen["status"] = "frozen"
    frozen["frozen_at_utc"] = datetime.now(timezone.utc).isoformat()
    frozen["frozen_digest"] = None
    frozen["frozen_digest"] = registry_frozen_digest(frozen)
    atomic_json(registry_path, frozen)
    validate_frozen_registry(frozen, registry_path)
    return frozen


def run_search(
    *,
    natural_selection: Path,
    static_calibration: Path,
    output: Path,
    protocol: SearchProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    search_path = output / "searched_modes.json"
    if search_path.exists():
        search = json.loads(search_path.read_text())
        validate_search_log(search, protocol)
    else:
        search = compute_search_log(
            natural_selection=natural_selection,
            static_calibration=static_calibration,
            output=output,
            protocol=protocol,
        )
    initialize_hypothesis_registry(
        search_log_path=search_path,
        registry_path=output / "hypothesis_registry.json",
        protocol=protocol,
    )
    return search


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "stage",
        choices=("prepare", "search", "all", "freeze-registry", "verify-seal"),
    )
    parser.add_argument(
        "--natural-selection",
        type=Path,
        default=ROOT / "artifacts/exp04b_confirmatory/natural_selection.pt",
    )
    parser.add_argument(
        "--models",
        type=Path,
        default=ROOT / "artifacts/exp04_ioi_mechanism/confirmation/models.pt",
    )
    parser.add_argument(
        "--static-calibration",
        type=Path,
        default=ROOT / "artifacts/exp04b_confirmatory/static_calibration.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "artifacts/exp05_decoder_advantage_discovery",
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-tokens", type=int, default=4_096)
    args = parser.parse_args()

    if args.stage in {"prepare", "all"}:
        prepare_reconstructions(
            natural_selection=args.natural_selection,
            models_path=args.models,
            output=args.output,
            device=_device(args.device),
            batch_tokens=args.batch_tokens,
        )
    if args.stage in {"search", "all"}:
        run_search(
            natural_selection=args.natural_selection,
            static_calibration=args.static_calibration,
            output=args.output,
        )
    registry_path = args.output / "hypothesis_registry.json"
    if args.stage == "freeze-registry":
        freeze_hypothesis_registry(registry_path=registry_path)
        print(f"frozen registry: {registry_path}", flush=True)
    if args.stage == "verify-seal":
        guard_range_access(
            DEFAULT_PROTOCOL.sealed_final_range,
            registry_path=registry_path,
        )
        print("sealed final range is authorized by the frozen registry", flush=True)


if __name__ == "__main__":
    main()
