#!/usr/bin/env python3
"""Fresh three-pair concept confirmation using Exp10's exact evaluators.

This driver owns authorization, checkpoint adaptation, immutable-cache
adoption, sharding, confirmation statistics, and promotion.  Sparse and
companion fitting remain in :mod:`experiments.exp10_concept_discovery`.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import signal
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from sklearn.metrics import roc_auc_score

from experiments import exp10_concept_discovery as exp10
from experiments import exp12_pythia_fresh_confirmation as exp12


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/exp13_concept_confirmation.json"
DEFAULT_OUTPUT = ROOT / "artifacts/exp13_concept_confirmation"
METHODS = ("mse", "dpsae")


class RunAbortedError(RuntimeError):
    """Raised when another Exp13 process has requested a fail-fast stop."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def file_sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def verify_file_record(record: Mapping[str, Any], *, label: str) -> Path:
    if set(record) != {"path", "bytes", "sha256"}:
        raise RuntimeError(f"{label} file-record schema drift")
    path = Path(str(record["path"])).resolve()
    if file_record(path) != dict(record):
        raise RuntimeError(f"{label} changed after it was frozen")
    return path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def request_abort(
    *, config: Mapping[str, Any], output_root: Path, reason: str
) -> dict[str, Any]:
    path = output_root / "abort_requested.json"
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing.get("config_digest") != canonical_digest(config):
            raise RuntimeError("Exp13 abort marker belongs to another config")
        return existing
    payload = {
        "schema_version": 1,
        "complete": True,
        "abort_requested": True,
        "config_digest": canonical_digest(config),
        "reason": reason,
        "written_at_unix_seconds": time.time(),
    }
    atomic_json(path, payload)
    return payload


def check_abort(config: Mapping[str, Any], output_root: Path) -> None:
    path = output_root / "abort_requested.json"
    if not path.is_file():
        return
    value = json.loads(path.read_text())
    if value.get("config_digest") != canonical_digest(config):
        raise RuntimeError("Exp13 abort marker belongs to another config")
    raise RunAbortedError(str(value.get("reason", "another Exp13 process failed")))


def repository_state() -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    if status:
        raise RuntimeError("Exp13 requires a clean committed repository")
    return {"revision": revision, "dirty": False, "status": []}


def source_hashes(config: Mapping[str, Any]) -> dict[str, str]:
    paths = config["provenance"]["source_files"]
    if not isinstance(paths, list) or not paths or len(paths) != len(set(paths)):
        raise ValueError("Exp13 source paths must be a unique nonempty list")
    result = {}
    for relative in paths:
        path = (ROOT / str(relative)).resolve()
        if ROOT not in path.parents or not path.is_file():
            raise FileNotFoundError(path)
        result[str(relative)] = file_sha256(path)
    return result


def shard_template(pair_count: int = 3, seed_count: int = 10) -> dict[str, Any]:
    sparse = [[] for _ in range(4)]
    companion = [[] for _ in range(4)]
    sparse_order = [
        {"pair_slot": pair, "method": method, "seed_slot": seed}
        for pair in range(pair_count)
        for method in METHODS
        for seed in range(seed_count)
    ]
    companion_order = [
        {"pair_slot": pair, "seed_slot": seed}
        for pair in range(pair_count)
        for seed in range(seed_count)
    ]
    for index, row in enumerate(sparse_order):
        sparse[index % 4].append(row)
    for index, row in enumerate(companion_order):
        companion[index % 4].append(row)
    return {
        "worker_count": 4,
        "pair_count": pair_count,
        "seed_count": seed_count,
        "workers": [
            {"worker_index": index, "sparse": sparse[index], "companion": companion[index]}
            for index in range(4)
        ],
    }


def realized_shards(pair_seeds: Sequence[int], probe_seeds: Sequence[int]) -> dict[str, Any]:
    template = shard_template(len(pair_seeds), len(probe_seeds))
    workers = []
    for worker in template["workers"]:
        workers.append(
            {
                "worker_index": worker["worker_index"],
                "sparse": [
                    {
                        "pair_seed": int(pair_seeds[row["pair_slot"]]),
                        "method": row["method"],
                        "probe_seed": int(probe_seeds[row["seed_slot"]]),
                    }
                    for row in worker["sparse"]
                ],
                "companion": [
                    {
                        "pair_seed": int(pair_seeds[row["pair_slot"]]),
                        "probe_seed": int(probe_seeds[row["seed_slot"]]),
                    }
                    for row in worker["companion"]
                ],
            }
        )
    return {**template, "workers": workers}


def validate_config(config: Mapping[str, Any]) -> None:
    if (
        config.get("schema_version") != 1
        or config.get("experiment_id") != "exp13_concept_confirmation"
    ):
        raise ValueError("not an Exp13 concept-confirmation config")
    runtime = config["runtime"]
    if (
        int(runtime["worker_count"]) != 4
        or int(runtime["pair_count"]) != 3
        or int(runtime["probe_seed_count"]) != 10
        or runtime["method_order"] != list(METHODS)
        or runtime["sparse_jobs_per_worker"] != [15, 15, 15, 15]
        or runtime["companion_jobs_per_worker"] != [8, 8, 7, 7]
        or int(runtime["worker_timeout_seconds"]) != 27_000
    ):
        raise ValueError("Exp13 fixed fleet shape changed")
    template = shard_template()
    if runtime["shard_template_sha256"] != canonical_digest(template):
        raise ValueError("Exp13 shard template digest changed")
    projection = runtime["timing_projection"]
    if (
        int(projection["pair_multiplier"]) != 3
        or float(projection["maximum_projected_pod_hours"]) != 7.5
        or projection["concept_metrics_forbidden"] is not True
        or int(config["base_experiment"]["required_timing_schema_version"]) != 7
    ):
        raise ValueError("Exp13 timing gate changed")
    authorization = config["authorization"]
    if (
        int(authorization["required_pair_count"]) != 3
        or authorization["required_authorized_value"] is not True
        or authorization["required_excluded_pilot_checkpoint_id"]
        != "pythia160m_block8_s0_25m"
        or authorization["required_confirmatory_inference"]
        != {
            "global_family_block_interval_is_gate_forming": True,
            "family_specific_p_values": (
                "one_sided_centered_paired_stratified_heldout_example_bootstrap"
            ),
            "holm_adjustment_scope": "family_specific_reporting_only",
            "family_significance_is_gate_forming": False,
            "individual_concepts": "descriptive_only",
        }
        or authorization["require_common_selected_checkpoint"] is not True
        or authorization["require_all_matched_quality_gates"] is not True
        or runtime["cache_adoption_requires_all_file_sha256"] is not True
        or runtime["cache_regeneration_forbidden"] is not True
    ):
        raise ValueError("Exp13 authorization or immutable-cache contract changed")
    statistics = config["statistics"]
    if (
        int(statistics["primary_k"]) != 5
        or float(statistics["minimum_median_pair_macro"]) != 0.005
        or statistics["all_pair_macros_strictly_positive"] is not True
        or statistics["pooled_lower95_strictly_positive"] is not True
        or statistics["family_tests"]["gate_forming"] is not False
        or statistics["family_tests"]["reporting_only"] is not True
    ):
        raise ValueError("Exp13 confirmatory inference changed")
    candidates = config["candidates"]
    if (
        int(candidates["minimum_probe_seed_count"]) != 5
        or int(candidates["probe_seed_denominator"]) != 10
        or int(candidates["maximum_per_method"]) != 300
        or candidates["equal_method_budgets"] is not True
        or candidates["never_quota_fill"] is not True
        or candidates["ranking"]
        != [
            "descending_probe_seed_frequency",
            "descending_mean_absolute_weight",
            "dataset",
            "feature_id",
        ]
    ):
        raise ValueError("Exp13 candidate promotion changed")


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = json.loads(path.read_text())
    validate_config(config)
    return config


def project_runtime(config: Mapping[str, Any], pilot_timing: Mapping[str, Any]) -> dict[str, Any]:
    required_schema = int(config["base_experiment"]["required_timing_schema_version"])
    if (
        pilot_timing.get("schema_version") != required_schema
        or pilot_timing.get("complete") is not True
        or pilot_timing.get("passed") is not True
        or pilot_timing.get("names_and_concept_results_suppressed") is not True
        or int(pilot_timing.get("saved_concept_metric_count", -1)) != 0
    ):
        raise RuntimeError("final Exp10 timing report is not the blind schema-v7 pass")
    source_hours = float(pilot_timing.get("projection", {}).get("projected_pod_hours", math.nan))
    if not math.isfinite(source_hours) or source_hours <= 0:
        raise RuntimeError("final Exp10 timing report has no finite projection")
    multiplier = int(config["runtime"]["timing_projection"]["pair_multiplier"])
    projected = source_hours * multiplier
    maximum = float(config["runtime"]["timing_projection"]["maximum_projected_pod_hours"])
    return {
        "schema_version": 1,
        "complete": True,
        "passed": projected <= maximum,
        "source_schema_version": required_schema,
        "source_projected_pod_hours": source_hours,
        "pair_multiplier": multiplier,
        "projected_pod_hours": projected,
        "maximum_projected_pod_hours": maximum,
        "accounting": "three_times_complete_pilot_projection_including_fixed_terms",
        "concept_metrics_opened": False,
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    if not p_values:
        return {}
    ordered = sorted((float(value), key) for key, value in p_values.items())
    if any(not 0 <= value <= 1 or not math.isfinite(value) for value, _ in ordered):
        raise ValueError("Holm inputs must be finite probabilities")
    count = len(ordered)
    adjusted: dict[str, float] = {}
    running = 0.0
    for rank, (value, key) in enumerate(ordered):
        running = max(running, min(1.0, (count - rank) * value))
        adjusted[key] = running
    return adjusted


def _auc_draws_from_stratified_counts(
    *,
    scores: np.ndarray,
    positive_indices: np.ndarray,
    negative_indices: np.ndarray,
    positive_counts: np.ndarray,
    negative_counts: np.ndarray,
) -> np.ndarray:
    positive_scores = scores[positive_indices]
    negative_scores = scores[negative_indices]
    if (
        positive_counts.ndim != 2
        or negative_counts.ndim != 2
        or positive_counts.shape[0] != negative_counts.shape[0]
        or positive_counts.shape[1] != len(positive_scores)
        or negative_counts.shape[1] != len(negative_scores)
    ):
        raise ValueError("stratified bootstrap count matrices are misaligned")
    negative_order = np.argsort(negative_scores, kind="stable")
    ordered_negative = negative_scores[negative_order]
    cumulative = np.cumsum(negative_counts[:, negative_order], axis=1)
    cumulative = np.concatenate(
        [np.zeros((len(cumulative), 1), dtype=cumulative.dtype), cumulative], axis=1
    )
    left = np.searchsorted(ordered_negative, positive_scores, side="left")
    right = np.searchsorted(ordered_negative, positive_scores, side="right")
    lower = cumulative[:, left]
    through_equal = cumulative[:, right]
    credit = lower + 0.5 * (through_equal - lower)
    denominator = float(len(positive_scores) * len(negative_scores))
    return np.sum(positive_counts * credit, axis=1, dtype=np.float64) / denominator


def centered_paired_stratified_bootstrap_pvalue(
    records: Sequence[Mapping[str, Any]], *, samples: int, seed: int
) -> dict[str, float | int]:
    if not records or samples <= 0:
        raise ValueError("family bootstrap requires records and positive samples")
    observed_rows = []
    parsed = []
    for index, record in enumerate(records):
        labels = np.asarray(record["label"], dtype=np.int64)
        mse = np.asarray(record["mse_score"], dtype=np.float64)
        dpsae = np.asarray(record["dpsae_score"], dtype=np.float64)
        if labels.ndim != 1 or mse.shape != labels.shape or dpsae.shape != labels.shape:
            raise ValueError(f"family bootstrap record {index} has misaligned arrays")
        classes = sorted(np.unique(labels).tolist())
        if classes != [0, 1]:
            raise ValueError("family bootstrap requires both binary classes in every stratum")
        if not np.isfinite(mse).all() or not np.isfinite(dpsae).all():
            raise ValueError("family bootstrap scores must be finite")
        negative_indices = np.flatnonzero(labels == 0)
        positive_indices = np.flatnonzero(labels == 1)
        parsed.append((mse, dpsae, positive_indices, negative_indices))
        observed_rows.append(float(roc_auc_score(labels, dpsae) - roc_auc_score(labels, mse)))
    observed = float(np.mean(observed_rows))
    rng = np.random.default_rng(seed)
    draws = np.zeros(samples, dtype=np.float64)
    for mse, dpsae, positive_indices, negative_indices in parsed:
        positive_counts = rng.multinomial(
            len(positive_indices),
            np.full(len(positive_indices), 1 / len(positive_indices)),
            size=samples,
        )
        negative_counts = rng.multinomial(
            len(negative_indices),
            np.full(len(negative_indices), 1 / len(negative_indices)),
            size=samples,
        )
        common = {
            "positive_indices": positive_indices,
            "negative_indices": negative_indices,
            "positive_counts": positive_counts,
            "negative_counts": negative_counts,
        }
        draws += _auc_draws_from_stratified_counts(scores=dpsae, **common)
        draws -= _auc_draws_from_stratified_counts(scores=mse, **common)
    draws /= len(parsed)
    centered = draws - observed
    p_value = float((1 + np.count_nonzero(centered >= observed)) / (samples + 1))
    return {
        "estimate": observed,
        "p_value_one_sided_centered": p_value,
        "bootstrap_samples": samples,
        "record_count": len(records),
    }


def confirmation_checks(
    *,
    pair_macros: Mapping[int, float],
    pooled_interval: Mapping[str, float],
    complete_matrix: bool,
    matched_gates: bool,
    minimum_median: float,
) -> dict[str, bool]:
    values = [float(value) for value in pair_macros.values()]
    if len(values) != 3 or any(not math.isfinite(value) for value in values):
        raise ValueError("confirmation requires exactly three finite pair macros")
    return {
        "all_pair_macros_positive": all(value > 0 for value in values),
        "median_pair_macro": float(np.median(values)) >= minimum_median,
        "pooled_family_block_lower95": float(pooled_interval["lower"]) > 0,
        "complete_matrix": bool(complete_matrix),
        "all_matched_quality_gates": bool(matched_gates),
    }


def confirmation_gate_record(
    *,
    contract: Mapping[str, Any],
    contexts: Mapping[int, Mapping[str, Any]],
    checks: Mapping[str, bool],
    passed: bool,
) -> dict[str, Any]:
    pair_seeds = [int(value) for value in contract["pair_seeds"]]
    if len(pair_seeds) != 3 or set(pair_seeds) != set(contexts):
        raise RuntimeError("confirmation gate requires the exact three authorized pairs")
    return {
        "passed": bool(passed),
        "checkpoint_count": len(pair_seeds),
        "checkpoint_pair_count": len(pair_seeds),
        "pair_seeds": pair_seeds,
        "checkpoint_ids": {
            str(pair_seed): contexts[pair_seed]["config"]["pilot_checkpoint"][
                "checkpoint_id"
            ]
            for pair_seed in pair_seeds
        },
        "selected_requested_snapshot_tokens": int(
            contract["selected_requested_snapshot_tokens"]
        ),
        "checks": dict(checks),
    }


def _copy_file_atomic(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    shutil.copyfile(source, temporary)
    temporary.replace(destination)


def _payload_names(bundle: Mapping[str, Mapping[str, Any]], pair_seed: int) -> dict[str, str]:
    names: dict[str, str] = {}
    for name, payload in bundle.items():
        spec = payload.get("spec", {})
        if int(spec.get("seed", -1)) != pair_seed:
            continue
        method = str(spec.get("method"))
        if method in METHODS:
            if method in names:
                raise RuntimeError(f"pair {pair_seed} has duplicate {method} payloads")
            names[method] = str(name)
    if set(names) != set(METHODS):
        raise RuntimeError(f"pair {pair_seed} source bundle is incomplete")
    return names


def _selected_snapshot_sources(
    *,
    exp12_config: Mapping[str, Any],
    authorization: Mapping[str, Any],
) -> tuple[dict[int, dict[str, Any]], Path, Mapping[str, Any]]:
    decision_path = verify_file_record(
        authorization["prerequisites"]["maturity_stop_decision"],
        label="Exp12 maturity decision",
    )
    decision = json.loads(decision_path.read_text())
    selected = int(authorization["selected_requested_snapshot_tokens"])
    if (
        decision.get("complete") is not True
        or decision.get("common_checkpoint_selected") is not True
        or int(decision.get("selected_requested_snapshot_tokens", -1)) != selected
        or decision.get("config_digest") != exp12.canonical_digest(exp12_config)
    ):
        raise RuntimeError("Exp12 maturity decision changed before Exp13 freeze")
    pair_seeds = [int(value) for value in authorization["pair_seeds"]]
    sources: dict[int, dict[str, Any]] = {}
    for pair_seed in pair_seeds:
        manifest_record = decision["maturity_inputs"][str(pair_seed)]["snapshots"][
            str(selected)
        ]["manifest"]
        manifest_path = verify_file_record(
            manifest_record, label=f"Exp12 pair {pair_seed} snapshot manifest"
        )
        manifest = json.loads(manifest_path.read_text())
        models_record = manifest["artifacts"]["models"]
        models_path = verify_file_record(
            models_record, label=f"Exp12 pair {pair_seed} selected models"
        )
        sources[pair_seed] = {
            "snapshot_manifest": dict(manifest_record),
            "models": dict(models_record),
            "models_path": models_path,
        }
    shared_ready_path = decision_path.parent / "shared_ready.json"
    shared_ready = json.loads(shared_ready_path.read_text())
    if shared_ready.get("config_digest") != exp12.canonical_digest(exp12_config):
        raise RuntimeError("Exp12 shared inputs belong to another config")
    calibration_path = verify_file_record(
        shared_ready["artifacts"]["calibration"], label="Exp12 shared calibration"
    )
    return sources, calibration_path, decision


def _matched_quality_passed(decision: Mapping[str, Any], selected: int) -> bool:
    rows = [
        row
        for row in decision.get("candidate_checkpoints", [])
        if int(row.get("requested_snapshot_tokens", -1)) == selected
    ]
    if len(rows) != 1:
        return False
    matched = rows[0].get("matched_quality", {})
    return bool(matched) and all(
        bool(checks) and all(bool(value) for value in checks.values())
        for checks in matched.values()
    )


def _validate_pilot_inputs(
    *,
    config: Mapping[str, Any],
    base_config: Mapping[str, Any],
    pilot_root: Path,
    pilot_audit_path: Path,
    source_cache_ready_path: Path,
    model_cache: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    resolved_path = pilot_root / "resolved_config.json"
    timing_path = pilot_root / "timing_smoke.json"
    resolved = json.loads(resolved_path.read_text())
    if resolved.get("config_digest") != exp10.canonical_digest(base_config):
        raise RuntimeError("final Exp10 root does not match the pinned base config")
    audit = json.loads(pilot_audit_path.read_text())
    manifest_path = Path(str(audit.get("manifest_path", ""))).resolve()
    if (
        audit.get("complete") is not True
        or audit.get("passed") is not True
        or audit.get("phase") != config["base_experiment"]["required_final_audit_phase"]
        or audit.get("config_digest") != resolved["config_digest"]
        or not manifest_path.is_file()
        or file_sha256(manifest_path) != audit.get("manifest_sha256")
    ):
        raise RuntimeError("final Exp10 artifact audit is missing or stale")
    timing = json.loads(timing_path.read_text())
    projection = project_runtime(config, timing)
    if not projection["passed"]:
        raise RuntimeError("blind Exp13 projection exceeds 7.5 pod-hours")
    cache = json.loads(source_cache_ready_path.read_text())
    if (
        cache.get("config_digest") != resolved["config_digest"]
        or cache.get("dataset_manifest_sha256")
        != base_config["benchmark"]["dataset_manifest_sha256"]
        or set(cache.get("files", {})) != set(base_config["benchmark"]["datasets"])
    ):
        raise RuntimeError("final Exp10 cache manifest changed")
    expected_paths = exp10.cache_files(base_config, model_cache)
    for dataset, path in expected_paths.items():
        record = cache["files"][dataset]
        if (
            str(path.resolve()) != record.get("path")
            or path.stat().st_size != record.get("bytes")
            or file_sha256(path) != record.get("sha256")
        ):
            raise RuntimeError(f"final Exp10 cache file changed: {dataset}")
    return resolved, audit, projection


def _derived_pair_config(
    base_config: Mapping[str, Any], *, pair_seed: int, selected_tokens: int
) -> dict[str, Any]:
    value = copy.deepcopy(base_config)
    pilot = value["pilot_checkpoint"]
    pilot.update(
        {
            "checkpoint_id": f"pythia160m_block8_pair_{pair_seed}_{selected_tokens}",
            "artifact_directory": "exp13_generated_at_freeze",
            "model_payload_names": {
                "mse": f"mse_s{pair_seed}",
                "dpsae": f"dpsae_s{pair_seed}",
            },
            "training_tokens": selected_tokens,
        }
    )
    return value


def _write_pair_bundle(
    *,
    base_config: Mapping[str, Any],
    output_root: Path,
    pair_seed: int,
    selected_tokens: int,
    source: Mapping[str, Any],
    calibration_source: Path,
    source_cache_ready_path: Path,
    source_cache: Mapping[str, Any],
    model_cache: Path,
) -> dict[str, Any]:
    pair_root = output_root / "pairs" / f"seed_{pair_seed}"
    checkpoint = pair_root / "checkpoint"
    source_bundle = torch.load(source["models_path"], map_location="cpu", weights_only=False)
    names = _payload_names(source_bundle, pair_seed)
    if names != {"mse": f"mse_s{pair_seed}", "dpsae": f"dpsae_s{pair_seed}"}:
        raise RuntimeError("Exp12 selected payload names differ from the frozen pair convention")
    pair_bundle = {name: source_bundle[name] for name in names.values()}
    models_path = checkpoint / "models.pt"
    calibration_path = checkpoint / "calibration.pt"
    atomic_torch(models_path, pair_bundle)
    _copy_file_atomic(calibration_source, calibration_path)
    pair_config = _derived_pair_config(
        base_config, pair_seed=pair_seed, selected_tokens=selected_tokens
    )
    adapter_contract = {}
    for method in METHODS:
        adapter = exp10.load_adapter(pair_config, checkpoint, method, torch.device("cpu"))
        adapter_contract[method] = {
            "input_width": int(adapter.W_dec.shape[1]),
            "dictionary_width": int(adapter.W_dec.shape[0]),
            "activation_threshold": float(adapter.activation_threshold),
            "decoder_norm_max_deviation": float(
                (adapter.W_dec.norm(dim=1) - 1).abs().max()
            ),
        }
        del adapter
    evaluation_path = checkpoint / "evaluation.json"
    evaluation = {
        "schema_version": 1,
        "complete": True,
        "experiment": "exp13_exp10_compatible_checkpoint",
        "pair_seed": pair_seed,
        "selected_requested_snapshot_tokens": selected_tokens,
        "models_sha256": file_sha256(models_path),
        "calibration_sha256": file_sha256(calibration_path),
        "source_snapshot_manifest": source["snapshot_manifest"],
        "source_models": source["models"],
        "adapter_contract": adapter_contract,
        "concept_metrics_opened": False,
    }
    atomic_json(evaluation_path, evaluation)
    config_path = pair_root / "exp10_compatible_config.json"
    atomic_json(config_path, pair_config)
    artifacts = {
        "models_sha256": file_sha256(models_path),
        "calibration_sha256": file_sha256(calibration_path),
        "evaluation_sha256": file_sha256(evaluation_path),
    }
    resolved = {
        "schema_version": 1,
        "experiment_id": pair_config["experiment_id"],
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "config_digest": exp10.canonical_digest(pair_config),
        "checkpoint_directory": str(checkpoint.resolve()),
        "artifact_hashes": artifacts,
        "exp13_pair_seed": pair_seed,
        "source_snapshot_manifest": source["snapshot_manifest"],
        "source_models": source["models"],
    }
    atomic_json(pair_root / "resolved_config.json", resolved)
    adopted_cache = copy.deepcopy(source_cache)
    adopted_cache.update(
        {
            "config_digest": resolved["config_digest"],
            "adopted_from_cache_ready_path": str(source_cache_ready_path.resolve()),
            "adopted_from_cache_ready_sha256": file_sha256(source_cache_ready_path),
            "generation_timing_source": "exp13_hash_bound_no_regeneration",
            "exp13_pair_seed": pair_seed,
        }
    )
    atomic_json(pair_root / "cache_ready.json", adopted_cache)
    exp10.verify_cache_ready(pair_config, pair_root, model_cache)
    return {
        "pair_seed": pair_seed,
        "pair_root": str(pair_root.resolve()),
        "checkpoint_directory": str(checkpoint.resolve()),
        "config": file_record(config_path),
        "resolved": file_record(pair_root / "resolved_config.json"),
        "cache_ready": file_record(pair_root / "cache_ready.json"),
        "models": file_record(models_path),
        "calibration": file_record(calibration_path),
        "evaluation": file_record(evaluation_path),
        "source_snapshot_manifest": dict(source["snapshot_manifest"]),
        "source_models": dict(source["models"]),
    }


def freeze_run(
    *,
    config_path: Path,
    base_config_path: Path,
    exp12_config_path: Path,
    exp12_root: Path,
    pilot_root: Path,
    pilot_audit_path: Path,
    source_cache_ready_path: Path,
    model_cache: Path,
    saebench_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    existing_path = output_root / "run_contract.json"
    if existing_path.is_file():
        return load_contract(output_root, config)
    repository = repository_state()
    base_config = exp10.load_config(base_config_path)
    if base_config.get("experiment_id") != config["base_experiment"]["required_experiment_id"]:
        raise RuntimeError("Exp13 base experiment identity changed")
    if int(base_config["statistics"]["primary_k"]) != int(
        config["statistics"]["primary_k"]
    ):
        raise RuntimeError("Exp13 and its pinned Exp10 base config disagree on primary k")
    exp12_config = exp12.load_config(exp12_config_path, require_frozen=True)
    authorization_path = exp12_root / config["authorization"]["artifact_name"]
    authorization = json.loads(authorization_path.read_text())
    if (
        authorization.get("complete") is not True
        or authorization.get("authorized") is not True
        or authorization.get("experiment") != config["authorization"]["required_experiment"]
        or authorization.get("config_digest") != exp12.canonical_digest(exp12_config)
        or authorization.get("excluded_pilot_checkpoint_id")
        != config["authorization"]["required_excluded_pilot_checkpoint_id"]
        or authorization.get("confirmatory_inference")
        != config["authorization"]["required_confirmatory_inference"]
        or authorization.get("maturity_inputs_were_concept_blind") is not True
        or authorization.get("concept_outcomes_opened_by_this_stage") is not False
    ):
        raise RuntimeError("Exp12 did not authorize confirmatory concept evaluation")
    pair_seeds = [int(value) for value in authorization["pair_seeds"]]
    if len(pair_seeds) != 3 or len(set(pair_seeds)) != 3:
        raise RuntimeError("Exp12 authorization does not contain three unique pairs")
    selected = int(authorization["selected_requested_snapshot_tokens"])
    sources, calibration_source, decision = _selected_snapshot_sources(
        exp12_config=exp12_config, authorization=authorization
    )
    if not _matched_quality_passed(decision, selected):
        raise RuntimeError("Exp12 selected checkpoint no longer passes every matched gate")
    pilot_resolved, pilot_audit, projection = _validate_pilot_inputs(
        config=config,
        base_config=base_config,
        pilot_root=pilot_root,
        pilot_audit_path=pilot_audit_path,
        source_cache_ready_path=source_cache_ready_path,
        model_cache=model_cache,
    )
    dependency = exp10.verify_saebench_environment(base_config, saebench_root)
    source_cache = json.loads(source_cache_ready_path.read_text())
    pair_records = [
        _write_pair_bundle(
            base_config=base_config,
            output_root=output_root,
            pair_seed=pair_seed,
            selected_tokens=selected,
            source=sources[pair_seed],
            calibration_source=calibration_source,
            source_cache_ready_path=source_cache_ready_path,
            source_cache=source_cache,
            model_cache=model_cache,
        )
        for pair_seed in pair_seeds
    ]
    shards = realized_shards(pair_seeds, base_config["benchmark"]["probe_seeds"])
    shard_path = output_root / "shard_manifest.json"
    atomic_json(shard_path, shards)
    timing_path = output_root / "timing_gate.json"
    atomic_json(timing_path, projection)
    contract = {
        "schema_version": 1,
        "complete": True,
        "experiment_id": config["experiment_id"],
        "config_digest": canonical_digest(config),
        "repository": repository,
        "source_hashes": source_hashes(config),
        "inputs": {
            "config": file_record(config_path),
            "base_config": file_record(base_config_path),
            "exp12_config": file_record(exp12_config_path),
            "exp12_authorization": file_record(authorization_path),
            "pilot_resolved": file_record(pilot_root / "resolved_config.json"),
            "pilot_timing": file_record(pilot_root / "timing_smoke.json"),
            "pilot_final_audit": file_record(pilot_audit_path),
            "pilot_final_audit_manifest": file_record(Path(pilot_audit["manifest_path"])),
            "source_cache_ready": file_record(source_cache_ready_path),
            "exp12_calibration": file_record(calibration_source),
        },
        "pilot_resolved_config_digest": pilot_resolved["config_digest"],
        "dependency_preflight": dependency,
        "selected_requested_snapshot_tokens": selected,
        "pair_seeds": pair_seeds,
        "probe_seeds": [int(value) for value in base_config["benchmark"]["probe_seeds"]],
        "pairs": pair_records,
        "shard_manifest": file_record(shard_path),
        "timing_gate": file_record(timing_path),
        "timing_projection": projection,
        "model_cache": str(model_cache.resolve()),
        "saebench_root": str(saebench_root.resolve()),
        "concept_metrics_opened_by_freeze": False,
    }
    contract["contract_sha256"] = canonical_digest(contract)
    atomic_json(existing_path, contract)
    atomic_json(
        output_root / "status.json",
        {
            "schema_version": 1,
            "complete": False,
            "state": "frozen_ready_for_workers",
            "contract_sha256": contract["contract_sha256"],
            "projected_pod_hours": projection["projected_pod_hours"],
        },
    )
    return contract


def load_contract(output_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = output_root / "run_contract.json"
    if not path.is_file():
        raise RuntimeError("freeze Exp13 before running workers")
    contract = json.loads(path.read_text())
    digest = contract.pop("contract_sha256", None)
    if digest != canonical_digest(contract):
        raise RuntimeError("Exp13 run-contract digest changed")
    contract["contract_sha256"] = digest
    if (
        contract.get("complete") is not True
        or contract.get("config_digest") != canonical_digest(config)
        or contract.get("timing_projection", {}).get("passed") is not True
        or contract.get("concept_metrics_opened_by_freeze") is not False
    ):
        raise RuntimeError("Exp13 run contract is incomplete or stale")
    for label, record in contract["inputs"].items():
        verify_file_record(record, label=f"Exp13 input {label}")
    verify_file_record(contract["shard_manifest"], label="Exp13 shard manifest")
    verify_file_record(contract["timing_gate"], label="Exp13 timing gate")
    return contract


def _pair_contexts(contract: Mapping[str, Any]) -> dict[int, dict[str, Any]]:
    contexts = {}
    for record in contract["pairs"]:
        pair_seed = int(record["pair_seed"])
        config_path = verify_file_record(
            record["config"], label=f"pair {pair_seed} derived config"
        )
        verify_file_record(record["resolved"], label=f"pair {pair_seed} resolved config")
        verify_file_record(record["cache_ready"], label=f"pair {pair_seed} cache adoption")
        for name in ("models", "calibration", "evaluation"):
            verify_file_record(record[name], label=f"pair {pair_seed} {name}")
        for name in ("source_snapshot_manifest", "source_models"):
            verify_file_record(record[name], label=f"pair {pair_seed} {name}")
        pair_config = json.loads(config_path.read_text())
        pair_root = Path(record["pair_root"]).resolve()
        resolved = exp10.load_resolved(pair_root, pair_config)
        if int(resolved.get("exp13_pair_seed", -1)) != pair_seed:
            raise RuntimeError("pair resolved-config identity drift")
        contexts[pair_seed] = {
            "config": pair_config,
            "root": pair_root,
            "checkpoint": Path(record["checkpoint_directory"]).resolve(),
        }
    if set(contexts) != {int(value) for value in contract["pair_seeds"]}:
        raise RuntimeError("Exp13 pair contexts do not cover the authorization")
    return contexts


def _status_path(output_root: Path, worker_index: int) -> Path:
    return output_root / "worker_status" / f"worker_{worker_index}.json"


def _write_worker_status(
    output_root: Path,
    *,
    worker_index: int,
    state: str,
    contract_sha256: str,
    extra: Mapping[str, Any] | None = None,
) -> None:
    atomic_json(
        _status_path(output_root, worker_index),
        {
            "schema_version": 1,
            "complete": state in {"complete", "failed"},
            "failed": state == "failed",
            "state": state,
            "worker_index": worker_index,
            "contract_sha256": contract_sha256,
            "written_at_unix_seconds": time.time(),
            **dict(extra or {}),
        },
    )


def run_worker(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    worker_index: int,
    device: str,
) -> dict[str, Any]:
    contract = load_contract(output_root, config)
    if worker_index not in range(4):
        raise ValueError("Exp13 worker index must be in [0, 3]")
    shards = json.loads(Path(contract["shard_manifest"]["path"]).read_text())
    expected_shards = realized_shards(contract["pair_seeds"], contract["probe_seeds"])
    if shards != expected_shards:
        raise RuntimeError("Exp13 realized shard manifest changed")
    shard = shards["workers"][worker_index]
    _write_worker_status(
        output_root,
        worker_index=worker_index,
        state="running",
        contract_sha256=contract["contract_sha256"],
        extra={
            "sparse_job_count": len(shard["sparse"]),
            "companion_job_count": len(shard["companion"]),
        },
    )
    started = time.monotonic()
    try:
        check_abort(config, output_root)
        base_config = exp10.load_config(Path(contract["inputs"]["base_config"]["path"]))
        dependency = exp10.verify_saebench_environment(
            base_config, Path(contract["saebench_root"])
        )
        contexts = _pair_contexts(contract)
        torch_device = torch.device(device)
        adapters = {
            pair_seed: {
                method: exp10.load_adapter(
                    context["config"], context["checkpoint"], method, torch_device
                )
                for method in METHODS
            }
            for pair_seed, context in contexts.items()
        }
        sparse_done = []
        companion_done = []
        model_cache = Path(contract["model_cache"])
        for assignment in shard["sparse"]:
            check_abort(config, output_root)
            pair_seed = int(assignment["pair_seed"])
            method = str(assignment["method"])
            seed = int(assignment["probe_seed"])
            context = contexts[pair_seed]
            exp10.run_sparse_job(
                config=context["config"],
                output_root=context["root"],
                checkpoint_dir=context["checkpoint"],
                model_cache=model_cache,
                method=method,
                probe_seed=seed,
                device=device,
                adapter=adapters[pair_seed][method],
            )
            checkpoint_id = context["config"]["pilot_checkpoint"]["checkpoint_id"]
            path = context["root"] / "jobs" / checkpoint_id / method / f"seed_{seed}/done.json"
            sparse_done.append({**assignment, "done": file_record(path)})
        for assignment in shard["companion"]:
            check_abort(config, output_root)
            pair_seed = int(assignment["pair_seed"])
            seed = int(assignment["probe_seed"])
            context = contexts[pair_seed]
            exp10.run_companion_job(
                config=context["config"],
                output_root=context["root"],
                checkpoint_dir=context["checkpoint"],
                model_cache=model_cache,
                probe_seed=seed,
                device=device,
                adapters=adapters[pair_seed],
            )
            checkpoint_id = context["config"]["pilot_checkpoint"]["checkpoint_id"]
            path = context["root"] / "companion" / checkpoint_id / f"seed_{seed}/done.json"
            companion_done.append({**assignment, "done": file_record(path)})
        result = {
            "schema_version": 1,
            "complete": True,
            "failed": False,
            "worker_index": worker_index,
            "contract_sha256": contract["contract_sha256"],
            "shard_sha256": canonical_digest(shard),
            "dependency_preflight": dependency,
            "device": device,
            "sparse": sparse_done,
            "companion": companion_done,
            "wall_seconds": time.monotonic() - started,
        }
        summary_path = output_root / "workers" / f"worker_{worker_index}.json"
        atomic_json(summary_path, result)
        _write_worker_status(
            output_root,
            worker_index=worker_index,
            state="complete",
            contract_sha256=contract["contract_sha256"],
            extra={"summary": file_record(summary_path)},
        )
        return result
    except Exception as error:
        request_abort(config=config, output_root=output_root, reason=str(error))
        _write_worker_status(
            output_root,
            worker_index=worker_index,
            state="failed",
            contract_sha256=contract["contract_sha256"],
            extra={"error_type": type(error).__name__, "error": str(error)},
        )
        raise


def wait_for_workers(output_root: Path, *, timeout_seconds: float, poll_seconds: float) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        statuses = []
        for index in range(4):
            path = _status_path(output_root, index)
            statuses.append(json.loads(path.read_text()) if path.is_file() else None)
        failures = [row for row in statuses if row and row.get("failed") is True]
        if failures:
            raise RuntimeError(f"Exp13 worker failed: {failures[0].get('error')}")
        if all(row and row.get("state") == "complete" for row in statuses):
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for Exp13 workers")
        time.sleep(min(poll_seconds, max(0.1, deadline - time.monotonic())))


def _validate_worker_summaries(
    contract: Mapping[str, Any], output_root: Path
) -> list[dict[str, Any]]:
    shards = realized_shards(contract["pair_seeds"], contract["probe_seeds"])
    summaries = []
    sparse_observed = []
    companion_observed = []
    for index in range(4):
        status = json.loads(_status_path(output_root, index).read_text())
        if (
            status.get("state") != "complete"
            or status.get("failed") is not False
            or status.get("contract_sha256") != contract["contract_sha256"]
        ):
            raise RuntimeError(f"Exp13 worker {index} is incomplete")
        summary_path = verify_file_record(status["summary"], label=f"worker {index} summary")
        summary = json.loads(summary_path.read_text())
        shard = shards["workers"][index]
        if (
            summary.get("worker_index") != index
            or summary.get("contract_sha256") != contract["contract_sha256"]
            or summary.get("shard_sha256") != canonical_digest(shard)
            or len(summary.get("sparse", [])) != len(shard["sparse"])
            or len(summary.get("companion", [])) != len(shard["companion"])
        ):
            raise RuntimeError(f"Exp13 worker {index} summary differs from its shard")
        for row in summary["sparse"]:
            verify_file_record(row["done"], label="sparse done")
            sparse_observed.append(
                (int(row["pair_seed"]), str(row["method"]), int(row["probe_seed"]))
            )
        for row in summary["companion"]:
            verify_file_record(row["done"], label="companion done")
            companion_observed.append((int(row["pair_seed"]), int(row["probe_seed"])))
        summaries.append(summary)
    expected_sparse = [
        (int(pair), method, int(seed))
        for pair in contract["pair_seeds"]
        for method in METHODS
        for seed in contract["probe_seeds"]
    ]
    expected_companion = [
        (int(pair), int(seed))
        for pair in contract["pair_seeds"]
        for seed in contract["probe_seeds"]
    ]
    if Counter(sparse_observed) != Counter(expected_sparse):
        raise RuntimeError("Exp13 sparse worker matrix is incomplete or duplicated")
    if Counter(companion_observed) != Counter(expected_companion):
        raise RuntimeError("Exp13 companion worker matrix is incomplete or duplicated")
    return summaries


def _scientific_artifact_records(output_root: Path, *, phase: str) -> list[dict[str, Any]]:
    excluded = {
        output_root / f"artifact_manifest_{phase.replace('-', '_')}.jsonl",
        output_root / f"artifact_audit_{phase.replace('-', '_')}.json",
        output_root / "finalizer_status.json",
    }
    records = []
    for path in sorted(output_root.rglob("*")):
        if path in excluded or "logs" in path.relative_to(output_root).parts:
            continue
        if path.is_symlink():
            raise RuntimeError(f"Exp13 artifacts may not contain symlinks: {path}")
        if path.is_file():
            if path.name.endswith((".tmp", ".partial", ".part")):
                raise RuntimeError(f"unfinished Exp13 artifact exists: {path}")
            records.append(
                {
                    "path": str(path.relative_to(output_root)),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha256(path),
                }
            )
    return records


def audit_artifacts(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    phase: str,
    wait_seconds: float = 0,
) -> dict[str, Any]:
    if phase not in {"pre-aggregate", "final"}:
        raise ValueError("Exp13 audit phase must be pre-aggregate or final")
    if wait_seconds > 0:
        wait_for_workers(
            output_root,
            timeout_seconds=wait_seconds,
            poll_seconds=float(config["runtime"]["status_poll_seconds"]),
        )
    contract = load_contract(output_root, config)
    summaries = _validate_worker_summaries(contract, output_root)
    contexts = _pair_contexts(contract)
    sparse_count = 0
    companion_count = 0
    for context in contexts.values():
        sparse = exp10._load_sparse_records(context["config"], context["root"])
        companion = exp10._load_companion_records(context["config"], context["root"])
        sparse_count += len(sparse)
        companion_count += len(companion)
    datasets = len(next(iter(contexts.values()))["config"]["benchmark"]["datasets"])
    expected_sparse = 3 * 2 * 10 * datasets
    expected_companion = 3 * 10 * datasets
    if sparse_count != expected_sparse or companion_count != expected_companion:
        raise RuntimeError("Exp13 audited matrix has missing or extra dataset records")
    if phase == "final":
        aggregate_path = output_root / "confirmation_report.json"
        candidates_path = output_root / "candidate_associations.jsonl"
        candidate_manifest_path = output_root / "candidate_manifest.json"
        for path in (aggregate_path, candidates_path, candidate_manifest_path):
            if not path.is_file():
                raise RuntimeError(f"Exp13 final artifact is missing: {path}")
        aggregate = json.loads(aggregate_path.read_text())
        candidate_manifest = json.loads(candidate_manifest_path.read_text())
        candidates = [
            json.loads(line)
            for line in candidates_path.read_text().splitlines()
            if line.strip()
        ]
        expected_autointerp = bool(aggregate.get("confirmation_passed")) and bool(
            candidates
        )
        if (
            aggregate.get("complete") is not True
            or aggregate.get("config_digest") != canonical_digest(config)
            or aggregate.get("candidate_manifest_sha256") != file_sha256(candidate_manifest_path)
            or candidate_manifest.get("candidate_jsonl_sha256") != file_sha256(candidates_path)
            or candidate_manifest.get("confirmation_passed")
            != aggregate.get("confirmation_passed")
            or candidate_manifest.get("confirmation_gate")
            != aggregate.get("confirmation_gate")
            or int(candidate_manifest.get("candidate_count", -1)) != len(candidates)
            or candidate_manifest.get("autointerp_eligible") != expected_autointerp
            or any(row.get("autointerp_eligible") is not True for row in candidates)
        ):
            raise RuntimeError("Exp13 aggregate or candidate manifest changed")
    records = _scientific_artifact_records(output_root, phase=phase)
    token = phase.replace("-", "_")
    manifest_path = output_root / f"artifact_manifest_{token}.jsonl"
    audit_path = output_root / f"artifact_audit_{token}.json"
    atomic_jsonl(manifest_path, records)
    report = {
        "schema_version": 1,
        "complete": True,
        "passed": True,
        "phase": phase,
        "config_digest": canonical_digest(config),
        "contract_sha256": contract["contract_sha256"],
        "worker_count": len(summaries),
        "sparse_dataset_record_count": sparse_count,
        "companion_dataset_record_count": companion_count,
        "expected_sparse_dataset_record_count": expected_sparse,
        "expected_companion_dataset_record_count": expected_companion,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_entry_count": len(records),
        "manifest_sha256": file_sha256(manifest_path),
    }
    atomic_json(audit_path, report)
    return report


def _primary_results(
    contract: Mapping[str, Any], contexts: Mapping[int, Mapping[str, Any]]
) -> tuple[
    dict[int, dict[str, dict[str, dict[str, float]]]],
    dict[int, dict[str, float]],
    dict[int, dict[tuple[str, int, str], dict[str, Any]]],
]:
    pair_metrics = {}
    pair_task_deltas = {}
    sparse_by_pair = {}
    for pair_seed, context in contexts.items():
        config = context["config"]
        sparse = exp10._load_sparse_records(config, context["root"])
        sparse_by_pair[pair_seed] = sparse
        metrics: dict[str, dict[str, dict[str, float]]] = {}
        deltas = {}
        for dataset in config["benchmark"]["datasets"]:
            metrics[dataset] = {}
            for method in METHODS:
                rows = []
                for probe_seed in contract["probe_seeds"]:
                    record = sparse[(method, int(probe_seed), dataset)]
                    row = next(
                        value
                        for value in record["rows"]
                        if int(value["k"]) == int(config["statistics"]["primary_k"])
                    )
                    rows.append(row["metrics"])
                metrics[dataset][method] = {
                    metric: float(np.mean([float(row[metric]) for row in rows]))
                    for metric in ("test_auc", "test_acc", "test_f1")
                }
            deltas[dataset] = (
                metrics[dataset]["dpsae"]["test_auc"]
                - metrics[dataset]["mse"]["test_auc"]
            )
        pair_metrics[pair_seed] = metrics
        pair_task_deltas[pair_seed] = deltas
    return pair_metrics, pair_task_deltas, sparse_by_pair


def _companion_results(
    contract: Mapping[str, Any], contexts: Mapping[int, Mapping[str, Any]]
) -> dict[int, dict[str, Any]]:
    result = {}
    for pair_seed, context in contexts.items():
        config = context["config"]
        companion = exp10._load_companion_records(config, context["root"])
        tasks = {}
        for dataset in config["benchmark"]["datasets"]:
            rows = [
                companion[(int(probe_seed), dataset)]["metrics"]
                for probe_seed in contract["probe_seeds"]
            ]
            original = {
                metric: float(np.mean([row["original_residual"][metric] for row in rows]))
                for metric in ("test_auc", "test_acc", "test_f1")
            }
            methods = {}
            for method in METHODS:
                methods[method] = {}
                for representation in ("reconstruction", "full_code"):
                    methods[method][representation] = {
                        metric: float(
                            np.mean(
                                [row["methods"][method][representation][metric] for row in rows]
                            )
                        )
                        for metric in ("test_auc", "test_acc", "test_f1")
                    }
            tasks[dataset] = {"original_residual": original, "methods": methods}
        result[pair_seed] = tasks
    return result


def _family_prediction_records(
    *,
    contract: Mapping[str, Any],
    contexts: Mapping[int, Mapping[str, Any]],
    family: str,
) -> list[dict[str, Any]]:
    records = []
    for pair_seed, context in contexts.items():
        config = context["config"]
        checkpoint_id = config["pilot_checkpoint"]["checkpoint_id"]
        datasets = [
            dataset
            for dataset in config["benchmark"]["datasets"]
            if config["benchmark"]["family_by_dataset"][dataset] == family
        ]
        for dataset in datasets:
            for probe_seed in contract["probe_seeds"]:
                payloads = {}
                identities = {}
                for method in METHODS:
                    root = (
                        context["root"]
                        / "jobs"
                        / checkpoint_id
                        / method
                        / f"seed_{probe_seed}"
                    )
                    provenance_path = root / "provenance" / f"{dataset}.json"
                    provenance = json.loads(provenance_path.read_text())
                    prediction_path = root / "predictions" / f"{dataset}.pt"
                    if provenance.get("heldout_predictions_sha256") != file_sha256(
                        prediction_path
                    ):
                        raise RuntimeError("family-test prediction hash changed")
                    payloads[method] = torch.load(
                        prediction_path, map_location="cpu", weights_only=False
                    )
                    identities[method] = {
                        key: payloads[method][key]
                        for key in (
                            "split",
                            "split_id",
                            "example_id_policy",
                            "example_ids",
                        )
                    }
                if identities["mse"] != identities["dpsae"]:
                    raise RuntimeError("MSE and DPSAE held-out identities do not align")
                mse_label = torch.as_tensor(payloads["mse"]["label"]).numpy()
                dpsae_label = torch.as_tensor(payloads["dpsae"]["label"]).numpy()
                if not np.array_equal(mse_label, dpsae_label):
                    raise RuntimeError("MSE and DPSAE held-out labels do not align")
                if len(identities["mse"]["example_ids"]) != len(mse_label):
                    raise RuntimeError("held-out example IDs and labels do not align")
                records.append(
                    {
                        "label": mse_label,
                        "mse_score": torch.as_tensor(
                            payloads["mse"]["by_k"]["5"]["decision_score"]
                        ).numpy(),
                        "dpsae_score": torch.as_tensor(
                            payloads["dpsae"]["by_k"]["5"]["decision_score"]
                        ).numpy(),
                    }
                )
    return records


def _candidate_pool(
    *,
    config: Mapping[str, Any],
    contract: Mapping[str, Any],
    contexts: Mapping[int, Mapping[str, Any]],
    pair_task_deltas: Mapping[int, Mapping[str, float]],
    sparse_by_pair: Mapping[
        int, Mapping[tuple[str, int, str], Mapping[str, Any]]
    ],
) -> tuple[list[dict[str, Any]], list[str]]:
    positive_tasks = sorted(
        dataset
        for dataset in next(iter(pair_task_deltas.values()))
        if all(float(pair_task_deltas[pair_seed][dataset]) > 0 for pair_seed in pair_task_deltas)
    )
    minimum = int(config["candidates"]["minimum_probe_seed_count"])
    primary_k = int(config["statistics"]["primary_k"])
    by_method: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    pair_contracts = {int(row["pair_seed"]): row for row in contract["pairs"]}
    if set(pair_contracts) != set(contexts):
        raise RuntimeError("candidate checkpoint records do not cover every pair")
    for pair_seed, context in contexts.items():
        pair_config = context["config"]
        checkpoint_id = pair_config["pilot_checkpoint"]["checkpoint_id"]
        sparse = sparse_by_pair[pair_seed]
        for method in METHODS:
            for dataset in positive_tasks:
                features: dict[
                    int, dict[int, tuple[float, dict[str, Any], dict[str, Any]]]
                ] = {}
                for probe_seed in contract["probe_seeds"]:
                    record = sparse[(method, int(probe_seed), dataset)]
                    row = next(value for value in record["rows"] if int(value["k"]) == primary_k)
                    root = (
                        context["root"]
                        / "jobs"
                        / checkpoint_id
                        / method
                        / f"seed_{probe_seed}"
                    )
                    provenance_record = file_record(root / "provenance" / f"{dataset}.json")
                    prediction_record = file_record(root / "predictions" / f"{dataset}.pt")
                    for feature in row["feature_weights"]:
                        feature_id = int(feature["feature_id"])
                        by_seed = features.setdefault(feature_id, {})
                        if int(probe_seed) in by_seed:
                            raise RuntimeError("selected feature repeats within one sparse fit")
                        by_seed[int(probe_seed)] = (
                            float(feature["weight"]),
                            provenance_record,
                            prediction_record,
                        )
                for feature_id, by_seed in features.items():
                    values = [(seed, *by_seed[seed]) for seed in sorted(by_seed)]
                    if len(values) < minimum:
                        continue
                    identity = {
                        "pair_seed": pair_seed,
                        "checkpoint_id": checkpoint_id,
                        "method": method,
                        "dataset": dataset,
                        "feature_id": feature_id,
                    }
                    by_method[method].append(
                        {
                            "candidate_id": "exp13_candidate_" + canonical_digest(identity)[:24],
                            **identity,
                            "autointerp_eligible": True,
                            "family": pair_config["benchmark"]["family_by_dataset"][dataset],
                            "checkpoint_artifacts": {
                                name: dict(pair_contracts[pair_seed][name])
                                for name in (
                                    "models",
                                    "calibration",
                                    "evaluation",
                                    "source_snapshot_manifest",
                                    "source_models",
                                )
                            },
                            "probe_seed_count": len(values),
                            "probe_seed_frequency": len(values) / len(contract["probe_seeds"]),
                            "probe_seeds": [value[0] for value in values],
                            "mean_weight": float(np.mean([value[1] for value in values])),
                            "mean_absolute_weight": float(
                                np.mean([abs(value[1]) for value in values])
                            ),
                            "contributing_artifacts": [
                                {
                                    "probe_seed": seed,
                                    "provenance": provenance,
                                    "predictions": predictions,
                                }
                                for seed, _weight, provenance, predictions in values
                            ],
                        }
                    )
    for method in METHODS:
        by_method[method].sort(
            key=lambda row: (
                -row["probe_seed_frequency"],
                -row["mean_absolute_weight"],
                row["dataset"],
                row["feature_id"],
                row["pair_seed"],
            )
        )
    budget = min(
        int(config["candidates"]["maximum_per_method"]),
        len(by_method["mse"]),
        len(by_method["dpsae"]),
    )
    selected = [*by_method["mse"][:budget], *by_method["dpsae"][:budget]]
    return selected, positive_tasks


def aggregate_confirmation(
    *, config: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    contract = load_contract(output_root, config)
    pre_audit_path = output_root / "artifact_audit_pre_aggregate.json"
    pre_audit = json.loads(pre_audit_path.read_text())
    if (
        pre_audit.get("complete") is not True
        or pre_audit.get("passed") is not True
        or pre_audit.get("contract_sha256") != contract["contract_sha256"]
        or file_sha256(Path(pre_audit["manifest_path"])) != pre_audit["manifest_sha256"]
    ):
        raise RuntimeError("Exp13 requires a valid pre-aggregation audit")
    contexts = _pair_contexts(contract)
    pair_metrics, pair_task_deltas, sparse_by_pair = _primary_results(contract, contexts)
    companion = _companion_results(contract, contexts)
    pair_macros = {
        pair_seed: float(np.mean(list(deltas.values())))
        for pair_seed, deltas in pair_task_deltas.items()
    }
    pooled_task_deltas = {
        dataset: float(
            np.mean([pair_task_deltas[pair_seed][dataset] for pair_seed in pair_task_deltas])
        )
        for dataset in next(iter(pair_task_deltas.values()))
    }
    base_config = next(iter(contexts.values()))["config"]
    stats = config["statistics"]
    pooled_interval = exp10.family_block_bootstrap(
        pooled_task_deltas,
        base_config["benchmark"]["family_by_dataset"],
        samples=int(stats["bootstrap_samples"]),
        seed=int(stats["bootstrap_seed"]),
        confidence_level=float(stats["confidence_level"]),
    )
    authorization = json.loads(Path(contract["inputs"]["exp12_authorization"]["path"]).read_text())
    decision = json.loads(
        Path(authorization["prerequisites"]["maturity_stop_decision"]["path"]).read_text()
    )
    matched = _matched_quality_passed(
        decision, int(contract["selected_requested_snapshot_tokens"])
    )
    complete_matrix = (
        pre_audit["sparse_dataset_record_count"]
        == pre_audit["expected_sparse_dataset_record_count"]
        and pre_audit["companion_dataset_record_count"]
        == pre_audit["expected_companion_dataset_record_count"]
    )
    checks = confirmation_checks(
        pair_macros=pair_macros,
        pooled_interval=pooled_interval,
        complete_matrix=complete_matrix,
        matched_gates=matched,
        minimum_median=float(stats["minimum_median_pair_macro"]),
    )
    confirmation_passed = all(checks.values())

    family_tests = {}
    raw_p = {}
    families = sorted(set(base_config["benchmark"]["family_by_dataset"].values()))
    family_spec = stats["family_tests"]
    for index, family in enumerate(families):
        records = _family_prediction_records(
            contract=contract, contexts=contexts, family=family
        )
        result = centered_paired_stratified_bootstrap_pvalue(
            records,
            samples=int(family_spec["samples"]),
            seed=int(family_spec["seed"]) + index,
        )
        family_tests[family] = result
        raw_p[family] = float(result["p_value_one_sided_centered"])
    adjusted = holm_adjust(raw_p)
    for family in families:
        family_tests[family]["p_value_holm"] = adjusted[family]
        family_tests[family]["gate_forming"] = False

    candidates = []
    positive_tasks = sorted(
        dataset
        for dataset in pooled_task_deltas
        if all(pair_task_deltas[pair_seed][dataset] > 0 for pair_seed in pair_task_deltas)
    )
    if confirmation_passed:
        candidates, positive_tasks = _candidate_pool(
            config=config,
            contract=contract,
            contexts=contexts,
            pair_task_deltas=pair_task_deltas,
            sparse_by_pair=sparse_by_pair,
        )
    candidate_path = output_root / "candidate_associations.jsonl"
    atomic_jsonl(candidate_path, candidates)
    counts = Counter(row["method"] for row in candidates)
    confirmation_gate = confirmation_gate_record(
        contract=contract,
        contexts=contexts,
        checks=checks,
        passed=confirmation_passed,
    )
    candidate_manifest = {
        "schema_version": 1,
        "complete": True,
        "config_digest": canonical_digest(config),
        "confirmation_passed": confirmation_passed,
        "confirmation_gate": confirmation_gate,
        "candidate_count": len(candidates),
        "candidate_count_by_method": {method: counts.get(method, 0) for method in METHODS},
        "equal_method_budget": counts.get("mse", 0) == counts.get("dpsae", 0),
        "maximum_per_method": int(config["candidates"]["maximum_per_method"]),
        "positive_in_all_pairs_task_count": len(positive_tasks),
        "positive_in_all_pairs_tasks": positive_tasks,
        "never_quota_fill": True,
        "candidate_jsonl_sha256": file_sha256(candidate_path),
        "autointerp_eligible": confirmation_passed and bool(candidates),
    }
    candidate_manifest_path = output_root / "candidate_manifest.json"
    atomic_json(candidate_manifest_path, candidate_manifest)
    report = {
        "schema_version": 1,
        "complete": True,
        "experiment_id": config["experiment_id"],
        "config_digest": canonical_digest(config),
        "contract_sha256": contract["contract_sha256"],
        "primary_metric": "paired_dpsae_minus_mse_test_auc_at_k5",
        "probe_seed_aggregation": stats["probe_seed_aggregation"],
        "pair_macros": {str(key): value for key, value in pair_macros.items()},
        "median_pair_macro": float(np.median(list(pair_macros.values()))),
        "pooled_family_block_interval": pooled_interval,
        "checks": checks,
        "confirmation_gate": confirmation_gate,
        "confirmation_passed": confirmation_passed,
        "task_metrics_by_pair": {str(key): value for key, value in pair_metrics.items()},
        "task_deltas_by_pair": {str(key): value for key, value in pair_task_deltas.items()},
        "pooled_task_deltas": pooled_task_deltas,
        "companion_task_metrics_by_pair": {str(key): value for key, value in companion.items()},
        "family_tests_reporting_only": family_tests,
        "family_tests_gate_forming": False,
        "pre_aggregate_audit": file_record(pre_audit_path),
        "candidate_manifest_sha256": file_sha256(candidate_manifest_path),
    }
    atomic_json(output_root / "confirmation_report.json", report)
    atomic_json(
        output_root / "status.json",
        {
            "schema_version": 1,
            "complete": True,
            "state": "complete",
            "confirmation_passed": confirmation_passed,
            "contract_sha256": contract["contract_sha256"],
        },
    )
    return report


def finalize_run(
    *, config: Mapping[str, Any], output_root: Path, wait_seconds: float
) -> dict[str, Any]:
    contract = load_contract(output_root, config)
    status_path = output_root / "finalizer_status.json"
    atomic_json(
        status_path,
        {
            "schema_version": 1,
            "complete": False,
            "failed": False,
            "state": "waiting_for_workers",
            "contract_sha256": contract["contract_sha256"],
            "written_at_unix_seconds": time.time(),
        },
    )
    try:
        pre = audit_artifacts(
            config=config,
            output_root=output_root,
            phase="pre-aggregate",
            wait_seconds=wait_seconds,
        )
        aggregate = aggregate_confirmation(config=config, output_root=output_root)
        final = audit_artifacts(
            config=config, output_root=output_root, phase="final"
        )
        result = {
            "schema_version": 1,
            "complete": True,
            "failed": False,
            "state": "complete",
            "contract_sha256": contract["contract_sha256"],
            "confirmation_passed": bool(aggregate["confirmation_passed"]),
            "pre_aggregate_audit": file_record(
                output_root / "artifact_audit_pre_aggregate.json"
            ),
            "confirmation_report": file_record(
                output_root / "confirmation_report.json"
            ),
            "final_audit": file_record(output_root / "artifact_audit_final.json"),
            "written_at_unix_seconds": time.time(),
        }
        atomic_json(status_path, result)
        return {"pre_aggregate": pre, "aggregate": aggregate, "final": final}
    except Exception as error:
        request_abort(config=config, output_root=output_root, reason=str(error))
        atomic_json(
            status_path,
            {
                "schema_version": 1,
                "complete": True,
                "failed": True,
                "state": "failed",
                "contract_sha256": contract["contract_sha256"],
                "error_type": type(error).__name__,
                "error": str(error),
                "written_at_unix_seconds": time.time(),
            },
        )
        raise


def status_report(output_root: Path) -> dict[str, Any]:
    def read_optional(path: Path) -> Any:
        return json.loads(path.read_text()) if path.is_file() else None

    contract = read_optional(output_root / "run_contract.json")
    contract_digest_valid = None
    if contract is not None:
        digest = contract.get("contract_sha256")
        unsigned = {key: value for key, value in contract.items() if key != "contract_sha256"}
        contract_digest_valid = digest == canonical_digest(unsigned)
    workers = {
        str(index): read_optional(_status_path(output_root, index)) for index in range(4)
    }
    states = Counter(
        row.get("state", "unknown") for row in workers.values() if row is not None
    )
    return {
        "schema_version": 1,
        "output_root": str(output_root.resolve()),
        "contract_present": contract is not None,
        "contract_digest_valid": contract_digest_valid,
        "contract_sha256": contract.get("contract_sha256") if contract else None,
        "worker_states": dict(sorted(states.items())),
        "workers": workers,
        "abort": read_optional(output_root / "abort_requested.json"),
        "freeze_failure": read_optional(output_root / "freeze_failed.json"),
        "finalizer": read_optional(output_root / "finalizer_status.json"),
        "aggregate_status": read_optional(output_root / "status.json"),
        "pre_aggregate_audit": read_optional(
            output_root / "artifact_audit_pre_aggregate.json"
        ),
        "final_audit": read_optional(output_root / "artifact_audit_final.json"),
    }


def retain_entrypoint_failure(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    command: str,
    error: Exception,
    worker_index: int | None = None,
) -> None:
    try:
        request_abort(config=config, output_root=output_root, reason=str(error))
    except Exception:
        pass
    raw_contract = (
        json.loads((output_root / "run_contract.json").read_text())
        if (output_root / "run_contract.json").is_file()
        else {}
    )
    contract_sha256 = str(raw_contract.get("contract_sha256", "unavailable"))
    failure = {
        "error_type": type(error).__name__,
        "error": str(error),
        "entrypoint_command": command,
    }
    if command == "run-worker" and worker_index is not None:
        path = _status_path(output_root, worker_index)
        existing = json.loads(path.read_text()) if path.is_file() else {}
        if existing.get("state") != "failed":
            _write_worker_status(
                output_root,
                worker_index=worker_index,
                state="failed",
                contract_sha256=contract_sha256,
                extra=failure,
            )
    elif command == "finalize":
        path = output_root / "finalizer_status.json"
        existing = json.loads(path.read_text()) if path.is_file() else {}
        if existing.get("state") != "failed":
            atomic_json(
                path,
                {
                    "schema_version": 1,
                    "complete": True,
                    "failed": True,
                    "state": "failed",
                    "contract_sha256": contract_sha256,
                    "written_at_unix_seconds": time.time(),
                    **failure,
                },
            )


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    subparsers = parser.add_subparsers(dest="command", required=True)

    freeze = subparsers.add_parser("freeze")
    freeze.add_argument("--base-config", type=Path, required=True)
    freeze.add_argument("--exp12-config", type=Path, required=True)
    freeze.add_argument("--exp12-root", type=Path, required=True)
    freeze.add_argument("--pilot-root", type=Path, required=True)
    freeze.add_argument("--pilot-audit", type=Path, required=True)
    freeze.add_argument("--source-cache-ready", type=Path, required=True)
    freeze.add_argument("--model-cache", type=Path, required=True)
    freeze.add_argument("--saebench-root", type=Path, required=True)

    worker = subparsers.add_parser("run-worker")
    worker.add_argument("--worker-index", type=int, required=True)
    worker.add_argument("--device", default="cuda")

    audit = subparsers.add_parser("audit")
    audit.add_argument("--phase", choices=("pre-aggregate", "final"), required=True)
    audit.add_argument("--wait-seconds", type=float, default=0)

    subparsers.add_parser("aggregate")
    finalize = subparsers.add_parser("finalize")
    finalize.add_argument("--wait-seconds", type=float, required=True)
    failure = subparsers.add_parser("retain-failure")
    failure.add_argument("--stage", choices=("worker", "finalizer"), required=True)
    failure.add_argument("--worker-index", type=int)
    failure.add_argument("--message", required=True)
    subparsers.add_parser("status")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    output_root = args.output_root.expanduser().resolve()
    config_path = args.config.expanduser().resolve()
    if args.command == "status":
        print(json.dumps(status_report(output_root), indent=2, sort_keys=True))
        return 0
    config = load_config(config_path)

    def terminate_for_timeout(signum: int, _frame: Any) -> None:
        raise RunAbortedError(f"Exp13 received termination signal {signum}")

    if args.command in {"run-worker", "finalize"}:
        signal.signal(signal.SIGTERM, terminate_for_timeout)
    if args.command == "freeze":
        try:
            result = freeze_run(
                config_path=config_path,
                base_config_path=args.base_config.expanduser().resolve(),
                exp12_config_path=args.exp12_config.expanduser().resolve(),
                exp12_root=args.exp12_root.expanduser().resolve(),
                pilot_root=args.pilot_root.expanduser().resolve(),
                pilot_audit_path=args.pilot_audit.expanduser().resolve(),
                source_cache_ready_path=args.source_cache_ready.expanduser().resolve(),
                model_cache=args.model_cache.expanduser().resolve(),
                saebench_root=args.saebench_root.expanduser().resolve(),
                output_root=output_root,
            )
        except Exception as error:
            atomic_json(
                output_root / "freeze_failed.json",
                {
                    "schema_version": 1,
                    "complete": True,
                    "failed": True,
                    "state": "failed",
                    "config_digest": canonical_digest(config),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "written_at_unix_seconds": time.time(),
                },
            )
            raise
    elif args.command == "run-worker":
        try:
            result = run_worker(
                config=config,
                output_root=output_root,
                worker_index=args.worker_index,
                device=args.device,
            )
        except Exception as error:
            retain_entrypoint_failure(
                config=config,
                output_root=output_root,
                command=args.command,
                error=error,
                worker_index=args.worker_index,
            )
            raise
    elif args.command == "audit":
        result = audit_artifacts(
            config=config,
            output_root=output_root,
            phase=args.phase,
            wait_seconds=args.wait_seconds,
        )
    elif args.command == "aggregate":
        result = aggregate_confirmation(config=config, output_root=output_root)
    elif args.command == "finalize":
        try:
            result = finalize_run(
                config=config,
                output_root=output_root,
                wait_seconds=args.wait_seconds,
            )
        except Exception as error:
            retain_entrypoint_failure(
                config=config,
                output_root=output_root,
                command=args.command,
                error=error,
            )
            raise
    elif args.command == "retain-failure":
        if args.stage == "worker" and args.worker_index not in range(4):
            raise ValueError("worker failure retention requires an index in [0, 3]")
        if args.stage == "finalizer" and args.worker_index is not None:
            raise ValueError("finalizer failure retention does not take a worker index")
        retain_entrypoint_failure(
            config=config,
            output_root=output_root,
            command="run-worker" if args.stage == "worker" else "finalize",
            error=RuntimeError(args.message),
            worker_index=args.worker_index,
        )
        result = status_report(output_root)
    else:  # pragma: no cover - argparse enforces the command set.
        raise AssertionError(args.command)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
