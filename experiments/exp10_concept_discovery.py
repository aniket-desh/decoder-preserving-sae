#!/usr/bin/env python3
"""Frozen, resumable concept-discovery pilot for the Pythia SAE pair.

Heavy dependencies are imported only inside execution stages. This keeps the
repository testable in its normal environment while requiring an exact clean
checkout of the pinned SAEBench commit for benchmark work.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from dpsae.saebench_adapter import (
    NativeBatchTopKSAEBenchAdapter,
    load_native_saebench_adapter,
    one_based_resid_post_hook,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/exp10_concept_discovery.json"
DEFAULT_OUTPUT = ROOT / "artifacts/exp10_concept_discovery"


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = read_json(path)
    if config.get("schema_version") != 1 or config.get("experiment_id") != "exp10_concept_discovery":
        raise ValueError("not an exp10 concept-discovery config")
    benchmark = config["benchmark"]
    datasets = benchmark["datasets"]
    if len(datasets) != len(set(datasets)) or not datasets:
        raise ValueError("benchmark datasets must be nonempty and unique")
    if canonical_digest(datasets) != benchmark["dataset_manifest_sha256"]:
        raise ValueError("frozen dataset manifest hash disagrees with the dataset list")
    families = benchmark["family_by_dataset"]
    if set(families) != set(datasets) or any(not families[name] for name in datasets):
        raise ValueError("family map must cover every frozen dataset exactly once")
    seeds = benchmark["probe_seeds"]
    if len(seeds) != 10 or len(seeds) != len(set(seeds)):
        raise ValueError("exp10 requires exactly ten unique probe seeds")
    if benchmark["ks"] != [1, 2, 5] or benchmark["setting"] != "normal":
        raise ValueError("exp10 sparse-probe setting or k values changed")
    if benchmark["regularization"] != "l1":
        raise ValueError("the sparse-selection benchmark must remain L1")
    if benchmark["companion_regularization"] != "l2":
        raise ValueError("companion probes must match sae-probes' unfiltered L2 logreg baseline")
    if "unfiltered logreg baseline" not in benchmark["companion_regularization_rationale"]:
        raise ValueError("companion regularization rationale is missing")
    model = config["model"]
    layer, hook = one_based_resid_post_hook(model["one_based_block"])
    if layer != model["transformer_lens_hook_layer"] or hook != model["hook_name"]:
        raise ValueError("one-based block and TransformerLens hook disagree")
    if config["adapter"]["decoder_renormalization"] != "forbidden":
        raise ValueError("exp10 forbids adapter-side decoder renormalization")
    return config


def resolve_checkpoint_dir(config: Mapping[str, Any], override: Path | None) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    return (ROOT / config["pilot_checkpoint"]["artifact_directory"]).resolve()


def artifact_paths(config: Mapping[str, Any], checkpoint_dir: Path) -> dict[str, Path]:
    pilot = config["pilot_checkpoint"]
    return {
        "models": checkpoint_dir / pilot["models_file"],
        "calibration": checkpoint_dir / pilot["calibration_file"],
        "evaluation": checkpoint_dir / pilot["evaluation_file"],
    }


def artifact_hashes(config: Mapping[str, Any], checkpoint_dir: Path) -> dict[str, str]:
    paths = artifact_paths(config, checkpoint_dir)
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"checkpoint directory is missing artifacts: {missing}")
    return {f"{name}_sha256": file_sha256(path) for name, path in paths.items()}


def _finite(value: Any) -> bool:
    if isinstance(value, bool) or value is None or isinstance(value, str):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(float(value))
    if isinstance(value, Mapping):
        return all(_finite(item) for item in value.values())
    if isinstance(value, Sequence):
        return all(_finite(item) for item in value)
    return True


def assess_eligibility(
    config: Mapping[str, Any], checkpoint_dir: Path, *, device: torch.device | None = None
) -> dict[str, Any]:
    """Apply the stricter concept gate without opening any benchmark result."""

    paths = artifact_paths(config, checkpoint_dir)
    hashes = artifact_hashes(config, checkpoint_dir)
    evaluation = read_json(paths["evaluation"])
    pilot = config["pilot_checkpoint"]
    gate = config["eligibility"]
    names = pilot["model_payload_names"]
    models = evaluation.get("models", {})
    checks: dict[str, bool] = {}
    checks["evaluation_complete"] = evaluation.get("complete") is True
    checks["expected_evaluation_models"] = all(name in models for name in names.values())
    if not checks["expected_evaluation_models"]:
        return {
            "schema_version": 1,
            "passed": False,
            "checkpoint_directory": str(checkpoint_dir),
            "artifact_hashes": hashes,
            "checks": checks,
        }

    mse = models[names["mse"]]
    dpsae = models[names["dpsae"]]
    ratio = float(dpsae["nmse"]) / max(float(mse["nmse"]), 1e-12)
    target_l0 = float(pilot["target_l0"])
    relative_l0 = {
        method: abs(float(models[name]["inference_l0"]) - target_l0) / target_l0
        for method, name in names.items()
    }
    pair_relative_l0 = abs(
        float(dpsae["inference_l0"]) - float(mse["inference_l0"])
    ) / target_l0
    checks["nmse_ratio"] = ratio <= float(gate["maximum_dpsae_to_mse_nmse_ratio"])
    checks["l0_target"] = all(
        value <= float(gate["maximum_relative_l0_error"])
        for value in relative_l0.values()
    )
    checks["l0_pair_match"] = pair_relative_l0 <= float(
        gate["maximum_pair_relative_l0_difference"]
    )
    checks["finite"] = _finite({"mse": mse, "dpsae": dpsae, "nmse_ratio": ratio})
    if gate["require_artifact_hash_match"]:
        checks["models_hash"] = evaluation.get("models_sha256") == hashes["models_sha256"]
        checks["calibration_hash"] = (
            evaluation.get("calibration_sha256") == hashes["calibration_sha256"]
        )

    adapter_device = device or torch.device("cpu")
    adapter_checks: dict[str, Any] = {}
    for method, payload_name in names.items():
        adapter = load_adapter(config, checkpoint_dir, method, adapter_device)
        adapter_checks[method] = {
            "decoder_norm_max_deviation": float((adapter.W_dec.norm(dim=1) - 1).abs().max()),
            "threshold": float(adapter.activation_threshold),
            "threshold_updates": int(adapter.threshold_updates),
            "hook_name": adapter.cfg.hook_name,
        }
    checks["native_adapter"] = True
    passed = all(checks.values())
    return {
        "schema_version": 1,
        "passed": passed,
        "checkpoint_directory": str(checkpoint_dir),
        "artifact_hashes": hashes,
        "metrics": {
            "nmse_ratio": ratio,
            "relative_l0_error": relative_l0,
            "pair_relative_l0_difference": pair_relative_l0,
        },
        "limits": dict(gate),
        "adapter_checks": adapter_checks,
        "checks": checks,
    }


def load_adapter(
    config: Mapping[str, Any],
    checkpoint_dir: Path,
    method: str,
    device: torch.device,
) -> NativeBatchTopKSAEBenchAdapter:
    pilot = config["pilot_checkpoint"]
    model = config["model"]
    adapter = config["adapter"]
    if method not in pilot["model_payload_names"]:
        raise ValueError(f"unknown exp10 method: {method}")
    paths = artifact_paths(config, checkpoint_dir)
    return load_native_saebench_adapter(
        models_path=paths["models"],
        calibration_path=paths["calibration"],
        payload_name=pilot["model_payload_names"][method],
        model_name=model["transformer_lens_name"],
        one_based_block=int(model["one_based_block"]),
        context_size=int(model["context_size"]),
        device=device,
        decoder_norm_atol=float(adapter["decoder_norm_atol_float32"]),
        expected_method=method,
        expected_d_in=int(model["d_model"]),
        expected_d_sae=int(pilot["dictionary_size"]),
        expected_k=int(pilot["target_l0"]),
    )


EXPECTED_SIGNATURES = {
    "sae_bench.evals.sparse_probing_sae_probes.main.run_eval": (
        "config",
        "selected_saes",
        "device",
        "output_path",
        "force_rerun",
    ),
    "sae_probes.run_sae_evals.run_sae_evals": (
        "sae",
        "model_name",
        "hook_name",
        "reg_type",
        "setting",
        "ks",
        "binarize",
        "results_path",
        "model_cache_path",
        "datasets",
        "device",
        "mean_diff_normalization",
        "seed",
    ),
    "sae_probes.generate_sae_activations.generate_sae_activations": (
        "sae",
        "setting",
        "dataset",
        "hook_name",
        "model_name",
        "device",
        "num_train",
        "frac",
        "model_cache_path",
        "batch_size",
        "seed",
    ),
    "sae_probes.utils_training.find_best_reg": (
        "X_train",
        "y_train",
        "X_test",
        "y_test",
        "n_jobs",
        "parallel",
        "penalty",
        "seed",
    ),
    "sae_probes.utils_data.get_xy_traintest": (
        "num_train",
        "numbered_dataset_tag",
        "hook_name",
        "model_name",
        "model_cache_path",
        "MAX_AMT",
        "seed",
    ),
}


def verify_sae_probes_packaged_data(config: Mapping[str, Any]) -> dict[str, Any]:
    """Validate package data without importing sae-probes.

    Version 0.4 eagerly scans every packaged CSV during import and can take
    roughly a minute. The scan is intentional; interrupting it can surface a
    misleading pandas parser traceback even when every zstd file is valid.
    """

    import importlib.util

    spec = importlib.util.find_spec("sae_probes")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("sae-probes package cannot be located")
    package_root = Path(next(iter(spec.submodule_search_locations))).resolve()
    master = package_root / "data/probing_datasets_MASTER.csv.zst"
    if not master.is_file():
        raise RuntimeError(f"sae-probes master dataset is missing: {master}")
    observed_hash = file_sha256(master)
    expected_hash = config["dependencies"]["sae_probes_master_data_sha256"]
    if observed_hash != expected_hash:
        raise RuntimeError(
            "sae-probes packaged master dataset differs from the exact v0.4.0 source "
            f"(expected {expected_hash}, observed {observed_hash})"
        )
    cleaned = sorted((package_root / "data/cleaned_data").glob("*.csv.zst"))
    expected_count = int(config["dependencies"]["sae_probes_expected_cleaned_data_files"])
    if len(cleaned) != expected_count:
        raise RuntimeError(
            f"sae-probes packaged dataset count drift: expected {expected_count}, "
            f"observed {len(cleaned)}"
        )
    return {
        "package_root": str(package_root),
        "master_data_path": str(master),
        "master_data_sha256": observed_hash,
        "cleaned_data_file_count": len(cleaned),
        "policy": config["dependencies"]["sae_probes_data_policy"],
    }


def _resolve_object(qualified_name: str):
    module_name, object_name = qualified_name.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), object_name)


def verify_saebench_environment(
    config: Mapping[str, Any], saebench_root: Path
) -> dict[str, Any]:
    """Require the exact clean source checkout and public APIs used by exp10."""

    root = saebench_root.expanduser().resolve()
    if not (root / ".git").exists():
        raise RuntimeError("--saebench-root must be a source checkout with .git")
    revision = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    expected = config["dependencies"]["saebench_commit"]
    if revision != expected:
        raise RuntimeError(f"SAEBench revision drift: expected {expected}, observed {revision}")
    dirty = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    ).strip()
    if config["dependencies"]["require_clean_saebench_checkout"] and dirty:
        raise RuntimeError("pinned SAEBench checkout is dirty")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    packaged_data = verify_sae_probes_packaged_data(config)
    import_started = time.monotonic()
    sae_bench = importlib.import_module("sae_bench")
    eager_import_seconds = time.monotonic() - import_started
    module_path = Path(sae_bench.__file__).resolve()
    if not module_path.is_relative_to(root):
        raise RuntimeError(f"imported sae_bench from {module_path}, outside pinned checkout")
    versions = {
        "sae-bench": importlib.metadata.version("sae-bench"),
        "sae-probes": importlib.metadata.version("sae-probes"),
    }
    expected_versions = {
        "sae-bench": config["dependencies"]["saebench_version"],
        "sae-probes": config["dependencies"]["sae_probes_version"],
    }
    if versions != expected_versions:
        raise RuntimeError(
            f"benchmark package version drift: expected {expected_versions}, observed {versions}"
        )

    from sae_probes import DATASETS

    if list(DATASETS) != config["benchmark"]["datasets"]:
        raise RuntimeError("installed sae-probes DATASETS differs from frozen manifest")
    signatures = {}
    if config["dependencies"]["require_exact_function_signatures"]:
        for name, expected_parameters in EXPECTED_SIGNATURES.items():
            observed = tuple(inspect.signature(_resolve_object(name)).parameters)
            signatures[name] = list(observed)
            if observed != expected_parameters:
                raise RuntimeError(
                    f"pinned API signature drift for {name}: "
                    f"expected {expected_parameters}, observed {observed}"
                )
    return {
        "saebench_root": str(root),
        "saebench_revision": revision,
        "saebench_dirty": bool(dirty),
        "versions": versions,
        "signatures": signatures,
        "dataset_manifest_sha256": canonical_digest(list(DATASETS)),
        "sae_probes_packaged_data": packaged_data,
        "sae_probes_eager_import_seconds": eager_import_seconds,
        "sae_probes_eager_import_expected_seconds": config["dependencies"][
            "sae_probes_eager_import_expected_seconds"
        ],
    }


def source_hashes(config_path: Path) -> dict[str, str]:
    paths = [
        config_path if config_path.is_absolute() else ROOT / config_path,
        Path(__file__).resolve(),
        ROOT / "src/dpsae/saebench_adapter.py",
    ]
    resolved = [path.resolve() for path in paths]
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in resolved}


def freeze_run(
    *,
    config_path: Path,
    output_root: Path,
    checkpoint_dir: Path,
    saebench_root: Path,
) -> dict[str, Any]:
    config = load_config(config_path)
    eligibility = assess_eligibility(config, checkpoint_dir)
    atomic_json(output_root / "eligibility.json", eligibility)
    if not eligibility["passed"]:
        raise RuntimeError("Pythia pilot failed the frozen concept eligibility gate")
    environment = verify_saebench_environment(config, saebench_root)
    resolved = {
        "schema_version": 1,
        "experiment_id": config["experiment_id"],
        "config_path": str(config_path.resolve()),
        "config_sha256": file_sha256(config_path),
        "config_digest": canonical_digest(config),
        "checkpoint_directory": str(checkpoint_dir),
        "artifact_hashes": eligibility["artifact_hashes"],
        "environment": environment,
        "source_hashes": source_hashes(config_path),
    }
    path = output_root / "resolved_config.json"
    if path.exists() and read_json(path) != resolved:
        raise RuntimeError("resolved exp10 run changed; use a fresh output root")
    atomic_json(path, resolved)
    return resolved


def load_resolved(output_root: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    path = output_root / "resolved_config.json"
    if not path.exists():
        raise RuntimeError("run freeze before benchmark execution")
    value = read_json(path)
    if value.get("config_digest") != canonical_digest(config):
        raise RuntimeError("resolved config does not match the current frozen config")
    return value


def cache_files(config: Mapping[str, Any], model_cache: Path) -> dict[str, Path]:
    model = config["model"]
    base = model_cache / f"model_activations_{model['transformer_lens_name']}"
    return {
        dataset: base / f"{dataset}_{model['hook_name']}.pt"
        for dataset in config["benchmark"]["datasets"]
    }


def _cache_manifest(
    config: Mapping[str, Any],
    model_cache: Path,
    config_digest: str,
    *,
    include_hashes: bool = True,
) -> dict[str, Any]:
    files = cache_files(config, model_cache)
    missing = [str(path) for path in files.values() if not path.is_file()]
    if missing:
        raise RuntimeError(f"activation cache is incomplete ({len(missing)} missing files)")
    records = {}
    for dataset, path in files.items():
        value = torch.load(path, map_location="cpu", weights_only=True)
        if value.ndim != 2 or value.shape[1] != int(config["model"]["d_model"]):
            raise RuntimeError(f"bad cached activation shape for {dataset}: {tuple(value.shape)}")
        record = {
            "path": str(path.resolve()),
            "shape": list(value.shape),
            "bytes": path.stat().st_size,
        }
        if include_hashes:
            record["sha256"] = file_sha256(path)
        records[dataset] = record
    return {
        "schema_version": 1,
        "complete": True,
        "config_digest": config_digest,
        "model_cache": str(model_cache.resolve()),
        "dataset_manifest_sha256": config["benchmark"]["dataset_manifest_sha256"],
        "files": records,
    }


def verify_cache_ready(
    config: Mapping[str, Any], output_root: Path, model_cache: Path
) -> dict[str, Any]:
    resolved = load_resolved(output_root, config)
    path = output_root / "cache_ready.json"
    if not path.exists():
        raise RuntimeError("single-writer activation cache is not ready")
    observed = read_json(path)
    if observed.get("config_digest") != resolved["config_digest"]:
        raise RuntimeError("activation cache belongs to another resolved config")
    if set(observed.get("files", {})) != set(config["benchmark"]["datasets"]):
        raise RuntimeError("activation cache manifest dataset set changed")
    for dataset, cache_path in cache_files(config, model_cache).items():
        record = observed["files"][dataset]
        if str(cache_path.resolve()) != record.get("path") or not cache_path.is_file():
            raise RuntimeError(f"activation cache path changed for {dataset}")
        if cache_path.stat().st_size != record.get("bytes"):
            raise RuntimeError(f"activation cache size changed for {dataset}")
        if "sha256" not in record:
            raise RuntimeError(f"activation cache lacks a frozen hash for {dataset}")
    return observed


def prepare_cache(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    model_cache: Path,
    device: str,
) -> dict[str, Any]:
    resolved = load_resolved(output_root, config)
    ready = output_root / "cache_ready.json"
    if ready.exists():
        return verify_cache_ready(config, output_root, model_cache)
    lock = output_root / "cache_writer.lock"
    output_root.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    except FileExistsError as error:
        raise RuntimeError("another cache writer exists; inspect cache_writer.lock") from error
    try:
        os.write(descriptor, f"pid={os.getpid()}\n".encode())
        os.close(descriptor)
        from sae_probes.generate_model_activations import ensure_dataset_activations

        ensure_dataset_activations(
            model_name=config["model"]["transformer_lens_name"],
            dataset_short_names=config["benchmark"]["datasets"],
            hook_names=[config["model"]["hook_name"]],
            model_cache_path=model_cache,
            device=device,
        )
        manifest = _cache_manifest(config, model_cache, resolved["config_digest"])
        atomic_json(ready, manifest)
        return manifest
    finally:
        lock.unlink(missing_ok=True)


def wait_cache(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    model_cache: Path,
    timeout_seconds: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while not (output_root / "cache_ready.json").exists():
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for exp10 cache_ready.json")
        time.sleep(min(10, max(0.1, deadline - time.monotonic())))
    return verify_cache_ready(config, output_root, model_cache)


def _job_dir(
    config: Mapping[str, Any], output_root: Path, method: str, probe_seed: int
) -> Path:
    checkpoint_id = config["pilot_checkpoint"]["checkpoint_id"]
    return output_root / "jobs" / checkpoint_id / method / f"seed_{probe_seed}"


def _raw_sparse_files(job_dir: Path, model_name: str, hook_name: str) -> list[Path]:
    return sorted(
        job_dir.glob(
            f"raw/*_custom_sae/sae_probes_{model_name}/normal_setting/"
            f"*_{hook_name}_l1.json"
        )
    )


def _parse_dataset(path: Path) -> str:
    marker = "_blocks."
    if marker not in path.stem:
        raise ValueError(f"cannot parse sae-probes dataset from {path.name}")
    return path.stem.split(marker, 1)[0]


def _metrics_dict(result: Any) -> dict[str, float]:
    return {key: float(value) for key, value in asdict(result.metrics).items()}


def _classifier_state(result: Any) -> dict[str, Any]:
    classifier = result.classifier
    return {
        "coefficient": torch.as_tensor(classifier.coef_).float(),
        "intercept": torch.as_tensor(classifier.intercept_).float(),
        "classes": torch.as_tensor(classifier.classes_),
        "C": float(classifier.C),
    }


def _heldout_identity(
    config: Mapping[str, Any], dataset: str, probe_seed: int, example_count: int
) -> dict[str, Any]:
    """Assign stable IDs to sae-probes' deterministic positional test split."""

    split_payload = {
        "dataset": dataset,
        "probe_seed": probe_seed,
        "split": "test",
        "splitter": "sae_probes.utils_data.get_xy_traintest",
        "model_name": config["model"]["transformer_lens_name"],
        "hook_name": config["model"]["hook_name"],
    }
    split_id = f"exp10-test-{canonical_digest(split_payload)[:20]}"
    example_ids = [
        f"{split_id}-{index:05d}" for index in range(example_count)
    ]
    return {
        "split": "test",
        "split_id": split_id,
        "example_id_policy": "sha256_of_frozen_split_identity_plus_positional_index",
        "example_ids": example_ids,
    }


def _heldout_classifier_outputs(result: Any, X_test: Any) -> dict[str, Tensor]:
    classifier = result.classifier
    decision = np.asarray(classifier.decision_function(X_test))
    prediction = np.asarray(classifier.predict(X_test))
    return {
        "decision_score": torch.as_tensor(decision).detach().float().cpu(),
        "prediction": torch.as_tensor(prediction).detach().cpu(),
    }


def capture_sparse_provenance(
    *,
    config: Mapping[str, Any],
    adapter: NativeBatchTopKSAEBenchAdapter,
    method: str,
    probe_seed: int,
    job_dir: Path,
    model_cache: Path,
    device: str,
    config_digest: str,
) -> dict[str, str]:
    from sae_probes.generate_sae_activations import generate_sae_activations
    from sae_probes.run_sae_evals import get_sorted_indices, mean_act_normalization
    from sae_probes.utils_training import find_best_reg

    model = config["model"]
    benchmark = config["benchmark"]
    raw_by_dataset = {_parse_dataset(path): path for path in _raw_sparse_files(
        job_dir, model["transformer_lens_name"], model["hook_name"]
    )}
    expected = set(benchmark["datasets"])
    if set(raw_by_dataset) != expected:
        missing = sorted(expected.difference(raw_by_dataset))
        extra = sorted(set(raw_by_dataset).difference(expected))
        raise RuntimeError(f"sparse result missingness: missing={missing}, extra={extra}")

    hashes: dict[str, str] = {}
    for dataset in benchmark["datasets"]:
        output = job_dir / "provenance" / f"{dataset}.json"
        predictions_path = job_dir / "predictions" / f"{dataset}.pt"
        if output.exists():
            observed = read_json(output)
            if observed.get("config_digest") != config_digest:
                raise RuntimeError(f"stale sparse provenance for {dataset}")
            if observed.get("raw_result_sha256") != file_sha256(raw_by_dataset[dataset]):
                raise RuntimeError(f"raw sparse result changed for {dataset}")
            if (
                not predictions_path.is_file()
                or observed.get("heldout_predictions_sha256")
                != file_sha256(predictions_path)
            ):
                raise RuntimeError(f"held-out sparse predictions changed for {dataset}")
            hashes[dataset] = file_sha256(output)
            continue
        raw_path = raw_by_dataset[dataset]
        entries = read_json(raw_path)
        if not isinstance(entries, list):
            raise RuntimeError(f"unexpected sae-probes result schema for {dataset}")
        by_k = {int(entry["k"]): entry for entry in entries if "k" in entry}
        if set(by_k) != set(benchmark["ks"]):
            raise RuntimeError(f"unexpected k rows for {dataset}: {sorted(by_k)}")
        activations = generate_sae_activations(
            sae=adapter,
            setting="normal",
            dataset=dataset,
            hook_name=model["hook_name"],
            model_name=model["transformer_lens_name"],
            device=device,
            num_train=None,
            frac=None,
            model_cache_path=model_cache,
            batch_size=128,
            seed=probe_seed,
        )
        ranking = get_sorted_indices(
            activations.X_train,
            activations.y_train,
            normalize_fn=mean_act_normalization,
        )
        rows = []
        test_labels = torch.as_tensor(activations.y_test).detach().cpu()
        heldout = _heldout_identity(config, dataset, probe_seed, len(test_labels))
        prediction_rows: dict[str, Any] = {}
        for k in benchmark["ks"]:
            raw = by_k[k]
            feature_ids = [int(value) for value in ranking[:k].tolist()]
            if raw.get("indices") != feature_ids:
                raise RuntimeError(f"selected-feature drift for {dataset} k={k}")
            fit = find_best_reg(
                X_train=activations.X_train[:, feature_ids],
                y_train=activations.y_train,
                X_test=activations.X_test[:, feature_ids],
                y_test=activations.y_test,
                penalty="l1",
                seed=probe_seed,
            )
            metrics = _metrics_dict(fit)
            for metric in ("test_auc", "test_acc", "test_f1", "val_auc"):
                if not math.isclose(metrics[metric], float(raw[metric]), abs_tol=1e-8, rel_tol=0):
                    raise RuntimeError(
                        f"provenance refit drift for {dataset} k={k} {metric}: "
                        f"{metrics[metric]} != {raw[metric]}"
                    )
            coefficients = fit.classifier.coef_[0]
            prediction_rows[str(k)] = _heldout_classifier_outputs(
                fit, activations.X_test[:, feature_ids]
            )
            rows.append(
                {
                    "k": k,
                    "metrics": metrics,
                    "feature_ids": feature_ids,
                    "feature_weights": [
                        {"feature_id": feature_id, "weight": float(weight)}
                        for feature_id, weight in zip(feature_ids, coefficients)
                    ],
                    "intercept": float(fit.classifier.intercept_[0]),
                    "regularization_C": float(fit.classifier.C),
                }
            )
        prediction_artifact = {
            "schema_version": 1,
            "config_digest": config_digest,
            "checkpoint_id": config["pilot_checkpoint"]["checkpoint_id"],
            "method": method,
            "probe_seed": probe_seed,
            "dataset": dataset,
            **heldout,
            "label": test_labels,
            "by_k": prediction_rows,
            "decision_score_semantics": "sklearn_binary_logistic_score_for_classifier_classes_index_1",
        }
        atomic_torch(predictions_path, prediction_artifact)
        record = {
            "schema_version": 2,
            "config_digest": config_digest,
            "checkpoint_id": config["pilot_checkpoint"]["checkpoint_id"],
            "method": method,
            "probe_seed": probe_seed,
            "dataset": dataset,
            "family": benchmark["family_by_dataset"][dataset],
            "raw_result_sha256": file_sha256(raw_path),
            "selection": "sae_probes_mean_activation_normalized_absolute_class_difference",
            "probe": "sae_probes_find_best_reg_l1",
            "heldout_split_id": heldout["split_id"],
            "heldout_example_count": len(test_labels),
            "heldout_example_id_policy": heldout["example_id_policy"],
            "heldout_predictions_sha256": file_sha256(predictions_path),
            "rows": rows,
        }
        atomic_json(output, record)
        hashes[dataset] = file_sha256(output)
    return hashes


def run_sparse_job(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    checkpoint_dir: Path,
    model_cache: Path,
    method: str,
    probe_seed: int,
    device: str,
) -> dict[str, Any]:
    resolved = load_resolved(output_root, config)
    verify_cache_ready(config, output_root, model_cache)
    if method not in {"mse", "dpsae"}:
        raise ValueError("method must be mse or dpsae")
    if probe_seed not in config["benchmark"]["probe_seeds"]:
        raise ValueError("probe seed is not in the frozen seed list")
    job_dir = _job_dir(config, output_root, method, probe_seed)
    done = job_dir / "done.json"
    if done.exists():
        value = read_json(done)
        if value.get("config_digest") != resolved["config_digest"]:
            raise RuntimeError("completed sparse job belongs to another config")
        for dataset, digest in value.get("provenance_hashes", {}).items():
            path = job_dir / "provenance" / f"{dataset}.json"
            if not path.is_file() or file_sha256(path) != digest:
                raise RuntimeError(f"completed sparse provenance changed for {dataset}")
            record = read_json(path)
            predictions = job_dir / "predictions" / f"{dataset}.pt"
            if (
                not predictions.is_file()
                or record.get("heldout_predictions_sha256") != file_sha256(predictions)
            ):
                raise RuntimeError(f"completed sparse predictions changed for {dataset}")
        return value
    adapter = load_adapter(config, checkpoint_dir, method, torch.device(device))

    from sae_bench.evals.sparse_probing_sae_probes.eval_config import (
        SparseProbingSaeProbesEvalConfig,
    )
    from sae_bench.evals.sparse_probing_sae_probes.main import run_eval

    payload_hash = resolved["artifact_hashes"]["models_sha256"][:12]
    release = f"exp10_{config['pilot_checkpoint']['checkpoint_id']}_{method}_{payload_hash}"
    bench_config = SparseProbingSaeProbesEvalConfig(
        model_name=config["model"]["transformer_lens_name"],
        random_seed=probe_seed,
        dataset_names=config["benchmark"]["datasets"],
        reg_type="l1",
        setting="normal",
        ks=config["benchmark"]["ks"],
        binarize=False,
        results_path=str(job_dir / "raw"),
        model_cache_path=str(model_cache),
        include_llm_baseline=True,
        baseline_method="logreg",
    )
    run_eval(
        bench_config,
        [(release, adapter)],
        device,
        str(job_dir / "saebench_output"),
        force_rerun=False,
    )
    provenance_hashes = capture_sparse_provenance(
        config=config,
        adapter=adapter,
        method=method,
        probe_seed=probe_seed,
        job_dir=job_dir,
        model_cache=model_cache,
        device=device,
        config_digest=resolved["config_digest"],
    )
    final_results = sorted((job_dir / "saebench_output").glob("*_eval_results.json"))
    if len(final_results) != 1:
        raise RuntimeError("expected exactly one SAEBench aggregate result")
    result = {
        "schema_version": 1,
        "complete": True,
        "config_digest": resolved["config_digest"],
        "artifact_hashes": resolved["artifact_hashes"],
        "method": method,
        "probe_seed": probe_seed,
        "dataset_count": len(provenance_hashes),
        "dataset_manifest_sha256": config["benchmark"]["dataset_manifest_sha256"],
        "saebench_result_sha256": file_sha256(final_results[0]),
        "provenance_hashes": provenance_hashes,
    }
    atomic_json(done, result)
    return result


@torch.inference_mode()
def _representations(
    adapter: NativeBatchTopKSAEBenchAdapter,
    activation: Tensor,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> tuple[Tensor, Tensor]:
    codes, reconstructions = [], []
    for batch in activation.split(batch_size):
        code = adapter.encode(batch.to(device)).cpu()
        reconstruction = adapter.decode(code.to(device)).cpu()
        codes.append(code)
        reconstructions.append(reconstruction)
    return torch.cat(codes), torch.cat(reconstructions)


def run_companion_job(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    checkpoint_dir: Path,
    model_cache: Path,
    probe_seed: int,
    device: str,
) -> dict[str, Any]:
    resolved = load_resolved(output_root, config)
    verify_cache_ready(config, output_root, model_cache)
    if probe_seed not in config["benchmark"]["probe_seeds"]:
        raise ValueError("probe seed is not in the frozen seed list")
    checkpoint_id = config["pilot_checkpoint"]["checkpoint_id"]
    job_dir = output_root / "companion" / checkpoint_id / f"seed_{probe_seed}"
    done = job_dir / "done.json"
    if done.exists():
        value = read_json(done)
        if value.get("config_digest") != resolved["config_digest"]:
            raise RuntimeError("completed companion job belongs to another config")
        return value

    from sae_probes.run_sae_evals import DATASET_SIZES
    from sae_probes.utils_data import get_xy_traintest
    from sae_probes.utils_training import find_best_reg

    torch_device = torch.device(device)
    adapters = {
        method: load_adapter(config, checkpoint_dir, method, torch_device)
        for method in ("mse", "dpsae")
    }
    dataset_hashes = {}
    for dataset in config["benchmark"]["datasets"]:
        metrics_path = job_dir / "metrics" / f"{dataset}.json"
        weights_path = job_dir / "weights" / f"{dataset}.pt"
        if metrics_path.exists() and weights_path.exists():
            record = read_json(metrics_path)
            if record.get("config_digest") != resolved["config_digest"]:
                raise RuntimeError(f"stale companion result for {dataset}")
            if record.get("weights_sha256") != file_sha256(weights_path):
                raise RuntimeError(f"companion weights changed for {dataset}")
            dataset_hashes[dataset] = file_sha256(metrics_path)
            continue
        num_train = min(int(DATASET_SIZES[dataset]) - 100, 1024)
        X_train, y_train, X_test, y_test = get_xy_traintest(
            num_train,
            dataset,
            config["model"]["hook_name"],
            model_name=config["model"]["transformer_lens_name"],
            model_cache_path=model_cache,
            seed=probe_seed,
        )
        test_labels = torch.as_tensor(y_test).detach().cpu()
        heldout_identity = _heldout_identity(config, dataset, probe_seed, len(test_labels))
        original = find_best_reg(X_train, y_train, X_test, y_test, seed=probe_seed)
        metrics: dict[str, Any] = {"original_residual": _metrics_dict(original), "methods": {}}
        weights: dict[str, Any] = {
            "schema_version": 2,
            "config_digest": resolved["config_digest"],
            "checkpoint_id": checkpoint_id,
            "probe_seed": probe_seed,
            "dataset": dataset,
            "heldout": {
                **heldout_identity,
                "label": test_labels,
                "decision_score_semantics": "sklearn_binary_logistic_score_for_classifier_classes_index_1",
                "original_residual": _heldout_classifier_outputs(original, X_test),
                "methods": {},
            },
            "original_residual": _classifier_state(original),
        }
        for method, adapter in adapters.items():
            train_code, train_reconstruction = _representations(
                adapter, X_train, device=torch_device
            )
            test_code, test_reconstruction = _representations(
                adapter, X_test, device=torch_device
            )
            full_code = find_best_reg(
                train_code, y_train, test_code, y_test, seed=probe_seed
            )
            reconstruction = find_best_reg(
                train_reconstruction,
                y_train,
                test_reconstruction,
                y_test,
                seed=probe_seed,
            )
            metrics["methods"][method] = {
                "full_code": _metrics_dict(full_code),
                "reconstruction": _metrics_dict(reconstruction),
            }
            weights[method] = {
                "full_code": _classifier_state(full_code),
                "reconstruction": _classifier_state(reconstruction),
            }
            weights["heldout"]["methods"][method] = {
                "full_code": _heldout_classifier_outputs(full_code, test_code),
                "reconstruction": _heldout_classifier_outputs(
                    reconstruction, test_reconstruction
                ),
            }
        atomic_torch(weights_path, weights)
        record = {
            "schema_version": 2,
            "config_digest": resolved["config_digest"],
            "checkpoint_id": checkpoint_id,
            "probe_seed": probe_seed,
            "dataset": dataset,
            "family": config["benchmark"]["family_by_dataset"][dataset],
            "num_train": num_train,
            "regularization": "sae_probes_find_best_reg_l2",
            "heldout_split_id": heldout_identity["split_id"],
            "heldout_example_count": len(test_labels),
            "heldout_example_id_policy": heldout_identity["example_id_policy"],
            "metrics": metrics,
            "weights_sha256": file_sha256(weights_path),
        }
        atomic_json(metrics_path, record)
        dataset_hashes[dataset] = file_sha256(metrics_path)
    result = {
        "schema_version": 1,
        "complete": True,
        "config_digest": resolved["config_digest"],
        "artifact_hashes": resolved["artifact_hashes"],
        "probe_seed": probe_seed,
        "dataset_count": len(dataset_hashes),
        "dataset_manifest_sha256": config["benchmark"]["dataset_manifest_sha256"],
        "dataset_hashes": dataset_hashes,
    }
    atomic_json(done, result)
    return result


def run_worker(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    checkpoint_dir: Path,
    model_cache: Path,
    cache_role: str,
    method: str,
    probe_seeds: Sequence[int],
    include_companion: bool,
    device: str,
    dependency_preflight: Mapping[str, Any],
    cache_wait_seconds: float = 21600,
) -> dict[str, Any]:
    """Run several seeds after one expensive sae-probes eager import."""

    seeds = [int(seed) for seed in probe_seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("worker probe seeds must be nonempty and unique")
    frozen_seeds = set(config["benchmark"]["probe_seeds"])
    if not set(seeds).issubset(frozen_seeds):
        raise ValueError("worker probe seeds must be a subset of the frozen seed list")
    if cache_role not in {"prepare", "wait"}:
        raise ValueError("cache role must be prepare or wait")
    if method not in {"mse", "dpsae"}:
        raise ValueError("worker method must be mse or dpsae")
    if include_companion and method != "mse":
        raise ValueError("companion jobs are assigned once, to the MSE workers")

    started = time.monotonic()
    if cache_role == "prepare":
        cache = prepare_cache(
            config=config,
            output_root=output_root,
            model_cache=model_cache,
            device=device,
        )
    else:
        cache = wait_cache(
            config=config,
            output_root=output_root,
            model_cache=model_cache,
            timeout_seconds=cache_wait_seconds,
        )

    sparse_results = []
    companion_results = []
    for seed in seeds:
        sparse_results.append(
            run_sparse_job(
                config=config,
                output_root=output_root,
                checkpoint_dir=checkpoint_dir,
                model_cache=model_cache,
                method=method,
                probe_seed=seed,
                device=device,
            )
        )
        if include_companion:
            companion_results.append(
                run_companion_job(
                    config=config,
                    output_root=output_root,
                    checkpoint_dir=checkpoint_dir,
                    model_cache=model_cache,
                    probe_seed=seed,
                    device=device,
                )
            )

    result = {
        "schema_version": 1,
        "complete": True,
        "config_digest": canonical_digest(config),
        "cache_role": cache_role,
        "cache_manifest_sha256": canonical_digest(cache),
        "method": method,
        "probe_seeds": seeds,
        "include_companion": include_companion,
        "device": device,
        "dependency_preflight": dict(dependency_preflight),
        "worker_seconds_excluding_dependency_preflight": time.monotonic() - started,
        "sparse_job_count": len(sparse_results),
        "companion_job_count": len(companion_results),
    }
    worker_name = f"{method}_{seeds[0]}_{seeds[-1]}"
    atomic_json(output_root / "workers" / f"{worker_name}.json", result)
    return result


def family_block_bootstrap(
    task_deltas: Mapping[str, float],
    family_by_task: Mapping[str, str],
    *,
    samples: int,
    seed: int,
    confidence_level: float,
) -> dict[str, float]:
    if set(task_deltas) != set(family_by_task):
        raise ValueError("task deltas and family map must have identical keys")
    grouped: dict[str, list[float]] = {}
    for task, delta in task_deltas.items():
        grouped.setdefault(family_by_task[task], []).append(float(delta))
    families = sorted(grouped)
    if len(families) < 2:
        raise ValueError("family bootstrap requires at least two families")
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.float64)
    for index in range(samples):
        selected = rng.choice(families, size=len(families), replace=True)
        draws[index] = np.mean([value for family in selected for value in grouped[family]])
    alpha = (1 - confidence_level) / 2
    return {
        "estimate": float(np.mean(list(task_deltas.values()))),
        "lower": float(np.quantile(draws, alpha)),
        "upper": float(np.quantile(draws, 1 - alpha)),
        "bootstrap_samples": samples,
        "family_count": len(families),
    }


def _load_sparse_records(
    config: Mapping[str, Any], output_root: Path
) -> dict[tuple[str, int, str], dict[str, Any]]:
    records = {}
    for method in ("mse", "dpsae"):
        for seed in config["benchmark"]["probe_seeds"]:
            job = _job_dir(config, output_root, method, seed)
            done = read_json(job / "done.json")
            if not done.get("complete") or done.get("config_digest") != canonical_digest(config):
                raise RuntimeError(f"incomplete sparse job: {job}")
            for dataset in config["benchmark"]["datasets"]:
                record = read_json(job / "provenance" / f"{dataset}.json")
                records[(method, seed, dataset)] = record
    return records


def _load_companion_records(
    config: Mapping[str, Any], output_root: Path
) -> dict[tuple[int, str], dict[str, Any]]:
    checkpoint_id = config["pilot_checkpoint"]["checkpoint_id"]
    records = {}
    for seed in config["benchmark"]["probe_seeds"]:
        job = output_root / "companion" / checkpoint_id / f"seed_{seed}"
        done = read_json(job / "done.json")
        if not done.get("complete") or done.get("config_digest") != canonical_digest(config):
            raise RuntimeError(f"incomplete companion job: {job}")
        for dataset in config["benchmark"]["datasets"]:
            path = job / "metrics" / f"{dataset}.json"
            if done["dataset_hashes"].get(dataset) != file_sha256(path):
                raise RuntimeError(f"companion result changed for {dataset}, seed {seed}")
            records[(seed, dataset)] = read_json(path)
    return records


def _candidate_records(
    config: Mapping[str, Any],
    sparse: Mapping[tuple[str, int, str], Mapping[str, Any]],
) -> list[dict[str, Any]]:
    primary_k = int(config["statistics"]["primary_k"])
    seeds = config["benchmark"]["probe_seeds"]
    minimum = math.ceil(config["candidates"]["minimum_probe_seed_frequency"] * len(seeds))
    candidates = []
    for method in ("mse", "dpsae"):
        for dataset in config["benchmark"]["datasets"]:
            feature_values: dict[int, list[tuple[int, float]]] = {}
            for seed in seeds:
                record = sparse[(method, seed, dataset)]
                row = next(item for item in record["rows"] if item["k"] == primary_k)
                for feature in row["feature_weights"]:
                    feature_values.setdefault(int(feature["feature_id"]), []).append(
                        (seed, float(feature["weight"]))
                    )
            for feature_id, values in feature_values.items():
                if len(values) < minimum:
                    continue
                identity = {
                    "checkpoint_id": config["pilot_checkpoint"]["checkpoint_id"],
                    "method": method,
                    "dataset": dataset,
                    "feature_id": feature_id,
                }
                candidates.append(
                    {
                        "candidate_id": "candidate_" + canonical_digest(identity)[:24],
                        **identity,
                        "family": config["benchmark"]["family_by_dataset"][dataset],
                        "probe_seed_frequency": len(values) / len(seeds),
                        "probe_seeds": [seed for seed, _ in values],
                        "mean_weight": float(np.mean([weight for _, weight in values])),
                        "mean_absolute_weight": float(
                            np.mean([abs(weight) for _, weight in values])
                        ),
                        "autointerp_eligible": False,
                        "blocked_on": "three_fresh_paired_checkpoint_confirmation",
                    }
                )
    return sorted(
        candidates,
        key=lambda item: (
            item["method"],
            -item["probe_seed_frequency"],
            -item["mean_absolute_weight"],
            item["dataset"],
            item["feature_id"],
        ),
    )


def aggregate_pilot(
    *, config: Mapping[str, Any], output_root: Path, wait_seconds: float = 0
) -> dict[str, Any]:
    if wait_seconds > 0:
        deadline = time.monotonic() + wait_seconds
        expected_done = [
            _job_dir(config, output_root, method, seed) / "done.json"
            for method in ("mse", "dpsae")
            for seed in config["benchmark"]["probe_seeds"]
        ] + [
            output_root
            / "companion"
            / config["pilot_checkpoint"]["checkpoint_id"]
            / f"seed_{seed}"
            / "done.json"
            for seed in config["benchmark"]["probe_seeds"]
        ]
        while not all(path.exists() for path in expected_done):
            if time.monotonic() >= deadline:
                missing = [str(path) for path in expected_done if not path.exists()]
                raise TimeoutError(f"timed out waiting for {len(missing)} exp10 jobs")
            time.sleep(10)
    resolved = load_resolved(output_root, config)
    eligibility = read_json(output_root / "eligibility.json")
    sparse = _load_sparse_records(config, output_root)
    companion = _load_companion_records(config, output_root)
    seeds = config["benchmark"]["probe_seeds"]
    primary_k = int(config["statistics"]["primary_k"])
    task_method_metrics: dict[str, dict[str, dict[str, float]]] = {}
    seed_deltas: dict[int, list[float]] = {seed: [] for seed in seeds}
    for dataset in config["benchmark"]["datasets"]:
        task_method_metrics[dataset] = {}
        for method in ("mse", "dpsae"):
            metric_rows = []
            for seed in seeds:
                row = next(
                    item
                    for item in sparse[(method, seed, dataset)]["rows"]
                    if item["k"] == primary_k
                )
                metric_rows.append(row["metrics"])
            task_method_metrics[dataset][method] = {
                metric: float(np.mean([row[metric] for row in metric_rows]))
                for metric in ("test_auc", "test_acc", "test_f1")
            }
        for seed in seeds:
            dpsae_row = next(
                item
                for item in sparse[("dpsae", seed, dataset)]["rows"]
                if item["k"] == primary_k
            )
            mse_row = next(
                item
                for item in sparse[("mse", seed, dataset)]["rows"]
                if item["k"] == primary_k
            )
            seed_deltas[seed].append(
                float(dpsae_row["metrics"]["test_auc"])
                - float(mse_row["metrics"]["test_auc"])
            )

    task_deltas = {
        dataset: metrics["dpsae"]["test_auc"] - metrics["mse"]["test_auc"]
        for dataset, metrics in task_method_metrics.items()
    }
    companion_task_metrics: dict[str, Any] = {}
    for dataset in config["benchmark"]["datasets"]:
        rows = [companion[(seed, dataset)]["metrics"] for seed in seeds]
        original = {
            metric: float(np.mean([row["original_residual"][metric] for row in rows]))
            for metric in ("test_auc", "test_acc", "test_f1")
        }
        methods = {}
        for method in ("mse", "dpsae"):
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
        full_code_delta = (
            methods["dpsae"]["full_code"]["test_auc"]
            - methods["mse"]["full_code"]["test_auc"]
        )
        reconstruction_delta = (
            methods["dpsae"]["reconstruction"]["test_auc"]
            - methods["mse"]["reconstruction"]["test_auc"]
        )
        companion_task_metrics[dataset] = {
            "original_residual": original,
            "methods": methods,
            "paired_full_code_auc_delta": full_code_delta,
            "paired_reconstruction_auc_delta": reconstruction_delta,
            "excess_sparse_gain_auc": task_deltas[dataset] - full_code_delta,
        }
    stats = config["statistics"]
    interval = family_block_bootstrap(
        task_deltas,
        config["benchmark"]["family_by_dataset"],
        samples=int(stats["bootstrap_samples"]),
        seed=int(stats["bootstrap_seed"]),
        confidence_level=float(stats["confidence_level"]),
    )
    seed_macro = {seed: float(np.mean(values)) for seed, values in seed_deltas.items()}
    reseed_se = float(np.std(list(seed_macro.values()), ddof=1) / math.sqrt(len(seeds)))
    family_deltas: dict[str, list[float]] = {}
    for dataset, delta in task_deltas.items():
        family = config["benchmark"]["family_by_dataset"][dataset]
        family_deltas.setdefault(family, []).append(delta)
    family_means = {family: float(np.mean(values)) for family, values in family_deltas.items()}
    positive_families = sum(value > 0 for value in family_means.values())
    checks = {
        "eligibility": eligibility.get("passed") is True,
        "complete_matrix": (
            len(sparse) == 2 * len(seeds) * len(task_deltas)
            and len(companion) == len(seeds) * len(task_deltas)
        ),
        "point_delta": interval["estimate"] > float(stats["minimum_point_delta_auc"]),
        "lower_confidence_bound": interval["lower"] > 0,
        "probe_reseed_stability": interval["estimate"]
        > float(stats["minimum_effect_to_probe_reseed_se_ratio"]) * reseed_se,
        "multiple_positive_families": positive_families
        >= int(stats["minimum_positive_families"]),
    }
    advance = all(checks.values())
    candidates = _candidate_records(config, sparse)
    candidate_path = output_root / "candidate_associations.jsonl"
    atomic_jsonl(candidate_path, candidates)
    candidate_manifest = {
        "schema_version": 1,
        "config_digest": resolved["config_digest"],
        "candidate_count": len(candidates),
        "candidate_jsonl_sha256": file_sha256(candidate_path),
        "pilot_gate": {"passed": advance, "checks": checks},
        "confirmation_gate": {
            "passed": False,
            "required_checkpoint_count": 3,
            "reason": "pilot is excluded from fresh confirmation",
        },
        "autointerp_eligible": False,
    }
    atomic_json(output_root / "candidate_manifest.json", candidate_manifest)
    report = {
        "schema_version": 1,
        "complete": True,
        "config_digest": resolved["config_digest"],
        "artifact_hashes": resolved["artifact_hashes"],
        "primary": {
            "metric": "paired_dpsae_minus_mse_test_auc",
            "k": primary_k,
            "family_block_interval": interval,
            "probe_reseed_standard_error": reseed_se,
            "probe_seed_macro_deltas": seed_macro,
            "family_mean_deltas": family_means,
            "positive_family_count": positive_families,
        },
        "checks": checks,
        "advance_fresh_confirmation": advance,
        "advance_autointerp": False,
        "autointerp_blocked_on": "fresh_three_seed_confirmation",
        "task_metrics": task_method_metrics,
        "companion_task_metrics": companion_task_metrics,
        "companion_macro": {
            "paired_full_code_auc_delta": float(
                np.mean(
                    [value["paired_full_code_auc_delta"] for value in companion_task_metrics.values()]
                )
            ),
            "paired_reconstruction_auc_delta": float(
                np.mean(
                    [
                        value["paired_reconstruction_auc_delta"]
                        for value in companion_task_metrics.values()
                    ]
                )
            ),
            "excess_sparse_gain_auc": float(
                np.mean([value["excess_sparse_gain_auc"] for value in companion_task_metrics.values()])
            ),
        },
        "candidate_manifest_sha256": file_sha256(output_root / "candidate_manifest.json"),
    }
    atomic_json(output_root / "advancement_report.json", report)
    return report


def _path(value: str) -> Path:
    return Path(value).expanduser()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=_path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=_path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=_path)
    parser.add_argument("--saebench-root", type=_path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    eligibility_parser = subparsers.add_parser("eligibility")
    eligibility_parser.add_argument("--device", default="cpu")

    subparsers.add_parser("freeze")

    cache_parser = subparsers.add_parser("prepare-cache")
    cache_parser.add_argument("--model-cache", type=_path, required=True)
    cache_parser.add_argument("--device", default="cuda:0")

    wait_parser = subparsers.add_parser("wait-cache")
    wait_parser.add_argument("--model-cache", type=_path, required=True)
    wait_parser.add_argument("--timeout-seconds", type=float, default=21600)

    sparse_parser = subparsers.add_parser("run-sparse")
    sparse_parser.add_argument("--model-cache", type=_path, required=True)
    sparse_parser.add_argument("--method", choices=("mse", "dpsae"), required=True)
    sparse_parser.add_argument("--probe-seed", type=int, required=True)
    sparse_parser.add_argument("--device", default="cuda:0")

    companion_parser = subparsers.add_parser("run-companion")
    companion_parser.add_argument("--model-cache", type=_path, required=True)
    companion_parser.add_argument("--probe-seed", type=int, required=True)
    companion_parser.add_argument("--device", default="cuda:0")

    worker_parser = subparsers.add_parser("run-worker")
    worker_parser.add_argument("--model-cache", type=_path, required=True)
    worker_parser.add_argument("--cache-role", choices=("prepare", "wait"), required=True)
    worker_parser.add_argument("--method", choices=("mse", "dpsae"), required=True)
    worker_parser.add_argument("--probe-seeds", type=int, nargs="+", required=True)
    worker_parser.add_argument("--include-companion", action="store_true")
    worker_parser.add_argument("--cache-wait-seconds", type=float, default=21600)
    worker_parser.add_argument("--device", default="cuda:0")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--wait-seconds", type=float, default=0)

    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint_dir = resolve_checkpoint_dir(config, args.checkpoint_dir)
    args.output_root = args.output_root.resolve()

    if args.command == "eligibility":
        report = assess_eligibility(config, checkpoint_dir, device=torch.device(args.device))
        atomic_json(args.output_root / "eligibility.json", report)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(2)
    elif args.command == "freeze":
        if args.saebench_root is None:
            parser.error("freeze requires --saebench-root")
        print(
            json.dumps(
                freeze_run(
                    config_path=args.config,
                    output_root=args.output_root,
                    checkpoint_dir=checkpoint_dir,
                    saebench_root=args.saebench_root,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "prepare-cache":
        if args.saebench_root is None:
            parser.error("prepare-cache requires --saebench-root")
        verify_saebench_environment(config, args.saebench_root)
        print(
            json.dumps(
                prepare_cache(
                    config=config,
                    output_root=args.output_root,
                    model_cache=args.model_cache.resolve(),
                    device=args.device,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "wait-cache":
        print(
            json.dumps(
                wait_cache(
                    config=config,
                    output_root=args.output_root,
                    model_cache=args.model_cache.resolve(),
                    timeout_seconds=args.timeout_seconds,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "run-sparse":
        if args.saebench_root is None:
            parser.error("run-sparse requires --saebench-root")
        verify_saebench_environment(config, args.saebench_root)
        print(
            json.dumps(
                run_sparse_job(
                    config=config,
                    output_root=args.output_root,
                    checkpoint_dir=checkpoint_dir,
                    model_cache=args.model_cache.resolve(),
                    method=args.method,
                    probe_seed=args.probe_seed,
                    device=args.device,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "run-companion":
        if args.saebench_root is None:
            parser.error("run-companion requires --saebench-root")
        verify_saebench_environment(config, args.saebench_root)
        print(
            json.dumps(
                run_companion_job(
                    config=config,
                    output_root=args.output_root,
                    checkpoint_dir=checkpoint_dir,
                    model_cache=args.model_cache.resolve(),
                    probe_seed=args.probe_seed,
                    device=args.device,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "run-worker":
        if args.saebench_root is None:
            parser.error("run-worker requires --saebench-root")
        preflight_started = time.monotonic()
        environment = verify_saebench_environment(config, args.saebench_root)
        environment = {
            **environment,
            "total_dependency_preflight_seconds": time.monotonic() - preflight_started,
        }
        print(
            json.dumps(
                run_worker(
                    config=config,
                    output_root=args.output_root,
                    checkpoint_dir=checkpoint_dir,
                    model_cache=args.model_cache.resolve(),
                    cache_role=args.cache_role,
                    method=args.method,
                    probe_seeds=args.probe_seeds,
                    include_companion=args.include_companion,
                    device=args.device,
                    dependency_preflight=environment,
                    cache_wait_seconds=args.cache_wait_seconds,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "aggregate":
        print(
            json.dumps(
                aggregate_pilot(
                    config=config,
                    output_root=args.output_root,
                    wait_seconds=args.wait_seconds,
                ),
                indent=2,
                sort_keys=True,
            )
        )


if __name__ == "__main__":
    main()
