#!/usr/bin/env python3
"""Frozen, resumable concept-discovery pilot for the Pythia SAE pair.

Heavy dependencies are imported only inside execution stages. This keeps the
repository testable in its normal environment while requiring an exact clean
checkout of the pinned SAEBench commit for benchmark work.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
import resource
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, NamedTuple, Sequence

import numpy as np
import torch
from torch import Tensor

from dpsae.cpu_quota import resolve_cpu_budget
from dpsae.saebench_adapter import (
    NativeBatchTopKSAEBenchAdapter,
    load_native_saebench_adapter,
    one_based_resid_post_hook,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/exp10_concept_discovery.json"
DEFAULT_OUTPUT = ROOT / "artifacts/exp10_concept_discovery"


@dataclass
class CompanionProbeMetrics:
    test_f1: float
    test_acc: float
    test_auc: float
    val_auc: float


class CompanionProbeResult(NamedTuple):
    metrics: CompanionProbeMetrics
    classifier: Any
    scaler: None


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
    if (
        config.get("schema_version") != 1
        or config.get("experiment_id") != "exp10_concept_discovery"
    ):
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
    if benchmark.get("companion_full_code_matrix_format") != "scipy_csr_exact_values":
        raise ValueError("companion full-code probes must use the frozen exact-value CSR format")
    if (
        benchmark.get("companion_l2_path_optimization")
        != "batched_all_representations_independent_cold_C_loky_cold_selected_C_refit"
    ):
        raise ValueError("companion L2 probes must use the frozen parallel cold-C optimization")
    if "unfiltered logreg baseline" not in benchmark["companion_regularization_rationale"]:
        raise ValueError("companion regularization rationale is missing")
    model = config["model"]
    layer, hook = one_based_resid_post_hook(model["one_based_block"])
    if layer != model["transformer_lens_hook_layer"] or hook != model["hook_name"]:
        raise ValueError("one-based block and TransformerLens hook disagree")
    if config["adapter"]["decoder_renormalization"] != "forbidden":
        raise ValueError("exp10 forbids adapter-side decoder renormalization")
    source_files = config.get("provenance", {}).get("source_files")
    required_source_files = {
        "configs/exp10_concept_discovery.json",
        "experiments/exp10_concept_discovery.py",
        "src/dpsae/saebench_adapter.py",
        "src/dpsae/cpu_quota.py",
        "scripts/audit_exp10_artifacts.py",
        "scripts/run_exp10_concept_4xa40.sh",
        "scripts/run_exp10_timing_smoke_a40.sh",
        "scripts/run_steps1_4_autonomous_runpod.sh",
    }
    if not isinstance(source_files, list) or set(source_files) != required_source_files:
        raise ValueError("exp10 provenance source_files changed")
    if benchmark.get("saebench_include_llm_baseline") is not False:
        raise ValueError(
            "SAEBench's duplicate residual baseline must remain disabled; the companion "
            "evaluator retains it once per seed and task"
        )
    runtime = config["runtime"]
    worker_count = int(runtime["worker_count"])
    if worker_count != 4 or len(runtime["sparse_worker_shards"]) != worker_count:
        raise ValueError("exp10 requires exactly four frozen sparse worker shards")
    if len(runtime["companion_seed_shards"]) != worker_count:
        raise ValueError("exp10 requires exactly four frozen companion seed shards")
    if int(runtime.get("companion_full_code_cold_C_jobs_per_worker", 0)) != 8:
        raise ValueError("exp10 requires eight parallel cold-C jobs per full-code worker")
    if runtime.get("thread_quota_policy") != (
        "floor_effective_cpu_count_divided_by_worker_count"
    ):
        raise ValueError("exp10 must derive thread quotas from the effective cgroup CPU budget")
    expected_resources = runtime.get("resource_identity")
    if expected_resources != {
        "cgroup_quota_cores": 32.3,
        "effective_cpu_count": 32,
        "threads_per_worker": 8,
    }:
        raise ValueError("exp10 runtime resource identity changed")
    sparse_pairs = [
        (shard["method"], int(seed))
        for shard in runtime["sparse_worker_shards"]
        for seed in shard["probe_seeds"]
    ]
    expected_sparse_pairs = [(method, int(seed)) for method in ("mse", "dpsae") for seed in seeds]
    if sorted(sparse_pairs) != sorted(expected_sparse_pairs):
        raise ValueError("sparse worker shards must cover each method/seed pair exactly once")
    companion_seeds = [int(seed) for shard in runtime["companion_seed_shards"] for seed in shard]
    if sorted(companion_seeds) != sorted(seeds) or sorted(
        len(shard) for shard in runtime["companion_seed_shards"]
    ) != [2, 2, 3, 3]:
        raise ValueError("companion shards must cover ten seeds exactly once as 3/3/2/2")
    smoke = runtime["timing_smoke"]
    if (
        runtime.get("cold_cache_timing_source_policy")
        != "in_process_monotonic_or_hash_bound_external_provenance"
        or int(smoke["probe_seed"]) in seeds
        or int(smoke["task_count"]) != 8
        or smoke.get("topology_mode") != "four_worker_same_tasks"
        or int(smoke.get("measured_worker_count", 0)) != worker_count
        or smoke.get("same_task_set_per_worker") is not True
        or float(smoke.get("barrier_timeout_seconds", 0)) <= 0
        or float(smoke.get("maximum_start_skew_seconds", 0)) <= 0
        or int(smoke["quartile_count"]) != 4
        or int(smoke["tasks_per_quartile"]) != 2
        or float(smoke["headroom_multiplier"]) != 1.3
        or float(smoke["maximum_projected_pod_hours"]) != 3.0
        or int(smoke["projection_pair_units"]) != len(datasets) * len(seeds)
    ):
        raise ValueError("blind timing-smoke contract changed")
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
    pair_relative_l0 = abs(float(dpsae["inference_l0"]) - float(mse["inference_l0"])) / target_l0
    checks["nmse_ratio"] = ratio <= float(gate["maximum_dpsae_to_mse_nmse_ratio"])
    checks["l0_target"] = all(
        value <= float(gate["maximum_relative_l0_error"]) for value in relative_l0.values()
    )
    checks["l0_pair_match"] = pair_relative_l0 <= float(gate["maximum_pair_relative_l0_difference"])
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


def verify_saebench_environment(config: Mapping[str, Any], saebench_root: Path) -> dict[str, Any]:
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
    resolved_config = config_path if config_path.is_absolute() else ROOT / config_path
    config = read_json(resolved_config)
    source_files = config.get("provenance", {}).get("source_files")
    if not isinstance(source_files, list) or not source_files:
        raise ValueError("exp10 provenance requires a nonempty source_files list")
    if len(source_files) != len(set(source_files)):
        raise ValueError("exp10 provenance source_files must be unique")
    paths = [ROOT / str(path) for path in source_files]
    if resolved_config.resolve() not in [path.resolve() for path in paths]:
        raise ValueError("exp10 provenance source_files must include its config")
    resolved = [path.resolve() for path in paths]
    if any(ROOT not in path.parents for path in resolved):
        raise ValueError("exp10 provenance source_files must remain inside the repository")
    return {str(path.relative_to(ROOT)): file_sha256(path) for path in resolved}


def repository_state() -> dict[str, Any]:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=ROOT,
        text=True,
    ).splitlines()
    if status:
        raise RuntimeError("exp10 requires a clean repository revision")
    return {"revision": revision, "dirty": False, "status": status}


def stable_resolved_contract(value: Mapping[str, Any]) -> dict[str, Any]:
    stable = json.loads(json.dumps(value))
    stable.get("environment", {}).pop("sae_probes_eager_import_seconds", None)
    return stable


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
        "repository": repository_state(),
        "environment": environment,
        "source_hashes": source_hashes(config_path),
    }
    path = output_root / "resolved_config.json"
    if path.exists():
        existing = read_json(path)
        if stable_resolved_contract(existing) != stable_resolved_contract(resolved):
            raise RuntimeError("resolved exp10 run changed; use a fresh output root")
        return existing
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
    _verify_cache_manifest_inputs(config, observed, model_cache)
    return observed


def _verify_cache_manifest_inputs(
    config: Mapping[str, Any], observed: Mapping[str, Any], model_cache: Path
) -> None:
    if set(observed.get("files", {})) != set(config["benchmark"]["datasets"]):
        raise RuntimeError("activation cache manifest dataset set changed")
    if observed.get("dataset_manifest_sha256") != config["benchmark"]["dataset_manifest_sha256"]:
        raise RuntimeError("activation cache dataset manifest changed")
    for dataset, cache_path in cache_files(config, model_cache).items():
        record = observed["files"][dataset]
        if str(cache_path.resolve()) != record.get("path") or not cache_path.is_file():
            raise RuntimeError(f"activation cache path changed for {dataset}")
        if cache_path.stat().st_size != record.get("bytes"):
            raise RuntimeError(f"activation cache size changed for {dataset}")
        if "sha256" not in record:
            raise RuntimeError(f"activation cache lacks a frozen hash for {dataset}")


def prepare_cache(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    model_cache: Path,
    device: str,
    cold_cache_provenance: Path | None = None,
) -> dict[str, Any]:
    resolved = load_resolved(output_root, config)
    ready = output_root / "cache_ready.json"
    if ready.exists():
        observed = read_json(ready)
        if observed.get("config_digest") == resolved["config_digest"]:
            return verify_cache_ready(config, output_root, model_cache)
        if cold_cache_provenance is None or not cold_cache_provenance.is_file():
            raise RuntimeError(
                "existing cache belongs to the prior config; provide hash-bound cold-cache "
                "timing provenance to adopt it without regeneration"
            )
        return _adopt_cache_from_provenance(
            config=config,
            resolved=resolved,
            output_root=output_root,
            model_cache=model_cache,
            source_cache_ready=ready,
            provenance_path=cold_cache_provenance,
        )
    if cold_cache_provenance is not None and cold_cache_provenance.is_file():
        provenance = read_json(cold_cache_provenance)
        source_value = provenance.get("source_cache_ready_path")
        if isinstance(source_value, str) and Path(source_value).expanduser().is_file():
            return _adopt_cache_from_provenance(
                config=config,
                resolved=resolved,
                output_root=output_root,
                model_cache=model_cache,
                source_cache_ready=Path(source_value).expanduser().resolve(),
                provenance_path=cold_cache_provenance,
            )
    generation_started = time.monotonic()
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
        manifest["generation_seconds"] = time.monotonic() - generation_started
        manifest["generation_timing_source"] = "in_process_monotonic"
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


def _job_dir(config: Mapping[str, Any], output_root: Path, method: str, probe_seed: int) -> Path:
    checkpoint_id = config["pilot_checkpoint"]["checkpoint_id"]
    return output_root / "jobs" / checkpoint_id / method / f"seed_{probe_seed}"


def _raw_sparse_files(job_dir: Path, model_name: str, hook_name: str) -> list[Path]:
    return sorted(
        job_dir.glob(
            f"raw/*_custom_sae/sae_probes_{model_name}/normal_setting/*_{hook_name}_l1.json"
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
    example_ids = [f"{split_id}-{index:05d}" for index in range(example_count)]
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
    raw_by_dataset = {
        _parse_dataset(path): path
        for path in _raw_sparse_files(job_dir, model["transformer_lens_name"], model["hook_name"])
    }
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
            if not predictions_path.is_file() or observed.get(
                "heldout_predictions_sha256"
            ) != file_sha256(predictions_path):
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
    adapter: NativeBatchTopKSAEBenchAdapter | None = None,
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
            if not predictions.is_file() or record.get("heldout_predictions_sha256") != file_sha256(
                predictions
            ):
                raise RuntimeError(f"completed sparse predictions changed for {dataset}")
        return value
    if adapter is None:
        adapter = load_adapter(config, checkpoint_dir, method, torch.device(device))
    elif adapter.method != method:
        raise ValueError(f"cached adapter method {adapter.method!r} does not match {method!r}")

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
        include_llm_baseline=bool(config["benchmark"]["saebench_include_llm_baseline"]),
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


def _full_code_csr(code: Tensor) -> Any:
    """Preserve a BatchTopK code exactly while exposing its sparsity to sklearn.

    ``find_best_reg`` accepts scipy CSR matrices without changing its CV grid,
    solver, seed, or fitted outputs. BatchTopK codes are extremely wide but
    contain only a few exact nonzeros per row, so passing the dense tensor makes
    every lbfgs matrix product scan thousands of known-zero columns.
    """

    from scipy.sparse import csr_matrix

    class _RowCountCSR(csr_matrix):
        """CSR compatibility shim for sae-probes' ``len(X_train)`` call."""

        def __len__(self) -> int:
            return int(self.shape[0])

    if code.ndim != 2 or code.layout != torch.strided:
        raise ValueError("full-code CSR conversion requires a dense rank-two tensor")
    if code.device.type != "cpu":
        raise ValueError("full-code CSR conversion requires a CPU tensor")
    if not torch.isfinite(code).all():
        raise ValueError("full-code CSR conversion requires finite values")
    dense = code.detach().contiguous().numpy()
    sparse = _RowCountCSR(dense, copy=True)
    sparse.sum_duplicates()
    sparse.sort_indices()
    if sparse.nnz != int(torch.count_nonzero(code)):
        raise RuntimeError("full-code CSR conversion changed the exact nonzero support")
    return sparse


def _probe_matrix(value: Any) -> Any:
    """Normalize torch/numpy/scipy inputs without changing any numeric value."""

    from scipy.sparse import issparse

    if issparse(value):
        return value.tocsr(copy=False)
    if isinstance(value, Tensor):
        if value.device.type != "cpu":
            raise ValueError("companion probe matrices must be on CPU")
        return value.detach().contiguous().numpy()
    return np.asarray(value)


def _companion_cv_splits(X_train: Any, y_train: np.ndarray, seed: int) -> list[Any]:
    """Mirror sae-probes v0.4.0 ``get_cv`` and ``get_splits`` exactly."""

    from sklearn.model_selection import LeavePOut, StratifiedKFold

    n_samples = X_train.shape[0]
    if n_samples <= 12:
        cv: Any = LeavePOut(2)
    elif n_samples < 128:
        cv = StratifiedKFold(n_splits=6, shuffle=True, random_state=seed)
    else:
        val_size = min(int(0.2 * n_samples), 100)
        train_size = n_samples - val_size
        cv = [(list(range(train_size)), list(range(train_size, n_samples)))]
    if not hasattr(cv, "split"):
        return list(cv)
    return [
        (train_index, val_index)
        for train_index, val_index in cv.split(X_train, y_train)
        if len(np.unique(y_train[val_index])) == 2
    ]


def _cold_l2_validation_score(
    C: float,
    train: Any,
    train_labels: np.ndarray,
    splits: Sequence[Any],
    seed: int,
) -> float:
    """Evaluate one frozen C with fresh cold estimators on every CV fold."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    fold_scores = []
    for train_index, val_index in splits:
        fold_labels = train_labels[train_index]
        if len(np.unique(fold_labels)) < 2:
            fold_scores.append(0.5)
            continue
        model = LogisticRegression(C=C, random_state=seed, max_iter=1000)
        model.fit(train[train_index], fold_labels)
        probability = model.predict_proba(train[val_index])[:, 1]
        fold_scores.append(float(roc_auc_score(train_labels[val_index], probability)))
    return float(np.mean(fold_scores))


def _finalize_cold_l2_result(
    *,
    best_C: float | None,
    avg_scores: Sequence[float],
    train: Any,
    train_labels: np.ndarray,
    test: Any,
    test_labels: np.ndarray,
    seed: int,
) -> CompanionProbeResult:
    """Mirror the upstream selected-C refit and held-out metrics exactly."""

    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

    final_model = LogisticRegression(
        **({"C": best_C} if best_C is not None else {}),
        random_state=seed,
        max_iter=1000,
    )
    rng = np.random.RandomState(seed)
    shuffle_idx = rng.permutation(train.shape[0])
    shuffled_train = train[shuffle_idx]
    shuffled_labels = train_labels[shuffle_idx]
    final_model.fit(shuffled_train, shuffled_labels)

    prediction = final_model.predict(test)
    probability = final_model.predict_proba(test)[:, 1]
    if best_C is None:
        val_auc = float(
            roc_auc_score(
                shuffled_labels,
                final_model.predict_proba(shuffled_train)[:, 1],
            )
        )
    else:
        val_auc = float(np.max(avg_scores))
    metrics = CompanionProbeMetrics(
        test_f1=float(f1_score(test_labels, prediction, average="weighted")),
        test_acc=float(accuracy_score(test_labels, prediction)),
        test_auc=float(roc_auc_score(test_labels, probability)),
        val_auc=val_auc,
    )
    return CompanionProbeResult(metrics=metrics, classifier=final_model, scaler=None)


def find_best_reg_l2_parallel_cold_C_batch(
    representations: Mapping[str, tuple[Any, Any]],
    y_train: Any,
    y_test: Any,
    *,
    seed: int,
    n_jobs: int,
    _parallel_backend: str = "loky",
) -> dict[str, CompanionProbeResult]:
    """Fit several frozen L2 probes through one exact cold-C work queue.

    Each representation retains its own ten candidates, estimators, scores,
    original-order tie rule, and selected-C refit.  Pooling only the independent
    validation jobs prevents five short probe grids from repeatedly leaving the
    worker pool underfilled.
    """

    from joblib import Parallel, delayed, parallel_config

    if not representations:
        raise ValueError("batched companion probes require at least one representation")
    if n_jobs < 1:
        raise ValueError("parallel cold-C job count must be positive")
    if _parallel_backend not in {"loky", "threading"}:
        raise ValueError("parallel cold-C backend must be loky or threading")

    train_labels = np.asarray(y_train)
    test_labels = np.asarray(y_test)
    normalized: dict[str, tuple[Any, Any]] = {}
    for name, (X_train, X_test) in representations.items():
        if not isinstance(name, str) or not name:
            raise ValueError("batched companion representation names must be nonempty strings")
        train = _probe_matrix(X_train)
        test = _probe_matrix(X_test)
        if train.shape[0] != len(train_labels) or test.shape[0] != len(test_labels):
            raise ValueError("companion probe matrix/label lengths differ")
        if train.shape[1] != test.shape[1]:
            raise ValueError("companion train/test widths differ")
        normalized[name] = (train, test)

    original_cs = np.logspace(5, -5, 10)
    names = list(normalized)
    average_scores: dict[str, list[float]] = {name: [] for name in names}
    best_cs: dict[str, float | None] = {name: None for name in names}
    backend_options = {"inner_max_num_threads": 1} if _parallel_backend == "loky" else {}
    jobs: list[tuple[str, float, Any]] = []
    if len(train_labels) > 3:
        first_train = next(iter(normalized.values()))[0]
        splits = _companion_cv_splits(first_train, train_labels, seed)
        jobs = [
            (name, float(C), train)
            for name, (train, _test) in normalized.items()
            for C in original_cs
        ]
    with parallel_config(backend=_parallel_backend, **backend_options):
        with Parallel(n_jobs=min(n_jobs, max(len(jobs), len(names)))) as parallel:
            scores = parallel(
                delayed(_cold_l2_validation_score)(
                    C,
                    train,
                    train_labels,
                    splits,
                    seed,
                )
                for _name, C, train in jobs
            )
            if jobs:
                offset = 0
                for name in names:
                    values = [
                        float(value) for value in scores[offset : offset + len(original_cs)]
                    ]
                    average_scores[name] = values
                    best_cs[name] = float(original_cs[int(np.argmax(values))])
                    offset += len(original_cs)
                if offset != len(scores):
                    raise RuntimeError("batched cold-C score accounting drifted")

            finalized = parallel(
                delayed(_finalize_cold_l2_result)(
                    best_C=best_cs[name],
                    avg_scores=average_scores[name],
                    train=normalized[name][0],
                    train_labels=train_labels,
                    test=normalized[name][1],
                    test_labels=test_labels,
                    seed=seed,
                )
                for name in names
            )
    return dict(zip(names, finalized, strict=True))


def find_best_reg_l2_parallel_cold_C(
    X_train: Any,
    y_train: Any,
    X_test: Any,
    y_test: Any,
    *,
    seed: int,
    n_jobs: int,
    _parallel_backend: str = "loky",
) -> CompanionProbeResult:
    """Run the exact sae-probes cold L2 fits concurrently across C values.

    Every candidate remains a fresh cold sklearn estimator. Joblib returns
    scores in the original high-to-low-C order, preserving ``np.argmax`` ties,
    and the selected C receives the exact upstream cold shuffled full refit.
    """

    return find_best_reg_l2_parallel_cold_C_batch(
        {"single": (X_train, X_test)},
        y_train,
        y_test,
        seed=seed,
        n_jobs=n_jobs,
        _parallel_backend=_parallel_backend,
    )["single"]


def run_companion_job(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    checkpoint_dir: Path,
    model_cache: Path,
    probe_seed: int,
    device: str,
    adapters: Mapping[str, NativeBatchTopKSAEBenchAdapter] | None = None,
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
    torch_device = torch.device(device)
    if adapters is None:
        adapters = {
            method: load_adapter(config, checkpoint_dir, method, torch_device)
            for method in ("mse", "dpsae")
        }
    if set(adapters) != {"mse", "dpsae"} or any(
        adapters[method].method != method for method in ("mse", "dpsae")
    ):
        raise ValueError("cached companion adapters must contain exact MSE and DPSAE models")
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
        method_representations: dict[str, tuple[Tensor, Tensor, Tensor, Tensor]] = {}
        probe_representations: dict[str, tuple[Any, Any]] = {
            "original_residual": (X_train, X_test)
        }
        for method, adapter in adapters.items():
            train_code, train_reconstruction = _representations(
                adapter, X_train, device=torch_device
            )
            test_code, test_reconstruction = _representations(
                adapter, X_test, device=torch_device
            )
            method_representations[method] = (
                train_code,
                train_reconstruction,
                test_code,
                test_reconstruction,
            )
            probe_representations[f"{method}.full_code"] = (
                _full_code_csr(train_code),
                _full_code_csr(test_code),
            )
            probe_representations[f"{method}.reconstruction"] = (
                train_reconstruction,
                test_reconstruction,
            )
        probe_results = find_best_reg_l2_parallel_cold_C_batch(
            probe_representations,
            y_train,
            y_test,
            seed=probe_seed,
            n_jobs=int(config["runtime"]["companion_full_code_cold_C_jobs_per_worker"]),
        )
        original = probe_results["original_residual"]
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
        for method in adapters:
            _train_code, _train_reconstruction, test_code, test_reconstruction = (
                method_representations[method]
            )
            full_code = probe_results[f"{method}.full_code"]
            reconstruction = probe_results[f"{method}.reconstruction"]
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
                "reconstruction": _heldout_classifier_outputs(reconstruction, test_reconstruction),
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
            "full_code_matrix_format": "scipy_csr_exact_values",
            "l2_path_optimization": (
                "batched_all_representations_independent_cold_C_loky_cold_selected_C_refit"
            ),
            "full_code_cold_C_jobs": int(
                config["runtime"]["companion_full_code_cold_C_jobs_per_worker"]
            ),
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


@torch.inference_mode()
def _encode_only(
    adapter: NativeBatchTopKSAEBenchAdapter,
    activation: Tensor,
    *,
    device: torch.device,
    batch_size: int = 128,
) -> Tensor:
    return torch.cat(
        [adapter.encode(batch.to(device)).cpu() for batch in activation.split(batch_size)]
    )


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed_call(device: torch.device, function: Callable[[], Any]) -> tuple[Any, float]:
    _synchronize(device)
    started = time.perf_counter()
    value = function()
    _synchronize(device)
    return value, time.perf_counter() - started


def select_timing_smoke_tasks(
    config: Mapping[str, Any], dataset_sizes: Mapping[str, int]
) -> list[dict[str, Any]]:
    """Select two tasks per size quartile without consulting concept outcomes."""

    datasets = list(config["benchmark"]["datasets"])
    if set(datasets) != set(dataset_sizes):
        missing = sorted(set(datasets).difference(dataset_sizes))
        extra = sorted(set(dataset_sizes).difference(datasets))
        raise ValueError(f"timing dataset-size manifest drift: missing={missing}, extra={extra}")
    manifest_index = {dataset: index for index, dataset in enumerate(datasets)}
    ordered = sorted(datasets, key=lambda name: (int(dataset_sizes[name]), manifest_index[name]))
    buckets = [list(bucket) for bucket in np.array_split(np.asarray(ordered, dtype=object), 4)]
    selected = []
    for quartile, bucket in enumerate(buckets):
        if len(bucket) < 2:
            raise RuntimeError("timing smoke needs at least two tasks in every size quartile")
        positions = [len(bucket) // 3, (2 * len(bucket)) // 3]
        if positions[0] == positions[1]:
            positions[1] = min(len(bucket) - 1, positions[0] + 1)
        for sample, position in enumerate(positions):
            dataset = str(bucket[position])
            size = int(dataset_sizes[dataset])
            selected.append(
                {
                    "slot": f"quartile_{quartile}_sample_{sample}",
                    "quartile": quartile,
                    "dataset": dataset,
                    "dataset_size": size,
                    "n_train": min(size - 100, 1024),
                }
            )
    if len(selected) != 8 or len({item["dataset"] for item in selected}) != 8:
        raise RuntimeError("timing smoke must select eight distinct tasks")
    return selected


def _fit_sparse_smoke(
    *,
    codes: Mapping[str, tuple[Tensor, Tensor]],
    y_train: Any,
    y_test: Any,
    ks: Sequence[int],
    probe_seed: int,
) -> None:
    from sae_probes.run_sae_evals import get_sorted_indices, mean_act_normalization
    from sae_probes.utils_training import find_best_reg

    for train_code, test_code in codes.values():
        ranking = get_sorted_indices(train_code, y_train, normalize_fn=mean_act_normalization)
        for k in ks:
            feature_ids = ranking[: int(k)]
            result = find_best_reg(
                X_train=train_code[:, feature_ids],
                y_train=y_train,
                X_test=test_code[:, feature_ids],
                y_test=y_test,
                penalty="l1",
                seed=probe_seed,
            )
            del result


def _parent_peak_rss_mib() -> float:
    """Return parent-process peak RSS; loky children are watched separately."""

    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024 * 1024)
    return value / 1024


def _timing_thread_quota(config: Mapping[str, Any]) -> dict[str, Any]:
    worker_count = int(config["runtime"]["worker_count"])
    budget = asdict(resolve_cpu_budget(worker_count))
    _validate_cpu_budget(config, budget, label="live runtime")
    variables = (
        "LOKY_MAX_CPU_COUNT",
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    )
    observed = {name: os.environ.get(name) for name in variables}
    expected_environment = {
        "LOKY_MAX_CPU_COUNT": str(budget["effective_cpu_count"]),
        "OMP_NUM_THREADS": str(budget["threads_per_worker"]),
        "MKL_NUM_THREADS": str(budget["threads_per_worker"]),
        "OPENBLAS_NUM_THREADS": str(budget["threads_per_worker"]),
        "NUMEXPR_NUM_THREADS": str(budget["threads_per_worker"]),
    }
    if observed != expected_environment:
        raise RuntimeError(
            "runtime thread-cap identity mismatch: "
            f"expected {expected_environment}, observed {observed}"
        )
    return {**budget, "environment": observed}


def _validate_cpu_budget(
    config: Mapping[str, Any], value: Any, *, label: str
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{label} CPU budget is missing")
    budget = dict(value)
    expected_keys = {
        "visible_cpu_count",
        "cgroup_quota_cores",
        "effective_cpu_count",
        "worker_count",
        "threads_per_worker",
    }
    if set(budget) != expected_keys:
        raise RuntimeError(f"{label} CPU budget schema drift")
    expected = config["runtime"]["resource_identity"]
    for key in ("cgroup_quota_cores", "effective_cpu_count", "threads_per_worker"):
        if budget.get(key) != expected[key]:
            raise RuntimeError(
                f"{label} CPU resource drift for {key}: "
                f"expected {expected[key]!r}, observed {budget.get(key)!r}"
            )
    worker_count = int(config["runtime"]["worker_count"])
    if budget.get("worker_count") != worker_count:
        raise RuntimeError(f"{label} CPU worker count drift")
    visible = budget.get("visible_cpu_count")
    if not isinstance(visible, int) or isinstance(visible, bool) or visible < int(
        expected["effective_cpu_count"]
    ):
        raise RuntimeError(f"{label} visible CPU count is incompatible")
    return budget


def _validate_reported_runtime_resources(
    config: Mapping[str, Any], value: Any
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("timing-smoke runtime_resources is missing")
    resources = dict(value)
    budget = {key: resources.get(key) for key in resources if key != "environment"}
    _validate_cpu_budget(config, budget, label="timing-smoke")
    expected = config["runtime"]["resource_identity"]
    expected_environment = {
        "LOKY_MAX_CPU_COUNT": str(expected["effective_cpu_count"]),
        "OMP_NUM_THREADS": str(expected["threads_per_worker"]),
        "MKL_NUM_THREADS": str(expected["threads_per_worker"]),
        "OPENBLAS_NUM_THREADS": str(expected["threads_per_worker"]),
        "NUMEXPR_NUM_THREADS": str(expected["threads_per_worker"]),
    }
    if resources.get("environment") != expected_environment:
        raise RuntimeError("timing-smoke thread-cap environment drift")
    return resources


def _cache_file_hashes_digest(cache: Mapping[str, Any]) -> str:
    files = cache.get("files", {})
    hashes = {dataset: record.get("sha256") for dataset, record in files.items()}
    if not hashes or any(not value for value in hashes.values()):
        raise RuntimeError("cache timing provenance requires every frozen cache-file hash")
    return canonical_digest(hashes)


def _validate_external_cache_timing(
    cache: Mapping[str, Any],
    *,
    output_root: Path,
    model_cache: Path,
    provenance_path: Path,
    source_ready_sha256: str | None = None,
    source_ready_path: Path | None = None,
) -> dict[str, Any]:
    provenance = read_json(provenance_path)
    if provenance.get("schema_version") != 1 or provenance.get("complete") is not True:
        raise RuntimeError("cold-cache timing provenance schema is invalid")
    expected_source_hash = (
        source_ready_sha256
        or cache.get("adopted_from_cache_ready_sha256")
        or file_sha256(output_root / "cache_ready.json")
    )
    expected_model_cache = str(model_cache.resolve())
    expected_file_hashes = _cache_file_hashes_digest(cache)
    provenance_source_path = provenance.get("source_cache_ready_path")
    if not isinstance(provenance_source_path, str) or not provenance_source_path:
        raise RuntimeError("cold-cache timing provenance lacks its source manifest path")
    if (
        source_ready_path is not None
        and Path(provenance_source_path).expanduser().resolve() != source_ready_path.resolve()
    ):
        raise RuntimeError("cold-cache timing provenance source path changed")
    if provenance.get("source_cache_ready_sha256") != expected_source_hash:
        raise RuntimeError("cold-cache timing provenance is bound to another cache manifest")
    if (
        provenance.get("model_cache_path") != expected_model_cache
        or cache.get("model_cache") != expected_model_cache
    ):
        raise RuntimeError("cold-cache timing provenance model-cache path changed")
    if provenance.get("cache_file_hashes_sha256") != expected_file_hashes:
        raise RuntimeError("cold-cache timing provenance file-hash manifest changed")
    start = float(provenance.get("start_unix_seconds", math.nan))
    end = float(provenance.get("end_unix_seconds", math.nan))
    elapsed = float(provenance.get("generation_seconds", math.nan))
    if (
        not all(math.isfinite(value) for value in (start, end, elapsed))
        or elapsed <= 0
        or end <= start
        or not math.isclose(end - start, elapsed, abs_tol=1e-6, rel_tol=0)
    ):
        raise RuntimeError("cold-cache timing interval is invalid")
    provenance_sha256 = file_sha256(provenance_path)
    expected_provenance_sha256 = cache.get("cache_generation_timing_provenance_sha256")
    if expected_provenance_sha256 is not None and expected_provenance_sha256 != provenance_sha256:
        raise RuntimeError("cold-cache timing provenance hash changed after cache adoption")
    return {
        "source": "external_hash_bound_provenance",
        "generation_seconds": elapsed,
        "start_unix_seconds": start,
        "end_unix_seconds": end,
        "provenance_path": str(provenance_path.resolve()),
        "provenance_sha256": provenance_sha256,
        "source_cache_ready_sha256": expected_source_hash,
        "source_cache_ready_path": provenance_source_path,
        "model_cache_path": expected_model_cache,
        "cache_file_hashes_sha256": expected_file_hashes,
    }


def _adopt_cache_from_provenance(
    *,
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
    output_root: Path,
    model_cache: Path,
    source_cache_ready: Path,
    provenance_path: Path,
) -> dict[str, Any]:
    observed = read_json(source_cache_ready)
    _verify_cache_manifest_inputs(config, observed, model_cache)
    source_ready_sha256 = file_sha256(source_cache_ready)
    timing = _validate_external_cache_timing(
        observed,
        output_root=output_root,
        model_cache=model_cache,
        provenance_path=provenance_path,
        source_ready_sha256=source_ready_sha256,
        source_ready_path=source_cache_ready,
    )
    adopted = dict(observed)
    adopted.update(
        {
            "config_digest": resolved["config_digest"],
            "adopted_from_cache_ready_path": str(source_cache_ready.resolve()),
            "adopted_from_cache_ready_sha256": source_ready_sha256,
            "generation_seconds": timing["generation_seconds"],
            "generation_timing_source": "external_hash_bound_provenance",
            "cache_generation_timing_provenance_sha256": timing["provenance_sha256"],
        }
    )
    atomic_json(output_root / "cache_ready.json", adopted)
    return verify_cache_ready(config, output_root, model_cache)


def record_cold_cache_timing_provenance(
    *,
    config: Mapping[str, Any],
    source_cache_ready: Path,
    model_cache: Path,
    start_unix_seconds: int,
    end_unix_seconds: int,
    expected_generation_seconds: int,
    output_path: Path,
) -> dict[str, Any]:
    source_cache_ready = source_cache_ready.resolve()
    model_cache = model_cache.resolve()
    cache = read_json(source_cache_ready)
    _verify_cache_manifest_inputs(config, cache, model_cache)
    elapsed = int(end_unix_seconds) - int(start_unix_seconds)
    if elapsed <= 0:
        raise ValueError("cold-cache end time must be later than its start time")
    if elapsed != int(expected_generation_seconds):
        raise RuntimeError("supplied cold-cache generation seconds disagree with end minus start")
    recorded_elapsed = cache.get("generation_seconds")
    if recorded_elapsed is not None and not math.isclose(
        float(recorded_elapsed), float(elapsed), abs_tol=1e-6, rel_tol=0
    ):
        raise RuntimeError(
            "cache manifest generation time disagrees with the supplied wall-clock interval"
        )
    provenance = {
        "schema_version": 1,
        "complete": True,
        "source": "external_wall_clock_interval",
        "start_unix_seconds": int(start_unix_seconds),
        "end_unix_seconds": int(end_unix_seconds),
        "generation_seconds": elapsed,
        "source_cache_ready_path": str(source_cache_ready),
        "source_cache_ready_sha256": file_sha256(source_cache_ready),
        "model_cache_path": str(model_cache),
        "cache_file_hashes_sha256": _cache_file_hashes_digest(cache),
    }
    atomic_json(output_path, provenance)
    validated = _validate_external_cache_timing(
        cache,
        output_root=source_cache_ready.parent,
        model_cache=model_cache,
        provenance_path=output_path,
        source_ready_sha256=provenance["source_cache_ready_sha256"],
        source_ready_path=source_cache_ready,
    )
    return {
        "schema_version": 1,
        "complete": True,
        "output_path": str(output_path.resolve()),
        "output_sha256": file_sha256(output_path),
        "generation_seconds": validated["generation_seconds"],
        "source_cache_ready_sha256": validated["source_cache_ready_sha256"],
        "cache_file_hashes_sha256": validated["cache_file_hashes_sha256"],
    }


def _resolve_cache_generation_timing(
    cache: Mapping[str, Any],
    *,
    output_root: Path,
    model_cache: Path,
    provenance_path: Path | None,
) -> dict[str, Any]:
    if provenance_path is not None and provenance_path.is_file():
        return _validate_external_cache_timing(
            cache,
            output_root=output_root,
            model_cache=model_cache,
            provenance_path=provenance_path,
        )
    if cache.get("generation_timing_source") == "external_hash_bound_provenance":
        raise RuntimeError("adopted cache requires its exact cold-cache provenance JSON")
    elapsed = float(cache.get("generation_seconds", math.nan))
    if not math.isfinite(elapsed) or elapsed <= 0:
        raise RuntimeError(
            "cache manifest lacks cold generation time; provide hash-bound external provenance"
        )
    return {
        "source": "in_process_monotonic",
        "generation_seconds": elapsed,
        "source_cache_ready_sha256": file_sha256(output_root / "cache_ready.json"),
        "model_cache_path": str(model_cache.resolve()),
        "cache_file_hashes_sha256": _cache_file_hashes_digest(cache),
    }


def _timing_topology(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = config["runtime"]
    smoke = runtime["timing_smoke"]
    return {
        "mode": smoke["topology_mode"],
        "measured_worker_count": int(smoke["measured_worker_count"]),
        "tasks_per_worker": int(smoke["task_count"]),
        "same_task_set_per_worker": bool(smoke["same_task_set_per_worker"]),
        "barrier_synchronized": True,
        "cold_c_jobs_per_worker": int(runtime["companion_full_code_cold_C_jobs_per_worker"]),
        "parent_threads_per_worker": int(runtime["resource_identity"]["threads_per_worker"]),
        "loky_inner_max_num_threads": 1,
    }


def _read_cgroup_cpu_stat() -> dict[str, Any]:
    candidates = [Path("/sys/fs/cgroup/cpu.stat")]
    membership = Path("/proc/self/cgroup")
    if membership.is_file():
        for line in membership.read_text().splitlines():
            fields = line.split(":", 2)
            if len(fields) != 3:
                continue
            controllers = fields[1].split(",") if fields[1] else []
            relative = fields[2].lstrip("/")
            if not controllers:
                candidates.append(Path("/sys/fs/cgroup") / relative / "cpu.stat")
            elif "cpu" in controllers:
                for mount in ("cpu", "cpu,cpuacct"):
                    candidates.append(Path("/sys/fs/cgroup") / mount / relative / "cpu.stat")
    candidates.extend(
        [
            Path("/sys/fs/cgroup/cpu/cpu.stat"),
            Path("/sys/fs/cgroup/cpu,cpuacct/cpu.stat"),
        ]
    )
    for path in dict.fromkeys(candidates):
        if not path.is_file():
            continue
        counters: dict[str, int] = {}
        for line in path.read_text().splitlines():
            fields = line.split()
            if len(fields) == 2:
                counters[fields[0]] = int(fields[1])
        if {"nr_periods", "nr_throttled"}.issubset(counters) and (
            "throttled_time" in counters or "throttled_usec" in counters
        ):
            return {"path": str(path.resolve()), "counters": counters}
    raise RuntimeError("cgroup cpu.stat with throttling counters is unavailable")


def _cgroup_cpu_stat_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    if before.get("path") != after.get("path"):
        raise RuntimeError("cgroup cpu.stat path changed during timing")
    start = dict(before.get("counters", {}))
    end = dict(after.get("counters", {}))
    if set(start) != set(end):
        raise RuntimeError("cgroup cpu.stat counter schema changed during timing")
    delta = {key: int(end[key]) - int(start[key]) for key in start}
    if any(value < 0 for value in delta.values()):
        raise RuntimeError("cgroup cpu.stat counter decreased during timing")
    return {"path": before["path"], "before": start, "after": end, "delta": delta}


def _validate_cgroup_cpu_stat_delta(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping) or set(value) != {"path", "before", "after", "delta"}:
        raise RuntimeError("timing worker cgroup cpu.stat delta schema drift")
    if not isinstance(value["path"], str) or not value["path"]:
        raise RuntimeError("timing worker cgroup cpu.stat path is invalid")
    before, after, delta = value["before"], value["after"], value["delta"]
    if not all(isinstance(item, Mapping) for item in (before, after, delta)):
        raise RuntimeError("timing worker cgroup cpu.stat counters are invalid")
    if set(before) != set(after) or set(before) != set(delta):
        raise RuntimeError("timing worker cgroup cpu.stat counter sets differ")
    if not {"nr_periods", "nr_throttled"}.issubset(delta) or not (
        "throttled_time" in delta or "throttled_usec" in delta
    ):
        raise RuntimeError("timing worker cgroup cpu.stat lacks throttling counters")
    for key in delta:
        if any(
            not isinstance(item[key], int) or isinstance(item[key], bool)
            for item in (before, after, delta)
        ):
            raise RuntimeError("timing worker cgroup cpu.stat counter is not integral")
        if after[key] - before[key] != delta[key] or delta[key] < 0:
            raise RuntimeError("timing worker cgroup cpu.stat delta is inconsistent")
    if delta["nr_periods"] <= 0:
        raise RuntimeError("timing worker cgroup cpu.stat observed no active quota periods")
    return dict(value)


_TIMING_TASK_KEYS = {
    "slot",
    "quartile",
    "n_train",
    "n_test",
    "stage_seconds",
    "total_seconds",
    "parent_peak_rss_mib",
    "peak_gpu_allocated_bytes",
    "peak_gpu_reserved_bytes",
}
_SPARSE_TIMING_STAGE_KEYS = {
    "primary_data_load",
    "primary_encode",
    "primary_l1",
    "provenance_data_load",
    "provenance_encode",
    "provenance_l1",
    "total",
}
_COMPANION_TIMING_STAGE_KEYS = {
    "data_load",
    "encode_decode",
    "batched_l2",
    "total",
}
_TIMING_READY_KEYS = {
    "schema_version",
    "complete",
    "worker_index",
    "config_digest",
    "artifact_hashes",
    "source_hashes_sha256",
    "dependency_environment_sha256",
    "cache_ready_sha256",
    "cache_file_hashes_sha256",
    "selection_manifest_sha256",
    "selected_opaque_slots",
    "runtime_resources",
    "cpu_budget_sha256",
    "cache_generation_timing",
    "topology",
    "initialization_seconds",
    "ready_monotonic_seconds",
}
_TIMING_WORKER_KEYS = _TIMING_READY_KEYS | {
    "barrier_start_sha256",
    "barrier_start_monotonic_seconds",
    "measurement_started_monotonic_seconds",
    "measurement_finished_monotonic_seconds",
    "measurement_seconds",
    "task_count",
    "names_and_concept_results_suppressed",
    "saved_concept_metric_count",
    "cgroup_cpu_stat_delta",
    "tasks",
}
_TIMING_START_KEYS = {
    "schema_version",
    "complete",
    "topology",
    "worker_count",
    "common_identity_sha256",
    "start_monotonic_seconds",
    "start_unix_seconds",
    "ready_reports",
}
_TIMING_EXIT_KEYS = {
    "schema_version",
    "complete",
    "worker_index",
    "exit_code",
    "worker_report_sha256",
}
_TIMING_REPORT_KEYS = {
    "schema_version",
    "complete",
    "passed",
    "config_digest",
    "artifact_hashes",
    "source_hashes_sha256",
    "probe_seed",
    "task_count",
    "measured_task_count",
    "measured_worker_count",
    "topology",
    "selection_policy",
    "selection_manifest_sha256",
    "names_and_concept_results_suppressed",
    "saved_concept_metric_count",
    "companion_full_code_matrix_format",
    "companion_l2_path_optimization",
    "companion_full_code_cold_C_jobs_per_worker",
    "runtime_resources",
    "cpu_budget_sha256",
    "cache_generation_timing",
    "timing_worker_reports",
    "timing_worker_exit_sentinels",
    "barrier",
    "cgroup_cpu_stat_deltas",
    "projection",
}
_TIMING_BARRIER_PROOF_KEYS = {
    "synchronized",
    "start_path",
    "start_sha256",
    "common_identity_sha256",
    "ready_reports",
    "start_monotonic_seconds",
    "observed_start_skew_seconds",
    "maximum_start_skew_seconds",
}


def _validate_timing_rows(config: Mapping[str, Any], rows: Any) -> list[dict[str, Any]]:
    smoke = config["runtime"]["timing_smoke"]
    if not isinstance(rows, list) or len(rows) != int(smoke["task_count"]):
        raise RuntimeError("timing worker must contain exactly eight opaque task rows")
    observed: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if not isinstance(row, Mapping) or set(row) != _TIMING_TASK_KEYS:
            raise RuntimeError("timing task row schema or privacy boundary drift")
        expected_slot = f"quartile_{index // 2}_sample_{index % 2}"
        if row.get("slot") != expected_slot or row.get("quartile") != index // 2:
            raise RuntimeError("timing task opaque slot order drift")
        if any(
            not isinstance(row.get(key), int)
            or isinstance(row.get(key), bool)
            or int(row[key]) <= 0
            for key in ("n_train", "n_test")
        ):
            raise RuntimeError("timing task sample counts are invalid")
        stages = row.get("stage_seconds")
        if not isinstance(stages, Mapping) or set(stages) != {
            "sparse_method_0",
            "sparse_method_1",
            "companion",
        }:
            raise RuntimeError("timing task stage schema drift")
        for name, expected_keys in (
            ("sparse_method_0", _SPARSE_TIMING_STAGE_KEYS),
            ("sparse_method_1", _SPARSE_TIMING_STAGE_KEYS),
            ("companion", _COMPANION_TIMING_STAGE_KEYS),
        ):
            stage = stages[name]
            if not isinstance(stage, Mapping) or set(stage) != expected_keys:
                raise RuntimeError("timing task component stage schema drift")
            values = [float(stage[key]) for key in expected_keys]
            if not all(math.isfinite(item) and item >= 0 for item in values):
                raise RuntimeError("timing task component contains invalid duration")
            components = [float(value) for key, value in stage.items() if key != "total"]
            if not math.isclose(float(stage["total"]), sum(components), rel_tol=1e-9, abs_tol=1e-6):
                raise RuntimeError("timing task component total is inconsistent")
        for key in ("total_seconds", "parent_peak_rss_mib"):
            value = float(row[key])
            if not math.isfinite(value) or value <= 0:
                raise RuntimeError("timing task total or RSS is invalid")
        for key in ("peak_gpu_allocated_bytes", "peak_gpu_reserved_bytes"):
            if not isinstance(row[key], int) or isinstance(row[key], bool) or row[key] < 0:
                raise RuntimeError("timing task GPU memory counter is invalid")
        observed.append(dict(row))
    return observed


def _timing_common_identity(value: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "config_digest",
        "artifact_hashes",
        "source_hashes_sha256",
        "dependency_environment_sha256",
        "cache_ready_sha256",
        "cache_file_hashes_sha256",
        "selection_manifest_sha256",
        "selected_opaque_slots",
        "runtime_resources",
        "cpu_budget_sha256",
        "cache_generation_timing",
        "topology",
    )
    return {key: value.get(key) for key in keys}


def _expected_selected_slots(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [
        {"slot": f"quartile_{index // 2}_sample_{index % 2}", "quartile": index // 2}
        for index in range(int(config["runtime"]["timing_smoke"]["task_count"]))
    ]


def _expected_timing_common_identity(
    config: Mapping[str, Any], output_root: Path, candidate: Mapping[str, Any]
) -> dict[str, Any]:
    resolved = load_resolved(output_root, config)
    checkpoint_dir = Path(str(resolved.get("checkpoint_directory", "")))
    if artifact_hashes(config, checkpoint_dir) != resolved.get("artifact_hashes"):
        raise RuntimeError("resolved checkpoint artifact hashes changed")
    config_path = Path(str(resolved.get("config_path", "")))
    observed_source_hashes = source_hashes(config_path)
    if observed_source_hashes != resolved.get("source_hashes"):
        raise RuntimeError("resolved source hashes changed")
    cache_path = output_root / "cache_ready.json"
    cache = read_json(cache_path)
    model_cache = Path(str(cache.get("model_cache", "")))
    _verify_cache_manifest_inputs(config, cache, model_cache)
    for record in cache["files"].values():
        cached_path = Path(record["path"])
        if file_sha256(cached_path) != record.get("sha256"):
            raise RuntimeError("activation cache content hash changed")
    cpu_budget_path = output_root / "cpu_budget.json"
    cpu_budget = _validate_cpu_budget(config, read_json(cpu_budget_path), label="launcher")
    runtime_resources = _timing_thread_quota(config)
    if {key: runtime_resources[key] for key in cpu_budget} != cpu_budget:
        raise RuntimeError("live resources differ from the launcher CPU budget")
    from sae_probes.run_sae_evals import DATASET_SIZES

    selected = select_timing_smoke_tasks(
        config,
        {dataset: int(DATASET_SIZES[dataset]) for dataset in config["benchmark"]["datasets"]},
    )
    selection_digest = canonical_digest(
        [
            {"slot": item["slot"], "dataset": item["dataset"], "size": item["dataset_size"]}
            for item in selected
        ]
    )
    dependency_environment = stable_resolved_contract(
        {"environment": resolved.get("environment", {})}
    )["environment"]
    timing_candidate = candidate.get("cache_generation_timing")
    if not isinstance(timing_candidate, Mapping):
        raise RuntimeError("timing cache-generation identity is missing")
    provenance_path = (
        Path(str(timing_candidate.get("provenance_path")))
        if timing_candidate.get("source") == "external_hash_bound_provenance"
        else None
    )
    cache_timing = _resolve_cache_generation_timing(
        cache,
        output_root=output_root,
        model_cache=model_cache,
        provenance_path=provenance_path,
    )
    return {
        "config_digest": resolved["config_digest"],
        "artifact_hashes": resolved["artifact_hashes"],
        "source_hashes_sha256": canonical_digest(observed_source_hashes),
        "dependency_environment_sha256": canonical_digest(dependency_environment),
        "cache_ready_sha256": file_sha256(cache_path),
        "cache_file_hashes_sha256": _cache_file_hashes_digest(cache),
        "selection_manifest_sha256": selection_digest,
        "selected_opaque_slots": _expected_selected_slots(config),
        "runtime_resources": runtime_resources,
        "cpu_budget_sha256": file_sha256(cpu_budget_path),
        "cache_generation_timing": cache_timing,
        "topology": _timing_topology(config),
    }


def _timing_ready_path(output_root: Path, worker_index: int) -> Path:
    return output_root / "timing_barrier" / f"ready_{worker_index}.json"


def _timing_worker_path(output_root: Path, worker_index: int) -> Path:
    return output_root / "timing_workers" / f"worker_{worker_index}.json"


def _timing_exit_path(output_root: Path, worker_index: int) -> Path:
    return output_root / "timing_workers" / f"exit_{worker_index}.json"


def _wait_for_timing_start(
    output_root: Path, ready_path: Path, common_identity_sha256: str, timeout_seconds: float
) -> dict[str, Any]:
    start_path = output_root / "timing_barrier" / "start.json"
    deadline = time.monotonic() + timeout_seconds
    while not start_path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for the four-worker timing barrier")
        time.sleep(min(0.25, max(0.01, deadline - time.monotonic())))
    start = read_json(start_path)
    refs = start.get("ready_reports")
    if (
        set(start) != _TIMING_START_KEYS
        or start.get("schema_version") != 1
        or start.get("complete") is not True
        or start.get("topology") != "four_worker_same_tasks"
        or start.get("common_identity_sha256") != common_identity_sha256
        or not isinstance(refs, list)
        or not any(
            set(ref) == {"worker_index", "path", "sha256"}
            and ref.get("path") == str(ready_path.relative_to(output_root))
            and ref.get("sha256") == file_sha256(ready_path)
            for ref in refs
            if isinstance(ref, Mapping)
        )
    ):
        raise RuntimeError("timing barrier start artifact does not authorize this worker")
    return start


def start_timing_barrier(
    config: Mapping[str, Any], output_root: Path, *, timeout_seconds: float | None = None
) -> dict[str, Any]:
    smoke = config["runtime"]["timing_smoke"]
    worker_count = int(smoke["measured_worker_count"])
    timeout = float(timeout_seconds or smoke["barrier_timeout_seconds"])
    start_path = output_root / "timing_barrier" / "start.json"
    if start_path.exists():
        raise RuntimeError("timing barrier start artifact already exists; use a fresh timing root")
    deadline = time.monotonic() + timeout
    ready_paths = [_timing_ready_path(output_root, index) for index in range(worker_count)]
    while not all(path.is_file() for path in ready_paths):
        for index in range(worker_count):
            exit_path = _timing_exit_path(output_root, index)
            if exit_path.is_file():
                sentinel = read_json(exit_path)
                if sentinel.get("exit_code") != 0:
                    raise RuntimeError(f"timing worker {index} exited before the barrier")
        if time.monotonic() >= deadline:
            missing = [path.name for path in ready_paths if not path.is_file()]
            raise TimeoutError(f"timed out waiting for timing ready artifacts: {missing}")
        time.sleep(min(0.25, max(0.01, deadline - time.monotonic())))
    ready = [read_json(path) for path in ready_paths]
    common = _timing_common_identity(ready[0])
    common_sha256 = canonical_digest(common)
    expected_common = _expected_timing_common_identity(config, output_root, ready[0])
    if common != expected_common:
        raise RuntimeError("timing ready identity differs from frozen and live artifacts")
    for index, value in enumerate(ready):
        if (
            set(value) != _TIMING_READY_KEYS
            or value.get("schema_version") != 1
            or value.get("complete") is not True
            or value.get("worker_index") != index
            or value.get("topology") != _timing_topology(config)
            or canonical_digest(_timing_common_identity(value)) != common_sha256
            or value.get("selected_opaque_slots") != _expected_selected_slots(config)
        ):
            raise RuntimeError("timing ready reports do not share one frozen identity")
        initialization = float(value.get("initialization_seconds", math.nan))
        if not math.isfinite(initialization) or initialization <= 0:
            raise RuntimeError("timing ready report has invalid initialization duration")
    started_monotonic = time.monotonic()
    start = {
        "schema_version": 1,
        "complete": True,
        "topology": "four_worker_same_tasks",
        "worker_count": worker_count,
        "common_identity_sha256": common_sha256,
        "start_monotonic_seconds": started_monotonic,
        "start_unix_seconds": time.time(),
        "ready_reports": [
            {
                "worker_index": index,
                "path": str(path.relative_to(output_root)),
                "sha256": file_sha256(path),
            }
            for index, path in enumerate(ready_paths)
        ],
    }
    atomic_json(start_path, start)
    return start


def run_timing_worker(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    checkpoint_dir: Path,
    model_cache: Path,
    worker_index: int,
    device: str,
    dependency_environment: Mapping[str, Any],
    process_started_monotonic: float,
    cold_cache_provenance: Path | None = None,
) -> dict[str, Any]:
    """Measure one barrier-synchronized worker without writing the authoritative report."""

    topology = _timing_topology(config)
    if worker_index not in range(int(topology["measured_worker_count"])):
        raise ValueError("timing worker index is outside the frozen four-worker topology")
    worker_path = _timing_worker_path(output_root, worker_index)
    ready_path = _timing_ready_path(output_root, worker_index)
    if worker_path.exists() or ready_path.exists():
        raise RuntimeError("timing worker artifacts already exist; use a fresh timing root")
    resolved = load_resolved(output_root, config)
    cache = verify_cache_ready(config, output_root, model_cache)
    cache_generation_timing = _resolve_cache_generation_timing(
        cache,
        output_root=output_root,
        model_cache=model_cache,
        provenance_path=cold_cache_provenance,
    )
    smoke = config["runtime"]["timing_smoke"]
    probe_seed = int(smoke["probe_seed"])
    if probe_seed in config["benchmark"]["probe_seeds"]:
        raise RuntimeError("timing smoke seed must remain outside the report seed set")
    runtime_resources = _timing_thread_quota(config)
    cpu_budget_path = output_root / "cpu_budget.json"
    if not cpu_budget_path.is_file():
        raise RuntimeError("timing worker requires the launcher CPU-budget artifact")
    cpu_budget = _validate_cpu_budget(config, read_json(cpu_budget_path), label="launcher")
    if {key: runtime_resources[key] for key in cpu_budget} != cpu_budget:
        raise RuntimeError("live runtime resources differ from the launcher CPU budget")

    from sae_probes.run_sae_evals import DATASET_SIZES
    from sae_probes.utils_data import get_xy_traintest
    selected = select_timing_smoke_tasks(
        config,
        {dataset: int(DATASET_SIZES[dataset]) for dataset in config["benchmark"]["datasets"]},
    )
    selection_digest = canonical_digest(
        [
            {"slot": item["slot"], "dataset": item["dataset"], "size": item["dataset_size"]}
            for item in selected
        ]
    )
    selected_slots = [
        {"slot": item["slot"], "quartile": int(item["quartile"])} for item in selected
    ]
    torch_device = torch.device(device)
    adapters = {
        method: load_adapter(config, checkpoint_dir, method, torch_device)
        for method in ("mse", "dpsae")
    }
    stable_dependency = stable_resolved_contract({"environment": dependency_environment})[
        "environment"
    ]
    frozen_dependency = stable_resolved_contract(
        {"environment": resolved.get("environment", {})}
    )["environment"]
    if stable_dependency != frozen_dependency:
        raise RuntimeError("timing worker dependency environment differs from the frozen run")
    ready = {
        "schema_version": 1,
        "complete": True,
        "worker_index": worker_index,
        "config_digest": resolved["config_digest"],
        "artifact_hashes": resolved["artifact_hashes"],
        "source_hashes_sha256": canonical_digest(resolved["source_hashes"]),
        "dependency_environment_sha256": canonical_digest(stable_dependency),
        "cache_ready_sha256": file_sha256(output_root / "cache_ready.json"),
        "cache_file_hashes_sha256": _cache_file_hashes_digest(cache),
        "selection_manifest_sha256": selection_digest,
        "selected_opaque_slots": selected_slots,
        "runtime_resources": runtime_resources,
        "cpu_budget_sha256": file_sha256(cpu_budget_path),
        "cache_generation_timing": cache_generation_timing,
        "topology": topology,
        "initialization_seconds": time.monotonic() - process_started_monotonic,
        "ready_monotonic_seconds": time.monotonic(),
    }
    atomic_json(ready_path, ready)
    common_sha256 = canonical_digest(_timing_common_identity(ready))
    start = _wait_for_timing_start(
        output_root,
        ready_path,
        common_sha256,
        float(smoke["barrier_timeout_seconds"]),
    )
    measurement_started = time.monotonic()
    cpu_stat_before = _read_cgroup_cpu_stat()

    method_slots = {"mse": "sparse_method_0", "dpsae": "sparse_method_1"}
    rows = []
    for item in selected:
        if torch_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch_device)
        task_started = time.perf_counter()
        dataset = item["dataset"]
        n_train = int(item["n_train"])

        def load_task() -> tuple[Any, Any, Any, Any]:
            return get_xy_traintest(
                n_train,
                dataset,
                config["model"]["hook_name"],
                model_name=config["model"]["transformer_lens_name"],
                model_cache_path=model_cache,
                seed=probe_seed,
            )

        sparse_stages: dict[str, dict[str, float]] = {}
        for method, adapter in adapters.items():
            stage: dict[str, float] = {}
            for pass_name in ("primary", "provenance"):
                loaded, load_seconds = _timed_call(torch_device, load_task)
                X_train, y_train, X_test, y_test = loaded
                codes, encode_seconds = _timed_call(
                    torch_device,
                    lambda: {
                        method: (
                            _encode_only(adapter, X_train, device=torch_device),
                            _encode_only(adapter, X_test, device=torch_device),
                        )
                    },
                )
                _, fit_seconds = _timed_call(
                    torch_device,
                    lambda: _fit_sparse_smoke(
                        codes=codes,
                        y_train=y_train,
                        y_test=y_test,
                        ks=config["benchmark"]["ks"],
                        probe_seed=probe_seed,
                    ),
                )
                stage[f"{pass_name}_data_load"] = load_seconds
                stage[f"{pass_name}_encode"] = encode_seconds
                stage[f"{pass_name}_l1"] = fit_seconds
                codes = None
                loaded = None
            stage["total"] = sum(stage.values())
            sparse_stages[method_slots[method]] = stage

        loaded, companion_data_seconds = _timed_call(torch_device, load_task)
        X_train, y_train, X_test, y_test = loaded
        companion_representations, companion_encode_seconds = _timed_call(
            torch_device,
            lambda: {
                method: (
                    *_representations(adapter, X_train, device=torch_device),
                    *_representations(adapter, X_test, device=torch_device),
                )
                for method, adapter in adapters.items()
            },
        )
        batched_representations: dict[str, tuple[Any, Any]] = {
            "original_residual": (X_train, X_test)
        }
        for method, (
            train_code,
            train_reconstruction,
            test_code,
            test_reconstruction,
        ) in companion_representations.items():
            batched_representations[f"{method}.full_code"] = (
                _full_code_csr(train_code),
                _full_code_csr(test_code),
            )
            batched_representations[f"{method}.reconstruction"] = (
                train_reconstruction,
                test_reconstruction,
            )
        _, batched_l2_seconds = _timed_call(
            torch_device,
            lambda: find_best_reg_l2_parallel_cold_C_batch(
                batched_representations,
                y_train,
                y_test,
                seed=probe_seed,
                n_jobs=int(config["runtime"]["companion_full_code_cold_C_jobs_per_worker"]),
            ),
        )
        companion_stage = {
            "data_load": companion_data_seconds,
            "encode_decode": companion_encode_seconds,
            "batched_l2": batched_l2_seconds,
        }
        companion_stage["total"] = sum(companion_stage.values())
        gpu_allocated = (
            int(torch.cuda.max_memory_allocated(torch_device)) if torch_device.type == "cuda" else 0
        )
        gpu_reserved = (
            int(torch.cuda.max_memory_reserved(torch_device)) if torch_device.type == "cuda" else 0
        )
        rows.append(
            {
                "slot": item["slot"],
                "quartile": item["quartile"],
                "n_train": len(X_train),
                "n_test": len(X_test),
                "stage_seconds": {**sparse_stages, "companion": companion_stage},
                "total_seconds": time.perf_counter() - task_started,
                "parent_peak_rss_mib": _parent_peak_rss_mib(),
                "peak_gpu_allocated_bytes": gpu_allocated,
                "peak_gpu_reserved_bytes": gpu_reserved,
            }
        )
        companion_representations = None
        loaded = None
        gc.collect()
        if torch_device.type == "cuda":
            torch.cuda.empty_cache()

    measurement_finished = time.monotonic()
    report = {
        **ready,
        "schema_version": 1,
        "complete": True,
        "barrier_start_sha256": file_sha256(output_root / "timing_barrier" / "start.json"),
        "barrier_start_monotonic_seconds": float(start["start_monotonic_seconds"]),
        "measurement_started_monotonic_seconds": measurement_started,
        "measurement_finished_monotonic_seconds": measurement_finished,
        "measurement_seconds": measurement_finished - measurement_started,
        "task_count": len(rows),
        "names_and_concept_results_suppressed": True,
        "saved_concept_metric_count": 0,
        "cgroup_cpu_stat_delta": _cgroup_cpu_stat_delta(
            cpu_stat_before, _read_cgroup_cpu_stat()
        ),
        "tasks": rows,
    }
    _validate_timing_worker_report(config, report, worker_index=worker_index)
    atomic_json(worker_path, report)
    return report


def _validate_timing_worker_report(
    config: Mapping[str, Any], value: Any, *, worker_index: int
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError("timing worker report is not an object")
    report = dict(value)
    if set(report) != _TIMING_WORKER_KEYS:
        raise RuntimeError("timing worker top-level schema or privacy boundary drift")
    required = {
        "schema_version": 1,
        "complete": True,
        "worker_index": worker_index,
        "config_digest": canonical_digest(config),
        "topology": _timing_topology(config),
        "task_count": int(config["runtime"]["timing_smoke"]["task_count"]),
        "names_and_concept_results_suppressed": True,
        "saved_concept_metric_count": 0,
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise RuntimeError(f"timing worker {worker_index} identity drift for {key}")
    _validate_reported_runtime_resources(config, report.get("runtime_resources"))
    _validate_timing_rows(config, report.get("tasks"))
    _validate_cgroup_cpu_stat_delta(report.get("cgroup_cpu_stat_delta"))
    for key in (
        "initialization_seconds",
        "ready_monotonic_seconds",
        "barrier_start_monotonic_seconds",
        "measurement_started_monotonic_seconds",
        "measurement_finished_monotonic_seconds",
        "measurement_seconds",
    ):
        number = float(report.get(key, math.nan))
        if not math.isfinite(number) or number <= 0:
            raise RuntimeError(f"timing worker {worker_index} has invalid {key}")
    if report["measurement_started_monotonic_seconds"] < report["barrier_start_monotonic_seconds"]:
        raise RuntimeError("timing worker started before the release barrier")
    if not math.isclose(
        report["measurement_finished_monotonic_seconds"]
        - report["measurement_started_monotonic_seconds"],
        report["measurement_seconds"],
        rel_tol=1e-9,
        abs_tol=1e-6,
    ):
        raise RuntimeError("timing worker measurement interval is inconsistent")
    return report


def _load_timing_fleet(
    config: Mapping[str, Any], output_root: Path
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    worker_count = int(config["runtime"]["timing_smoke"]["measured_worker_count"])
    workers = []
    exits = []
    for index in range(worker_count):
        worker_path = _timing_worker_path(output_root, index)
        if not worker_path.is_file():
            raise RuntimeError(f"missing timing worker report {index}")
        workers.append(
            _validate_timing_worker_report(config, read_json(worker_path), worker_index=index)
        )
        exit_path = _timing_exit_path(output_root, index)
        if not exit_path.is_file():
            raise RuntimeError(f"missing timing worker exit sentinel {index}")
        sentinel = read_json(exit_path)
        if (
            set(sentinel) != _TIMING_EXIT_KEYS
            or sentinel.get("schema_version") != 1
            or sentinel.get("complete") is not True
            or sentinel.get("worker_index") != index
            or sentinel.get("exit_code") != 0
            or sentinel.get("worker_report_sha256") != file_sha256(worker_path)
        ):
            raise RuntimeError(f"timing worker {index} did not exit successfully")
        exits.append(sentinel)
    common_sha256 = canonical_digest(_timing_common_identity(workers[0]))
    task_identity = [
        (row["slot"], row["quartile"], row["n_train"], row["n_test"])
        for row in workers[0]["tasks"]
    ]
    for worker in workers[1:]:
        if canonical_digest(_timing_common_identity(worker)) != common_sha256:
            raise RuntimeError("timing workers do not share one frozen identity")
        observed = [
            (row["slot"], row["quartile"], row["n_train"], row["n_test"])
            for row in worker["tasks"]
        ]
        if observed != task_identity:
            raise RuntimeError("timing workers did not measure the same opaque task set")
    start_path = output_root / "timing_barrier" / "start.json"
    if not start_path.is_file():
        raise RuntimeError("timing barrier start artifact is missing")
    start = read_json(start_path)
    refs = start.get("ready_reports")
    if (
        set(start) != _TIMING_START_KEYS
        or start.get("schema_version") != 1
        or start.get("complete") is not True
        or start.get("topology") != "four_worker_same_tasks"
        or start.get("worker_count") != worker_count
        or start.get("common_identity_sha256") != common_sha256
        or not isinstance(refs, list)
        or len(refs) != worker_count
    ):
        raise RuntimeError("timing barrier start artifact schema drift")
    expected_common = _expected_timing_common_identity(config, output_root, workers[0])
    if _timing_common_identity(workers[0]) != expected_common:
        raise RuntimeError("timing fleet identity differs from frozen and live artifacts")
    for index, ref in enumerate(refs):
        ready_path = _timing_ready_path(output_root, index)
        if (
            not isinstance(ref, Mapping)
            or set(ref) != {"worker_index", "path", "sha256"}
            or not ready_path.is_file()
            or ref.get("worker_index") != index
            or ref.get("path") != str(ready_path.relative_to(output_root))
            or ref.get("sha256") != file_sha256(ready_path)
            or set(read_json(ready_path)) != _TIMING_READY_KEYS
            or canonical_digest(_timing_common_identity(read_json(ready_path))) != common_sha256
        ):
            raise RuntimeError("timing barrier ready artifact hash or identity drift")
    start_sha256 = file_sha256(start_path)
    if any(worker.get("barrier_start_sha256") != start_sha256 for worker in workers):
        raise RuntimeError("timing worker barrier hash drift")
    starts = [float(worker["measurement_started_monotonic_seconds"]) for worker in workers]
    start_skew = max(starts) - min(starts)
    maximum_skew = float(config["runtime"]["timing_smoke"]["maximum_start_skew_seconds"])
    if start_skew > maximum_skew:
        raise RuntimeError("timing worker start skew exceeds the frozen maximum")
    return workers, start, exits


def _project_timing_smoke(
    config: Mapping[str, Any],
    worker_reports: Sequence[Mapping[str, Any]],
    *,
    cache_generation_seconds: float,
) -> dict[str, Any]:
    expected_workers = int(config["runtime"]["timing_smoke"]["measured_worker_count"])
    if len(worker_reports) != expected_workers:
        raise ValueError("timing projection rejects isolated or pooled measurements")
    smoke = config["runtime"]["timing_smoke"]
    method_slots = {"mse": "sparse_method_0", "dpsae": "sparse_method_1"}
    dataset_count = len(config["benchmark"]["datasets"])
    pair_units = int(smoke["projection_pair_units"])
    if pair_units != dataset_count * len(config["benchmark"]["probe_seeds"]):
        raise RuntimeError("timing projection units differ from the frozen dataset-seed grid")
    headroom = float(smoke["headroom_multiplier"])
    worker_projections = []
    for worker_index, report in enumerate(worker_reports):
        rows = _validate_timing_rows(config, report.get("tasks"))
        components = {
            slot: float(
                np.quantile(
                    np.asarray([row["stage_seconds"][slot]["total"] for row in rows]), 0.95
                )
            )
            for slot in (*method_slots.values(), "companion")
        }
        sparse_shard = config["runtime"]["sparse_worker_shards"][worker_index]
        sparse_slot = method_slots[sparse_shard["method"]]
        sparse_seed_count = len(sparse_shard["probe_seeds"])
        companion_seed_count = len(config["runtime"]["companion_seed_shards"][worker_index])
        unpadded = dataset_count * (
            sparse_seed_count * components[sparse_slot]
            + companion_seed_count * components["companion"]
        )
        worker_projections.append(
            {
                "worker_index": worker_index,
                "component_p95_seconds": components,
                "task_p95_seconds": float(
                    np.quantile(np.asarray([row["total_seconds"] for row in rows]), 0.95)
                ),
                "sparse_slot": sparse_slot,
                "sparse_seed_count": sparse_seed_count,
                "companion_seed_count": companion_seed_count,
                "p95_workload_seconds": unpadded,
                "p95_workload_seconds_with_headroom": unpadded * headroom,
            }
        )
    slowest = max(worker_projections, key=lambda item: item["p95_workload_seconds_with_headroom"])
    maximum_initialization = max(float(report["initialization_seconds"]) for report in worker_reports)
    workload = float(slowest["p95_workload_seconds_with_headroom"])
    return {
        "aggregation": "slowest_measured_worker",
        "headroom_multiplier": headroom,
        "dataset_count": dataset_count,
        "pair_units": pair_units,
        "worker_projections": worker_projections,
        "slowest_worker_index": int(slowest["worker_index"]),
        "maximum_initialization_seconds": maximum_initialization,
        "initialization_accounting": "maximum_pre_barrier_initialization_added_once",
        "cache_generation_seconds": float(cache_generation_seconds),
        "projected_workload_seconds_with_headroom": workload,
        "projected_pod_hours": (
            float(cache_generation_seconds) + maximum_initialization + workload
        )
        / 3600,
        "maximum_projected_pod_hours": float(smoke["maximum_projected_pod_hours"]),
    }


def _assemble_timing_smoke_report(
    config: Mapping[str, Any],
    *,
    resolved: Mapping[str, Any],
    worker_reports: Sequence[Mapping[str, Any]],
    worker_report_refs: Sequence[Mapping[str, Any]],
    worker_exit_refs: Sequence[Mapping[str, Any]],
    barrier: Mapping[str, Any],
) -> dict[str, Any]:
    smoke = config["runtime"]["timing_smoke"]
    cache_timing = dict(worker_reports[0]["cache_generation_timing"])
    projection = _project_timing_smoke(
        config,
        worker_reports,
        cache_generation_seconds=float(cache_timing["generation_seconds"]),
    )
    return {
        "schema_version": 7,
        "complete": True,
        "passed": projection["projected_pod_hours"]
        <= float(smoke["maximum_projected_pod_hours"]),
        "config_digest": resolved["config_digest"],
        "artifact_hashes": resolved["artifact_hashes"],
        "source_hashes_sha256": canonical_digest(resolved["source_hashes"]),
        "probe_seed": int(smoke["probe_seed"]),
        "task_count": int(smoke["task_count"]),
        "measured_task_count": len(worker_reports) * int(smoke["task_count"]),
        "measured_worker_count": len(worker_reports),
        "topology": _timing_topology(config),
        "selection_policy": smoke["selection"],
        "selection_manifest_sha256": worker_reports[0]["selection_manifest_sha256"],
        "names_and_concept_results_suppressed": True,
        "saved_concept_metric_count": 0,
        "companion_full_code_matrix_format": "scipy_csr_exact_values",
        "companion_l2_path_optimization": (
            "batched_all_representations_independent_cold_C_loky_cold_selected_C_refit"
        ),
        "companion_full_code_cold_C_jobs_per_worker": int(
            config["runtime"]["companion_full_code_cold_C_jobs_per_worker"]
        ),
        "runtime_resources": dict(worker_reports[0]["runtime_resources"]),
        "cpu_budget_sha256": worker_reports[0]["cpu_budget_sha256"],
        "cache_generation_timing": cache_timing,
        "timing_worker_reports": list(worker_report_refs),
        "timing_worker_exit_sentinels": list(worker_exit_refs),
        "barrier": dict(barrier),
        "cgroup_cpu_stat_deltas": [worker["cgroup_cpu_stat_delta"] for worker in worker_reports],
        "projection": projection,
    }


def finalize_timing_smoke(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    final_path = output_root / "timing_smoke.json"
    if final_path.exists():
        raise RuntimeError("authoritative timing report already exists; use a fresh timing root")
    resolved = load_resolved(output_root, config)
    workers, start, _exits = _load_timing_fleet(config, output_root)
    worker_refs = [
        {
            "worker_index": index,
            "path": str(_timing_worker_path(output_root, index).relative_to(output_root)),
            "sha256": file_sha256(_timing_worker_path(output_root, index)),
            "task_count": int(workers[index]["task_count"]),
        }
        for index in range(len(workers))
    ]
    exit_refs = [
        {
            "worker_index": index,
            "path": str(_timing_exit_path(output_root, index).relative_to(output_root)),
            "sha256": file_sha256(_timing_exit_path(output_root, index)),
            "exit_code": 0,
        }
        for index in range(len(workers))
    ]
    starts = [float(worker["measurement_started_monotonic_seconds"]) for worker in workers]
    barrier = {
        "synchronized": True,
        "start_path": "timing_barrier/start.json",
        "start_sha256": file_sha256(output_root / "timing_barrier" / "start.json"),
        "common_identity_sha256": start["common_identity_sha256"],
        "ready_reports": start["ready_reports"],
        "start_monotonic_seconds": start["start_monotonic_seconds"],
        "observed_start_skew_seconds": max(starts) - min(starts),
        "maximum_start_skew_seconds": float(
            config["runtime"]["timing_smoke"]["maximum_start_skew_seconds"]
        ),
    }
    report = _assemble_timing_smoke_report(
        config,
        resolved=resolved,
        worker_reports=workers,
        worker_report_refs=worker_refs,
        worker_exit_refs=exit_refs,
        barrier=barrier,
    )
    atomic_json(final_path, report)
    return report


def verify_timing_smoke_gate(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    path = output_root / "timing_smoke.json"
    if not path.is_file():
        raise RuntimeError("run the four-worker blind timing smoke before exp10 workers")
    report = read_json(path)
    smoke = config["runtime"]["timing_smoke"]
    if not isinstance(report, Mapping) or set(report) != _TIMING_REPORT_KEYS:
        raise RuntimeError("timing-smoke top-level schema or privacy boundary drift")
    required = {
        "schema_version": 7,
        "complete": True,
        "passed": True,
        "config_digest": canonical_digest(config),
        "probe_seed": int(smoke["probe_seed"]),
        "task_count": int(smoke["task_count"]),
        "measured_task_count": int(smoke["task_count"])
        * int(smoke["measured_worker_count"]),
        "measured_worker_count": int(smoke["measured_worker_count"]),
        "topology": _timing_topology(config),
        "names_and_concept_results_suppressed": True,
        "saved_concept_metric_count": 0,
        "companion_full_code_matrix_format": "scipy_csr_exact_values",
        "companion_l2_path_optimization": (
            "batched_all_representations_independent_cold_C_loky_cold_selected_C_refit"
        ),
        "companion_full_code_cold_C_jobs_per_worker": int(
            config["runtime"]["companion_full_code_cold_C_jobs_per_worker"]
        ),
    }
    for key, expected in required.items():
        if report.get(key) != expected:
            raise RuntimeError(
                f"timing-smoke gate failed for {key}: expected {expected!r}, "
                f"observed {report.get(key)!r}"
            )
    resolved = load_resolved(output_root, config)
    if report.get("source_hashes_sha256") != canonical_digest(resolved["source_hashes"]):
        raise RuntimeError("timing-smoke source hash identity drift")
    _validate_reported_runtime_resources(config, report.get("runtime_resources"))
    cpu_budget_path = output_root / "cpu_budget.json"
    if not cpu_budget_path.is_file():
        raise RuntimeError("timing-smoke CPU-budget artifact is missing")
    cpu_budget = _validate_cpu_budget(config, read_json(cpu_budget_path), label="launcher")
    if report.get("cpu_budget_sha256") != file_sha256(cpu_budget_path):
        raise RuntimeError("timing-smoke CPU-budget artifact hash drift")
    if any(report["runtime_resources"].get(key) != value for key, value in cpu_budget.items()):
        raise RuntimeError("timing-smoke resources differ from the CPU-budget artifact")
    workers, start, _exits = _load_timing_fleet(config, output_root)
    expected_worker_refs = [
        {
            "worker_index": index,
            "path": str(_timing_worker_path(output_root, index).relative_to(output_root)),
            "sha256": file_sha256(_timing_worker_path(output_root, index)),
            "task_count": int(smoke["task_count"]),
        }
        for index in range(len(workers))
    ]
    expected_exit_refs = [
        {
            "worker_index": index,
            "path": str(_timing_exit_path(output_root, index).relative_to(output_root)),
            "sha256": file_sha256(_timing_exit_path(output_root, index)),
            "exit_code": 0,
        }
        for index in range(len(workers))
    ]
    if report.get("timing_worker_reports") != expected_worker_refs:
        raise RuntimeError("timing-smoke worker report hashes drift")
    if report.get("timing_worker_exit_sentinels") != expected_exit_refs:
        raise RuntimeError("timing-smoke worker exit sentinel hashes drift")
    starts = [float(worker["measurement_started_monotonic_seconds"]) for worker in workers]
    expected_barrier = {
        "synchronized": True,
        "start_path": "timing_barrier/start.json",
        "start_sha256": file_sha256(output_root / "timing_barrier" / "start.json"),
        "common_identity_sha256": start["common_identity_sha256"],
        "ready_reports": start["ready_reports"],
        "start_monotonic_seconds": start["start_monotonic_seconds"],
        "observed_start_skew_seconds": max(starts) - min(starts),
        "maximum_start_skew_seconds": float(smoke["maximum_start_skew_seconds"]),
    }
    if (
        not isinstance(report.get("barrier"), Mapping)
        or set(report["barrier"]) != _TIMING_BARRIER_PROOF_KEYS
        or report.get("barrier") != expected_barrier
    ):
        raise RuntimeError("timing-smoke barrier proof drift")
    timing = report.get("cache_generation_timing", {})
    projection = _project_timing_smoke(
        config,
        workers,
        cache_generation_seconds=float(timing.get("generation_seconds", math.nan)),
    )
    if report.get("projection") != projection:
        raise RuntimeError("timing-smoke projection was not rebuilt from four worker reports")
    if projection.get("aggregation") != "slowest_measured_worker":
        raise RuntimeError("timing-smoke pooled projection is forbidden")
    if projection["projected_pod_hours"] > float(smoke["maximum_projected_pod_hours"]):
        raise RuntimeError("blind timing projection exceeds the frozen pod-hour limit")
    if report.get("cgroup_cpu_stat_deltas") != [
        worker["cgroup_cpu_stat_delta"] for worker in workers
    ]:
        raise RuntimeError("timing-smoke cgroup cpu.stat delta proof drift")
    ready_path = output_root / "cache_ready.json"
    ready = read_json(ready_path)
    generation_seconds = float(timing.get("generation_seconds", math.nan))
    source = timing.get("source")
    if source == "external_hash_bound_provenance":
        provenance_path = Path(str(timing.get("provenance_path", "")))
        if not provenance_path.is_file() or file_sha256(provenance_path) != timing.get(
            "provenance_sha256"
        ):
            raise RuntimeError("timing-smoke external cache provenance changed")
        provenance = read_json(provenance_path)
        for key in (
            "generation_seconds",
            "start_unix_seconds",
            "end_unix_seconds",
            "source_cache_ready_path",
            "source_cache_ready_sha256",
            "model_cache_path",
            "cache_file_hashes_sha256",
        ):
            if provenance.get(key) != timing.get(key):
                raise RuntimeError(f"timing-smoke cache provenance drift for {key}")
        expected_source_hash = ready.get("adopted_from_cache_ready_sha256") or file_sha256(
            ready_path
        )
        if timing.get("source_cache_ready_sha256") != expected_source_hash:
            raise RuntimeError("timing-smoke source cache manifest changed")
    elif source == "in_process_monotonic":
        if not math.isclose(
            float(ready.get("generation_seconds", math.nan)),
            generation_seconds,
            abs_tol=1e-6,
            rel_tol=0,
        ):
            raise RuntimeError("timing-smoke in-process cache duration changed")
    else:
        raise RuntimeError("timing-smoke cache timing source is invalid")
    return report


def run_worker(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    checkpoint_dir: Path,
    model_cache: Path,
    worker_index: int,
    cache_role: str,
    method: str,
    probe_seeds: Sequence[int],
    companion_seeds: Sequence[int],
    device: str,
    dependency_preflight: Mapping[str, Any],
    cache_wait_seconds: float = 21600,
) -> dict[str, Any]:
    """Run several seeds after one expensive sae-probes eager import."""

    seeds = [int(seed) for seed in probe_seeds]
    companions = [int(seed) for seed in companion_seeds]
    if not seeds or len(seeds) != len(set(seeds)):
        raise ValueError("worker probe seeds must be nonempty and unique")
    if len(companions) != len(set(companions)):
        raise ValueError("worker companion seeds must be unique")
    frozen_seeds = set(config["benchmark"]["probe_seeds"])
    if not set(seeds).issubset(frozen_seeds):
        raise ValueError("worker probe seeds must be a subset of the frozen seed list")
    if cache_role not in {"prepare", "wait"}:
        raise ValueError("cache role must be prepare or wait")
    if method not in {"mse", "dpsae"}:
        raise ValueError("worker method must be mse or dpsae")
    runtime = config["runtime"]
    if worker_index not in range(int(runtime["worker_count"])):
        raise ValueError("worker index is outside the frozen four-worker fleet")
    frozen_sparse = runtime["sparse_worker_shards"][worker_index]
    frozen_companions = [int(seed) for seed in runtime["companion_seed_shards"][worker_index]]
    if method != frozen_sparse["method"] or seeds != [
        int(seed) for seed in frozen_sparse["probe_seeds"]
    ]:
        raise ValueError("worker sparse assignment differs from the frozen runtime shard")
    if companions != frozen_companions:
        raise ValueError("worker companion assignment differs from the frozen 3/3/2/2 shard")

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
    runtime_resources = _timing_thread_quota(config)
    timing_smoke = verify_timing_smoke_gate(config, output_root)
    torch_device = torch.device(device)
    adapters = {
        adapter_method: load_adapter(config, checkpoint_dir, adapter_method, torch_device)
        for adapter_method in ("mse", "dpsae")
    }

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
                adapter=adapters[method],
            )
        )
    for seed in companions:
        companion_results.append(
            run_companion_job(
                config=config,
                output_root=output_root,
                checkpoint_dir=checkpoint_dir,
                model_cache=model_cache,
                probe_seed=seed,
                device=device,
                adapters=adapters,
            )
        )

    result = {
        "schema_version": 1,
        "complete": True,
        "config_digest": canonical_digest(config),
        "worker_index": worker_index,
        "cache_role": cache_role,
        "cache_manifest_sha256": canonical_digest(cache),
        "timing_smoke_sha256": file_sha256(output_root / "timing_smoke.json"),
        "timing_projected_pod_hours": timing_smoke["projection"]["projected_pod_hours"],
        "method": method,
        "probe_seeds": seeds,
        "companion_seeds": companions,
        "device": device,
        "runtime_resources": runtime_resources,
        "dependency_preflight": dict(dependency_preflight),
        "worker_seconds_excluding_dependency_preflight": time.monotonic() - started,
        "sparse_job_count": len(sparse_results),
        "companion_job_count": len(companion_results),
    }
    worker_name = f"worker_{worker_index}_{method}_{seeds[0]}_{seeds[-1]}"
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
                provenance_path = job / "provenance" / f"{dataset}.json"
                expected_hash = done.get("provenance_hashes", {}).get(dataset)
                if expected_hash != file_sha256(provenance_path):
                    raise RuntimeError(
                        f"sparse provenance changed for {method}, {dataset}, seed {seed}"
                    )
                record = read_json(provenance_path)
                predictions_path = job / "predictions" / f"{dataset}.pt"
                if not predictions_path.is_file() or record.get(
                    "heldout_predictions_sha256"
                ) != file_sha256(predictions_path):
                    raise RuntimeError(
                        f"sparse predictions changed for {method}, {dataset}, seed {seed}"
                    )
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
            record = read_json(path)
            weights_path = job / "weights" / f"{dataset}.pt"
            if not weights_path.is_file() or record.get("weights_sha256") != file_sha256(
                weights_path
            ):
                raise RuntimeError(f"companion weights changed for {dataset}, seed {seed}")
            records[(seed, dataset)] = record
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
                item for item in sparse[("dpsae", seed, dataset)]["rows"] if item["k"] == primary_k
            )
            mse_row = next(
                item for item in sparse[("mse", seed, dataset)]["rows"] if item["k"] == primary_k
            )
            seed_deltas[seed].append(
                float(dpsae_row["metrics"]["test_auc"]) - float(mse_row["metrics"]["test_auc"])
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
                        np.mean([row["methods"][method][representation][metric] for row in rows])
                    )
                    for metric in ("test_auc", "test_acc", "test_f1")
                }
        full_code_delta = (
            methods["dpsae"]["full_code"]["test_auc"] - methods["mse"]["full_code"]["test_auc"]
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
        "multiple_positive_families": positive_families >= int(stats["minimum_positive_families"]),
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
                    [
                        value["paired_full_code_auc_delta"]
                        for value in companion_task_metrics.values()
                    ]
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
                np.mean(
                    [value["excess_sparse_gain_auc"] for value in companion_task_metrics.values()]
                )
            ),
        },
        "candidate_manifest_sha256": file_sha256(output_root / "candidate_manifest.json"),
    }
    atomic_json(output_root / "advancement_report.json", report)
    return report


def _path(value: str) -> Path:
    return Path(value).expanduser()


def main() -> None:
    process_started_monotonic = time.monotonic()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=_path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=_path, default=DEFAULT_OUTPUT)
    parser.add_argument("--checkpoint-dir", type=_path)
    parser.add_argument("--saebench-root", type=_path)
    subparsers = parser.add_subparsers(dest="command", required=True)

    eligibility_parser = subparsers.add_parser("eligibility")
    eligibility_parser.add_argument("--device", default="cpu")

    subparsers.add_parser("freeze")

    cold_cache_parser = subparsers.add_parser("record-cold-cache-timing")
    cold_cache_parser.add_argument("--source-cache-ready", type=_path, required=True)
    cold_cache_parser.add_argument("--model-cache", type=_path, required=True)
    cold_cache_parser.add_argument("--start-unix-seconds", type=int, required=True)
    cold_cache_parser.add_argument("--end-unix-seconds", type=int, required=True)
    cold_cache_parser.add_argument("--expected-generation-seconds", type=int, required=True)
    cold_cache_parser.add_argument("--output", type=_path)

    cache_parser = subparsers.add_parser("prepare-cache")
    cache_parser.add_argument("--model-cache", type=_path, required=True)
    cache_parser.add_argument("--cold-cache-provenance", type=_path)
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

    timing_parser = subparsers.add_parser("timing-smoke")
    timing_parser.add_argument("--model-cache", type=_path, required=True)
    timing_parser.add_argument("--cold-cache-provenance", type=_path)
    timing_parser.add_argument("--device", default="cuda:0")
    timing_preflight_parser = subparsers.add_parser("timing-preflight")
    timing_preflight_parser.add_argument("--model-cache", type=_path, required=True)
    timing_preflight_parser.add_argument("--cold-cache-provenance", type=_path)
    timing_preflight_parser.add_argument("--device", default="cuda:0")
    timing_worker_parser = subparsers.add_parser("timing-worker")
    timing_worker_parser.add_argument("--model-cache", type=_path, required=True)
    timing_worker_parser.add_argument("--cold-cache-provenance", type=_path)
    timing_worker_parser.add_argument("--worker-index", type=int, required=True)
    timing_worker_parser.add_argument("--device", default="cuda:0")
    timing_barrier_parser = subparsers.add_parser("timing-start-barrier")
    timing_barrier_parser.add_argument("--timeout-seconds", type=float)
    subparsers.add_parser("timing-finalize")
    subparsers.add_parser("timing-gate")

    worker_parser = subparsers.add_parser("run-worker")
    worker_parser.add_argument("--model-cache", type=_path, required=True)
    worker_parser.add_argument("--worker-index", type=int, required=True)
    worker_parser.add_argument("--cache-role", choices=("prepare", "wait"), required=True)
    worker_parser.add_argument("--method", choices=("mse", "dpsae"), required=True)
    worker_parser.add_argument("--probe-seeds", type=int, nargs="+", required=True)
    worker_parser.add_argument("--companion-seeds", type=int, nargs="*", default=[])
    worker_parser.add_argument("--cache-wait-seconds", type=float, default=21600)
    worker_parser.add_argument("--device", default="cuda:0")

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--wait-seconds", type=float, default=0)

    args = parser.parse_args()
    config = load_config(args.config)
    checkpoint_dir = resolve_checkpoint_dir(config, args.checkpoint_dir)
    args.output_root = args.output_root.resolve()

    if args.command == "record-cold-cache-timing":
        output_path = (
            args.output.resolve()
            if args.output is not None
            else args.output_root / "cold_cache_timing_provenance.json"
        )
        print(
            json.dumps(
                record_cold_cache_timing_provenance(
                    config=config,
                    source_cache_ready=args.source_cache_ready.resolve(),
                    model_cache=args.model_cache.resolve(),
                    start_unix_seconds=args.start_unix_seconds,
                    end_unix_seconds=args.end_unix_seconds,
                    expected_generation_seconds=args.expected_generation_seconds,
                    output_path=output_path,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "eligibility":
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
                    cold_cache_provenance=args.cold_cache_provenance,
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
    elif args.command == "timing-smoke":
        parser.error(
            "isolated timing-smoke is forbidden; use four timing-worker processes and "
            "timing-start-barrier"
        )
    elif args.command == "timing-preflight":
        parser.error(
            "isolated timing-preflight is forbidden; use the four-worker timing launcher"
        )
    elif args.command == "timing-worker":
        if args.saebench_root is None:
            parser.error("timing-worker requires --saebench-root")
        dependency_environment = verify_saebench_environment(config, args.saebench_root)
        report = run_timing_worker(
            config=config,
            output_root=args.output_root,
            checkpoint_dir=checkpoint_dir,
            model_cache=args.model_cache.resolve(),
            worker_index=args.worker_index,
            device=args.device,
            dependency_environment=dependency_environment,
            process_started_monotonic=process_started_monotonic,
            cold_cache_provenance=args.cold_cache_provenance,
        )
        print(json.dumps(report, indent=2, sort_keys=True))
    elif args.command == "timing-start-barrier":
        print(
            json.dumps(
                start_timing_barrier(
                    config,
                    args.output_root,
                    timeout_seconds=args.timeout_seconds,
                ),
                indent=2,
                sort_keys=True,
            )
        )
    elif args.command == "timing-finalize":
        report = finalize_timing_smoke(config, args.output_root)
        print(json.dumps(report, indent=2, sort_keys=True))
        if not report["passed"]:
            raise SystemExit(2)
    elif args.command == "timing-gate":
        print(
            json.dumps(
                verify_timing_smoke_gate(config, args.output_root),
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
                    worker_index=args.worker_index,
                    cache_role=args.cache_role,
                    method=args.method,
                    probe_seeds=args.probe_seeds,
                    companion_seeds=args.companion_seeds,
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
