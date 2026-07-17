#!/usr/bin/env python3
"""Fail-closed integrity audit for exp10 concept-discovery artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch


METHODS = ("mse", "dpsae")
METRIC_KEYS = {"test_f1", "test_acc", "test_auc", "val_auc"}
TIMING_TASK_KEYS = {
    "slot",
    "quartile",
    "n_train",
    "n_test",
    "stage_seconds",
    "total_seconds",
    "peak_rss_mib",
    "peak_gpu_allocated_bytes",
    "peak_gpu_reserved_bytes",
}
TIMING_READY_KEYS = {
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
TIMING_WORKER_KEYS = TIMING_READY_KEYS | {
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
TIMING_REPORT_KEYS = {
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
TIMING_BARRIER_PROOF_KEYS = {
    "synchronized",
    "start_path",
    "start_sha256",
    "common_identity_sha256",
    "ready_reports",
    "start_monotonic_seconds",
    "observed_start_skew_seconds",
    "maximum_start_skew_seconds",
}


class AuditError(RuntimeError):
    """Raised when any expected artifact or cross-artifact invariant fails."""


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


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_json(path: Path, value: Any) -> None:
    _atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_jsonl(path: Path, values: Sequence[Mapping[str, Any]]) -> None:
    text = "".join(json.dumps(value, sort_keys=True, allow_nan=False) + "\n" for value in values)
    _atomic_text(path, text)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def _finite_number(value: Any, label: str, *, positive: bool = False) -> float:
    _require(
        not isinstance(value, bool) and isinstance(value, (int, float)), f"{label} is not numeric"
    )
    number = float(value)
    _require(math.isfinite(number), f"{label} is not finite")
    if positive:
        _require(number > 0, f"{label} is not positive")
    return number


def _validate_runtime_resources(
    config: Mapping[str, Any], value: Any, label: str
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} runtime resources are not an object")
    resources = dict(value)
    expected_keys = {
        "visible_cpu_count",
        "cgroup_quota_cores",
        "effective_cpu_count",
        "worker_count",
        "threads_per_worker",
        "environment",
    }
    _require(set(resources) == expected_keys, f"{label} runtime resource schema drift")
    frozen = config["runtime"]["resource_identity"]
    for key in ("cgroup_quota_cores", "effective_cpu_count", "threads_per_worker"):
        _require(resources.get(key) == frozen[key], f"{label} {key} drift")
    worker_count = int(config["runtime"]["worker_count"])
    _require(resources.get("worker_count") == worker_count, f"{label} worker count drift")
    visible = resources.get("visible_cpu_count")
    _require(
        isinstance(visible, int)
        and not isinstance(visible, bool)
        and visible >= int(frozen["effective_cpu_count"]),
        f"{label} visible CPU count drift",
    )
    expected_environment = {
        "LOKY_MAX_CPU_COUNT": str(frozen["effective_cpu_count"]),
        "OMP_NUM_THREADS": str(frozen["threads_per_worker"]),
        "MKL_NUM_THREADS": str(frozen["threads_per_worker"]),
        "OPENBLAS_NUM_THREADS": str(frozen["threads_per_worker"]),
        "NUMEXPR_NUM_THREADS": str(frozen["threads_per_worker"]),
    }
    _require(
        resources.get("environment") == expected_environment,
        f"{label} thread-cap environment drift",
    )
    return resources


def _validate_cpu_budget(
    config: Mapping[str, Any], value: Any, label: str
) -> dict[str, Any]:
    _require(isinstance(value, Mapping), f"{label} CPU budget is not an object")
    budget = dict(value)
    expected_keys = {
        "visible_cpu_count",
        "cgroup_quota_cores",
        "effective_cpu_count",
        "worker_count",
        "threads_per_worker",
    }
    _require(set(budget) == expected_keys, f"{label} CPU budget schema drift")
    frozen = config["runtime"]["resource_identity"]
    for key in ("cgroup_quota_cores", "effective_cpu_count", "threads_per_worker"):
        _require(budget.get(key) == frozen[key], f"{label} {key} drift")
    _require(
        budget.get("worker_count") == int(config["runtime"]["worker_count"]),
        f"{label} worker count drift",
    )
    visible = budget.get("visible_cpu_count")
    _require(
        isinstance(visible, int)
        and not isinstance(visible, bool)
        and visible >= int(frozen["effective_cpu_count"]),
        f"{label} visible CPU count drift",
    )
    return budget


def _timing_topology(config: Mapping[str, Any]) -> dict[str, Any]:
    runtime = config["runtime"]
    timing = runtime["timing_smoke"]
    return {
        "mode": timing["topology_mode"],
        "measured_worker_count": int(timing["measured_worker_count"]),
        "tasks_per_worker": int(timing["task_count"]),
        "same_task_set_per_worker": bool(timing["same_task_set_per_worker"]),
        "barrier_synchronized": True,
        "cold_c_jobs_per_worker": int(runtime["companion_full_code_cold_C_jobs_per_worker"]),
        "parent_threads_per_worker": int(runtime["resource_identity"]["threads_per_worker"]),
        "loky_inner_max_num_threads": 1,
    }


def _validate_cpu_stat_delta(value: Any, label: str) -> None:
    _require(
        isinstance(value, Mapping) and set(value) == {"path", "before", "after", "delta"},
        f"{label} cgroup cpu.stat schema drift",
    )
    before, after, delta = value["before"], value["after"], value["delta"]
    _require(
        all(isinstance(item, Mapping) for item in (before, after, delta))
        and set(before) == set(after) == set(delta),
        f"{label} cgroup cpu.stat counters drift",
    )
    _require(
        {"nr_periods", "nr_throttled"}.issubset(delta)
        and ("throttled_time" in delta or "throttled_usec" in delta),
        f"{label} cgroup cpu.stat throttling counters missing",
    )
    for key in delta:
        _require(
            all(isinstance(item[key], int) and not isinstance(item[key], bool) for item in (before, after, delta))
            and after[key] - before[key] == delta[key]
            and delta[key] >= 0,
            f"{label} cgroup cpu.stat delta invalid for {key}",
        )
    _require(delta["nr_periods"] > 0, f"{label} observed no cgroup quota periods")


def _validate_timing_rows(config: Mapping[str, Any], value: Any, label: str) -> list[Any]:
    count = int(config["runtime"]["timing_smoke"]["task_count"])
    _require(isinstance(value, list) and len(value) == count, f"{label} task count drift")
    identity = []
    for index, row in enumerate(value):
        _require(
            isinstance(row, Mapping) and set(row) == TIMING_TASK_KEYS,
            f"{label} task schema or privacy boundary drift",
        )
        slot = f"quartile_{index // 2}_sample_{index % 2}"
        _require(
            row.get("slot") == slot and row.get("quartile") == index // 2,
            f"{label} opaque slot order drift",
        )
        stages = row.get("stage_seconds")
        _require(
            isinstance(stages, Mapping)
            and set(stages) == {"sparse_method_0", "sparse_method_1", "companion"},
            f"{label} timing stage schema drift",
        )
        identity.append((row["slot"], row["quartile"], row["n_train"], row["n_test"]))
    return identity


def _read_json(path: Path) -> Any:
    _require(
        path.is_file() and not path.is_symlink(), f"missing or symlinked JSON artifact: {path}"
    )
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise AuditError(f"invalid JSON artifact {path}: {error}") from error


def _torch_load(path: Path) -> Any:
    _require(
        path.is_file() and not path.is_symlink(), f"missing or symlinked torch artifact: {path}"
    )
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise AuditError(f"invalid torch artifact {path}: {error}") from error


def _exact_files(directory: Path, names: set[str], label: str) -> dict[str, Path]:
    _require(
        directory.is_dir() and not directory.is_symlink(), f"missing {label} directory: {directory}"
    )
    observed = {path.name for path in directory.iterdir() if path.is_file()}
    _require(
        observed == names,
        f"{label} file set drift: expected={sorted(names)}, observed={sorted(observed)}",
    )
    return {name: directory / name for name in names}


def _exact_directories(directory: Path, names: set[str], label: str) -> None:
    _require(
        directory.is_dir() and not directory.is_symlink(), f"missing {label} directory: {directory}"
    )
    observed = {path.name for path in directory.iterdir() if path.is_dir()}
    _require(
        observed == names,
        f"{label} directory set drift: expected={sorted(names)}, observed={sorted(observed)}",
    )


def _validate_metrics(value: Any, label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} metrics are not an object")
    _require(set(value) == METRIC_KEYS, f"{label} metric schema drift: {sorted(value)}")
    for key, metric in value.items():
        _finite_number(metric, f"{label}.{key}")


def _validate_outputs(value: Any, count: int, label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} held-out outputs are not an object")
    _require(
        set(value) == {"decision_score", "prediction"}, f"{label} held-out output schema drift"
    )
    decision = torch.as_tensor(value["decision_score"])
    prediction = torch.as_tensor(value["prediction"])
    _require(decision.ndim == 1 and len(decision) == count, f"{label} decision-score shape drift")
    _require(prediction.ndim == 1 and len(prediction) == count, f"{label} prediction shape drift")
    _require(torch.isfinite(decision).all().item(), f"{label} decision scores are non-finite")


def _expected_split(
    config: Mapping[str, Any], dataset: str, seed: int, count: int
) -> dict[str, Any]:
    payload = {
        "dataset": dataset,
        "probe_seed": seed,
        "split": "test",
        "splitter": "sae_probes.utils_data.get_xy_traintest",
        "model_name": config["model"]["transformer_lens_name"],
        "hook_name": config["model"]["hook_name"],
    }
    split_id = f"exp10-test-{canonical_digest(payload)[:20]}"
    return {
        "split": "test",
        "split_id": split_id,
        "example_id_policy": "sha256_of_frozen_split_identity_plus_positional_index",
        "example_ids": [f"{split_id}-{index:05d}" for index in range(count)],
    }


def _validate_split(value: Mapping[str, Any], expected: Mapping[str, Any], label: str) -> None:
    for key, wanted in expected.items():
        _require(value.get(key) == wanted, f"{label} held-out split mismatch for {key}")


def _validate_classifier(value: Any, width: int, label: str) -> None:
    _require(isinstance(value, Mapping), f"{label} classifier is not an object")
    _require(
        set(value) == {"coefficient", "intercept", "classes", "C"},
        f"{label} classifier schema drift",
    )
    coefficient = torch.as_tensor(value["coefficient"])
    intercept = torch.as_tensor(value["intercept"])
    classes = torch.as_tensor(value["classes"])
    _require(tuple(coefficient.shape) == (1, width), f"{label} coefficient shape drift")
    _require(tuple(intercept.shape) == (1,), f"{label} intercept shape drift")
    _require(tuple(classes.shape) == (2,), f"{label} classes shape drift")
    _require(torch.isfinite(coefficient).all().item(), f"{label} coefficients are non-finite")
    _require(torch.isfinite(intercept).all().item(), f"{label} intercept is non-finite")
    _finite_number(value["C"], f"{label}.C", positive=True)


class ArtifactAuditor:
    def __init__(self, config: Mapping[str, Any], config_path: Path, output_root: Path) -> None:
        self.config = config
        self.config_path = config_path.resolve()
        self.output_root = output_root.resolve()
        self.datasets = [str(value) for value in config["benchmark"]["datasets"]]
        self.seeds = [int(value) for value in config["benchmark"]["probe_seeds"]]
        self.ks = [int(value) for value in config["benchmark"]["ks"]]
        self.checkpoint = str(config["pilot_checkpoint"]["checkpoint_id"])
        self.config_digest = canonical_digest(config)
        self.records: dict[str, dict[str, Any]] = {}
        self.counts: Counter[str] = Counter()
        self.split_digests: dict[tuple[int, str], str] = {}
        self.resolved: Mapping[str, Any] = {}

        _require(
            self.datasets and len(self.datasets) == len(set(self.datasets)),
            "datasets are not unique",
        )
        _require(
            self.seeds and len(self.seeds) == len(set(self.seeds)), "probe seeds are not unique"
        )
        _require(self.ks and len(self.ks) == len(set(self.ks)), "k values are not unique")
        expected_manifest = config["benchmark"].get("dataset_manifest_sha256")
        _require(
            expected_manifest == canonical_digest(self.datasets), "dataset manifest digest drift"
        )
        _require(
            set(config["benchmark"]["family_by_dataset"]) == set(self.datasets), "family map drift"
        )

    def record(self, path: Path, kind: str, **identity: Any) -> str:
        path = path.resolve()
        _require(path.is_file() and not path.is_symlink(), f"missing or symlinked artifact: {path}")
        try:
            display = str(path.relative_to(self.output_root))
        except ValueError:
            display = str(path)
        _require(display not in self.records, f"artifact was audited twice: {display}")
        digest = file_sha256(path)
        self.records[display] = {
            "path": display,
            "kind": kind,
            "bytes": path.stat().st_size,
            "sha256": digest,
            **identity,
        }
        self.counts[kind] += 1
        return digest

    def audit_base(self) -> None:
        resolved_path = self.output_root / "resolved_config.json"
        resolved = _read_json(resolved_path)
        _require(resolved.get("schema_version") == 1, "resolved-config schema drift")
        _require(
            resolved.get("config_digest") == self.config_digest, "resolved-config digest drift"
        )
        if "config_sha256" in resolved:
            _require(
                resolved["config_sha256"] == file_sha256(self.config_path), "config file hash drift"
            )
        self.resolved = resolved
        self.record(resolved_path, "resolved_config")

        eligibility_path = self.output_root / "eligibility.json"
        eligibility = _read_json(eligibility_path)
        _require(
            eligibility.get("schema_version") == 1 and eligibility.get("passed") is True,
            "eligibility gate did not pass",
        )
        _require(
            eligibility.get("artifact_hashes") == resolved.get("artifact_hashes"),
            "eligibility artifact hashes drift",
        )
        self.record(eligibility_path, "eligibility")

        ready_path = self.output_root / "cache_ready.json"
        ready = _read_json(ready_path)
        _require(
            ready.get("schema_version") == 1 and ready.get("complete") is True,
            "cache-ready schema drift",
        )
        _require(
            ready.get("config_digest") == self.config_digest, "cache-ready config digest drift"
        )
        _require(
            ready.get("dataset_manifest_sha256")
            == self.config["benchmark"]["dataset_manifest_sha256"],
            "cache-ready dataset digest drift",
        )
        _require(set(ready.get("files", {})) == set(self.datasets), "cache-ready dataset set drift")
        for dataset in self.datasets:
            metadata = ready["files"][dataset]
            _require(isinstance(metadata, Mapping), f"cache metadata is invalid for {dataset}")
            cache_path = Path(metadata.get("path", ""))
            _require(cache_path.is_absolute(), f"cache path is not absolute for {dataset}")
            _require(
                cache_path.stat().st_size == metadata.get("bytes"),
                f"cache byte count drift for {dataset}",
            )
            digest = self.record(cache_path, "cache_activation", dataset=dataset)
            _require(digest == metadata.get("sha256"), f"cache hash drift for {dataset}")
            tensor = torch.as_tensor(_torch_load(cache_path))
            wanted_shape = metadata.get("shape")
            _require(
                list(tensor.shape) == wanted_shape, f"cache shape manifest drift for {dataset}"
            )
            _require(
                tensor.ndim == 2 and tensor.shape[1] == int(self.config["model"]["d_model"]),
                f"cache activation shape drift for {dataset}",
            )
        self.record(ready_path, "cache_ready")

        cpu_budget_path = self.output_root / "cpu_budget.json"
        cpu_budget = _validate_cpu_budget(
            self.config, _read_json(cpu_budget_path), "launcher"
        )
        cpu_budget_hash = self.record(cpu_budget_path, "cpu_budget")

        runtime = self.config.get("runtime", {})
        timing = runtime.get("timing_smoke") if isinstance(runtime, Mapping) else None
        if isinstance(timing, Mapping) and timing.get("require_passed_report_before_workers"):
            timing_path = self.output_root / "timing_smoke.json"
            report = _read_json(timing_path)
            _require(
                isinstance(report, Mapping)
                and set(report) == TIMING_REPORT_KEYS
                and report.get("schema_version") == 6
                and report.get("complete") is True
                and report.get("passed") is True,
                "timing-smoke gate did not pass",
            )
            _require(
                report.get("companion_full_code_matrix_format")
                == "scipy_csr_exact_values",
                "timing-smoke full-code matrix format drift",
            )
            _require(
                report.get("companion_l2_path_optimization")
                == "parallel_independent_cold_C_loky_cold_selected_C_refit",
                "timing-smoke companion L2 path optimization drift",
            )
            _require(
                report.get("companion_full_code_cold_C_jobs_per_worker")
                == int(self.config["runtime"]["companion_full_code_cold_C_jobs_per_worker"]),
                "timing-smoke parallel cold-C job count drift",
            )
            _require(
                report.get("config_digest") == self.config_digest,
                "timing-smoke config digest drift",
            )
            _require(
                report.get("probe_seed") == int(timing["probe_seed"]), "timing-smoke seed drift"
            )
            _require(
                report.get("task_count") == int(timing["task_count"]),
                "timing-smoke task count drift",
            )
            _require(
                report.get("saved_concept_metric_count") == 0,
                "timing smoke retained concept metrics",
            )
            _validate_runtime_resources(
                self.config, report.get("runtime_resources"), "timing-smoke"
            )
            _require(
                report.get("cpu_budget_sha256") == cpu_budget_hash,
                "timing-smoke CPU-budget hash drift",
            )
            _require(
                all(
                    report["runtime_resources"].get(key) == value
                    for key, value in cpu_budget.items()
                ),
                "timing-smoke resources differ from launcher CPU budget",
            )
            topology = _timing_topology(self.config)
            worker_count = int(timing["measured_worker_count"])
            task_count = int(timing["task_count"])
            _require(report.get("topology") == topology, "timing-smoke topology drift")
            _require(
                report.get("measured_worker_count") == worker_count
                and report.get("measured_task_count") == worker_count * task_count,
                "timing-smoke measured fleet cardinality drift",
            )
            worker_refs = report.get("timing_worker_reports")
            exit_refs = report.get("timing_worker_exit_sentinels")
            _require(
                isinstance(worker_refs, list)
                and isinstance(exit_refs, list)
                and len(worker_refs) == len(exit_refs) == worker_count,
                "timing-smoke worker proof cardinality drift",
            )
            workers = []
            task_identity = None
            common_identity = None
            common_keys = (
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
            for index, ref in enumerate(worker_refs):
                _require(
                    isinstance(ref, Mapping)
                    and set(ref) == {"worker_index", "path", "sha256", "task_count"}
                    and ref.get("worker_index") == index
                    and ref.get("task_count") == task_count,
                    f"timing worker reference {index} drift",
                )
                relative = Path(str(ref["path"]))
                _require(not relative.is_absolute(), f"timing worker path {index} is absolute")
                worker_path = (self.output_root / relative).resolve()
                _require(
                    self.output_root.resolve() in worker_path.parents,
                    f"timing worker path {index} escapes the output root",
                )
                digest = self.record(worker_path, "timing_worker_report", worker_index=index)
                _require(digest == ref.get("sha256"), f"timing worker hash {index} drift")
                worker = _read_json(worker_path)
                _require(
                    isinstance(worker, Mapping)
                    and set(worker) == TIMING_WORKER_KEYS
                    and worker.get("schema_version") == 1
                    and worker.get("complete") is True
                    and worker.get("worker_index") == index
                    and worker.get("config_digest") == self.config_digest
                    and worker.get("topology") == topology
                    and worker.get("task_count") == task_count
                    and worker.get("names_and_concept_results_suppressed") is True
                    and worker.get("saved_concept_metric_count") == 0,
                    f"timing worker {index} schema or privacy drift",
                )
                _validate_runtime_resources(
                    self.config, worker.get("runtime_resources"), f"timing worker {index}"
                )
                _require(
                    worker.get("cpu_budget_sha256") == cpu_budget_hash,
                    f"timing worker {index} CPU-budget hash drift",
                )
                observed_identity = {key: worker.get(key) for key in common_keys}
                if common_identity is None:
                    common_identity = observed_identity
                _require(
                    observed_identity == common_identity,
                    f"timing worker {index} common identity drift",
                )
                observed_tasks = _validate_timing_rows(
                    self.config, worker.get("tasks"), f"timing worker {index}"
                )
                if task_identity is None:
                    task_identity = observed_tasks
                _require(
                    observed_tasks == task_identity,
                    f"timing worker {index} did not measure the same opaque tasks",
                )
                _validate_cpu_stat_delta(
                    worker.get("cgroup_cpu_stat_delta"), f"timing worker {index}"
                )
                workers.append(worker)

                exit_ref = exit_refs[index]
                _require(
                    isinstance(exit_ref, Mapping)
                    and set(exit_ref) == {"worker_index", "path", "sha256", "exit_code"}
                    and exit_ref.get("worker_index") == index
                    and exit_ref.get("exit_code") == 0,
                    f"timing worker exit reference {index} drift",
                )
                exit_relative = Path(str(exit_ref["path"]))
                _require(not exit_relative.is_absolute(), f"timing exit path {index} is absolute")
                exit_path = (self.output_root / exit_relative).resolve()
                _require(
                    self.output_root.resolve() in exit_path.parents,
                    f"timing exit path {index} escapes the output root",
                )
                exit_hash = self.record(
                    exit_path, "timing_worker_exit", worker_index=index
                )
                _require(exit_hash == exit_ref.get("sha256"), f"timing exit hash {index} drift")
                sentinel = _read_json(exit_path)
                _require(
                    set(sentinel)
                    == {
                        "schema_version",
                        "complete",
                        "worker_index",
                        "exit_code",
                        "worker_report_sha256",
                    }
                    and sentinel.get("schema_version") == 1
                    and sentinel.get("complete") is True
                    and sentinel.get("worker_index") == index
                    and sentinel.get("exit_code") == 0
                    and sentinel.get("worker_report_sha256") == digest,
                    f"timing worker exit sentinel {index} is not bound to its report",
                )

            barrier = report.get("barrier")
            _require(
                isinstance(barrier, Mapping)
                and set(barrier) == TIMING_BARRIER_PROOF_KEYS
                and barrier.get("synchronized") is True
                and barrier.get("start_path") == "timing_barrier/start.json"
                and barrier.get("maximum_start_skew_seconds")
                == float(timing["maximum_start_skew_seconds"]),
                "timing barrier proof drift",
            )
            start_path = self.output_root / "timing_barrier/start.json"
            start_hash = self.record(start_path, "timing_barrier_start")
            _require(start_hash == barrier.get("start_sha256"), "timing barrier hash drift")
            start = _read_json(start_path)
            _require(
                set(start)
                == {
                    "schema_version",
                    "complete",
                    "topology",
                    "worker_count",
                    "common_identity_sha256",
                    "start_monotonic_seconds",
                    "start_unix_seconds",
                    "ready_reports",
                }
                and start.get("schema_version") == 1
                and start.get("complete") is True
                and start.get("topology") == "four_worker_same_tasks"
                and start.get("worker_count") == worker_count
                and start.get("common_identity_sha256") == canonical_digest(common_identity),
                "timing barrier start schema or identity drift",
            )
            ready_refs = start.get("ready_reports")
            _require(
                isinstance(ready_refs, list) and len(ready_refs) == worker_count,
                "timing ready proof cardinality drift",
            )
            for index, ref in enumerate(ready_refs):
                _require(
                    isinstance(ref, Mapping)
                    and set(ref) == {"worker_index", "path", "sha256"}
                    and ref.get("worker_index") == index,
                    f"timing ready reference {index} drift",
                )
                ready_artifact_path = (self.output_root / str(ref["path"])).resolve()
                _require(
                    self.output_root.resolve() in ready_artifact_path.parents,
                    f"timing ready path {index} escapes the output root",
                )
                ready_hash = self.record(
                    ready_artifact_path, "timing_barrier_ready", worker_index=index
                )
                _require(ready_hash == ref.get("sha256"), f"timing ready hash {index} drift")
                ready_artifact = _read_json(ready_artifact_path)
                _require(
                    set(ready_artifact) == TIMING_READY_KEYS
                    and ready_artifact.get("worker_index") == index
                    and {key: ready_artifact.get(key) for key in common_keys}
                    == common_identity,
                    f"timing ready artifact {index} identity drift",
                )
            starts = [float(worker["measurement_started_monotonic_seconds"]) for worker in workers]
            observed_skew = max(starts) - min(starts)
            _require(
                math.isclose(
                    float(barrier.get("observed_start_skew_seconds", math.nan)),
                    observed_skew,
                    abs_tol=1e-6,
                    rel_tol=1e-9,
                )
                and observed_skew <= float(timing["maximum_start_skew_seconds"]),
                "timing barrier start skew drift",
            )
            projection = report.get("projection")
            _require(
                isinstance(projection, Mapping)
                and projection.get("aggregation") == "slowest_measured_worker"
                and projection.get("initialization_accounting")
                == "maximum_pre_barrier_initialization_added_once"
                and math.isclose(
                    float(projection.get("maximum_initialization_seconds", math.nan)),
                    max(float(worker["initialization_seconds"]) for worker in workers),
                    abs_tol=1e-6,
                    rel_tol=1e-9,
                ),
                "timing slowest-worker projection or initialization accounting drift",
            )
            _require(
                report.get("cgroup_cpu_stat_deltas")
                == [worker["cgroup_cpu_stat_delta"] for worker in workers],
                "timing cgroup cpu.stat proof drift",
            )
            self.record(timing_path, "timing_smoke")

    def _validate_sparse_prediction(
        self, path: Path, method: str, seed: int, dataset: str, provenance: Mapping[str, Any]
    ) -> None:
        value = _torch_load(path)
        _require(
            isinstance(value, Mapping) and value.get("schema_version") == 1,
            f"sparse prediction schema drift for {method}/{seed}/{dataset}",
        )
        identity = {
            "config_digest": self.config_digest,
            "checkpoint_id": self.checkpoint,
            "method": method,
            "probe_seed": seed,
            "dataset": dataset,
        }
        for key, wanted in identity.items():
            _require(
                value.get(key) == wanted,
                f"sparse prediction {key} drift for {method}/{seed}/{dataset}",
            )
        label = torch.as_tensor(value.get("label"))
        _require(
            label.ndim == 1, f"sparse labels are not one-dimensional for {method}/{seed}/{dataset}"
        )
        count = len(label)
        expected = _expected_split(self.config, dataset, seed, count)
        _validate_split(value, expected, f"sparse {method}/{seed}/{dataset}")
        _require(
            provenance.get("heldout_split_id") == expected["split_id"],
            f"sparse provenance split drift for {method}/{seed}/{dataset}",
        )
        _require(
            provenance.get("heldout_example_count") == count,
            f"sparse provenance count drift for {method}/{seed}/{dataset}",
        )
        _require(
            provenance.get("heldout_example_id_policy") == expected["example_id_policy"],
            f"sparse provenance ID policy drift for {method}/{seed}/{dataset}",
        )
        by_k = value.get("by_k")
        _require(
            isinstance(by_k, Mapping) and set(by_k) == {str(k) for k in self.ks},
            f"sparse prediction k set drift for {method}/{seed}/{dataset}",
        )
        for k in self.ks:
            _validate_outputs(by_k[str(k)], count, f"sparse {method}/{seed}/{dataset}/k={k}")
        split_digest = canonical_digest({"split": expected, "label": label.tolist()})
        key = (seed, dataset)
        prior = self.split_digests.setdefault(key, split_digest)
        _require(
            prior == split_digest,
            f"held-out label/split mismatch across methods for seed={seed}, dataset={dataset}",
        )

    def audit_sparse(self) -> None:
        root = self.output_root / "jobs" / self.checkpoint
        _exact_directories(root, set(METHODS), "sparse method")
        seed_names = {f"seed_{seed}" for seed in self.seeds}
        for method in METHODS:
            method_root = root / method
            _exact_directories(method_root, seed_names, f"sparse {method} seed")
            for seed in self.seeds:
                job = method_root / f"seed_{seed}"
                done_path = job / "done.json"
                done = _read_json(done_path)
                _require(
                    done.get("schema_version") == 1 and done.get("complete") is True,
                    f"incomplete sparse job {method}/{seed}",
                )
                required = {
                    "config_digest": self.config_digest,
                    "method": method,
                    "probe_seed": seed,
                    "dataset_count": len(self.datasets),
                    "dataset_manifest_sha256": self.config["benchmark"]["dataset_manifest_sha256"],
                    "artifact_hashes": self.resolved.get("artifact_hashes"),
                }
                for key, wanted in required.items():
                    _require(
                        done.get(key) == wanted, f"sparse done {key} drift for {method}/{seed}"
                    )
                _require(
                    set(done.get("provenance_hashes", {})) == set(self.datasets),
                    f"sparse done provenance set drift for {method}/{seed}",
                )

                provenance_files = _exact_files(
                    job / "provenance",
                    {f"{dataset}.json" for dataset in self.datasets},
                    f"sparse provenance {method}/{seed}",
                )
                prediction_files = _exact_files(
                    job / "predictions",
                    {f"{dataset}.pt" for dataset in self.datasets},
                    f"sparse predictions {method}/{seed}",
                )
                raw_paths = sorted((job / "raw").glob("**/*.json"))
                _require(
                    len(raw_paths) == len(self.datasets),
                    f"raw sparse result count drift for {method}/{seed}",
                )
                raw_by_dataset: dict[str, tuple[Path, list[Any]]] = {}
                for raw_path in raw_paths:
                    entries = _read_json(raw_path)
                    _require(
                        isinstance(entries, list) and entries,
                        f"raw sparse result is not a nonempty list: {raw_path}",
                    )
                    dataset_values = {
                        entry.get("dataset") for entry in entries if isinstance(entry, Mapping)
                    }
                    _require(
                        len(dataset_values) == 1, f"raw sparse dataset identity drift: {raw_path}"
                    )
                    dataset = str(next(iter(dataset_values)))
                    _require(
                        dataset in self.datasets and dataset not in raw_by_dataset,
                        f"raw sparse dataset set drift for {method}/{seed}: {dataset}",
                    )
                    raw_by_dataset[dataset] = (raw_path, entries)
                _require(
                    set(raw_by_dataset) == set(self.datasets),
                    f"raw sparse dataset coverage drift for {method}/{seed}",
                )

                aggregate_paths = sorted((job / "saebench_output").glob("*_eval_results.json"))
                _require(
                    len(aggregate_paths) == 1, f"SAEBench aggregate count drift for {method}/{seed}"
                )
                aggregate_hash = self.record(
                    aggregate_paths[0], "saebench_result", method=method, probe_seed=seed
                )
                _require(
                    aggregate_hash == done.get("saebench_result_sha256"),
                    f"SAEBench aggregate hash drift for {method}/{seed}",
                )

                for dataset in self.datasets:
                    raw_path, raw_entries = raw_by_dataset[dataset]
                    raw_hash = self.record(
                        raw_path, "sparse_raw", method=method, probe_seed=seed, dataset=dataset
                    )
                    by_k = {
                        int(entry["k"]): entry
                        for entry in raw_entries
                        if isinstance(entry, Mapping) and "k" in entry
                    }
                    _require(
                        set(by_k) == set(self.ks) and len(raw_entries) == len(self.ks),
                        f"raw sparse k set drift for {method}/{seed}/{dataset}",
                    )

                    provenance_path = provenance_files[f"{dataset}.json"]
                    provenance = _read_json(provenance_path)
                    _require(
                        provenance.get("schema_version") == 2,
                        f"provenance schema drift for {method}/{seed}/{dataset}",
                    )
                    identity = {
                        "config_digest": self.config_digest,
                        "checkpoint_id": self.checkpoint,
                        "method": method,
                        "probe_seed": seed,
                        "dataset": dataset,
                        "family": self.config["benchmark"]["family_by_dataset"][dataset],
                    }
                    for key, wanted in identity.items():
                        _require(
                            provenance.get(key) == wanted,
                            f"provenance {key} drift for {method}/{seed}/{dataset}",
                        )
                    _require(
                        provenance.get("raw_result_sha256") == raw_hash,
                        f"provenance raw hash drift for {method}/{seed}/{dataset}",
                    )
                    rows = provenance.get("rows")
                    _require(
                        isinstance(rows, list) and len(rows) == len(self.ks),
                        f"provenance row count drift for {method}/{seed}/{dataset}",
                    )
                    provenance_by_k = {
                        int(row["k"]): row
                        for row in rows
                        if isinstance(row, Mapping) and "k" in row
                    }
                    _require(
                        set(provenance_by_k) == set(self.ks),
                        f"provenance k set drift for {method}/{seed}/{dataset}",
                    )
                    for k in self.ks:
                        row = provenance_by_k[k]
                        raw = by_k[k]
                        _validate_metrics(
                            row.get("metrics"), f"provenance {method}/{seed}/{dataset}/k={k}"
                        )
                        feature_ids = row.get("feature_ids")
                        _require(
                            isinstance(feature_ids, list)
                            and len(feature_ids) == k
                            and len(set(feature_ids)) == k
                            and all(
                                isinstance(item, int) and not isinstance(item, bool)
                                for item in feature_ids
                            ),
                            f"feature IDs drift for {method}/{seed}/{dataset}/k={k}",
                        )
                        _require(
                            raw.get("indices") == feature_ids,
                            f"raw/provenance feature alignment drift for {method}/{seed}/{dataset}/k={k}",
                        )
                        _require(
                            raw.get("dataset") == dataset
                            and raw.get("hook_name") == self.config["model"]["hook_name"]
                            and raw.get("reg_type") == "l1"
                            and raw.get("binarize") is False,
                            f"raw sparse schema drift for {method}/{seed}/{dataset}/k={k}",
                        )
                        for metric in METRIC_KEYS:
                            _require(
                                float(raw.get(metric)) == float(row["metrics"][metric]),
                                f"raw/provenance metric drift for {method}/{seed}/{dataset}/k={k}/{metric}",
                            )
                        weights = row.get("feature_weights")
                        _require(
                            isinstance(weights, list)
                            and [item.get("feature_id") for item in weights] == feature_ids,
                            f"feature-weight IDs drift for {method}/{seed}/{dataset}/k={k}",
                        )
                        for item in weights:
                            _finite_number(
                                item.get("weight"),
                                f"feature weight {method}/{seed}/{dataset}/k={k}",
                            )
                        _finite_number(
                            row.get("intercept"), f"intercept {method}/{seed}/{dataset}/k={k}"
                        )
                        _finite_number(
                            row.get("regularization_C"),
                            f"C {method}/{seed}/{dataset}/k={k}",
                            positive=True,
                        )

                    prediction_path = prediction_files[f"{dataset}.pt"]
                    prediction_hash = self.record(
                        prediction_path,
                        "sparse_predictions",
                        method=method,
                        probe_seed=seed,
                        dataset=dataset,
                    )
                    _require(
                        provenance.get("heldout_predictions_sha256") == prediction_hash,
                        f"provenance prediction hash drift for {method}/{seed}/{dataset}",
                    )
                    self._validate_sparse_prediction(
                        prediction_path, method, seed, dataset, provenance
                    )
                    provenance_hash = self.record(
                        provenance_path,
                        "sparse_provenance",
                        method=method,
                        probe_seed=seed,
                        dataset=dataset,
                    )
                    _require(
                        done["provenance_hashes"].get(dataset) == provenance_hash,
                        f"done/provenance hash drift for {method}/{seed}/{dataset}",
                    )
                self.record(done_path, "sparse_done", method=method, probe_seed=seed)

    def _validate_companion_weights(self, value: Any, seed: int, dataset: str, count: int) -> None:
        _require(
            isinstance(value, Mapping) and value.get("schema_version") == 2,
            f"companion weight schema drift for {seed}/{dataset}",
        )
        identity = {
            "config_digest": self.config_digest,
            "checkpoint_id": self.checkpoint,
            "probe_seed": seed,
            "dataset": dataset,
        }
        for key, wanted in identity.items():
            _require(value.get(key) == wanted, f"companion weight {key} drift for {seed}/{dataset}")
        heldout = value.get("heldout")
        _require(
            isinstance(heldout, Mapping), f"companion heldout block missing for {seed}/{dataset}"
        )
        label = torch.as_tensor(heldout.get("label"))
        _require(
            label.ndim == 1 and len(label) == count,
            f"companion label shape drift for {seed}/{dataset}",
        )
        expected = _expected_split(self.config, dataset, seed, count)
        _validate_split(heldout, expected, f"companion {seed}/{dataset}")
        split_digest = canonical_digest({"split": expected, "label": label.tolist()})
        _require(
            self.split_digests.get((seed, dataset)) == split_digest,
            f"held-out split/labels do not align across sparse and companion artifacts for seed={seed}, dataset={dataset}",
        )
        _validate_outputs(
            heldout.get("original_residual"), count, f"companion original {seed}/{dataset}"
        )
        heldout_methods = heldout.get("methods")
        _require(
            isinstance(heldout_methods, Mapping) and set(heldout_methods) == set(METHODS),
            f"companion heldout method set drift for {seed}/{dataset}",
        )
        d_model = int(self.config["model"]["d_model"])
        d_sae = int(self.config["pilot_checkpoint"]["dictionary_size"])
        _validate_classifier(
            value.get("original_residual"), d_model, f"companion original {seed}/{dataset}"
        )
        for method in METHODS:
            method_state = value.get(method)
            _require(
                isinstance(method_state, Mapping)
                and set(method_state) == {"full_code", "reconstruction"},
                f"companion classifier representation set drift for {method}/{seed}/{dataset}",
            )
            _validate_classifier(
                method_state["full_code"], d_sae, f"companion {method} full_code {seed}/{dataset}"
            )
            _validate_classifier(
                method_state["reconstruction"],
                d_model,
                f"companion {method} reconstruction {seed}/{dataset}",
            )
            outputs = heldout_methods[method]
            _require(
                isinstance(outputs, Mapping) and set(outputs) == {"full_code", "reconstruction"},
                f"companion heldout representation set drift for {method}/{seed}/{dataset}",
            )
            for representation in ("full_code", "reconstruction"):
                _validate_outputs(
                    outputs[representation],
                    count,
                    f"companion {method}/{representation}/{seed}/{dataset}",
                )

    def audit_companion(self) -> None:
        root = self.output_root / "companion" / self.checkpoint
        _exact_directories(root, {f"seed_{seed}" for seed in self.seeds}, "companion seed")
        for seed in self.seeds:
            job = root / f"seed_{seed}"
            done_path = job / "done.json"
            done = _read_json(done_path)
            _require(
                done.get("schema_version") == 1 and done.get("complete") is True,
                f"incomplete companion job {seed}",
            )
            required = {
                "config_digest": self.config_digest,
                "probe_seed": seed,
                "dataset_count": len(self.datasets),
                "dataset_manifest_sha256": self.config["benchmark"]["dataset_manifest_sha256"],
                "artifact_hashes": self.resolved.get("artifact_hashes"),
            }
            for key, wanted in required.items():
                _require(done.get(key) == wanted, f"companion done {key} drift for {seed}")
            _require(
                set(done.get("dataset_hashes", {})) == set(self.datasets),
                f"companion done dataset set drift for {seed}",
            )
            metric_files = _exact_files(
                job / "metrics",
                {f"{dataset}.json" for dataset in self.datasets},
                f"companion metrics {seed}",
            )
            weight_files = _exact_files(
                job / "weights",
                {f"{dataset}.pt" for dataset in self.datasets},
                f"companion weights {seed}",
            )
            for dataset in self.datasets:
                metrics_path = metric_files[f"{dataset}.json"]
                metrics = _read_json(metrics_path)
                _require(
                    metrics.get("schema_version") == 2,
                    f"companion metrics schema drift for {seed}/{dataset}",
                )
                identity = {
                    "config_digest": self.config_digest,
                    "checkpoint_id": self.checkpoint,
                    "probe_seed": seed,
                    "dataset": dataset,
                    "family": self.config["benchmark"]["family_by_dataset"][dataset],
                    "regularization": "sae_probes_find_best_reg_l2",
                    "full_code_matrix_format": "scipy_csr_exact_values",
                    "l2_path_optimization": (
                        "parallel_independent_cold_C_loky_cold_selected_C_refit"
                    ),
                    "full_code_cold_C_jobs": int(
                        self.config["runtime"]["companion_full_code_cold_C_jobs_per_worker"]
                    ),
                }
                for key, wanted in identity.items():
                    _require(
                        metrics.get(key) == wanted,
                        f"companion metrics {key} drift for {seed}/{dataset}",
                    )
                count = metrics.get("heldout_example_count")
                _require(
                    isinstance(count, int) and not isinstance(count, bool) and count > 0,
                    f"companion heldout count drift for {seed}/{dataset}",
                )
                expected = _expected_split(self.config, dataset, seed, count)
                _require(
                    metrics.get("heldout_split_id") == expected["split_id"]
                    and metrics.get("heldout_example_id_policy") == expected["example_id_policy"],
                    f"companion metrics split drift for {seed}/{dataset}",
                )
                metric_block = metrics.get("metrics")
                _require(
                    isinstance(metric_block, Mapping)
                    and set(metric_block) == {"original_residual", "methods"},
                    f"companion metric block schema drift for {seed}/{dataset}",
                )
                _validate_metrics(
                    metric_block["original_residual"], f"companion original {seed}/{dataset}"
                )
                _require(
                    isinstance(metric_block["methods"], Mapping)
                    and set(metric_block["methods"]) == set(METHODS),
                    f"companion metric method set drift for {seed}/{dataset}",
                )
                for method in METHODS:
                    reps = metric_block["methods"][method]
                    _require(
                        isinstance(reps, Mapping) and set(reps) == {"full_code", "reconstruction"},
                        f"companion metric representation set drift for {method}/{seed}/{dataset}",
                    )
                    for representation in reps:
                        _validate_metrics(
                            reps[representation],
                            f"companion {method}/{representation}/{seed}/{dataset}",
                        )
                weights_path = weight_files[f"{dataset}.pt"]
                weight_hash = self.record(
                    weights_path, "companion_weights", probe_seed=seed, dataset=dataset
                )
                _require(
                    metrics.get("weights_sha256") == weight_hash,
                    f"companion weight hash drift for {seed}/{dataset}",
                )
                self._validate_companion_weights(_torch_load(weights_path), seed, dataset, count)
                metric_hash = self.record(
                    metrics_path, "companion_metrics", probe_seed=seed, dataset=dataset
                )
                _require(
                    done["dataset_hashes"].get(dataset) == metric_hash,
                    f"companion done/metric hash drift for {seed}/{dataset}",
                )
            self.record(done_path, "companion_done", probe_seed=seed)

    def audit_workers(self) -> None:
        runtime = self.config.get("runtime")
        if not isinstance(runtime, Mapping):
            return
        worker_count = int(runtime.get("worker_count", 0))
        sparse_shards = runtime.get("sparse_worker_shards", [])
        companion_shards = runtime.get("companion_seed_shards", [])
        _require(
            len(sparse_shards) == worker_count and len(companion_shards) == worker_count,
            "worker shard count drift",
        )
        sparse_assignments = [
            (str(shard["method"]), int(seed))
            for shard in sparse_shards
            for seed in shard["probe_seeds"]
        ]
        expected_sparse = [(method, seed) for method in METHODS for seed in self.seeds]
        _require(
            Counter(sparse_assignments) == Counter(expected_sparse),
            "sparse worker mapping is not an exact method-seed partition",
        )
        companion_assignments = [int(seed) for shard in companion_shards for seed in shard]
        _require(
            Counter(companion_assignments) == Counter(self.seeds),
            "companion worker mapping is not an exact seed partition",
        )
        worker_root = self.output_root / "workers"
        files = sorted(worker_root.glob("*.json")) if worker_root.is_dir() else []
        _require(
            len(files) == worker_count,
            f"worker summary count drift: expected={worker_count}, observed={len(files)}",
        )
        by_index: dict[int, tuple[Path, Mapping[str, Any]]] = {}
        for path in files:
            report = _read_json(path)
            index = report.get("worker_index")
            _require(
                isinstance(index, int) and not isinstance(index, bool) and index not in by_index,
                f"worker index drift in {path}",
            )
            by_index[index] = (path, report)
        _require(set(by_index) == set(range(worker_count)), "worker index coverage drift")
        timing_path = self.output_root / "timing_smoke.json"
        timing_hash = file_sha256(timing_path) if timing_path.is_file() else None
        timing_resources = None
        if timing_path.is_file():
            timing_resources = _validate_runtime_resources(
                self.config,
                _read_json(timing_path).get("runtime_resources"),
                "timing-smoke",
            )
        for index in range(worker_count):
            path, report = by_index[index]
            shard = sparse_shards[index]
            expected = {
                "schema_version": 1,
                "complete": True,
                "config_digest": self.config_digest,
                "worker_index": index,
                "method": shard["method"],
                "probe_seeds": [int(seed) for seed in shard["probe_seeds"]],
                "companion_seeds": [int(seed) for seed in companion_shards[index]],
                "sparse_job_count": len(shard["probe_seeds"]),
                "companion_job_count": len(companion_shards[index]),
            }
            for key, wanted in expected.items():
                _require(report.get(key) == wanted, f"worker {index} {key} drift")
            if timing_hash is not None:
                _require(
                    report.get("timing_smoke_sha256") == timing_hash,
                    f"worker {index} timing-smoke hash drift",
                )
            worker_resources = _validate_runtime_resources(
                self.config, report.get("runtime_resources"), f"worker {index}"
            )
            _require(
                worker_resources == timing_resources,
                f"worker {index} runtime resources differ from timing-smoke",
            )
            self.record(path, "worker_summary", worker_index=index)

    def audit_final(self) -> None:
        candidate_path = self.output_root / "candidate_associations.jsonl"
        _require(
            candidate_path.is_file() and not candidate_path.is_symlink(),
            "candidate associations are missing",
        )
        candidates = []
        for line_number, line in enumerate(
            candidate_path.read_text(encoding="utf-8").splitlines(), 1
        ):
            _require(line.strip() != "", f"blank candidate JSONL line {line_number}")
            try:
                candidates.append(json.loads(line))
            except json.JSONDecodeError as error:
                raise AuditError(f"invalid candidate JSONL line {line_number}: {error}") from error
        candidate_ids = []
        for row in candidates:
            _require(isinstance(row, Mapping), "candidate row is not an object")
            method, dataset = row.get("method"), row.get("dataset")
            _require(
                method in METHODS and dataset in self.datasets, "candidate method or dataset drift"
            )
            _require(row.get("checkpoint_id") == self.checkpoint, "candidate checkpoint drift")
            _require(
                row.get("family") == self.config["benchmark"]["family_by_dataset"][dataset],
                "candidate family drift",
            )
            _require(
                isinstance(row.get("feature_id"), int)
                and not isinstance(row.get("feature_id"), bool),
                "candidate feature ID drift",
            )
            candidate_seeds = row.get("probe_seeds")
            _require(
                isinstance(candidate_seeds, list)
                and len(candidate_seeds) == len(set(candidate_seeds))
                and set(candidate_seeds).issubset(self.seeds),
                "candidate probe-seed set drift",
            )
            _require(
                float(row.get("probe_seed_frequency")) == len(candidate_seeds) / len(self.seeds),
                "candidate probe-seed frequency drift",
            )
            _require(
                row.get("autointerp_eligible") is False,
                "pilot candidate was incorrectly made autointerp eligible",
            )
            candidate_ids.append(row.get("candidate_id"))
        _require(
            all(isinstance(value, str) and value for value in candidate_ids)
            and len(candidate_ids) == len(set(candidate_ids)),
            "candidate IDs are missing or duplicated",
        )
        candidate_hash = self.record(candidate_path, "candidate_associations")

        manifest_path = self.output_root / "candidate_manifest.json"
        manifest = _read_json(manifest_path)
        _require(
            manifest.get("schema_version") == 1
            and manifest.get("config_digest") == self.config_digest,
            "candidate manifest schema/config drift",
        )
        _require(
            manifest.get("candidate_count") == len(candidates), "candidate manifest count drift"
        )
        _require(
            manifest.get("candidate_jsonl_sha256") == candidate_hash,
            "candidate manifest JSONL hash drift",
        )
        _require(
            manifest.get("autointerp_eligible") is False,
            "pilot candidate manifest was incorrectly made eligible",
        )
        manifest_hash = self.record(manifest_path, "candidate_manifest")

        report_path = self.output_root / "advancement_report.json"
        report = _read_json(report_path)
        _require(
            report.get("schema_version") == 1 and report.get("complete") is True,
            "advancement report schema drift",
        )
        _require(
            report.get("config_digest") == self.config_digest
            and report.get("artifact_hashes") == self.resolved.get("artifact_hashes"),
            "advancement report config/artifact drift",
        )
        _require(
            report.get("candidate_manifest_sha256") == manifest_hash,
            "advancement report candidate-manifest hash drift",
        )
        _require(
            report.get("primary", {}).get("k") == int(self.config["statistics"]["primary_k"]),
            "advancement primary k drift",
        )
        _require(
            set(report.get("task_metrics", {})) == set(self.datasets), "advancement task set drift"
        )
        _require(
            set(report.get("companion_task_metrics", {})) == set(self.datasets),
            "advancement companion task set drift",
        )
        for dataset in self.datasets:
            _require(
                set(report["task_metrics"][dataset]) == set(METHODS),
                f"advancement method set drift for {dataset}",
            )
        checks = report.get("checks")
        _require(
            isinstance(checks, Mapping) and checks.get("complete_matrix") is True,
            "advancement complete-matrix check failed",
        )
        _require(
            report.get("advance_autointerp") is False,
            "pilot report incorrectly advanced autointerp",
        )
        _require(
            manifest.get("pilot_gate", {}).get("checks") == checks,
            "candidate/report gate checks drift",
        )
        _require(
            manifest.get("pilot_gate", {}).get("passed")
            == report.get("advance_fresh_confirmation"),
            "candidate/report pilot gate drift",
        )
        self.record(report_path, "advancement_report")

    def expected_counts(self, phase: str) -> dict[str, int]:
        datasets, seeds = len(self.datasets), len(self.seeds)
        runtime = self.config.get("runtime", {})
        worker_count = int(runtime.get("worker_count", 0)) if isinstance(runtime, Mapping) else 0
        counts = {
            "resolved_config": 1,
            "eligibility": 1,
            "cache_ready": 1,
            "cpu_budget": 1,
            "cache_activation": datasets,
            "sparse_done": len(METHODS) * seeds,
            "saebench_result": len(METHODS) * seeds,
            "sparse_raw": len(METHODS) * seeds * datasets,
            "sparse_provenance": len(METHODS) * seeds * datasets,
            "sparse_predictions": len(METHODS) * seeds * datasets,
            "companion_done": seeds,
            "companion_metrics": seeds * datasets,
            "companion_weights": seeds * datasets,
            "worker_summary": worker_count,
        }
        timing = runtime.get("timing_smoke") if isinstance(runtime, Mapping) else None
        if isinstance(timing, Mapping) and timing.get("require_passed_report_before_workers"):
            counts["timing_smoke"] = 1
            measured_workers = int(timing["measured_worker_count"])
            counts["timing_worker_report"] = measured_workers
            counts["timing_worker_exit"] = measured_workers
            counts["timing_barrier_ready"] = measured_workers
            counts["timing_barrier_start"] = 1
        if phase == "final":
            counts.update(
                {"candidate_associations": 1, "candidate_manifest": 1, "advancement_report": 1}
            )
        return counts


def audit_artifacts(
    *,
    config_path: Path,
    output_root: Path,
    phase: str,
    audit_path: Path | None = None,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    _require(phase in {"pre-aggregate", "final"}, f"unknown audit phase: {phase}")
    config = _read_json(config_path.resolve())
    _require(isinstance(config, Mapping), "exp10 config is not an object")
    auditor = ArtifactAuditor(config, config_path, output_root)
    auditor.audit_base()
    auditor.audit_sparse()
    auditor.audit_companion()
    auditor.audit_workers()
    if phase == "final":
        auditor.audit_final()

    expected = auditor.expected_counts(phase)
    observed = dict(sorted(auditor.counts.items()))
    _require(
        observed == dict(sorted(expected.items())),
        f"audited artifact counts drift: expected={expected}, observed={observed}",
    )
    manifest_records = sorted(auditor.records.values(), key=lambda item: item["path"])
    token = phase.replace("-", "_")
    manifest_path = (manifest_path or output_root / f"artifact_manifest_{token}.jsonl").resolve()
    audit_path = (audit_path or output_root / f"artifact_audit_{token}.json").resolve()
    atomic_jsonl(manifest_path, manifest_records)
    report = {
        "schema_version": 1,
        "complete": True,
        "passed": True,
        "phase": phase,
        "config_digest": auditor.config_digest,
        "checkpoint_id": auditor.checkpoint,
        "dataset_count": len(auditor.datasets),
        "probe_seed_count": len(auditor.seeds),
        "methods": list(METHODS),
        "ks": auditor.ks,
        "expected_counts": expected,
        "observed_counts": observed,
        "heldout_split_alignment_count": len(auditor.split_digests),
        "manifest_path": str(manifest_path),
        "manifest_entry_count": len(manifest_records),
        "manifest_sha256": file_sha256(manifest_path),
    }
    atomic_json(audit_path, report)
    return report


def wait_for_completion(config_path: Path, output_root: Path, wait_seconds: float) -> None:
    config = _read_json(config_path.resolve())
    checkpoint = config["pilot_checkpoint"]["checkpoint_id"]
    seeds = config["benchmark"]["probe_seeds"]
    expected_done = [
        output_root / "jobs" / checkpoint / method / f"seed_{seed}" / "done.json"
        for method in METHODS
        for seed in seeds
    ] + [output_root / "companion" / checkpoint / f"seed_{seed}" / "done.json" for seed in seeds]
    worker_count = int(config["runtime"]["worker_count"])
    deadline = time.monotonic() + wait_seconds
    while True:
        missing = [path for path in expected_done if not path.is_file()]
        observed_workers = len(list((output_root / "workers").glob("*.json")))
        if not missing and observed_workers >= worker_count:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                "timed out waiting for exp10 artifacts: "
                f"missing_done={len(missing)}, workers={observed_workers}/{worker_count}"
            )
        time.sleep(min(10.0, max(0.1, deadline - time.monotonic())))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--phase", choices=("pre-aggregate", "final"), required=True)
    parser.add_argument("--audit-path", type=Path)
    parser.add_argument("--manifest-path", type=Path)
    parser.add_argument("--wait-seconds", type=float, default=0)
    args = parser.parse_args()
    try:
        if args.wait_seconds > 0:
            wait_for_completion(args.config, args.output_root, args.wait_seconds)
        report = audit_artifacts(
            config_path=args.config,
            output_root=args.output_root,
            phase=args.phase,
            audit_path=args.audit_path,
            manifest_path=args.manifest_path,
        )
    except Exception as error:
        token = args.phase.replace("-", "_")
        failure_path = (
            args.audit_path or args.output_root / f"artifact_audit_{token}.json"
        ).resolve()
        failure = {
            "schema_version": 1,
            "complete": True,
            "passed": False,
            "phase": args.phase,
            "error_type": type(error).__name__,
            "error": str(error),
        }
        atomic_json(failure_path, failure)
        print(json.dumps(failure, sort_keys=True), file=sys.stderr)
        return 2
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
