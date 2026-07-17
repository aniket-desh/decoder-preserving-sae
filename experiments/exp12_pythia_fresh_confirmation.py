#!/usr/bin/env python3
"""Conditional fresh Pythia confirmation with a concept-blind maturity gate.

The checked-in config is intentionally blocked until every outcome-facing
maturity choice in docs/experiment_plan.md is specified.  ``preflight`` writes
the exact unresolved set and exits before loading the pilot report or a model.
Once frozen, the same driver trains one paired MSE/DPSAE fleet per process,
retains optimizer-bearing maturity snapshots, applies the common stop rule,
and runs the decoder and frozen-network hooks before authorizing concept work.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import signal
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from dpsae.corpus import prepare_token_memmap
from dpsae.decoder_distance import calibrate_ridge
from dpsae.exp04b_natural_text import (
    bootstrap_ratio_interval,
    exact_identity_decoder_statistics,
)
from dpsae.exp04b_training import probe_seed_for_step, stage_seeds
from dpsae.language_model import ActivationStats, estimate_activation_stats
from dpsae.language_training import SAETrainSpec, TrainingFleet
from dpsae.mech_analysis import load_sae

from experiments.exp06_generality import GenericActivationModel, TargetSpec


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/exp12_pythia_fresh_confirmation.json"
DEFAULT_OUTPUT = ROOT / "artifacts/exp12_pythia_fresh_confirmation"
METHODS = ("mse", "dpsae")
MATURITY_METRICS = (
    "nmse",
    "exact_decoder_distortion",
    "inference_l0",
    "dead_feature_fraction",
)
GIB = 2**30

UNRESOLVED_PATHS = (
    "status",
    "model.revision",
    "confirmation.pair_seeds",
    "corpus.cache_absolute_range",
    "corpus.calibration_absolute_range",
    "corpus.training_absolute_range",
    "corpus.maturity_evaluation_absolute_range",
    "corpus.frozen_network_absolute_range",
    "corpus.allow_training_cache_reuse",
    "training.maximum_tokens",
    "training.scheduler_horizon_tokens",
    "maturity_evaluation.exact_decoder.bootstrap_seed",
    "maturity_stop_rule.rule_version",
    "maturity_stop_rule.minimum_checkpoint_tokens",
    "maturity_stop_rule.plateau_consecutive_intervals",
    "maturity_stop_rule.maximum_relative_change.nmse",
    "maturity_stop_rule.maximum_relative_change.exact_decoder_distortion",
    "maturity_stop_rule.maximum_relative_change.inference_l0",
    "maturity_stop_rule.maximum_relative_change.dead_feature_fraction",
    "maturity_stop_rule.extension_to_500m_policy",
    "maturity_stop_rule.minimum_unique_exposure_for_500m",
    "maturity_stop_rule.maximum_cache_reuse_count_for_500m",
    "maturity_stop_rule.no_plateau_by_maximum_policy",
    "frozen_network_evaluation.sampling_seed",
    "frozen_network_evaluation.bootstrap_seed",
    "randomness.base_seed",
)


class FreezeBlockedError(RuntimeError):
    def __init__(self, choices: Sequence[str]) -> None:
        self.choices = tuple(choices)
        super().__init__("fresh Pythia confirmation is not frozen: " + ", ".join(self.choices))


class RunAbortedError(RuntimeError):
    """Raised when another Exp12 process has requested a fail-fast stop."""


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_bytes), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def promoted_file_record(source: Path, destination: Path) -> dict[str, Any]:
    """Describe a temporary file at its immutable post-promotion path."""

    if not source.is_file():
        raise FileNotFoundError(source)
    return {
        "path": str(destination.resolve()),
        "bytes": source.stat().st_size,
        "sha256": file_sha256(source),
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def _stage_paths(
    output_root: Path, stage: str, pair_seed: int | None = None
) -> tuple[Path, Path]:
    if stage in {"train-pair", "wait-shared"}:
        if pair_seed is None:
            raise ValueError(f"{stage} status requires a pair seed")
        root = output_root / "pairs" / f"seed_{pair_seed}"
        return root / "pair_status.json", root / "pair_failed.json"
    if stage == "coordinator":
        return output_root / "coordinator_status.json", output_root / "coordinator_failed.json"
    if stage == "timing-smoke":
        root = output_root / "nonreport_timing_smoke"
        return root / "smoke_status.json", root / "smoke_failed.json"
    raise ValueError(f"stage has no retained status contract: {stage}")


def write_stage_status(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    stage: str,
    state: str,
    pair_seed: int | None = None,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if state not in {"running", "complete", "failed"}:
        raise ValueError(f"unknown retained stage state: {state}")
    status_path, _ = _stage_paths(output_root, stage, pair_seed)
    payload = {
        "schema_version": 1,
        "complete": state in {"complete", "failed"},
        "failed": state == "failed",
        "state": state,
        "stage": stage,
        "pair_seed": pair_seed,
        "config_digest": canonical_digest(config),
        "written_at_unix_seconds": time.time(),
        **dict(extra or {}),
    }
    atomic_json(status_path, payload)
    return payload


def write_stage_failure(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    stage: str,
    error: Exception,
    pair_seed: int | None = None,
) -> dict[str, Any]:
    status_path, failure_path = _stage_paths(output_root, stage, pair_seed)
    payload = {
        "schema_version": 1,
        "complete": True,
        "failed": True,
        "state": "failed",
        "stage": stage,
        "pair_seed": pair_seed,
        "config_digest": canonical_digest(config),
        "error_type": type(error).__name__,
        "error": str(error),
        "written_at_unix_seconds": time.time(),
    }
    atomic_json(failure_path, payload)
    atomic_json(status_path, payload)
    return payload


def request_abort(
    *, config: Mapping[str, Any], output_root: Path, reason: str
) -> dict[str, Any]:
    path = output_root / "abort_requested.json"
    if path.is_file():
        existing = json.loads(path.read_text())
        if existing.get("config_digest") != canonical_digest(config):
            raise RuntimeError("Exp12 abort marker belongs to another config")
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


def _check_abort_or_deadline(config: Mapping[str, Any], output_root: Path) -> None:
    abort_path = output_root / "abort_requested.json"
    if abort_path.is_file():
        abort = json.loads(abort_path.read_text())
        if abort.get("config_digest") != canonical_digest(config):
            raise RuntimeError("Exp12 abort marker belongs to another config")
        raise RunAbortedError(str(abort.get("reason", "another Exp12 process failed")))
    coordinator_path = output_root / "coordinator_status.json"
    if not coordinator_path.is_file():
        raise RuntimeError("Exp12 coordinator status is missing")
    coordinator = json.loads(coordinator_path.read_text())
    if coordinator.get("config_digest") != canonical_digest(config):
        raise RuntimeError("Exp12 coordinator status belongs to another config")
    deadline = float(coordinator.get("deadline_unix_seconds", 0))
    if deadline <= 0:
        raise RuntimeError("Exp12 coordinator status lacks a valid deadline")
    if time.time() >= deadline:
        raise TimeoutError("Exp12 exceeded its frozen wall-time ceiling")


def repository_state() -> dict[str, Any]:
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"], cwd=ROOT, text=True
    ).splitlines()
    return {"revision": revision, "dirty": bool(status), "status": status}


def _at_path(value: Mapping[str, Any], dotted: str) -> Any:
    current: Any = value
    for key in dotted.split("."):
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def unresolved_choices(config: Mapping[str, Any]) -> list[str]:
    unresolved = []
    for path in UNRESOLVED_PATHS:
        value = _at_path(config, path)
        if path == "status":
            if value != "frozen":
                unresolved.append("status must be changed to 'frozen' after all choices are reviewed")
        elif value is None:
            unresolved.append(path)
    return unresolved


def _range(config: Mapping[str, Any], key: str) -> tuple[int, int]:
    value = config["corpus"][key]
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 2:
        raise ValueError(f"corpus.{key} must be [absolute_start, absolute_stop]")
    start, stop = (int(item) for item in value)
    if start < 0 or stop <= start:
        raise ValueError(f"corpus.{key} is empty or negative")
    return start, stop


def realized_tokens(requested: int, tokens_per_batch: int) -> int:
    if requested <= 0 or tokens_per_batch <= 0:
        raise ValueError("token budgets and minibatches must be positive")
    return math.ceil(requested / tokens_per_batch) * tokens_per_batch


def snapshot_budgets(config: Mapping[str, Any]) -> list[int]:
    maximum = int(config["training"]["maximum_tokens"])
    return [int(value) for value in config["training"]["snapshot_tokens"] if int(value) <= maximum]


def validate_config(config: Mapping[str, Any], *, require_frozen: bool = True) -> None:
    if config.get("schema_version") != 1 or config.get("experiment_id") != "exp12_pythia_fresh_confirmation":
        raise ValueError("not an exp12 fresh-Pythia config")
    fixed = {
        "model.name": "EleutherAI/pythia-160m-deduped",
        "model.architecture": "gpt_neox",
        "model.one_based_block": 8,
        "model.hidden_size": 768,
        "sae.dictionary_size": 16_384,
        "sae.target_l0": 32,
        "sae.decoder_weight": 0.03125,
        "sae.sparsity_mode": "batch_topk",
        "confirmation.pair_count": 3,
        "confirmation.pairing": (
            "same_initialization_activation_stream_minibatch_order_and_schedule_within_pair"
        ),
        "confirmation.replication_unit": "trained_mse_dpsae_checkpoint_pair",
        "corpus.stream_order": "sequential_nonoverlapping_sequences_v1",
        "corpus.reuse_tracking": (
            "exact_delivered_tokens_unique_tokens_cache_epoch_and_reused_tokens"
        ),
        "corpus.allow_training_cache_reuse": False,
        "training.snapshot_boundary_rule": "first_completed_minibatch_at_or_above_requested_tokens",
        "training.maximum_tokens": 250_000_000,
        "training.scheduler": "linear_warmup_then_cosine_decay",
        "training.scheduler_horizon_tokens": 250_000_000,
        "training.resume_requires_optimizer_scheduler_stream_and_hash_match": True,
        "maturity_evaluation.exact_decoder.retain_group_numerators_and_denominators": True,
        "maturity_evaluation.matched_quality.maximum_dead_feature_fraction": 0.01,
        "maturity_stop_rule.require_plateau_for_both_methods_in_every_pair": True,
        "maturity_stop_rule.require_matched_quality_in_every_pair": True,
        "maturity_stop_rule.common_checkpoint_selection": (
            "earliest_snapshot_passing_all_frozen_conditions"
        ),
        "maturity_stop_rule.extension_to_500m_policy": "not_applicable_maximum_250m",
        "maturity_stop_rule.minimum_unique_exposure_for_500m": 500_000_000,
        "maturity_stop_rule.maximum_cache_reuse_count_for_500m": 0,
        "maturity_stop_rule.no_plateau_by_maximum_policy": "fail_without_common_checkpoint",
        "frozen_network_evaluation.retain_per_sequence_sufficient_statistics": True,
        "concept_authorization.requires_stop_rule_contract_before_training": True,
        "concept_authorization.requires_common_maturity_decision": True,
        "concept_authorization.requires_decoder_distortion_outputs": True,
        "concept_authorization.requires_frozen_network_outputs": True,
        "concept_authorization.concept_results_forbidden_as_maturity_inputs": True,
        "concept_authorization.confirmatory_inference.global_family_block_interval_is_gate_forming": True,
        "concept_authorization.confirmatory_inference.family_specific_p_values": (
            "one_sided_centered_paired_stratified_heldout_example_bootstrap"
        ),
        "concept_authorization.confirmatory_inference.holm_adjustment_scope": (
            "family_specific_reporting_only"
        ),
        "concept_authorization.confirmatory_inference.family_significance_is_gate_forming": False,
        "concept_authorization.confirmatory_inference.individual_concepts": "descriptive_only",
        "randomness.pair_seed_derivation": "blake2s_stage_stream_pair_v1",
        "timing_smoke.reportable": False,
        "timing_smoke.pair_seed": 2_027_071_899,
        "timing_smoke.cache_absolute_range": [0, 2_066_432],
        "timing_smoke.calibration_absolute_range": [0, 65_536],
        "timing_smoke.training_absolute_range": [65_536, 2_066_432],
        "timing_smoke.requested_training_tokens": 2_000_000,
        "timing_smoke.projection_headroom_multiplier": 1.3,
        "timing_smoke.maximum_projected_confirmation_wall_hours": 7.25,
        "timing_smoke.maximum_smoke_wall_minutes": 45,
        "timing_smoke.maximum_peak_reserved_gib": 44,
        "timing_smoke.artifact_subdirectory": "nonreport_timing_smoke",
        "runpod.maximum_wall_hours": 8,
    }
    for path, expected in fixed.items():
        if _at_path(config, path) != expected:
            raise ValueError(f"frozen exp12 setting changed: {path}")
    if [int(value) for value in config["training"]["snapshot_tokens"]] != [
        25_000_000,
        50_000_000,
        100_000_000,
        250_000_000,
        500_000_000,
    ]:
        raise ValueError("maturity snapshot schedule changed")
    if [int(value) for value in config["training"]["required_snapshot_tokens"]] != [
        25_000_000,
        50_000_000,
        100_000_000,
        250_000_000,
    ]:
        raise ValueError("required maturity snapshots changed")
    if list(config["maturity_evaluation"]["metrics"]) != list(MATURITY_METRICS):
        raise ValueError("concept-blind maturity metrics changed")
    if config["maturity_stop_rule"].get("concept_blind") is not True:
        raise ValueError("maturity stop rule must remain concept blind")
    choices = unresolved_choices(config)
    if choices:
        if require_frozen:
            raise FreezeBlockedError(choices)
        return

    revision = str(config["model"]["revision"])
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("model.revision must be a full lowercase commit SHA")
    seeds = [int(value) for value in config["confirmation"]["pair_seeds"]]
    if len(seeds) != 3 or len(seeds) != len(set(seeds)) or any(seed < 0 for seed in seeds):
        raise ValueError("confirmation.pair_seeds must contain three unique nonnegative seeds")
    if int(config["randomness"]["base_seed"]) < 0:
        raise ValueError("randomness.base_seed must be nonnegative")
    smoke = config["timing_smoke"]
    smoke_seed = int(smoke["pair_seed"])
    if smoke_seed in seeds:
        raise ValueError("the nonreport timing-smoke seed overlaps a confirmation seed")

    names = (
        "calibration_absolute_range",
        "training_absolute_range",
        "maturity_evaluation_absolute_range",
        "frozen_network_absolute_range",
    )
    ranges = {name: _range(config, name) for name in names}
    cache = _range(config, "cache_absolute_range")
    if any(boundary % int(config["training"]["sequence_length"]) for boundary in cache):
        raise ValueError("cache_absolute_range must align to the frozen sequence length")
    for name, interval in ranges.items():
        if interval[0] < cache[0] or interval[1] > cache[1]:
            raise ValueError(f"{name} lies outside cache_absolute_range")
        if any(boundary % int(config["training"]["sequence_length"]) for boundary in interval):
            raise ValueError(f"{name} must align to the frozen sequence length")
    for index, left_name in enumerate(names):
        for right_name in names[index + 1 :]:
            left, right = ranges[left_name], ranges[right_name]
            if max(left[0], right[0]) < min(left[1], right[1]):
                raise ValueError(f"fresh corpus ranges overlap: {left_name}, {right_name}")
    sequence_length = int(config["training"]["sequence_length"])
    batch_size = int(config["training"]["sequences_per_batch"])
    tokens_per_batch = sequence_length * batch_size
    if sequence_length <= 0 or batch_size <= 0:
        raise ValueError("training sequence and batch sizes must be positive")
    if int(config["calibration"]["activation_tokens"]) > (
        ranges["calibration_absolute_range"][1] - ranges["calibration_absolute_range"][0]
    ):
        raise ValueError("calibration range is too short")
    maturity_tokens = int(config["maturity_evaluation"]["activation_tokens"])
    if maturity_tokens > (
        ranges["maturity_evaluation_absolute_range"][1]
        - ranges["maturity_evaluation_absolute_range"][0]
    ):
        raise ValueError("maturity evaluation range is too short")
    if maturity_tokens % int(config["maturity_evaluation"]["exact_decoder"]["group_size"]):
        raise ValueError("maturity tokens must divide exact decoder groups")
    maximum = int(config["training"]["maximum_tokens"])
    if maximum not in {250_000_000, 500_000_000}:
        raise ValueError("training.maximum_tokens must be frozen to 250M or 500M")
    if int(config["training"]["scheduler_horizon_tokens"]) != maximum:
        raise ValueError("cosine scheduler horizon must equal the frozen maximum token budget")
    if any(value > maximum for value in config["training"]["required_snapshot_tokens"]):
        raise ValueError("maximum token budget omits a required 250M maturity snapshot")
    usable_training = (ranges["training_absolute_range"][1] - ranges["training_absolute_range"][0]) // sequence_length * sequence_length
    allow_reuse = config["corpus"]["allow_training_cache_reuse"]
    if not isinstance(allow_reuse, bool):
        raise ValueError("allow_training_cache_reuse must be boolean")
    if not allow_reuse and usable_training < realized_tokens(maximum, tokens_per_batch):
        raise ValueError("fresh sequential training range is too short for the realized maximum budget")
    smoke_cache = tuple(int(value) for value in smoke["cache_absolute_range"])
    smoke_calibration = tuple(int(value) for value in smoke["calibration_absolute_range"])
    smoke_training = tuple(int(value) for value in smoke["training_absolute_range"])
    if any(boundary % sequence_length for boundary in (*smoke_cache, *smoke_calibration, *smoke_training)):
        raise ValueError("timing-smoke ranges must align to the frozen sequence length")
    if not (
        smoke_cache[0] <= smoke_calibration[0] < smoke_calibration[1] <= smoke_cache[1]
        and smoke_cache[0] <= smoke_training[0] < smoke_training[1] <= smoke_cache[1]
        and smoke_calibration[1] <= smoke_training[0]
    ):
        raise ValueError("timing-smoke calibration and training ranges are invalid")
    if max(smoke_cache[0], cache[0]) < min(smoke_cache[1], cache[1]):
        raise ValueError("nonreport timing-smoke tokens overlap the confirmation cache")
    smoke_requested = int(smoke["requested_training_tokens"])
    smoke_capacity = smoke_training[1] - smoke_training[0]
    if smoke_capacity < realized_tokens(smoke_requested, tokens_per_batch):
        raise ValueError("timing-smoke training range is too short")
    headroom = float(smoke["projection_headroom_multiplier"])
    maximum_projection = float(smoke["maximum_projected_confirmation_wall_hours"])
    maximum_smoke_minutes = float(smoke["maximum_smoke_wall_minutes"])
    if not math.isfinite(headroom) or headroom < 1:
        raise ValueError("timing-smoke projection headroom must be at least one")
    if maximum_projection <= 0 or maximum_smoke_minutes <= 0:
        raise ValueError("timing-smoke wall-time gates must be positive")

    rule = config["maturity_stop_rule"]
    if int(rule["minimum_checkpoint_tokens"]) not in snapshot_budgets(config):
        raise ValueError("minimum maturity checkpoint is outside the frozen snapshot schedule")
    intervals = int(rule["plateau_consecutive_intervals"])
    if intervals < 1:
        raise ValueError("plateau_consecutive_intervals must be positive")
    minimum_index = snapshot_budgets(config).index(int(rule["minimum_checkpoint_tokens"]))
    if minimum_index < intervals:
        raise ValueError("minimum checkpoint does not have enough prior intervals for plateau testing")
    for metric in MATURITY_METRICS:
        tolerance = float(rule["maximum_relative_change"][metric])
        if not math.isfinite(tolerance) or tolerance < 0:
            raise ValueError(f"invalid plateau tolerance for {metric}")
    dead_ceiling = float(
        config["maturity_evaluation"]["matched_quality"][
            "maximum_dead_feature_fraction"
        ]
    )
    if not math.isfinite(dead_ceiling) or not 0 <= dead_ceiling <= 1:
        raise ValueError("maximum_dead_feature_fraction must lie in [0, 1]")
    extension_policy = rule["extension_to_500m_policy"]
    expected_extension_policy = (
        "always_if_fresh_capacity"
        if maximum == 500_000_000
        else "not_applicable_maximum_250m"
    )
    if extension_policy != expected_extension_policy:
        raise ValueError(
            "the 500M policy must agree with the outcome-independent frozen maximum "
            f"({expected_extension_policy!r})"
        )
    if rule["no_plateau_by_maximum_policy"] != "fail_without_common_checkpoint":
        raise ValueError("fresh confirmation must fail when no common plateau passes")
    if maximum == 500_000_000:
        if allow_reuse:
            raise ValueError("500M confirmation forbids training-cache reuse")
        if int(rule["minimum_unique_exposure_for_500m"]) < 500_000_000:
            raise ValueError("500M requires at least 500M unique-token exposure")
        if int(rule["maximum_cache_reuse_count_for_500m"]) != 0:
            raise ValueError("500M maximum cache reuse count must be zero")
    frozen = config["frozen_network_evaluation"]
    if int(frozen["sequence_length"]) != 256 or int(frozen["sequences"]) != 2048:
        raise ValueError("frozen-network sequence contract changed")
    frozen_range = ranges["frozen_network_absolute_range"]
    if frozen_range[1] - frozen_range[0] < int(frozen["sequence_length"]) * int(frozen["sequences"]):
        raise ValueError("frozen-network range cannot supply nonoverlapping sequences")
    runpod = config["runpod"]
    if (
        runpod["gpu_type"] != "A40"
        or int(runpod["gpu_count"]) != 4
        or [int(value) for value in runpod["pair_gpu_indices"]] != [0, 1, 2]
        or int(runpod["coordinator_gpu_index"]) != 3
    ):
        raise ValueError("four-A40 worker mapping changed")
    maximum_wall_hours = float(runpod["maximum_wall_hours"])
    if not math.isfinite(maximum_wall_hours) or maximum_wall_hours <= 0:
        raise ValueError("runpod.maximum_wall_hours must be positive")
    if maximum_projection + maximum_smoke_minutes / 60 > maximum_wall_hours:
        raise ValueError("timing smoke and projected confirmation exceed the pod wall-time ceiling")
    if float(smoke["maximum_peak_reserved_gib"]) > float(runpod["maximum_gpu_reserved_gib"]):
        raise ValueError("timing-smoke memory gate exceeds the fleet memory guard")


def timeout_seconds_for_run(config: Mapping[str, Any], timeout_hours: float) -> float:
    maximum_hours = float(config["runpod"]["maximum_wall_hours"])
    if not math.isfinite(timeout_hours) or timeout_hours <= 0:
        raise ValueError("--timeout-hours must be positive")
    if timeout_hours > maximum_hours:
        raise ValueError(
            f"--timeout-hours={timeout_hours:g} exceeds the frozen {maximum_hours:g}-hour ceiling"
        )
    return timeout_hours * 3600


def load_config(path: Path, *, require_frozen: bool = True) -> dict[str, Any]:
    config = json.loads(path.read_text())
    validate_config(config, require_frozen=require_frozen)
    return config


class SequentialTokenStream:
    """Deterministic nonoverlapping corpus stream with exact reuse accounting."""

    def __init__(
        self,
        path: Path,
        *,
        cache_absolute_range: tuple[int, int],
        training_absolute_range: tuple[int, int],
        sequence_length: int,
        batch_size: int,
        allow_reuse: bool,
    ) -> None:
        cache_start, cache_stop = cache_absolute_range
        train_start, train_stop = training_absolute_range
        if train_start < cache_start or train_stop > cache_stop:
            raise ValueError("training stream lies outside the token cache")
        if sequence_length <= 0 or batch_size <= 0:
            raise ValueError("stream sequence length and batch size must be positive")
        self.path = path
        self.cache_start = cache_start
        self.cache_stop = cache_stop
        self.train_start = train_start
        self.train_stop = train_stop
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.allow_reuse = allow_reuse
        self.available_sequences = (train_stop - train_start) // sequence_length
        if self.available_sequences < batch_size:
            raise ValueError("training range is smaller than one full minibatch")
        self.usable_tokens = self.available_sequences * sequence_length
        self.total_sequences = 0
        self.tokens = np.memmap(
            path,
            mode="r",
            dtype=np.uint16,
            shape=(cache_stop - cache_start,),
        )

    def batch(self) -> tuple[Tensor, dict[str, Any]]:
        stop_sequence = self.total_sequences + self.batch_size
        if not self.allow_reuse and stop_sequence > self.available_sequences:
            raise RuntimeError("fresh sequential corpus exhausted before the frozen budget")
        logical = torch.arange(self.total_sequences, stop_sequence, dtype=torch.int64)
        physical = logical.remainder(self.available_sequences)
        starts = self.train_start + physical * self.sequence_length
        rows = np.stack(
            [
                self.tokens[
                    int(start) - self.cache_start : int(start) - self.cache_start + self.sequence_length
                ]
                for start in starts
            ]
        ).astype(np.int64, copy=False)
        self.total_sequences = stop_sequence
        delivered = self.total_sequences * self.sequence_length
        unique = min(delivered, self.usable_tokens)
        reused = delivered - unique
        epoch = (self.total_sequences - 1) // self.available_sequences
        record = {
            "absolute_sequence_starts": starts.tolist(),
            "delivered_tokens": delivered,
            "unique_corpus_exposure_tokens": unique,
            "reused_tokens": reused,
            "cache_epoch": epoch,
            "cache_reuse_count": epoch,
            "unique_absolute_interval": [self.train_start, self.train_start + unique],
        }
        return torch.from_numpy(rows), record

    def state_dict(self) -> dict[str, Any]:
        return {
            "total_sequences": self.total_sequences,
            "stream_contract": self.contract(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("stream_contract") != self.contract():
            raise RuntimeError("training stream contract changed across resume")
        total = int(state.get("total_sequences", -1))
        if total < 0 or (not self.allow_reuse and total > self.available_sequences):
            raise RuntimeError("invalid resumed sequential-stream position")
        self.total_sequences = total

    def contract(self) -> dict[str, Any]:
        return {
            "cache_absolute_range": [self.cache_start, self.cache_stop],
            "training_absolute_range": [self.train_start, self.train_stop],
            "sequence_length": self.sequence_length,
            "batch_size": self.batch_size,
            "allow_reuse": self.allow_reuse,
            "available_sequences": self.available_sequences,
        }


def _contains_forbidden_key(value: Any, forbidden: Sequence[str], prefix: str = "") -> str | None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            name = str(key).lower()
            path = f"{prefix}.{key}" if prefix else str(key)
            allowed_metric_key = name in MATURITY_METRICS or name == "dead_feature_count"
            if not allowed_metric_key and any(token.lower() in name for token in forbidden):
                return path
            found = _contains_forbidden_key(child, forbidden, path)
            if found is not None:
                return found
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        for index, child in enumerate(value):
            found = _contains_forbidden_key(child, forbidden, f"{prefix}[{index}]")
            if found is not None:
                return found
    return None


def _matched_quality(config: Mapping[str, Any], snapshot: Mapping[str, Any]) -> dict[str, bool]:
    models = snapshot["models"]
    mse, dpsae = models["mse"], models["dpsae"]
    gates = config["maturity_evaluation"]["matched_quality"]
    target = float(config["sae"]["target_l0"])
    return {
        "nmse": float(dpsae["nmse"]) / max(float(mse["nmse"]), 1e-12)
        <= float(gates["maximum_dpsae_to_mse_nmse_ratio"]),
        "l0_target": all(
            abs(float(models[method]["inference_l0"]) - target) / target
            <= float(gates["maximum_relative_l0_error"])
            for method in METHODS
        ),
        "l0_pair": abs(float(dpsae["inference_l0"]) - float(mse["inference_l0"])) / target
        <= float(gates["maximum_pair_relative_l0_difference"]),
        "dead_features": all(
            float(models[method]["dead_feature_fraction"])
            <= float(gates["maximum_dead_feature_fraction"])
            for method in METHODS
        ),
    }


def build_maturity_decision(
    config: Mapping[str, Any],
    snapshots_by_seed: Mapping[int, Mapping[int, Mapping[str, Any]]],
    *,
    stop_rule_contract_sha256: str,
) -> dict[str, Any]:
    """Apply only the frozen natural-text maturity metrics to all three pairs."""

    validate_config(config, require_frozen=True)
    seeds = [int(value) for value in config["confirmation"]["pair_seeds"]]
    budgets = snapshot_budgets(config)
    if set(snapshots_by_seed) != set(seeds):
        raise ValueError("maturity inputs do not cover the frozen pair seeds")
    forbidden = config["maturity_stop_rule"]["forbidden_input_key_substrings"]
    found = _contains_forbidden_key(snapshots_by_seed, forbidden)
    if found is not None:
        raise ValueError(f"concept-facing maturity input is forbidden: {found}")
    for seed in seeds:
        if set(snapshots_by_seed[seed]) != set(budgets):
            raise ValueError(f"seed {seed} maturity snapshots do not cover the frozen budget grid")
        for budget in budgets:
            snapshot = snapshots_by_seed[seed][budget]
            if set(snapshot.get("models", {})) != set(METHODS):
                raise ValueError(f"seed {seed}, budget {budget} lacks a paired model metric block")
            if int(snapshot.get("requested_snapshot_tokens", -1)) != budget:
                raise ValueError(f"seed {seed}, budget {budget} has the wrong snapshot identity")
            for method in METHODS:
                for metric in MATURITY_METRICS:
                    value = float(snapshot["models"][method][metric])
                    if not math.isfinite(value) or value < 0:
                        raise ValueError(f"nonfinite maturity metric: {seed}/{budget}/{method}/{metric}")

    rule = config["maturity_stop_rule"]
    intervals = int(rule["plateau_consecutive_intervals"])
    minimum = int(rule["minimum_checkpoint_tokens"])
    candidate_rows = []
    selected: int | None = None
    for index, budget in enumerate(budgets):
        matched = {
            str(seed): _matched_quality(config, snapshots_by_seed[seed][budget])
            for seed in seeds
        }
        matched_all = all(all(checks.values()) for checks in matched.values())
        deltas: dict[str, Any] = {}
        plateau = index >= intervals and budget >= minimum
        if plateau:
            for transition in range(index - intervals + 1, index + 1):
                previous_budget, current_budget = budgets[transition - 1], budgets[transition]
                transition_key = f"{previous_budget}_to_{current_budget}"
                deltas[transition_key] = {}
                for seed in seeds:
                    deltas[transition_key][str(seed)] = {}
                    for method in METHODS:
                        deltas[transition_key][str(seed)][method] = {}
                        for metric in MATURITY_METRICS:
                            previous = float(
                                snapshots_by_seed[seed][previous_budget]["models"][method][metric]
                            )
                            current = float(
                                snapshots_by_seed[seed][current_budget]["models"][method][metric]
                            )
                            denominator = max(abs(previous), 1e-12)
                            if metric == "dead_feature_fraction":
                                # A zero-dead checkpoint is common. Measure a later
                                # one-feature change against one dictionary atom rather
                                # than an arbitrary floating-point epsilon.
                                denominator = max(
                                    abs(previous),
                                    1 / int(config["sae"]["dictionary_size"]),
                                )
                            change = abs(current - previous) / denominator
                            tolerance = float(rule["maximum_relative_change"][metric])
                            passed = change <= tolerance or math.isclose(
                                change,
                                tolerance,
                                rel_tol=0,
                                abs_tol=1e-12,
                            )
                            deltas[transition_key][str(seed)][method][metric] = {
                                "relative_change": change,
                                "reference_denominator": denominator,
                                "maximum": tolerance,
                                "passed": passed,
                            }
                            plateau = plateau and passed
        fresh_500 = True
        if budget == 500_000_000:
            fresh_500 = all(
                int(snapshots_by_seed[seed][budget]["corpus_exposure"]["unique_corpus_exposure_tokens"])
                >= int(rule["minimum_unique_exposure_for_500m"])
                and int(snapshots_by_seed[seed][budget]["corpus_exposure"]["cache_reuse_count"])
                <= int(rule["maximum_cache_reuse_count_for_500m"])
                for seed in seeds
            )
        eligible = budget >= minimum and plateau and matched_all and fresh_500
        candidate_rows.append(
            {
                "requested_snapshot_tokens": budget,
                "matched_quality": matched,
                "matched_quality_passed": matched_all,
                "plateau_passed": plateau,
                "fresh_500m_exposure_passed": fresh_500,
                "relative_changes": deltas,
                "eligible": eligible,
            }
        )
        if eligible and selected is None:
            selected = budget
    return {
        "schema_version": 1,
        "complete": True,
        "concept_blind": True,
        "config_digest": canonical_digest(config),
        "stop_rule_contract_sha256": stop_rule_contract_sha256,
        "pair_seeds": seeds,
        "candidate_checkpoints": candidate_rows,
        "selected_requested_snapshot_tokens": selected,
        "common_checkpoint_selected": selected is not None,
        "fallback_applied": False,
        "no_plateau_by_maximum_policy": rule["no_plateau_by_maximum_policy"],
    }


def _source_hashes(config_path: Path) -> dict[str, str]:
    paths = (
        config_path.resolve(),
        Path(__file__).resolve(),
        ROOT / "src/dpsae/corpus.py",
        ROOT / "src/dpsae/decoder_distance.py",
        ROOT / "src/dpsae/exp04b_natural_text.py",
        ROOT / "src/dpsae/exp04b_training.py",
        ROOT / "src/dpsae/language_model.py",
        ROOT / "src/dpsae/language_sae.py",
        ROOT / "src/dpsae/language_training.py",
        ROOT / "src/dpsae/mech_analysis.py",
        ROOT / "experiments/exp06_generality.py",
        ROOT / "experiments/exp08_language_evidence.py",
        ROOT / "experiments/exp09_frozen_network.py",
    )
    return {
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): file_sha256(path)
        for path in paths
    }


def write_preflight_contract(
    *, config_path: Path, output_root: Path, pilot_report_path: Path
) -> dict[str, Any]:
    """Freeze the maturity rule before any fresh training starts."""

    config = load_config(config_path, require_frozen=False)
    choices = unresolved_choices(config)
    if choices:
        blocked = {
            "schema_version": 1,
            "complete": True,
            "ready_for_training": False,
            "experiment_id": config["experiment_id"],
            "unresolved_choices": choices,
            "pilot_report_opened": False,
            "model_loaded": False,
            "concept_outcomes_opened_by_this_stage": False,
        }
        atomic_json(output_root / "freeze_blocked.json", blocked)
        raise FreezeBlockedError(choices)
    validate_config(config, require_frozen=True)
    repository = repository_state()
    if repository["dirty"]:
        raise RuntimeError("fresh confirmation requires a clean repository revision")
    pilot = json.loads(pilot_report_path.read_text())
    gate = config["pilot_gate"]
    if pilot.get("complete") is not True or pilot.get(gate["report_field"]) is not gate["required_value"]:
        raise RuntimeError("concept pilot did not authorize fresh Pythia confirmation")
    if pilot.get("advance_autointerp") is True:
        raise RuntimeError("fresh training must precede any pilot-driven autointerp expansion")
    resolved = {
        "schema_version": 1,
        "complete": True,
        "config_digest": canonical_digest(config),
        "config": file_record(config_path),
        "repository": repository,
        "source_hashes": _source_hashes(config_path),
        "pilot_report": file_record(pilot_report_path),
        "pilot_gate": {
            "field": gate["report_field"],
            "required_value": gate["required_value"],
            "observed_value": pilot[gate["report_field"]],
            "excluded_pilot_checkpoint_id": gate["exclude_pilot_checkpoint_id"],
        },
    }
    resolved_path = output_root / "resolved_config.json"
    if resolved_path.exists() and json.loads(resolved_path.read_text()) != resolved:
        raise RuntimeError("fresh confirmation resolved config changed; use a new run ID")
    atomic_json(resolved_path, resolved)
    rule = {
        "schema_version": 1,
        "complete": True,
        "frozen_before_training": True,
        "concept_blind": True,
        "config_digest": resolved["config_digest"],
        "maturity_evaluation": config["maturity_evaluation"],
        "maturity_stop_rule": config["maturity_stop_rule"],
        "snapshot_tokens": snapshot_budgets(config),
        "pair_seeds": config["confirmation"]["pair_seeds"],
        "maximum_tokens": config["training"]["maximum_tokens"],
        "scheduler_horizon_tokens": config["training"]["scheduler_horizon_tokens"],
        "corpus_contract": config["corpus"],
        "pilot_report": resolved["pilot_report"],
        "pilot_gate": resolved["pilot_gate"],
        "forbidden_inputs": config["maturity_stop_rule"]["forbidden_input_key_substrings"],
    }
    rule["rule_digest"] = canonical_digest(rule)
    rule_path = output_root / "stop_rule_contract.json"
    if rule_path.exists() and json.loads(rule_path.read_text()) != rule:
        raise RuntimeError("stop-rule contract changed after it was frozen")
    atomic_json(rule_path, rule)
    return rule


def _require_contract(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    resolved_path = output_root / "resolved_config.json"
    contract_path = output_root / "stop_rule_contract.json"
    if not resolved_path.is_file() or not contract_path.is_file():
        raise RuntimeError("run preflight before preparing or training")
    resolved = json.loads(resolved_path.read_text())
    contract = json.loads(contract_path.read_text())
    digest = canonical_digest(config)
    if resolved.get("config_digest") != digest or contract.get("config_digest") != digest:
        raise RuntimeError("frozen preflight contract belongs to another config")
    _verify_file_record(resolved["config"], "frozen config")
    _verify_file_record(resolved["pilot_report"], "pilot advancement report")
    for name, expected_sha256 in resolved.get("source_hashes", {}).items():
        source = Path(name) if Path(name).is_absolute() else ROOT / name
        if not source.is_file() or file_sha256(source) != expected_sha256:
            raise RuntimeError(f"source changed after exp12 preflight: {name}")
    if contract.get("frozen_before_training") is not True or contract.get("concept_blind") is not True:
        raise RuntimeError("maturity stop-rule contract is not frozen and concept blind")
    expected_rule = dict(contract)
    rule_digest = expected_rule.pop("rule_digest", None)
    if rule_digest != canonical_digest(expected_rule):
        raise RuntimeError("stop-rule contract digest changed")
    return contract


def _dtype(name: str) -> torch.dtype:
    values = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in values:
        raise ValueError(f"unsupported model dtype: {name}")
    return values[name]


def load_lm(
    config: Mapping[str, Any], device: torch.device, *, local_files_only: bool
) -> GenericActivationModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config["model"]
    common = {
        "revision": str(model_config["revision"]),
        "local_files_only": local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(str(model_config["name"]), **common)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_config["name"]), dtype=_dtype(str(model_config["dtype"])), **common
    ).to(device)
    target = TargetSpec(
        "exp12-pythia-block8",
        str(model_config["name"]),
        int(model_config["one_based_block"]),
        str(model_config["architecture"]),
    )
    wrapped = GenericActivationModel(model, tokenizer, target=target, device=device)
    if wrapped.resolved_model_revision != str(model_config["revision"]):
        raise RuntimeError("loaded Pythia revision differs from the frozen config")
    if wrapped.hidden_size != int(model_config["hidden_size"]):
        raise RuntimeError("loaded Pythia hidden size differs from the frozen config")
    return wrapped


def _cache_metadata_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".json")


def _validate_cache(
    path: Path, metadata: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    absolute = _range(config, "cache_absolute_range")
    expected = {
        "dataset_name": config["corpus"]["dataset_name"],
        "dataset_config": config["corpus"]["dataset_config"],
        "dataset_revision": config["corpus"]["dataset_revision"],
        "split": config["corpus"]["dataset_split"],
        "token_count": absolute[1] - absolute[0],
        "token_offset": absolute[0],
        "dtype": "uint16",
        "tokenizer": config["corpus"]["tokenizer"],
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("fresh confirmation token cache metadata drift")
    if not path.is_file() or path.stat().st_size != 2 * int(expected["token_count"]):
        raise RuntimeError("fresh confirmation token cache byte count drift")


def _read_sequences(
    token_cache: Path,
    *,
    cache_absolute_range: tuple[int, int],
    starts: Tensor,
    sequence_length: int,
) -> Tensor:
    cache_start, cache_stop = cache_absolute_range
    tokens = np.memmap(
        token_cache, mode="r", dtype=np.uint16, shape=(cache_stop - cache_start,)
    )
    rows = np.stack(
        [
            tokens[
                int(start) - cache_start : int(start) - cache_start + sequence_length
            ]
            for start in starts
        ]
    ).astype(np.int64, copy=False)
    return torch.from_numpy(rows)


def _first_aligned_starts(interval: tuple[int, int], *, sequence_length: int, count: int) -> Tensor:
    start, stop = interval
    if count <= 0 or start + count * sequence_length > stop:
        raise ValueError("frozen range is too short for the requested aligned sequences")
    return start + torch.arange(count, dtype=torch.int64) * sequence_length


def _random_nonoverlapping_starts(
    interval: tuple[int, int], *, sequence_length: int, count: int, seed: int
) -> Tensor:
    start, stop = interval
    blocks = (stop - start) // sequence_length
    if count <= 0 or count > blocks:
        raise ValueError("frozen range has too few nonoverlapping blocks")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    return start + torch.randperm(blocks, generator=generator)[:count] * sequence_length


@torch.inference_mode()
def prepare_shared_inputs(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    token_cache: Path,
    device: torch.device,
    local_files_only: bool,
) -> dict[str, Any]:
    """Single-writer corpus, calibration, and natural-text cache preparation."""

    contract = _require_contract(config, output_root)
    ready_path = output_root / "shared_ready.json"
    if ready_path.exists():
        ready = json.loads(ready_path.read_text())
        if ready.get("config_digest") != canonical_digest(config):
            raise RuntimeError("shared cache belongs to another config")
        for record in ready.get("artifacts", {}).values():
            path = Path(record["path"])
            if file_record(path) != record:
                raise RuntimeError("shared artifact changed after preparation")
        return ready
    metadata_path = _cache_metadata_path(token_cache)
    if token_cache.exists() != metadata_path.exists():
        raise RuntimeError("token cache and metadata must both exist or both be absent")
    if token_cache.exists():
        metadata = json.loads(metadata_path.read_text())
        _validate_cache(token_cache, metadata, config)
    else:
        from transformers import AutoTokenizer

        model = config["model"]
        tokenizer = AutoTokenizer.from_pretrained(
            model["name"], revision=model["revision"], local_files_only=local_files_only
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        absolute = _range(config, "cache_absolute_range")
        metadata = prepare_token_memmap(
            token_cache,
            tokenizer=tokenizer,
            token_count=absolute[1] - absolute[0],
            token_offset=absolute[0],
            dataset_name=config["corpus"]["dataset_name"],
            dataset_config=config["corpus"]["dataset_config"],
            split=config["corpus"]["dataset_split"],
            dataset_revision=config["corpus"]["dataset_revision"],
        )
        _validate_cache(token_cache, metadata, config)
    lm = load_lm(config, device, local_files_only=local_files_only)
    sequence_length = int(config["training"]["sequence_length"])
    cache_range = _range(config, "cache_absolute_range")
    calibration_count = math.ceil(int(config["calibration"]["activation_tokens"]) / sequence_length)
    calibration_starts = _first_aligned_starts(
        _range(config, "calibration_absolute_range"),
        sequence_length=sequence_length,
        count=calibration_count,
    )
    calibration_ids = _read_sequences(
        token_cache,
        cache_absolute_range=cache_range,
        starts=calibration_starts,
        sequence_length=sequence_length,
    )
    calibration_chunks = []
    batch_sequences = int(config["training"]["sequences_per_batch"])
    for ids in calibration_ids.split(batch_sequences):
        calibration_chunks.append(lm.activations(ids).cpu())
    activations = torch.cat(calibration_chunks).flatten(0, 1)[: int(config["calibration"]["activation_tokens"])].to(device)
    stats = estimate_activation_stats(activations)
    normalized = stats.normalize(activations)
    group_size = int(config["sae"]["group_size"])
    ridge_tokens = int(config["sae"]["ridge_calibration_groups"]) * group_size
    groups = normalized[:ridge_tokens].reshape(-1, group_size, lm.hidden_size)
    ridge_values = [
        calibrate_ridge(group, float(config["sae"]["ridge_dof_fraction"]))
        for group in groups
    ]
    calibration_path = output_root / "shared/calibration.pt"
    atomic_torch(
        calibration_path,
        {
            "schema_version": 1,
            "config_digest": canonical_digest(config),
            "model_name": config["model"]["name"],
            "model_revision": config["model"]["revision"],
            "one_based_block": config["model"]["one_based_block"],
            "hidden_size": lm.hidden_size,
            "activation_stats": stats.state_dict(),
            "ridge": float(np.median(ridge_values)),
            "ridge_values": ridge_values,
            "absolute_range": list(_range(config, "calibration_absolute_range")),
            "starts": calibration_starts,
            "activation_tokens": len(activations),
            "token_cache_sha256": file_sha256(token_cache),
            "stop_rule_contract_sha256": file_sha256(output_root / "stop_rule_contract.json"),
        },
    )
    del activations, normalized, groups, calibration_chunks

    maturity_count = math.ceil(int(config["maturity_evaluation"]["activation_tokens"]) / sequence_length)
    maturity_starts = _first_aligned_starts(
        _range(config, "maturity_evaluation_absolute_range"),
        sequence_length=sequence_length,
        count=maturity_count,
    )
    maturity_ids = _read_sequences(
        token_cache,
        cache_absolute_range=cache_range,
        starts=maturity_starts,
        sequence_length=sequence_length,
    )
    maturity_activations = []
    for ids in maturity_ids.split(batch_sequences):
        maturity_activations.append(stats.normalize(lm.activations(ids)).cpu().half())
    maturity_path = output_root / "shared/maturity_evaluation.pt"
    atomic_torch(
        maturity_path,
        {
            "schema_version": 1,
            "config_digest": canonical_digest(config),
            "absolute_range": list(_range(config, "maturity_evaluation_absolute_range")),
            "starts": maturity_starts,
            "input_ids": maturity_ids,
            "activations": torch.cat(maturity_activations)[:maturity_count],
            "calibration_sha256": file_sha256(calibration_path),
            "token_cache_sha256": file_sha256(token_cache),
        },
    )

    frozen = config["frozen_network_evaluation"]
    frozen_starts = _random_nonoverlapping_starts(
        _range(config, "frozen_network_absolute_range"),
        sequence_length=int(frozen["sequence_length"]),
        count=int(frozen["sequences"]),
        seed=int(frozen["sampling_seed"]),
    )
    frozen_ids = _read_sequences(
        token_cache,
        cache_absolute_range=cache_range,
        starts=frozen_starts,
        sequence_length=int(frozen["sequence_length"]),
    )
    frozen_path = output_root / "shared/frozen_network_inputs.pt"
    atomic_torch(
        frozen_path,
        {
            "schema_version": 1,
            "config_digest": canonical_digest(config),
            "absolute_range": list(_range(config, "frozen_network_absolute_range")),
            "starts": frozen_starts,
            "input_ids": frozen_ids,
            "sampling_algorithm": frozen["sampling_algorithm"],
            "sampling_seed": frozen["sampling_seed"],
            "token_cache_sha256": file_sha256(token_cache),
        },
    )
    del lm, maturity_activations
    if device.type == "cuda":
        torch.cuda.empty_cache()
    ready = {
        "schema_version": 1,
        "complete": True,
        "single_writer": True,
        "config_digest": canonical_digest(config),
        "stop_rule_contract_sha256": file_sha256(output_root / "stop_rule_contract.json"),
        "token_cache_metadata": metadata,
        "artifacts": {
            "token_cache": file_record(token_cache),
            "token_cache_metadata": file_record(metadata_path),
            "calibration": file_record(calibration_path),
            "maturity_evaluation": file_record(maturity_path),
            "frozen_network_inputs": file_record(frozen_path),
        },
        "rule_digest": contract["rule_digest"],
    }
    atomic_json(ready_path, ready)
    return ready


def _validate_shared(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    _require_contract(config, output_root)
    path = output_root / "shared_ready.json"
    if not path.is_file():
        raise RuntimeError("shared single-writer preparation is incomplete")
    ready = json.loads(path.read_text())
    if ready.get("config_digest") != canonical_digest(config):
        raise RuntimeError("shared inputs belong to another config")
    for name, record in ready.get("artifacts", {}).items():
        if file_record(Path(record["path"])) != record:
            raise RuntimeError(f"shared artifact changed after preparation: {name}")
    return ready


def _matched_pair_specs(config: Mapping[str, Any], pair_seed: int) -> list[SAETrainSpec]:
    k = int(config["sae"]["target_l0"])
    return [
        SAETrainSpec(f"mse_s{pair_seed}", "mse", pair_seed, k),
        SAETrainSpec(
            f"dpsae_s{pair_seed}",
            "dpsae",
            pair_seed,
            k,
            decoder_weight=float(config["sae"]["decoder_weight"]),
        ),
    ]


def pair_specs(config: Mapping[str, Any], pair_seed: int) -> list[SAETrainSpec]:
    seeds = [int(value) for value in config["confirmation"]["pair_seeds"]]
    if pair_seed not in seeds:
        raise ValueError("pair seed is outside the frozen confirmation set")
    return _matched_pair_specs(config, pair_seed)


def _learning_rate(config: Mapping[str, Any], step: int) -> float:
    training = config["training"]
    tokens_per_batch = int(training["sequence_length"]) * int(training["sequences_per_batch"])
    horizon_steps = math.ceil(int(training["scheduler_horizon_tokens"]) / tokens_per_batch)
    progress = min(step / horizon_steps, 1.0)
    warmup = float(config["sae"]["warmup_fraction"])
    if progress < warmup:
        scale = progress / warmup
    else:
        scale = 0.5 * (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup)))
    return float(config["sae"]["learning_rate"]) * scale


def build_timing_smoke_gate(
    config: Mapping[str, Any], measurements: Mapping[str, Any]
) -> dict[str, Any]:
    """Project only runtime and memory; never expose smoke model outcomes."""

    smoke = config["timing_smoke"]
    sequence_length = int(config["training"]["sequence_length"])
    batch_size = int(config["training"]["sequences_per_batch"])
    tokens_per_batch = sequence_length * batch_size
    smoke_realized = realized_tokens(int(smoke["requested_training_tokens"]), tokens_per_batch)
    full_realized = realized_tokens(int(config["training"]["maximum_tokens"]), tokens_per_batch)
    observed_realized = int(measurements["smoke_realized_training_tokens"])
    if observed_realized != smoke_realized:
        raise ValueError("timing smoke did not execute its frozen realized-token budget")
    timing_fields = (
        "cache_wall_seconds",
        "setup_wall_seconds",
        "training_wall_seconds",
        "smoke_wall_seconds",
        "peak_reserved_gpu_gib",
    )
    values = {name: float(measurements[name]) for name in timing_fields}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("timing-smoke measurements must be finite and nonnegative")
    smoke_cache = smoke["cache_absolute_range"]
    full_cache = config["corpus"]["cache_absolute_range"]
    smoke_streamed_tokens = int(smoke_cache[1])
    full_streamed_tokens = int(full_cache[1])
    full_cache_reused = bool(measurements["full_confirmation_cache_reused"])
    cache_projection = (
        0.0
        if full_cache_reused
        else values["cache_wall_seconds"] * full_streamed_tokens / smoke_streamed_tokens
    )
    training_projection = (
        values["training_wall_seconds"] * full_realized / smoke_realized
    )
    setup_projection = values["setup_wall_seconds"]
    raw_projection = cache_projection + setup_projection + training_projection
    headroom = float(smoke["projection_headroom_multiplier"])
    projected_seconds = raw_projection * headroom
    runtime_limit_seconds = (
        float(smoke["maximum_projected_confirmation_wall_hours"]) * 3600
    )
    smoke_limit_seconds = float(smoke["maximum_smoke_wall_minutes"]) * 60
    memory_limit = float(smoke["maximum_peak_reserved_gib"])
    gates = {
        "projected_confirmation_wall_time": projected_seconds
        <= runtime_limit_seconds,
        "smoke_wall_time": values["smoke_wall_seconds"] <= smoke_limit_seconds,
        "peak_reserved_gpu_memory": values["peak_reserved_gpu_gib"] <= memory_limit,
    }
    return {
        "reportable": False,
        "pair_seed": int(smoke["pair_seed"]),
        "smoke_requested_training_tokens": int(smoke["requested_training_tokens"]),
        "smoke_realized_training_tokens": smoke_realized,
        "full_requested_training_tokens": int(config["training"]["maximum_tokens"]),
        "full_realized_training_tokens": full_realized,
        "full_confirmation_cache_reused": full_cache_reused,
        "measurements": values,
        "projection": {
            "cache_seconds": cache_projection,
            "smoke_cache_streamed_tokens": smoke_streamed_tokens,
            "full_cache_streamed_tokens": full_streamed_tokens,
            "setup_seconds": setup_projection,
            "training_seconds": training_projection,
            "raw_seconds": raw_projection,
            "headroom_multiplier": headroom,
            "projected_seconds": projected_seconds,
            "projected_hours": projected_seconds / 3600,
            "maximum_hours": runtime_limit_seconds / 3600,
        },
        "limits": {
            "maximum_smoke_wall_minutes": smoke_limit_seconds / 60,
            "maximum_peak_reserved_gib": memory_limit,
        },
        "gates": gates,
        "passed": all(gates.values()),
        "model_quality_metrics_retained": False,
        "concept_outcomes_opened_by_this_stage": False,
    }


def _validate_smoke_cache(
    path: Path, metadata: Mapping[str, Any], config: Mapping[str, Any]
) -> None:
    smoke_range = config["timing_smoke"]["cache_absolute_range"]
    expected = {
        "dataset_name": config["corpus"]["dataset_name"],
        "dataset_config": config["corpus"]["dataset_config"],
        "dataset_revision": config["corpus"]["dataset_revision"],
        "split": config["corpus"]["dataset_split"],
        "token_count": int(smoke_range[1]) - int(smoke_range[0]),
        "token_offset": int(smoke_range[0]),
        "dtype": "uint16",
        "tokenizer": config["corpus"]["tokenizer"],
    }
    if any(metadata.get(key) != value for key, value in expected.items()):
        raise RuntimeError("nonreport timing-smoke token cache metadata drift")
    if not path.is_file() or path.stat().st_size != 2 * int(expected["token_count"]):
        raise RuntimeError("nonreport timing-smoke token cache byte count drift")


def run_timing_smoke(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    token_cache: Path,
    device: torch.device,
    local_files_only: bool,
) -> dict[str, Any]:
    """Run a nonreport 2M-token pair and block an unsafe full confirmation."""

    validate_config(config, require_frozen=True)
    _require_contract(config, output_root)
    smoke = config["timing_smoke"]
    smoke_root = output_root / str(smoke["artifact_subdirectory"])
    gate_path = smoke_root / "timing_smoke_gate.json"
    if gate_path.is_file():
        gate = json.loads(gate_path.read_text())
        if (
            gate.get("complete") is not True
            or gate.get("config_digest") != canonical_digest(config)
            or gate.get("reportable") is not False
        ):
            raise RuntimeError("timing-smoke gate belongs to another contract")
        if set(gate.get("artifacts", {})) != {
            "token_cache",
            "token_cache_metadata",
            "cache_timing",
        }:
            raise RuntimeError("timing-smoke gate lacks its exact artifact set")
        for record in gate["artifacts"].values():
            if file_record(Path(record["path"])) != record:
                raise RuntimeError("timing-smoke artifact changed")
        if gate.get("passed") is not True:
            raise RuntimeError("retained timing-smoke gate failed")
        return gate

    resources = _resource_guard(config, output_root, device)
    smoke_started = time.monotonic()
    smoke_deadline = smoke_started + float(smoke["maximum_smoke_wall_minutes"]) * 60
    full_metadata_path = _cache_metadata_path(token_cache)
    if token_cache.exists() != full_metadata_path.exists():
        raise RuntimeError("full confirmation token cache is only partially present")
    full_cache_reused = token_cache.is_file()
    if full_cache_reused:
        _validate_cache(token_cache, json.loads(full_metadata_path.read_text()), config)

    from transformers import AutoTokenizer

    model = config["model"]
    tokenizer = AutoTokenizer.from_pretrained(
        model["name"],
        revision=model["revision"],
        local_files_only=local_files_only,
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    smoke_cache = smoke_root / "tokens.bin"
    smoke_metadata_path = _cache_metadata_path(smoke_cache)
    cache_timing_path = smoke_root / "cache_timing.json"
    if smoke_cache.exists() != smoke_metadata_path.exists():
        raise RuntimeError("timing-smoke token cache is only partially present")
    if smoke_cache.is_file():
        metadata = json.loads(smoke_metadata_path.read_text())
        _validate_smoke_cache(smoke_cache, metadata, config)
        if not cache_timing_path.is_file():
            raise RuntimeError("timing-smoke cache exists without its timing record")
        cache_timing = json.loads(cache_timing_path.read_text())
        if (
            cache_timing.get("config_digest") != canonical_digest(config)
            or file_record(smoke_cache) != cache_timing.get("token_cache")
            or file_record(smoke_metadata_path) != cache_timing.get("metadata")
        ):
            raise RuntimeError("timing-smoke cache timing record changed")
        cache_wall_seconds = float(cache_timing["wall_seconds"])
    else:
        cache_started = time.monotonic()
        smoke_range = smoke["cache_absolute_range"]
        metadata = prepare_token_memmap(
            smoke_cache,
            tokenizer=tokenizer,
            token_count=int(smoke_range[1]) - int(smoke_range[0]),
            token_offset=int(smoke_range[0]),
            dataset_name=config["corpus"]["dataset_name"],
            dataset_config=config["corpus"]["dataset_config"],
            split=config["corpus"]["dataset_split"],
            dataset_revision=config["corpus"]["dataset_revision"],
        )
        cache_wall_seconds = time.monotonic() - cache_started
        _validate_smoke_cache(smoke_cache, metadata, config)
        atomic_json(
            cache_timing_path,
            {
                "schema_version": 1,
                "config_digest": canonical_digest(config),
                "wall_seconds": cache_wall_seconds,
                "token_cache": file_record(smoke_cache),
                "metadata": file_record(smoke_metadata_path),
            },
        )
    if time.monotonic() >= smoke_deadline:
        raise TimeoutError("timing smoke exceeded its frozen wall-time ceiling")

    setup_started = time.monotonic()
    lm = load_lm(config, device, local_files_only=local_files_only)
    sequence_length = int(config["training"]["sequence_length"])
    calibration_range = tuple(int(value) for value in smoke["calibration_absolute_range"])
    calibration_count = math.ceil(
        int(config["calibration"]["activation_tokens"]) / sequence_length
    )
    calibration_starts = _first_aligned_starts(
        calibration_range,
        sequence_length=sequence_length,
        count=calibration_count,
    )
    cache_range = tuple(int(value) for value in smoke["cache_absolute_range"])
    calibration_ids = _read_sequences(
        smoke_cache,
        cache_absolute_range=cache_range,
        starts=calibration_starts,
        sequence_length=sequence_length,
    )
    calibration_chunks = [
        lm.activations(ids).cpu()
        for ids in calibration_ids.split(int(config["training"]["sequences_per_batch"]))
    ]
    activations = torch.cat(calibration_chunks).flatten(0, 1)[
        : int(config["calibration"]["activation_tokens"])
    ].to(device)
    stats = estimate_activation_stats(activations)
    normalized = stats.normalize(activations)
    group_size = int(config["sae"]["group_size"])
    ridge_tokens = int(config["sae"]["ridge_calibration_groups"]) * group_size
    groups = normalized[:ridge_tokens].reshape(-1, group_size, lm.hidden_size)
    ridge = float(
        np.median(
            [
                calibrate_ridge(group, float(config["sae"]["ridge_dof_fraction"]))
                for group in groups
            ]
        )
    )
    smoke_seed = int(smoke["pair_seed"])
    fleet = TrainingFleet(
        _matched_pair_specs(config, smoke_seed),
        input_dim=int(config["model"]["hidden_size"]),
        dictionary_size=int(config["sae"]["dictionary_size"]),
        learning_rate=float(config["sae"]["learning_rate"]),
        device=device,
        aux_weight=float(config["sae"]["aux_weight"]),
        dead_after_steps=int(config["sae"]["dead_after_steps"]),
        aux_k=int(config["sae"]["aux_k"]),
        sparsity_mode=str(config["sae"]["sparsity_mode"]),
    )
    stream = SequentialTokenStream(
        smoke_cache,
        cache_absolute_range=cache_range,
        training_absolute_range=tuple(
            int(value) for value in smoke["training_absolute_range"]
        ),
        sequence_length=sequence_length,
        batch_size=int(config["training"]["sequences_per_batch"]),
        allow_reuse=False,
    )
    randomness = stage_seeds(smoke_seed, "exp12_nonreport_timing_smoke", replicate=0)
    del activations, normalized, groups, calibration_chunks
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    setup_wall_seconds = time.monotonic() - setup_started

    tokens_per_batch = sequence_length * int(config["training"]["sequences_per_batch"])
    smoke_realized = realized_tokens(int(smoke["requested_training_tokens"]), tokens_per_batch)
    step = 0
    tokens_seen = 0
    training_started = time.monotonic()
    while tokens_seen < smoke_realized:
        if step % int(config["training"]["log_every_steps"]) == 0:
            if time.monotonic() >= smoke_deadline:
                raise TimeoutError("timing smoke exceeded its frozen wall-time ceiling")
        step += 1
        fleet.set_learning_rate(_learning_rate(config, step))
        ids, _ = stream.batch()
        activation = stats.normalize(lm.activations(ids)).flatten(0, 1)
        fleet.train_batch(
            activation,
            step=step,
            ridge=ridge,
            group_size=group_size,
            probes=int(config["sae"]["decoder_probes"]),
            probe_seed=probe_seed_for_step(randomness, step - 1),
        )
        tokens_seen += len(activation)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    training_wall_seconds = time.monotonic() - training_started
    smoke_wall_seconds = (
        cache_wall_seconds + setup_wall_seconds + training_wall_seconds
    )
    peak_reserved = torch.cuda.max_memory_reserved(device) / GIB
    resources["peak_reserved_gpu_gib"] = peak_reserved
    measurements = {
        "cache_wall_seconds": cache_wall_seconds,
        "setup_wall_seconds": setup_wall_seconds,
        "training_wall_seconds": training_wall_seconds,
        "smoke_wall_seconds": smoke_wall_seconds,
        "peak_reserved_gpu_gib": peak_reserved,
        "smoke_realized_training_tokens": tokens_seen,
        "full_confirmation_cache_reused": full_cache_reused,
    }
    gate = {
        "schema_version": 1,
        "complete": True,
        "experiment": "exp12_nonreport_timing_smoke",
        "config_digest": canonical_digest(config),
        **build_timing_smoke_gate(config, measurements),
        "resources": resources,
        "artifacts": {
            "token_cache": file_record(smoke_cache),
            "token_cache_metadata": file_record(smoke_metadata_path),
            "cache_timing": file_record(cache_timing_path),
        },
    }
    atomic_json(gate_path, gate)
    del fleet, lm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    if gate["passed"] is not True:
        raise RuntimeError(f"timing-smoke gate failed: {gate['gates']}")
    return gate


def _require_timing_smoke_gate(
    config: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    smoke = config["timing_smoke"]
    path = (
        output_root
        / str(smoke["artifact_subdirectory"])
        / "timing_smoke_gate.json"
    )
    if not path.is_file():
        raise RuntimeError("run the nonreport timing smoke before full confirmation work")
    gate = json.loads(path.read_text())
    if (
        gate.get("complete") is not True
        or gate.get("passed") is not True
        or gate.get("reportable") is not False
        or gate.get("config_digest") != canonical_digest(config)
        or int(gate.get("pair_seed", -1)) != int(smoke["pair_seed"])
    ):
        raise RuntimeError("nonreport timing-smoke gate changed or did not pass")
    if set(gate.get("artifacts", {})) != {
        "token_cache",
        "token_cache_metadata",
        "cache_timing",
    }:
        raise RuntimeError("nonreport timing-smoke gate lacks its exact artifact set")
    for record in gate["artifacts"].values():
        if file_record(Path(record["path"])) != record:
            raise RuntimeError("nonreport timing-smoke artifact changed")
    return gate


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")


def _trim_jsonl(path: Path, maximum_step: int) -> None:
    if not path.exists():
        return
    records = [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip() and int(json.loads(line)["step"]) <= maximum_step
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records))
    temporary.replace(path)


@torch.inference_mode()
def evaluate_loaded_pair(
    *,
    config: Mapping[str, Any],
    fleet: TrainingFleet,
    evaluation_cache: Mapping[str, Any],
    ridge: float,
    step: int,
    device: torch.device,
) -> dict[str, Any]:
    """Evaluate only the four allowed concept-blind maturity diagnostics."""

    activations = evaluation_cache["activations"].float()
    sequence_count, sequence_length, width = activations.shape
    group_size = int(config["maturity_evaluation"]["exact_decoder"]["group_size"])
    if sequence_count * sequence_length % group_size:
        raise ValueError("maturity activation count does not divide exact decoder groups")
    models: dict[str, Any] = {}
    per_sequence: dict[str, Any] = {}
    name_by_method = {spec.method: spec.name for spec in fleet.specs}
    for method in METHODS:
        model = fleet.models[name_by_method[method]].eval()
        reconstruction_chunks, token_l0_chunks = [], []
        for batch in activations.flatten(0, 1).split(4096):
            reconstruction, code = model(batch.to(device), use_threshold=True)
            reconstruction_chunks.append(reconstruction.cpu())
            token_l0_chunks.append((code != 0).sum(1).cpu())
        reconstruction = torch.cat(reconstruction_chunks).reshape_as(activations)
        token_l0 = torch.cat(token_l0_chunks).reshape(sequence_count, sequence_length)
        grouped_original = activations.reshape(-1, group_size, width).to(device)
        grouped_reconstruction = reconstruction.reshape_as(grouped_original).to(device)
        numerator, denominator = exact_identity_decoder_statistics(
            grouped_original, grouped_reconstruction, ridge=ridge
        )
        numerator, denominator = numerator.cpu(), denominator.cpu()
        interval = bootstrap_ratio_interval(
            numerator,
            denominator,
            samples=int(config["maturity_evaluation"]["exact_decoder"]["bootstrap_samples"]),
            seed=int(config["maturity_evaluation"]["exact_decoder"]["bootstrap_seed"]),
        )
        sequence_sse = (reconstruction - activations).square().sum(dim=(1, 2))
        sequence_energy = activations.square().sum(dim=(1, 2))
        sequence_l0 = token_l0.sum(1)
        token_count = sequence_count * sequence_length
        dead = (step - model.last_active_step) >= int(config["sae"]["dead_after_steps"])
        models[method] = {
            "nmse": float(sequence_sse.sum() / sequence_energy.sum().clamp_min(1e-12)),
            "exact_decoder_distortion": float(numerator.sum() / denominator.sum().clamp_min(1e-12)),
            "exact_decoder_distortion_ci95": [interval["low"], interval["high"]],
            "inference_l0": float(sequence_l0.sum() / token_count),
            "dead_feature_fraction": float(dead.float().mean()),
            "dead_feature_count": int(dead.sum()),
            "dictionary_size": int(dead.numel()),
            "evaluation_tokens": token_count,
            "exact_group_count": len(numerator),
            "exact_numerator_by_group": numerator.tolist(),
            "exact_denominator_by_group": denominator.tolist(),
        }
        per_sequence[method] = [
            {
                "sequence": index,
                "absolute_start": int(evaluation_cache["starts"][index]),
                "nmse_numerator": float(sequence_sse[index]),
                "nmse_denominator": float(sequence_energy[index]),
                "l0_sum": int(sequence_l0[index]),
                "l0_count": sequence_length,
            }
            for index in range(sequence_count)
        ]
        del reconstruction, token_l0, grouped_original, grouped_reconstruction
    return {
        "models": models,
        "per_sequence": per_sequence,
        "evaluation_window": {
            "absolute_range": evaluation_cache["absolute_range"],
            "absolute_starts": evaluation_cache["starts"].tolist(),
            "sequence_length": sequence_length,
            "sequence_count": sequence_count,
        },
    }


def _snapshot_dir(pair_root: Path, requested_tokens: int) -> Path:
    return pair_root / "snapshots" / f"requested_{requested_tokens}"


def _save_snapshot(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    pair_root: Path,
    pair_seed: int,
    requested_tokens: int,
    step: int,
    tokens_seen: int,
    learning_rate: float,
    fleet: TrainingFleet,
    stream: SequentialTokenStream,
    exposure: Mapping[str, Any],
    maturity: Mapping[str, Any],
    calibration_sha256: str,
    cumulative_wall_seconds: float,
) -> dict[str, Any]:
    snapshot = _snapshot_dir(pair_root, requested_tokens)
    partial = snapshot.with_name(f".{snapshot.name}.partial")
    if snapshot.exists():
        manifest_path = snapshot / "manifest.json"
        if not manifest_path.is_file():
            raise RuntimeError("completed maturity snapshot exists without a manifest")
        manifest = json.loads(manifest_path.read_text())
        if (
            manifest.get("complete") is not True
            or manifest.get("config_digest") != canonical_digest(config)
            or int(manifest.get("pair_seed", -1)) != pair_seed
            or int(manifest.get("requested_snapshot_tokens", -1)) != requested_tokens
            or set(manifest.get("artifacts", {})) != {"state", "models", "maturity"}
        ):
            raise RuntimeError("completed maturity snapshot identity changed")
        for record in manifest["artifacts"].values():
            if file_record(Path(record["path"])) != record:
                raise RuntimeError("completed maturity snapshot changed")
        return manifest
    if partial.exists():
        if partial.is_dir():
            shutil.rmtree(partial)
        else:
            partial.unlink()
    partial.mkdir(parents=True)
    state_path = partial / "state.pt"
    models_path = partial / "models.pt"
    maturity_path = partial / "maturity.json"
    state = fleet.state_dict(step=step, tokens_seen=tokens_seen)
    state.update(
        {
            "schema_version": 1,
            "config_digest": canonical_digest(config),
            "stop_rule_contract_sha256": file_sha256(output_root / "stop_rule_contract.json"),
            "calibration_sha256": calibration_sha256,
            "pair_seed": pair_seed,
            "requested_snapshot_tokens": requested_tokens,
            "realized_snapshot_tokens": tokens_seen,
            "learning_rate": learning_rate,
            "scheduler": config["training"]["scheduler"],
            "scheduler_horizon_tokens": config["training"]["scheduler_horizon_tokens"],
            "stream": stream.state_dict(),
            "corpus_exposure": dict(exposure),
        }
    )
    atomic_torch(state_path, state)
    atomic_torch(models_path, fleet.export_models())
    maturity_payload = {
        "schema_version": 1,
        "complete": True,
        "concept_blind": True,
        "config_digest": canonical_digest(config),
        "pair_seed": pair_seed,
        "requested_snapshot_tokens": requested_tokens,
        "realized_snapshot_tokens": tokens_seen,
        "step": step,
        "learning_rate": learning_rate,
        "cumulative_training_wall_seconds": cumulative_wall_seconds,
        "written_at_unix_seconds": time.time(),
        "corpus_exposure": dict(exposure),
        **maturity,
    }
    atomic_json(maturity_path, maturity_payload)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "config_digest": canonical_digest(config),
        "pair_seed": pair_seed,
        "requested_snapshot_tokens": requested_tokens,
        "realized_snapshot_tokens": tokens_seen,
        "optimizer_scheduler_state_retained": True,
        "corpus_exposure": dict(exposure),
        "artifacts": {
            "state": promoted_file_record(state_path, snapshot / "state.pt"),
            "models": promoted_file_record(models_path, snapshot / "models.pt"),
            "maturity": promoted_file_record(maturity_path, snapshot / "maturity.json"),
        },
    }
    atomic_json(partial / "manifest.json", manifest)
    partial.replace(snapshot)
    return manifest


def train_pair(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    token_cache: Path,
    pair_seed: int,
    device: torch.device,
    local_files_only: bool,
) -> dict[str, Any]:
    """Train one matched pair on one GPU and retain every frozen maturity snapshot."""

    validate_config(config, require_frozen=True)
    _check_abort_or_deadline(config, output_root)
    ready = _validate_shared(config, output_root)
    pair_root = output_root / "pairs" / f"seed_{pair_seed}"
    done_path = pair_root / "pair_done.json"
    if done_path.exists():
        done = json.loads(done_path.read_text())
        if done.get("config_digest") != canonical_digest(config):
            raise RuntimeError("completed pair belongs to another config")
        for record in done.get("snapshot_manifests", {}).values():
            if file_record(Path(record["path"])) != record:
                raise RuntimeError("completed pair snapshot manifest changed")
        return done
    calibration_path = Path(ready["artifacts"]["calibration"]["path"])
    maturity_cache_path = Path(ready["artifacts"]["maturity_evaluation"]["path"])
    calibration = torch.load(calibration_path, map_location="cpu", weights_only=False)
    evaluation_cache = torch.load(maturity_cache_path, map_location="cpu", weights_only=False)
    lm = load_lm(config, device, local_files_only=local_files_only)
    stats = ActivationStats.from_state_dict(calibration["activation_stats"], device)
    specs = pair_specs(config, pair_seed)
    fleet = TrainingFleet(
        specs,
        input_dim=int(config["model"]["hidden_size"]),
        dictionary_size=int(config["sae"]["dictionary_size"]),
        learning_rate=float(config["sae"]["learning_rate"]),
        device=device,
        aux_weight=float(config["sae"]["aux_weight"]),
        dead_after_steps=int(config["sae"]["dead_after_steps"]),
        aux_k=int(config["sae"]["aux_k"]),
        sparsity_mode=str(config["sae"]["sparsity_mode"]),
    )
    stream = SequentialTokenStream(
        token_cache,
        cache_absolute_range=_range(config, "cache_absolute_range"),
        training_absolute_range=_range(config, "training_absolute_range"),
        sequence_length=int(config["training"]["sequence_length"]),
        batch_size=int(config["training"]["sequences_per_batch"]),
        allow_reuse=bool(config["corpus"]["allow_training_cache_reuse"]),
    )
    randomness = stage_seeds(
        int(config["randomness"]["base_seed"]), "exp12_fresh_pair", replicate=pair_seed
    )
    latest_path = pair_root / "latest_state.pt"
    start_step, tokens_seen = 0, 0
    prior_wall_seconds = 0.0
    if latest_path.exists():
        state = torch.load(latest_path, map_location=device, weights_only=False)
        if (
            state.get("config_digest") != canonical_digest(config)
            or state.get("calibration_sha256") != file_sha256(calibration_path)
            or state.get("specs") != [asdict(spec) for spec in specs]
            or state.get("stop_rule_contract_sha256")
            != file_sha256(output_root / "stop_rule_contract.json")
        ):
            raise RuntimeError("pair resume checkpoint contract changed")
        start_step, tokens_seen = fleet.load_state_dict(state)
        stream.load_state_dict(state["stream"])
        prior_wall_seconds = float(state.get("cumulative_training_wall_seconds", 0.0))
    log_path = pair_root / "training.jsonl"
    _trim_jsonl(log_path, start_step)
    batch_tokens = int(config["training"]["sequence_length"]) * int(
        config["training"]["sequences_per_batch"]
    )
    maximum_realized = realized_tokens(int(config["training"]["maximum_tokens"]), batch_tokens)
    budgets = snapshot_budgets(config)
    pending = [budget for budget in budgets if not (_snapshot_dir(pair_root, budget) / "manifest.json").is_file()]
    checkpoint_interval = int(config["training"]["resume_checkpoint_tokens"])
    next_checkpoint = (tokens_seen // checkpoint_interval + 1) * checkpoint_interval
    exposure: Mapping[str, Any] = {
        "delivered_tokens": tokens_seen,
        "unique_corpus_exposure_tokens": min(tokens_seen, stream.usable_tokens),
        "reused_tokens": max(0, tokens_seen - stream.usable_tokens),
        "cache_epoch": max(0, (stream.total_sequences - 1) // stream.available_sequences),
        "cache_reuse_count": max(0, (stream.total_sequences - 1) // stream.available_sequences),
        "unique_absolute_interval": [
            stream.train_start,
            stream.train_start + min(tokens_seen, stream.usable_tokens),
        ],
    }
    started = time.time()
    step = start_step
    while tokens_seen < maximum_realized:
        if (step - start_step) % int(config["training"]["log_every_steps"]) == 0:
            _check_abort_or_deadline(config, output_root)
        step += 1
        learning_rate = _learning_rate(config, step)
        fleet.set_learning_rate(learning_rate)
        ids, exposure = stream.batch()
        activation = stats.normalize(lm.activations(ids)).flatten(0, 1)
        metrics = fleet.train_batch(
            activation,
            step=step,
            ridge=float(calibration["ridge"]),
            group_size=int(config["sae"]["group_size"]),
            probes=int(config["sae"]["decoder_probes"]),
            probe_seed=probe_seed_for_step(randomness, step - 1),
        )
        tokens_seen += len(activation)
        if (step - 1) % int(config["training"]["log_every_steps"]) == 0:
            _append_jsonl(
                log_path,
                {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "learning_rate": learning_rate,
                    "cumulative_training_wall_seconds": prior_wall_seconds
                    + time.time()
                    - started,
                    "corpus_exposure": dict(exposure),
                    "models": metrics,
                },
            )
        while pending and tokens_seen >= pending[0]:
            _check_abort_or_deadline(config, output_root)
            requested = pending.pop(0)
            maturity = evaluate_loaded_pair(
                config=config,
                fleet=fleet,
                evaluation_cache=evaluation_cache,
                ridge=float(calibration["ridge"]),
                step=step,
                device=device,
            )
            _save_snapshot(
                config=config,
                output_root=output_root,
                pair_root=pair_root,
                pair_seed=pair_seed,
                requested_tokens=requested,
                step=step,
                tokens_seen=tokens_seen,
                learning_rate=learning_rate,
                fleet=fleet,
                stream=stream,
                exposure=exposure,
                maturity=maturity,
                calibration_sha256=file_sha256(calibration_path),
                cumulative_wall_seconds=prior_wall_seconds + time.time() - started,
            )
        if tokens_seen >= next_checkpoint or tokens_seen >= maximum_realized:
            state = fleet.state_dict(step=step, tokens_seen=tokens_seen)
            state.update(
                {
                    "schema_version": 1,
                    "config_digest": canonical_digest(config),
                    "calibration_sha256": file_sha256(calibration_path),
                    "stop_rule_contract_sha256": file_sha256(
                        output_root / "stop_rule_contract.json"
                    ),
                    "stream": stream.state_dict(),
                    "randomness": asdict(randomness),
                    "learning_rate": learning_rate,
                    "corpus_exposure": dict(exposure),
                    "cumulative_training_wall_seconds": prior_wall_seconds
                    + time.time()
                    - started,
                }
            )
            atomic_torch(latest_path, state)
            next_checkpoint = (tokens_seen // checkpoint_interval + 1) * checkpoint_interval
    _check_abort_or_deadline(config, output_root)
    if pending:
        raise RuntimeError(f"training ended before maturity snapshots: {pending}")
    manifests = {
        str(budget): file_record(_snapshot_dir(pair_root, budget) / "manifest.json")
        for budget in budgets
    }
    done = {
        "schema_version": 1,
        "complete": True,
        "config_digest": canonical_digest(config),
        "pair_seed": pair_seed,
        "specs": [asdict(spec) for spec in specs],
        "maximum_requested_tokens": int(config["training"]["maximum_tokens"]),
        "maximum_realized_tokens": tokens_seen,
        "final_corpus_exposure": dict(exposure),
        "snapshot_manifests": manifests,
        "calibration_sha256": file_sha256(calibration_path),
        "shared_ready_sha256": file_sha256(output_root / "shared_ready.json"),
        "stop_rule_contract_sha256": file_sha256(output_root / "stop_rule_contract.json"),
        "cumulative_training_wall_seconds": prior_wall_seconds + time.time() - started,
    }
    atomic_json(done_path, done)
    return done


def _verify_file_record(record: Mapping[str, Any], label: str) -> Path:
    path = Path(str(record.get("path", "")))
    if not path.is_file() or file_record(path) != dict(record):
        raise RuntimeError(f"{label} changed or is incomplete")
    return path


def load_maturity_inputs(
    config: Mapping[str, Any], output_root: Path
) -> tuple[dict[int, dict[int, dict[str, Any]]], dict[str, Any]]:
    """Load and hash only the retained concept-blind maturity artifacts."""

    _validate_shared(config, output_root)
    digest = canonical_digest(config)
    snapshots: dict[int, dict[int, dict[str, Any]]] = {}
    provenance: dict[str, Any] = {}
    for seed in (int(value) for value in config["confirmation"]["pair_seeds"]):
        pair_root = output_root / "pairs" / f"seed_{seed}"
        done_path = pair_root / "pair_done.json"
        if not done_path.is_file():
            raise RuntimeError(f"fresh pair seed {seed} has not completed")
        done = json.loads(done_path.read_text())
        if (
            done.get("complete") is not True
            or done.get("config_digest") != digest
            or int(done.get("pair_seed", -1)) != seed
        ):
            raise RuntimeError(f"fresh pair seed {seed} completion contract changed")
        snapshots[seed] = {}
        pair_records: dict[str, Any] = {"pair_done": file_record(done_path), "snapshots": {}}
        expected = {str(value) for value in snapshot_budgets(config)}
        if set(done.get("snapshot_manifests", {})) != expected:
            raise RuntimeError(f"fresh pair seed {seed} has the wrong maturity snapshot set")
        for budget in snapshot_budgets(config):
            manifest_record = done["snapshot_manifests"][str(budget)]
            manifest_path = _verify_file_record(
                manifest_record, f"seed {seed}, budget {budget} manifest"
            )
            manifest = json.loads(manifest_path.read_text())
            if (
                manifest.get("complete") is not True
                or manifest.get("config_digest") != digest
                or int(manifest.get("pair_seed", -1)) != seed
                or int(manifest.get("requested_snapshot_tokens", -1)) != budget
                or manifest.get("optimizer_scheduler_state_retained") is not True
            ):
                raise RuntimeError(f"seed {seed}, budget {budget} manifest contract changed")
            for name, artifact in manifest.get("artifacts", {}).items():
                _verify_file_record(artifact, f"seed {seed}, budget {budget} {name}")
            maturity_record = manifest.get("artifacts", {}).get("maturity")
            if maturity_record is None:
                raise RuntimeError(f"seed {seed}, budget {budget} lacks maturity diagnostics")
            maturity_path = Path(maturity_record["path"])
            maturity = json.loads(maturity_path.read_text())
            if (
                maturity.get("complete") is not True
                or maturity.get("concept_blind") is not True
                or maturity.get("config_digest") != digest
                or int(maturity.get("pair_seed", -1)) != seed
                or int(maturity.get("requested_snapshot_tokens", -1)) != budget
            ):
                raise RuntimeError(f"seed {seed}, budget {budget} maturity contract changed")
            snapshots[seed][budget] = maturity
            pair_records["snapshots"][str(budget)] = {
                "manifest": dict(manifest_record),
                "maturity": dict(maturity_record),
            }
        provenance[str(seed)] = pair_records
    return snapshots, provenance


def decide_maturity(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    """Write the immutable common-checkpoint decision before any concept evaluation."""

    contract = _require_contract(config, output_root)
    snapshots, provenance = load_maturity_inputs(config, output_root)
    contract_path = output_root / "stop_rule_contract.json"
    decision = build_maturity_decision(
        config,
        snapshots,
        stop_rule_contract_sha256=file_sha256(contract_path),
    )
    decision.update(
        {
            "rule_digest": contract["rule_digest"],
            "maturity_inputs": provenance,
            "decision_written_before_concept_evaluation": True,
            "concept_outcomes_opened_by_this_stage": False,
        }
    )
    path = output_root / "maturity_stop_decision.json"
    if path.exists() and json.loads(path.read_text()) != decision:
        raise RuntimeError("maturity stop decision changed after it was written")
    atomic_json(path, decision)
    return decision


def _load_selected_decision(config: Mapping[str, Any], output_root: Path) -> dict[str, Any]:
    path = output_root / "maturity_stop_decision.json"
    if not path.is_file():
        raise RuntimeError("write the frozen maturity decision before selected-checkpoint evaluation")
    decision = json.loads(path.read_text())
    if (
        decision.get("complete") is not True
        or decision.get("concept_blind") is not True
        or decision.get("config_digest") != canonical_digest(config)
        or decision.get("decision_written_before_concept_evaluation") is not True
    ):
        raise RuntimeError("maturity stop decision contract changed")
    if not decision.get("common_checkpoint_selected"):
        raise RuntimeError("the frozen stop rule selected no common maturity checkpoint")
    selected = int(decision["selected_requested_snapshot_tokens"])
    if selected not in snapshot_budgets(config):
        raise RuntimeError("maturity decision selected a checkpoint outside the frozen grid")
    return decision


def run_selected_decoder_evaluation(
    config: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    """Seal the selected snapshots' already-computed exact decoder statistics."""

    decision = _load_selected_decision(config, output_root)
    snapshots, provenance = load_maturity_inputs(config, output_root)
    if decision.get("maturity_inputs") != provenance:
        raise RuntimeError("maturity artifacts changed after the common-checkpoint decision")
    selected = int(decision["selected_requested_snapshot_tokens"])
    rows: list[dict[str, Any]] = []
    for seed in (int(value) for value in config["confirmation"]["pair_seeds"]):
        snapshot = snapshots[seed][selected]
        for method in METHODS:
            model = snapshot["models"][method]
            numerator = [float(value) for value in model["exact_numerator_by_group"]]
            denominator = [float(value) for value in model["exact_denominator_by_group"]]
            if not numerator or len(numerator) != len(denominator):
                raise RuntimeError("selected decoder evaluation lacks retained exact group statistics")
            pooled = sum(numerator) / max(sum(denominator), 1e-30)
            reported = float(model["exact_decoder_distortion"])
            if not math.isclose(pooled, reported, rel_tol=1e-6, abs_tol=1e-9):
                raise RuntimeError("selected exact decoder statistic does not match retained groups")
            rows.append(
                {
                    "pair_seed": seed,
                    "method": method,
                    "requested_snapshot_tokens": selected,
                    "realized_snapshot_tokens": int(snapshot["realized_snapshot_tokens"]),
                    "exact_decoder_distortion": reported,
                    "exact_decoder_distortion_ci95": model[
                        "exact_decoder_distortion_ci95"
                    ],
                    "evaluation_tokens": int(model["evaluation_tokens"]),
                    "group_size": int(
                        config["maturity_evaluation"]["exact_decoder"]["group_size"]
                    ),
                    "exact_group_count": len(numerator),
                    "exact_numerator_by_group": numerator,
                    "exact_denominator_by_group": denominator,
                    "snapshot_manifest": provenance[str(seed)]["snapshots"][str(selected)][
                        "manifest"
                    ],
                }
            )
    output = {
        "schema_version": 1,
        "complete": True,
        "experiment": "exp12_selected_exact_decoder_distortion",
        "config_digest": canonical_digest(config),
        "selected_requested_snapshot_tokens": selected,
        "common_checkpoint_across_all_pairs": True,
        "estimand": "exact_identity_decoder_disagreement_ratio_on_frozen_natural_text_activations",
        "rows": rows,
        "maturity_stop_decision": file_record(output_root / "maturity_stop_decision.json"),
        "concept_outcomes_opened_by_this_stage": False,
    }
    path = output_root / "selected_decoder_distortion.json"
    if path.exists() and json.loads(path.read_text()) != output:
        raise RuntimeError("selected decoder-distortion output changed")
    atomic_json(path, output)
    return output


def _pair_names(payloads: Mapping[str, Mapping[str, Any]]) -> dict[int, dict[str, str]]:
    names: dict[int, dict[str, str]] = {}
    for name, payload in payloads.items():
        spec = payload["spec"]
        seed, method = int(spec["seed"]), str(spec["method"])
        if method not in METHODS or method in names.setdefault(seed, {}):
            raise RuntimeError("selected model bundle contains duplicate or unknown pair members")
        names[seed][method] = name
    expected = sorted(int(value) for value in payloads_seeds(payloads))
    if sorted(names) != expected or any(set(pair) != set(METHODS) for pair in names.values()):
        raise RuntimeError("selected model bundle is not a complete paired fleet")
    return names


def payloads_seeds(payloads: Mapping[str, Mapping[str, Any]]) -> set[int]:
    return {int(payload["spec"]["seed"]) for payload in payloads.values()}


def _selected_model_payloads(
    config: Mapping[str, Any], output_root: Path, selected: int
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payloads: dict[str, dict[str, Any]] = {}
    records: dict[str, Any] = {}
    expected_seeds = {int(value) for value in config["confirmation"]["pair_seeds"]}
    decision = _load_selected_decision(config, output_root)
    for seed in sorted(expected_seeds):
        expected_manifest = decision["maturity_inputs"][str(seed)]["snapshots"][str(selected)][
            "manifest"
        ]
        manifest_path = _verify_file_record(
            expected_manifest, f"seed {seed} post-decision selected manifest"
        )
        manifest = json.loads(manifest_path.read_text())
        models_record = manifest["artifacts"]["models"]
        models_path = _verify_file_record(models_record, f"seed {seed} selected models")
        bundle = torch.load(models_path, map_location="cpu", weights_only=False)
        expected_names = {spec.name for spec in pair_specs(config, seed)}
        if set(bundle) != expected_names:
            raise RuntimeError(f"seed {seed} selected model bundle changed membership")
        for name, payload in bundle.items():
            if name in payloads or int(payload["spec"]["seed"]) != seed:
                raise RuntimeError("selected model payload seed or name changed")
            payloads[name] = payload
        records[str(seed)] = dict(models_record)
    if payloads_seeds(payloads) != expected_seeds:
        raise RuntimeError("selected model payloads do not cover the frozen seeds")
    _pair_names(payloads)
    return payloads, records


def _resource_guard(
    config: Mapping[str, Any], output_root: Path, device: torch.device
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    free_disk = shutil.disk_usage(output_root).free / GIB
    minimum_disk = float(config["runpod"]["minimum_free_disk_gib"])
    if free_disk < minimum_disk:
        raise RuntimeError(f"only {free_disk:.2f} GiB free; exp12 requires {minimum_disk:.2f} GiB")
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("fresh Pythia confirmation requires a CUDA A40 process")
    gpu_name = torch.cuda.get_device_name(device)
    if "A40" not in gpu_name:
        raise RuntimeError(f"fresh Pythia confirmation requires A40 GPUs, found {gpu_name!r}")
    fraction = float(config["runpod"]["gpu_memory_fraction"])
    torch.cuda.set_per_process_memory_fraction(fraction, device)
    torch.cuda.reset_peak_memory_stats(device)
    reserved = torch.cuda.memory_reserved(device) / GIB
    maximum_reserved = float(config["runpod"]["maximum_gpu_reserved_gib"])
    if reserved >= maximum_reserved:
        raise RuntimeError("GPU reserved-memory guard failed before exp12 work")
    return {
        "device": str(device),
        "gpu_name": gpu_name,
        "gpu_memory_fraction": fraction,
        "free_disk_gib_at_start": free_disk,
        "reserved_gpu_gib_at_start": reserved,
    }


def _finalize_resources(
    resources: dict[str, Any], config: Mapping[str, Any], device: torch.device
) -> None:
    peak = torch.cuda.max_memory_reserved(device) / GIB
    resources["peak_reserved_gpu_gib"] = peak
    if peak > float(config["runpod"]["maximum_gpu_reserved_gib"]):
        raise RuntimeError("exp12 exceeded its frozen peak GPU-memory guard")


def _next_token_statistics(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    log_probability = F.log_softmax(logits[:, :-1].float(), dim=-1)
    nll = -log_probability.gather(-1, targets[:, :, None]).squeeze(-1)
    correct = log_probability.argmax(-1).eq(targets)
    return log_probability, nll.sum(1), correct.sum(1)


@torch.inference_mode()
def run_selected_frozen_network_evaluation(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    device: torch.device,
    local_files_only: bool,
) -> dict[str, Any]:
    """Evaluate every selected pair by inserting reconstructions into frozen Pythia."""

    from experiments.exp08_language_evidence import aggregate_frozen_rows
    from experiments.exp09_frozen_network import (
        bootstrap_natural_pair,
        identity_gate,
        natural_retention_rows,
        pooled_kl_ratio,
    )

    decision = _load_selected_decision(config, output_root)
    selected = int(decision["selected_requested_snapshot_tokens"])
    ready = _validate_shared(config, output_root)
    payloads, model_records = _selected_model_payloads(config, output_root, selected)
    output_path = output_root / "selected_frozen_network.json"
    inputs = {
        "maturity_stop_decision": file_record(output_root / "maturity_stop_decision.json"),
        "calibration": ready["artifacts"]["calibration"],
        "frozen_network_inputs": ready["artifacts"]["frozen_network_inputs"],
        "selected_model_bundles": model_records,
    }
    if output_path.exists():
        existing = json.loads(output_path.read_text())
        if (
            existing.get("complete") is True
            and existing.get("config_digest") == canonical_digest(config)
            and existing.get("inputs") == inputs
        ):
            return existing
        raise RuntimeError("refusing to reuse stale selected frozen-network output")

    resources = _resource_guard(config, output_root, device)
    started = time.time()
    frozen_cache = torch.load(
        Path(ready["artifacts"]["frozen_network_inputs"]["path"]),
        map_location="cpu",
        weights_only=False,
    )
    calibration = torch.load(
        Path(ready["artifacts"]["calibration"]["path"]),
        map_location="cpu",
        weights_only=False,
    )
    if (
        frozen_cache.get("config_digest") != canonical_digest(config)
        or calibration.get("config_digest") != canonical_digest(config)
    ):
        raise RuntimeError("selected frozen-network inputs belong to another config")
    input_ids = frozen_cache["input_ids"]
    starts = frozen_cache["starts"].to(torch.int64)
    frozen = config["frozen_network_evaluation"]
    if input_ids.shape != (
        int(frozen["sequences"]),
        int(frozen["sequence_length"]),
    ):
        raise RuntimeError("frozen-network input cache shape changed")
    lm = load_lm(config, device, local_files_only=local_files_only)
    stats = ActivationStats.from_state_dict(calibration["activation_stats"], device)
    models = {
        name: load_sae(payload, input_dim=int(config["model"]["hidden_size"]), device=device)
        for name, payload in payloads.items()
    }
    common_rows: list[dict[str, Any]] = []
    model_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    identity_max, identity_total, identity_elements = 0.0, 0.0, 0

    def identity_replacement(hidden: Tensor) -> Tensor:
        return hidden

    def mean_replacement(hidden: Tensor) -> Tensor:
        return stats.mean.reshape(1, 1, -1).expand_as(hidden)

    batch_size = int(frozen["batch_sequences"])
    for start in range(0, len(input_ids), batch_size):
        ids = input_ids[start : start + batch_size]
        targets = ids[:, 1:].to(device)
        original_logits = lm.logits(ids)
        original_log_prob, original_nll, original_correct = _next_token_statistics(
            original_logits, targets
        )
        original_prob = original_log_prob.exp()
        original_top1 = original_log_prob.argmax(-1)
        identity_logits = lm.logits(ids, replacement=identity_replacement)
        identity_delta = (identity_logits - original_logits).abs()
        identity_flat = identity_delta.flatten(1)
        identity_max_by_sequence = identity_flat.max(1).values
        identity_mean_by_sequence = identity_flat.double().mean(1)
        identity_max = max(identity_max, float(identity_delta.max()))
        identity_total += float(identity_delta.double().sum())
        identity_elements += identity_delta.numel()
        if identity_max > float(frozen["identity_max_abs_logit_difference"]):
            raise RuntimeError("identity-hook maximum tolerance failed before SAE evaluation")
        identity_log_prob, identity_nll, identity_correct = _next_token_statistics(
            identity_logits, targets
        )
        identity_kl = (
            original_prob * (original_log_prob - identity_log_prob)
        ).sum(-1).sum(1)
        identity_agreement = identity_log_prob.argmax(-1).eq(original_top1).sum(1)
        del original_logits, identity_logits, identity_delta, identity_log_prob

        normalized_hidden = stats.normalize(lm.activations(ids))
        activation_energy = normalized_hidden.square().sum(dim=(1, 2))
        mean_logits = lm.logits(ids, replacement=mean_replacement)
        mean_log_prob, mean_nll, mean_correct = _next_token_statistics(mean_logits, targets)
        mean_kl = (original_prob * (original_log_prob - mean_log_prob)).sum(-1).sum(1)
        mean_agreement = mean_log_prob.argmax(-1).eq(original_top1).sum(1)
        del mean_logits, mean_log_prob
        tokens_per_sequence = int(targets.shape[1])
        for row in range(len(ids)):
            sequence_ids = ids[row].detach().cpu().contiguous()
            common_rows.append(
                {
                    "sequence": start + row,
                    "absolute_start": int(starts[start + row]),
                    "sequence_sha256": hashlib.sha256(sequence_ids.numpy().tobytes()).hexdigest(),
                    "tokens": tokens_per_sequence,
                    "original_nll": float(original_nll[row]),
                    "original_kl": 0.0,
                    "original_agreement": tokens_per_sequence,
                    "original_correct": int(original_correct[row]),
                    "identity_nll": float(identity_nll[row]),
                    "identity_kl": float(identity_kl[row]),
                    "identity_agreement": int(identity_agreement[row]),
                    "identity_correct": int(identity_correct[row]),
                    "identity_max_abs_logit_difference": float(identity_max_by_sequence[row]),
                    "identity_mean_abs_logit_difference": float(identity_mean_by_sequence[row]),
                    "identity_reconstruction_sse": 0.0,
                    "mean_nll": float(mean_nll[row]),
                    "mean_kl": float(mean_kl[row]),
                    "mean_agreement": int(mean_agreement[row]),
                    "mean_correct": int(mean_correct[row]),
                    "mean_reconstruction_sse": float(activation_energy[row]),
                    "activation_energy": float(activation_energy[row]),
                    "activation_tokens": int(normalized_hidden.shape[1]),
                }
            )
        for name, model in models.items():
            shape = normalized_hidden.shape
            reconstruction, code = model(
                normalized_hidden.reshape(-1, shape[-1]), use_threshold=True
            )
            reconstruction = reconstruction.reshape(shape)
            code = code.reshape(shape[0], shape[1], -1)

            def replacement(_hidden: Tensor, value: Tensor = reconstruction) -> Tensor:
                if _hidden.shape != value.shape:
                    raise RuntimeError("frozen reconstruction shape changed within a batch")
                return stats.denormalize(value)

            reconstructed_logits = lm.logits(ids, replacement=replacement)
            reconstructed_log_prob, reconstructed_nll, reconstructed_correct = (
                _next_token_statistics(reconstructed_logits, targets)
            )
            kl = (
                original_prob * (original_log_prob - reconstructed_log_prob)
            ).sum(-1).sum(1)
            agreement = reconstructed_log_prob.argmax(-1).eq(original_top1).sum(1)
            for row in range(len(ids)):
                model_rows[name].append(
                    {
                        "sequence": start + row,
                        "checkpoint": name,
                        "checkpoint_seed": int(payloads[name]["spec"]["seed"]),
                        "condition": str(payloads[name]["spec"]["method"]),
                        "reconstructed_nll": float(reconstructed_nll[row]),
                        "kl": float(kl[row]),
                        "agreement": int(agreement[row]),
                        "reconstructed_correct": int(reconstructed_correct[row]),
                        "reconstruction_sse": float(
                            (reconstruction[row] - normalized_hidden[row]).square().sum()
                        ),
                        "l0_count": float((code[row] != 0).sum()),
                    }
                )
            del reconstructed_logits, reconstructed_log_prob, reconstruction, code
        del original_prob, original_log_prob, normalized_hidden

    identity = identity_gate(
        maximum=identity_max,
        total=identity_total,
        elements=identity_elements,
        max_tolerance=float(frozen["identity_max_abs_logit_difference"]),
        mean_tolerance=float(frozen["identity_max_mean_abs_logit_difference"]),
    )
    reports = {
        name: {"spec": dict(payloads[name]["spec"]), **aggregate_frozen_rows(common_rows, rows)}
        for name, rows in model_rows.items()
    }
    pairs = []
    for seed, names in sorted(_pair_names(payloads).items()):
        mse_rows, dpsae_rows = model_rows[names["mse"]], model_rows[names["dpsae"]]
        mse, dpsae = reports[names["mse"]], reports[names["dpsae"]]
        row: dict[str, Any] = {
            "seed": seed,
            "bootstrap_seed": int(frozen["bootstrap_seed"]) + seed,
            "baseline": names["mse"],
            "candidate": names["dpsae"],
            "kl_ratio_dpsae_to_mse": pooled_kl_ratio(mse_rows, dpsae_rows),
            "kl_difference_dpsae_minus_mse": dpsae["original_to_reconstruction_kl"]
            - mse["original_to_reconstruction_kl"],
            "loss_recovered_difference_dpsae_minus_mse": dpsae["loss_recovered"]
            - mse["loss_recovered"],
            "cross_entropy_increase_difference_dpsae_minus_mse": dpsae[
                "cross_entropy_increase"
            ]
            - mse["cross_entropy_increase"],
            "top1_agreement_difference_dpsae_minus_mse": dpsae[
                "top1_agreement_with_original"
            ]
            - mse["top1_agreement_with_original"],
            "next_token_accuracy_difference_dpsae_minus_mse": dpsae[
                "reconstruction_next_token_accuracy"
            ]
            - mse["reconstruction_next_token_accuracy"],
            "activation_nmse_ratio_dpsae_to_mse": dpsae["activation_nmse"]
            / mse["activation_nmse"],
            "inference_l0_difference_dpsae_minus_mse": dpsae["inference_l0"]
            - mse["inference_l0"],
        }
        row.update(
            bootstrap_natural_pair(
                common_rows,
                mse_rows,
                dpsae_rows,
                samples=int(frozen["bootstrap_samples"]),
                seed=int(frozen["bootstrap_seed"]) + seed,
                quantiles=frozen["confidence_interval"],
            )
        )
        pairs.append(row)
    retained = natural_retention_rows(
        common_rows,
        model_rows,
        payloads,
        bootstrap_seed=int(frozen["bootstrap_seed"]),
    )
    _finalize_resources(resources, config, device)
    output = {
        "schema_version": 1,
        "complete": True,
        "experiment": "exp12_selected_frozen_pythia_natural_text",
        "config_digest": canonical_digest(config),
        "selected_requested_snapshot_tokens": selected,
        "identity_hook": identity,
        "mean_ablation": "replace normalized block-8 activation with zero, then denormalize",
        "models": reports,
        "paired": pairs,
        "per_sequence": {
            "schema": "frozen_network_sufficient_statistics_v1",
            "condition_rows": retained,
            "common": common_rows,
            "models": model_rows,
        },
        "protocol": dict(frozen),
        "inputs": inputs,
        "resources": resources,
        "wall_seconds": time.time() - started,
        "concept_outcomes_opened_by_this_stage": False,
    }
    atomic_json(output_path, output)
    return output


def authorize_concept_evaluation(
    config: Mapping[str, Any], output_root: Path
) -> dict[str, Any]:
    """Authorize later concept evaluation only after all outcome-blind work exists."""

    decision = _load_selected_decision(config, output_root)
    _require_timing_smoke_gate(config, output_root)
    required = {
        "stop_rule_contract": output_root / "stop_rule_contract.json",
        "timing_smoke_gate": output_root
        / str(config["timing_smoke"]["artifact_subdirectory"])
        / "timing_smoke_gate.json",
        "maturity_stop_decision": output_root / "maturity_stop_decision.json",
        "selected_decoder_distortion": output_root / "selected_decoder_distortion.json",
        "selected_frozen_network": output_root / "selected_frozen_network.json",
    }
    records = {name: file_record(path) for name, path in required.items()}
    decoder = json.loads(required["selected_decoder_distortion"].read_text())
    frozen = json.loads(required["selected_frozen_network"].read_text())
    selected = int(decision["selected_requested_snapshot_tokens"])
    for label, payload in (("decoder", decoder), ("frozen network", frozen)):
        if (
            payload.get("complete") is not True
            or payload.get("config_digest") != canonical_digest(config)
            or int(payload.get("selected_requested_snapshot_tokens", -1)) != selected
            or payload.get("concept_outcomes_opened_by_this_stage") is not False
        ):
            raise RuntimeError(f"{label} prerequisite changed before concept authorization")
    if frozen.get("identity_hook", {}).get("passed") is not True:
        raise RuntimeError("frozen-network identity hook did not pass before concept authorization")
    authorization = {
        "schema_version": 1,
        "complete": True,
        "authorized": True,
        "experiment": "exp12_concept_evaluation_authorization",
        "config_digest": canonical_digest(config),
        "selected_requested_snapshot_tokens": selected,
        "pair_seeds": config["confirmation"]["pair_seeds"],
        "excluded_pilot_checkpoint_id": config["pilot_gate"]["exclude_pilot_checkpoint_id"],
        "prerequisites": records,
        "maturity_inputs_were_concept_blind": True,
        "confirmatory_inference": config["concept_authorization"]["confirmatory_inference"],
        "concept_outcomes_opened_by_this_stage": False,
    }
    path = output_root / "concept_evaluation_authorization.json"
    if path.exists() and json.loads(path.read_text()) != authorization:
        raise RuntimeError("concept authorization changed after it was written")
    atomic_json(path, authorization)
    return authorization


def wait_for_files(
    paths: Sequence[Path],
    *,
    timeout_seconds: float,
    poll_seconds: float = 20,
    failure_paths: Sequence[Path] = (),
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        for failure_path in failure_paths:
            if failure_path.is_file():
                try:
                    failure = json.loads(failure_path.read_text())
                    reason = failure.get("error") or failure.get("reason") or "unknown failure"
                except (OSError, json.JSONDecodeError):
                    reason = "failure marker could not be decoded"
                raise RunAbortedError(f"{failure_path}: {reason}")
        missing = [path for path in paths if not path.is_file()]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError("timed out waiting for: " + ", ".join(str(path) for path in missing))
        time.sleep(min(poll_seconds, max(0.1, deadline - time.monotonic())))


def run_coordinator(
    *,
    config: Mapping[str, Any],
    output_root: Path,
    token_cache: Path,
    device: torch.device,
    local_files_only: bool,
    timeout_seconds: float,
) -> dict[str, Any]:
    validate_config(config, require_frozen=True)
    maximum_seconds = float(config["runpod"]["maximum_wall_hours"]) * 3600
    if not math.isfinite(timeout_seconds) or timeout_seconds <= 0 or timeout_seconds > maximum_seconds:
        raise ValueError("coordinator timeout exceeds the frozen wall-time contract")
    status_path, _ = _stage_paths(output_root, "coordinator")
    started = time.time()
    deadline = started + timeout_seconds
    if status_path.is_file():
        prior = json.loads(status_path.read_text())
        if prior.get("config_digest") != canonical_digest(config):
            raise RuntimeError("Exp12 coordinator status belongs to another config")
        if prior.get("started_at_unix_seconds") is not None:
            started = float(prior["started_at_unix_seconds"])
            deadline = float(prior["deadline_unix_seconds"])
    write_stage_status(
        config=config,
        output_root=output_root,
        stage="coordinator",
        state="running",
        extra={
            "started_at_unix_seconds": started,
            "deadline_unix_seconds": deadline,
        },
    )
    try:
        _check_abort_or_deadline(config, output_root)
        _require_timing_smoke_gate(config, output_root)
        _resource_guard(config, output_root, device)
        prepare_shared_inputs(
            config=config,
            output_root=output_root,
            token_cache=token_cache,
            device=device,
            local_files_only=local_files_only,
        )
        _check_abort_or_deadline(config, output_root)
        seeds = [int(seed) for seed in config["confirmation"]["pair_seeds"]]
        done_paths = [
            output_root / "pairs" / f"seed_{seed}" / "pair_done.json" for seed in seeds
        ]
        failure_paths = [
            output_root / "pairs" / f"seed_{seed}" / "pair_failed.json" for seed in seeds
        ] + [output_root / "abort_requested.json"]
        remaining = deadline - time.time()
        if remaining <= 0:
            raise TimeoutError("Exp12 exceeded its frozen wall-time ceiling")
        wait_for_files(
            done_paths,
            timeout_seconds=remaining,
            failure_paths=failure_paths,
        )
        decision = decide_maturity(config, output_root)
        if not decision["common_checkpoint_selected"]:
            write_stage_status(
                config=config,
                output_root=output_root,
                stage="coordinator",
                state="complete",
                extra={
                    "started_at_unix_seconds": started,
                    "deadline_unix_seconds": deadline,
                    "common_checkpoint_selected": False,
                    "result": file_record(output_root / "maturity_stop_decision.json"),
                },
            )
            return decision
        run_selected_decoder_evaluation(config, output_root)
        run_selected_frozen_network_evaluation(
            config=config,
            output_root=output_root,
            device=device,
            local_files_only=local_files_only,
        )
        authorization = authorize_concept_evaluation(config, output_root)
        write_stage_status(
            config=config,
            output_root=output_root,
            stage="coordinator",
            state="complete",
            extra={
                "started_at_unix_seconds": started,
                "deadline_unix_seconds": deadline,
                "common_checkpoint_selected": True,
                "result": file_record(output_root / "concept_evaluation_authorization.json"),
            },
        )
        return authorization
    except Exception as error:
        request_abort(config=config, output_root=output_root, reason=str(error))
        write_stage_failure(
            config=config,
            output_root=output_root,
            stage="coordinator",
            error=error,
        )
        raise


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "stage",
        choices=(
            "preflight",
            "timing-smoke",
            "prepare-shared",
            "wait-shared",
            "train-pair",
            "decide-maturity",
            "evaluate-decoder",
            "evaluate-frozen",
            "authorize-concept",
            "coordinator",
        ),
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--token-cache",
        type=Path,
        default=DEFAULT_OUTPUT / "shared/fineweb_pythia_tokens.bin",
    )
    parser.add_argument(
        "--pilot-report",
        type=Path,
        default=ROOT / "artifacts/exp10_concept_discovery/advancement_report.json",
    )
    parser.add_argument("--pair-seed", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--timeout-hours", type=float, default=8.0)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.stage == "preflight":
        try:
            write_preflight_contract(
                config_path=args.config,
                output_root=args.output_root,
                pilot_report_path=args.pilot_report,
            )
        except FreezeBlockedError as error:
            print(str(error), file=sys.stderr)
            return 2
        return 0
    config = load_config(args.config, require_frozen=True)
    timeout_seconds = timeout_seconds_for_run(config, args.timeout_hours)

    def terminate_for_timeout(signum: int, _frame: Any) -> None:
        raise RunAbortedError(f"Exp12 received termination signal {signum}")

    signal.signal(signal.SIGTERM, terminate_for_timeout)
    device = torch.device(args.device)
    if args.stage == "timing-smoke":
        write_stage_status(
            config=config,
            output_root=args.output_root,
            stage="timing-smoke",
            state="running",
        )
        try:
            run_timing_smoke(
                config=config,
                output_root=args.output_root,
                token_cache=args.token_cache,
                device=device,
                local_files_only=args.local_files_only,
            )
            write_stage_status(
                config=config,
                output_root=args.output_root,
                stage="timing-smoke",
                state="complete",
                extra={
                    "result": file_record(
                        args.output_root
                        / str(config["timing_smoke"]["artifact_subdirectory"])
                        / "timing_smoke_gate.json"
                    )
                },
            )
        except Exception as error:
            write_stage_failure(
                config=config,
                output_root=args.output_root,
                stage="timing-smoke",
                error=error,
            )
            raise
    elif args.stage == "prepare-shared":
        _resource_guard(config, args.output_root, device)
        prepare_shared_inputs(
            config=config,
            output_root=args.output_root,
            token_cache=args.token_cache,
            device=device,
            local_files_only=args.local_files_only,
        )
    elif args.stage == "wait-shared":
        if args.pair_seed is None:
            raise ValueError("wait-shared requires --pair-seed")
        write_stage_status(
            config=config,
            output_root=args.output_root,
            stage="wait-shared",
            pair_seed=args.pair_seed,
            state="running",
        )
        try:
            wait_for_files(
                [args.output_root / "shared_ready.json"],
                timeout_seconds=timeout_seconds,
                failure_paths=[
                    args.output_root / "coordinator_failed.json",
                    args.output_root / "abort_requested.json",
                    args.output_root
                    / str(config["timing_smoke"]["artifact_subdirectory"])
                    / "smoke_failed.json",
                ],
            )
            _check_abort_or_deadline(config, args.output_root)
            _validate_shared(config, args.output_root)
            write_stage_status(
                config=config,
                output_root=args.output_root,
                stage="wait-shared",
                pair_seed=args.pair_seed,
                state="complete",
            )
        except Exception as error:
            write_stage_failure(
                config=config,
                output_root=args.output_root,
                stage="wait-shared",
                pair_seed=args.pair_seed,
                error=error,
            )
            raise
    elif args.stage == "train-pair":
        if args.pair_seed is None:
            raise ValueError("train-pair requires --pair-seed")
        write_stage_status(
            config=config,
            output_root=args.output_root,
            stage="train-pair",
            pair_seed=args.pair_seed,
            state="running",
        )
        try:
            _check_abort_or_deadline(config, args.output_root)
            _resource_guard(config, args.output_root, device)
            train_pair(
                config=config,
                output_root=args.output_root,
                token_cache=args.token_cache,
                pair_seed=args.pair_seed,
                device=device,
                local_files_only=args.local_files_only,
            )
            write_stage_status(
                config=config,
                output_root=args.output_root,
                stage="train-pair",
                pair_seed=args.pair_seed,
                state="complete",
                extra={
                    "result": file_record(
                        args.output_root
                        / "pairs"
                        / f"seed_{args.pair_seed}"
                        / "pair_done.json"
                    )
                },
            )
        except Exception as error:
            write_stage_failure(
                config=config,
                output_root=args.output_root,
                stage="train-pair",
                pair_seed=args.pair_seed,
                error=error,
            )
            raise
    elif args.stage == "decide-maturity":
        decide_maturity(config, args.output_root)
    elif args.stage == "evaluate-decoder":
        run_selected_decoder_evaluation(config, args.output_root)
    elif args.stage == "evaluate-frozen":
        run_selected_frozen_network_evaluation(
            config=config,
            output_root=args.output_root,
            device=device,
            local_files_only=args.local_files_only,
        )
    elif args.stage == "authorize-concept":
        authorize_concept_evaluation(config, args.output_root)
    elif args.stage == "coordinator":
        run_coordinator(
            config=config,
            output_root=args.output_root,
            token_cache=args.token_cache,
            device=device,
            local_files_only=args.local_files_only,
            timeout_seconds=timeout_seconds,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
