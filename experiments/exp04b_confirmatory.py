#!/usr/bin/env python3
"""Confirmatory natural-text and static-baseline audit for Experiment 4b.

The driver deliberately keeps data preparation, static calibration, training,
and evaluation as independently resumable stages.  IOI execution is wired as
an explicit hook so its causal protocol can evolve without coupling it to the
natural-text runner.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch import Tensor

from dpsae.corpus import MemmapTokenBatcher, TokenRange, prepare_token_memmap
from dpsae.decoder_distance import (
    batched_ridge_predict,
    calibrate_ridge,
)
from dpsae.exp04b_natural_text import (
    apply_geometry_groups,
    bootstrap_paired_reduction_interval,
    bootstrap_ratio_interval,
    exact_decoder_sweep,
    geometry_group_indices,
)
from dpsae.exp04b_training import (
    confirmation_replicate_config,
    confirmation_specs,
    probe_seed_for_step,
    screen_specs,
    select_static_baselines,
    stage_seeds,
)
from dpsae.language_model import ActivationStats, GPT2ActivationModel
from dpsae.language_training import (
    SAETrainSpec,
    TrainingFleet,
    spectral_surrogate_operator,
)
from dpsae.mech_analysis import load_sae


ROOT = Path(__file__).resolve().parents[1]
STAGES = (
    "prepare-tail",
    "cache-natural",
    "calibrate-static",
    "baseline-screen",
    "baseline-confirm",
    "natural-evaluate",
    "ioi-confirm",
)


@dataclass(frozen=True)
class ExperimentPaths:
    output: Path
    tail_tokens: Path
    natural_selection: Path
    natural_test: Path
    static_calibration: Path
    baseline_selection: Path
    source_artifact: Path
    source_tokens: Path
    source_calibration: Path


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    config = json.loads(path.read_text())
    source_path = root / config["source_config"]
    config["source"] = json.loads(source_path.read_text())
    config["source_config_resolved"] = str(source_path.resolve())
    config["repository"] = repository_state(root)
    _validate_config(config)
    return config


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


def _validate_config(config: Mapping[str, Any]) -> None:
    source = config["source"]
    fresh = config["fresh_corpus"]
    if int(fresh["token_offset"]) < int(source["corpus"]["token_count"]):
        raise ValueError("fresh corpus overlaps the source corpus")
    token_count = int(fresh["token_count"])
    ranges = [TokenRange(*fresh[name]) for name in ("selection_range", "test_range")]
    if any(item.start < 0 or item.stop > token_count or item.size <= 0 for item in ranges):
        raise ValueError("fresh selection/test ranges must lie inside the tail shard")
    if ranges[0].stop > ranges[1].start:
        raise ValueError("fresh selection and test ranges overlap")
    sequence_length = int(source["training"]["sequence_length"])
    for key in ("activation_tokens", "exact_tokens"):
        if int(config["natural_text"][key]) % sequence_length:
            raise ValueError(f"natural_text.{key} must divide into full sequences")


def experiment_paths(config: Mapping[str, Any], *, root: Path = ROOT) -> ExperimentPaths:
    output = root / "artifacts" / str(config["experiment"])
    source = root / str(config["source_artifact"])
    return ExperimentPaths(
        output=output,
        tail_tokens=output / "fineweb_gpt2_tail_tokens.bin",
        natural_selection=output / "natural_selection.pt",
        natural_test=output / "natural_test.pt",
        static_calibration=output / "static_calibration.pt",
        baseline_selection=output / "baseline_selection.json",
        source_artifact=source,
        source_tokens=source / "fineweb_gpt2_tokens.bin",
        source_calibration=source / "calibration.pt",
    )


def stage_sequence(stage: str, *, fleet: str = "all") -> list[tuple[str, str | None]]:
    if stage not in (*STAGES, "all"):
        raise ValueError(f"unknown stage: {stage}")
    if fleet not in {"source", "baseline", "all"}:
        raise ValueError(f"unknown fleet: {fleet}")
    if stage == "natural-evaluate":
        fleets = ("source", "baseline") if fleet == "all" else (fleet,)
        return [(stage, item) for item in fleets]
    if stage != "all":
        return [(stage, None)]
    result = [(name, None) for name in STAGES[:5]]
    fleets = ("source", "baseline") if fleet == "all" else (fleet,)
    result.extend(("natural-evaluate", item) for item in fleets)
    return result


def load_lm(config: Mapping[str, Any], device: torch.device) -> GPT2ActivationModel:
    source = config["source"]
    return GPT2ActivationModel.from_pretrained(
        source["model_name"], layer=int(source["layer"]), device=device
    )


def _completed(path: Path) -> bool:
    return path.exists() and bool(json.loads(path.read_text()).get("complete"))


def prepare_tail(
    config: Mapping[str, Any], paths: ExperimentPaths, device: torch.device
) -> None:
    done = paths.output / "prepare_tail.json"
    metadata_path = paths.tail_tokens.with_suffix(paths.tail_tokens.suffix + ".json")
    partial_path = paths.tail_tokens.with_suffix(paths.tail_tokens.suffix + ".partial")
    partial_metadata = partial_path.with_suffix(partial_path.suffix + ".json")
    if paths.tail_tokens.exists() and not metadata_path.exists() and partial_metadata.exists():
        partial_metadata.replace(metadata_path)
    if metadata_path.exists() and not paths.tail_tokens.exists() and partial_path.exists():
        partial_path.replace(paths.tail_tokens)
    existing = (paths.tail_tokens.exists(), metadata_path.exists())
    if any(existing) and not all(existing):
        raise RuntimeError("tail shard is partial; move it aside before retrying")
    fresh = config["fresh_corpus"]
    if all(existing):
        metadata = json.loads(metadata_path.read_text())
        expected = {
            "token_count": int(fresh["token_count"]),
            "token_offset": int(fresh["token_offset"]),
        }
        if any(int(metadata.get(key, -1)) != value for key, value in expected.items()):
            raise RuntimeError("existing immutable tail shard has different coordinates")
        if paths.tail_tokens.stat().st_size != 2 * expected["token_count"]:
            raise RuntimeError("existing immutable tail shard has the wrong byte size")
    else:
        lm = load_lm(config, device)
        source_corpus = config["source"]["corpus"]
        metadata = prepare_token_memmap(
            partial_path,
            tokenizer=lm.tokenizer,
            token_count=int(fresh["token_count"]),
            token_offset=int(fresh["token_offset"]),
            dataset_name=source_corpus["dataset_name"],
            dataset_config=source_corpus.get("dataset_config"),
            split=source_corpus["split"],
        )
        partial_path.replace(paths.tail_tokens)
        partial_metadata.replace(metadata_path)
    atomic_json(done, {"complete": True, "metadata": metadata})


def _tail_batcher(
    config: Mapping[str, Any],
    paths: ExperimentPaths,
    split: str,
) -> MemmapTokenBatcher:
    fresh = config["fresh_corpus"]
    natural = config["natural_text"]
    source_training = config["source"]["training"]
    return MemmapTokenBatcher(
        paths.tail_tokens,
        token_count=int(fresh["token_count"]),
        token_range=TokenRange(*fresh[f"{split}_range"]),
        sequence_length=int(source_training["sequence_length"]),
        batch_size=int(source_training["sequences_per_batch"]),
        seed=int(natural[f"{split}_seed"]),
    )


@torch.inference_mode()
def cache_natural(
    config: Mapping[str, Any],
    paths: ExperimentPaths,
    device: torch.device,
    *,
    splits: Sequence[str] = ("selection", "test"),
) -> None:
    if not (paths.tail_tokens.exists() and paths.source_calibration.exists()):
        raise FileNotFoundError("tail tokens and source calibration are required")
    lm = load_lm(config, device)
    calibration = torch.load(paths.source_calibration, map_location="cpu", weights_only=False)
    stats = ActivationStats.from_state_dict(calibration["activation_stats"], device)
    target_tokens = int(config["natural_text"]["activation_tokens"])
    sequence_length = int(config["source"]["training"]["sequence_length"])
    target_sequences = target_tokens // sequence_length
    if not splits or any(split not in {"selection", "test"} for split in splits):
        raise ValueError("natural cache splits must be selection and/or test")
    if len(set(splits)) != len(splits):
        raise ValueError("natural cache splits must be unique")
    outputs = {
        "selection": paths.natural_selection,
        "test": paths.natural_test,
    }
    for split in splits:
        output = outputs[split]
        if output.exists():
            payload = torch.load(output, map_location="cpu", weights_only=False)
            expected_range = [int(value) for value in config["fresh_corpus"][f"{split}_range"]]
            expected_offset = int(config["fresh_corpus"]["token_offset"])
            absolute_start = expected_offset + expected_range[0]
            absolute_stop = expected_offset + expected_range[1]
            if (
                payload.get("split") != split
                or [int(value) for value in payload.get("token_range", ())]
                != expected_range
                or int(payload.get("token_offset", -1)) != expected_offset
                or payload.get("normalized_with_sha256")
                != sha256_file(paths.source_calibration)
                or payload.get("repository") != config["repository"]
                or payload["input_ids"].shape[0] != target_sequences
                or payload["activations"].shape[:2] != payload["input_ids"].shape
                or payload["starts"].shape != (target_sequences,)
                or int(payload["starts"].min()) < absolute_start
                or int(payload["starts"].max()) + sequence_length > absolute_stop
            ):
                raise RuntimeError(f"incompatible immutable natural cache: {output}")
            continue
        batcher = _tail_batcher(config, paths, split)
        ids, activations, starts = [], [], []
        while sum(len(value) for value in ids) < target_sequences:
            token_ids, relative_starts = batcher.batch_with_starts()
            activation = stats.normalize(lm.activations(token_ids))
            ids.append(token_ids.cpu())
            activations.append(activation.cpu().half())
            starts.append(relative_starts + int(config["fresh_corpus"]["token_offset"]))
        payload = {
            "split": split,
            "input_ids": torch.cat(ids)[:target_sequences],
            "activations": torch.cat(activations)[:target_sequences],
            "starts": torch.cat(starts)[:target_sequences],
            "eos_token_id": int(lm.tokenizer.eos_token_id),
            "token_range": list(config["fresh_corpus"][f"{split}_range"]),
            "token_offset": int(config["fresh_corpus"]["token_offset"]),
            "normalized_with": str(paths.source_calibration),
            "normalized_with_sha256": sha256_file(paths.source_calibration),
            "repository": config["repository"],
        }
        atomic_torch(output, payload)
        print(f"cached {split}: {target_tokens:,} normalized tokens", flush=True)
    atomic_json(paths.output / "cache_natural.json", {"complete": True})


def _source_batcher(
    config: Mapping[str, Any],
    paths: ExperimentPaths,
    range_name: str,
    *,
    seed: int,
) -> MemmapTokenBatcher:
    source = config["source"]
    return MemmapTokenBatcher(
        paths.source_tokens,
        token_count=int(source["corpus"]["token_count"]),
        token_range=TokenRange(*source["corpus"]["ranges"][range_name]),
        sequence_length=int(source["training"]["sequence_length"]),
        batch_size=int(source["training"]["sequences_per_batch"]),
        seed=seed,
    )


@torch.inference_mode()
def calibrate_static(
    config: Mapping[str, Any], paths: ExperimentPaths, device: torch.device
) -> None:
    if paths.static_calibration.exists():
        state = torch.load(paths.static_calibration, map_location="cpu", weights_only=False)
        required = {"spectral", "whitening", "ridge", "ridges_by_dof_fraction"}
        if not required <= state.keys():
            raise RuntimeError("static calibration artifact is incomplete")
        return
    source_state = torch.load(paths.source_calibration, map_location="cpu", weights_only=False)
    lm = load_lm(config, device)
    stats = ActivationStats.from_state_dict(source_state["activation_stats"], device)
    source = config["source"]
    batcher = _source_batcher(config, paths, "calibration", seed=int(source["seed"]))
    target = int(source["geometry"]["calibration_tokens"])
    chunks: list[Tensor] = []
    while sum(len(chunk) for chunk in chunks) < target:
        chunks.append(stats.normalize(lm.activations(batcher.batch())).flatten(0, 1))
    activations = torch.cat(chunks)[:target].float()
    ridge = float(source_state["ridge"])
    spectral = spectral_surrogate_operator(activations, ridge=ridge)

    group_size = int(source["geometry"]["group_size"])
    group_count = int(config["natural_text"]["ridge_calibration_groups"])
    ridge_groups = activations[: group_count * group_size].reshape(
        group_count, group_size, -1
    )
    ridges_by_fraction = {}
    for fraction in config["natural_text"]["ridge_fractions"]:
        values = [calibrate_ridge(group, float(fraction)) for group in ridge_groups]
        ridges_by_fraction[format(float(fraction), ".12g")] = {
            "ridge": float(np.median(values)),
            "values": values,
        }
    ridges_by_group_size = {}
    target_fraction = float(source["geometry"]["ridge_dof_fraction"])
    for size in config["natural_text"]["group_sizes"]:
        size = int(size)
        groups = activations[: group_count * size].reshape(group_count, size, -1)
        values = [calibrate_ridge(group, target_fraction) for group in groups]
        ridges_by_group_size[str(size)] = {
            "ridge": ridge if size == group_size else float(np.median(values)),
            "values": values,
            "dof_fraction": target_fraction,
        }
    atomic_torch(
        paths.static_calibration,
        {
            "spectral": spectral.cpu(),
            "whitening": source_state["whitening"],
            "ridge": ridge,
            "ridges_by_dof_fraction": ridges_by_fraction,
            "ridges_by_group_size": ridges_by_group_size,
            "source_range": list(source["corpus"]["ranges"]["calibration"]),
            "activation_tokens": target,
            "model_name": source["model_name"],
            "layer": int(source["layer"]),
        },
    )


def _source_decoder_weight(paths: ExperimentPaths) -> float:
    selection = json.loads((paths.source_artifact / "screening_selection.json").read_text())
    weight = float(selection["selected_decoder_weight"])
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("source DPSAE weight must be finite and positive")
    return weight


def training_specs(
    config: Mapping[str, Any], paths: ExperimentPaths, stage: str
) -> list[SAETrainSpec]:
    source = config["source"]
    k = int(source["sae"]["primary_k"])
    weight = _source_decoder_weight(paths)
    if stage == "baseline-screen":
        return screen_specs(
            k=k,
            seed=0,
            dpsae_weight=weight,
            beta_grid=config["baseline"]["beta_grid"],
        )
    if stage == "baseline-confirm":
        selection = json.loads(paths.baseline_selection.read_text())
        return confirmation_specs(
            k=k,
            seeds=config["baseline"]["confirmation_seeds"],
            dpsae_weight=weight,
            selection=selection,
        )
    raise ValueError(f"unknown training stage: {stage}")


def _stage_randomness(config: Mapping[str, Any], stage: str):
    source = config["source"]
    if stage == "baseline-confirm":
        changed = confirmation_replicate_config(
            source, replicate=int(config["baseline"]["confirmation_replicate"])
        )
        return stage_seeds(
            int(source["seed"]),
            "confirmation",
            replicate=int(changed["randomness"]["replicate"]),
        )
    return stage_seeds(int(source["seed"]), "baseline_screen")


def _learning_rate(source: Mapping[str, Any], step: int, total_steps: int) -> float:
    progress = step / total_steps
    warmup = float(source["sae"]["warmup_fraction"])
    if progress < warmup:
        scale = progress / warmup
    else:
        scale = 0.5 * (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup)))
    return float(source["sae"]["learning_rate"]) * scale


def _trim_log(path: Path, maximum_step: int) -> None:
    if not path.exists():
        return
    records = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if int(record["step"]) <= maximum_step:
            records.append(json.dumps(record))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(line + "\n" for line in records))
    temporary.replace(path)


def _load_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    required = {"input_ids", "activations", "starts"}
    if not required <= payload.keys():
        raise ValueError(f"natural cache is missing {sorted(required - payload.keys())}: {path}")
    return payload


@dataclass
class SampledGeometry:
    original: Tensor
    reference: Tensor
    targets: Tensor
    denominator_by_group: Tensor
    indices: Tensor


def sampled_geometry(
    activations: Tensor,
    input_ids: Tensor,
    *,
    ridge: float,
    group_size: int,
    probes: int,
    seed: int,
) -> SampledGeometry:
    indices = geometry_group_indices(input_ids, group_size, "contiguous")
    original = apply_geometry_groups(activations, indices).float()
    generator = torch.Generator(device=original.device).manual_seed(seed)
    targets = torch.randn(
        len(original), group_size, probes, generator=generator, device=original.device
    )
    targets.div_(targets.square().mean(1, keepdim=True).sqrt().clamp_min(1e-6))
    reference = batched_ridge_predict(original, targets, ridge)
    denominator = reference.square().sum(dim=(1, 2))
    return SampledGeometry(original, reference, targets, denominator, indices)


def sampled_model_report(
    geometry: SampledGeometry,
    activations: Tensor,
    reconstruction: Tensor,
    *,
    ridge: float,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    grouped = apply_geometry_groups(reconstruction, geometry.indices).float()
    prediction = batched_ridge_predict(grouped, geometry.targets, ridge)
    numerator = (prediction - geometry.reference).square().sum(dim=(1, 2))
    interval = bootstrap_ratio_interval(
        numerator,
        geometry.denominator_by_group,
        samples=bootstrap_samples,
        seed=seed,
    )
    return {
        "nmse": float(
            (reconstruction.float() - activations.float()).square().sum()
            / activations.float().square().sum().clamp_min(1e-12)
        ),
        "decoder": interval["estimate"],
        "decoder_distortion": interval["estimate"],
        "ci_low": interval["low"],
        "ci_high": interval["high"],
        "groups": len(numerator),
        "numerator_by_group": numerator.detach().cpu().tolist(),
        "denominator_by_group": geometry.denominator_by_group.detach().cpu().tolist(),
    }


@torch.inference_mode()
def _reconstruct(model, activations: Tensor, *, batch_tokens: int = 4096) -> Tensor:
    chunks = []
    flat = activations.flatten(0, 1)
    for batch in flat.split(batch_tokens):
        reconstruction, _ = model(batch.float(), use_threshold=True)
        chunks.append(reconstruction)
    return torch.cat(chunks).reshape_as(activations)


@torch.inference_mode()
def _evaluate_training_fleet(
    fleet: TrainingFleet,
    cache: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    ridge: float,
) -> dict[str, dict[str, Any]]:
    activations = cache["activations"].to(fleet.device).float()
    input_ids = cache["input_ids"]
    source_geometry = config["source"]["geometry"]
    natural = config["natural_text"]
    geometry = sampled_geometry(
        activations,
        input_ids,
        ridge=ridge,
        group_size=int(source_geometry["group_size"]),
        probes=int(natural["sampled_probes"]),
        seed=int(natural["sampled_probe_seed"]),
    )
    result = {}
    for spec in fleet.specs:
        reconstruction = _reconstruct(fleet.models[spec.name].eval(), activations)
        result[spec.name] = sampled_model_report(
            geometry,
            activations,
            reconstruction,
            ridge=ridge,
            bootstrap_samples=int(natural["bootstrap_samples"]),
            seed=int(natural["test_seed"]),
        )
    return result


def _checkpoint_matches(state: Mapping[str, Any], specs: Sequence[SAETrainSpec]) -> None:
    expected = [asdict(spec) for spec in specs]
    if state.get("specs") != expected:
        raise RuntimeError("checkpoint specs do not match the resolved stage configuration")


def train_baselines(
    config: Mapping[str, Any],
    paths: ExperimentPaths,
    device: torch.device,
    stage: str,
) -> None:
    output = paths.output / stage.replace("-", "_")
    done_path = output / "done.json"
    if _completed(done_path):
        split = "selection" if stage == "baseline-screen" else "test"
        required = (output / "models.pt", output / f"{split}.json")
        if not all(path.exists() for path in required):
            raise RuntimeError(f"completed {stage} is missing required artifacts")
        return
    if not paths.static_calibration.exists():
        raise FileNotFoundError("calibrate-static must complete before baseline training")
    specs = training_specs(config, paths, stage)
    source = config["source"]
    static = torch.load(paths.static_calibration, map_location="cpu", weights_only=False)
    source_state = torch.load(paths.source_calibration, map_location="cpu", weights_only=False)
    lm = load_lm(config, device)
    stats = ActivationStats.from_state_dict(source_state["activation_stats"], device)
    fleet = TrainingFleet(
        specs,
        input_dim=int(lm.model.config.n_embd),
        dictionary_size=int(source["sae"]["dictionary_size"]),
        learning_rate=float(source["sae"]["learning_rate"]),
        device=device,
        whitening=static["whitening"],
        spectral=static["spectral"],
        aux_weight=float(source["sae"]["aux_weight"]),
        dead_after_steps=int(source["sae"]["dead_after_steps"]),
        aux_k=int(source["sae"]["aux_k"]),
    )
    randomness = _stage_randomness(config, stage)
    range_name = "screen" if stage == "baseline-screen" else "confirmation"
    batcher = _source_batcher(
        config, paths, range_name, seed=int(randomness.data_order)
    )
    budget_key = "screen_tokens" if stage == "baseline-screen" else "confirmation_tokens"
    token_budget = int(config["baseline"][budget_key])
    tokens_per_step = int(source["training"]["sequence_length"]) * int(
        source["training"]["sequences_per_batch"]
    )
    total_steps = math.ceil(token_budget / tokens_per_step)
    checkpoint_every = max(
        1, int(config["baseline"]["checkpoint_tokens"]) // tokens_per_step
    )
    checkpoint = output / "checkpoint.pt"
    start_step, tokens_seen = 0, 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        _checkpoint_matches(state, specs)
        start_step, tokens_seen = fleet.load_state_dict(state)
        batcher.load_generator_state(state["batcher_generator_state"])
    log_path = output / "training.jsonl"
    output.mkdir(parents=True, exist_ok=True)
    _trim_log(log_path, start_step)
    started = time.monotonic()
    ridge = float(static["ridge"])
    for zero_step in range(start_step, total_steps):
        step = zero_step + 1
        learning_rate = _learning_rate(source, step, total_steps)
        for optimizer in fleet.optimizers.values():
            optimizer.param_groups[0]["lr"] = learning_rate
        activation = stats.normalize(lm.activations(batcher.batch())).flatten(0, 1)
        metrics = fleet.train_batch(
            activation,
            step=step,
            ridge=ridge,
            group_size=int(source["geometry"]["group_size"]),
            probes=int(source["geometry"]["probes"]),
            probe_seed=probe_seed_for_step(randomness, zero_step),
        )
        tokens_seen += len(activation)
        if zero_step % int(source["training"]["log_every_steps"]) == 0 or step == total_steps:
            record = {
                "step": step,
                "tokens_seen": tokens_seen,
                "learning_rate": learning_rate,
                "elapsed_seconds": time.monotonic() - started,
                "models": metrics,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"{stage} {step:,}/{total_steps:,}", flush=True)
        if step % checkpoint_every == 0 or step == total_steps:
            state = fleet.state_dict(step=step, tokens_seen=tokens_seen)
            state["batcher_generator_state"] = batcher.generator.get_state()
            state["randomness"] = asdict(randomness)
            atomic_torch(checkpoint, state)

    split = "selection" if stage == "baseline-screen" else "test"
    cache = _load_cache(
        paths.natural_selection if split == "selection" else paths.natural_test
    )
    evaluation = _evaluate_training_fleet(fleet, cache, config, ridge=ridge)
    atomic_json(output / f"{split}.json", evaluation)
    atomic_torch(output / "models.pt", fleet.export_models())
    if stage == "baseline-screen":
        selection = select_static_baselines(evaluation, specs, split="selection")
        atomic_json(paths.baseline_selection, selection)
    atomic_json(
        done_path,
        {
            "complete": True,
            "stage": stage,
            "tokens_seen": tokens_seen,
            "randomness": asdict(randomness),
            "evaluated_on": split,
        },
    )


def one_factor_settings(
    *,
    base_ridge: float,
    base_group_size: int,
    ridges: Sequence[float],
    group_sizes: Sequence[int],
    groupings: Sequence[str],
    group_ridges: Mapping[int, float] | None = None,
) -> list[tuple[str, float, int, str]]:
    """Return a deduplicated audit where exactly one geometry factor varies."""

    settings = []
    settings.extend(("ridge", float(ridge), base_group_size, "contiguous") for ridge in ridges)
    settings.extend(
        (
            "group_size",
            base_ridge if group_ridges is None else float(group_ridges[int(size)]),
            int(size),
            "contiguous",
        )
        for size in group_sizes
    )
    settings.extend(
        ("grouping", base_ridge, base_group_size, str(grouping))
        for grouping in groupings
    )
    result, seen = [], set()
    for axis, ridge, size, grouping in settings:
        key = (ridge, size, grouping)
        if key not in seen:
            result.append((axis, ridge, size, grouping))
            seen.add(key)
    return result


def _fleet_payloads(paths: ExperimentPaths, fleet: str) -> dict[str, dict[str, Any]]:
    if fleet == "baseline":
        model_path = paths.output / "baseline_confirm" / "models.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"missing {fleet} fleet: {model_path}")
        return torch.load(model_path, map_location="cpu", weights_only=False)
    payloads = {}
    for stage in ("confirmation", "robustness16", "robustness64"):
        model_path = paths.source_artifact / stage / "models.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"missing source fleet: {model_path}")
        stage_payloads = torch.load(model_path, map_location="cpu", weights_only=False)
        overlap = set(payloads) & set(stage_payloads)
        if overlap:
            raise RuntimeError(f"duplicate source model names: {sorted(overlap)}")
        payloads.update(stage_payloads)
    return payloads


def _partial_result(path: Path, expected: set[str], protocol: Mapping[str, Any]) -> dict:
    if not path.exists():
        return {"protocol": dict(protocol), "models": {}, "complete": False}
    result = json.loads(path.read_text())
    if result.get("protocol") != dict(protocol):
        raise RuntimeError(f"partial evaluation protocol changed: {path}")
    if set(result.get("models", {})) - expected:
        raise RuntimeError(f"partial evaluation contains unexpected models: {path}")
    return result


def _paired_reductions(
    models: Mapping[str, Mapping[str, Any]],
    exact_rows: Sequence[Mapping[str, Any]],
    *,
    base_ridge: float,
    base_group_size: int,
    samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    exact_by_name = {
        str(row["model"]): row
        for row in exact_rows
        if float(row["ridge"]) == base_ridge
        and int(row["group_size"]) == base_group_size
        and row["grouping"] == "contiguous"
    }
    by_seed = {
        (int(value["spec"]["seed"]), int(value["spec"]["k"])): name
        for name, value in models.items()
        if value["spec"]["method"] == "mse"
    }
    rows = []
    for name, value in models.items():
        spec = value["spec"]
        key = (int(spec["seed"]), int(spec["k"]))
        if spec["method"] == "mse" or key not in by_seed:
            continue
        baseline = by_seed[key]
        sampled = bootstrap_paired_reduction_interval(
            torch.tensor(models[baseline]["sampled_primary"]["numerator_by_group"]),
            torch.tensor(value["sampled_primary"]["numerator_by_group"]),
            samples=samples,
            seed=seed,
        )
        exact = bootstrap_paired_reduction_interval(
            torch.tensor(exact_by_name[baseline]["numerator_by_group"]),
            torch.tensor(exact_by_name[name]["numerator_by_group"]),
            samples=samples,
            seed=seed,
        )
        rows.append(
            {
                "baseline": baseline,
                "candidate": name,
                "method": spec["method"],
                "seed": int(spec["seed"]),
                "sampled_reduction": sampled,
                "exact_identity_reduction": exact,
            }
        )
    return rows


@torch.inference_mode()
def natural_evaluate(
    config: Mapping[str, Any],
    paths: ExperimentPaths,
    device: torch.device,
    fleet_name: str,
) -> None:
    payloads = _fleet_payloads(paths, fleet_name)
    static = torch.load(paths.static_calibration, map_location="cpu", weights_only=False)
    cache = _load_cache(paths.natural_test)
    natural = config["natural_text"]
    source = config["source"]
    base_ridge = float(static["ridge"])
    base_group_size = int(source["geometry"]["group_size"])
    protocol = {
        "fleet": fleet_name,
        "repository": config["repository"],
        "split": "test",
        "starts": cache["starts"].tolist(),
        "sampled_probe_seed": int(natural["sampled_probe_seed"]),
        "sampled_probes": int(natural["sampled_probes"]),
        "base_ridge": base_ridge,
        "base_group_size": base_group_size,
        "exact_tokens": int(natural["exact_tokens"]),
    }
    output = paths.output / f"natural_evaluation_{fleet_name}.json"
    result = _partial_result(output, set(payloads), protocol)
    if result.get("complete"):
        return
    activations = cache["activations"].to(device).float()
    input_ids = cache["input_ids"]
    geometry = sampled_geometry(
        activations,
        input_ids,
        ridge=base_ridge,
        group_size=base_group_size,
        probes=int(natural["sampled_probes"]),
        seed=int(natural["sampled_probe_seed"]),
    )
    exact_sequences = int(natural["exact_tokens"]) // activations.shape[1]
    reconstruction_dir = paths.output / "exact_reconstructions" / fleet_name
    for name, payload in payloads.items():
        cache_path = reconstruction_dir / f"{name}.pt"
        if name in result["models"] and cache_path.exists():
            continue
        print(f"natural evaluation {fleet_name}/{name}", flush=True)
        model = load_sae(payload, input_dim=activations.shape[-1], device=device)
        reconstruction = _reconstruct(model, activations)
        sampled = sampled_model_report(
            geometry,
            activations,
            reconstruction,
            ridge=base_ridge,
            bootstrap_samples=int(natural["bootstrap_samples"]),
            seed=int(natural["test_seed"]),
        )
        atomic_torch(cache_path, reconstruction[:exact_sequences].cpu().half())
        result["models"][name] = {"spec": payload["spec"], "sampled_primary": sampled}
        atomic_json(output, result)
        del model, reconstruction
        if device.type == "cuda":
            torch.cuda.empty_cache()

    ridge_values = [base_ridge] + [
        float(value["ridge"])
        for value in static["ridges_by_dof_fraction"].values()
    ]
    settings = one_factor_settings(
        base_ridge=base_ridge,
        base_group_size=base_group_size,
        ridges=ridge_values,
        group_sizes=natural["group_sizes"],
        groupings=natural["groupings"],
        group_ridges={
            int(size): float(value["ridge"])
            for size, value in static["ridges_by_group_size"].items()
        },
    )
    settings_payload = [
        {
            "axis": axis,
            "ridge": ridge,
            "group_size": group_size,
            "grouping": grouping,
        }
        for axis, ridge, group_size, grouping in settings
    ]
    exact_path = paths.output / f"natural_exact_audit_{fleet_name}.json"
    if exact_path.exists():
        exact_state = json.loads(exact_path.read_text())
        if exact_state.get("settings") != settings_payload:
            raise RuntimeError(f"exact audit settings changed: {exact_path}")
    else:
        exact_state = {"settings": settings_payload, "completed": [], "rows": []}
    completed = {tuple(value) for value in exact_state["completed"]}
    if len(completed) < len(settings):
        exact_original = activations[:exact_sequences]
        exact_ids = input_ids[:exact_sequences]
        reconstructions = {
            name: torch.load(
                reconstruction_dir / f"{name}.pt", map_location=device, weights_only=False
            ).float()
            for name in payloads
        }
        eos_token_id = int(cache["eos_token_id"])
        for axis, ridge, group_size, grouping in settings:
            key = (axis, ridge, group_size, grouping)
            if key in completed:
                continue
            rows = exact_decoder_sweep(
                exact_original,
                reconstructions,
                exact_ids,
                ridges=[ridge],
                group_sizes=[group_size],
                groupings=[grouping],
                eos_token_id=eos_token_id,
                max_groups=int(natural["exact_max_groups"]),
                bootstrap_samples=int(natural["bootstrap_samples"]),
                seed=int(natural["test_seed"]),
            )
            for row in rows:
                row["audit_axis"] = axis
            exact_state["rows"].extend(rows)
            exact_state["completed"].append(list(key))
            atomic_json(exact_path, exact_state)
    exact_rows = exact_state["rows"]

    primary_rows = {
        str(row["model"]): row
        for row in exact_rows
        if float(row["ridge"]) == base_ridge
        and int(row["group_size"]) == base_group_size
        and row["grouping"] == "contiguous"
    }
    for name in payloads:
        result["models"][name]["exact_identity_primary"] = primary_rows[name]
    result["exact_identity_audit"] = exact_rows
    result["paired_reductions"] = _paired_reductions(
        result["models"],
        exact_rows,
        base_ridge=base_ridge,
        base_group_size=base_group_size,
        samples=int(natural["bootstrap_samples"]),
        seed=int(natural["test_seed"]),
    )
    result["complete"] = True
    atomic_json(output, result)


def ioi_confirm_hook(*_args, **_kwargs) -> None:
    """Reserved execution hook for the separately implemented causal protocol."""

    raise NotImplementedError(
        "Experiment 4b IOI execution is intentionally separate from this natural-text driver"
    )


def run_stage(
    stage: str,
    config: Mapping[str, Any],
    paths: ExperimentPaths,
    device: torch.device,
    *,
    fleet: str | None = None,
    natural_splits: Sequence[str] = ("selection", "test"),
) -> None:
    if stage == "prepare-tail":
        prepare_tail(config, paths, device)
    elif stage == "cache-natural":
        cache_natural(config, paths, device, splits=natural_splits)
    elif stage == "calibrate-static":
        calibrate_static(config, paths, device)
    elif stage in {"baseline-screen", "baseline-confirm"}:
        train_baselines(config, paths, device, stage)
    elif stage == "natural-evaluate":
        if fleet is None:
            raise ValueError("natural-evaluate requires a fleet")
        natural_evaluate(config, paths, device, fleet)
    elif stage == "ioi-confirm":
        ioi_confirm_hook(config, paths, device)
    else:
        raise ValueError(stage)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=(*STAGES, "all"))
    parser.add_argument(
        "--config", type=Path, default=ROOT / "configs" / "exp04b_confirmatory.json"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--fleet", choices=("source", "baseline", "all"), default="all")
    parser.add_argument(
        "--natural-split",
        choices=("selection", "test", "all"),
        default="all",
        help="limit cache-natural so sealed test contexts need not be opened early",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    paths = experiment_paths(config)
    paths.output.mkdir(parents=True, exist_ok=True)
    atomic_json(paths.output / "resolved_config.json", config)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    natural_splits = (
        ("selection", "test")
        if args.natural_split == "all"
        else (args.natural_split,)
    )
    for stage, fleet in stage_sequence(args.stage, fleet=args.fleet):
        print(f"=== {stage}{'' if fleet is None else f' ({fleet})'} ===", flush=True)
        run_stage(
            stage,
            config,
            paths,
            device,
            fleet=fleet,
            natural_splits=natural_splits,
        )


if __name__ == "__main__":
    main()
