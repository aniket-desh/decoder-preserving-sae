#!/usr/bin/env python3
"""Matched-NMSE static spectral control with a sealed advancement gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import torch

from dpsae.corpus import MemmapTokenBatcher, TokenRange
from dpsae.exp04b_training import probe_seed_for_step, stage_seeds
from dpsae.exp11_static import (
    confirmation_specs,
    screen_specs,
    select_matched_spectral,
    summarize_confirmation,
)
from dpsae.language_model import ActivationStats, GPT2ActivationModel
from dpsae.language_training import SAETrainSpec, TrainingFleet

from experiments.exp04b_confirmatory import (
    sampled_geometry,
    sampled_model_report,
)


ROOT = Path(__file__).resolve().parents[1]
STAGES = ("validate", "screen", "confirm", "finalize", "all")


@dataclass(frozen=True)
class ExperimentPaths:
    output: Path
    source_artifact: Path
    evaluation_artifact: Path
    source_tokens: Path
    source_calibration: Path
    source_selection: Path
    static_calibration: Path
    selection_cache: Path
    test_cache: Path
    reference_root: Path
    reference_models: Path
    decision: Path


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


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def repository_state(root: Path = ROOT) -> dict[str, Any]:
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        status = subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=root, text=True
        ).splitlines()
    except (OSError, subprocess.CalledProcessError):
        revision, status = "unknown", ["repository state unavailable"]
    return {"revision": revision, "dirty": bool(status), "status": status}


def load_config(path: Path, *, root: Path = ROOT) -> dict[str, Any]:
    config = json.loads(path.read_text())
    source_path = root / config["source_config"]
    evaluation_path = root / config["evaluation_config"]
    config["source"] = json.loads(source_path.read_text())
    config["evaluation"] = json.loads(evaluation_path.read_text())
    config["natural_text"] = config["evaluation"]["natural_text"]
    config["config_path"] = str(path.resolve())
    config["config_sha256"] = sha256_file(path)
    config["repository"] = repository_state(root)
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    source = config["source"]
    expected = config["expected_source"]
    for key in ("model_name", "layer"):
        if source[key] != expected[key]:
            raise ValueError(f"source {key} does not match the sealed protocol")
    if int(source["sae"]["primary_k"]) != int(expected["primary_k"]):
        raise ValueError("source primary_k does not match the sealed protocol")
    screen = config["screen"]
    if [float(value) for value in screen["beta_grid"]] != [2, 4, 8, 16, 32]:
        raise ValueError("the sealed spectral beta grid is [2, 4, 8, 16, 32]")
    if float(screen["target_nmse_ratio"]) != 1.07:
        raise ValueError("the sealed NMSE target is 1.07")
    if float(screen["matching_tolerance"]) != 0.01:
        raise ValueError("the sealed NMSE matching tolerance is 0.01")
    if float(screen["decoder_reduction_margin"]) != 0.02:
        raise ValueError("the sealed decoder-reduction margin is 0.02")
    if int(screen["training_tokens"]) != 25_000_000:
        raise ValueError("the sealed screen budget is 25M tokens")
    confirmation = config["confirmation"]
    if list(confirmation["seeds"]) != [0, 1, 2]:
        raise ValueError("the sealed confirmation seeds are [0, 1, 2]")
    if int(confirmation["training_tokens"]) != 100_000_000:
        raise ValueError("the sealed confirmation budget is 100M tokens")
    if screen["evaluation_split"] != "selection":
        raise ValueError("the screen must use the selection cache")
    if confirmation["evaluation_split"] != "test":
        raise ValueError("confirmation must use the test cache")


def experiment_paths(
    config: Mapping[str, Any],
    *,
    root: Path = ROOT,
    reference_artifact_root: Path | None = None,
) -> ExperimentPaths:
    output = root / "artifacts" / str(config["experiment"])
    source = root / str(config["source_artifact"])
    evaluation = root / str(config["evaluation_artifact"])
    reference = (
        reference_artifact_root
        if reference_artifact_root is not None
        else root / str(config["reference_confirmation"]["artifact_root"])
    )
    return ExperimentPaths(
        output=output,
        source_artifact=source,
        evaluation_artifact=evaluation,
        source_tokens=source / "fineweb_gpt2_tokens.bin",
        source_calibration=source / "calibration.pt",
        source_selection=source / "screening_selection.json",
        static_calibration=evaluation / "static_calibration.pt",
        selection_cache=evaluation / "natural_selection.pt",
        test_cache=evaluation / "natural_test.pt",
        reference_root=reference,
        reference_models=reference / str(config["reference_confirmation"]["models_file"]),
        decision=output / "screen" / "decision.json",
    )


def _required_inputs(paths: ExperimentPaths, stage: str) -> dict[str, Path]:
    common = {
        "source_tokens": paths.source_tokens,
        "source_calibration": paths.source_calibration,
        "source_selection": paths.source_selection,
        "static_calibration": paths.static_calibration,
        "reference_models": paths.reference_models,
    }
    if stage == "screen":
        return {**common, "selection_cache": paths.selection_cache}
    if stage == "confirmation":
        return {**common, "test_cache": paths.test_cache, "decision": paths.decision}
    raise ValueError(stage)


def validate_inputs(
    config: Mapping[str, Any], paths: ExperimentPaths, stage: str
) -> dict[str, Any]:
    inputs = _required_inputs(paths, stage)
    missing = [str(path) for path in inputs.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("missing Exp11 inputs: " + ", ".join(missing))
    source_selection = json.loads(paths.source_selection.read_text())
    observed_weight = float(source_selection["selected_decoder_weight"])
    expected_weight = float(config["expected_source"]["decoder_weight"])
    if observed_weight != expected_weight:
        raise RuntimeError(
            f"source decoder weight {observed_weight} != sealed {expected_weight}"
        )
    observed_reference_sha = sha256_file(paths.reference_models)
    expected_reference_sha = str(
        config["reference_confirmation"]["expected_models_sha256"]
    )
    if observed_reference_sha != expected_reference_sha:
        raise RuntimeError(
            "reference confirmation bundle hash does not match the sealed artifact"
        )
    cache_path = paths.selection_cache if stage == "screen" else paths.test_cache
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    expected_split = config[stage]["evaluation_split"]
    if cache.get("split") != expected_split:
        raise RuntimeError(f"{cache_path} is not the sealed {expected_split} cache")
    if not {"input_ids", "activations", "starts"} <= cache.keys():
        raise RuntimeError(f"{cache_path} lacks required evaluation tensors")
    result = {
        name: {
            "path": str(path.resolve()),
            "bytes": path.stat().st_size,
            "sha256": (
                observed_reference_sha
                if name == "reference_models"
                else sha256_file(path)
            ),
        }
        for name, path in inputs.items()
    }
    result["reference_inventory"] = {
        "root": str(paths.reference_root.resolve()),
        "files": [
            {
                "path": str(path.relative_to(paths.reference_root)),
                "bytes": path.stat().st_size,
                "sha256": (
                    observed_reference_sha
                    if path == paths.reference_models
                    else sha256_file(path)
                ),
            }
            for path in sorted(paths.reference_root.rglob("*"))
            if path.is_file()
            and path.name != "checkpoint.pt"
            and (path.suffix in {".json", ".jsonl"} or path == paths.reference_models)
        ],
    }
    return result


def _provenance(
    config: Mapping[str, Any],
    *,
    stage: str,
    device: torch.device,
    inputs: Mapping[str, Any],
    randomness: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "stage": stage,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": config["config_path"],
        "config_sha256": config["config_sha256"],
        "protocol_version": config["protocol_version"],
        "protocol_frozen_utc": config["protocol_frozen_utc"],
        "execution_allocation": config["execution_allocation"],
        "repository": config["repository"],
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "randomness": dict(randomness),
        "inputs": dict(inputs),
    }


def stage_specs(
    config: Mapping[str, Any], stage: str, decision: Mapping[str, Any] | None = None
) -> list[SAETrainSpec]:
    expected = config["expected_source"]
    k = int(expected["primary_k"])
    decoder_weight = float(expected["decoder_weight"])
    if stage == "screen":
        return screen_specs(
            k=k,
            seed=int(config["screen"]["seed"]),
            decoder_weight=decoder_weight,
            beta_grid=config["screen"]["beta_grid"],
        )
    if stage == "confirmation":
        if not decision or not decision.get("advance"):
            raise ValueError("confirmation requires an advancing screen decision")
        return confirmation_specs(
            k=k,
            seeds=config["confirmation"]["seeds"],
            decoder_weight=decoder_weight,
            spectral_beta=float(decision["selected"]["spec"]["loss_weight"]),
        )
    raise ValueError(stage)


def _stage_randomness(config: Mapping[str, Any], stage: str):
    section = config[stage]
    return stage_seeds(
        int(config["source"]["seed"]),
        str(section["randomness_stage"]),
        replicate=int(section["randomness_replicate"]),
    )


def _learning_rate(source: Mapping[str, Any], step: int, total_steps: int) -> float:
    progress = step / total_steps
    warmup = float(source["sae"]["warmup_fraction"])
    if progress < warmup:
        scale = progress / warmup
    else:
        scale = 0.5 * (1 + math.cos(math.pi * (progress - warmup) / (1 - warmup)))
    return float(source["sae"]["learning_rate"]) * scale


def _batcher(
    config: Mapping[str, Any], paths: ExperimentPaths, stage: str, seed: int
) -> MemmapTokenBatcher:
    source = config["source"]
    range_name = str(config[stage]["corpus_range"])
    return MemmapTokenBatcher(
        paths.source_tokens,
        token_count=int(source["corpus"]["token_count"]),
        token_range=TokenRange(*source["corpus"]["ranges"][range_name]),
        sequence_length=int(source["training"]["sequence_length"]),
        batch_size=int(source["training"]["sequences_per_batch"]),
        seed=seed,
    )


def _load_cache(path: Path) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not {"input_ids", "activations", "starts"} <= payload.keys():
        raise ValueError(f"incomplete evaluation cache: {path}")
    return payload


@torch.inference_mode()
def _reconstruct(model, activations: torch.Tensor, batch_tokens: int = 4096) -> torch.Tensor:
    chunks = []
    flat = activations.flatten(0, 1)
    for batch in flat.split(batch_tokens):
        reconstruction, _ = model(batch.float(), use_threshold=True)
        chunks.append(reconstruction)
    return torch.cat(chunks).reshape_as(activations)


@torch.inference_mode()
def evaluate_fleet(
    fleet: TrainingFleet,
    cache: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    ridge: float,
) -> dict[str, dict[str, Any]]:
    activations = cache["activations"].to(fleet.device).float()
    natural = config["natural_text"]
    geometry_config = config["source"]["geometry"]
    geometry = sampled_geometry(
        activations,
        cache["input_ids"],
        ridge=ridge,
        group_size=int(geometry_config["group_size"]),
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


def _trim_log(path: Path, maximum_step: int) -> None:
    if not path.exists():
        return
    retained = []
    for line in path.read_text().splitlines():
        record = json.loads(line)
        if int(record["step"]) <= maximum_step:
            retained.append(json.dumps(record))
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(line + "\n" for line in retained))
    temporary.replace(path)


def train_stage(
    config: Mapping[str, Any],
    paths: ExperimentPaths,
    device: torch.device,
    stage: str,
    *,
    decision: Mapping[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    output = paths.output / stage
    done = output / "done.json"
    results_path = output / "results.json"
    models_path = output / "models.pt"
    if done.exists() and json.loads(done.read_text()).get("complete"):
        done_record = json.loads(done.read_text())
        if done_record.get("config_sha256") != config["config_sha256"]:
            raise RuntimeError(f"completed {stage} uses a different sealed config")
        if done_record.get("repository", {}).get("revision") != config["repository"][
            "revision"
        ]:
            raise RuntimeError(f"completed {stage} uses a different code revision")
        if not results_path.exists() or not models_path.exists():
            raise RuntimeError(f"completed {stage} is missing results or models")
        return json.loads(results_path.read_text())

    inputs = validate_inputs(config, paths, stage)
    specs = stage_specs(config, stage, decision)
    randomness = _stage_randomness(config, stage)
    provenance = _provenance(
        config,
        stage=stage,
        device=device,
        inputs=inputs,
        randomness=asdict(randomness),
    )
    atomic_json(output / "provenance.json", provenance)
    input_sha256 = {
        name: record["sha256"]
        for name, record in inputs.items()
        if isinstance(record, Mapping) and "sha256" in record
    }

    source = config["source"]
    static = torch.load(paths.static_calibration, map_location="cpu", weights_only=False)
    source_state = torch.load(paths.source_calibration, map_location="cpu", weights_only=False)
    lm = GPT2ActivationModel.from_pretrained(
        source["model_name"], layer=int(source["layer"]), device=device
    )
    stats = ActivationStats.from_state_dict(source_state["activation_stats"], device)
    fleet = TrainingFleet(
        specs,
        input_dim=int(lm.model.config.n_embd),
        dictionary_size=int(source["sae"]["dictionary_size"]),
        learning_rate=float(source["sae"]["learning_rate"]),
        device=device,
        spectral=static["spectral"],
        aux_weight=float(source["sae"]["aux_weight"]),
        dead_after_steps=int(source["sae"]["dead_after_steps"]),
        aux_k=int(source["sae"]["aux_k"]),
    )
    batcher = _batcher(config, paths, stage, randomness.data_order)
    tokens_per_step = int(source["training"]["sequence_length"]) * int(
        source["training"]["sequences_per_batch"]
    )
    total_steps = math.ceil(int(config[stage]["training_tokens"]) / tokens_per_step)
    checkpoint_every = max(1, int(config["checkpoint_tokens"]) // tokens_per_step)
    checkpoint = output / "checkpoint.pt"
    start_step, tokens_seen = 0, 0
    if checkpoint.exists():
        state = torch.load(checkpoint, map_location=device, weights_only=False)
        if state.get("specs") != [asdict(spec) for spec in specs]:
            raise RuntimeError("checkpoint specs do not match the sealed stage")
        if state.get("config_sha256") != config["config_sha256"]:
            raise RuntimeError("checkpoint config hash does not match the sealed stage")
        if state.get("input_sha256") != input_sha256:
            raise RuntimeError("checkpoint inputs do not match the sealed stage")
        if state.get("repository_revision") != config["repository"]["revision"]:
            raise RuntimeError("checkpoint code revision does not match the sealed stage")
        start_step, tokens_seen = fleet.load_state_dict(state)
        batcher.load_generator_state(state["batcher_generator_state"])
    output.mkdir(parents=True, exist_ok=True)
    log_path = output / "training.jsonl"
    _trim_log(log_path, start_step)
    started = time.monotonic()
    ridge = float(static["ridge"])
    for zero_step in range(start_step, total_steps):
        step = zero_step + 1
        fleet.set_learning_rate(_learning_rate(source, step, total_steps))
        activations = stats.normalize(lm.activations(batcher.batch())).flatten(0, 1)
        metrics = fleet.train_batch(
            activations,
            step=step,
            ridge=ridge,
            group_size=int(source["geometry"]["group_size"]),
            probes=int(source["geometry"]["probes"]),
            probe_seed=probe_seed_for_step(randomness, zero_step),
        )
        tokens_seen += len(activations)
        if zero_step % int(source["training"]["log_every_steps"]) == 0 or step == total_steps:
            record = {
                "step": step,
                "tokens_seen": tokens_seen,
                "learning_rate": _learning_rate(source, step, total_steps),
                "elapsed_seconds": time.monotonic() - started,
                "models": metrics,
            }
            with log_path.open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            print(f"{stage} {step:,}/{total_steps:,}", flush=True)
        if step % checkpoint_every == 0 or step == total_steps:
            state = fleet.state_dict(step=step, tokens_seen=tokens_seen)
            state.update(
                batcher_generator_state=batcher.generator.get_state(),
                randomness=asdict(randomness),
                config_sha256=config["config_sha256"],
                input_sha256=input_sha256,
                repository_revision=config["repository"]["revision"],
            )
            atomic_torch(checkpoint, state)

    cache_path = paths.selection_cache if stage == "screen" else paths.test_cache
    results = evaluate_fleet(
        fleet, _load_cache(cache_path), config, ridge=ridge
    )
    atomic_json(results_path, results)
    atomic_torch(models_path, fleet.export_models())
    atomic_json(
        done,
        {
            "complete": True,
            "stage": stage,
            "tokens_seen": tokens_seen,
            "evaluated_on": str(config[stage]["evaluation_split"]),
            "config_sha256": config["config_sha256"],
            "repository": config["repository"],
        },
    )
    return results


def run_screen(
    config: Mapping[str, Any], paths: ExperimentPaths, device: torch.device
) -> dict[str, Any]:
    if paths.decision.exists():
        decision = json.loads(paths.decision.read_text())
        if decision.get("config_sha256") != config["config_sha256"]:
            raise RuntimeError("screen decision does not match the sealed config")
        if decision.get("repository", {}).get("revision") != config["repository"][
            "revision"
        ]:
            raise RuntimeError("screen decision does not match the current code revision")
        return decision
    specs = stage_specs(config, "screen")
    results = train_stage(config, paths, device, "screen")
    screen = config["screen"]
    decision = select_matched_spectral(
        results,
        specs,
        split=str(screen["evaluation_split"]),
        target_nmse_ratio=float(screen["target_nmse_ratio"]),
        matching_tolerance=float(screen["matching_tolerance"]),
        decoder_reduction_margin=float(screen["decoder_reduction_margin"]),
    )
    decision["config_sha256"] = config["config_sha256"]
    decision["repository"] = config["repository"]
    atomic_json(paths.decision, decision)
    return decision


def run_confirmation(
    config: Mapping[str, Any], paths: ExperimentPaths, device: torch.device
) -> dict[str, Any]:
    if not paths.decision.exists():
        raise FileNotFoundError("screen decision is required before confirmation")
    decision = json.loads(paths.decision.read_text())
    if decision.get("config_sha256") != config["config_sha256"]:
        raise RuntimeError("screen decision does not match the sealed config")
    if decision.get("repository", {}).get("revision") != config["repository"]["revision"]:
        raise RuntimeError("screen decision does not match the current code revision")
    if not decision.get("advance"):
        report = {
            "complete": True,
            "status": "not_run_by_predeclared_gate",
            "screen_status": decision.get("status"),
            "config_sha256": config["config_sha256"],
        }
        atomic_json(paths.output / "confirmation" / "not_run.json", report)
        return report
    metrics = train_stage(config, paths, device, "confirmation", decision=decision)
    summary = summarize_confirmation(metrics, config["confirmation"]["seeds"])
    summary.update(
        status="complete",
        selected_spectral_beta=decision["selected"]["spec"]["loss_weight"],
        config_sha256=config["config_sha256"],
        repository=config["repository"],
    )
    atomic_json(paths.output / "confirmation" / "summary.json", summary)
    return summary


def finalize(config: Mapping[str, Any], paths: ExperimentPaths) -> dict[str, Any]:
    if not paths.decision.exists():
        raise FileNotFoundError("screen decision is required before finalization")
    decision = json.loads(paths.decision.read_text())
    if decision.get("config_sha256") != config["config_sha256"]:
        raise RuntimeError("screen decision does not match the sealed config")
    if decision.get("advance"):
        confirmation_path = paths.output / "confirmation" / "summary.json"
    else:
        confirmation_path = paths.output / "confirmation" / "not_run.json"
    if not confirmation_path.exists():
        raise FileNotFoundError("confirmation completion record is missing")
    report = {
        "complete": True,
        "experiment": config["experiment"],
        "config_sha256": config["config_sha256"],
        "repository": config["repository"],
        "screen": decision,
        "confirmation": json.loads(confirmation_path.read_text()),
    }
    atomic_json(paths.output / "summary.json", report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=STAGES)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "exp11_static_matched_nmse.json",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--reference-artifact-root",
        type=Path,
        help="directory containing the exact restored confirmation models.pt",
    )
    parser.add_argument(
        "--allow-dirty",
        action="store_true",
        help="development-only override; paid runs must use a clean revision",
    )
    args = parser.parse_args()
    config = load_config(args.config)
    if (
        config["require_clean_repository"]
        and config["repository"]["dirty"]
        and not args.allow_dirty
    ):
        raise RuntimeError("Exp11 requires a clean repository revision")
    paths = experiment_paths(
        config, reference_artifact_root=args.reference_artifact_root
    )
    paths.output.mkdir(parents=True, exist_ok=True)
    atomic_json(paths.output / "resolved_config.json", config)
    device = torch.device(args.device)
    torch.set_float32_matmul_precision("high")
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.stage == "validate":
        inputs = validate_inputs(config, paths, "screen")
        print(json.dumps({"complete": True, "validated": inputs}, indent=2))
        return
    if args.stage in {"screen", "all"}:
        decision = run_screen(config, paths, device)
        print(json.dumps({"screen_status": decision["status"]}, indent=2))
    if args.stage in {"confirm", "all"}:
        report = run_confirmation(config, paths, device)
        print(json.dumps({"confirmation_status": report["status"]}, indent=2))
    if args.stage in {"finalize", "all"}:
        report = finalize(config, paths)
        print(json.dumps({"complete": report["complete"]}, indent=2))


if __name__ == "__main__":
    main()
