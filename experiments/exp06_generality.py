#!/usr/bin/env python3
"""Bounded generality screens for decoder-preserving SAEs.

The two preregistered targets are GPT-2 small block 4 and
EleutherAI/pythia-160m-deduped block 8.  Each target calibrates its own
activation normalization and ridge, trains one paired MSE/DPSAE seed at a
caller-supplied frozen gamma, and receives one exact held-out evaluation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
from torch import Tensor

from dpsae.corpus import MemmapTokenBatcher, TokenRange, prepare_token_memmap
from dpsae.decoder_distance import calibrate_ridge
from dpsae.exp04b_natural_text import (
    bootstrap_paired_reduction_interval,
    exact_identity_decoder_statistics,
)
from dpsae.exp04b_training import probe_seed_for_step, stage_seeds
from dpsae.language_model import (
    ActivationStats,
    GPT2ActivationModel,
    estimate_activation_stats,
)
from dpsae.language_training import SAETrainSpec, TrainingFleet
from dpsae.mech_analysis import load_sae


ROOT = Path(__file__).resolve().parents[1]
GIB = 2**30


@dataclass(frozen=True)
class TargetSpec:
    key: str
    model_name: str
    layer: int
    architecture: str


TARGETS = {
    "gpt2-block4": TargetSpec(
        "gpt2-block4", "openai-community/gpt2", 4, "gpt2"
    ),
    "pythia-block8": TargetSpec(
        "pythia-block8",
        "EleutherAI/pythia-160m-deduped",
        8,
        "gpt_neox",
    ),
}


@dataclass(frozen=True)
class ScreenProtocol:
    dataset_name: str = "HuggingFaceFW/fineweb"
    dataset_config: str = "sample-10BT"
    split: str = "train"
    token_count: int = 50_000_000
    calibration_range: tuple[int, int] = (0, 10_000_000)
    training_range: tuple[int, int] = (10_000_000, 40_000_000)
    heldout_range: tuple[int, int] = (40_000_000, 50_000_000)
    calibration_tokens: int = 65_536
    evaluation_tokens: int = 16_384
    train_tokens: int = 25_000_000
    sequence_length: int = 256
    sequences_per_batch: int = 8
    dictionary_size: int = 16_384
    k: int = 32
    learning_rate: float = 3e-4
    warmup_fraction: float = 0.02
    aux_weight: float = 1 / 32
    dead_after_steps: int = 2_000
    aux_k: int = 512
    group_size: int = 128
    probes: int = 16
    ridge_dof_fraction: float = 0.25
    ridge_calibration_groups: int = 32
    checkpoint_tokens: int = 5_000_000
    log_every_steps: int = 25
    model_seed: int = 0
    base_seed: int = 2_027_071_601

    def __post_init__(self) -> None:
        ranges = [
            TokenRange(*self.calibration_range),
            TokenRange(*self.training_range),
            TokenRange(*self.heldout_range),
        ]
        if ranges[0].stop > ranges[1].start or ranges[1].stop > ranges[2].start:
            raise ValueError("calibration, training, and held-out ranges must be disjoint")
        if ranges[0].start < 0 or ranges[-1].stop > self.token_count:
            raise ValueError("all corpus ranges must lie inside the bounded shard")
        if self.sequence_length * self.sequences_per_batch % self.group_size:
            raise ValueError("training batches must divide into geometry groups")
        for value in (self.calibration_tokens, self.evaluation_tokens):
            if value <= 0 or value % self.sequence_length:
                raise ValueError("evaluation budgets must contain whole sequences")
        if self.train_tokens <= 0 or self.dictionary_size <= 0 or self.k <= 0:
            raise ValueError("training budgets and SAE sizes must be positive")
        if not 0 < self.ridge_dof_fraction < 1:
            raise ValueError("ridge degrees-of-freedom fraction must lie in (0, 1)")


DEFAULT_PROTOCOL = ScreenProtocol()


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


def canonical_digest(value: Any) -> str:
    encoded = json.dumps(
        value, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode()
    return hashlib.sha256(encoded).hexdigest()


def file_sha256(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
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
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"], cwd=root, text=True
            ).strip()
        )
    except (OSError, subprocess.CalledProcessError):
        revision, dirty = "unknown", None
    return {"revision": revision, "dirty": dirty}


def corpus_metadata_path(token_cache: Path) -> Path:
    return token_cache.with_suffix(token_cache.suffix + ".json")


def validate_corpus_metadata(
    metadata: Mapping[str, Any],
    target: TargetSpec,
    protocol: ScreenProtocol = DEFAULT_PROTOCOL,
) -> None:
    if metadata.get("tokenizer") != target.model_name:
        raise RuntimeError(
            f"{target.key} requires tokenizer {target.model_name}; "
            f"found {metadata.get('tokenizer')!r}"
        )
    if int(metadata.get("token_count", -1)) < protocol.token_count:
        raise RuntimeError("token cache is smaller than the bounded screen shard")
    if metadata.get("dtype") != "uint16":
        raise RuntimeError("generality token caches must use uint16")
    expected = (protocol.dataset_name, protocol.dataset_config, protocol.split)
    observed = (
        metadata.get("dataset_name"),
        metadata.get("dataset_config"),
        metadata.get("split"),
    )
    if observed != expected:
        raise RuntimeError("token cache comes from a different FineWeb source")


def validate_token_cache(
    token_cache: Path,
    metadata: Mapping[str, Any],
    target: TargetSpec,
    protocol: ScreenProtocol = DEFAULT_PROTOCOL,
) -> None:
    validate_corpus_metadata(metadata, target, protocol)
    expected_bytes = int(metadata["token_count"]) * np.dtype(np.uint16).itemsize
    if token_cache.stat().st_size != expected_bytes:
        raise RuntimeError("token cache size disagrees with its metadata")


def check_disk_guard(path: Path, *, minimum_free_gib: float) -> dict[str, float]:
    if minimum_free_gib <= 0:
        raise ValueError("minimum free disk must be positive")
    path.mkdir(parents=True, exist_ok=True)
    usage = shutil.disk_usage(path)
    free_gib = usage.free / GIB
    used_fraction = usage.used / usage.total
    if free_gib < minimum_free_gib or used_fraction >= 0.80:
        raise RuntimeError(
            f"disk guard failed: free={free_gib:.2f} GiB used={used_fraction:.1%}"
        )
    return {"free_gib": free_gib, "used_fraction": used_fraction}


def gpu_memory_limit_bytes(
    total_bytes: int, *, maximum_reserved_gib: float, maximum_fraction: float
) -> int:
    if total_bytes <= 0 or maximum_reserved_gib <= 0 or not 0 < maximum_fraction < 1:
        raise ValueError("GPU guard limits must be positive and fractional")
    return min(int(maximum_reserved_gib * GIB), int(maximum_fraction * total_bytes))


def check_gpu_guard(
    device: torch.device,
    *,
    maximum_reserved_gib: float,
    maximum_fraction: float,
) -> dict[str, float] | None:
    if device.type != "cuda":
        return None
    index = device.index if device.index is not None else torch.cuda.current_device()
    total = torch.cuda.get_device_properties(index).total_memory
    limit = gpu_memory_limit_bytes(
        total,
        maximum_reserved_gib=maximum_reserved_gib,
        maximum_fraction=maximum_fraction,
    )
    free, _ = torch.cuda.mem_get_info(index)
    reserved = torch.cuda.memory_reserved(index)
    externally_used = total - free
    if reserved >= limit or externally_used >= limit:
        raise RuntimeError(
            "GPU guard failed: "
            f"reserved={reserved / GIB:.2f} GiB used={externally_used / GIB:.2f} GiB "
            f"limit={limit / GIB:.2f} GiB"
        )
    return {
        "total_gib": total / GIB,
        "free_gib": free / GIB,
        "reserved_gib": reserved / GIB,
        "limit_gib": limit / GIB,
    }


class GenericActivationModel(GPT2ActivationModel):
    """Existing GPT-2/GPT-NeoX adapter plus target and revision metadata."""

    def __init__(
        self,
        model,
        tokenizer,
        *,
        target: TargetSpec,
        device: torch.device,
    ) -> None:
        if target.architecture == "gpt2":
            blocks = getattr(getattr(model, "transformer", None), "h", None)
        elif target.architecture == "gpt_neox":
            blocks = getattr(getattr(model, "gpt_neox", None), "layers", None)
        else:
            raise ValueError(f"unsupported architecture: {target.architecture}")
        if blocks is None:
            raise ValueError("target architecture is incompatible with the loaded model")
        super().__init__(model, tokenizer, layer=target.layer, device=device)
        hidden_size = getattr(model.config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(model.config, "n_embd", None)
        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError("model config has no valid hidden size")
        self.target = target
        self.hidden_size = hidden_size

    @classmethod
    def from_pretrained(
        cls,
        target: TargetSpec,
        *,
        device: torch.device,
        local_files_only: bool = False,
    ) -> "GenericActivationModel":
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            target.model_name, local_files_only=local_files_only
        )
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            target.model_name,
            dtype=torch.bfloat16,
            local_files_only=local_files_only,
        ).to(device)
        return cls(model, tokenizer, target=target, device=device)

    @property
    def resolved_model_revision(self) -> str | None:
        return getattr(self.model.config, "_commit_hash", None)


def load_tokenizer(target: TargetSpec, *, local_files_only: bool = False):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        target.model_name, local_files_only=local_files_only
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def prepare_corpus(
    *,
    target: TargetSpec,
    token_cache: Path,
    output: Path,
    minimum_free_disk_gib: float,
    local_files_only: bool = False,
    protocol: ScreenProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    disk = check_disk_guard(output, minimum_free_gib=minimum_free_disk_gib)
    metadata_path = corpus_metadata_path(token_cache)
    if token_cache.exists() != metadata_path.exists():
        raise RuntimeError("token cache and metadata must either both exist or both be absent")
    if token_cache.exists():
        metadata = json.loads(metadata_path.read_text())
        validate_token_cache(token_cache, metadata, target, protocol)
    else:
        tokenizer = load_tokenizer(target, local_files_only=local_files_only)
        metadata = prepare_token_memmap(
            token_cache,
            tokenizer=tokenizer,
            token_count=protocol.token_count,
            dataset_name=protocol.dataset_name,
            dataset_config=protocol.dataset_config,
            split=protocol.split,
        )
        validate_token_cache(token_cache, metadata, target, protocol)
    result = {
        "target": asdict(target),
        "protocol": asdict(protocol),
        "token_cache": {
            "path": str(token_cache.resolve()),
            "sha256": file_sha256(token_cache),
            "metadata": metadata,
        },
        "repository": repository_state(),
        "disk": disk,
    }
    atomic_json(output / "corpus.json", result)
    return result


def _stable_randomness(target: TargetSpec, protocol: ScreenProtocol):
    return stage_seeds(protocol.base_seed, f"exp06_{target.key}")


def resolved_config(
    *,
    target: TargetSpec,
    gamma: float,
    token_cache: Path,
    protocol: ScreenProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    if not math.isfinite(gamma) or gamma <= 0:
        raise ValueError("frozen gamma must be finite and strictly positive")
    metadata = json.loads(corpus_metadata_path(token_cache).read_text())
    validate_token_cache(token_cache, metadata, target, protocol)
    randomness = _stable_randomness(target, protocol)
    result = {
        "target": asdict(target),
        "protocol": asdict(protocol),
        "frozen_gamma": gamma,
        "randomness": asdict(randomness),
        "token_cache": {
            "path": str(token_cache.resolve()),
            "sha256": file_sha256(token_cache),
            "metadata_sha256": file_sha256(corpus_metadata_path(token_cache)),
        },
        "repository": repository_state(),
        "code_sha256": {
            str(path.relative_to(ROOT)): file_sha256(path)
            for path in (
                Path(__file__).resolve(),
                ROOT / "src/dpsae/corpus.py",
                ROOT / "src/dpsae/decoder_distance.py",
                ROOT / "src/dpsae/exp04b_natural_text.py",
                ROOT / "src/dpsae/exp04b_training.py",
                ROOT / "src/dpsae/language_model.py",
                ROOT / "src/dpsae/language_sae.py",
                ROOT / "src/dpsae/language_training.py",
                ROOT / "src/dpsae/mech_analysis.py",
            )
        },
    }
    result["config_digest"] = canonical_digest(result)
    return result


def _batcher(
    token_cache: Path,
    token_range: tuple[int, int],
    *,
    seed: int,
    protocol: ScreenProtocol,
) -> MemmapTokenBatcher:
    return MemmapTokenBatcher(
        token_cache,
        token_count=protocol.token_count,
        token_range=TokenRange(*token_range),
        sequence_length=protocol.sequence_length,
        batch_size=protocol.sequences_per_batch,
        seed=seed,
    )


def _load_or_write_config(
    output: Path,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    path = output / "resolved_config.json"
    if path.exists():
        observed = json.loads(path.read_text())
        if observed != expected:
            raise RuntimeError("resolved generality config changed")
        return observed
    atomic_json(path, expected)
    return dict(expected)


@torch.inference_mode()
def calibrate(
    *,
    target: TargetSpec,
    config: Mapping[str, Any],
    token_cache: Path,
    output: Path,
    device: torch.device,
    local_files_only: bool = False,
    protocol: ScreenProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    path = output / "calibration.pt"
    if path.exists():
        state = torch.load(path, map_location="cpu", weights_only=False)
        if state.get("config_digest") != config["config_digest"]:
            raise RuntimeError("calibration belongs to another resolved config")
        return state
    lm = GenericActivationModel.from_pretrained(
        target, device=device, local_files_only=local_files_only
    )
    model_revision = lm.resolved_model_revision
    if not isinstance(model_revision, str) or not model_revision:
        raise RuntimeError("loaded model did not expose a resolved Hub revision")
    batcher = _batcher(
        token_cache,
        protocol.calibration_range,
        seed=config["randomness"]["data_order"],
        protocol=protocol,
    )
    chunks = []
    while sum(len(chunk) for chunk in chunks) < protocol.calibration_tokens:
        chunks.append(lm.activations(batcher.batch()).flatten(0, 1).cpu())
    activations = torch.cat(chunks)[: protocol.calibration_tokens].to(device)
    stats = estimate_activation_stats(activations)
    normalized = stats.normalize(activations)
    ridge_tokens = protocol.ridge_calibration_groups * protocol.group_size
    groups = normalized[:ridge_tokens].reshape(-1, protocol.group_size, lm.hidden_size)
    ridge_values = [
        calibrate_ridge(group, protocol.ridge_dof_fraction) for group in groups
    ]
    state = {
        "config_digest": config["config_digest"],
        "activation_stats": stats.state_dict(),
        "ridge": float(np.median(ridge_values)),
        "ridge_values": ridge_values,
        "model_name": target.model_name,
        "model_revision": model_revision,
        "layer": target.layer,
        "architecture": target.architecture,
        "hidden_size": lm.hidden_size,
        "calibration_tokens": protocol.calibration_tokens,
    }
    atomic_torch(path, state)
    del lm, activations, normalized, groups
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return state


def _learning_rate(step: int, total_steps: int, protocol: ScreenProtocol) -> float:
    progress = step / total_steps
    if progress < protocol.warmup_fraction:
        scale = progress / protocol.warmup_fraction
    else:
        position = (progress - protocol.warmup_fraction) / (
            1 - protocol.warmup_fraction
        )
        scale = 0.5 * (1 + math.cos(math.pi * position))
    return protocol.learning_rate * scale


def training_specs(gamma: float, protocol: ScreenProtocol) -> list[SAETrainSpec]:
    return [
        SAETrainSpec("mse_s0", "mse", protocol.model_seed, protocol.k),
        SAETrainSpec(
            "dpsae_s0",
            "dpsae",
            protocol.model_seed,
            protocol.k,
            decoder_weight=gamma,
        ),
    ]


def _checkpoint_matches(
    checkpoint: Mapping[str, Any],
    *,
    config_digest: str,
    calibration_sha256: str,
    specs: list[SAETrainSpec],
) -> None:
    if checkpoint.get("config_digest") != config_digest:
        raise RuntimeError("checkpoint resolved-config digest changed")
    if checkpoint.get("calibration_sha256") != calibration_sha256:
        raise RuntimeError("checkpoint calibration hash changed")
    if checkpoint.get("specs") != [asdict(spec) for spec in specs]:
        raise RuntimeError("checkpoint model specs changed")


def _validate_model_identity(
    model: GenericActivationModel,
    calibration: Mapping[str, Any],
) -> None:
    expected = {
        "model_name": model.target.model_name,
        "model_revision": model.resolved_model_revision,
        "layer": model.target.layer,
        "architecture": model.target.architecture,
        "hidden_size": model.hidden_size,
    }
    observed = {key: calibration.get(key) for key in expected}
    if observed != expected:
        raise RuntimeError("loaded model identity changed after calibration")


def _append_jsonl(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")


def _trim_log(path: Path, maximum_step: int) -> None:
    if not path.exists():
        return
    records = [
        json.dumps(record)
        for record in map(json.loads, path.read_text().splitlines())
        if int(record["step"]) <= maximum_step
    ]
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(record + "\n" for record in records))
    temporary.replace(path)


def train(
    *,
    target: TargetSpec,
    config: Mapping[str, Any],
    token_cache: Path,
    output: Path,
    device: torch.device,
    minimum_free_disk_gib: float,
    maximum_gpu_reserved_gib: float,
    maximum_gpu_fraction: float,
    local_files_only: bool = False,
    protocol: ScreenProtocol = DEFAULT_PROTOCOL,
) -> None:
    done = output / "train_done.json"
    if done.exists():
        state = json.loads(done.read_text())
        models_path = output / "models.pt"
        if state.get("config_digest") != config["config_digest"]:
            raise RuntimeError("completed training belongs to another resolved config")
        if not models_path.exists():
            raise RuntimeError("completed training is missing models.pt")
        if state.get("models_sha256") != file_sha256(models_path):
            raise RuntimeError("completed model artifact hash changed")
        if state.get("calibration_sha256") != file_sha256(output / "calibration.pt"):
            raise RuntimeError("completed training calibration hash changed")
        return
    check_disk_guard(output, minimum_free_gib=minimum_free_disk_gib)
    check_gpu_guard(
        device,
        maximum_reserved_gib=maximum_gpu_reserved_gib,
        maximum_fraction=maximum_gpu_fraction,
    )
    calibration = torch.load(
        output / "calibration.pt", map_location="cpu", weights_only=False
    )
    calibration_hash = file_sha256(output / "calibration.pt")
    if calibration.get("config_digest") != config["config_digest"]:
        raise RuntimeError("training calibration does not match the resolved config")
    lm = GenericActivationModel.from_pretrained(
        target, device=device, local_files_only=local_files_only
    )
    _validate_model_identity(lm, calibration)
    stats = ActivationStats.from_state_dict(calibration["activation_stats"], device)
    specs = training_specs(float(config["frozen_gamma"]), protocol)
    fleet = TrainingFleet(
        specs,
        input_dim=lm.hidden_size,
        dictionary_size=protocol.dictionary_size,
        learning_rate=protocol.learning_rate,
        device=device,
        aux_weight=protocol.aux_weight,
        dead_after_steps=protocol.dead_after_steps,
        aux_k=protocol.aux_k,
    )
    randomness = _stable_randomness(target, protocol)
    batcher = _batcher(
        token_cache,
        protocol.training_range,
        seed=randomness.data_order,
        protocol=protocol,
    )
    tokens_per_step = protocol.sequence_length * protocol.sequences_per_batch
    total_steps = math.ceil(protocol.train_tokens / tokens_per_step)
    if total_steps < 1:
        raise ValueError("training token budget is smaller than one batch")
    checkpoint_every = max(1, protocol.checkpoint_tokens // tokens_per_step)
    checkpoint_path = output / "checkpoint.pt"
    start_step, tokens_seen = 0, 0
    if checkpoint_path.exists():
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        _checkpoint_matches(
            checkpoint,
            config_digest=config["config_digest"],
            calibration_sha256=calibration_hash,
            specs=specs,
        )
        start_step, tokens_seen = fleet.load_state_dict(checkpoint)
        batcher.load_generator_state(checkpoint["batcher_generator_state"])
    log_path = output / "training.jsonl"
    _trim_log(log_path, start_step)
    started = time.monotonic()
    for zero_step in range(start_step, total_steps):
        step = zero_step + 1
        learning_rate = _learning_rate(step, total_steps, protocol)
        for optimizer in fleet.optimizers.values():
            optimizer.param_groups[0]["lr"] = learning_rate
        activation = stats.normalize(lm.activations(batcher.batch())).flatten(0, 1)
        try:
            metrics = fleet.train_batch(
                activation,
                step=step,
                ridge=float(calibration["ridge"]),
                group_size=protocol.group_size,
                probes=protocol.probes,
                probe_seed=probe_seed_for_step(randomness, zero_step),
            )
        except torch.cuda.OutOfMemoryError as error:
            raise RuntimeError("GPU OOM guard stopped the generality fleet") from error
        tokens_seen += len(activation)
        if zero_step % protocol.log_every_steps == 0 or step == total_steps:
            resources = check_gpu_guard(
                device,
                maximum_reserved_gib=maximum_gpu_reserved_gib,
                maximum_fraction=maximum_gpu_fraction,
            )
            _append_jsonl(
                log_path,
                {
                    "step": step,
                    "tokens_seen": tokens_seen,
                    "learning_rate": learning_rate,
                    "elapsed_seconds": time.monotonic() - started,
                    "models": metrics,
                    "gpu": resources,
                },
            )
        if step % checkpoint_every == 0 or step == total_steps:
            check_disk_guard(output, minimum_free_gib=minimum_free_disk_gib)
            checkpoint = fleet.state_dict(step=step, tokens_seen=tokens_seen)
            checkpoint.update(
                config_digest=config["config_digest"],
                calibration_sha256=calibration_hash,
                batcher_generator_state=batcher.generator.get_state(),
                randomness=asdict(randomness),
            )
            atomic_torch(checkpoint_path, checkpoint)
    atomic_torch(output / "models.pt", fleet.export_models())
    atomic_json(
        done,
        {
            "complete": True,
            "config_digest": config["config_digest"],
            "tokens_seen": tokens_seen,
            "requested_train_tokens": protocol.train_tokens,
            "steps": total_steps,
            "models_sha256": file_sha256(output / "models.pt"),
            "calibration_sha256": calibration_hash,
        },
    )


@torch.inference_mode()
def prepare_evaluation_cache(
    *,
    target: TargetSpec,
    config: Mapping[str, Any],
    token_cache: Path,
    output: Path,
    device: torch.device,
    local_files_only: bool = False,
    protocol: ScreenProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    path = output / "evaluation_cache.pt"
    calibration_hash = file_sha256(output / "calibration.pt")
    if path.exists():
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("config_digest") != config["config_digest"]:
            raise RuntimeError("evaluation cache belongs to another resolved config")
        if value.get("calibration_sha256") != calibration_hash:
            raise RuntimeError("evaluation cache calibration hash changed")
        return value
    calibration = torch.load(
        output / "calibration.pt", map_location="cpu", weights_only=False
    )
    if calibration.get("config_digest") != config["config_digest"]:
        raise RuntimeError("evaluation calibration does not match the resolved config")
    lm = GenericActivationModel.from_pretrained(
        target, device=device, local_files_only=local_files_only
    )
    _validate_model_identity(lm, calibration)
    stats = ActivationStats.from_state_dict(calibration["activation_stats"], device)
    randomness = _stable_randomness(target, protocol)
    batcher = _batcher(
        token_cache,
        protocol.heldout_range,
        seed=randomness.data_order + 1,
        protocol=protocol,
    )
    sequence_count = protocol.evaluation_tokens // protocol.sequence_length
    ids, starts, activations = [], [], []
    while sum(len(chunk) for chunk in ids) < sequence_count:
        batch_ids, batch_starts = batcher.batch_with_starts()
        ids.append(batch_ids.cpu())
        starts.append(batch_starts.cpu())
        activations.append(stats.normalize(lm.activations(batch_ids)).cpu().half())
    value = {
        "config_digest": config["config_digest"],
        "token_range": list(protocol.heldout_range),
        "input_ids": torch.cat(ids)[:sequence_count],
        "starts": torch.cat(starts)[:sequence_count],
        "activations": torch.cat(activations)[:sequence_count],
        "source_token_cache_sha256": config["token_cache"]["sha256"],
        "calibration_sha256": calibration_hash,
        "model_revision": calibration["model_revision"],
    }
    atomic_torch(path, value)
    del lm
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return value


@torch.inference_mode()
def evaluate_one_model(
    payload: Mapping[str, Any],
    activations: Tensor,
    *,
    ridge: float,
    group_size: int,
    device: torch.device,
    batch_tokens: int = 4_096,
) -> dict[str, Any]:
    model = load_sae(dict(payload), input_dim=activations.shape[-1], device=device)
    reconstruction_chunks = []
    active, token_count = 0, 0
    for batch in activations.flatten(0, 1).split(batch_tokens):
        reconstruction, code = model(batch.to(device).float(), use_threshold=True)
        reconstruction_chunks.append(reconstruction.cpu())
        active += int((code != 0).sum())
        token_count += len(code)
    reconstruction = torch.cat(reconstruction_chunks).reshape_as(activations).float()
    original = activations.float().reshape(-1, group_size, activations.shape[-1])
    candidate = reconstruction.reshape_as(original)
    numerator, denominator = exact_identity_decoder_statistics(
        original, candidate, ridge=ridge
    )
    result = {
        "spec": dict(payload["spec"]),
        "nmse": float(
            (reconstruction - activations.float()).square().sum()
            / activations.float().square().sum().clamp_min(1e-12)
        ),
        "inference_l0": active / token_count,
        "decoder_distortion": float(
            numerator.sum() / denominator.sum().clamp_min(1e-12)
        ),
        "numerator_by_group": numerator.tolist(),
        "denominator_by_group": denominator.tolist(),
    }
    del model, reconstruction, candidate
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def _paired_payload_names(
    payloads: Mapping[str, Mapping[str, Any]], protocol: ScreenProtocol
) -> dict[str, str]:
    result = {}
    for name, payload in payloads.items():
        spec = payload.get("spec", {})
        if spec.get("seed") != protocol.model_seed or spec.get("k") != protocol.k:
            continue
        method = spec.get("method")
        if method in {"mse", "dpsae"}:
            if method in result:
                raise ValueError(f"multiple {method} models in screen payload")
            result[method] = name
    if set(result) != {"mse", "dpsae"}:
        raise ValueError("evaluation requires one paired MSE and DPSAE model")
    return result


def evaluate(
    *,
    target: TargetSpec,
    config: Mapping[str, Any],
    token_cache: Path,
    output: Path,
    device: torch.device,
    minimum_free_disk_gib: float,
    maximum_gpu_reserved_gib: float,
    maximum_gpu_fraction: float,
    local_files_only: bool = False,
    protocol: ScreenProtocol = DEFAULT_PROTOCOL,
) -> dict[str, Any]:
    check_disk_guard(output, minimum_free_gib=minimum_free_disk_gib)
    check_gpu_guard(
        device,
        maximum_reserved_gib=maximum_gpu_reserved_gib,
        maximum_fraction=maximum_gpu_fraction,
    )
    cache = prepare_evaluation_cache(
        target=target,
        config=config,
        token_cache=token_cache,
        output=output,
        device=device,
        local_files_only=local_files_only,
        protocol=protocol,
    )
    result_path = output / "evaluation.json"
    artifact_hashes = {
        "evaluation_cache_sha256": file_sha256(output / "evaluation_cache.pt"),
        "models_sha256": file_sha256(output / "models.pt"),
        "calibration_sha256": file_sha256(output / "calibration.pt"),
    }
    result = (
        json.loads(result_path.read_text())
        if result_path.exists()
        else {
            "config_digest": config["config_digest"],
            "repository": repository_state(),
            **artifact_hashes,
            "models": {},
            "complete": False,
        }
    )
    if result.get("config_digest") != config["config_digest"]:
        raise RuntimeError("partial evaluation belongs to another resolved config")
    if any(result.get(key) != value for key, value in artifact_hashes.items()):
        raise RuntimeError("partial evaluation input artifact hash changed")
    payloads = torch.load(output / "models.pt", map_location="cpu", weights_only=False)
    names = _paired_payload_names(payloads, protocol)
    selected = {name: payloads[name] for name in names.values()}
    del payloads
    calibration = torch.load(
        output / "calibration.pt", map_location="cpu", weights_only=False
    )
    activations = cache["activations"]
    for method in ("mse", "dpsae"):
        name = names[method]
        payload = selected.pop(name)
        if name not in result["models"]:
            result["models"][name] = evaluate_one_model(
                payload,
                activations,
                ridge=float(calibration["ridge"]),
                group_size=protocol.group_size,
                device=device,
            )
            atomic_json(result_path, result)
        del payload
    mse = result["models"][names["mse"]]
    dpsae = result["models"][names["dpsae"]]
    interval = bootstrap_paired_reduction_interval(
        torch.tensor(mse["numerator_by_group"]),
        torch.tensor(dpsae["numerator_by_group"]),
        samples=10_000,
        seed=_stable_randomness(target, protocol).probe_sequence + 1,
    )
    nmse_ratio = dpsae["nmse"] / max(mse["nmse"], 1e-12)
    result.update(
        paired_reduction=interval,
        nmse_ratio=nmse_ratio,
        screen_gate={
            "minimum_decoder_reduction": 0.10,
            "maximum_nmse_ratio": 1.10,
            "passes": interval["estimate"] >= 0.10 and nmse_ratio <= 1.10,
        },
        complete=True,
    )
    atomic_json(result_path, result)
    return result


def _device(value: str) -> torch.device:
    if value == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(value)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def default_output(target: TargetSpec) -> Path:
    return ROOT / "artifacts/exp06_generality" / target.key


def default_token_cache(target: TargetSpec, output: Path) -> Path:
    if target.key == "gpt2-block4":
        return ROOT / "artifacts/exp04_ioi_mechanism/fineweb_gpt2_tokens.bin"
    return output / "fineweb_pythia_tokens.bin"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "calibrate", "train", "evaluate", "all"))
    parser.add_argument("--target", choices=tuple(TARGETS), required=True)
    parser.add_argument("--gamma", type=float, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--token-cache", type=Path)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--min-free-disk-gib", type=float, default=12.0)
    parser.add_argument("--max-gpu-reserved-gib", type=float, default=40.0)
    parser.add_argument("--max-gpu-fraction", type=float, default=0.80)
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    target = TARGETS[args.target]
    output = args.output or default_output(target)
    token_cache = args.token_cache or default_token_cache(target, output)
    device = _device(args.device)
    if args.stage in {"prepare", "all"}:
        prepare_corpus(
            target=target,
            token_cache=token_cache,
            output=output,
            minimum_free_disk_gib=args.min_free_disk_gib,
            local_files_only=args.local_files_only,
        )
    if not token_cache.exists() or not corpus_metadata_path(token_cache).exists():
        raise FileNotFoundError("prepare the tokenizer-matched corpus before this stage")
    config = resolved_config(target=target, gamma=args.gamma, token_cache=token_cache)
    _load_or_write_config(output, config)
    if args.stage in {"calibrate", "all"}:
        calibrate(
            target=target,
            config=config,
            token_cache=token_cache,
            output=output,
            device=device,
            local_files_only=args.local_files_only,
        )
    if args.stage in {"train", "all"}:
        train(
            target=target,
            config=config,
            token_cache=token_cache,
            output=output,
            device=device,
            minimum_free_disk_gib=args.min_free_disk_gib,
            maximum_gpu_reserved_gib=args.max_gpu_reserved_gib,
            maximum_gpu_fraction=args.max_gpu_fraction,
            local_files_only=args.local_files_only,
        )
    if args.stage in {"evaluate", "all"}:
        evaluate(
            target=target,
            config=config,
            token_cache=token_cache,
            output=output,
            device=device,
            minimum_free_disk_gib=args.min_free_disk_gib,
            maximum_gpu_reserved_gib=args.max_gpu_reserved_gib,
            maximum_gpu_fraction=args.max_gpu_fraction,
            local_files_only=args.local_files_only,
        )


if __name__ == "__main__":
    main()
