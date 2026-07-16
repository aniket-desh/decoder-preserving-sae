#!/usr/bin/env python3
"""Confirmatory frozen-network compatibility on natural text and IOI.

The confirmatory natural-text range is only opened by the ``prepare`` stage.
Smoke mode requires an already-opened cache and cannot dispatch ``prepare``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from dpsae.corpus import prepare_token_memmap
from dpsae.ioi import (
    CANONICAL_BABA_TEMPLATES,
    TEMPLATE_SPLITS,
    canonical_name_splits,
    generate_ioi_examples,
    tokenize_ioi_examples,
)
from dpsae.language_model import (
    ActivationStats,
    GPT2ActivationModel,
    answer_logit_difference,
    final_token_logits,
)
from dpsae.mech_analysis import load_sae, make_replacement
if __package__:
    from experiments.exp08_language_evidence import (
        aggregate_frozen_rows,
        pair_names,
        selected_payloads,
    )
else:
    from exp08_language_evidence import aggregate_frozen_rows, pair_names, selected_payloads


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/exp09_frozen_network.json"
DEFAULT_OUTPUT = ROOT / "artifacts/exp09_frozen_network"
DEFAULT_MODELS = ROOT / "artifacts/exp08_experiment_figure/confirmation/models.pt"
DEFAULT_CALIBRATION = ROOT / "artifacts/exp04_ioi_mechanism/calibration.pt"
DEFAULT_TRAINING_DONE = ROOT / "artifacts/exp08_experiment_figure/confirmation/done.json"
DEFAULT_CONFIRMATION_SUMMARY = ROOT / "artifacts/exp08_experiment_figure/confirmation_summary.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("prepare", "natural", "ioi", "validate", "all"))
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--models", type=Path, default=DEFAULT_MODELS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--training-done", type=Path, default=DEFAULT_TRAINING_DONE)
    parser.add_argument(
        "--confirmation-summary", type=Path, default=DEFAULT_CONFIRMATION_SUMMARY
    )
    parser.add_argument("--natural-cache", type=Path)
    parser.add_argument("--smoke-cache", type=Path)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--maximum-sequences", type=int, default=0)
    parser.add_argument("--maximum-ioi-examples", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--maximum-peak-gpu-gib", type=float, default=12.0)
    parser.add_argument("--minimum-free-gib", type=float, default=10.0)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    return parser.parse_args()


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def file_record(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def validate_record_is_current(record: Mapping[str, Any], label: str) -> None:
    path = Path(str(record.get("path", "")))
    if not path.is_file():
        raise FileNotFoundError(f"recorded {label} is missing: {path}")
    if file_record(path) != dict(record):
        raise ValueError(f"recorded {label} changed after the stage completed")


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def atomic_torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def repository_state(root: Path = ROOT) -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=root, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=root, text=True
    ).splitlines()
    return {"revision": revision, "dirty": bool(status), "status": status}


def load_config(path: Path) -> dict[str, Any]:
    config = json.loads(path.read_text())
    validate_config(config)
    return config


def validate_config(config: Mapping[str, Any]) -> None:
    natural = config["natural_text"]
    checkpoints = config["checkpoints"]
    ioi = config["ioi"]
    fixed = {
        "natural_text.sequences": (int(natural["sequences"]), 2_048),
        "natural_text.sequence_length": (int(natural["sequence_length"]), 256),
        "natural_text.bootstrap_samples": (int(natural["bootstrap_samples"]), 10_000),
        "natural_text.noninferiority_margin": (
            float(natural["noninferiority_margin"]),
            1.01,
        ),
        "checkpoints.selected_decoder_weight": (
            float(checkpoints["selected_decoder_weight"]),
            0.03125,
        ),
        "checkpoints.expected_k": (int(checkpoints["expected_k"]), 32),
        "checkpoints.input_dimension": (int(checkpoints["input_dimension"]), 768),
        "checkpoints.dictionary_size": (int(checkpoints["dictionary_size"]), 16_384),
        "natural_text.identity_max_abs_logit_difference": (
            float(natural["identity_max_abs_logit_difference"]),
            1e-5,
        ),
        "natural_text.identity_max_mean_abs_logit_difference": (
            float(natural["identity_max_mean_abs_logit_difference"]),
            1e-7,
        ),
        "ioi.examples": (int(ioi["examples"]), 2_048),
        "ioi.bootstrap_samples": (int(ioi["bootstrap_samples"]), 10_000),
    }
    for name, (observed, expected) in fixed.items():
        if observed != expected:
            raise ValueError(f"frozen Exp09 setting changed: {name}={observed}, expected {expected}")
    if [int(value) for value in checkpoints["expected_seeds"]] != [0, 1, 2]:
        raise ValueError("Exp09 requires exactly checkpoint seeds [0, 1, 2]")
    absolute = tuple(int(value) for value in natural["absolute_range"])
    if absolute != (200_000_000, 210_000_000):
        raise ValueError("Exp09 confirmatory range must be [200M,210M)")
    if absolute[1] - absolute[0] < int(natural["sequences"]) * int(
        natural["sequence_length"]
    ):
        raise ValueError("confirmatory range is too short for non-overlapping sequences")
    expected_algorithm = "random_permutation_of_nonoverlapping_aligned_blocks_v1"
    if natural["sampling_algorithm"] != expected_algorithm:
        raise ValueError("Exp09 non-overlapping sampling algorithm changed")
    for section in (natural, ioi):
        quantiles = [float(value) for value in section["confidence_interval"]]
        if quantiles != [0.025, 0.975]:
            raise ValueError("Exp09 uses the two-sided 95% percentile interval")
    if ioi["name_split"] != "test" or ioi["template_split"] != "test":
        raise ValueError("Exp09 IOI must use the frozen test names and templates")
    revision = str(natural["dataset_revision"])
    if len(revision) != 40 or any(character not in "0123456789abcdef" for character in revision):
        raise ValueError("Exp09 FineWeb revision must be a full lowercase commit SHA")


def validate_exact_file(path: Path, contract: Mapping[str, Any], label: str) -> dict[str, Any]:
    record = file_record(path)
    if record["bytes"] != int(contract["bytes"]):
        raise ValueError(f"{label} byte count differs from its frozen contract")
    if record["sha256"] != str(contract["sha256"]):
        raise ValueError(f"{label} SHA-256 differs from its frozen contract")
    return record


def validate_checkpoint_provenance(
    *,
    models: Path,
    training_done: Path,
    confirmation_summary: Path,
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    contract = config["checkpoint_provenance"]
    records = {
        "models": validate_exact_file(models, config["checkpoints"], "checkpoint bundle"),
        "training_done": validate_exact_file(
            training_done, contract["training_done"], "checkpoint training manifest"
        ),
        "confirmation_summary": validate_exact_file(
            confirmation_summary,
            contract["confirmation_summary"],
            "checkpoint confirmation summary",
        ),
    }
    done = json.loads(training_done.read_text())
    summary = json.loads(confirmation_summary.read_text())
    expected_revision = str(contract["training_revision"])
    expected_interval = [int(value) for value in contract["training_interval"]]
    if not done.get("complete") or done.get("repository", {}).get("revision") != expected_revision:
        raise ValueError("checkpoint training manifest has the wrong completion or revision")
    if [int(value) for value in done.get("stream", {}).get("range", ())] != expected_interval:
        raise ValueError("checkpoint training manifest has the wrong training interval")
    frozen_specs = {
        (int(spec["seed"]), str(spec["method"]), float(spec["decoder_weight"]), int(spec["k"]))
        for spec in done.get("specs", ())
    }
    expected_specs = {
        (seed, method, 0.0 if method == "mse" else 0.03125, 32)
        for seed in (0, 1, 2)
        for method in ("mse", "dpsae")
    }
    if frozen_specs != expected_specs:
        raise ValueError("checkpoint training manifest has the wrong six model specs")
    if (
        not summary.get("complete")
        or not summary.get("gate_passed")
        or summary.get("repository", {}).get("revision") != expected_revision
        or [int(value) for value in summary.get("expected_seeds", ())] != [0, 1, 2]
        or not math.isclose(
            float(summary.get("selected_decoder_weight", -1)), 0.03125, rel_tol=0, abs_tol=1e-12
        )
    ):
        raise ValueError("checkpoint confirmation summary differs from the frozen contract")
    summary_models = summary.get("inputs", {}).get("models", {})
    if (
        int(summary_models.get("bytes", -1)) != records["models"]["bytes"]
        or summary_models.get("sha256") != records["models"]["sha256"]
    ):
        raise ValueError("checkpoint confirmation summary names another model bundle")
    return records


def deterministic_nonoverlapping_starts(
    *, start: int, stop: int, sequence_length: int, count: int, seed: int
) -> Tensor:
    """Sample aligned blocks without replacement, preserving a deterministic random order."""

    if start < 0 or stop <= start or sequence_length <= 0 or count <= 0:
        raise ValueError("invalid sequence-sampling coordinates")
    blocks = (stop - start) // sequence_length
    if count > blocks:
        raise ValueError("not enough disjoint aligned blocks for the requested sample")
    generator = torch.Generator(device="cpu").manual_seed(seed)
    chosen = torch.randperm(blocks, generator=generator)[:count]
    return start + chosen.to(torch.int64) * sequence_length


def validate_natural_inputs(
    payload: Mapping[str, Any],
    *,
    expected_range: tuple[int, int] | None = None,
    expected_sequences: int | None = None,
    require_nonoverlap: bool,
) -> tuple[int, int]:
    input_ids = payload.get("input_ids")
    starts = payload.get("starts")
    if not isinstance(input_ids, Tensor) or input_ids.ndim != 2:
        raise ValueError("natural input cache is missing rank-2 input_ids")
    if not isinstance(starts, Tensor) or starts.shape != (len(input_ids),):
        raise ValueError("natural input cache has inconsistent start positions")
    if expected_sequences is not None and len(input_ids) != expected_sequences:
        raise ValueError("natural input cache has the wrong sequence count")
    observed_range = tuple(int(value) for value in payload.get("absolute_range", ()))
    if len(observed_range) != 2:
        token_offset = payload.get("token_offset")
        token_range = payload.get("token_range")
        if token_offset is None or not isinstance(token_range, Sequence) or len(token_range) != 2:
            raise ValueError("natural input cache has no absolute range provenance")
        observed_range = tuple(int(token_offset) + int(value) for value in token_range)
    if expected_range is not None and observed_range != expected_range:
        raise ValueError(
            f"natural input range {observed_range} differs from {expected_range}"
        )
    sequence_length = int(input_ids.shape[1])
    if len(starts) and (
        int(starts.min()) < observed_range[0]
        or int(starts.max()) + sequence_length > observed_range[1]
    ):
        raise ValueError("natural input cache contains an out-of-range sequence")
    sorted_starts = starts.to(torch.int64).sort().values
    if require_nonoverlap and len(torch.unique(sorted_starts)) != len(sorted_starts):
        raise ValueError("natural input cache contains duplicate sequence starts")
    if require_nonoverlap and len(sorted_starts) > 1:
        if bool(((sorted_starts[1:] - sorted_starts[:-1]) < sequence_length).any()):
            raise ValueError("natural input cache contains overlapping sequences")
    return observed_range


def _dtype(name: str) -> torch.dtype:
    mapping = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}
    if name not in mapping:
        raise ValueError(f"unsupported model dtype: {name}")
    return mapping[name]


def load_lm(
    config: Mapping[str, Any],
    device: torch.device,
    *,
    local_files_only: bool,
) -> GPT2ActivationModel:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_config = config["model"]
    common = {
        "revision": str(model_config["revision"]),
        "local_files_only": local_files_only,
    }
    tokenizer = AutoTokenizer.from_pretrained(str(model_config["name"]), **common)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        str(model_config["name"]), dtype=_dtype(str(model_config["dtype"])), **common
    )
    model.to(device)
    return GPT2ActivationModel(
        model,
        tokenizer,
        layer=int(model_config["layer"]),
        device=device,
    )


def load_selected_models(
    path: Path,
    config: Mapping[str, Any],
    *,
    input_dim: int,
    device: torch.device,
) -> tuple[dict[str, Mapping[str, Any]], dict[str, Any]]:
    checkpoint_record = validate_exact_file(path, config["checkpoints"], "checkpoint bundle")
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, Mapping):
        raise ValueError("checkpoint bundle is not a model mapping")
    checkpoints = config["checkpoints"]
    payloads = selected_payloads(
        raw,
        float(checkpoints["selected_decoder_weight"]),
        expected_seeds=checkpoints["expected_seeds"],
    )
    if set(payloads) != set(raw) or len(payloads) != 6:
        raise ValueError("checkpoint bundle must contain exactly the six frozen Exp09 models")
    if input_dim != int(checkpoints["input_dimension"]):
        raise ValueError("language model width differs from the frozen checkpoint contract")
    expected_shape = (
        int(checkpoints["dictionary_size"]),
        int(checkpoints["input_dimension"]),
    )
    for name, payload in payloads.items():
        spec = payload["spec"]
        if int(spec["k"]) != int(checkpoints["expected_k"]):
            raise ValueError(f"checkpoint {name} has the wrong sparsity target")
        if tuple(payload["state_dict"]["decoder_weight"].shape) != expected_shape:
            raise ValueError(f"checkpoint {name} has the wrong SAE dimensions")
    models = {
        name: load_sae(dict(payload), input_dim=input_dim, device=device).eval()
        for name, payload in payloads.items()
    }
    return dict(payloads), {"record": checkpoint_record, "models": models}


def load_stats(
    path: Path, config: Mapping[str, Any], device: torch.device
) -> tuple[ActivationStats, dict[str, Any]]:
    record = validate_exact_file(path, config["calibration"], "activation calibration")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model_name") != config["model"]["name"]:
        raise ValueError("activation calibration targets another model")
    if int(payload.get("layer", -1)) != int(config["model"]["layer"]):
        raise ValueError("activation calibration targets another layer")
    return ActivationStats.from_state_dict(payload["activation_stats"], device), record


def natural_input_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    natural = config["natural_text"]
    return {
        "absolute_range": [int(value) for value in natural["absolute_range"]],
        "sequence_length": int(natural["sequence_length"]),
        "sequences": int(natural["sequences"]),
        "sampling_seed": int(natural["sampling_seed"]),
        "sampling_algorithm": str(natural["sampling_algorithm"]),
        "dataset_name": str(natural["dataset_name"]),
        "dataset_config": str(natural["dataset_config"]),
        "dataset_split": str(natural["dataset_split"]),
        "dataset_revision": str(natural["dataset_revision"]),
        "model": dict(config["model"]),
    }


def prepare_inputs(
    config: Mapping[str, Any],
    output_dir: Path,
    repository: Mapping[str, Any],
    *,
    local_files_only: bool,
) -> Path:
    """Open the fresh range once and create the lean immutable input cache."""

    from transformers import AutoTokenizer

    natural = config["natural_text"]
    contract = natural_input_contract(config)
    absolute = tuple(int(value) for value in natural["absolute_range"])
    sequence_length = int(natural["sequence_length"])
    cache_path = output_dir / "natural_inputs.pt"
    if cache_path.exists():
        payload = torch.load(cache_path, map_location="cpu", weights_only=False)
        validate_natural_inputs(
            payload,
            expected_range=absolute,
            expected_sequences=int(natural["sequences"]),
            require_nonoverlap=True,
        )
        if payload.get("repository") != repository:
            raise RuntimeError("existing fresh cache belongs to another repository state")
        if payload.get("contract") != contract:
            raise RuntimeError("existing fresh cache differs from the frozen input contract")
        return cache_path

    output_dir.mkdir(parents=True, exist_ok=True)
    tail_path = output_dir / "fineweb_gpt2_200m_210m_tokens.bin"
    tokenizer = AutoTokenizer.from_pretrained(
        str(config["model"]["name"]),
        revision=str(config["model"]["revision"]),
        local_files_only=local_files_only,
    )
    tokenizer.pad_token = tokenizer.eos_token
    metadata = prepare_token_memmap(
        tail_path,
        tokenizer=tokenizer,
        token_count=absolute[1] - absolute[0],
        token_offset=absolute[0],
        dataset_name=str(natural["dataset_name"]),
        dataset_config=str(natural["dataset_config"]),
        split=str(natural["dataset_split"]),
        dataset_revision=str(natural["dataset_revision"]),
    )
    expected_metadata = {
        "dataset_name": str(natural["dataset_name"]),
        "dataset_config": str(natural["dataset_config"]),
        "dataset_revision": str(natural["dataset_revision"]),
        "split": str(natural["dataset_split"]),
        "token_count": absolute[1] - absolute[0],
        "token_offset": absolute[0],
    }
    if any(metadata.get(key) != value for key, value in expected_metadata.items()):
        raise RuntimeError("FineWeb tail metadata differs from the frozen input contract")
    starts = deterministic_nonoverlapping_starts(
        start=absolute[0],
        stop=absolute[1],
        sequence_length=sequence_length,
        count=int(natural["sequences"]),
        seed=int(natural["sampling_seed"]),
    )
    memmap = np.memmap(
        tail_path,
        mode="r",
        dtype=np.uint16,
        shape=(absolute[1] - absolute[0],),
    )
    rows = np.stack(
        [
            memmap[int(start) - absolute[0] : int(start) - absolute[0] + sequence_length]
            for start in starts
        ]
    ).astype(np.int64, copy=False)
    payload = {
        "complete": True,
        "confirmatory": True,
        "fresh_range_opened": True,
        "absolute_range": list(absolute),
        "input_ids": torch.from_numpy(rows),
        "starts": starts,
        "sampling_seed": int(natural["sampling_seed"]),
        "sampling_algorithm": str(natural["sampling_algorithm"]),
        "contract": contract,
        "tail": file_record(tail_path),
        "tail_metadata": metadata,
        "model": dict(config["model"]),
        "repository": dict(repository),
    }
    validate_natural_inputs(
        payload,
        expected_range=absolute,
        expected_sequences=int(natural["sequences"]),
        require_nonoverlap=True,
    )
    atomic_torch(cache_path, payload)
    atomic_json(
        output_dir / "prepare_manifest.json",
        {
            "complete": True,
            "confirmatory": True,
            "fresh_range_opened": True,
            "natural_inputs": file_record(cache_path),
            "tail": file_record(tail_path),
            "repository": repository,
        },
    )
    return cache_path


def identity_gate(
    *,
    maximum: float,
    total: float,
    elements: int,
    max_tolerance: float,
    mean_tolerance: float,
) -> dict[str, Any]:
    if elements <= 0:
        raise ValueError("identity control evaluated no logits")
    mean = total / elements
    passed = maximum <= max_tolerance and mean <= mean_tolerance
    result = {
        "passed": passed,
        "maximum_absolute_logit_difference": maximum,
        "mean_absolute_logit_difference": mean,
        "maximum_tolerance": max_tolerance,
        "mean_tolerance": mean_tolerance,
        "elements": elements,
    }
    if not passed:
        raise RuntimeError(f"identity-hook control failed: {result}")
    return result


def _interval(values: Tensor, quantiles: Sequence[float]) -> list[float]:
    finite = values[torch.isfinite(values)]
    if not len(finite):
        raise ValueError("bootstrap statistic has no finite draws")
    q = torch.tensor(list(quantiles), dtype=torch.float64)
    return [float(value) for value in finite.quantile(q)]


def pooled_kl_ratio(
    baseline_rows: Sequence[Mapping[str, float]],
    candidate_rows: Sequence[Mapping[str, float]],
) -> float:
    baseline = sum(float(row["kl"]) for row in baseline_rows)
    candidate = sum(float(row["kl"]) for row in candidate_rows)
    if not math.isfinite(baseline) or baseline <= 0:
        raise ValueError("pooled MSE KL denominator must be finite and positive")
    if not math.isfinite(candidate) or candidate < 0:
        raise ValueError("pooled DPSAE KL numerator must be finite and nonnegative")
    return candidate / baseline


def bootstrap_natural_pair(
    common_rows: Sequence[Mapping[str, float]],
    baseline_rows: Sequence[Mapping[str, float]],
    candidate_rows: Sequence[Mapping[str, float]],
    *,
    samples: int,
    seed: int,
    quantiles: Sequence[float],
    chunk_size: int = 256,
) -> dict[str, Any]:
    if not (
        len(common_rows) == len(baseline_rows) == len(candidate_rows) and common_rows
    ):
        raise ValueError("paired natural rows must be nonempty and aligned")
    for common, baseline, candidate in zip(common_rows, baseline_rows, candidate_rows):
        ids = {common.get("sequence"), baseline.get("sequence"), candidate.get("sequence")}
        if len(ids) != 1:
            raise ValueError("paired natural rows have misaligned sequence IDs")

    def tensor(rows: Sequence[Mapping[str, float]], key: str) -> Tensor:
        return torch.tensor([float(row[key]) for row in rows], dtype=torch.float64)

    common = {
        key: tensor(common_rows, key)
        for key in (
            "tokens",
            "original_nll",
            "mean_nll",
            "activation_energy",
            "activation_tokens",
        )
    }
    baseline = {
        key: tensor(baseline_rows, key)
        for key in (
            "reconstructed_nll",
            "kl",
            "agreement",
            "reconstructed_correct",
            "reconstruction_sse",
            "l0_count",
        )
    }
    candidate = {
        key: tensor(candidate_rows, key) for key in baseline
    }
    draws_by_metric: dict[str, list[Tensor]] = {
        key: []
        for key in (
            "kl_ratio",
            "kl_difference",
            "loss_recovered_difference",
            "cross_entropy_increase_difference",
            "agreement_difference",
            "accuracy_difference",
            "nmse_ratio",
            "l0_difference",
        )
    }
    valid_loss_recovered = 0
    valid_kl_ratio = 0
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for offset in range(0, samples, chunk_size):
        size = min(chunk_size, samples - offset)
        indices = torch.randint(len(common_rows), (size, len(common_rows)), generator=generator)

        def summed(values: Tensor) -> Tensor:
            return values[indices].sum(1)

        tokens = summed(common["tokens"])
        original = summed(common["original_nll"])
        mean = summed(common["mean_nll"])
        mse_nll = summed(baseline["reconstructed_nll"])
        dpsae_nll = summed(candidate["reconstructed_nll"])
        mse_kl = summed(baseline["kl"])
        dpsae_kl = summed(candidate["kl"])
        valid_kl = torch.isfinite(mse_kl) & torch.isfinite(dpsae_kl) & (mse_kl > 0)
        valid_kl_ratio += int(valid_kl.sum())
        ratio = torch.full_like(mse_kl, torch.nan)
        ratio[valid_kl] = dpsae_kl[valid_kl] / mse_kl[valid_kl]
        denominator = mean - original
        valid_loss = denominator > 0
        valid_loss_recovered += int(valid_loss.sum())
        recovered = torch.full_like(denominator, torch.nan)
        recovered[valid_loss] = -(
            dpsae_nll[valid_loss] - mse_nll[valid_loss]
        ) / denominator[valid_loss]
        mse_sse = summed(baseline["reconstruction_sse"])
        dpsae_sse = summed(candidate["reconstruction_sse"])
        draws_by_metric["kl_ratio"].append(ratio)
        draws_by_metric["kl_difference"].append((dpsae_kl - mse_kl) / tokens)
        draws_by_metric["loss_recovered_difference"].append(recovered)
        draws_by_metric["cross_entropy_increase_difference"].append(
            (dpsae_nll - mse_nll) / tokens
        )
        draws_by_metric["agreement_difference"].append(
            (summed(candidate["agreement"]) - summed(baseline["agreement"])) / tokens
        )
        draws_by_metric["accuracy_difference"].append(
            (
                summed(candidate["reconstructed_correct"])
                - summed(baseline["reconstructed_correct"])
            )
            / tokens
        )
        draws_by_metric["nmse_ratio"].append(dpsae_sse / mse_sse)
        draws_by_metric["l0_difference"].append(
            (
                summed(candidate["l0_count"]) - summed(baseline["l0_count"])
            )
            / summed(common["activation_tokens"])
        )

    if valid_kl_ratio != samples:
        raise ValueError("one or more bootstrap draws had a nonpositive MSE KL denominator")
    intervals = {
        key: _interval(torch.cat(chunks), quantiles)
        for key, chunks in draws_by_metric.items()
    }
    return {
        "kl_ratio_dpsae_to_mse_ci95": intervals["kl_ratio"],
        "kl_difference_dpsae_minus_mse_ci95": intervals["kl_difference"],
        "loss_recovered_difference_dpsae_minus_mse_ci95": intervals[
            "loss_recovered_difference"
        ],
        "cross_entropy_increase_difference_dpsae_minus_mse_ci95": intervals[
            "cross_entropy_increase_difference"
        ],
        "top1_agreement_difference_dpsae_minus_mse_ci95": intervals[
            "agreement_difference"
        ],
        "next_token_accuracy_difference_dpsae_minus_mse_ci95": intervals[
            "accuracy_difference"
        ],
        "activation_nmse_ratio_dpsae_to_mse_ci95": intervals["nmse_ratio"],
        "inference_l0_difference_dpsae_minus_mse_ci95": intervals["l0_difference"],
        "valid_kl_ratio_draw_fraction": valid_kl_ratio / samples,
        "valid_loss_recovered_draw_fraction": valid_loss_recovered / samples,
    }


def all_seed_noninferiority(
    rows: Sequence[Mapping[str, Any]], *, expected_seeds: Sequence[int], margin: float
) -> bool:
    by_seed = {int(row["seed"]): row for row in rows}
    if sorted(by_seed) != sorted(int(seed) for seed in expected_seeds):
        raise ValueError("noninferiority rows do not cover the frozen seed set")
    return all(
        float(by_seed[int(seed)]["kl_ratio_dpsae_to_mse_ci95"][1]) < margin
        for seed in expected_seeds
    )


def natural_retention_rows(
    common_rows: Sequence[Mapping[str, Any]],
    model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
    payloads: Mapping[str, Mapping[str, Any]],
    *,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    """Flatten every paired condition into the exact Panel F sufficient statistics."""

    if not common_rows:
        raise ValueError("cannot retain an empty natural evaluation")
    for name, rows in model_rows.items():
        if len(rows) != len(common_rows):
            raise ValueError(f"checkpoint {name} has incomplete per-sequence rows")
    retained: list[dict[str, Any]] = []
    for pair_seed, names in sorted(pair_names(payloads).items()):
        actual_bootstrap_seed = bootstrap_seed + int(pair_seed)
        for index, common in enumerate(common_rows):
            shared = {
                "sequence": int(common["sequence"]),
                "absolute_start": int(common["absolute_start"]),
                "sequence_sha256": str(common["sequence_sha256"]),
                "valid_token_count": int(common["tokens"]),
                "activation_nmse_denominator": float(common["activation_energy"]),
                "checkpoint_pair_seed": int(pair_seed),
                "mse_checkpoint": names["mse"],
                "dpsae_checkpoint": names["dpsae"],
                "bootstrap_seed": actual_bootstrap_seed,
            }
            controls = (
                (
                    "original",
                    common["original_kl"],
                    common["original_nll"],
                    common["original_agreement"],
                    common["original_correct"],
                    0.0,
                ),
                (
                    "identity",
                    common["identity_kl"],
                    common["identity_nll"],
                    common["identity_agreement"],
                    common["identity_correct"],
                    common["identity_reconstruction_sse"],
                ),
                (
                    "mean_ablation",
                    common["mean_kl"],
                    common["mean_nll"],
                    common["mean_agreement"],
                    common["mean_correct"],
                    common["mean_reconstruction_sse"],
                ),
            )
            for condition, kl, cross_entropy, agreement, correct, nmse_numerator in controls:
                row = {
                    **shared,
                    "condition": condition,
                    "checkpoint": None,
                    "summed_original_to_condition_kl": float(kl),
                    "summed_cross_entropy": float(cross_entropy),
                    "original_condition_top1_agreement_count": int(agreement),
                    "next_token_correct_count": int(correct),
                    "activation_nmse_numerator": float(nmse_numerator),
                    "l0_sum": None,
                    "l0_count": None,
                }
                if condition == "identity":
                    row.update(
                        maximum_abs_logit_difference=float(
                            common["identity_max_abs_logit_difference"]
                        ),
                        mean_abs_logit_difference=float(
                            common["identity_mean_abs_logit_difference"]
                        ),
                    )
                retained.append(row)
            for condition in ("mse", "dpsae"):
                checkpoint = names[condition]
                model = model_rows[checkpoint][index]
                if int(model["sequence"]) != int(common["sequence"]):
                    raise ValueError("natural retention rows are not sequence aligned")
                retained.append(
                    {
                        **shared,
                        "condition": condition,
                        "checkpoint": checkpoint,
                        "summed_original_to_condition_kl": float(model["kl"]),
                        "summed_cross_entropy": float(model["reconstructed_nll"]),
                        "original_condition_top1_agreement_count": int(
                            model["agreement"]
                        ),
                        "next_token_correct_count": int(
                            model["reconstructed_correct"]
                        ),
                        "activation_nmse_numerator": float(
                            model["reconstruction_sse"]
                        ),
                        "l0_sum": float(model["l0_count"]),
                        "l0_count": int(common["activation_tokens"]),
                    }
                )
    return retained


def _next_token_statistics(logits: Tensor, targets: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    log_prob = F.log_softmax(logits[:, :-1].float(), dim=-1)
    nll = -log_prob.gather(-1, targets[:, :, None]).squeeze(-1)
    correct = log_prob.argmax(-1).eq(targets)
    return log_prob, nll.sum(1), correct.sum(1)


def _reuse_or_refuse(
    path: Path,
    *,
    inputs: Mapping[str, Any],
    repository: Mapping[str, Any],
    protocol: Mapping[str, Any],
) -> bool:
    if not path.exists():
        return False
    payload = json.loads(path.read_text())
    if not payload.get("complete"):
        return False
    if (
        payload.get("inputs") != inputs
        or payload.get("repository") != repository
        or payload.get("protocol") != protocol
    ):
        raise RuntimeError(f"refusing to reuse stale complete output: {path}")
    return True


def prepare_resources(
    output_dir: Path,
    device: torch.device,
    *,
    gpu_memory_fraction: float,
    minimum_free_gib: float,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output_dir).free / 2**30
    if free_gib < minimum_free_gib:
        raise RuntimeError(f"only {free_gib:.2f} GiB free")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device)
        torch.cuda.reset_peak_memory_stats(device)
    return {
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "gpu_memory_fraction": gpu_memory_fraction if device.type == "cuda" else None,
        "free_gib_at_start": free_gib,
        "torch_version": torch.__version__,
    }


def _finalize_resources(
    resources: dict[str, Any], device: torch.device, maximum_peak_gpu_gib: float
) -> None:
    if device.type != "cuda":
        return
    resources["peak_allocated_gpu_gib"] = torch.cuda.max_memory_allocated(device) / 2**30
    if resources["peak_allocated_gpu_gib"] > maximum_peak_gpu_gib:
        raise RuntimeError("Exp09 exceeded its peak GPU-memory guard")


@torch.inference_mode()
def run_natural(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    repository: Mapping[str, Any],
) -> Path:
    smoke = bool(args.smoke)
    natural = config["natural_text"]
    cache_path = args.smoke_cache if smoke else args.natural_cache
    if cache_path is None:
        cache_path = args.output_dir / "natural_inputs.pt"
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    expected_range = None if smoke else tuple(int(v) for v in natural["absolute_range"])
    expected_sequences = None if smoke else int(natural["sequences"])
    observed_range = validate_natural_inputs(
        cache,
        expected_range=expected_range,
        expected_sequences=expected_sequences,
        require_nonoverlap=not smoke,
    )
    if not smoke:
        if not cache.get("confirmatory") or not cache.get("fresh_range_opened"):
            raise ValueError("confirmatory evaluation requires the fresh immutable cache")
        if cache.get("contract") != natural_input_contract(config):
            raise ValueError("confirmatory cache differs from the frozen input contract")
        if cache.get("repository") != repository:
            raise ValueError("confirmatory cache belongs to another repository state")
        validate_record_is_current(cache["tail"], "FineWeb tail")
    input_ids = cache["input_ids"]
    starts = cache["starts"].to(torch.int64)
    maximum = args.maximum_sequences
    if smoke:
        maximum = maximum or int(config["smoke"]["natural_sequences"])
    elif maximum:
        raise ValueError("confirmatory natural evaluation cannot truncate its frozen sample")
    if maximum:
        input_ids, starts = input_ids[:maximum], starts[:maximum]

    bootstrap_samples = (
        int(config["smoke"]["bootstrap_samples"])
        if smoke
        else int(natural["bootstrap_samples"])
    )
    provenance = validate_checkpoint_provenance(
        models=args.models,
        training_done=args.training_done,
        confirmation_summary=args.confirmation_summary,
        config=config,
    )
    inputs = {
        "config": file_record(args.config),
        "calibration": validate_exact_file(
            args.calibration, config["calibration"], "activation calibration"
        ),
        "natural_cache": file_record(cache_path),
        **provenance,
    }
    protocol = {
        "confirmatory": not smoke,
        "fresh_range_opened": bool(cache.get("fresh_range_opened", False)) and not smoke,
        "absolute_range": list(observed_range),
        "sequences": len(input_ids),
        "sequence_length": int(input_ids.shape[1]),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": int(natural["bootstrap_seed"]),
        "confidence_interval": list(natural["confidence_interval"]),
        "kl_estimand": "ratio of pooled token-level KL sums",
        "noninferiority_margin": float(natural["noninferiority_margin"]),
        "raw_retention": (
            "per-sequence sufficient statistics for original, identity, mean ablation, "
            "and all six reconstructions; no full-vocabulary logits"
        ),
    }
    output = args.output_dir / "natural_results.json"
    if _reuse_or_refuse(output, inputs=inputs, repository=repository, protocol=protocol):
        return output

    started = time.time()
    device = torch.device(args.device)
    resources = prepare_resources(
        args.output_dir,
        device,
        gpu_memory_fraction=args.gpu_memory_fraction,
        minimum_free_gib=args.minimum_free_gib,
    )
    resources["allocation"] = dict(config["runpod"])
    lm = load_lm(config, device, local_files_only=args.local_files_only)
    stats, _ = load_stats(args.calibration, config, device)
    payloads, loaded = load_selected_models(
        args.models,
        config,
        input_dim=int(lm.model.config.n_embd),
        device=device,
    )
    models = loaded["models"]
    common_rows: list[dict[str, Any]] = []
    model_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    identity_max = 0.0
    identity_total = 0.0
    identity_elements = 0

    def identity_replacement(hidden: Tensor) -> Tensor:
        return hidden

    def mean_replacement(hidden: Tensor) -> Tensor:
        return stats.mean.reshape(1, 1, -1).expand_as(hidden)

    batch_size = int(natural["batch_sequences"])
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
        identity_delta_flat = identity_delta.flatten(1)
        identity_delta_max_by_sequence = identity_delta_flat.max(1).values
        identity_delta_mean_by_sequence = identity_delta_flat.double().mean(1)
        identity_max = max(identity_max, float(identity_delta.max()))
        identity_total += float(identity_delta.double().sum())
        identity_elements += identity_delta.numel()
        if identity_max > float(natural["identity_max_abs_logit_difference"]):
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
        mean_kl = (
            original_prob * (original_log_prob - mean_log_prob)
        ).sum(-1).sum(1)
        mean_agreement = mean_log_prob.argmax(-1).eq(original_top1).sum(1)
        del mean_logits, mean_log_prob
        tokens_per_sequence = int(targets.shape[1])
        for row in range(len(ids)):
            sequence_ids = ids[row].detach().cpu().contiguous()
            common_rows.append(
                {
                    "sequence": start + row,
                    "absolute_start": int(starts[start + row]),
                    "sequence_sha256": hashlib.sha256(
                        sequence_ids.numpy().tobytes()
                    ).hexdigest(),
                    "tokens": tokens_per_sequence,
                    "original_nll": float(original_nll[row]),
                    "original_kl": 0.0,
                    "original_agreement": tokens_per_sequence,
                    "mean_nll": float(mean_nll[row]),
                    "mean_kl": float(mean_kl[row]),
                    "mean_agreement": int(mean_agreement[row]),
                    "mean_correct": int(mean_correct[row]),
                    "mean_reconstruction_sse": float(activation_energy[row]),
                    "identity_nll": float(identity_nll[row]),
                    "identity_kl": float(identity_kl[row]),
                    "identity_agreement": int(identity_agreement[row]),
                    "identity_correct": int(identity_correct[row]),
                    "identity_max_abs_logit_difference": float(
                        identity_delta_max_by_sequence[row]
                    ),
                    "identity_mean_abs_logit_difference": float(
                        identity_delta_mean_by_sequence[row]
                    ),
                    "identity_reconstruction_sse": 0.0,
                    "original_correct": int(original_correct[row]),
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
                    raise ValueError("frozen reconstruction shape changed within a batch")
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
        print(f"natural: {min(start + len(ids), len(input_ids))}/{len(input_ids)}", flush=True)

    identity = identity_gate(
        maximum=identity_max,
        total=identity_total,
        elements=identity_elements,
        max_tolerance=float(natural["identity_max_abs_logit_difference"]),
        mean_tolerance=float(natural["identity_max_mean_abs_logit_difference"]),
    )
    reports = {
        name: {"spec": dict(payloads[name]["spec"]), **aggregate_frozen_rows(common_rows, rows)}
        for name, rows in model_rows.items()
    }
    paired = []
    for seed, names in sorted(pair_names(payloads).items()):
        mse_rows = model_rows[names["mse"]]
        dpsae_rows = model_rows[names["dpsae"]]
        mse = reports[names["mse"]]
        dpsae = reports[names["dpsae"]]
        row: dict[str, Any] = {
            "seed": seed,
            "bootstrap_seed": int(natural["bootstrap_seed"]) + seed,
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
                samples=bootstrap_samples,
                seed=int(natural["bootstrap_seed"]) + seed,
                quantiles=natural["confidence_interval"],
            )
        )
        row["noninferior_at_margin"] = (
            row["kl_ratio_dpsae_to_mse_ci95"][1]
            < float(natural["noninferiority_margin"])
        )
        paired.append(row)
    noninferiority_passed = all_seed_noninferiority(
        paired,
        expected_seeds=config["checkpoints"]["expected_seeds"],
        margin=float(natural["noninferiority_margin"]),
    )
    retention_rows = natural_retention_rows(
        common_rows,
        model_rows,
        payloads,
        bootstrap_seed=int(natural["bootstrap_seed"]),
    )
    _finalize_resources(resources, device, args.maximum_peak_gpu_gib)
    atomic_json(
        output,
        {
            "complete": True,
            "experiment": "exp09_frozen_network_natural",
            "confirmatory": not smoke,
            "primary_gate_passed": noninferiority_passed,
            "identity_hook": identity,
            "mean_ablation": "replace normalized activation with zero, then denormalize",
            "models": reports,
            "paired": paired,
            "per_sequence": {
                "schema": "panel_f_sufficient_statistics_v1",
                "condition_rows": retention_rows,
                "common": common_rows,
                "models": model_rows,
            },
            "inputs": inputs,
            "repository": repository,
            "protocol": protocol,
            "resources": resources,
            "wall_seconds": time.time() - started,
        },
    )
    return output


def build_ioi_prompt_payload(
    config: Mapping[str, Any], tokenizer, repository: Mapping[str, Any]
) -> dict[str, Any]:
    ioi = config["ioi"]
    names_by_split = canonical_name_splits(tokenizer, seed=int(ioi["name_split_seed"]))
    names = names_by_split[str(ioi["name_split"])]
    families = TEMPLATE_SPLITS[str(ioi["template_split"])]
    examples = generate_ioi_examples(
        count=int(ioi["examples"]),
        names=names,
        template_families=families,
        seed=int(ioi["generator_seed"]),
    )
    return {
        "complete": True,
        "experiment": "exp09_frozen_ioi_prompts",
        "protocol": {
            "examples": int(ioi["examples"]),
            "name_split": str(ioi["name_split"]),
            "template_split": str(ioi["template_split"]),
            "name_split_seed": int(ioi["name_split_seed"]),
            "generator_seed": int(ioi["generator_seed"]),
            "names": list(names),
            "template_families": list(families),
            "templates": [CANONICAL_BABA_TEMPLATES[index] for index in families],
            "construction": "balanced BABA/ABBA with recorded ABC and swapped controls",
        },
        "examples": [example.to_dict() for example in examples],
        "model": dict(config["model"]),
        "repository": dict(repository),
    }


def bootstrap_ioi_pair(
    baseline_rows: Sequence[Mapping[str, float]],
    candidate_rows: Sequence[Mapping[str, float]],
    *,
    samples: int,
    seed: int,
    quantiles: Sequence[float],
    chunk_size: int = 512,
) -> dict[str, list[float]]:
    if len(baseline_rows) != len(candidate_rows) or not baseline_rows:
        raise ValueError("paired IOI rows must be nonempty and aligned")
    for baseline, candidate in zip(baseline_rows, candidate_rows):
        if int(baseline["prompt_index"]) != int(candidate["prompt_index"]):
            raise ValueError("paired IOI rows have misaligned prompt IDs")

    def tensor(rows: Sequence[Mapping[str, float]], key: str) -> Tensor:
        return torch.tensor([float(row[key]) for row in rows], dtype=torch.float64)

    metrics = ("absolute_logit_difference_error", "preferred_answer_agreement", "accuracy")
    baseline = {key: tensor(baseline_rows, key) for key in metrics}
    candidate = {key: tensor(candidate_rows, key) for key in metrics}
    draws: dict[str, list[Tensor]] = {key: [] for key in metrics}
    generator = torch.Generator(device="cpu").manual_seed(seed)
    for offset in range(0, samples, chunk_size):
        size = min(chunk_size, samples - offset)
        indices = torch.randint(len(baseline_rows), (size, len(baseline_rows)), generator=generator)
        for key in metrics:
            draws[key].append(
                (candidate[key][indices] - baseline[key][indices]).mean(1)
            )
    return {
        "absolute_logit_difference_error_dpsae_minus_mse_ci95": _interval(
            torch.cat(draws["absolute_logit_difference_error"]), quantiles
        ),
        "preferred_answer_agreement_dpsae_minus_mse_ci95": _interval(
            torch.cat(draws["preferred_answer_agreement"]), quantiles
        ),
        "accuracy_dpsae_minus_mse_ci95": _interval(
            torch.cat(draws["accuracy"]), quantiles
        ),
    }


@torch.inference_mode()
def run_ioi(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    repository: Mapping[str, Any],
) -> Path:
    smoke = bool(args.smoke)
    ioi = config["ioi"]
    bootstrap_samples = (
        int(config["smoke"]["bootstrap_samples"])
        if smoke
        else int(ioi["bootstrap_samples"])
    )
    maximum = args.maximum_ioi_examples
    if smoke:
        maximum = maximum or int(config["smoke"]["ioi_examples"])
    elif maximum:
        raise ValueError("confirmatory IOI evaluation cannot truncate its frozen prompt set")
    prompt_path = args.output_dir / "ioi_prompts.json"

    device = torch.device(args.device)
    resources = prepare_resources(
        args.output_dir,
        device,
        gpu_memory_fraction=args.gpu_memory_fraction,
        minimum_free_gib=args.minimum_free_gib,
    )
    resources["allocation"] = dict(config["runpod"])
    started = time.time()
    lm = load_lm(config, device, local_files_only=args.local_files_only)
    expected_prompts = build_ioi_prompt_payload(config, lm.tokenizer, repository)
    if prompt_path.exists():
        if json.loads(prompt_path.read_text()) != expected_prompts:
            raise RuntimeError("existing frozen IOI prompt artifact differs from its contract")
    else:
        atomic_json(prompt_path, expected_prompts)
    prompt_payload = expected_prompts
    example_dicts = prompt_payload["examples"][:maximum] if maximum else prompt_payload["examples"]
    examples = generate_ioi_examples(
        count=len(example_dicts),
        names=prompt_payload["protocol"]["names"],
        template_families=prompt_payload["protocol"]["template_families"],
        seed=int(prompt_payload["protocol"]["generator_seed"]),
    )
    if [example.to_dict() for example in examples] != example_dicts:
        raise RuntimeError("regenerated IOI prompts differ from the recorded artifact")

    provenance = validate_checkpoint_provenance(
        models=args.models,
        training_done=args.training_done,
        confirmation_summary=args.confirmation_summary,
        config=config,
    )
    inputs = {
        "config": file_record(args.config),
        "calibration": validate_exact_file(
            args.calibration, config["calibration"], "activation calibration"
        ),
        "prompts": file_record(prompt_path),
        **provenance,
    }
    protocol = {
        "confirmatory": not smoke,
        "prompts": len(examples),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": int(ioi["bootstrap_seed"]),
        "confidence_interval": list(ioi["confidence_interval"]),
        "full_reconstruction": True,
        "raw_retention": (
            "per-prompt correct/incorrect logits, preferences, errors, and immutable "
            "prompt hashes; no full-vocabulary logits"
        ),
    }
    output = args.output_dir / "ioi_results.json"
    if _reuse_or_refuse(output, inputs=inputs, repository=repository, protocol=protocol):
        return output

    stats, _ = load_stats(args.calibration, config, device)
    payloads, loaded = load_selected_models(
        args.models,
        config,
        input_dim=int(lm.model.config.n_embd),
        device=device,
    )
    models = loaded["models"]
    checkpoint_pairs: dict[str, dict[str, Any]] = {}
    for pair_seed, names in pair_names(payloads).items():
        for checkpoint in names.values():
            checkpoint_pairs[checkpoint] = {
                "checkpoint_pair_seed": int(pair_seed),
                "mse_checkpoint": names["mse"],
                "dpsae_checkpoint": names["dpsae"],
                "bootstrap_seed": int(ioi["bootstrap_seed"]) + int(pair_seed),
            }
    common_rows: list[dict[str, Any]] = []
    model_rows: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
    identity_max = 0.0
    identity_total = 0.0
    identity_elements = 0

    def identity_replacement(hidden: Tensor) -> Tensor:
        return hidden

    batch_size = int(ioi["batch_size"])
    for start in range(0, len(examples), batch_size):
        batch = examples[start : start + batch_size]
        tokenized = tokenize_ioi_examples(batch, lm.tokenizer, variant="prompt")
        original_logits = lm.logits(tokenized["input_ids"], tokenized["attention_mask"])
        identity_logits = lm.logits(
            tokenized["input_ids"],
            tokenized["attention_mask"],
            replacement=identity_replacement,
        )
        delta = (identity_logits - original_logits).abs()
        delta_flat = delta.flatten(1)
        delta_max_by_prompt = delta_flat.max(1).values
        delta_mean_by_prompt = delta_flat.double().mean(1)
        identity_max = max(identity_max, float(delta.max()))
        identity_total += float(delta.double().sum())
        identity_elements += delta.numel()
        if identity_max > float(ioi["identity_max_abs_logit_difference"]):
            raise RuntimeError("IOI identity-hook maximum tolerance failed")
        original_final = final_token_logits(
            original_logits, tokenized["attention_mask"]
        ).cpu()
        identity_final = final_token_logits(
            identity_logits, tokenized["attention_mask"]
        ).cpu()
        final_rows = torch.arange(len(batch))
        correct_ids = tokenized["io_token_id"].to(torch.long)
        incorrect_ids = tokenized["subject_token_id"].to(torch.long)
        original_correct_logits = original_final[final_rows, correct_ids]
        original_incorrect_logits = original_final[final_rows, incorrect_ids]
        identity_correct_logits = identity_final[final_rows, correct_ids]
        identity_incorrect_logits = identity_final[final_rows, incorrect_ids]
        original_difference = original_correct_logits - original_incorrect_logits
        identity_difference = identity_correct_logits - identity_incorrect_logits
        for row, value in enumerate(original_difference):
            example = batch[row]
            common_rows.append(
                {
                    "prompt_index": start + row,
                    "prompt_sha256": hashlib.sha256(example.prompt.encode()).hexdigest(),
                    "abc_prompt_sha256": hashlib.sha256(
                        example.abc_prompt.encode()
                    ).hexdigest(),
                    "swapped_prompt_sha256": hashlib.sha256(
                        example.swapped_prompt.encode()
                    ).hexdigest(),
                    "generator_seed": int(ioi["generator_seed"]),
                    "name_split_id": str(ioi["name_split"]),
                    "template_family": int(example.template_family),
                    "template_id": int(example.template_family),
                    "order": example.order,
                    "io_name": example.io_name,
                    "subject_name": example.subject_name,
                    "third_name": example.third_name,
                    "name_triplet_sha256": hashlib.sha256(
                        json.dumps(
                            [example.io_name, example.subject_name, example.third_name],
                            separators=(",", ":"),
                        ).encode()
                    ).hexdigest(),
                    "correct_token_id": int(correct_ids[row]),
                    "incorrect_token_id": int(incorrect_ids[row]),
                    "original_correct_logit": float(original_correct_logits[row]),
                    "original_incorrect_logit": float(original_incorrect_logits[row]),
                    "original_logit_difference": float(value),
                    "original_preference": int(value >= 0),
                    "original_accuracy": int(value >= 0),
                    "identity_correct_logit": float(identity_correct_logits[row]),
                    "identity_incorrect_logit": float(identity_incorrect_logits[row]),
                    "identity_logit_difference": float(identity_difference[row]),
                    "identity_preference_agreement": int(
                        (identity_difference[row] >= 0) == (value >= 0)
                    ),
                    "identity_accuracy": int(identity_difference[row] >= 0),
                    "identity_absolute_logit_difference_error": float(
                        (identity_difference[row] - value).abs()
                    ),
                    "identity_max_abs_logit_difference": float(delta_max_by_prompt[row]),
                    "identity_mean_abs_logit_difference": float(delta_mean_by_prompt[row]),
                }
            )
        del identity_logits, original_logits, delta
        for name, model in models.items():
            logits = lm.logits(
                tokenized["input_ids"],
                tokenized["attention_mask"],
                replacement=make_replacement(model, stats),
            )
            reconstructed = answer_logit_difference(
                logits,
                tokenized["attention_mask"],
                tokenized["io_token_id"],
                tokenized["subject_token_id"],
            ).cpu()
            reconstructed_final = final_token_logits(
                logits, tokenized["attention_mask"]
            ).cpu()
            reconstructed_correct_logits = reconstructed_final[final_rows, correct_ids]
            reconstructed_incorrect_logits = reconstructed_final[final_rows, incorrect_ids]
            for row, value in enumerate(reconstructed):
                original = original_difference[row]
                model_rows[name].append(
                    {
                        "prompt_index": start + row,
                        "checkpoint": name,
                        "checkpoint_seed": int(payloads[name]["spec"]["seed"]),
                        "condition": str(payloads[name]["spec"]["method"]),
                        **checkpoint_pairs[name],
                        "correct_logit": float(reconstructed_correct_logits[row]),
                        "incorrect_logit": float(reconstructed_incorrect_logits[row]),
                        "reconstructed_logit_difference": float(value),
                        "absolute_logit_difference_error": float((value - original).abs()),
                        "reconstructed_preference": int(value >= 0),
                        "preferred_answer_agreement": int((value >= 0) == (original >= 0)),
                        "accuracy": int(value >= 0),
                    }
                )
            del logits
        print(f"ioi: {min(start + len(batch), len(examples))}/{len(examples)}", flush=True)

    identity = identity_gate(
        maximum=identity_max,
        total=identity_total,
        elements=identity_elements,
        max_tolerance=float(ioi["identity_max_abs_logit_difference"]),
        mean_tolerance=float(ioi["identity_max_mean_abs_logit_difference"]),
    )
    reports = {}
    for name, rows in model_rows.items():
        reports[name] = {
            "spec": dict(payloads[name]["spec"]),
            "absolute_logit_difference_error": sum(
                row["absolute_logit_difference_error"] for row in rows
            )
            / len(rows),
            "preferred_answer_agreement": sum(
                row["preferred_answer_agreement"] for row in rows
            )
            / len(rows),
            "accuracy": sum(row["accuracy"] for row in rows) / len(rows),
        }
    paired = []
    for seed, names in sorted(pair_names(payloads).items()):
        mse_rows = model_rows[names["mse"]]
        dpsae_rows = model_rows[names["dpsae"]]
        mse = reports[names["mse"]]
        dpsae = reports[names["dpsae"]]
        row: dict[str, Any] = {
            "seed": seed,
            "bootstrap_seed": int(ioi["bootstrap_seed"]) + seed,
            "baseline": names["mse"],
            "candidate": names["dpsae"],
            "absolute_logit_difference_error_dpsae_minus_mse": dpsae[
                "absolute_logit_difference_error"
            ]
            - mse["absolute_logit_difference_error"],
            "preferred_answer_agreement_dpsae_minus_mse": dpsae[
                "preferred_answer_agreement"
            ]
            - mse["preferred_answer_agreement"],
            "accuracy_dpsae_minus_mse": dpsae["accuracy"] - mse["accuracy"],
        }
        row.update(
            bootstrap_ioi_pair(
                mse_rows,
                dpsae_rows,
                samples=bootstrap_samples,
                seed=int(ioi["bootstrap_seed"]) + seed,
                quantiles=ioi["confidence_interval"],
            )
        )
        paired.append(row)
    original_accuracy = sum(row["original_accuracy"] for row in common_rows) / len(common_rows)
    _finalize_resources(resources, device, args.maximum_peak_gpu_gib)
    atomic_json(
        output,
        {
            "complete": True,
            "experiment": "exp09_frozen_network_ioi",
            "confirmatory": not smoke,
            "identity_hook": identity,
            "original_model": {"accuracy": original_accuracy},
            "models": reports,
            "paired": paired,
            "per_prompt": {
                "schema": "panel_g_endpoint_statistics_v1",
                "common": common_rows,
                "models": model_rows,
            },
            "inputs": inputs,
            "repository": repository,
            "protocol": protocol,
            "resources": resources,
            "wall_seconds": time.time() - started,
        },
    )
    return output


def validate_outputs(
    args: argparse.Namespace,
    config: Mapping[str, Any],
    repository: Mapping[str, Any],
) -> Path:
    natural_path = args.output_dir / "natural_results.json"
    ioi_path = args.output_dir / "ioi_results.json"
    prompts_path = args.output_dir / "ioi_prompts.json"
    natural = json.loads(natural_path.read_text())
    ioi = json.loads(ioi_path.read_text())
    if not natural.get("complete") or not ioi.get("complete"):
        raise ValueError("Exp09 stage output is incomplete")
    if natural.get("confirmatory") == args.smoke or ioi.get("confirmatory") == args.smoke:
        raise ValueError("Exp09 stage mode does not match validation mode")
    if not natural["identity_hook"]["passed"] or not ioi["identity_hook"]["passed"]:
        raise ValueError("Exp09 identity-hook gate did not pass")
    current_common = {
        "config": file_record(args.config),
        "calibration": validate_exact_file(
            args.calibration, config["calibration"], "activation calibration"
        ),
        **validate_checkpoint_provenance(
            models=args.models,
            training_done=args.training_done,
            confirmation_summary=args.confirmation_summary,
            config=config,
        ),
    }
    for stage_name, stage in (("natural", natural), ("ioi", ioi)):
        for name, record in current_common.items():
            if stage["inputs"].get(name) != record:
                raise ValueError(f"{stage_name} output used another {name}")
        for name, record in stage["inputs"].items():
            validate_record_is_current(record, f"{stage_name} {name}")
    if not args.smoke:
        expected_range = [int(value) for value in config["natural_text"]["absolute_range"]]
        if natural["protocol"]["absolute_range"] != expected_range:
            raise ValueError("confirmatory natural output used the wrong range")
        if not natural["protocol"]["fresh_range_opened"]:
            raise ValueError("confirmatory natural output does not identify the fresh range")
        if int(natural["protocol"]["sequences"]) != int(config["natural_text"]["sequences"]):
            raise ValueError("confirmatory natural output used the wrong sample size")
    output = args.output_dir / "completion_manifest.json"
    payload = {
        "complete": True,
        "experiment": "exp09_frozen_network",
        "confirmatory": not args.smoke,
        "natural_noninferiority_passed": bool(natural["primary_gate_passed"]),
        "allocation": dict(config["runpod"]),
        "inputs": {
            **current_common,
            "natural_results": file_record(natural_path),
            "ioi_results": file_record(ioi_path),
            "ioi_prompts": file_record(prompts_path),
        },
        "repository": repository,
    }
    if output.exists():
        existing = json.loads(output.read_text())
        if existing != payload:
            raise RuntimeError("existing Exp09 completion manifest is stale")
        return output
    atomic_json(output, payload)
    return output


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    repository = repository_state()
    if repository["dirty"] and not args.allow_dirty:
        raise RuntimeError(f"Exp09 requires a clean repository: {repository['status']}")
    if args.smoke and args.stage in {"prepare", "all"}:
        raise ValueError("smoke mode cannot dispatch the fresh-data prepare stage")
    if not args.smoke and args.smoke_cache is not None:
        raise ValueError("--smoke-cache requires --smoke")
    if args.smoke:
        args.output_dir = args.output_dir / "smoke"
    if args.natural_cache is None:
        args.natural_cache = args.output_dir / "natural_inputs.pt"
    if args.stage in {"prepare", "all"}:
        prepare_inputs(
            config,
            args.output_dir,
            repository,
            local_files_only=args.local_files_only,
        )
    if args.stage in {"natural", "all"}:
        run_natural(args, config, repository)
    if args.stage in {"ioi", "all"}:
        run_ioi(args, config, repository)
    if args.stage in {"validate", "all"}:
        validate_outputs(args, config, repository)


if __name__ == "__main__":
    main()
