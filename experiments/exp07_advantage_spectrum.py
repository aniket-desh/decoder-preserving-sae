#!/usr/bin/env python3
"""Bridge exact decoder distortion to taskwise advantage spectra."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from dpsae.mech_analysis import load_sae
from dpsae.task_fidelity import advantage_operators


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs/exp07_task_fidelity.json"


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


def sha256_file(path: Path, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def input_record(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def repository_state() -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=ROOT, text=True
    ).splitlines()
    return {"revision": revision, "dirty": bool(status), "status": status}


def quantiles(values: Tensor, probabilities: list[float]) -> dict[str, float]:
    values = values.detach().cpu().double().flatten()
    result = torch.quantile(values, torch.tensor(probabilities, dtype=torch.float64))
    return {f"q{int(100 * probability):02d}": float(value) for probability, value in zip(probabilities, result)}


def wilson_interval(successes: int, trials: int, z: float = 1.959963984540054) -> list[float]:
    proportion = successes / trials
    denominator = 1 + z * z / trials
    center = (proportion + z * z / (2 * trials)) / denominator
    half_width = z * math.sqrt(
        proportion * (1 - proportion) / trials + z * z / (4 * trials * trials)
    ) / denominator
    return [center - half_width, center + half_width]


@torch.inference_mode()
def reconstruct(model, activations: Tensor, *, batch_tokens: int, use_threshold: bool) -> Tensor:
    chunks = []
    for batch in activations.split(batch_tokens):
        reconstruction, _ = model(batch.float(), use_threshold=use_threshold)
        chunks.append(reconstruction)
    return torch.cat(chunks)


def separation_witness(config: dict[str, Any], output: Path) -> dict[str, Any]:
    settings = config["counterexample"]
    ridge = float(settings["ridge"])
    delta = float(settings["delta"])
    cases = []
    for label, a, b in (
        ("equal_aggregate_task_exchange", float(settings["equal_a"]), float(settings["equal_a"])),
        ("positive_trace_but_indefinite", float(settings["unequal_a"]), float(settings["unequal_b"])),
    ):
        source = torch.tensor([[[a, 0.0], [0.0, b]]], dtype=torch.float64)
        baseline = torch.tensor([[[a + delta, 0.0], [0.0, b]]], dtype=torch.float64)
        candidate = torch.tensor([[[a, 0.0], [0.0, b + delta]]], dtype=torch.float64)
        result = advantage_operators(source, baseline, candidate, ridge=ridge)
        baseline_mse = float((source - baseline).square().mean())
        candidate_mse = float((source - candidate).square().mean())
        baseline_l0 = float((baseline != 0).sum(2).double().mean())
        candidate_l0 = float((candidate != 0).sum(2).double().mean())
        eigenvalues = result["eigenvalues"][0]
        cases.append(
            {
                "case": label,
                "a": a,
                "b": b,
                "delta": delta,
                "ridge": ridge,
                "baseline_mse": baseline_mse,
                "candidate_mse": candidate_mse,
                "baseline_l0": baseline_l0,
                "candidate_l0": candidate_l0,
                "baseline_decoder_numerator": float(result["baseline_numerator"][0]),
                "candidate_decoder_numerator": float(result["candidate_numerator"][0]),
                "advantage_trace": float(result["trace"][0]),
                "advantage_eigenvalues": eigenvalues.tolist(),
                "mixed_sign": bool(eigenvalues[0] < 0 < eigenvalues[-1]),
            }
        )
    payload = {
        "complete": True,
        "experiment": "equal_mse_equal_l0_task_separation",
        "cases": cases,
        "checks": {
            "mse_and_l0_match": all(
                math.isclose(row["baseline_mse"], row["candidate_mse"], abs_tol=1e-14)
                and row["baseline_l0"] == row["candidate_l0"] == 1
                for row in cases
            ),
            "equal_case_matches_aggregate_distortion": math.isclose(
                cases[0]["baseline_decoder_numerator"],
                cases[0]["candidate_decoder_numerator"],
                rel_tol=1e-12,
                abs_tol=1e-14,
            ),
            "positive_trace_case_is_indefinite": (
                cases[1]["advantage_trace"] > 0 and cases[1]["mixed_sign"]
            ),
        },
        "inputs": {"config": input_record(Path(config["_config_path"]))},
        "repository": repository_state(),
    }
    if not all(payload["checks"].values()):
        raise RuntimeError(f"counterexample checks failed: {payload['checks']}")
    atomic_json(output / "counterexample.json", payload)
    return payload


def stored_exact_rows(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    payload = json.loads(path.read_text())
    rows = {str(row["model"]): row for row in payload["exact_rows"]}
    return rows, payload["paired_frontier"][0]


def direction_bank(groups: int, samples: int, directions: int, seed: int) -> Tensor:
    result = []
    for group in range(groups):
        generator = torch.Generator().manual_seed(seed + group)
        values = torch.randn(directions, samples, generator=generator, dtype=torch.float64)
        values /= values.norm(dim=1, keepdim=True).clamp_min(1e-30)
        result.append(values)
    return torch.stack(result)


def shared_direction_scores(directions: Tensor, operators: Tensor) -> Tensor:
    """Evaluate shared sample-coordinate tasks against batched operators."""

    if directions.ndim != 3 or operators.ndim != 3:
        raise ValueError("directions and operators must both be rank 3")
    if directions.shape[0] != operators.shape[0]:
        raise ValueError("directions and operators must have the same group count")
    if directions.shape[2] != operators.shape[1] or operators.shape[1] != operators.shape[2]:
        raise ValueError("operator dimensions must match the direction dimension")
    return torch.einsum("gdi,gij,gdj->gd", directions, operators, directions)


def spectrum_audit(
    config: dict[str, Any],
    output: Path,
    *,
    device: torch.device,
    gpu_memory_fraction: float,
    minimum_free_gib: float,
) -> dict[str, Any]:
    started = time.time()
    settings = config["advantage_spectrum"]
    models_path = ROOT / settings["models"]
    cache_path = ROOT / settings["cache"]
    confirmation_paths = {
        int(seed): ROOT / settings["confirmation_template"].format(seed=seed)
        for seed in settings["seeds"]
    }
    output.mkdir(parents=True, exist_ok=True)
    free_gib = shutil.disk_usage(output).free / 2**30
    if free_gib < minimum_free_gib:
        raise RuntimeError(f"only {free_gib:.2f} GiB free; guard requires {minimum_free_gib:.2f}")
    if device.type == "cuda":
        torch.cuda.set_per_process_memory_fraction(gpu_memory_fraction, device)
        torch.cuda.reset_peak_memory_stats(device)

    payloads = torch.load(models_path, map_location="cpu", weights_only=False)
    cache = torch.load(cache_path, map_location="cpu", weights_only=False)
    exact_tokens = int(settings["exact_tokens"])
    group_size = int(settings["group_size"])
    groups = exact_tokens // group_size
    if exact_tokens % group_size or exact_tokens > cache["activations"].numel() // cache["activations"].shape[-1]:
        raise ValueError("exact token budget must form complete available groups")
    flat_activations = cache["activations"].flatten(0, 1)[:exact_tokens].to(device).float()
    grouped_original = flat_activations.reshape(groups, group_size, -1).double()
    ridge = float(settings["ridge"])
    directions = direction_bank(
        groups,
        group_size,
        int(settings["random_directions_per_group"]),
        int(settings["random_direction_seed"]),
    )
    direction_scores = []
    seed_payloads = []

    for seed in settings["seeds"]:
        seed = int(seed)
        baseline_name = settings["baseline_template"].format(seed=seed)
        candidate_name = settings["candidate_template"].format(seed=seed)
        reconstructions = {}
        for name in (baseline_name, candidate_name):
            model = load_sae(payloads[name], input_dim=flat_activations.shape[-1], device=device)
            model.eval()
            reconstructions[name] = reconstruct(
                model,
                flat_activations,
                batch_tokens=int(settings["reconstruction_batch_tokens"]),
                use_threshold=True,
            ).reshape_as(grouped_original).double()
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        exact = advantage_operators(
            grouped_original,
            reconstructions[baseline_name],
            reconstructions[candidate_name],
            ridge=ridge,
        )
        raw_q = exact["advantage"]
        symmetry_residual = torch.linalg.matrix_norm(raw_q - raw_q.mT, ord="fro", dim=(1, 2)) / torch.linalg.matrix_norm(
            raw_q, ord="fro", dim=(1, 2)
        ).clamp_min(1e-30)
        if float(symmetry_residual.max()) > float(settings["maximum_symmetry_residual"]):
            raise RuntimeError(f"seed {seed} advantage operator failed symmetry gate")
        q = (raw_q + raw_q.mT) / 2
        eigenvalues = torch.linalg.eigvalsh(q)
        baseline_error_eigenvalues = torch.linalg.eigvalsh(exact["baseline_error"])
        candidate_error_eigenvalues = torch.linalg.eigvalsh(exact["candidate_error"])
        scale = baseline_error_eigenvalues.abs().amax(1).square() + candidate_error_eigenvalues.abs().amax(1).square()
        epsilon = torch.finfo(torch.float64).eps
        numerical_tolerance = torch.maximum(
            torch.full_like(scale, float(settings["minimum_numerical_tolerance"])),
            float(settings["roundoff_multiplier"]) * group_size * epsilon * scale,
        )
        baseline_average_error = exact["baseline_numerator"] / group_size
        material_tolerance = torch.maximum(
            1000 * numerical_tolerance,
            float(settings["material_fraction_of_baseline_average_error"]) * baseline_average_error,
        )

        trace_error = (
            exact["trace"]
            - (exact["baseline_numerator"] - exact["candidate_numerator"])
        ).abs()
        trace_limit = float(settings["group_trace_relative_tolerance"]) * torch.maximum(
            torch.ones_like(trace_error),
            exact["baseline_numerator"] + exact["candidate_numerator"],
        )
        if bool((trace_error > trace_limit).any()):
            raise RuntimeError(f"seed {seed} internal group trace reconciliation failed")
        summed_trace_error = abs(
            float(exact["trace"].sum())
            - float(exact["baseline_numerator"].sum() - exact["candidate_numerator"].sum())
        )
        summed_trace_limit = float(settings["sum_trace_relative_tolerance"]) * max(
            1.0,
            float(exact["baseline_numerator"].sum() + exact["candidate_numerator"].sum()),
        )
        if summed_trace_error > summed_trace_limit:
            raise RuntimeError(f"seed {seed} summed trace reconciliation failed")
        if float(eigenvalues.min()) < -1 - float(settings["eigenvalue_bound_tolerance"]) or float(eigenvalues.max()) > 1 + float(settings["eigenvalue_bound_tolerance"]):
            raise RuntimeError(f"seed {seed} eigenvalues exceeded the ridge-contraction bound")

        directions_device = directions.to(device)
        # Keep every random task in the shared sample-coordinate basis.  Reusing
        # eigenvalue coordinates would silently rotate the task bank separately
        # for each checkpoint seed, invalidating cross-seed paired comparisons.
        scores = shared_direction_scores(directions_device, q)
        direction_scores.append(scores.detach().cpu())
        group_rows = []
        for group in range(groups):
            eig = eigenvalues[group]
            num_tol = float(numerical_tolerance[group])
            mat_tol = float(material_tolerance[group])
            positive_mass = float(eig.clamp_min(0).sum())
            negative_mass = float((-eig.clamp_max(0)).sum())
            normalized_scores = scores[group] / baseline_average_error[group].clamp_min(1e-30)
            numerical_positive = int((scores[group] > num_tol).sum())
            numerical_negative = int((scores[group] < -num_tol).sum())
            material_positive = int((scores[group] > mat_tol).sum())
            material_negative = int((scores[group] < -mat_tol).sum())
            trials = scores.shape[1]
            group_rows.append(
                {
                    "group": group,
                    "baseline_numerator": float(exact["baseline_numerator"][group]),
                    "candidate_numerator": float(exact["candidate_numerator"][group]),
                    "source_energy": float(exact["source_energy"][group]),
                    "trace": float(exact["trace"][group]),
                    "trace_numerical_tolerance": group_size * num_tol,
                    "numerically_positive_trace": bool(
                        exact["trace"][group] > group_size * numerical_tolerance[group]
                    ),
                    "group_reduction": float(exact["trace"][group] / exact["baseline_numerator"][group].clamp_min(1e-30)),
                    "minimum_eigenvalue": float(eig[0]),
                    "maximum_eigenvalue": float(eig[-1]),
                    "numerical_tolerance": num_tol,
                    "material_tolerance": mat_tol,
                    "numerical_positive_eigenvalues": int((eig > num_tol).sum()),
                    "numerical_negative_eigenvalues": int((eig < -num_tol).sum()),
                    "material_positive_eigenvalues": int((eig > mat_tol).sum()),
                    "material_negative_eigenvalues": int((eig < -mat_tol).sum()),
                    "positive_spectral_mass": positive_mass,
                    "negative_spectral_mass": negative_mass,
                    "largest_positive_mode_share": float(eig[-1].clamp_min(0) / max(positive_mass, 1e-30)),
                    "largest_negative_mode_share": float((-eig[0].clamp_max(0)) / max(negative_mass, 1e-30)),
                    "materially_indefinite": bool((eig > mat_tol).any() and (eig < -mat_tol).any()),
                    "material_psd_dominance": bool(not (eig < -mat_tol).any()),
                    "random_direction_numerical_positive_probability": numerical_positive / trials,
                    "random_direction_numerical_positive_wilson95": wilson_interval(numerical_positive, trials),
                    "random_direction_numerical_negative_probability": numerical_negative / trials,
                    "random_direction_numerical_negative_wilson95": wilson_interval(numerical_negative, trials),
                    "random_direction_material_positive_probability": material_positive / trials,
                    "random_direction_material_positive_wilson95": wilson_interval(material_positive, trials),
                    "random_direction_material_negative_probability": material_negative / trials,
                    "random_direction_material_negative_wilson95": wilson_interval(material_negative, trials),
                    "random_direction_relative_advantage_quantiles": quantiles(
                        normalized_scores,
                        [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99],
                    ),
                    "random_direction_mean_score": float(scores[group].mean()),
                    "expected_random_direction_mean_score": float(
                        exact["trace"][group] / group_size
                    ),
                }
            )

        stored_rows, stored_pair = stored_exact_rows(confirmation_paths[seed])
        stored_baseline = torch.tensor(stored_rows[baseline_name]["numerator_by_group"], dtype=torch.float64)
        stored_candidate = torch.tensor(stored_rows[candidate_name]["numerator_by_group"], dtype=torch.float64)
        stored_baseline_relative_error = abs(float(exact["baseline_numerator"].sum().cpu()) - float(stored_baseline.sum())) / float(stored_baseline.sum())
        stored_candidate_relative_error = abs(float(exact["candidate_numerator"].sum().cpu()) - float(stored_candidate.sum())) / float(stored_candidate.sum())
        stored_tolerance = float(settings["stored_fp32_relative_tolerance"])
        if max(stored_baseline_relative_error, stored_candidate_relative_error) > stored_tolerance:
            raise RuntimeError(f"seed {seed} does not reconcile with stored exact numerators")

        trace = exact["trace"].detach().cpu()
        trace_numerical_tolerance = group_size * numerical_tolerance.detach().cpu()
        positive_trace = torch.where(
            trace > trace_numerical_tolerance,
            trace,
            torch.zeros_like(trace),
        )
        positive_trace_sorted = positive_trace.sort(descending=True).values
        top_five_count = max(1, math.ceil(0.05 * groups))
        reductions = trace / exact["baseline_numerator"].detach().cpu().clamp_min(1e-30)
        numerical_positive_probabilities = torch.tensor(
            [row["random_direction_numerical_positive_probability"] for row in group_rows]
        )
        material_positive_probabilities = torch.tensor(
            [row["random_direction_material_positive_probability"] for row in group_rows]
        )
        numerical_negative_probabilities = torch.tensor(
            [row["random_direction_numerical_negative_probability"] for row in group_rows]
        )
        material_negative_probabilities = torch.tensor(
            [row["random_direction_material_negative_probability"] for row in group_rows]
        )
        expected_direction_mean = trace / group_size
        sampled_direction_means = scores.detach().cpu().mean(1)
        direction_mean_mc_se = torch.sqrt(
            scores.detach().cpu().var(1, unbiased=True).sum()
            / (groups * groups * scores.shape[1])
        )
        direction_mean_error = float(
            sampled_direction_means.mean() - expected_direction_mean.mean()
        )
        seed_summary = {
            "seed": seed,
            "baseline": baseline_name,
            "candidate": candidate_name,
            "groups": groups,
            "samples_per_group": group_size,
            "summed_trace": float(trace.sum()),
            "headline_distortion_difference": float(trace.sum() / exact["source_energy"].sum().cpu()),
            "paired_reduction": float(trace.sum() / exact["baseline_numerator"].sum().cpu()),
            "stored_paired_reduction": float(stored_pair["exact_decoder_reduction"]),
            "stored_baseline_numerator_relative_error": stored_baseline_relative_error,
            "stored_candidate_numerator_relative_error": stored_candidate_relative_error,
            "maximum_symmetry_residual": float(symmetry_residual.max()),
            "maximum_group_trace_reconciliation_error": float(trace_error.max()),
            "summed_trace_reconciliation_error": summed_trace_error,
            "fraction_groups_numerically_positive_trace": float(
                (trace > trace_numerical_tolerance).double().mean()
            ),
            "fraction_groups_materially_indefinite": sum(row["materially_indefinite"] for row in group_rows) / groups,
            "fraction_groups_material_psd_dominance": sum(row["material_psd_dominance"] for row in group_rows) / groups,
            "group_reduction_quantiles": quantiles(reductions, [0.10, 0.25, 0.50, 0.75, 0.90]),
            "top_one_group_share_of_positive_trace": float(positive_trace_sorted[0] / positive_trace.sum().clamp_min(1e-30)),
            "top_five_percent_groups_share_of_positive_trace": float(positive_trace_sorted[:top_five_count].sum() / positive_trace.sum().clamp_min(1e-30)),
            "total_positive_spectral_mass": sum(row["positive_spectral_mass"] for row in group_rows),
            "total_negative_spectral_mass": sum(row["negative_spectral_mass"] for row in group_rows),
            "mean_random_direction_numerical_positive_probability": float(numerical_positive_probabilities.mean()),
            "conditional_mc_se_random_direction_numerical_positive": float(
                torch.sqrt((numerical_positive_probabilities * (1 - numerical_positive_probabilities)).sum()) / (groups * math.sqrt(scores.shape[1]))
            ),
            "mean_random_direction_material_positive_probability": float(material_positive_probabilities.mean()),
            "conditional_mc_se_random_direction_material_positive": float(
                torch.sqrt((material_positive_probabilities * (1 - material_positive_probabilities)).sum()) / (groups * math.sqrt(scores.shape[1]))
            ),
            "mean_random_direction_numerical_negative_probability": float(
                numerical_negative_probabilities.mean()
            ),
            "mean_random_direction_material_negative_probability": float(
                material_negative_probabilities.mean()
            ),
            "sampled_random_direction_mean_score": float(sampled_direction_means.mean()),
            "expected_random_direction_mean_score": float(expected_direction_mean.mean()),
            "random_direction_mean_score_error": direction_mean_error,
            "conditional_mc_se_random_direction_mean_score": float(direction_mean_mc_se),
            "random_direction_mean_score_error_z": direction_mean_error
            / max(float(direction_mean_mc_se), 1e-30),
        }
        seed_payload = {
            "complete": True,
            "experiment": "taskwise_decoder_advantage_spectrum",
            "summary": seed_summary,
            "groups": group_rows,
            "eigenvalues": eigenvalues.detach().cpu().tolist(),
        }
        atomic_json(output / f"advantage_spectrum_seed{seed}.json", seed_payload)
        seed_payloads.append(seed_payload)
        del reconstructions, exact, q, eigenvalues, scores, directions_device
        if device.type == "cuda":
            torch.cuda.empty_cache()

    score_tensor = torch.stack(direction_scores)
    material_tolerances = torch.tensor(
        [[row["material_tolerance"] for row in payload["groups"]] for payload in seed_payloads],
        dtype=torch.float64,
    )
    numerical_tolerances = torch.tensor(
        [[row["numerical_tolerance"] for row in payload["groups"]] for payload in seed_payloads],
        dtype=torch.float64,
    )
    shared_numerical = (score_tensor > numerical_tolerances[:, :, None]).all(0)
    shared_material = (score_tensor > material_tolerances[:, :, None]).all(0)
    aligned_positive_trace = torch.tensor(
        [
            [row["numerically_positive_trace"] for row in payload["groups"]]
            for payload in seed_payloads
        ]
    ).all(0)
    aligned_indefinite = torch.tensor(
        [[row["materially_indefinite"] for row in payload["groups"]] for payload in seed_payloads]
    ).all(0)
    atomic_torch(
        output / "random_direction_scores.pt",
        {
            "scores": score_tensor,
            "material_tolerances": material_tolerances,
            "numerical_tolerances": numerical_tolerances,
            "direction_seed": int(settings["random_direction_seed"]),
        },
    )
    resources = {
        "device": str(device),
        "free_gib_at_start": free_gib,
        "minimum_free_gib_guard": minimum_free_gib,
        "gpu_memory_fraction_cap": gpu_memory_fraction if device.type == "cuda" else None,
        "peak_allocated_gpu_gib": (
            torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else None
        ),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
    }
    summary = {
        "complete": True,
        "experiment": "taskwise_decoder_advantage_spectrum_summary",
        "seed_summaries": [payload["summary"] for payload in seed_payloads],
        "across_seeds": {
            "fraction_aligned_group_positions_numerically_positive_trace_all_seeds": float(aligned_positive_trace.double().mean()),
            "fraction_aligned_group_positions_materially_indefinite_all_seeds": float(aligned_indefinite.double().mean()),
            "probability_shared_random_direction_numerically_improves_all_seeds": float(shared_numerical.double().mean()),
            "conditional_mc_se_shared_random_direction_numerically_improves_all_seeds": float(
                torch.sqrt(
                    (
                        shared_numerical.double().mean(1)
                        * (1 - shared_numerical.double().mean(1))
                    ).sum()
                )
                / (groups * math.sqrt(shared_numerical.shape[1]))
            ),
            "probability_shared_random_direction_materially_improves_all_seeds": float(shared_material.double().mean()),
            "conditional_mc_se_shared_random_direction_materially_improves_all_seeds": float(
                torch.sqrt(
                    (
                        shared_material.double().mean(1)
                        * (1 - shared_material.double().mean(1))
                    ).sum()
                )
                / (groups * math.sqrt(shared_material.shape[1]))
            ),
        },
        "protocol": {
            **settings,
            "random_direction_bank_reused_across_seeds": True,
            "spectrum_precision": "float64",
            "reconstruction_support": "stored inference threshold, matching headline exact audit",
        },
        "inputs": {
            "models": input_record(models_path),
            "cache": input_record(cache_path),
            "config": input_record(Path(config["_config_path"])),
            "evaluator": input_record(Path(__file__)),
            "task_fidelity_module": input_record(ROOT / "src/dpsae/task_fidelity.py"),
            "confirmation": {str(seed): input_record(path) for seed, path in confirmation_paths.items()},
        },
        "repository": repository_state(),
        "resources": resources,
        "wall_seconds": time.time() - started,
    }
    atomic_json(output / "advantage_spectrum_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["counterexample", "spectrum", "all"])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--gpu-memory-fraction", type=float, default=0.25)
    parser.add_argument("--minimum-free-gib", type=float, default=20.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text())
    config["_config_path"] = str(args.config.resolve())
    output = args.output or ROOT / config["output"]
    if args.mode in {"counterexample", "all"}:
        print(json.dumps(separation_witness(config, output)["checks"], indent=2), flush=True)
    if args.mode in {"spectrum", "all"}:
        summary = spectrum_audit(
            config,
            output,
            device=torch.device(args.device),
            gpu_memory_fraction=args.gpu_memory_fraction,
            minimum_free_gib=args.minimum_free_gib,
        )
        print(json.dumps(summary["across_seeds"], indent=2), flush=True)


if __name__ == "__main__":
    main()
