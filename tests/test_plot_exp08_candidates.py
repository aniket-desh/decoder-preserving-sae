import csv
import hashlib
import json
from pathlib import Path

import pytest

from scripts import plot_exp08_candidates as candidates


REPOSITORY = {"revision": "abc123", "dirty": False, "status": []}
WEIGHTS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 4.0]


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _external_record(path: Path) -> dict:
    return {"path": str(path), "bytes": path.stat().st_size, "sha256": _hash(path)}


def _build_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    experiment = tmp_path / "artifacts/exp08_experiment_figure"
    structured = tmp_path / "source/experiments/outputs/exp02_structured_prior"
    static_path = tmp_path / "source/artifacts/natural_evaluation_baseline.json"

    metrics = []
    groups = []
    methods = ("mse", *candidates.STRUCTURED_METHODS)
    factors = {
        "mse": (1.0, 1.0, 1.0, 1.0),
        "task_prior": (0.99, 0.72, 0.91, 0.86),
        "isotropic": (1.02, 0.88, 0.84, 0.86),
        "weighted_mse": (1.01, 0.80, 0.98, 0.94),
        "permuted_prior": (1.01, 1.03, 1.00, 1.01),
    }
    for seed in range(10):
        for method in methods:
            nmse, protected, unrelated, isotropic = factors[method]
            metrics.append(
                {
                    "seed": seed,
                    "method": method,
                    "test_nmse": 0.1 * nmse * (1 + seed / 100),
                    "protected_decoder_distortion": 0.2
                    * protected
                    * (1 + seed / 100),
                    "unrelated_decoder_distortion": 0.15
                    * unrelated
                    * (1 + seed / 100),
                    "isotropic_decoder_distortion": 0.18
                    * isotropic
                    * (1 + seed / 100),
                }
            )
            gain = {
                "mse": 0.0,
                "task_prior": 0.12,
                "isotropic": 0.05,
                "weighted_mse": 0.08,
                "permuted_prior": -0.01,
            }[method]
            groups.append(
                {
                    "seed": seed,
                    "method": method,
                    "group": "protected",
                    "matched_cosine": 0.4 + gain + seed / 1000,
                    "support_f1": 0.3 + gain / 2 + seed / 1000,
                }
            )
    _write_csv(structured / "metrics.csv", metrics)
    _write_csv(structured / "group_metrics.csv", groups)
    _write_json(
        structured / "metadata.json",
        {"smoke": False, "config": {"seeds": list(range(10))}},
    )
    (structured / "crossover.csv").write_text("relative_weight\n1.0\n")

    static_models = {}
    control_factors = {
        "mse": (1.0, 1.0),
        "dpsae": (1.005, 0.76),
        "whitening": (1.05, 1.12),
        "spectral": (1.01, 0.995),
    }
    for seed in candidates.EXPECTED_SEEDS:
        for method, (nmse_factor, decoder_factor) in control_factors.items():
            static_models[f"{method}_s{seed}"] = {
                "spec": {"method": method, "seed": seed},
                "sampled_primary": {"nmse": 0.025 * nmse_factor},
                "exact_identity_primary": {
                    "decoder_distortion": 0.04 * decoder_factor
                },
            }
    _write_json(
        static_path,
        {
            "complete": True,
            "protocol": {
                "repository": {"revision": "old-clean", "dirty": False}
            },
            "models": static_models,
        },
    )

    sweep_rows = []
    for seed in range(10):
        for weight in WEIGHTS:
            sweep_rows.append(
                {
                    "seed": seed,
                    "relative_weight": weight,
                    "protected_reduction_vs_mse": 0.04 * weight + seed / 1000,
                    "nmse_reduction_vs_mse": -0.002 * weight,
                }
            )
    _write_csv(experiment / "synthetic_prior_sweep/paired_metrics.csv", sweep_rows)
    _write_json(
        experiment / "synthetic_prior_sweep/metadata.json",
        {
            "complete": True,
            "experiment": "exp02_prior_weight_sweep",
            "git_revision": REPOSITORY["revision"],
            "config": {
                "seeds": list(range(10)),
                "empirical_crossover": {
                    "relative_weights": WEIGHTS,
                    "relative_weight_reference": (
                        "separate_two_direction_crossover_not_a_sparse_transition"
                    ),
                },
            },
            "weight_parameterization": {
                "relative_weight": (
                    "task weight divided by a separate 2D reference; "
                    "not a predicted sparse transition"
                )
            },
        },
    )

    gamma_rows = []
    for index, weight in enumerate((0.03125, 0.0625, 0.09375, 0.125, 0.25, 0.5, 1.0)):
        reduction = 0.11 + 0.025 * index
        gamma_rows.append(
            {
                "decoder_weight": weight,
                "nmse_change_percent": -0.2 + 0.25 * index,
                "exact_decoder_reduction": reduction,
                "exact_decoder_reduction_ci95": [reduction - 0.01, reduction + 0.01],
            }
        )
    _write_json(
        experiment / "gamma_sweep_selection.json",
        {
            "complete": True,
            "experiment": "paper_closure_frontier_existing",
            "repository": REPOSITORY,
            "paired_frontier": gamma_rows,
        },
    )
    _write_json(
        experiment / "gamma_sweep_choice.json",
        {
            "complete": True,
            "experiment": "paper_closure_frontier_selection",
            "repository": REPOSITORY,
            "selected_decoder_weight": 0.03125,
            "selection_rule": {
                "maximum_nmse_ratio": 1.01,
                "minimum_exact_decoder_reduction": 0.1,
            },
        },
    )

    confirmation = []
    for seed in candidates.EXPECTED_SEEDS:
        confirmation.append(
            {
                "seed": seed,
                "mse": {"nmse": 0.025, "exact_decoder_distortion": 0.040},
                "dpsae": {
                    "nmse": 0.0249 - seed / 100000,
                    "exact_decoder_distortion": 0.035 - seed / 10000,
                },
                "nmse_change_percent": -0.4 - seed / 10,
                "exact_decoder_reduction": 0.125 + seed / 100,
            }
        )
    _write_json(
        experiment / "confirmation_summary.json",
        {
            "complete": True,
            "experiment": "exp08_clean_confirmation_summary",
            "repository": REPOSITORY,
            "gate_passed": True,
            "rows": confirmation,
        },
    )

    frozen_rows = []
    for seed in candidates.EXPECTED_SEEDS:
        frozen_rows.append(
            {
                "seed": seed,
                "loss_recovered_difference_dpsae_minus_mse": 0.01 + seed / 1000,
                "loss_recovered_difference_dpsae_minus_mse_ci95": [-0.005, 0.025],
                "kl_difference_dpsae_minus_mse": -0.0004 - seed / 100000,
                "kl_difference_dpsae_minus_mse_ci95": [-0.0008, 0.0001],
                "cross_entropy_increase_difference_dpsae_minus_mse": -0.0003,
                "cross_entropy_increase_difference_dpsae_minus_mse_ci95": [
                    -0.0007,
                    0.0001,
                ],
                "top1_agreement_difference_dpsae_minus_mse": 0.002,
                "top1_agreement_difference_dpsae_minus_mse_ci95": [-0.001, 0.005],
                "activation_nmse_ratio_dpsae_to_mse": 0.998 + seed / 1000,
                "inference_l0_difference_dpsae_minus_mse": -0.1 + seed / 10,
            }
        )
    _write_json(
        experiment / "evidence/frozen_fidelity.json",
        {
            "complete": True,
            "experiment": "exp08_frozen_language_model_fidelity",
            "repository": REPOSITORY,
            "paired_differences": frozen_rows,
        },
    )

    settings = []
    robustness_rows = []
    axis_values = {
        "ridge": [("dof=0.125", 0.125), ("dof=0.25", 0.25), ("dof=0.5", 0.5)],
        "group_size": [("n=64", 64), ("n=128", 128), ("n=256", 256)],
        "grouping": [
            ("contiguous", "contiguous"),
            ("shuffled", "shuffled"),
            ("document balanced", "document_balanced"),
        ],
    }
    for axis, entries in axis_values.items():
        for index, (label, value) in enumerate(entries):
            settings.append(
                {
                    "audit_axis": axis,
                    "setting_label": label,
                    "setting_value": value,
                }
            )
            for seed in candidates.EXPECTED_SEEDS:
                robustness_rows.append(
                    {
                        "seed": seed,
                        "audit_axis": axis,
                        "setting_label": label,
                        "decoder_reduction_vs_mse": 0.1 + index / 100 + seed / 1000,
                    }
                )
    _write_json(
        experiment / "evidence/robustness.json",
        {
            "complete": True,
            "experiment": "exp08_matched_quality_robustness",
            "repository": REPOSITORY,
            "protocol": {"expected_seeds": list(candidates.EXPECTED_SEEDS)},
            "settings": settings,
            "paired_reductions": robustness_rows,
        },
    )
    _write_json(
        experiment / "task_spectrum/advantage_spectrum_summary.json",
        {
            "complete": True,
            "experiment": "taskwise_decoder_advantage_spectrum_summary",
            "repository": REPOSITORY,
            "seed_summaries": [
                {
                    "seed": seed,
                    "mean_random_direction_material_positive_probability": 0.62
                    + seed / 100,
                    "mean_random_direction_material_negative_probability": 0.18
                    - seed / 100,
                }
                for seed in candidates.EXPECTED_SEEDS
            ],
        },
    )

    external = {
        "static_baseline_evaluation": static_path,
        "structured_baseline_metrics": structured / "metrics.csv",
        "structured_baseline_group_metrics": structured / "group_metrics.csv",
        "structured_baseline_metadata": structured / "metadata.json",
        "structured_baseline_crossover": structured / "crossover.csv",
    }
    code_paths = (
        "scripts/plot_exp08_candidates.py",
        "src/dpsae/plot_style.py",
    )
    _write_json(
        experiment / "run_manifest.json",
        {
            "complete": True,
            "experiment": "exp08_experiment_figure_closure",
            "repository": REPOSITORY,
            "code": {
                relative: _external_record(candidates.ROOT / relative)
                for relative in code_paths
            },
            "external_inputs": {
                key: _external_record(path) for key, path in external.items()
            },
        },
    )
    return experiment, structured, static_path, experiment / "candidate_figures"


def test_transformed_interval_preserves_positive_axis_semantics() -> None:
    assert candidates.transformed_interval(
        -0.2, [-0.3, -0.1], sign=-1, scale=100
    ) == pytest.approx((20, 10, 30))


def test_end_to_end_candidate_render_and_manifest(tmp_path: Path, monkeypatch) -> None:
    experiment, structured, static_path, output = _build_inputs(tmp_path)
    monkeypatch.setattr(
        "sys.argv",
        [
            "plot_exp08_candidates.py",
            "--experiment-root",
            str(experiment),
            "--structured-baseline-dir",
            str(structured),
            "--static-baseline",
            str(static_path),
            "--output-dir",
            str(output),
        ],
    )

    candidates.main()

    expected = (
        "task_prior_candidates.pdf",
        "task_prior_candidates.png",
        "language_model_candidates.pdf",
        "language_model_candidates.png",
        "frozen_fidelity_review.pdf",
        "frozen_fidelity_review.png",
        "robustness_appendix.pdf",
        "robustness_appendix.png",
        "candidate_manifest.json",
    )
    assert all((output / name).stat().st_size > 0 for name in expected)
    manifest = json.loads((output / "candidate_manifest.json").read_text())
    assert manifest["complete"]
    assert not manifest["relative_weight_semantics"]["sparse_transition_claim"]
    assert manifest["status"] == "review_only_not_integrated_into_manuscript"


def test_renderer_rejects_output_outside_experiment_root(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be written exactly"):
        candidates.validate_output_location(tmp_path / "experiment", tmp_path / "paper")
