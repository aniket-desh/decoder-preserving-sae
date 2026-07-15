import pytest
import torch

from experiments.exp02_prior_weight_sweep import paired_sweep_rows
from experiments.exp02_structured_prior import expected_structured_prior_diagnostics


def test_paired_sweep_uses_one_seed_matched_mse_baseline() -> None:
    rows = [
        {
            "seed": 0,
            "method": "mse",
            "relative_weight": None,
            "protected_decoder_distortion": 4.0,
            "test_nmse": 2.0,
        },
        {
            "seed": 0,
            "method": "task_prior",
            "relative_weight": 0.5,
            "task_weight": 3.0,
            "protected_decoder_distortion": 3.0,
            "test_nmse": 2.2,
        },
        {
            "seed": 0,
            "method": "task_prior",
            "relative_weight": 2.0,
            "task_weight": 12.0,
            "protected_decoder_distortion": 2.0,
            "test_nmse": 1.8,
        },
    ]

    paired = paired_sweep_rows(rows, [0.5, 2.0])

    assert paired[0]["protected_reduction_vs_mse"] == pytest.approx(0.25)
    assert paired[0]["nmse_reduction_vs_mse"] == pytest.approx(-0.1)
    assert paired[1]["protected_reduction_vs_mse"] == pytest.approx(0.5)
    assert paired[1]["nmse_reduction_vs_mse"] == pytest.approx(0.1)


def test_paired_sweep_rejects_missing_weight() -> None:
    rows = [
        {
            "seed": 0,
            "method": "mse",
            "protected_decoder_distortion": 4.0,
            "test_nmse": 2.0,
        }
    ]

    with pytest.raises(ValueError, match="incomplete paired sweep"):
        paired_sweep_rows(rows, [0.5])


def test_expected_structured_prior_diagnostics_matches_conditional_formula() -> None:
    diagnostics = expected_structured_prior_diagnostics(
        torch.eye(2),
        torch.eye(2),
        group_size=2,
        task_weight=2.0,
    )

    assert diagnostics["expected_prior_extra_trace_ratio_mean"] == pytest.approx(1.0)
    assert diagnostics["expected_prior_min_eigenvalue_mean"] == pytest.approx(2.0)
    assert diagnostics["expected_prior_max_eigenvalue_mean"] == pytest.approx(2.0)
    assert diagnostics["expected_prior_normalized_commutator_mean"] == pytest.approx(0.0)
