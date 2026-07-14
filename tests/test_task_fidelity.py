import torch

from experiments.exp07_advantage_spectrum import shared_direction_scores
from experiments.exp07_gradient_fidelity import (
    block_bootstrap_gradient_summary,
    block_gradient_statistics,
    expected_gradient_metrics_from_weights,
    hierarchical_expected_gradient_summary,
)
from dpsae.decoder_distance import batched_ridge_predict
from dpsae.language_training import sampled_decoder_loss_from_reference
from dpsae.task_fidelity import (
    advantage_operators,
    exact_relative_gradients,
    fixed_radius_targets,
    ridge_gradient_factors,
    ridge_hat_matrices,
    sampled_relative_gradients,
)


def test_ridge_hat_matrices_match_identity_predictions():
    groups = torch.randn(3, 5, 7, dtype=torch.float64)
    identity = torch.eye(5, dtype=torch.float64).expand(3, 5, 5)
    expected = batched_ridge_predict(groups, identity, 0.4)
    assert torch.allclose(ridge_hat_matrices(groups, 0.4), expected)


def test_equal_mse_sparse_witness_has_mixed_task_advantage():
    source = torch.tensor([[[1.0, 0.0], [0.0, 2.0]]], dtype=torch.float64)
    baseline = torch.tensor([[[1.1, 0.0], [0.0, 2.0]]], dtype=torch.float64)
    candidate = torch.tensor([[[1.0, 0.0], [0.0, 2.1]]], dtype=torch.float64)
    assert torch.allclose((source - baseline).square().sum(), (source - candidate).square().sum())
    assert (baseline != 0).sum(2).float().mean() == 1
    assert (candidate != 0).sum(2).float().mean() == 1

    result = advantage_operators(source, baseline, candidate, ridge=0.5)
    eigenvalues = result["eigenvalues"][0]
    assert eigenvalues[0] < 0 < eigenvalues[1]
    assert result["trace"][0] > 0
    assert torch.allclose(
        result["trace"],
        result["baseline_numerator"] - result["candidate_numerator"],
        atol=1e-12,
    )


def test_shared_direction_scores_stay_in_sample_coordinates():
    directions = torch.tensor(
        [[[1.0, 0.0], [0.0, 1.0], [2**-0.5, 2**-0.5]]],
        dtype=torch.float64,
    )
    operators = torch.tensor(
        [
            [[2.0, 0.0], [0.0, -1.0]],
            [[-1.0, 0.0], [0.0, 2.0]],
        ],
        dtype=torch.float64,
    )
    repeated_directions = directions.expand(2, -1, -1)
    scores = shared_direction_scores(repeated_directions, operators)
    direct = torch.stack(
        [
            torch.tensor(
                [float(u @ operator @ u) for u in directions[0]],
                dtype=torch.float64,
            )
            for operator in operators
        ]
    )
    assert torch.allclose(scores, direct)
    assert not torch.equal(scores[0], scores[1])


def test_block_statistics_recover_u_statistical_gradient_bias():
    exact = torch.tensor([1.0, -0.5], dtype=torch.float64)
    sampled = torch.tensor(
        [[1.2, -0.4], [0.9, -0.7], [1.1, -0.3], [1.0, -0.6]],
        dtype=torch.float64,
    )
    fixed = torch.tensor(
        [[1.1, -0.5], [0.9, -0.6], [1.0, -0.4], [1.0, -0.5]],
        dtype=torch.float64,
    )
    statistics = block_gradient_statistics(sampled, fixed, exact)
    weights = torch.ones(1, len(sampled), dtype=torch.float64)
    sampled_metrics = expected_gradient_metrics_from_weights(
        statistics,
        weights,
        estimator="sampled",
    )
    paired_metrics = expected_gradient_metrics_from_weights(
        statistics,
        weights,
        estimator="sampled_minus_fixed",
    )

    residual = sampled - exact
    paired = sampled - fixed
    off_diagonal_residual = residual @ residual.mT
    off_diagonal_paired = paired @ paired.mT
    expected_bias_squared = (
        off_diagonal_residual.sum() - off_diagonal_residual.diagonal().sum()
    ) / (len(sampled) * (len(sampled) - 1) * exact.square().sum())
    expected_paired_squared = (
        off_diagonal_paired.sum() - off_diagonal_paired.diagonal().sum()
    ) / (len(sampled) * (len(sampled) - 1) * exact.square().sum())
    assert torch.allclose(
        sampled_metrics["relative_bias_squared_unclamped"],
        expected_bias_squared.reshape(1),
    )
    assert torch.allclose(
        paired_metrics["relative_bias_squared_unclamped"],
        expected_paired_squared.reshape(1),
    )
    bootstrap = block_bootstrap_gradient_summary(statistics, samples=64, seed=11)
    assert set(bootstrap) == {"sampled", "fixed", "sampled_minus_fixed"}
    assert len(bootstrap["sampled"]["cosine"]["bootstrap95"]) == 2

    raw = {
        str(batch): {
            "16": {
                "row_gram": {"bootstrap_sufficient_statistics": statistics}
            }
        }
        for batch in range(2)
    }
    hierarchical = hierarchical_expected_gradient_summary(
        raw,
        {
            "bootstrap_samples": 64,
            "batches": 2,
            "bootstrap_blocks": 4,
            "bootstrap_seed": 13,
        },
        probes=16,
        space="row_gram",
    )
    assert len(
        hierarchical["median_expected_relative_bias"][
            "hierarchical_bootstrap95"
        ]
    ) == 2


def test_fixed_radius_targets_have_training_norm():
    generator = torch.Generator().manual_seed(7)
    targets, clamp_hits = fixed_radius_targets(
        4,
        3,
        8,
        5,
        generator=generator,
        device=torch.device("cpu"),
        dtype=torch.float64,
    )
    assert clamp_hits == 0
    expected = torch.full((4, 3, 5), 8.0, dtype=torch.float64)
    assert torch.allclose(targets.square().sum(2), expected, atol=1e-12)


def test_closed_form_sampled_reconstruction_gradient_matches_autograd():
    torch.manual_seed(3)
    original = torch.randn(2, 4, 6, dtype=torch.float64)
    reconstructed = torch.randn(2, 4, 6, dtype=torch.float64)
    targets = torch.randn(2, 4, 3, dtype=torch.float64)
    targets /= targets.square().mean(1, keepdim=True).sqrt()
    covariance = (targets @ targets.mT).unsqueeze(0)

    factors = ridge_gradient_factors(original, reconstructed, ridge=0.7)
    _, analytic, denominator = sampled_relative_gradients(
        factors,
        covariance,
        probes=3,
    )
    variable = reconstructed.clone().requires_grad_(True)
    loss = sampled_decoder_loss_from_reference(
        original,
        variable,
        targets,
        ridge=0.7,
    )
    loss.backward()
    assert denominator.shape == (1,)
    assert torch.allclose(analytic[0], variable.grad, atol=1e-9, rtol=1e-8)


def test_closed_form_exact_gradient_matches_identity_autograd():
    torch.manual_seed(4)
    original = torch.randn(2, 4, 6, dtype=torch.float64)
    reconstructed = torch.randn(2, 4, 6, dtype=torch.float64)
    identity = torch.eye(4, dtype=torch.float64).expand(2, 4, 4)
    factors = ridge_gradient_factors(original, reconstructed, ridge=0.3)
    _, analytic = exact_relative_gradients(factors)

    variable = reconstructed.clone().requires_grad_(True)
    loss = sampled_decoder_loss_from_reference(
        original,
        variable,
        identity,
        ridge=0.3,
    )
    loss.backward()
    assert torch.allclose(analytic, variable.grad, atol=1e-9, rtol=1e-8)
