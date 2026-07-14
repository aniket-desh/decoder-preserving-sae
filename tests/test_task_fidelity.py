import torch

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
