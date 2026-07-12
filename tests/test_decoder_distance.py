import pytest
import torch

from dpsae.decoder_distance import (
    batched_ridge_predict,
    calibrate_ridge,
    decoder_distance,
    effective_degrees_of_freedom,
    ridge_hat_matrix,
    ridge_predict,
    sampled_decoder_loss,
)


def test_identical_representations_have_zero_distance() -> None:
    x = torch.randn(16, 7, dtype=torch.float64)
    assert decoder_distance(x, x, ridge=0.5).item() == pytest.approx(0.0, abs=1e-12)


def test_isotropic_covariance_matches_unweighted_distance() -> None:
    x = torch.randn(12, 5, dtype=torch.float64)
    x_hat = x + 0.2 * torch.randn_like(x)
    identity = torch.eye(x.shape[0], dtype=x.dtype)
    plain = decoder_distance(x, x_hat, ridge=1.3, reduction="sum")
    weighted = decoder_distance(
        x, x_hat, ridge=1.3, task_covariance=identity, reduction="sum"
    )
    assert weighted.item() == pytest.approx(plain.item(), rel=1e-12, abs=1e-12)


def test_hat_matrix_is_invariant_to_feature_rotation() -> None:
    x = torch.randn(20, 6, dtype=torch.float64)
    q, _ = torch.linalg.qr(torch.randn(6, 6, dtype=torch.float64))
    k_x = ridge_hat_matrix(x, ridge=0.7)
    k_rotated = ridge_hat_matrix(x @ q, ridge=0.7)
    torch.testing.assert_close(k_x, k_rotated, rtol=1e-10, atol=1e-10)


def test_hat_matrix_scale_and_ridge_invariance() -> None:
    x = torch.randn(20, 6, dtype=torch.float64)
    scale = 3.7
    torch.testing.assert_close(
        ridge_hat_matrix(x, ridge=0.4),
        ridge_hat_matrix(scale * x, ridge=scale**2 * 0.4),
        rtol=1e-10,
        atol=1e-10,
    )


def test_ridge_predict_matches_explicit_hat_matrix() -> None:
    x = torch.randn(11, 5, dtype=torch.float64)
    targets = torch.randn(11, 3, dtype=torch.float64)
    predicted = ridge_predict(x, targets, ridge=0.2)
    explicit = ridge_hat_matrix(x, ridge=0.2) @ targets
    torch.testing.assert_close(predicted, explicit, rtol=1e-10, atol=1e-10)


def test_batched_ridge_predict_matches_group_loop() -> None:
    x = torch.randn(4, 12, 5, dtype=torch.float64)
    targets = torch.randn(4, 12, 3, dtype=torch.float64)
    batched = batched_ridge_predict(x, targets, ridge=0.2)
    loop = torch.stack(
        [
            ridge_predict(x_group, y_group, ridge=0.2)
            for x_group, y_group in zip(x, targets)
        ]
    )
    torch.testing.assert_close(batched, loop, rtol=1e-10, atol=1e-10)


def test_sampled_loss_matches_manual_prediction_disagreement() -> None:
    x = torch.randn(13, 6, dtype=torch.float64)
    x_hat = x + 0.1 * torch.randn_like(x)
    targets = torch.randn(13, 7, dtype=torch.float64)
    manual = (
        ridge_predict(x, targets, 0.3) - ridge_predict(x_hat, targets, 0.3)
    ).square().sum()
    estimated = sampled_decoder_loss(
        x, x_hat, targets, ridge=0.3, relative=False
    )
    torch.testing.assert_close(estimated, manual, rtol=1e-10, atol=1e-10)


def test_ridge_calibration_hits_target_degrees_of_freedom() -> None:
    x = torch.randn(32, 24, dtype=torch.float64)
    ridge = calibrate_ridge(x, target_fraction=0.35)
    fraction = effective_degrees_of_freedom(x, ridge).item() / x.shape[0]
    assert fraction == pytest.approx(0.35, rel=1e-8, abs=1e-8)


def test_distance_backpropagates_to_reconstruction() -> None:
    x = torch.randn(10, 4)
    x_hat = torch.randn(10, 4, requires_grad=True)
    decoder_distance(x, x_hat).backward()
    assert x_hat.grad is not None
    assert torch.isfinite(x_hat.grad).all()


def test_invalid_covariance_shape_is_rejected() -> None:
    x = torch.randn(8, 3)
    with pytest.raises(ValueError, match="task_covariance"):
        decoder_distance(x, x, task_covariance=torch.eye(3))
