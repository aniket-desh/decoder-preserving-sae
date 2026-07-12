import pytest
import torch

from dpsae.decoder_distance import decoder_distance, ridge_hat_matrix


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

