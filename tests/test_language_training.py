import torch

from dpsae.language_training import (
    SAETrainSpec,
    TrainingFleet,
    sampled_decoder_loss_from_reference,
    spectral_surrogate_operator,
    whitening_operator,
)


def test_whitening_operator_makes_covariance_near_identity():
    x = torch.randn(2048, 8) @ torch.diag(torch.arange(1, 9, dtype=torch.float32))
    operator = whitening_operator(x, floor_fraction=1e-6)
    whitened = x @ operator
    covariance = whitened.mT @ whitened / len(x)
    assert torch.allclose(covariance, torch.eye(8), atol=2e-3)


def test_reference_decoder_loss_has_reconstruction_gradient():
    original = torch.randn(2, 8, 5)
    reconstructed = original.clone().requires_grad_()
    targets = torch.randn(2, 8, 3)
    loss = sampled_decoder_loss_from_reference(
        original, reconstructed, targets, ridge=0.2
    )
    loss.backward()
    assert reconstructed.grad is not None
    assert torch.isfinite(reconstructed.grad).all()


def test_spectral_surrogate_operator_has_theoretical_eigenweights():
    scales = torch.tensor([0.5, 1.0, 2.0])
    x = torch.eye(3).repeat(128, 1) * scales
    ridge = 0.2
    operator = spectral_surrogate_operator(x, ridge=ridge)
    covariance = x.mT @ x / len(x)
    expected = covariance.diag().sqrt() / (covariance.diag() + ridge)
    torch.testing.assert_close(operator.diag(), expected)
    torch.testing.assert_close(operator - torch.diag(operator.diag()), torch.zeros(3, 3))


def test_training_fleet_updates_matched_models():
    specs = [
        SAETrainSpec("mse_s0", "mse", 0, 2),
        SAETrainSpec("dpsae_s0", "dpsae", 0, 2, decoder_weight=0.5),
        SAETrainSpec("whitening_s0", "whitening", 0, 2),
        SAETrainSpec("spectral_s0", "spectral", 0, 2, loss_weight=0.5),
    ]
    x = torch.randn(16, 6)
    fleet = TrainingFleet(
        specs,
        input_dim=6,
        dictionary_size=12,
        learning_rate=1e-3,
        device=torch.device("cpu"),
        whitening=whitening_operator(torch.randn(128, 6)),
        spectral=spectral_surrogate_operator(torch.randn(128, 6), ridge=0.2),
        dead_after_steps=100,
    )
    metrics = fleet.train_batch(
        x, step=1, ridge=0.2, group_size=8, probes=3, probe_seed=4
    )
    assert set(metrics) == {spec.name for spec in specs}
    assert all(value["l0"] == 2 for value in metrics.values())
    assert metrics["spectral_s0"]["static"] > 0
