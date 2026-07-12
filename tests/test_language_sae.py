import torch

from dpsae.language_sae import BatchTopKSAE


def test_batch_topk_has_exact_batch_average_sparsity() -> None:
    model = BatchTopKSAE(8, 32, 3, seed=1)
    model.train()
    _, code = model(torch.randn(16, 8))
    assert int((code != 0).sum()) == 16 * 3
    assert model.threshold_updates.item() == 1


def test_eval_threshold_is_batch_independent() -> None:
    model = BatchTopKSAE(8, 32, 3, seed=2)
    calibration = torch.randn(64, 8)
    model.calibrate_threshold_(calibration)
    model.eval()
    first = model.encode(calibration[:16])
    second = model.encode(calibration[:32])[:16]
    torch.testing.assert_close(first, second)


def test_language_sae_backpropagates_and_normalizes_decoder() -> None:
    model = BatchTopKSAE(8, 32, 3, seed=3)
    x = torch.randn(16, 8)
    reconstruction, code = model(x)
    loss = (x - reconstruction).square().mean()
    loss.backward()
    model.project_decoder_grad_()
    model.normalize_decoder_()
    assert model.encoder_weight.grad is not None
    assert torch.isfinite(model.encoder_weight.grad).all()
    torch.testing.assert_close(model.decoder_weight.norm(dim=1), torch.ones(32))


def test_auxiliary_loss_uses_dead_features() -> None:
    model = BatchTopKSAE(8, 32, 3, seed=4)
    x = torch.randn(16, 8)
    reconstruction, code = model(x)
    model.update_activity_(code, step=1)
    aux = model.auxiliary_loss(
        x,
        reconstruction,
        step=20,
        dead_after_steps=5,
        aux_k=2,
    )
    assert aux.ndim == 0
    assert torch.isfinite(aux)
