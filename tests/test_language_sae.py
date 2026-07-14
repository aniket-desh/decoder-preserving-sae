import torch

from dpsae.language_sae import (
    BatchTopKSAE,
    jump_relu,
    jump_relu_target_l0_loss,
)


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


def test_token_topk_has_exact_per_token_support_and_is_batch_independent() -> None:
    model = BatchTopKSAE(8, 32, 3, seed=2, sparsity_mode="token_topk")
    model.eval()
    batch = torch.randn(16, 8)
    first = model.encode(batch)
    second = model.encode(torch.cat([batch, 10 * torch.randn(16, 8)]))[:16]
    assert torch.equal((first != 0).sum(1), torch.full((16,), 3))
    torch.testing.assert_close(first, second)


def test_jump_relu_uses_a_strict_learned_threshold_and_is_batch_independent() -> None:
    scores = torch.tensor([[0.49, 0.50, 0.51]])
    threshold = torch.full((3,), 0.50)
    code = jump_relu(scores, threshold, bandwidth=0.1)
    torch.testing.assert_close(code, torch.tensor([[0.0, 0.0, 0.51]]))

    model = BatchTopKSAE(
        8,
        32,
        3,
        seed=2,
        sparsity_mode="jump_relu",
        jump_relu_init_threshold=0.1,
        jump_relu_bandwidth=0.1,
    ).eval()
    batch = torch.randn(16, 8)
    first = model.encode(batch)
    second = model.encode(torch.cat([batch, 10 * torch.randn(16, 8)]))[:16]
    torch.testing.assert_close(first, second)
    assert (model.jump_threshold > 0).all()


def test_jump_relu_target_l0_estimator_moves_threshold_in_correct_direction() -> None:
    scores = torch.tensor([[0.6, 0.6]], requires_grad=True)
    log_threshold = torch.full((2,), 0.5).log().requires_grad_()
    threshold = log_threshold.exp()
    loss = jump_relu_target_l0_loss(
        scores,
        threshold,
        bandwidth=1.0,
        target_l0=1,
    )

    loss.backward()

    assert loss.item() == 1.0
    torch.testing.assert_close(scores.grad, torch.zeros_like(scores))
    assert torch.isfinite(log_threshold.grad).all()
    assert (log_threshold.grad < 0).all()


def test_jump_relu_topk_quantile_initialization_sets_exact_global_l0() -> None:
    model = BatchTopKSAE(
        2,
        4,
        2,
        sparsity_mode="jump_relu",
        jump_relu_init_threshold=0.001,
    )
    scores = torch.tensor(
        [
            [0.1, 0.2, 0.3, 0.4],
            [0.5, 0.6, 0.7, 0.8],
        ]
    )

    cutoff = model.initialize_jump_threshold_(scores)
    code = jump_relu(scores, model.jump_threshold, bandwidth=0.1)

    assert cutoff < 0.5
    assert torch.unique(model.jump_threshold).numel() == 1
    assert int((code != 0).sum()) == len(scores) * model.k
    assert model.threshold_updates.item() == 1
    assert model.activation_threshold.item() == cutoff


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
