import math

import torch

from dpsae.language_training import (
    SAETrainSpec,
    TrainingFleet,
    sampled_decoder_loss_from_reference,
    spectral_surrogate_operator,
    whitening_operator,
)
from dpsae.mech_analysis import load_sae


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
    assert all("sparsity" not in value for value in metrics.values())
    assert all(
        "sparsity_config" not in payload for payload in fleet.export_models().values()
    )
    assert metrics["spectral_s0"]["static"] > 0


def test_training_fleet_supports_tokenwise_topk_and_exports_mode():
    specs = [
        SAETrainSpec("mse_s0", "mse", 0, 2),
        SAETrainSpec("dpsae_s0", "dpsae", 0, 2, decoder_weight=0.5),
    ]
    fleet = TrainingFleet(
        specs,
        input_dim=6,
        dictionary_size=12,
        learning_rate=1e-3,
        device=torch.device("cpu"),
        dead_after_steps=100,
        sparsity_mode="token_topk",
    )
    metrics = fleet.train_batch(
        torch.randn(16, 6),
        step=1,
        ridge=0.2,
        group_size=8,
        probes=3,
        probe_seed=4,
    )
    assert all(value["l0"] == 2 for value in metrics.values())
    assert all(
        payload["sparsity_mode"] == "token_topk"
        for payload in fleet.export_models().values()
    )


def test_training_fleet_rejects_checkpoint_from_other_sparsity_mode():
    specs = [SAETrainSpec("mse_s0", "mse", 0, 2)]
    batch = TrainingFleet(
        specs,
        input_dim=6,
        dictionary_size=12,
        learning_rate=1e-3,
        device=torch.device("cpu"),
    )
    token = TrainingFleet(
        specs,
        input_dim=6,
        dictionary_size=12,
        learning_rate=1e-3,
        device=torch.device("cpu"),
        sparsity_mode="token_topk",
    )
    with torch.no_grad():
        state = batch.state_dict(step=0, tokens_seen=0)
    try:
        token.load_state_dict(state)
    except ValueError as error:
        assert "sparsity mode" in str(error)
    else:
        raise AssertionError("checkpoint mode mismatch must fail")


def _jump_fleet(
    *,
    learning_rate: float = 1e-3,
    sparsity_weight: float = 1.0,
    init_mode: str = "topk_quantile",
    threshold_lr_multiplier: float = 1.0,
):
    return TrainingFleet(
        [SAETrainSpec("mse_s0", "mse", 0, 2)],
        input_dim=6,
        dictionary_size=12,
        learning_rate=learning_rate,
        device=torch.device("cpu"),
        dead_after_steps=100,
        sparsity_mode="jump_relu",
        jump_relu_init_threshold=0.1,
        jump_relu_init_mode=init_mode,
        jump_relu_bandwidth=1.0,
        jump_relu_sparsity_weight=sparsity_weight,
        jump_relu_threshold_lr_multiplier=threshold_lr_multiplier,
    )


def test_training_fleet_dispatches_jump_relu_and_exports_controller_config():
    fleet = _jump_fleet()
    metrics = fleet.train_batch(
        torch.randn(16, 6),
        step=1,
        ridge=0.2,
        group_size=8,
        probes=3,
        probe_seed=4,
    )["mse_s0"]
    payload = fleet.export_models()["mse_s0"]

    assert metrics["sparsity"] >= 0
    assert metrics["l0"] == 2
    assert metrics["threshold_initialized"]
    assert metrics["initialization_cutoff"] > 0
    assert metrics["threshold_learning_rate"] == 1e-3
    assert math.isfinite(metrics["loss"])
    assert torch.isfinite(fleet.models["mse_s0"].log_threshold.grad).all()
    assert payload["sparsity_mode"] == "jump_relu"
    assert payload["sparsity_config"] == {
        "init_threshold": 0.1,
        "initialization": "topk_quantile",
        "bandwidth": 1.0,
        "target_l0_loss_weight": 1.0,
        "threshold_lr_multiplier": 1.0,
    }
    assert fleet.jump_threshold_summary()["mse_s0"]["initialization_cutoff"] > 0


def test_jump_relu_threshold_lr_multiplier_isolated_and_scheduled():
    fleet = _jump_fleet(
        learning_rate=3e-4,
        threshold_lr_multiplier=32.0,
    )
    model = fleet.models["mse_s0"]
    optimizer = fleet.optimizers["mse_s0"]

    assert len(optimizer.param_groups) == 2
    assert all(
        parameter is not model.log_threshold
        for parameter in optimizer.param_groups[0]["params"]
    )
    assert optimizer.param_groups[1]["params"] == [model.log_threshold]
    assert optimizer.param_groups[0]["lr"] == 3e-4
    assert optimizer.param_groups[1]["lr"] == 32 * 3e-4

    fleet.set_learning_rate(2e-4)

    assert optimizer.param_groups[0]["lr"] == 2e-4
    assert optimizer.param_groups[1]["lr"] == 32 * 2e-4
    state = fleet.state_dict(step=0, tokens_seen=0)
    assert state["sparsity_config"]["threshold_lr_multiplier"] == 32.0
    assert state["optimizers"]["mse_s0"]["param_groups"][1][
        "lr_multiplier"
    ] == 32.0


def test_jump_relu_threshold_multiplier_changes_only_threshold_update():
    ordinary = _jump_fleet(learning_rate=1e-4, threshold_lr_multiplier=1.0)
    accelerated = _jump_fleet(learning_rate=1e-4, threshold_lr_multiplier=32.0)
    x = torch.randn(16, 6, generator=torch.Generator().manual_seed(18))

    for fleet in (ordinary, accelerated):
        fleet.train_batch(
            x,
            step=1,
            ridge=0.2,
            group_size=8,
            probes=3,
            probe_seed=4,
        )

    ordinary_model = ordinary.models["mse_s0"]
    accelerated_model = accelerated.models["mse_s0"]
    for name, parameter in ordinary_model.named_parameters():
        if name != "log_threshold":
            torch.testing.assert_close(
                parameter,
                dict(accelerated_model.named_parameters())[name],
            )
    cutoff_log = ordinary_model.activation_threshold.log()
    ordinary_move = (ordinary_model.log_threshold - cutoff_log).abs().mean()
    accelerated_move = (accelerated_model.log_threshold - cutoff_log).abs().mean()
    assert accelerated_move > 20 * ordinary_move


def test_jump_relu_checkpoint_resume_export_and_load_round_trip():
    fleet = _jump_fleet(threshold_lr_multiplier=32.0)
    x = torch.randn(16, 6)
    fleet.train_batch(
        x, step=1, ridge=0.2, group_size=8, probes=3, probe_seed=4
    )
    state = fleet.state_dict(step=1, tokens_seen=16)
    restored = _jump_fleet(threshold_lr_multiplier=32.0)

    assert restored.load_state_dict(state) == (1, 16)
    torch.testing.assert_close(
        restored.models["mse_s0"].jump_threshold,
        fleet.models["mse_s0"].jump_threshold,
    )
    payload = restored.export_models()["mse_s0"]
    loaded = load_sae(payload, input_dim=6, device=torch.device("cpu"))
    with torch.inference_mode():
        expected = restored.models["mse_s0"].eval().encode(x)
        actual = loaded.encode(x)
    torch.testing.assert_close(actual, expected)
    assert restored.optimizers["mse_s0"].param_groups[1]["lr_multiplier"] == 32.0
    assert payload["sparsity_config"]["threshold_lr_multiplier"] == 32.0


def test_jump_relu_short_finite_integration_stays_near_target():
    fleet = _jump_fleet(learning_rate=3e-4, sparsity_weight=10.0)
    model = fleet.models["mse_s0"]
    x = torch.randn(16, 6, generator=torch.Generator().manual_seed(9))
    metrics = None
    initial_l0 = None
    for step in range(1, 9):
        metrics = fleet.train_batch(
            x,
            step=step,
            ridge=0.2,
            group_size=8,
            probes=3,
            probe_seed=step,
        )["mse_s0"]
        initial_l0 = metrics["l0"] if initial_l0 is None else initial_l0
        assert all(
            math.isfinite(metrics[key])
            for key in ("loss", "nmse", "sparsity", "aux", "l0")
        )

    assert metrics is not None and initial_l0 == 2
    assert abs(metrics["l0"] - 2) <= 0.25


def test_jump_relu_quantile_initialization_is_not_reapplied_after_resume():
    fleet = _jump_fleet(learning_rate=0.0)
    first = torch.randn(16, 6)
    fleet.train_batch(
        first, step=1, ridge=0.2, group_size=8, probes=3, probe_seed=1
    )
    state = fleet.state_dict(step=1, tokens_seen=16)
    restored = _jump_fleet(learning_rate=0.0)
    restored.load_state_dict(state)
    threshold = restored.models["mse_s0"].jump_threshold.detach().clone()

    metrics = restored.train_batch(
        100 * first,
        step=2,
        ridge=0.2,
        group_size=8,
        probes=3,
        probe_seed=2,
    )["mse_s0"]

    torch.testing.assert_close(restored.models["mse_s0"].jump_threshold, threshold)
    assert not metrics["threshold_initialized"]
    assert restored.models["mse_s0"].threshold_updates.item() == 1


def test_jump_relu_library_default_keeps_published_fixed_initialization():
    fleet = TrainingFleet(
        [SAETrainSpec("mse_s0", "mse", 0, 2)],
        input_dim=6,
        dictionary_size=12,
        learning_rate=0.0,
        device=torch.device("cpu"),
        sparsity_mode="jump_relu",
        jump_relu_init_threshold=0.001,
        jump_relu_bandwidth=0.001,
    )
    metrics = fleet.train_batch(
        torch.randn(16, 6),
        step=1,
        ridge=0.2,
        group_size=8,
        probes=3,
        probe_seed=1,
    )["mse_s0"]

    assert fleet.sparsity_config()["initialization"] == "fixed"
    assert fleet.sparsity_config()["threshold_lr_multiplier"] == 1.0
    assert fleet.models["mse_s0"].threshold_updates.item() == 0
    assert not metrics["threshold_initialized"]
