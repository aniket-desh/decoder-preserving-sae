import torch

from dpsae.mech_analysis import (
    binary_metrics,
    fit_ridge_binary,
    make_replacement,
    matched_random_features,
    score_ridge_binary,
    standardized_mean_difference,
)
from dpsae.language_model import ActivationStats
from dpsae.language_sae import BatchTopKSAE


def test_ridge_binary_separates_simple_data():
    positive = torch.randn(64, 4) + torch.tensor([2.0, 0, 0, 0])
    negative = torch.randn(64, 4) - torch.tensor([2.0, 0, 0, 0])
    x = torch.cat([positive, negative])
    labels = torch.cat([torch.ones(64), -torch.ones(64)])
    probe = fit_ridge_binary(x, labels)
    metrics = binary_metrics(score_ridge_binary(probe, x), labels)
    assert metrics["accuracy"] > 0.9
    assert metrics["auc"] > 0.95


def test_standardized_mean_difference_ranks_signal():
    positive = torch.randn(256, 5)
    negative = torch.randn(256, 5)
    positive[:, 3] += 2
    assert standardized_mean_difference(positive, negative).abs().argmax() == 3


def test_random_features_are_unique_and_frequency_matched():
    rates = torch.tensor([0.01, 0.011, 0.1, 0.11, 0.5, 0.51])
    selected = torch.tensor([0, 2, 4])
    matched = matched_random_features(selected, rates)
    assert torch.equal(matched, torch.tensor([1, 3, 5]))


def test_replacement_reports_incremental_activation_change():
    model = BatchTopKSAE(2, 2, 1)
    with torch.no_grad():
        model.encoder_weight.copy_(torch.eye(2))
        model.decoder_weight.copy_(torch.eye(2))
        model.encoder_bias.zero_()
        model.decoder_bias.zero_()
    stats = ActivationStats(torch.zeros(2), torch.tensor(1.0))
    diagnostics = {}
    replacement = make_replacement(
        model,
        stats,
        positions=torch.tensor([1]),
        features=torch.tensor([0]),
        diagnostics=diagnostics,
    )

    hidden = torch.tensor([[[0.0, 1.0], [2.0, 1.0]]])
    reconstructed = replacement(hidden)

    assert reconstructed[0, 1, 0] == 0
    assert diagnostics["relative_activation_change"][0] > 0
