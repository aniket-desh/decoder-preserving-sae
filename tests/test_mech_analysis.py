import torch

from dpsae.mech_analysis import (
    binary_metrics,
    fit_ridge_binary,
    matched_random_features,
    score_ridge_binary,
    standardized_mean_difference,
)


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
