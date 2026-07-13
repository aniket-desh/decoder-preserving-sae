import pytest
import torch

from dpsae.corpus import TokenRange
from dpsae.decoder_distance import ridge_hat_matrix
from dpsae.exp04b_natural_text import (
    apply_geometry_groups,
    bootstrap_paired_reduction_interval,
    bootstrap_ratio_interval,
    document_ids_from_tokens,
    exact_decoder_sweep,
    exact_identity_decoder_statistics,
    geometry_group_indices,
    selected_feature_exposure,
    split_selection_test_range,
)


def test_selection_test_ranges_are_disjoint_and_exhaust_source():
    split = split_selection_test_range(TokenRange(170, 180))

    assert split == {
        "selection": TokenRange(170, 175),
        "test": TokenRange(175, 180),
    }
    assert split["selection"].stop == split["test"].start


def test_selection_test_range_enforces_minimum_size():
    with pytest.raises(ValueError, match="minimum_size"):
        split_selection_test_range(TokenRange(0, 10), selection_tokens=8, minimum_size=3)


def test_document_ids_split_after_eos_and_between_sequences():
    tokens = torch.tensor([[1, 2, 9, 3], [4, 9, 5, 6]])

    assert torch.equal(
        document_ids_from_tokens(tokens, eos_token_id=9),
        torch.tensor([[0, 0, 0, 1], [2, 2, 3, 3]]),
    )


def test_geometry_group_constructions_are_paired_and_reproducible():
    tokens = torch.arange(16).reshape(4, 4)
    contiguous = geometry_group_indices(tokens, 4, "contiguous")
    shuffled = geometry_group_indices(tokens, 4, "shuffled", seed=7)
    balanced = geometry_group_indices(tokens, 4, "document_balanced", seed=7)

    assert torch.equal(contiguous, torch.arange(16).reshape(4, 4))
    assert torch.equal(shuffled, geometry_group_indices(tokens, 4, "shuffled", seed=7))
    assert not torch.equal(shuffled, contiguous)
    document_ids = document_ids_from_tokens(tokens, eos_token_id=None).flatten()
    for group in balanced:
        assert document_ids[group].unique().numel() == 4

    values = torch.arange(16).reshape(4, 4, 1)
    assert torch.equal(apply_geometry_groups(values, shuffled).squeeze(-1), shuffled)


def test_exact_identity_statistics_match_explicit_hat_matrices():
    generator = torch.Generator().manual_seed(3)
    original = torch.randn(2, 6, 3, generator=generator, dtype=torch.float64)
    reconstructed = original + 0.1 * torch.randn(
        original.shape, generator=generator, dtype=torch.float64
    )
    numerator, denominator = exact_identity_decoder_statistics(
        original, reconstructed, ridge=0.2
    )
    expected_numerator, expected_denominator = [], []
    for x, x_hat in zip(original, reconstructed):
        reference = ridge_hat_matrix(x.float(), 0.2)
        prediction = ridge_hat_matrix(x_hat.float(), 0.2)
        expected_numerator.append((prediction - reference).square().sum())
        expected_denominator.append(reference.square().sum())

    torch.testing.assert_close(numerator, torch.stack(expected_numerator))
    torch.testing.assert_close(denominator, torch.stack(expected_denominator))


def test_bootstrap_intervals_respect_constant_group_ratios():
    denominator = torch.tensor([1.0, 2.0, 4.0, 8.0])
    ratio = bootstrap_ratio_interval(
        0.3 * denominator, denominator, samples=128, seed=4
    )
    reduction = bootstrap_paired_reduction_interval(
        denominator, 0.7 * denominator, samples=128, seed=4
    )

    assert ratio == pytest.approx({"estimate": 0.3, "low": 0.3, "high": 0.3})
    assert reduction == pytest.approx({"estimate": 0.3, "low": 0.3, "high": 0.3})


def test_exact_decoder_sweep_reuses_paired_groups_and_limits_exact_work():
    generator = torch.Generator().manual_seed(9)
    tokens = torch.arange(32).reshape(4, 8)
    original = torch.randn(4, 8, 3, generator=generator)
    reconstructions = {
        "same": original.clone(),
        "noisy": original + 0.2 * torch.randn(original.shape, generator=generator),
    }
    rows = exact_decoder_sweep(
        original,
        reconstructions,
        tokens,
        ridges=[0.1, 0.4],
        group_sizes=[4],
        groupings=["contiguous", "shuffled"],
        max_groups=3,
        bootstrap_samples=64,
        seed=5,
    )

    assert len(rows) == 8
    assert all(row["groups"] == 3 for row in rows)
    assert all(row["decoder_distortion"] == 0 for row in rows if row["model"] == "same")
    assert all(row["decoder_distortion"] > 0 for row in rows if row["model"] == "noisy")
    for key in {(row["grouping"], row["ridge"]) for row in rows}:
        paired = [row for row in rows if (row["grouping"], row["ridge"]) == key]
        assert paired[0]["denominator_by_group"] == paired[1]["denominator_by_group"]


def test_selected_feature_exposure_reports_individual_and_joint_energy():
    codes = torch.tensor([[1.0, 0.0, 2.0], [0.0, 3.0, 0.0]])
    decoder = torch.eye(3)
    reference = codes.clone()
    metrics = selected_feature_exposure(
        codes,
        decoder,
        [0, 1],
        reference_activations=reference,
        collateral_kl=0.2,
    )

    assert metrics["active_frequency_by_feature"] == pytest.approx([0.5, 0.5])
    assert metrics["activation_mass_by_feature"] == pytest.approx([0.5, 1.5])
    assert metrics["decoded_energy_by_feature"] == pytest.approx([0.5, 4.5])
    assert metrics["summed_decoded_energy"] == pytest.approx(5.0)
    assert metrics["ablation_change_energy"] == pytest.approx(5.0)
    assert metrics["ablation_relative_activation_change"] == pytest.approx((10 / 14) ** 0.5)
    assert metrics["collateral_kl_per_activation_change"] == pytest.approx(
        0.2 / (10 / 14) ** 0.5
    )
