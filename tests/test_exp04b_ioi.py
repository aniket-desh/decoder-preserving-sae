import pytest
import torch

from dpsae.exp04b_ioi import (
    continuous_target_protocol,
    correct_minus_subject_target,
    dense_abc_patch_ceiling_report,
    exposure_matched_comparison,
    exposure_normalized_summary,
    frozen_test_report,
    frozen_test_report_by_model,
    logit_difference_report,
    matched_zero_ablation_report,
    paired_bootstrap_summary,
    rank_discovery_features,
    sample_natural_positions,
    select_feature_count,
    select_global_feature_count,
)
from dpsae.exp04b_natural_text import selected_feature_exposure


def test_discovery_ranking_uses_target_association():
    target = torch.linspace(-1, 1, 32)
    codes = torch.stack(
        [torch.ones(32), -0.5 * target + 0.2 * torch.sin(9 * target), target.square(), 3 * target],
        1,
    )
    ranking = rank_discovery_features(codes, target)
    assert ranking[0] == 3
    assert ranking[1] == 1


def test_feature_count_is_selected_on_validation_then_frozen_on_test():
    validation = [
        {"split": "validation", "features": 4, "ioi_effect": 1.0, "collateral_kl": 0.02},
        {"split": "validation", "features": 8, "ioi_effect": 2.0, "collateral_kl": 0.05},
        {"split": "validation", "features": 16, "ioi_effect": 9.0, "collateral_kl": 0.07},
    ]
    selection = select_feature_count(validation, kl_budget=0.06)
    assert selection.feature_count == 8

    report = frozen_test_report(
        selection,
        [
            {"split": "test", "features": 4, "ioi_effect": 100.0},
            {"split": "test", "features": 8, "ioi_effect": 1.5},
        ],
    )
    assert report["test"]["features"] == 8
    with pytest.raises(ValueError, match="validation rows only"):
        select_feature_count(
            [{"split": "test", "features": 1, "ioi_effect": 1.0, "collateral_kl": 0.0}],
            kl_budget=0.1,
        )


def test_global_selection_uses_six_core_models_worst_kl_and_ignores_whitening():
    rows = []
    for count, effect, maximum_kl in ((4, 1.0, 0.04), (8, 2.0, 0.061), (16, 1.0, 0.05)):
        for seed in range(3):
            for method in ("mse", "dpsae"):
                rows.append(
                    {
                        "split": "validation",
                        "model": f"{method}_s{seed}",
                        "method": method,
                        "features": count,
                        "ioi_effect": effect,
                        "collateral_kl": maximum_kl if seed == 2 else 0.01,
                    }
                )
        rows.append(
            {
                "split": "validation",
                "model": "whitening_s0",
                "method": "whitening",
                "features": count,
                "ioi_effect": 100.0,
                "collateral_kl": 0.0,
            }
        )
    selection = select_global_feature_count(rows, kl_budget=0.06)
    assert selection.feature_count == 4
    assert selection.models == 6
    assert selection.validation_kl == pytest.approx(0.04)

    test = frozen_test_report_by_model(
        selection,
        [
            {"split": "test", "model": method, "features": count}
            for method in ("mse_s0", "dpsae_s0", "whitening_s0")
            for count in (4, 8, 16)
        ],
    )
    assert {row["model"] for row in test["test_by_model"]} == {
        "mse_s0",
        "dpsae_s0",
        "whitening_s0",
    }
    assert {row["features"] for row in test["test_by_model"]} == {4}


def test_natural_position_sampling_is_one_per_sequence_and_excludes_final_token():
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 1, 1]])
    first = sample_natural_positions(mask, seed=7)
    second = sample_natural_positions(mask, seed=7)
    assert torch.equal(first, second)
    assert first.shape == (2,)
    assert first[0] < 2
    assert first[1] < 3

    lagged = sample_natural_positions(
        mask,
        seed=7,
        lag_distribution=torch.tensor([1]),
    )
    assert torch.equal(lagged, torch.tensor([1, 2]))


def test_matched_zero_ablation_reports_s2_effect_and_positionwise_kl():
    full_logits = torch.zeros(2, 3, 2)
    ablated_logits = full_logits.clone()
    ablated_logits[0, 1] = torch.tensor([2.0, -2.0])
    ablated_logits[1, 0] = torch.tensor([-1.0, 1.0])
    report = matched_zero_ablation_report(
        original_ioi_logit_difference=torch.tensor([3.0, 2.0]),
        full_ioi_logit_difference=torch.tensor([2.5, 1.5]),
        ablated_ioi_logit_difference=torch.tensor([1.0, 0.5]),
        full_natural_logits=full_logits,
        ablated_natural_logits=ablated_logits,
        natural_intervention_positions=torch.tensor([0, 1]),
        natural_readout_positions=torch.tensor([1, 0]),
        natural_relative_activation_change=0.2,
    )
    assert report["operator"]["ioi_position"] == "S2"
    assert report["ioi"]["intervention_effect"] == pytest.approx(1.25)
    assert report["natural_text"]["collateral_kl"] > 0
    assert report["natural_text"]["intervention_positions"] == [0, 1]
    assert report["natural_text"]["readout_positions"] == [1, 0]
    assert report["natural_text"]["relative_activation_change"] == 0.2


def test_exposure_and_normalization_use_selected_feature_totals():
    codes = torch.tensor([[1.0, 0.0, 2.0], [0.0, 0.0, 2.0]])
    decoder = torch.tensor([[1.0, 0.0], [0.0, 1.0], [0.0, 2.0]])
    exposure = selected_feature_exposure(codes, decoder, torch.tensor([0, 2]))
    assert exposure["summed_active_frequency"] == pytest.approx(1.5)
    assert exposure["summed_activation_mass"] == pytest.approx(2.5)
    assert exposure["summed_decoded_energy"] == pytest.approx(16.5)

    summary = exposure_normalized_summary(
        ioi_effect=2.0,
        collateral_kl=0.3,
        exposure=exposure,
        natural_relative_activation_change=0.1,
    )
    assert summary["ioi_effect_per_collateral_kl"] == pytest.approx(2 / 0.3)
    assert summary["kl_per_total_active_frequency"] == pytest.approx(0.2)
    assert summary["kl_per_natural_relative_activation_change"] == pytest.approx(3.0)


def test_paired_bootstrap_resamples_seed_pairs():
    summary = paired_bootstrap_summary(
        torch.tensor([1.0, 2.0, 3.0]),
        torch.tensor([2.0, 3.0, 4.0]),
        seed=9,
        bootstrap_samples=200,
    )
    assert summary["pairs"] == 3
    assert summary["paired_difference"] == 1.0
    assert summary["ci_low"] == 1.0
    assert summary["ci_high"] == 1.0


def test_exposure_matching_interpolates_only_on_common_support():
    reference = {"summed_active_frequency": 3.0, "collateral_kl": 0.2}
    curve = [
        {"summed_active_frequency": 2.0, "collateral_kl": 0.3},
        {"summed_active_frequency": 4.0, "collateral_kl": 0.5},
    ]
    result = exposure_matched_comparison(
        reference, curve, exposure_key="summed_active_frequency"
    )
    assert result is not None
    assert result["comparator_interpolated_outcome"] == pytest.approx(0.4)
    assert result["reference_minus_comparator"] == pytest.approx(-0.2)
    assert exposure_matched_comparison(
        {"summed_active_frequency": 5.0, "collateral_kl": 0.1},
        curve,
        exposure_key="summed_active_frequency",
    ) is None


def test_logit_difference_and_dense_patch_ceiling_have_explicit_references():
    original = torch.tensor([4.0, 2.0])
    full = torch.tensor([3.0, 1.0])
    selected = torch.tensor([2.0, 0.0])
    dense = torch.tensor([0.0, -2.0])
    logit_report = logit_difference_report(original, full, selected)
    assert logit_report["original_model_logit_difference"] == 3.0
    assert logit_report["full_sae_logit_difference"] == 2.0
    assert logit_report["intervention_effect"] == 1.0

    ceiling = dense_abc_patch_ceiling_report(
        original_ioi_logit_difference=original,
        full_ioi_logit_difference=full,
        selected_patch_logit_difference=selected,
        dense_patch_logit_difference=dense,
    )
    assert ceiling["dense_activation_abc_patch"]["reference"] == "original_model"
    assert ceiling["dense_activation_abc_patch"]["patch_effect"] == 4.0
    assert ceiling["selected_effect_fraction_of_dense_ceiling"] == 0.25


def test_correct_minus_subject_target_uses_final_unpadded_position():
    logits = torch.zeros(2, 3, 5)
    logits[0, 1, 3], logits[0, 1, 1] = 5, 2
    logits[1, 2, 4], logits[1, 2, 0] = 1, 3
    target = correct_minus_subject_target(
        logits,
        torch.tensor([[1, 1, 0], [1, 1, 1]]),
        torch.tensor([3, 4]),
        torch.tensor([1, 0]),
    )
    assert torch.equal(target, torch.tensor([3.0, -2.0]))


def test_continuous_protocol_selects_on_validation_and_scores_one_test_count():
    generator = torch.Generator().manual_seed(4)
    codes = torch.randn(120, 4, generator=generator)
    target = 2 * codes[:, 2] - codes[:, 0]
    selection = select_feature_count(
        [{"features": 2, "ioi_effect": 1.0, "collateral_kl": 0.04}],
        kl_budget=0.05,
    )
    result = continuous_target_protocol(
        discovery_codes=codes[:40],
        discovery_target=target[:40],
        validation_codes=codes[40:80],
        validation_target=target[40:80],
        test_codes=codes[80:],
        test_target=target[80:],
        feature_counts=[1, 2, 4],
        selection=selection,
    )
    assert result["target"] == "correct_minus_subject_logit_difference"
    assert result["selection"]["selected_on"] == "validation"
    assert result["selection"]["feature_count"] == 2
    assert result["test"]["features"] == 2
    assert result["test"]["r2"] > 0.99
