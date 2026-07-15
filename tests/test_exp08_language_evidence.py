import pytest
import torch

from experiments.exp08_language_evidence import (
    aggregate_frozen_rows,
    bootstrap_frozen_pair,
    disjoint_intervals,
    one_factor_settings,
    selected_payloads,
    validate_natural_cache,
)


def _payload(method: str, seed: int, weight: float = 0.0) -> dict:
    return {
        "spec": {
            "method": method,
            "seed": seed,
            "k": 32,
            "decoder_weight": weight,
        }
    }


def test_selected_payloads_keeps_one_matched_pair_per_seed() -> None:
    payloads = {
        "mse_s0": _payload("mse", 0),
        "dpsae_w0.03125_s0": _payload("dpsae", 0, 0.03125),
        "dpsae_w0.25_s0": _payload("dpsae", 0, 0.25),
        "mse_s1": _payload("mse", 1),
        "dpsae_w0.03125_s1": _payload("dpsae", 1, 0.03125),
    }

    selected = selected_payloads(payloads, 0.03125)

    assert set(selected) == {
        "mse_s0",
        "dpsae_w0.03125_s0",
        "mse_s1",
        "dpsae_w0.03125_s1",
    }


def test_one_factor_settings_keeps_a_shared_base_for_each_axis() -> None:
    config = {
        "natural_text": {
            "group_sizes": [64, 128, 256],
            "groupings": ["contiguous", "shuffled", "document_balanced"],
        }
    }
    source = {"geometry": {"group_size": 128}}
    static = {
        "ridge": 2.0,
        "ridges_by_dof_fraction": {
            "0.125": {"ridge": 1.0},
            "0.25": {"ridge": 2.0},
            "0.5": {"ridge": 3.0},
        },
        "ridges_by_group_size": {
            "64": {"ridge": 1.5},
            "128": {"ridge": 2.0},
            "256": {"ridge": 2.5},
        },
    }

    settings = one_factor_settings(config, source, static)

    assert len(settings) == 9
    assert any(
        row["audit_axis"] == "ridge"
        and row["setting_value"] == 0.25
        and row["ridge"] == 2.0
        for row in settings
    )
    assert any(
        row["audit_axis"] == "group_size"
        and row["setting_value"] == 128
        and row["ridge"] == 2.0
        for row in settings
    )
    assert any(
        row["audit_axis"] == "grouping"
        and row["setting_value"] == "contiguous"
        for row in settings
    )


def test_frozen_summary_and_paired_bootstrap_preserve_metric_directions() -> None:
    common = [
        {
            "tokens": 10.0,
            "original_nll": 20.0,
            "mean_nll": 40.0,
            "original_correct": 5.0,
            "activation_energy": 100.0,
            "activation_tokens": 10.0,
        }
        for _ in range(4)
    ]
    mse = [
        {
            "reconstructed_nll": 30.0,
            "kl": 2.0,
            "agreement": 8.0,
            "reconstructed_correct": 4.0,
            "reconstruction_sse": 10.0,
            "l0_count": 320.0,
        }
        for _ in range(4)
    ]
    dpsae = [
        {
            "reconstructed_nll": 26.0,
            "kl": 1.0,
            "agreement": 9.0,
            "reconstructed_correct": 5.0,
            "reconstruction_sse": 8.0,
            "l0_count": 310.0,
        }
        for _ in range(4)
    ]

    mse_summary = aggregate_frozen_rows(common, mse)
    dpsae_summary = aggregate_frozen_rows(common, dpsae)
    paired = bootstrap_frozen_pair(common, mse, dpsae, samples=100, seed=1)

    assert mse_summary["loss_recovered"] == pytest.approx(0.5)
    assert dpsae_summary["loss_recovered"] == pytest.approx(0.7)
    assert paired["loss_recovered_difference_dpsae_minus_mse_ci95"] == pytest.approx(
        [0.2, 0.2]
    )
    assert paired["kl_difference_dpsae_minus_mse_ci95"] == pytest.approx([-0.1, -0.1])
    assert paired[
        "cross_entropy_increase_difference_dpsae_minus_mse_ci95"
    ] == pytest.approx([-0.4, -0.4])
    assert paired["activation_nmse_ratio_dpsae_to_mse_ci95"] == pytest.approx(
        [0.8, 0.8]
    )
    assert paired[
        "inference_l0_difference_dpsae_minus_mse_ci95"
    ] == pytest.approx([-1.0, -1.0])
    assert paired["valid_loss_recovered_draw_fraction"] == 1


def test_selected_payloads_requires_the_frozen_seed_set() -> None:
    payloads = {
        "mse_s0": _payload("mse", 0),
        "dpsae_w0.03125_s0": _payload("dpsae", 0, 0.03125),
    }

    with pytest.raises(ValueError, match="differ from expected"):
        selected_payloads(payloads, 0.03125, expected_seeds=[0, 1, 2])


def test_natural_cache_validation_binds_split_range_offset_and_starts() -> None:
    cache = {
        "split": "test",
        "token_range": [5_000_000, 10_000_000],
        "token_offset": 190_000_000,
        "input_ids": torch.zeros(2, 256, dtype=torch.long),
        "activations": torch.zeros(2, 256, 4),
        "starts": torch.tensor([195_000_000, 199_999_744]),
    }
    config = {
        "fresh_corpus": {
            "token_offset": 190_000_000,
            "selection_range": [0, 5_000_000],
            "test_range": [5_000_000, 10_000_000],
        }
    }

    assert validate_natural_cache(cache, config, split="test") == (
        195_000_000,
        200_000_000,
    )
    cache["starts"][-1] += 1
    with pytest.raises(ValueError, match="outside its frozen range"):
        validate_natural_cache(cache, config, split="test")


def test_protocol_interval_validation_rejects_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        disjoint_intervals({"training": (50, 120), "evaluation": (119, 130)})
