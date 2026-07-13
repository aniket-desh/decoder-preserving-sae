import pytest

from dpsae.exp04b_training import (
    confirmation_replicate_config,
    confirmation_specs,
    probe_seed_for_step,
    screen_specs,
    select_static_baselines,
    stage_seeds,
)


def test_screen_specs_use_external_beta_grid_and_fixed_dpsae_anchor():
    specs = screen_specs(k=32, seed=4, dpsae_weight=0.25, beta_grid=[0.2, 0.8])

    assert [(spec.method, spec.loss_weight) for spec in specs] == [
        ("mse", 1.0),
        ("dpsae", 1.0),
        ("whitening", 0.2),
        ("whitening", 0.8),
        ("spectral", 0.2),
        ("spectral", 0.8),
    ]
    assert specs[1].decoder_weight == 0.25
    assert all(spec.seed == 4 and spec.k == 32 for spec in specs)


def test_selection_uses_fresh_split_nmse_gate_and_minimum_decoder():
    specs = screen_specs(k=32, seed=0, dpsae_weight=0.25, beta_grid=[0.1, 0.5])
    metrics = {
        "mse_s0": {"nmse": 0.10, "decoder": 0.20},
        "dpsae_s0": {"nmse": 0.108, "decoder": 0.14},
        "whitening_b0.1_s0": {"nmse": 0.105, "decoder": 0.17},
        "whitening_b0.5_s0": {"nmse": 0.111, "decoder": 0.12},
        "spectral_b0.1_s0": {"nmse": 0.109, "decoder": 0.16},
        "spectral_b0.5_s0": {"nmse": 0.108, "decoder": 0.13},
    }

    selection = select_static_baselines(metrics, specs, split="selection")

    whitening = selection["baselines"]["whitening"]
    spectral = selection["baselines"]["spectral"]
    assert selection["nmse_cap"] == pytest.approx(0.11)
    assert whitening["selected_spec"]["loss_weight"] == 0.1
    assert whitening["qualifying_count"] == 1
    assert [row["qualifies"] for row in whitening["candidates"]] == [True, False]
    assert spectral["selected_spec"]["loss_weight"] == 0.5
    assert spectral["selected_metrics"]["decoder"] == 0.13
    with pytest.raises(ValueError, match="fresh selection"):
        select_static_baselines(metrics, specs, split="test")


def test_selection_marks_no_qualifying_candidate_and_confirmation_omits_it():
    specs = screen_specs(k=32, seed=0, dpsae_weight=0.25, beta_grid=[0.1])
    selection = select_static_baselines(
        {
            "mse_s0": {"nmse": 0.10, "decoder": 0.20},
            "dpsae_s0": {"nmse": 0.108, "decoder": 0.14},
            "whitening_b0.1_s0": {"nmse": 0.12, "decoder": 0.10},
            "spectral_b0.1_s0": {"nmse": 0.109, "decoder": 0.15},
        },
        specs,
        split="selection",
    )

    whitening = selection["baselines"]["whitening"]
    assert whitening["status"] == "no_qualifying_candidate"
    assert whitening["selected_spec"] is None

    confirmation = confirmation_specs(
        k=32,
        seeds=[0, 1],
        dpsae_weight=0.25,
        selection=selection,
    )
    assert [spec.method for spec in confirmation] == [
        "mse",
        "dpsae",
        "spectral",
        "mse",
        "dpsae",
        "spectral",
    ]
    assert [spec.loss_weight for spec in confirmation if spec.method == "spectral"] == [
        0.1,
        0.1,
    ]


def test_confirmation_replicate_changes_both_seed_streams_without_mutating_config():
    config = {"seed": 20260712, "training": {"confirmation_tokens": 100}}
    first = confirmation_replicate_config(config, replicate=1)
    repeated = confirmation_replicate_config(config, replicate=1)
    second = confirmation_replicate_config(config, replicate=2)

    assert first == repeated
    assert "randomness" not in config
    assert first["randomness"]["data_order"] != second["randomness"]["data_order"]
    assert first["randomness"]["probe_sequence"] != second["randomness"]["probe_sequence"]

    seeds = stage_seeds(config["seed"], "confirmation", replicate=1)
    assert first["randomness"] == {
        "stage": "confirmation",
        "replicate": 1,
        "data_order": seeds.data_order,
        "probe_sequence": seeds.probe_sequence,
    }
    assert probe_seed_for_step(seeds, 9) == probe_seed_for_step(seeds, 9)
    assert probe_seed_for_step(seeds, 9) != probe_seed_for_step(seeds, 10)
