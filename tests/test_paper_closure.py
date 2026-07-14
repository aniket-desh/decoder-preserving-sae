import pytest

from experiments import paper_closure as runner


def test_frontier_specs_pair_each_seed_at_common_initialization():
    specs = runner._frontier_specs([0.03125], [0, 2])
    assert [spec.name for spec in specs] == [
        "mse_s0",
        "dpsae_w0.03125_s0",
        "mse_s2",
        "dpsae_w0.03125_s2",
    ]
    assert specs[0].seed == specs[1].seed == 0
    assert specs[2].seed == specs[3].seed == 2
    with pytest.raises(ValueError, match="seeds"):
        runner._frontier_specs([0.03125], [0, 0])


def test_screen_models_can_select_one_seed_from_confirmation_fleet():
    payloads = {
        "mse_s0": {"spec": {"method": "mse", "seed": 0}},
        "dpsae_w0.03125_s0": {"spec": {"method": "dpsae", "seed": 0}},
        "mse_s1": {"spec": {"method": "mse", "seed": 1}},
        "dpsae_w0.03125_s1": {"spec": {"method": "dpsae", "seed": 1}},
    }
    assert runner._screen_models(payloads, evaluation_seed=1) == [
        "mse_s1",
        "dpsae_w0.03125_s1",
    ]
    with pytest.raises(ValueError, match="exactly one MSE"):
        runner._screen_models(payloads)


def test_frontier_selection_uses_smallest_passing_weight_before_nmse():
    rows = [
        {
            "candidate": "small",
            "decoder_weight": 0.03125,
            "nmse_ratio_to_mse": 0.99,
            "exact_decoder_reduction": 0.18,
        },
        {
            "candidate": "lower_nmse",
            "decoder_weight": 0.0625,
            "nmse_ratio_to_mse": 0.94,
            "exact_decoder_reduction": 0.25,
        },
    ]
    selected = runner.select_frontier_candidate(
        rows,
        {
            "maximum_nmse_ratio": 1.01,
            "minimum_exact_decoder_reduction": 0.10,
            "selection_order": ["smaller_decoder_weight", "lower_nmse"],
        },
    )
    assert selected["candidate"] == "small"


def test_frontier_runner_dispatches_jump_relu_with_auditable_defaults(monkeypatch):
    monkeypatch.setattr(
        "sys.argv",
        [
            "paper_closure.py",
            "frontier-train-screen",
            "--sparsity-mode",
            "jump_relu",
        ],
    )

    args = runner.parse_args()

    assert args.sparsity_mode == "jump_relu"
    assert args.jump_relu_init_threshold == 0.001
    assert args.jump_relu_init_mode == "topk_quantile"
    assert args.jump_relu_bandwidth == 0.001
    assert args.jump_relu_sparsity_weight == 1.0
    assert args.jump_relu_threshold_lr_multiplier == 32.0
