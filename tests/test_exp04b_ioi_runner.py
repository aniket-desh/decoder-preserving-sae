import importlib.util
from pathlib import Path

import torch


SCRIPT = (
    Path(__file__).resolve().parents[1]
    / "experiments"
    / "exp04b_ioi_confirmatory.py"
)
SPEC = importlib.util.spec_from_file_location("exp04b_ioi_confirmatory", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_method_names_and_config_paths_are_stable():
    config, source = MODULE.load_configs(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "exp04b_confirmatory.json"
    )

    assert MODULE.method_name("mse_s0") == "mse"
    assert MODULE.method_name("dpsae_s2") == "dpsae"
    assert MODULE.method_name("whitening_s1") == "whitening"
    assert MODULE.method_name("spectral_s1") == "spectral"
    assert MODULE.output_path(config).name == "exp04b_confirmatory"
    assert source["model_name"] == "openai-community/gpt2"


def test_model_payloads_use_the_fully_paired_confirmation_fleet(tmp_path, monkeypatch):
    source_models = {
        "mse_s0": {"spec": {"method": "mse"}},
        "dpsae_s0": {"spec": {"method": "dpsae"}},
        "whitening_s0": {"spec": {"method": "whitening"}},
    }
    baseline_models = {
        "mse_s0": source_models["mse_s0"],
        "dpsae_s0": source_models["dpsae_s0"],
        "whitening_s0": {"spec": {"method": "whitening"}},
        "spectral_s0": {"spec": {"method": "spectral"}},
    }
    output = tmp_path / "output"
    (output / "baseline_confirm").mkdir(parents=True)
    torch.save(baseline_models, output / "baseline_confirm" / "models.pt")
    monkeypatch.setattr(MODULE, "output_path", lambda _config: output)

    payloads = MODULE._model_payloads({})

    assert set(payloads) == {
        "mse_s0",
        "dpsae_s0",
        "whitening_s0",
        "spectral_s0",
    }


def test_paired_test_summary_uses_frozen_count_and_per_example_pairs():
    def result(method, values, kl, r2):
        return {
            "method": method,
            "spec": {"seed": 0},
            "duplicate_state": {
                "ioi_zero_curve": [
                    {"features": 2, "effect_by_example": values}
                ],
                "natural_zero_curve": [
                    {"features": 2, "kl_by_sequence": kl}
                ],
                "exposure_curve": [
                    {
                        "features": 2,
                        "summed_active_frequency": 1.0,
                        "summed_activation_mass": 1.0,
                        "summed_decoded_energy": 1.0,
                        "collateral_kl": sum(kl) / len(kl),
                    }
                ],
            },
            "continuous_target": {"test": {"r2": r2}},
        }

    selection = MODULE.FrozenFeatureSelection(2, 0.06, 1.0, 0.01)
    rows = MODULE.paired_test_summary(
        {
            "mse_s0": result("mse", [1.0, 2.0], [0.2, 0.4], 0.3),
            "dpsae_s0": result("dpsae", [2.0, 4.0], [0.1, 0.2], 0.5),
        },
        selection,
        bootstrap_samples=32,
        seed=7,
    )

    assert len(rows) == 1
    assert rows[0]["ioi_effect_difference"]["paired_difference"] == 1.5
    assert rows[0]["natural_kl_difference"]["paired_difference"] < 0
    assert abs(rows[0]["continuous_target_r2_difference"] - 0.2) < 1e-12
