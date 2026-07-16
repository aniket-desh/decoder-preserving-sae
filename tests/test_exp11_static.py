import json
from pathlib import Path

import pytest

from dpsae.exp11_static import (
    confirmation_specs,
    screen_specs,
    select_matched_spectral,
    summarize_confirmation,
)
from experiments import exp11_static_matched_nmse as runner


ROOT = Path(__file__).resolve().parents[1]


def _screen_metrics():
    return {
        "mse_s0": {"nmse": 0.1, "decoder": 0.2},
        "dpsae_s0": {"nmse": 0.107, "decoder": 0.15},
        "spectral_b2_s0": {"nmse": 0.106, "decoder": 0.154},
        "spectral_b4_s0": {"nmse": 0.108, "decoder": 0.15},
        "spectral_b8_s0": {"nmse": 0.12, "decoder": 0.13},
    }


def test_config_freezes_grid_match_and_advancement_before_results():
    config = runner.load_config(ROOT / "configs" / "exp11_static_matched_nmse.json")

    assert config["screen"] == {
        "seed": 0,
        "training_tokens": 25_000_000,
        "corpus_range": "screen",
        "evaluation_split": "selection",
        "beta_grid": [2.0, 4.0, 8.0, 16.0, 32.0],
        "target_nmse_ratio": 1.07,
        "matching_tolerance": 0.01,
        "decoder_reduction_margin": 0.02,
        "randomness_stage": "exp11_spectral_screen",
        "randomness_replicate": 0,
    }
    assert config["confirmation"]["seeds"] == [0, 1, 2]
    assert config["confirmation"]["training_tokens"] == 100_000_000
    assert config["execution_allocation"] == {
        "provider": "RunPod",
        "gpu_model": "NVIDIA A40 48GB",
        "gpu_count": 4,
        "assigned_physical_gpu": 3,
        "pod_rate_usd_per_hour": 1.8,
        "gpu_rate_usd_per_hour": 0.45,
        "volume_gb": 200,
    }
    assert config["reference_confirmation"]["expected_models_sha256"] == (
        "227e60c2162e0a948dbd53b0c3afbb063435ec288fec47e33e5583193cc7c781"
    )


def test_screen_specs_are_paired_and_spectral_only():
    specs = screen_specs(
        k=32, seed=0, decoder_weight=0.25, beta_grid=[2, 4, 8]
    )

    assert [(spec.method, spec.loss_weight) for spec in specs] == [
        ("mse", 1.0),
        ("dpsae", 1.0),
        ("spectral", 2.0),
        ("spectral", 4.0),
        ("spectral", 8.0),
    ]
    assert specs[1].decoder_weight == 0.25
    assert all(spec.seed == 0 and spec.k == 32 for spec in specs)


def test_closest_inclusive_nmse_match_ties_to_smaller_beta_and_advances_at_margin():
    specs = screen_specs(
        k=32, seed=0, decoder_weight=0.25, beta_grid=[2, 4, 8]
    )

    report = select_matched_spectral(
        _screen_metrics(),
        specs,
        split="selection",
        target_nmse_ratio=1.07,
        matching_tolerance=0.01,
        decoder_reduction_margin=0.02,
    )

    assert report["selected"]["spec"]["loss_weight"] == 2
    assert report["selected"]["nmse_ratio"] == pytest.approx(1.06)
    assert report["decoder_reduction_gap"] == pytest.approx(0.02)
    assert report["advance"] is True
    assert report["status"] == "advance"


def test_matched_but_noncompetitive_point_does_not_advance():
    specs = screen_specs(k=32, seed=0, decoder_weight=0.25, beta_grid=[2])
    metrics = _screen_metrics()
    metrics["spectral_b2_s0"]["decoder"] = 0.17

    report = select_matched_spectral(
        metrics,
        specs,
        split="selection",
        target_nmse_ratio=1.07,
        matching_tolerance=0.01,
        decoder_reduction_margin=0.02,
    )

    assert report["selected"] is not None
    assert report["advance"] is False
    assert report["status"] == "noncompetitive_match"


def test_no_nmse_match_records_nonadvance_without_fallback():
    specs = screen_specs(k=32, seed=0, decoder_weight=0.25, beta_grid=[8])

    report = select_matched_spectral(
        _screen_metrics(),
        specs,
        split="selection",
        target_nmse_ratio=1.07,
        matching_tolerance=0.01,
        decoder_reduction_margin=0.02,
    )

    assert report["selected"] is None
    assert report["advance"] is False
    assert report["status"] == "no_matching_candidate"
    with pytest.raises(ValueError, match="selection data"):
        select_matched_spectral(
            _screen_metrics(),
            specs,
            split="test",
            target_nmse_ratio=1.07,
            matching_tolerance=0.01,
            decoder_reduction_margin=0.02,
        )


def test_confirmation_specs_and_summary_remain_paired():
    specs = confirmation_specs(
        k=32, seeds=[0, 1, 2], decoder_weight=0.25, spectral_beta=4
    )
    assert [spec.method for spec in specs] == ["mse", "dpsae", "spectral"] * 3
    assert all(
        spec.loss_weight == 4 for spec in specs if spec.method == "spectral"
    )
    metrics = {}
    for seed in range(3):
        metrics[f"mse_s{seed}"] = {"nmse": 0.1, "decoder": 0.2}
        metrics[f"dpsae_s{seed}"] = {"nmse": 0.107, "decoder": 0.15}
        metrics[f"spectral_s{seed}"] = {"nmse": 0.106, "decoder": 0.16}

    summary = summarize_confirmation(metrics, [0, 1, 2])

    assert summary["confirmatory_gate"] is None
    assert len(summary["seeds"]) == 3
    assert summary["seeds"][0]["dpsae"]["decoder_reduction"] == pytest.approx(0.25)
    assert summary["seeds"][0]["spectral"]["nmse_ratio"] == pytest.approx(1.06)


def test_reference_artifact_root_override_and_hash_are_validated(tmp_path, monkeypatch):
    config = runner.load_config(ROOT / "configs" / "exp11_static_matched_nmse.json")
    reference = tmp_path / "restored" / "confirmation_common"
    reference.mkdir(parents=True)
    (reference / "models.pt").write_bytes(b"exact-models")
    (reference / "done.json").write_text('{"complete": true}\n')
    paths = runner.experiment_paths(
        config, root=tmp_path, reference_artifact_root=reference
    )
    required = runner._required_inputs(paths, "screen")
    for name, path in required.items():
        if name != "reference_models":
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"placeholder")
    paths.source_selection.write_text('{"selected_decoder_weight": 0.25}\n')

    monkeypatch.setattr(
        runner,
        "sha256_file",
        lambda path, *_args, **_kwargs: (
            config["reference_confirmation"]["expected_models_sha256"]
            if path == paths.reference_models
            else "test-sha"
        ),
    )
    monkeypatch.setattr(
        runner.torch,
        "load",
        lambda *_args, **_kwargs: {
            "split": "selection",
            "input_ids": object(),
            "activations": object(),
            "starts": object(),
        },
    )

    manifest = runner.validate_inputs(config, paths, "screen")

    assert paths.reference_models == reference / "models.pt"
    assert manifest["reference_models"]["sha256"] == (
        config["reference_confirmation"]["expected_models_sha256"]
    )
    assert {item["path"] for item in manifest["reference_inventory"]["files"]} == {
        "done.json",
        "models.pt",
    }


def test_nonadvancing_screen_writes_machine_readable_confirmation_skip(tmp_path):
    config = {
        "config_sha256": "sealed",
        "repository": {"revision": "abc", "dirty": False, "status": []},
    }
    output = tmp_path / "exp11"
    paths = runner.ExperimentPaths(
        output=output,
        source_artifact=tmp_path / "source",
        evaluation_artifact=tmp_path / "evaluation",
        source_tokens=tmp_path / "tokens.bin",
        source_calibration=tmp_path / "calibration.pt",
        source_selection=tmp_path / "selection.json",
        static_calibration=tmp_path / "static.pt",
        selection_cache=tmp_path / "selection.pt",
        test_cache=tmp_path / "test.pt",
        reference_root=tmp_path / "reference",
        reference_models=tmp_path / "reference" / "models.pt",
        decision=output / "screen" / "decision.json",
    )
    paths.decision.parent.mkdir(parents=True)
    paths.decision.write_text(
        json.dumps(
            {
                "config_sha256": "sealed",
                "repository": {"revision": "abc", "dirty": False, "status": []},
                "advance": False,
                "status": "noncompetitive_match",
            }
        )
    )

    report = runner.run_confirmation(config, paths, runner.torch.device("cpu"))

    assert report["status"] == "not_run_by_predeclared_gate"
    assert (output / "confirmation" / "not_run.json").exists()
