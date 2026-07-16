import copy
import importlib.machinery
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dpsae.language_model import ActivationStats
from dpsae.language_sae import BatchTopKSAE
from dpsae.saebench_adapter import (
    NativeBatchTopKSAEBenchAdapter,
    one_based_resid_post_hook,
)
from experiments import exp10_concept_discovery as runner


def native_payload(*, method: str, d_in: int = 3, d_sae: int = 5, k: int = 2):
    model = BatchTopKSAE(d_in, d_sae, k, seed=4)
    model.calibrate_threshold_(torch.randn(32, d_in, generator=torch.Generator().manual_seed(5)))
    return {
        "spec": {"name": f"{method}_s0", "method": method, "seed": 0, "k": k},
        "sparsity_mode": "batch_topk",
        "state_dict": model.state_dict(),
    }, model


def test_frozen_config_hashes_dataset_family_and_regularization_contracts():
    config = runner.load_config()

    assert len(config["benchmark"]["datasets"]) == 113
    assert config["model"]["hook_name"] == "blocks.7.hook_resid_post"
    assert config["benchmark"]["regularization"] == "l1"
    assert config["benchmark"]["companion_regularization"] == "l2"
    assert "unfiltered logreg baseline" in config["benchmark"][
        "companion_regularization_rationale"
    ]
    assert set(config["benchmark"]["family_by_dataset"]) == set(
        config["benchmark"]["datasets"]
    )
    assert config["runpod"]["gpu_count"] == 4
    assert config["runpod"]["network_volume_gib"] == 200
    assert config["runpod"]["pod_hour_usd"] == 1.8


def test_source_hashes_accepts_repository_relative_config_path():
    hashes = runner.source_hashes(Path("configs/exp10_concept_discovery.json"))

    assert "configs/exp10_concept_discovery.json" in hashes
    assert "experiments/exp10_concept_discovery.py" in hashes
    assert "src/dpsae/saebench_adapter.py" in hashes


def test_one_based_block_8_maps_to_transformerlens_block_7():
    assert one_based_resid_post_hook(8) == (7, "blocks.7.hook_resid_post")
    with pytest.raises(ValueError, match="positive"):
        one_based_resid_post_hook(0)


def test_native_adapter_matches_normalized_checkpoint_exactly():
    payload, native = native_payload(method="mse")
    stats = ActivationStats(mean=torch.tensor([1.0, -2.0, 0.5]), scale=torch.tensor(2.5))
    adapter = NativeBatchTopKSAEBenchAdapter(
        payload=payload,
        activation_stats=stats.state_dict(),
        model_name="pythia-160m-deduped",
        one_based_block=8,
        context_size=1024,
        device=torch.device("cpu"),
        expected_method="mse",
        expected_d_in=3,
        expected_d_sae=5,
        expected_k=2,
    )
    raw = torch.randn(9, 3, generator=torch.Generator().manual_seed(6))
    normalized = stats.normalize(raw)
    native.eval()
    native_code = native.encode(normalized, use_threshold=True)
    native_reconstruction = stats.denormalize(native.decode(native_code))

    torch.testing.assert_close(adapter.encode(raw), native_code)
    torch.testing.assert_close(adapter.decode(native_code), native_reconstruction)
    torch.testing.assert_close(adapter(raw), native_reconstruction)
    torch.testing.assert_close(adapter.W_dec, payload["state_dict"]["decoder_weight"])
    with pytest.raises(ValueError, match="float32"):
        adapter.to(dtype=torch.bfloat16)


def test_native_adapter_rejects_nonunit_decoder_without_renormalizing():
    payload, _ = native_payload(method="mse")
    payload["state_dict"]["decoder_weight"][0].mul_(1.1)
    before = payload["state_dict"]["decoder_weight"].clone()

    with pytest.raises(ValueError, match="renormalization is forbidden"):
        NativeBatchTopKSAEBenchAdapter(
            payload=payload,
            activation_stats={"mean": torch.zeros(3), "scale": torch.tensor(1.0)},
            model_name="pythia-160m-deduped",
            one_based_block=8,
            context_size=1024,
            device=torch.device("cpu"),
        )

    torch.testing.assert_close(payload["state_dict"]["decoder_weight"], before)


def _tiny_eligibility_config() -> dict:
    config = copy.deepcopy(runner.load_config())
    config["model"]["d_model"] = 3
    config["pilot_checkpoint"]["dictionary_size"] = 5
    config["pilot_checkpoint"]["target_l0"] = 2
    return config


def test_eligibility_accepts_explicit_artifact_root_and_records_hashes(tmp_path: Path):
    config = _tiny_eligibility_config()
    mse, _ = native_payload(method="mse")
    dpsae, _ = native_payload(method="dpsae")
    models_path = tmp_path / "models.pt"
    calibration_path = tmp_path / "calibration.pt"
    evaluation_path = tmp_path / "evaluation.json"
    torch.save({"mse_s0": mse, "dpsae_s0": dpsae}, models_path)
    torch.save(
        {"activation_stats": {"mean": torch.zeros(3), "scale": torch.tensor(1.5)}},
        calibration_path,
    )
    evaluation = {
        "complete": True,
        "models_sha256": runner.file_sha256(models_path),
        "calibration_sha256": runner.file_sha256(calibration_path),
        "models": {
            "mse_s0": {"nmse": 0.1, "inference_l0": 2.0},
            "dpsae_s0": {"nmse": 0.1005, "inference_l0": 2.02},
        },
    }
    evaluation_path.write_text(json.dumps(evaluation))

    report = runner.assess_eligibility(config, tmp_path)

    assert report["passed"]
    assert report["checkpoint_directory"] == str(tmp_path)
    assert report["artifact_hashes"]["models_sha256"] == runner.file_sha256(models_path)


def test_family_block_bootstrap_is_deterministic_and_family_aware():
    deltas = {"a": 0.02, "b": 0.01, "c": -0.005, "d": 0.015}
    families = {"a": "x", "b": "x", "c": "y", "d": "z"}

    first = runner.family_block_bootstrap(
        deltas, families, samples=200, seed=11, confidence_level=0.95
    )
    second = runner.family_block_bootstrap(
        deltas, families, samples=200, seed=11, confidence_level=0.95
    )

    assert first == second
    assert first["family_count"] == 3
    assert first["estimate"] == pytest.approx(0.01)


def test_heldout_ids_and_classifier_outputs_are_stable():
    config = runner.load_config()
    first = runner._heldout_identity(config, "test_dataset", 17, 3)
    second = runner._heldout_identity(config, "test_dataset", 17, 3)
    changed = runner._heldout_identity(config, "test_dataset", 18, 3)

    assert first == second
    assert first["split_id"] != changed["split_id"]
    assert len(first["example_ids"]) == len(set(first["example_ids"])) == 3

    class Classifier:
        def decision_function(self, X):
            return np.asarray(X)[:, 0] - 0.5

        def predict(self, X):
            return (self.decision_function(X) >= 0).astype(np.int64)

    outputs = runner._heldout_classifier_outputs(
        SimpleNamespace(classifier=Classifier()), np.asarray([[0.25], [0.75]])
    )
    torch.testing.assert_close(outputs["decision_score"], torch.tensor([-0.25, 0.25]))
    torch.testing.assert_close(outputs["prediction"], torch.tensor([0, 1]))


def test_sae_probes_packaged_data_preflight_hashes_without_import(tmp_path, monkeypatch):
    config = copy.deepcopy(runner.load_config())
    package = tmp_path / "sae_probes"
    master = package / "data/probing_datasets_MASTER.csv.zst"
    cleaned = package / "data/cleaned_data"
    cleaned.mkdir(parents=True)
    master.write_bytes(b"frozen master")
    for index in range(3):
        (cleaned / f"dataset_{index}.csv.zst").write_bytes(b"zstd")
    config["dependencies"]["sae_probes_master_data_sha256"] = runner.file_sha256(master)
    config["dependencies"]["sae_probes_expected_cleaned_data_files"] = 3
    spec = importlib.machinery.ModuleSpec("sae_probes", loader=None, is_package=True)
    spec.submodule_search_locations = [str(package)]
    monkeypatch.setattr("importlib.util.find_spec", lambda name: spec)

    observed = runner.verify_sae_probes_packaged_data(config)

    assert observed["master_data_sha256"] == runner.file_sha256(master)
    assert observed["cleaned_data_file_count"] == 3
