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


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, sort_keys=True))


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
    assert config["benchmark"]["companion_full_code_matrix_format"] == "scipy_csr_exact_values"
    assert config["benchmark"]["saebench_include_llm_baseline"] is False
    assert "unfiltered logreg baseline" in config["benchmark"]["companion_regularization_rationale"]
    assert set(config["benchmark"]["family_by_dataset"]) == set(config["benchmark"]["datasets"])
    assert config["runpod"]["gpu_count"] == 4
    assert config["runpod"]["network_volume_gib"] == 200
    assert config["runpod"]["pod_hour_usd"] == 1.8
    runtime = config["runtime"]
    assert runtime["worker_count"] == 4
    assert [len(shard) for shard in runtime["companion_seed_shards"]] == [3, 3, 2, 2]
    assert runtime["cache_adapters_once_per_worker"] is True
    assert (
        runtime["cold_cache_timing_source_policy"]
        == "in_process_monotonic_or_hash_bound_external_provenance"
    )
    assert runtime["timing_smoke"]["probe_seed"] == 2027071799
    assert runtime["timing_smoke"]["task_count"] == 8
    assert runtime["timing_smoke"]["headroom_multiplier"] == 1.3
    assert runtime["timing_smoke"]["maximum_projected_pod_hours"] == 3.0


def test_source_hashes_accepts_repository_relative_config_path():
    hashes = runner.source_hashes(Path("configs/exp10_concept_discovery.json"))

    assert "configs/exp10_concept_discovery.json" in hashes
    assert "experiments/exp10_concept_discovery.py" in hashes
    assert "src/dpsae/saebench_adapter.py" in hashes


def test_repository_state_requires_clean_revision(monkeypatch):
    responses = iter(["abc123\n", ""])
    monkeypatch.setattr(runner.subprocess, "check_output", lambda *_a, **_k: next(responses))

    assert runner.repository_state() == {
        "revision": "abc123",
        "dirty": False,
        "status": [],
    }


def test_resolved_contract_ignores_only_measured_import_time():
    first = {
        "environment": {"sae_probes_eager_import_seconds": 61.0, "versions": {"x": "1"}},
        "source_hashes": {"runner": "abc"},
    }
    second = copy.deepcopy(first)
    second["environment"]["sae_probes_eager_import_seconds"] = 92.0

    assert runner.stable_resolved_contract(first) == runner.stable_resolved_contract(second)
    second["source_hashes"]["runner"] = "changed"
    assert runner.stable_resolved_contract(first) != runner.stable_resolved_contract(second)


def test_launcher_records_environment_and_deployed_sources():
    launcher = (runner.ROOT / "scripts/run_exp10_concept_4xa40.sh").read_text()
    assert "git status --porcelain=v1 --untracked-files=all" in launcher
    assert "environment-pip-freeze.txt" in launcher
    assert "deployed-source-sha256.txt" in launcher
    assert "--phase pre-aggregate --wait-seconds 172800 &&" in launcher
    assert "aggregate && $AUDITOR --phase final" in launcher
    assert "audit_exp10_artifacts.py" in launcher


def test_timing_launcher_self_detaches_into_tmux():
    launcher = (runner.ROOT / "scripts/run_exp10_timing_smoke_a40.sh").read_text()

    assert 'MODE="${1:-}"' in launcher
    assert 'tmux new-session -d -s "$SESSION"' in launcher
    assert "run_exp10_timing_smoke_a40.sh' --worker" in launcher
    assert "timing-preflight" in launcher
    assert "--cold-cache-provenance" in launcher


def test_timing_smoke_selection_is_size_only_deterministic_and_stratified():
    config = runner.load_config()
    sizes = {dataset: 200 + index for index, dataset in enumerate(config["benchmark"]["datasets"])}

    first = runner.select_timing_smoke_tasks(config, sizes)
    second = runner.select_timing_smoke_tasks(config, dict(reversed(list(sizes.items()))))

    assert first == second
    assert len(first) == len({item["dataset"] for item in first}) == 8
    assert [sum(item["quartile"] == quartile for item in first) for quartile in range(4)] == [
        2,
        2,
        2,
        2,
    ]
    assert all(item["n_train"] == min(item["dataset_size"] - 100, 1024) for item in first)


def test_timing_smoke_gate_accepts_only_the_frozen_blind_contract(tmp_path: Path):
    config = runner.load_config()
    smoke = config["runtime"]["timing_smoke"]
    (tmp_path / "cache_ready.json").write_text(
        json.dumps(
            {
                "generation_seconds": 100.0,
                "generation_timing_source": "in_process_monotonic",
            }
        )
    )
    report = {
        "schema_version": 2,
        "complete": True,
        "passed": True,
        "config_digest": runner.canonical_digest(config),
        "probe_seed": smoke["probe_seed"],
        "task_count": smoke["task_count"],
        "names_and_concept_results_suppressed": True,
        "saved_concept_metric_count": 0,
        "companion_full_code_matrix_format": "scipy_csr_exact_values",
        "cache_generation_timing": {
            "source": "in_process_monotonic",
            "generation_seconds": 100.0,
        },
        "projection": {
            "projected_pod_hours": 2.9,
            "cache_generation_seconds": 100.0,
        },
    }
    (tmp_path / "timing_smoke.json").write_text(json.dumps(report))

    assert runner.verify_timing_smoke_gate(config, tmp_path) == report
    report["names_and_concept_results_suppressed"] = False
    (tmp_path / "timing_smoke.json").write_text(json.dumps(report))
    with pytest.raises(RuntimeError, match="names_and_concept_results_suppressed"):
        runner.verify_timing_smoke_gate(config, tmp_path)


def test_full_code_csr_preserves_values_and_lbfgs_outputs():
    from sklearn.linear_model import LogisticRegression

    generator = torch.Generator().manual_seed(7)
    dense = torch.zeros(180, 256)
    indices = torch.randint(0, dense.shape[1], (180, 8), generator=generator)
    dense.scatter_(1, indices, torch.randn(180, 8, generator=generator))
    labels = ((dense[:, 3] - 0.7 * dense[:, 17]) > 0).numpy().astype(np.int64)
    train_dense, test_dense = dense[:140], dense[140:]
    train_csr = runner._full_code_csr(train_dense)
    test_csr = runner._full_code_csr(test_dense)

    assert len(train_csr) == len(train_dense)
    assert len(test_csr) == len(test_dense)
    np.testing.assert_array_equal(train_csr.toarray(), train_dense.numpy())
    np.testing.assert_array_equal(test_csr.toarray(), test_dense.numpy())
    assert train_csr.nnz == int(torch.count_nonzero(train_dense))
    assert test_csr.nnz == int(torch.count_nonzero(test_dense))

    dense_fit = LogisticRegression(C=1.0, random_state=11, max_iter=1000).fit(
        train_dense.numpy(), labels[:140]
    )
    csr_fit = LogisticRegression(C=1.0, random_state=11, max_iter=1000).fit(
        train_csr, labels[:140]
    )
    np.testing.assert_allclose(csr_fit.coef_, dense_fit.coef_, rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(
        csr_fit.decision_function(test_csr),
        dense_fit.decision_function(test_dense.numpy()),
        rtol=1e-10,
        atol=1e-10,
    )
    np.testing.assert_array_equal(
        csr_fit.predict(test_csr), dense_fit.predict(test_dense.numpy())
    )


def test_timing_gate_rejects_obsolete_dense_schema(tmp_path: Path):
    config = runner.load_config()
    smoke = config["runtime"]["timing_smoke"]
    (tmp_path / "timing_smoke.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "complete": True,
                "passed": True,
                "config_digest": runner.canonical_digest(config),
                "probe_seed": smoke["probe_seed"],
                "task_count": smoke["task_count"],
                "names_and_concept_results_suppressed": True,
                "saved_concept_metric_count": 0,
            }
        )
    )

    with pytest.raises(RuntimeError, match="schema_version"):
        runner.verify_timing_smoke_gate(config, tmp_path)


def test_timing_thread_quota_uses_four_way_cpu_affinity(monkeypatch):
    config = runner.load_config()
    monkeypatch.setattr(runner.os, "sched_getaffinity", lambda _pid: set(range(12)), raising=False)
    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.setenv(name, "3")

    observed = runner._timing_thread_quota(config)

    assert observed["visible_cpu_count"] == 12
    assert observed["threads_per_worker"] == 3


def test_timing_projection_uses_the_slowest_exact_worker_shard():
    config = runner.load_config()
    rows = [
        {
            "total_seconds": 13.0,
            "stage_seconds": {
                "sparse_method_0": {"total": 1.0},
                "sparse_method_1": {"total": 2.0},
                "companion": {"total": 10.0},
            },
        }
        for _ in range(8)
    ]

    projection = runner._project_timing_smoke(config, rows, cache_generation_seconds=100.0)

    expected_slowest_seconds = 113 * (5 * 1.0 + 3 * 10.0) * 1.3
    assert projection["worker_projections"][0][
        "p95_workload_seconds_with_headroom"
    ] == pytest.approx(expected_slowest_seconds)
    assert projection["projected_workload_seconds_with_headroom"] == pytest.approx(
        expected_slowest_seconds
    )
    assert projection["projected_pod_hours"] == pytest.approx(
        (100.0 + expected_slowest_seconds) / 3600
    )


def test_external_cold_cache_timing_is_hash_bound_recorded_and_projected(
    tmp_path: Path,
):
    config = runner.load_config()
    model_cache = tmp_path / "model-cache"
    model_cache.mkdir()
    cache = {
        "model_cache": str(model_cache.resolve()),
        "files": {
            "opaque_a": {"sha256": "a" * 64},
            "opaque_b": {"sha256": "b" * 64},
        },
    }
    _write_json(tmp_path / "cache_ready.json", cache)
    provenance = {
        "schema_version": 1,
        "complete": True,
        "start_unix_seconds": 1784234613,
        "end_unix_seconds": 1784236119,
        "generation_seconds": 1506,
        "source_cache_ready_path": str((tmp_path / "cache_ready.json").resolve()),
        "source_cache_ready_sha256": runner.file_sha256(tmp_path / "cache_ready.json"),
        "model_cache_path": str(model_cache.resolve()),
        "cache_file_hashes_sha256": runner._cache_file_hashes_digest(cache),
    }
    provenance_path = tmp_path / "cold-cache.json"
    _write_json(provenance_path, provenance)

    timing = runner._resolve_cache_generation_timing(
        cache,
        output_root=tmp_path,
        model_cache=model_cache,
        provenance_path=provenance_path,
    )
    rows = [
        {
            "total_seconds": 13.0,
            "stage_seconds": {
                "sparse_method_0": {"total": 1.0},
                "sparse_method_1": {"total": 2.0},
                "companion": {"total": 10.0},
            },
        }
        for _ in range(8)
    ]
    report = runner._assemble_timing_smoke_report(
        config,
        resolved={
            "config_digest": runner.canonical_digest(config),
            "artifact_hashes": {"models_sha256": "model"},
        },
        rows=rows,
        selection_digest="selection",
        thread_quota={"threads_per_worker": 1},
        cache_generation_timing=timing,
    )
    runner.atomic_json(tmp_path / "timing_smoke.json", report)
    observed = runner.read_json(tmp_path / "timing_smoke.json")

    expected_workload = 113 * (5 * 1.0 + 3 * 10.0) * 1.3
    assert observed["cache_generation_timing"]["generation_seconds"] == 1506
    assert observed["cache_generation_timing"]["provenance_sha256"] == runner.file_sha256(
        provenance_path
    )
    assert observed["projection"]["cache_generation_seconds"] == 1506
    assert observed["projection"]["projected_pod_hours"] == pytest.approx(
        (1506 + expected_workload) / 3600
    )
    assert runner.verify_timing_smoke_gate(config, tmp_path) == observed


def test_prepare_cache_adopts_prior_manifest_only_with_bound_provenance(
    tmp_path: Path,
):
    config = copy.deepcopy(runner.load_config())
    config["benchmark"]["datasets"] = ["opaque_task"]
    config["benchmark"]["dataset_manifest_sha256"] = runner.canonical_digest(
        config["benchmark"]["datasets"]
    )
    model_cache = tmp_path / "model-cache"
    cache_path = runner.cache_files(config, model_cache)["opaque_task"]
    cache_path.parent.mkdir(parents=True)
    cache_path.write_bytes(b"immutable-cache")
    old_cache = {
        "schema_version": 1,
        "complete": True,
        "config_digest": "prior-config",
        "model_cache": str(model_cache.resolve()),
        "dataset_manifest_sha256": config["benchmark"]["dataset_manifest_sha256"],
        "files": {
            "opaque_task": {
                "path": str(cache_path.resolve()),
                "shape": [1, config["model"]["d_model"]],
                "bytes": cache_path.stat().st_size,
                "sha256": runner.file_sha256(cache_path),
            }
        },
    }
    source_root = tmp_path / "prior-output"
    source_root.mkdir()
    source_cache_ready = source_root / "cache_ready.json"
    source_cache_ready.write_text(json.dumps(old_cache))
    output_root = tmp_path / "output"
    output_root.mkdir()
    _write_json(
        output_root / "resolved_config.json",
        {"config_digest": runner.canonical_digest(config)},
    )
    provenance_path = output_root / "cold-cache.json"
    with pytest.raises(RuntimeError, match="disagree with end minus start"):
        runner.record_cold_cache_timing_provenance(
            config=config,
            source_cache_ready=source_cache_ready,
            model_cache=model_cache,
            start_unix_seconds=1784234613,
            end_unix_seconds=1784236119,
            expected_generation_seconds=1505,
            output_path=provenance_path,
        )
    provenance_summary = runner.record_cold_cache_timing_provenance(
        config=config,
        source_cache_ready=source_cache_ready,
        model_cache=model_cache,
        start_unix_seconds=1784234613,
        end_unix_seconds=1784236119,
        expected_generation_seconds=1506,
        output_path=provenance_path,
    )
    provenance = runner.read_json(provenance_path)

    adopted = runner.prepare_cache(
        config=config,
        output_root=output_root,
        model_cache=model_cache,
        device="cpu",
        cold_cache_provenance=provenance_path,
    )

    assert adopted["config_digest"] == runner.canonical_digest(config)
    assert adopted["generation_seconds"] == 1506
    assert provenance_summary["generation_seconds"] == 1506
    assert adopted["generation_timing_source"] == "external_hash_bound_provenance"
    assert adopted["adopted_from_cache_ready_sha256"] == provenance["source_cache_ready_sha256"]


def test_worker_reuses_two_adapters_and_runs_sparse_before_companion(tmp_path: Path, monkeypatch):
    config = runner.load_config()
    runtime = config["runtime"]
    sparse = runtime["sparse_worker_shards"][0]
    companion = runtime["companion_seed_shards"][0]
    timing_report = {
        "projection": {"projected_pod_hours": 2.0},
    }
    (tmp_path / "timing_smoke.json").write_text(json.dumps(timing_report))
    adapters = {method: SimpleNamespace(method=method) for method in ("mse", "dpsae")}
    loaded = []
    calls = []

    monkeypatch.setattr(runner, "wait_cache", lambda **_kwargs: {"ready": True})
    monkeypatch.setattr(runner, "verify_timing_smoke_gate", lambda *_args, **_kwargs: timing_report)

    def fake_load_adapter(_config, _checkpoint_dir, method, _device):
        loaded.append(method)
        return adapters[method]

    def fake_sparse_job(**kwargs):
        calls.append(("sparse", kwargs["probe_seed"], kwargs["adapter"]))
        return {"complete": True}

    def fake_companion_job(**kwargs):
        calls.append(("companion", kwargs["probe_seed"], kwargs["adapters"]))
        return {"complete": True}

    monkeypatch.setattr(runner, "load_adapter", fake_load_adapter)
    monkeypatch.setattr(runner, "run_sparse_job", fake_sparse_job)
    monkeypatch.setattr(runner, "run_companion_job", fake_companion_job)

    result = runner.run_worker(
        config=config,
        output_root=tmp_path,
        checkpoint_dir=tmp_path,
        model_cache=tmp_path,
        worker_index=0,
        cache_role="wait",
        method=sparse["method"],
        probe_seeds=sparse["probe_seeds"],
        companion_seeds=companion,
        device="cpu",
        dependency_preflight={},
    )

    assert loaded == ["mse", "dpsae"]
    assert [kind for kind, _seed, _adapter in calls] == ["sparse"] * 5 + ["companion"] * 3
    assert all(call[2] is adapters["mse"] for call in calls[:5])
    assert all(call[2]["mse"] is adapters["mse"] for call in calls[5:])
    assert all(call[2]["dpsae"] is adapters["dpsae"] for call in calls[5:])
    assert result["sparse_job_count"] == 5
    assert result["companion_job_count"] == 3


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
