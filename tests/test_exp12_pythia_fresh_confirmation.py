import copy
import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from experiments import exp12_pythia_fresh_confirmation as runner


def frozen_config() -> dict:
    config = copy.deepcopy(
        runner.load_config(runner.DEFAULT_CONFIG, require_frozen=False)
    )
    config["status"] = "frozen"
    config["model"]["revision"] = "a" * 40
    config["confirmation"]["pair_seeds"] = [11, 22, 33]
    config["corpus"].update(
        {
            "cache_absolute_range": [50_000_128, 302_000_128],
            "calibration_absolute_range": [50_000_128, 50_065_664],
            "maturity_evaluation_absolute_range": [50_065_664, 50_131_200],
            "frozen_network_absolute_range": [50_131_200, 50_655_488],
            "training_absolute_range": [51_048_704, 302_000_128],
            "allow_training_cache_reuse": False,
        }
    )
    config["training"]["maximum_tokens"] = 250_000_000
    config["training"]["scheduler_horizon_tokens"] = 250_000_000
    config["maturity_evaluation"]["exact_decoder"]["bootstrap_seed"] = 700
    config["maturity_stop_rule"].update(
        {
            "rule_version": "pythia_block8_maturity_v1",
            "minimum_checkpoint_tokens": 100_000_000,
            "plateau_consecutive_intervals": 2,
            "maximum_relative_change": {
                "nmse": 0.02,
                "exact_decoder_distortion": 0.02,
                "inference_l0": 0.02,
                "dead_feature_fraction": 1.0,
            },
            "extension_to_500m_policy": "not_applicable_maximum_250m",
            "minimum_unique_exposure_for_500m": 500_000_000,
            "maximum_cache_reuse_count_for_500m": 0,
            "no_plateau_by_maximum_policy": "fail_without_common_checkpoint",
        }
    )
    config["frozen_network_evaluation"]["sampling_seed"] = 701
    config["frozen_network_evaluation"]["bootstrap_seed"] = 702
    config["randomness"]["base_seed"] = 703
    return config


def maturity_snapshots(config: dict) -> dict[int, dict[int, dict]]:
    values = {
        25_000_000: (1.20, 0.60, 31.8, 0.00100),
        50_000_000: (1.00, 0.50, 31.9, 0.00080),
        100_000_000: (0.99, 0.495, 32.0, 0.00079),
        250_000_000: (0.985, 0.492, 32.1, 0.000785),
    }
    result = {}
    for seed in config["confirmation"]["pair_seeds"]:
        result[seed] = {}
        for budget, (nmse, distortion, l0, dead) in values.items():
            result[seed][budget] = {
                "requested_snapshot_tokens": budget,
                "realized_snapshot_tokens": budget,
                "corpus_exposure": {
                    "unique_corpus_exposure_tokens": budget,
                    "cache_reuse_count": 0,
                },
                "models": {
                    "mse": {
                        "nmse": nmse,
                        "exact_decoder_distortion": distortion,
                        "inference_l0": l0,
                        "dead_feature_fraction": dead,
                    },
                    "dpsae": {
                        "nmse": nmse * 1.005,
                        "exact_decoder_distortion": distortion * 0.99,
                        "inference_l0": l0 + 0.05,
                        "dead_feature_fraction": dead * 0.99,
                    },
                },
            }
    return result


def test_checked_in_config_is_frozen_and_complete() -> None:
    config = runner.load_config(runner.DEFAULT_CONFIG, require_frozen=False)
    choices = runner.unresolved_choices(config)

    assert config["status"] == "frozen"
    assert len(runner.UNRESOLVED_PATHS) == 26
    assert choices == []
    assert config["model"]["revision"] == "582159a2dfe3e712a8d47ae83dec95ae3bde8e7e"
    assert config["confirmation"]["pair_seeds"] == [1, 2, 3]
    assert config["training"]["maximum_tokens"] == 250_000_000
    assert config["corpus"]["allow_training_cache_reuse"] is False
    assert config["maturity_stop_rule"]["no_plateau_by_maximum_policy"] == (
        "fail_without_common_checkpoint"
    )
    runner.validate_config(config, require_frozen=True)


def test_preflight_refuses_a_missing_pilot_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "run"
    monkeypatch.setattr(
        runner,
        "repository_state",
        lambda: {"commit": "abc", "dirty": False, "status": []},
    )

    with pytest.raises(FileNotFoundError):
        runner.write_preflight_contract(
            config_path=runner.DEFAULT_CONFIG,
            output_root=output,
            pilot_report_path=tmp_path / "must-not-be-opened.json",
        )

    assert not (output / "freeze_blocked.json").exists()


def test_fully_prespecified_contract_is_valid_and_hashes_the_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = frozen_config()
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    pilot_path = tmp_path / "pilot.json"
    pilot_path.write_text(
        json.dumps(
            {
                "complete": True,
                "advance_fresh_confirmation": True,
                "advance_autointerp": False,
            }
        )
    )
    monkeypatch.setattr(
        runner,
        "repository_state",
        lambda: {"revision": "deadbeef", "dirty": False, "status": []},
    )

    rule = runner.write_preflight_contract(
        config_path=config_path,
        output_root=tmp_path / "run",
        pilot_report_path=pilot_path,
    )

    expected = dict(rule)
    digest = expected.pop("rule_digest")
    assert digest == runner.canonical_digest(expected)
    assert rule["frozen_before_training"] is True
    assert rule["concept_blind"] is True
    assert rule["maximum_tokens"] == 250_000_000


def test_sequential_stream_tracks_unique_exposure_and_reuse_exactly(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    np.arange(32, dtype=np.uint16).tofile(path)
    stream = runner.SequentialTokenStream(
        path,
        cache_absolute_range=(100, 132),
        training_absolute_range=(104, 120),
        sequence_length=4,
        batch_size=2,
        allow_reuse=True,
    )

    first, first_exposure = stream.batch()
    second, second_exposure = stream.batch()
    third, third_exposure = stream.batch()

    assert first.tolist() == [list(range(4, 8)), list(range(8, 12))]
    assert second.tolist() == [list(range(12, 16)), list(range(16, 20))]
    assert first_exposure["absolute_sequence_starts"] == [104, 108]
    assert second_exposure["unique_corpus_exposure_tokens"] == 16
    assert third.tolist() == first.tolist()
    assert third_exposure["delivered_tokens"] == 24
    assert third_exposure["unique_corpus_exposure_tokens"] == 16
    assert third_exposure["reused_tokens"] == 8
    assert third_exposure["cache_epoch"] == 1
    assert third_exposure["cache_reuse_count"] == 1


def test_sequential_stream_refuses_unprespecified_reuse(tmp_path: Path) -> None:
    path = tmp_path / "tokens.bin"
    np.arange(16, dtype=np.uint16).tofile(path)
    stream = runner.SequentialTokenStream(
        path,
        cache_absolute_range=(0, 16),
        training_absolute_range=(0, 16),
        sequence_length=4,
        batch_size=2,
        allow_reuse=False,
    )

    stream.batch()
    stream.batch()
    with pytest.raises(RuntimeError, match="exhausted"):
        stream.batch()


def test_maturity_rule_selects_first_common_plateau_without_concept_inputs() -> None:
    config = frozen_config()
    snapshots = maturity_snapshots(config)

    decision = runner.build_maturity_decision(
        config,
        snapshots,
        stop_rule_contract_sha256="abc",
    )

    assert decision["common_checkpoint_selected"] is True
    assert decision["selected_requested_snapshot_tokens"] == 250_000_000
    assert decision["fallback_applied"] is False
    assert [row["eligible"] for row in decision["candidate_checkpoints"]] == [
        False,
        False,
        False,
        True,
    ]


def test_maturity_rule_rejects_concept_facing_keys_but_allows_dead_feature_metric() -> None:
    config = frozen_config()
    snapshots = maturity_snapshots(config)
    snapshots[11][25_000_000]["concept_auc"] = 0.9

    with pytest.raises(ValueError, match="concept-facing"):
        runner.build_maturity_decision(
            config,
            snapshots,
            stop_rule_contract_sha256="abc",
        )


def test_maturity_rule_requires_absolute_dead_feature_quality() -> None:
    config = frozen_config()
    snapshots = maturity_snapshots(config)
    snapshots[11][250_000_000]["models"]["dpsae"]["dead_feature_fraction"] = 0.0101

    decision = runner.build_maturity_decision(
        config,
        snapshots,
        stop_rule_contract_sha256="abc",
    )

    final = decision["candidate_checkpoints"][-1]
    assert final["matched_quality"]["11"]["dead_features"] is False
    assert decision["common_checkpoint_selected"] is False


def test_dead_feature_plateau_uses_one_atom_denominator_floor() -> None:
    config = frozen_config()
    snapshots = maturity_snapshots(config)
    atom = 1 / config["sae"]["dictionary_size"]
    for seed in config["confirmation"]["pair_seeds"]:
        for method in runner.METHODS:
            snapshots[seed][50_000_000]["models"][method]["dead_feature_fraction"] = 0.0
            snapshots[seed][100_000_000]["models"][method]["dead_feature_fraction"] = atom
            snapshots[seed][250_000_000]["models"][method]["dead_feature_fraction"] = atom

    decision = runner.build_maturity_decision(
        config,
        snapshots,
        stop_rule_contract_sha256="abc",
    )

    transition = decision["candidate_checkpoints"][2]["relative_changes"][
        "50000000_to_100000000"
    ]
    row = transition[str(config["confirmation"]["pair_seeds"][0])]["mse"][
        "dead_feature_fraction"
    ]
    assert row["reference_denominator"] == pytest.approx(atom)
    assert row["relative_change"] == pytest.approx(1.0)
    assert row["passed"] is True


def test_timeout_is_bounded_by_frozen_cost_ceiling() -> None:
    config = frozen_config()

    assert runner.timeout_seconds_for_run(config, 8) == 8 * 3600
    with pytest.raises(ValueError, match="positive"):
        runner.timeout_seconds_for_run(config, 0)
    with pytest.raises(ValueError, match="exceeds"):
        runner.timeout_seconds_for_run(config, 8.01)


def test_nonreport_smoke_projects_runtime_and_memory_without_quality_metrics() -> None:
    config = frozen_config()
    gate = runner.build_timing_smoke_gate(
        config,
        {
            "cache_wall_seconds": 10,
            "setup_wall_seconds": 20,
            "training_wall_seconds": 100,
            "smoke_wall_seconds": 140,
            "peak_reserved_gpu_gib": 40,
            "smoke_realized_training_tokens": runner.realized_tokens(2_000_000, 2048),
            "full_confirmation_cache_reused": False,
        },
    )

    assert gate["passed"] is True
    assert gate["reportable"] is False
    assert gate["model_quality_metrics_retained"] is False
    assert gate["concept_outcomes_opened_by_this_stage"] is False
    assert set(gate["measurements"]) == {
        "cache_wall_seconds",
        "setup_wall_seconds",
        "training_wall_seconds",
        "smoke_wall_seconds",
        "peak_reserved_gpu_gib",
    }

    memory_failure = runner.build_timing_smoke_gate(
        config,
        {
            "cache_wall_seconds": 0,
            "setup_wall_seconds": 20,
            "training_wall_seconds": 100,
            "smoke_wall_seconds": 140,
            "peak_reserved_gpu_gib": 44.01,
            "smoke_realized_training_tokens": runner.realized_tokens(2_000_000, 2048),
            "full_confirmation_cache_reused": True,
        },
    )
    assert memory_failure["passed"] is False
    assert memory_failure["gates"]["peak_reserved_gpu_memory"] is False

    runtime_failure = runner.build_timing_smoke_gate(
        config,
        {
            "cache_wall_seconds": 0,
            "setup_wall_seconds": 20,
            "training_wall_seconds": 200,
            "smoke_wall_seconds": 240,
            "peak_reserved_gpu_gib": 40,
            "smoke_realized_training_tokens": runner.realized_tokens(2_000_000, 2048),
            "full_confirmation_cache_reused": True,
        },
    )
    assert runtime_failure["passed"] is False
    assert runtime_failure["gates"]["projected_confirmation_wall_time"] is False


def test_timing_smoke_retains_only_nonreport_timing_artifacts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = frozen_config()
    config["calibration"]["activation_tokens"] = 256
    config["sae"]["ridge_calibration_groups"] = 1
    config["training"]["maximum_tokens"] = 2048
    config["training"]["scheduler_horizon_tokens"] = 2048
    config["corpus"]["cache_absolute_range"] = [4096, 6400]
    config["timing_smoke"].update(
        {
            "cache_absolute_range": [0, 2304],
            "calibration_absolute_range": [0, 256],
            "training_absolute_range": [256, 2304],
            "requested_training_tokens": 2048,
        }
    )
    monkeypatch.setattr(runner, "validate_config", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "_require_contract", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        runner,
        "_resource_guard",
        lambda *_args, **_kwargs: {"gpu_name": "fake A40"},
    )

    class Tokenizer:
        name_or_path = config["corpus"]["tokenizer"]
        vocab_size = 1024
        pad_token_id = 0
        eos_token_id = 0

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*_args, **_kwargs) -> Tokenizer:
            return Tokenizer()

    monkeypatch.setitem(
        sys.modules,
        "transformers",
        types.SimpleNamespace(AutoTokenizer=AutoTokenizer),
    )

    def prepare(path: Path, **kwargs) -> dict:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.zeros(int(kwargs["token_count"]), dtype=np.uint16).tofile(path)
        metadata = {
            "dataset_name": kwargs["dataset_name"],
            "dataset_config": kwargs["dataset_config"],
            "dataset_revision": kwargs["dataset_revision"],
            "split": kwargs["split"],
            "token_count": kwargs["token_count"],
            "token_offset": kwargs["token_offset"],
            "dtype": "uint16",
            "tokenizer": Tokenizer.name_or_path,
        }
        path.with_suffix(path.suffix + ".json").write_text(json.dumps(metadata))
        return metadata

    class LM:
        hidden_size = 768

        def activations(self, ids):
            return runner.torch.zeros((*ids.shape, self.hidden_size))

    class Stats:
        def normalize(self, value):
            return value

    class Fleet:
        def __init__(self, *_args, **_kwargs) -> None:
            pass

        def set_learning_rate(self, _learning_rate: float) -> None:
            pass

        def train_batch(self, *_args, **_kwargs) -> dict:
            return {}

    monkeypatch.setattr(runner, "prepare_token_memmap", prepare)
    monkeypatch.setattr(runner, "load_lm", lambda *_args, **_kwargs: LM())
    monkeypatch.setattr(runner, "estimate_activation_stats", lambda _value: Stats())
    monkeypatch.setattr(runner, "calibrate_ridge", lambda *_args, **_kwargs: 1.0)
    monkeypatch.setattr(runner, "TrainingFleet", Fleet)
    monkeypatch.setattr(runner.torch.cuda, "max_memory_reserved", lambda *_args: 0)

    gate = runner.run_timing_smoke(
        config=config,
        output_root=tmp_path,
        token_cache=tmp_path / "full-cache.bin",
        device=runner.torch.device("cpu"),
        local_files_only=True,
    )

    smoke_root = tmp_path / config["timing_smoke"]["artifact_subdirectory"]
    assert gate["passed"] is True
    assert gate["model_quality_metrics_retained"] is False
    assert not list(smoke_root.glob("*.pt"))
    assert not list(smoke_root.glob("*checkpoint*"))
    assert set(path.name for path in smoke_root.iterdir()) == {
        "cache_timing.json",
        "timing_smoke_gate.json",
        "tokens.bin",
        "tokens.bin.json",
    }


def test_wait_for_files_fails_immediately_on_retained_failure(tmp_path: Path) -> None:
    failure = tmp_path / "pair_failed.json"
    failure.write_text(json.dumps({"error": "synthetic worker failure"}))

    with pytest.raises(runner.RunAbortedError, match="synthetic worker failure"):
        runner.wait_for_files(
            [tmp_path / "never_created.json"],
            timeout_seconds=60,
            poll_seconds=0.01,
            failure_paths=[failure],
        )


def test_abort_and_deadline_checks_are_fail_closed(tmp_path: Path) -> None:
    config = frozen_config()
    runner.write_stage_status(
        config=config,
        output_root=tmp_path,
        stage="coordinator",
        state="running",
        extra={
            "started_at_unix_seconds": runner.time.time() - 2,
            "deadline_unix_seconds": runner.time.time() - 1,
        },
    )
    with pytest.raises(TimeoutError, match="wall-time"):
        runner._check_abort_or_deadline(config, tmp_path)

    second = tmp_path / "abort"
    runner.request_abort(config=config, output_root=second, reason="paired worker failed")
    with pytest.raises(runner.RunAbortedError, match="paired worker failed"):
        runner._check_abort_or_deadline(config, second)


def test_snapshot_partial_is_rebuilt_then_atomically_promoted(tmp_path: Path) -> None:
    config = frozen_config()
    output = tmp_path / "run"
    pair_root = output / "pairs" / "seed_11"
    (output / "stop_rule_contract.json").parent.mkdir(parents=True)
    (output / "stop_rule_contract.json").write_text("{}\n")
    snapshot = runner._snapshot_dir(pair_root, 25_000_000)
    partial = snapshot.with_name(f".{snapshot.name}.partial")
    partial.mkdir(parents=True)
    (partial / "interrupted").write_text("stale")

    class Fleet:
        def state_dict(self, *, step: int, tokens_seen: int) -> dict:
            return {"step": step, "tokens_seen": tokens_seen, "specs": []}

        def export_models(self) -> dict:
            return {"mse": {"weight": [1.0]}, "dpsae": {"weight": [2.0]}}

    class Stream:
        def state_dict(self) -> dict:
            return {"total_sequences": 1, "stream_contract": {"version": 1}}

    manifest = runner._save_snapshot(
        config=config,
        output_root=output,
        pair_root=pair_root,
        pair_seed=11,
        requested_tokens=25_000_000,
        step=1,
        tokens_seen=25_000_128,
        learning_rate=1e-4,
        fleet=Fleet(),
        stream=Stream(),
        exposure={"unique_corpus_exposure_tokens": 25_000_128},
        maturity={"models": {}},
        calibration_sha256="abc",
        cumulative_wall_seconds=1.0,
    )

    assert snapshot.is_dir()
    assert not partial.exists()
    for record in manifest["artifacts"].values():
        assert Path(record["path"]).parent == snapshot
        assert runner.file_record(Path(record["path"])) == record
    assert runner._save_snapshot(
        config=config,
        output_root=output,
        pair_root=pair_root,
        pair_seed=11,
        requested_tokens=25_000_000,
        step=1,
        tokens_seen=25_000_128,
        learning_rate=1e-4,
        fleet=Fleet(),
        stream=Stream(),
        exposure={"unique_corpus_exposure_tokens": 25_000_128},
        maturity={"models": {}},
        calibration_sha256="abc",
        cumulative_wall_seconds=1.0,
    ) == manifest


def test_coordinator_failure_requests_fleet_abort(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = frozen_config()
    failure = tmp_path / "pairs" / "seed_11" / "pair_failed.json"
    failure.parent.mkdir(parents=True)
    failure.write_text(json.dumps({"error": "synthetic pair crash"}))
    monkeypatch.setattr(runner, "_resource_guard", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "prepare_shared_inputs", lambda **_kwargs: {})
    monkeypatch.setattr(runner, "_require_timing_smoke_gate", lambda *_args: {})

    with pytest.raises(runner.RunAbortedError, match="synthetic pair crash"):
        runner.run_coordinator(
            config=config,
            output_root=tmp_path,
            token_cache=tmp_path / "tokens.bin",
            device=runner.torch.device("cpu"),
            local_files_only=True,
            timeout_seconds=1,
        )

    assert json.loads((tmp_path / "abort_requested.json").read_text())["abort_requested"]
    assert json.loads((tmp_path / "coordinator_failed.json").read_text())["failed"]


def test_concept_authorization_rechecks_identity_and_freezes_inference(tmp_path: Path) -> None:
    config = frozen_config()
    digest = runner.canonical_digest(config)
    selected = 250_000_000
    tmp_path.mkdir(exist_ok=True)
    (tmp_path / "stop_rule_contract.json").write_text("{}\n")
    smoke_root = tmp_path / config["timing_smoke"]["artifact_subdirectory"]
    smoke_root.mkdir()
    smoke_artifacts = {}
    for name in ("token_cache", "token_cache_metadata", "cache_timing"):
        artifact = smoke_root / name
        artifact.write_text(name)
        smoke_artifacts[name] = runner.file_record(artifact)
    (smoke_root / "timing_smoke_gate.json").write_text(
        json.dumps(
            {
                "complete": True,
                "passed": True,
                "reportable": False,
                "config_digest": digest,
                "pair_seed": config["timing_smoke"]["pair_seed"],
                "artifacts": smoke_artifacts,
            }
        )
    )
    (tmp_path / "maturity_stop_decision.json").write_text(
        json.dumps(
            {
                "complete": True,
                "concept_blind": True,
                "config_digest": digest,
                "decision_written_before_concept_evaluation": True,
                "common_checkpoint_selected": True,
                "selected_requested_snapshot_tokens": selected,
            }
        )
    )
    prerequisite = {
        "complete": True,
        "config_digest": digest,
        "selected_requested_snapshot_tokens": selected,
        "concept_outcomes_opened_by_this_stage": False,
    }
    (tmp_path / "selected_decoder_distortion.json").write_text(json.dumps(prerequisite))
    frozen = {**prerequisite, "identity_hook": {"passed": False}}
    (tmp_path / "selected_frozen_network.json").write_text(json.dumps(frozen))

    with pytest.raises(RuntimeError, match="identity hook"):
        runner.authorize_concept_evaluation(config, tmp_path)

    frozen["identity_hook"]["passed"] = True
    (tmp_path / "selected_frozen_network.json").write_text(json.dumps(frozen))
    authorization = runner.authorize_concept_evaluation(config, tmp_path)
    assert authorization["confirmatory_inference"] == config["concept_authorization"][
        "confirmatory_inference"
    ]


def test_current_contract_rejects_a_post_hoc_500m_extension() -> None:
    config = frozen_config()
    config["training"]["maximum_tokens"] = 500_000_000
    config["training"]["scheduler_horizon_tokens"] = 500_000_000
    config["corpus"]["cache_absolute_range"] = [50_000_128, 552_000_128]
    config["corpus"]["training_absolute_range"] = [51_048_704, 552_000_128]
    config["maturity_stop_rule"]["extension_to_500m_policy"] = "always_if_fresh_capacity"

    with pytest.raises(ValueError, match="frozen exp12 setting changed"):
        runner.validate_config(config)


def test_launcher_uses_three_pair_gpus_and_a_dedicated_coordinator() -> None:
    launcher = (runner.ROOT / "scripts/run_exp12_pythia_confirmation_4xa40.sh").read_text()

    assert "preflight" in launcher
    assert 'EXP12_USER_APPROVED:-}' in launcher
    assert "nvidia-smi --query-gpu=name" in launcher
    assert 'PAIR_GPUS=(0 1 2)' in launcher
    assert 'COORDINATOR_GPU=3' in launcher
    assert "wait-shared" in launcher
    assert "wait-shared --pair-seed '$seed'" in launcher
    assert "train-pair" in launcher
    assert "coordinator" in launcher
    assert "timing-smoke" in launcher
    assert launcher.index("'$RUNNER' timing-smoke") < launcher.index("'$RUNNER' coordinator")
    assert 'TIMEOUT_HOURS="${TIMEOUT_HOURS:-8}"' in launcher
    assert "timeout --signal=TERM --kill-after=60" in launcher
    pair_command = next(
        line for line in launcher.splitlines() if "wait-shared --pair-seed" in line
    )
    assert pair_command.count("timeout --signal=TERM --kill-after=60") == 1
    assert "concept_evaluation_authorization.json" not in launcher
