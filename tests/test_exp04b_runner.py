import json
from pathlib import Path

import pytest
import torch

from dpsae.exp04b_training import stage_seeds
from experiments import exp04b_confirmatory as runner


ROOT = Path(__file__).resolve().parents[1]


def test_repository_state_uses_shared_provenance_schema():
    state = runner.repository_state(ROOT)

    assert set(state) == {"revision", "dirty", "status"}
    assert state["dirty"] == bool(state["status"])
    assert isinstance(state["status"], list)


def test_repository_state_fails_closed_when_git_is_unavailable(monkeypatch):
    def fail(*_args, **_kwargs):
        raise OSError("git unavailable")

    monkeypatch.setattr(runner.subprocess, "check_output", fail)

    state = runner.repository_state(ROOT)

    assert state["revision"] == "unknown"
    assert state["dirty"] is True
    assert state["status"] == ["repository state unavailable"]


def test_driver_imports_config_and_uses_stable_cache_names():
    config = runner.load_config(ROOT / "configs" / "exp04b_confirmatory.json")
    paths = runner.experiment_paths(config)

    assert config["source"]["experiment"] == "exp04_ioi_mechanism"
    assert paths.natural_selection.name == "natural_selection.pt"
    assert paths.natural_test.name == "natural_test.pt"
    assert paths.tail_tokens.parent.name == "exp04b_confirmatory"


def test_stage_sequence_runs_both_fleets_but_leaves_ioi_explicit():
    assert runner.stage_sequence("natural-evaluate", fleet="source") == [
        ("natural-evaluate", "source")
    ]
    stages = runner.stage_sequence("all")
    assert ("natural-evaluate", "source") in stages
    assert ("natural-evaluate", "baseline") in stages
    assert all(stage != "ioi-confirm" for stage, _ in stages)
    with pytest.raises(NotImplementedError, match="intentionally separate"):
        runner.ioi_confirm_hook()


def test_confirmation_uses_changed_data_and_probe_streams():
    config = runner.load_config(ROOT / "configs" / "exp04b_confirmatory.json")
    changed = runner._stage_randomness(config, "baseline-confirm")
    original = stage_seeds(config["source"]["seed"], "confirmation", replicate=0)

    assert changed.replicate == config["baseline"]["confirmation_replicate"]
    assert changed.data_order != original.data_order
    assert changed.probe_sequence != original.probe_sequence


def test_one_factor_settings_change_at_most_one_geometry_factor():
    settings = runner.one_factor_settings(
        base_ridge=0.2,
        base_group_size=4,
        ridges=[0.1, 0.2, 0.3],
        group_sizes=[2, 4, 8],
        groupings=["contiguous", "shuffled", "document_balanced"],
        group_ridges={2: 0.15, 4: 0.2, 8: 0.25},
    )

    assert len(settings) == 7
    assert len({row[1:] for row in settings}) == len(settings)
    for _axis, ridge, group_size, grouping in settings:
        geometry_change = group_size != 4
        expected_ridge = {2: 0.15, 4: 0.2, 8: 0.25}.get(group_size, 0.2)
        changes = sum((ridge != expected_ridge, geometry_change, grouping != "contiguous"))
        assert changes <= 1


def test_sampled_primary_report_is_exactly_paired():
    generator = torch.Generator().manual_seed(5)
    activations = torch.randn(4, 4, 3, generator=generator)
    input_ids = torch.arange(16).reshape(4, 4)
    geometry = runner.sampled_geometry(
        activations,
        input_ids,
        ridge=0.2,
        group_size=4,
        probes=3,
        seed=8,
    )
    same = runner.sampled_model_report(
        geometry,
        activations,
        activations.clone(),
        ridge=0.2,
        bootstrap_samples=64,
        seed=9,
    )
    noisy = runner.sampled_model_report(
        geometry,
        activations,
        activations + 0.2 * torch.randn(activations.shape, generator=generator),
        ridge=0.2,
        bootstrap_samples=64,
        seed=9,
    )

    assert same["nmse"] == 0
    assert same["decoder"] == 0
    assert noisy["nmse"] > 0
    assert noisy["decoder"] > 0
    assert noisy["denominator_by_group"] == same["denominator_by_group"]


def test_natural_cache_contains_ids_normalized_activations_and_absolute_starts(
    tmp_path, monkeypatch
):
    output = tmp_path / "out"
    paths = runner.ExperimentPaths(
        output=output,
        tail_tokens=output / "tail.bin",
        natural_selection=output / "natural_selection.pt",
        natural_test=output / "natural_test.pt",
        static_calibration=output / "static.pt",
        baseline_selection=output / "selection.json",
        source_artifact=tmp_path / "source",
        source_tokens=tmp_path / "source" / "tokens.bin",
        source_calibration=tmp_path / "source" / "calibration.pt",
    )
    paths.tail_tokens.parent.mkdir(parents=True)
    paths.tail_tokens.touch()
    paths.source_calibration.parent.mkdir(parents=True)
    torch.save(
        {"activation_stats": {"mean": torch.zeros(2), "scale": torch.tensor(1.0)}},
        paths.source_calibration,
    )
    config = {
        "repository": {"revision": "test", "dirty": False, "status": []},
        "source": {"training": {"sequence_length": 4}},
        "fresh_corpus": {
            "token_offset": 100,
            "selection_range": [0, 8],
            "test_range": [8, 16],
        },
        "natural_text": {"activation_tokens": 8},
    }

    class FakeTokenizer:
        eos_token_id = 9

    class FakeLM:
        tokenizer = FakeTokenizer()

        @staticmethod
        def activations(ids):
            return ids.float().unsqueeze(-1).expand(*ids.shape, 2)

    class FakeBatcher:
        @staticmethod
        def batch_with_starts():
            return torch.arange(8).reshape(2, 4), torch.tensor([0, 4])

    monkeypatch.setattr(runner, "load_lm", lambda *_args: FakeLM())
    monkeypatch.setattr(runner, "_tail_batcher", lambda *_args: FakeBatcher())
    runner.cache_natural(config, paths, torch.device("cpu"))

    for split, path in (
        ("selection", paths.natural_selection),
        ("test", paths.natural_test),
    ):
        payload = torch.load(path, weights_only=False)
        assert payload["split"] == split
        assert set(("input_ids", "activations", "starts")) <= payload.keys()
        assert payload["input_ids"].shape == (2, 4)
        assert payload["activations"].shape == (2, 4, 2)
        assert torch.equal(payload["starts"], torch.tensor([100, 104]))
        assert payload["repository"] == config["repository"]
        assert payload["normalized_with_sha256"] == runner.sha256_file(
            paths.source_calibration
        )


def test_checkpoint_specs_are_validated_before_resume():
    spec = runner.SAETrainSpec("mse_s0", "mse", 0, 4)
    runner._checkpoint_matches({"specs": [runner.asdict(spec)]}, [spec])
    with pytest.raises(RuntimeError, match="checkpoint specs"):
        runner._checkpoint_matches({"specs": []}, [spec])


def test_source_fleet_merges_all_confirmed_sparsities(tmp_path):
    source = tmp_path / "source"
    for stage, name, k in (
        ("confirmation", "mse_s0", 32),
        ("robustness16", "mse_k16_s0", 16),
        ("robustness64", "mse_k64_s0", 64),
    ):
        (source / stage).mkdir(parents=True)
        torch.save(
            {name: {"spec": {"method": "mse", "seed": 0, "k": k}}},
            source / stage / "models.pt",
        )
    paths = runner.ExperimentPaths(
        output=tmp_path / "output",
        tail_tokens=tmp_path / "tail.bin",
        natural_selection=tmp_path / "selection.pt",
        natural_test=tmp_path / "test.pt",
        static_calibration=tmp_path / "static.pt",
        baseline_selection=tmp_path / "baseline.json",
        source_artifact=source,
        source_tokens=source / "tokens.bin",
        source_calibration=source / "calibration.pt",
    )

    assert set(runner._fleet_payloads(paths, "source")) == {
        "mse_s0",
        "mse_k16_s0",
        "mse_k64_s0",
    }


def test_trim_log_removes_uncheckpointed_records(tmp_path):
    path = tmp_path / "training.jsonl"
    path.write_text("".join(json.dumps({"step": step}) + "\n" for step in (1, 2, 3)))
    runner._trim_log(path, 2)
    assert [json.loads(line)["step"] for line in path.read_text().splitlines()] == [1, 2]
