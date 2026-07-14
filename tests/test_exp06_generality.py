import json
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from experiments import exp06_generality as runner


def tiny_protocol() -> runner.ScreenProtocol:
    return runner.ScreenProtocol(
        token_count=48,
        calibration_range=(0, 16),
        training_range=(16, 32),
        heldout_range=(32, 48),
        calibration_tokens=8,
        evaluation_tokens=8,
        train_tokens=16,
        sequence_length=4,
        sequences_per_batch=2,
        dictionary_size=8,
        k=2,
        group_size=4,
        probes=2,
        ridge_calibration_groups=2,
        checkpoint_tokens=8,
    )


def write_cache(
    tmp_path: Path,
    target: runner.TargetSpec,
    protocol: runner.ScreenProtocol,
) -> Path:
    path = tmp_path / f"{target.key}.bin"
    path.write_bytes(b"\x00\x00" * protocol.token_count)
    metadata = {
        "dataset_name": protocol.dataset_name,
        "dataset_config": protocol.dataset_config,
        "split": protocol.split,
        "token_count": protocol.token_count,
        "dtype": "uint16",
        "tokenizer": target.model_name,
    }
    runner.corpus_metadata_path(path).write_text(json.dumps(metadata))
    return path


class FakeCausalLM(torch.nn.Module):
    def __init__(self, architecture: str, *, blocks: int, width: int = 3) -> None:
        super().__init__()
        layers = torch.nn.ModuleList(torch.nn.Identity() for _ in range(blocks))
        if architecture == "gpt2":
            self.transformer = SimpleNamespace(h=layers)
        else:
            self.gpt_neox = SimpleNamespace(layers=layers)
        self.config = SimpleNamespace(hidden_size=width, _commit_hash="revision-a")
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def forward(self, input_ids, **_kwargs):
        hidden = input_ids.float().unsqueeze(-1).expand(*input_ids.shape, 3)
        count = (
            len(self.transformer.h)
            if hasattr(self, "transformer")
            else len(self.gpt_neox.layers)
        )
        return SimpleNamespace(hidden_states=tuple(hidden + index for index in range(count + 1)))


@pytest.mark.parametrize("key", ["gpt2-block4", "pythia-block8"])
def test_fixed_targets_capture_the_preregistered_hidden_state(key):
    target = runner.TARGETS[key]
    model = FakeCausalLM(target.architecture, blocks=target.layer)
    adapter = runner.GenericActivationModel(
        model,
        SimpleNamespace(),
        target=target,
        device=torch.device("cpu"),
    )

    input_ids = torch.tensor([[1, 2]])
    activation = adapter.activations(input_ids)

    assert target.layer in {4, 8}
    assert activation.shape == (1, 2, 3)
    assert torch.equal(activation[..., 0], input_ids + target.layer)
    assert adapter.resolved_model_revision == "revision-a"
    assert all(not parameter.requires_grad for parameter in model.parameters())


def test_adapter_rejects_model_family_mismatch():
    with pytest.raises(ValueError, match="architecture"):
        runner.GenericActivationModel(
            FakeCausalLM("gpt2", blocks=8),
            SimpleNamespace(),
            target=runner.TARGETS["pythia-block8"],
            device=torch.device("cpu"),
        )


def test_protocol_ranges_are_disjoint_and_bounded():
    protocol = runner.DEFAULT_PROTOCOL
    assert protocol.calibration_range == (0, 10_000_000)
    assert protocol.training_range == (10_000_000, 40_000_000)
    assert protocol.heldout_range == (40_000_000, 50_000_000)
    with pytest.raises(ValueError, match="disjoint"):
        runner.ScreenProtocol(training_range=(9_000_000, 40_000_000))


def test_pythia_refuses_a_gpt2_token_cache(tmp_path, monkeypatch):
    protocol = tiny_protocol()
    path = write_cache(tmp_path, runner.TARGETS["gpt2-block4"], protocol)
    monkeypatch.setattr(
        runner,
        "check_disk_guard",
        lambda *_args, **_kwargs: {"free_gib": 100.0, "used_fraction": 0.1},
    )

    with pytest.raises(RuntimeError, match="requires tokenizer"):
        runner.prepare_corpus(
            target=runner.TARGETS["pythia-block8"],
            token_cache=path,
            output=tmp_path / "output",
            minimum_free_disk_gib=1.0,
            protocol=protocol,
        )


def test_cache_size_must_match_metadata(tmp_path):
    protocol = tiny_protocol()
    target = runner.TARGETS["pythia-block8"]
    path = write_cache(tmp_path, target, protocol)
    path.write_bytes(b"\x00\x00")
    metadata = json.loads(runner.corpus_metadata_path(path).read_text())

    with pytest.raises(RuntimeError, match="size disagrees"):
        runner.validate_token_cache(path, metadata, target, protocol)


def test_resolved_config_hashes_code_data_revision_and_seeds(tmp_path):
    protocol = tiny_protocol()
    target = runner.TARGETS["pythia-block8"]
    path = write_cache(tmp_path, target, protocol)

    first = runner.resolved_config(
        target=target, gamma=0.25, token_cache=path, protocol=protocol
    )
    second = runner.resolved_config(
        target=target, gamma=0.25, token_cache=path, protocol=protocol
    )
    changed = runner.resolved_config(
        target=target, gamma=0.5, token_cache=path, protocol=protocol
    )

    assert first == second
    assert first["config_digest"] != changed["config_digest"]
    assert first["repository"]["revision"]
    assert "experiments/exp06_generality.py" in first["code_sha256"]
    assert set(first["randomness"]) == {
        "stage",
        "replicate",
        "data_order",
        "probe_sequence",
    }


def test_disk_and_gpu_limits_are_hard_caps(tmp_path, monkeypatch):
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(
        runner.shutil,
        "disk_usage",
        lambda _path: usage(100 * runner.GIB, 81 * runner.GIB, 19 * runner.GIB),
    )
    with pytest.raises(RuntimeError, match="disk guard"):
        runner.check_disk_guard(tmp_path, minimum_free_gib=10)

    assert runner.gpu_memory_limit_bytes(
        96 * runner.GIB,
        maximum_reserved_gib=40,
        maximum_fraction=0.8,
    ) == 40 * runner.GIB
    assert runner.gpu_memory_limit_bytes(
        32 * runner.GIB,
        maximum_reserved_gib=40,
        maximum_fraction=0.75,
    ) == 24 * runner.GIB


def test_resume_validation_and_log_trim(tmp_path):
    specs = runner.training_specs(0.25, tiny_protocol())
    checkpoint = {
        "config_digest": "digest",
        "calibration_sha256": "calibration",
        "specs": [runner.asdict(spec) for spec in specs],
    }
    runner._checkpoint_matches(
        checkpoint,
        config_digest="digest",
        calibration_sha256="calibration",
        specs=specs,
    )
    with pytest.raises(RuntimeError, match="digest"):
        runner._checkpoint_matches(
            checkpoint,
            config_digest="other",
            calibration_sha256="calibration",
            specs=specs,
        )

    log = tmp_path / "training.jsonl"
    log.write_text("".join(json.dumps({"step": step}) + "\n" for step in (1, 2, 3)))
    runner._trim_log(log, 2)
    assert [json.loads(line)["step"] for line in log.read_text().splitlines()] == [1, 2]


def test_exact_evaluation_uses_thresholded_reconstruction(monkeypatch):
    calls = []

    class FakeSAE:
        def __call__(self, activation, *, use_threshold):
            calls.append(use_threshold)
            return 0.5 * activation, torch.ones(len(activation), 2)

    monkeypatch.setattr(runner, "load_sae", lambda *_args, **_kwargs: FakeSAE())
    activations = torch.randn(2, 4, 3, generator=torch.Generator().manual_seed(4))
    result = runner.evaluate_one_model(
        {"spec": {"method": "mse", "seed": 0, "k": 2}, "state_dict": {}},
        activations,
        ridge=0.1,
        group_size=4,
        device=torch.device("cpu"),
        batch_tokens=3,
    )

    assert result["nmse"] == pytest.approx(0.25)
    assert result["inference_l0"] == 2
    assert result["decoder_distortion"] > 0
    assert calls and all(calls)


def test_evaluate_processes_one_paired_model_at_a_time(tmp_path, monkeypatch):
    protocol = tiny_protocol()
    output = tmp_path / "output"
    output.mkdir()
    cache = {"activations": torch.randn(2, 4, 3)}
    torch.save(cache, output / "evaluation_cache.pt")
    payloads = {
        spec.name: {"spec": runner.asdict(spec), "state_dict": {}}
        for spec in runner.training_specs(0.25, protocol)
    }
    torch.save(payloads, output / "models.pt")
    torch.save({"ridge": 0.1}, output / "calibration.pt")
    monkeypatch.setattr(runner, "check_disk_guard", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(runner, "check_gpu_guard", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(runner, "prepare_evaluation_cache", lambda **_kwargs: cache)
    calls = []

    def fake_evaluate(payload, _activations, **_kwargs):
        method = payload["spec"]["method"]
        calls.append(method)
        value = 1.0 if method == "mse" else 0.8
        return {
            "spec": payload["spec"],
            "nmse": value,
            "inference_l0": 2.0,
            "decoder_distortion": value,
            "numerator_by_group": [value, value],
            "denominator_by_group": [1.0, 1.0],
        }

    monkeypatch.setattr(runner, "evaluate_one_model", fake_evaluate)
    result = runner.evaluate(
        target=runner.TARGETS["gpt2-block4"],
        config={"config_digest": "digest"},
        token_cache=tmp_path / "unused.bin",
        output=output,
        device=torch.device("cpu"),
        minimum_free_disk_gib=1,
        maximum_gpu_reserved_gib=1,
        maximum_gpu_fraction=0.5,
        protocol=protocol,
    )

    assert calls == ["mse", "dpsae"]
    assert result["paired_reduction"]["estimate"] == pytest.approx(0.2)
    assert result["complete"] is True
