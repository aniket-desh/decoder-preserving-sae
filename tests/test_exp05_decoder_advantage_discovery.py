import json
from pathlib import Path

import pytest
import torch

from experiments import exp05_decoder_advantage_discovery as runner


def natural_cache(tmp_path: Path) -> tuple[Path, dict]:
    sequence_length = 256
    discovery_starts = torch.arange(80) * 1_000 + 180_010_000
    recurrence_starts = torch.arange(16) * 1_000 + 182_600_000
    starts = torch.cat([discovery_starts, recurrence_starts])
    input_ids = torch.arange(len(starts) * sequence_length).reshape(
        len(starts), sequence_length
    ) % 997
    activations = torch.arange(
        len(starts) * sequence_length * 2, dtype=torch.float32
    ).reshape(len(starts), sequence_length, 2).half()
    payload = {
        "split": "selection",
        "input_ids": input_ids,
        "activations": activations,
        "starts": starts,
        "eos_token_id": 996,
    }
    path = tmp_path / "natural_selection.pt"
    torch.save(payload, path)
    return path, payload


def fake_search(protocol: runner.SearchProtocol = runner.DEFAULT_PROTOCOL) -> dict:
    modes = []
    controls = []
    for seed in protocol.seeds:
        for group_slot in range(protocol.discovery_groups):
            controls.append(
                {
                    "control_id": f"s{seed}_g{group_slot:02d}",
                    "seed": seed,
                    "group_slot": group_slot,
                    "row_shuffle": {},
                    "random_directions": {},
                    "observed_operator": {},
                }
            )
            for side in ("top", "bottom"):
                for rank in range(1, protocol.extreme_modes_per_side + 1):
                    modes.append(
                        {
                            "mode_id": f"s{seed}_g{group_slot:02d}_{side}{rank}",
                            "seed": seed,
                            "group_slot": group_slot,
                            "group_position": group_slot,
                            "side": side,
                            "rank": rank,
                            "eigenvalue": 0.0,
                            "eigentask": [0.0] * protocol.group_size,
                            "controls": {},
                        }
                    )
    return {
        "schema_version": runner.SCHEMA_VERSION,
        "protocol_digest": runner.canonical_digest(runner.asdict(protocol)),
        "modes": modes,
        "controls": controls,
    }


def test_default_protocol_caps_exactly_192_modes_and_48_controls():
    assert runner.EXPECTED_MODE_COUNT == 192
    assert runner.EXPECTED_CONTROL_COUNT == 48
    runner.validate_search_log(fake_search())

    malformed = fake_search()
    malformed["modes"].pop()
    with pytest.raises(RuntimeError, match="exactly 192"):
        runner.validate_search_log(malformed)


def test_discovery_manifest_preselects_exact_tokens_and_fixed_groups(tmp_path):
    cache_path, cache = natural_cache(tmp_path)
    first = runner.build_discovery_manifest(cache, cache_path=cache_path)
    second = runner.build_discovery_manifest(cache, cache_path=cache_path)

    assert first["selected_sequence_rows"] == second["selected_sequence_rows"]
    assert first["group_positions"] == second["group_positions"]
    assert first["group_indices"] == second["group_indices"]
    assert len(first["selected_sequence_rows"]) * first["sequence_length"] == 16_384
    assert torch.tensor(first["group_indices"]).shape == (16, 128)
    assert len(set(first["group_positions"])) == 16
    assert all(
        180_000_000 <= start < 182_500_000
        for start in first["selected_sequence_starts"]
    )
    runner.validate_discovery_manifest(first)


def test_prepare_reconstructs_six_paired_models_in_fixed_sequence(tmp_path, monkeypatch):
    cache_path, _cache = natural_cache(tmp_path)
    models_path = tmp_path / "models.pt"
    payloads = {
        f"{method}_s{seed}": {
            "spec": {"method": method, "seed": seed, "k": 32},
            "state_dict": {},
        }
        for seed in (0, 1, 2)
        for method in ("mse", "dpsae")
    }
    torch.save(payloads, models_path)
    calls = []

    def fake_reconstruct(payload, activations, **_kwargs):
        calls.append((payload["spec"]["seed"], payload["spec"]["method"], activations.shape))
        return torch.zeros_like(activations)

    monkeypatch.setattr(runner, "reconstruct_one", fake_reconstruct)
    output = tmp_path / "output"
    runner.prepare_reconstructions(
        natural_selection=cache_path,
        models_path=models_path,
        output=output,
        device=torch.device("cpu"),
    )

    assert [(seed, method) for seed, method, _shape in calls] == [
        (seed, method) for seed in (0, 1, 2) for method in ("mse", "dpsae")
    ]
    assert all(shape[0] * shape[1] == 16_384 for _seed, _method, shape in calls)
    assert len(list((output / "reconstructions").glob("*.pt"))) == 6


def test_extreme_modes_log_top_bottom_tasks_and_both_controls():
    protocol = runner.SearchProtocol(
        source_selection_range=(0, 100),
        discovery_range=(0, 40),
        recurrence_range=(40, 100),
        sealed_final_range=(200, 300),
        exact_tokens=12,
        group_size=6,
        discovery_groups=2,
        extreme_modes_per_side=2,
        random_directions_per_group=8,
        seeds=(0,),
    )
    q = torch.diag(torch.tensor([-3.0, -2.0, -1.0, 1.0, 2.0, 3.0]))
    rows, control = runner.extreme_modes_for_group(
        q,
        0.5 * q,
        seed=0,
        group_slot=0,
        group_position=7,
        flat_token_indices=torch.arange(6),
        selected_sequence_starts=torch.tensor([10]),
        sequence_length=6,
        protocol=protocol,
    )

    assert [(row["side"], row["rank"]) for row in rows] == [
        ("top", 1),
        ("top", 2),
        ("bottom", 1),
        ("bottom", 2),
    ]
    assert [row["eigenvalue"] for row in rows] == pytest.approx([3, 2, -3, -2])
    assert all(len(row["eigentask"]) == 6 for row in rows)
    assert all("row_shuffle_rayleigh" in row["controls"] for row in rows)
    assert all("random_rayleigh_percentile" in row["controls"] for row in rows)
    assert control["random_directions"]["count"] == 8
    assert len(control["random_directions"]["rayleigh_values"]) == 8


def test_reconstruct_one_uses_threshold_and_preserves_exact_shape(monkeypatch):
    calls = []

    class FakeModel:
        def __call__(self, x, *, use_threshold):
            calls.append((len(x), use_threshold))
            return x + 1, torch.zeros(len(x), 1)

    monkeypatch.setattr(runner, "load_sae", lambda *_args, **_kwargs: FakeModel())
    activations = torch.zeros(2, 4, 3)
    reconstruction = runner.reconstruct_one(
        {"spec": {}, "state_dict": {}},
        activations,
        device=torch.device("cpu"),
        batch_tokens=3,
    )

    assert reconstruction.shape == activations.shape
    assert reconstruction.dtype == torch.float16
    assert torch.equal(reconstruction, torch.ones_like(reconstruction))
    assert all(use_threshold is True for _count, use_threshold in calls)
    assert sum(count for count, _use_threshold in calls) == activations.numel() // 3


def test_compute_search_log_streams_prepared_pairs_into_exact_mode_count(tmp_path):
    protocol = runner.SearchProtocol(
        source_selection_range=(0, 100),
        discovery_range=(0, 40),
        recurrence_range=(40, 100),
        sealed_final_range=(200, 300),
        exact_tokens=8,
        group_size=4,
        discovery_groups=2,
        extreme_modes_per_side=1,
        random_directions_per_group=4,
        seeds=(0,),
        k=2,
    )
    cache = {
        "split": "selection",
        "input_ids": torch.arange(8).reshape(4, 2),
        "activations": torch.randn(4, 2, 3, generator=torch.Generator().manual_seed(4)),
        "starts": torch.tensor([0, 2, 4, 6]),
        "eos_token_id": 99,
    }
    cache_path = tmp_path / "selection.pt"
    torch.save(cache, cache_path)
    output = tmp_path / "output"
    manifest = runner.build_discovery_manifest(
        cache, cache_path=cache_path, protocol=protocol
    )
    runner.atomic_json(output / "discovery_manifest.json", manifest)
    manifest_digest = runner.canonical_digest(manifest)
    selected = cache["activations"][manifest["selected_sequence_rows"]]
    for name, scale in (("mse_s0", 0.2), ("dpsae_s0", 0.1)):
        reconstruction = selected + scale * torch.randn(
            selected.shape, generator=torch.Generator().manual_seed(8)
        )
        runner.atomic_torch(
            output / "reconstructions" / f"{name}.pt",
            {
                "metadata": {
                    "model_name": name,
                    "manifest_digest": manifest_digest,
                },
                "reconstruction": reconstruction.half(),
            },
        )
    calibration = tmp_path / "static.pt"
    torch.save({"ridge": 0.2}, calibration)

    search = runner.compute_search_log(
        natural_selection=cache_path,
        static_calibration=calibration,
        output=output,
        protocol=protocol,
    )

    assert len(search["modes"]) == 4
    assert len(search["controls"]) == 2
    assert {row["side"] for row in search["modes"]} == {"top", "bottom"}
    runner.validate_search_log(search, protocol)


def test_registry_cannot_freeze_until_every_mode_is_dispositioned(tmp_path):
    search_path = tmp_path / "searched_modes.json"
    search_path.write_text(json.dumps(fake_search()))
    registry_path = tmp_path / "hypothesis_registry.json"
    registry = runner.initialize_hypothesis_registry(
        search_log_path=search_path,
        registry_path=registry_path,
    )
    assert registry["status"] == "open"
    assert len(registry["mode_dispositions"]) == 192

    with pytest.raises(ValueError, match="reviewed"):
        runner.freeze_hypothesis_registry(registry_path=registry_path)

    for disposition in registry["mode_dispositions"].values():
        disposition["status"] = "rejected"
    registry_path.write_text(json.dumps(registry))
    frozen = runner.freeze_hypothesis_registry(registry_path=registry_path)

    assert frozen["status"] == "frozen"
    runner.validate_frozen_registry(frozen, registry_path)
    runner.guard_range_access(
        runner.DEFAULT_PROTOCOL.sealed_final_range,
        registry_path=registry_path,
    )


def test_sealed_cache_is_refused_before_torch_load(tmp_path, monkeypatch):
    search_path = tmp_path / "searched_modes.json"
    search_path.write_text(json.dumps(fake_search()))
    registry_path = tmp_path / "hypothesis_registry.json"
    runner.initialize_hypothesis_registry(
        search_log_path=search_path,
        registry_path=registry_path,
    )
    called = False

    def forbidden_load(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("torch.load must not run before the seal guard")

    monkeypatch.setattr(runner.torch, "load", forbidden_load)
    with pytest.raises(PermissionError, match="sealed"):
        runner.guarded_load_natural_cache(
            tmp_path / "final.pt",
            requested_range=runner.DEFAULT_PROTOCOL.sealed_final_range,
            registry_path=registry_path,
        )
    assert called is False


def test_frozen_registry_detects_search_log_mutation(tmp_path):
    search_path = tmp_path / "searched_modes.json"
    search_path.write_text(json.dumps(fake_search()))
    registry_path = tmp_path / "hypothesis_registry.json"
    registry = runner.initialize_hypothesis_registry(
        search_log_path=search_path,
        registry_path=registry_path,
    )
    for disposition in registry["mode_dispositions"].values():
        disposition["status"] = "rejected"
    registry_path.write_text(json.dumps(registry))
    runner.freeze_hypothesis_registry(registry_path=registry_path)

    search_path.write_text(search_path.read_text() + "\n")
    with pytest.raises(PermissionError, match="changed"):
        runner.guard_range_access(
            runner.DEFAULT_PROTOCOL.sealed_final_range,
            registry_path=registry_path,
        )
