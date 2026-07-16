import sys
from pathlib import Path
import hashlib
import json

import pytest
import torch

from experiments.exp09_frozen_network import (
    DEFAULT_CONFIG,
    all_seed_noninferiority,
    bootstrap_ioi_pair,
    bootstrap_natural_pair,
    build_ioi_prompt_payload,
    deterministic_nonoverlapping_starts,
    identity_gate,
    load_config,
    main,
    natural_retention_rows,
    pooled_kl_ratio,
    validate_checkpoint_provenance,
    validate_natural_inputs,
)


class _SingleTokenNameTokenizer:
    def encode(self, text: str, *, add_special_tokens: bool = False) -> list[int]:
        assert not add_special_tokens
        return [sum(ord(character) for character in text)]


def _natural_rows(count: int = 4):
    common = [
        {
            "sequence": float(index),
            "tokens": 10.0,
            "original_nll": 20.0,
            "mean_nll": 40.0,
            "activation_energy": 100.0,
            "activation_tokens": 10.0,
        }
        for index in range(count)
    ]
    mse = [
        {
            "sequence": float(index),
            "reconstructed_nll": 30.0,
            "kl": 2.0,
            "agreement": 8.0,
            "reconstructed_correct": 4.0,
            "reconstruction_sse": 10.0,
            "l0_count": 320.0,
        }
        for index in range(count)
    ]
    dpsae = [
        {
            "sequence": float(index),
            "reconstructed_nll": 26.0,
            "kl": 1.0,
            "agreement": 9.0,
            "reconstructed_correct": 5.0,
            "reconstruction_sse": 8.0,
            "l0_count": 310.0,
        }
        for index in range(count)
    ]
    return common, mse, dpsae


def test_frozen_config_keeps_the_confirmatory_contract() -> None:
    config = load_config(DEFAULT_CONFIG)

    assert config["natural_text"]["absolute_range"] == [200_000_000, 210_000_000]
    assert config["natural_text"]["sequences"] == 2_048
    assert config["natural_text"]["confidence_interval"] == [0.025, 0.975]
    assert config["natural_text"]["noninferiority_margin"] == 1.01
    assert config["checkpoints"]["expected_seeds"] == [0, 1, 2]
    assert config["runpod"] == {
        "gpu_model": "NVIDIA A40 48GB",
        "gpu_count": 4,
        "gpu_hourly_usd": 0.45,
        "allocation_hourly_usd": 1.80,
        "network_volume_gib": 200,
    }


def test_nonoverlapping_starts_are_deterministic_unique_and_in_range() -> None:
    first = deterministic_nonoverlapping_starts(
        start=200_000_000,
        stop=210_000_000,
        sequence_length=256,
        count=2_048,
        seed=17,
    )
    second = deterministic_nonoverlapping_starts(
        start=200_000_000,
        stop=210_000_000,
        sequence_length=256,
        count=2_048,
        seed=17,
    )

    assert torch.equal(first, second)
    assert len(torch.unique(first)) == 2_048
    ordered = first.sort().values
    assert int(ordered.min()) >= 200_000_000
    assert int(ordered.max()) + 256 <= 210_000_000
    assert bool(((ordered[1:] - ordered[:-1]) >= 256).all())


def test_natural_input_validation_rejects_overlap_and_duplicates() -> None:
    payload = {
        "absolute_range": [100, 1_000],
        "input_ids": torch.zeros(3, 10, dtype=torch.long),
        "starts": torch.tensor([100, 105, 200]),
    }
    with pytest.raises(ValueError, match="overlapping"):
        validate_natural_inputs(payload, require_nonoverlap=True)

    payload["starts"] = torch.tensor([100, 100, 200])
    with pytest.raises(ValueError, match="duplicate"):
        validate_natural_inputs(payload, require_nonoverlap=True)


def test_pooled_ratio_and_bootstrap_use_paired_pooled_token_kl() -> None:
    common, mse, dpsae = _natural_rows()

    assert pooled_kl_ratio(mse, dpsae) == pytest.approx(0.5)
    result = bootstrap_natural_pair(
        common,
        mse,
        dpsae,
        samples=100,
        seed=3,
        quantiles=[0.025, 0.975],
        chunk_size=17,
    )

    assert result["kl_ratio_dpsae_to_mse_ci95"] == pytest.approx([0.5, 0.5])
    assert result["kl_difference_dpsae_minus_mse_ci95"] == pytest.approx([-0.1, -0.1])
    assert result["loss_recovered_difference_dpsae_minus_mse_ci95"] == pytest.approx(
        [0.2, 0.2]
    )
    assert result["activation_nmse_ratio_dpsae_to_mse_ci95"] == pytest.approx(
        [0.8, 0.8]
    )
    assert result["valid_kl_ratio_draw_fraction"] == 1


def test_pooled_ratio_rejects_nonpositive_mse_denominator() -> None:
    with pytest.raises(ValueError, match="denominator"):
        pooled_kl_ratio([{"kl": 0.0}], [{"kl": 1.0}])


def test_panel_f_retention_rows_are_complete_for_every_paired_condition() -> None:
    common, mse, dpsae = _natural_rows(count=1)
    common[0].update(
        absolute_start=200_000_000,
        sequence_sha256="abc",
        original_kl=0.0,
        original_agreement=10,
        original_correct=4,
        identity_kl=0.0,
        identity_nll=20.0,
        identity_agreement=10,
        identity_correct=4,
        identity_reconstruction_sse=0.0,
        identity_max_abs_logit_difference=0.0,
        identity_mean_abs_logit_difference=0.0,
        mean_kl=3.0,
        mean_agreement=2,
        mean_correct=1,
        mean_reconstruction_sse=100.0,
    )
    payloads = {
        "mse_seed0": {"spec": {"seed": 0, "method": "mse"}},
        "dpsae_seed0": {"spec": {"seed": 0, "method": "dpsae"}},
    }

    rows = natural_retention_rows(
        common,
        {"mse_seed0": mse, "dpsae_seed0": dpsae},
        payloads,
        bootstrap_seed=100,
    )

    assert [row["condition"] for row in rows] == [
        "original",
        "identity",
        "mean_ablation",
        "mse",
        "dpsae",
    ]
    assert all(row["bootstrap_seed"] == 100 for row in rows)
    assert all(row["valid_token_count"] == 10 for row in rows)
    assert rows[-1]["l0_sum"] == 310.0
    assert rows[-1]["l0_count"] == 10


def test_explicit_checkpoint_bundle_is_bound_to_training_provenance(tmp_path: Path) -> None:
    config = load_config(DEFAULT_CONFIG)
    models = tmp_path / "models.pt"
    done = tmp_path / "done.json"
    summary = tmp_path / "summary.json"
    models.write_bytes(b"frozen-model-bundle")
    specs = [
        {
            "seed": seed,
            "method": method,
            "decoder_weight": 0.0 if method == "mse" else 0.03125,
            "k": 32,
        }
        for seed in (0, 1, 2)
        for method in ("mse", "dpsae")
    ]
    done.write_text(
        json.dumps(
            {
                "complete": True,
                "repository": {"revision": config["checkpoint_provenance"]["training_revision"]},
                "stream": {"range": [50_000_000, 120_000_000]},
                "specs": specs,
            }
        )
    )
    model_hash = hashlib.sha256(models.read_bytes()).hexdigest()
    summary.write_text(
        json.dumps(
            {
                "complete": True,
                "gate_passed": True,
                "repository": {"revision": config["checkpoint_provenance"]["training_revision"]},
                "expected_seeds": [0, 1, 2],
                "selected_decoder_weight": 0.03125,
                "inputs": {
                    "models": {"bytes": models.stat().st_size, "sha256": model_hash}
                },
            }
        )
    )
    config["checkpoints"].update(bytes=models.stat().st_size, sha256=model_hash)
    for key, path in (("training_done", done), ("confirmation_summary", summary)):
        config["checkpoint_provenance"][key] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    records = validate_checkpoint_provenance(
        models=models,
        training_done=done,
        confirmation_summary=summary,
        config=config,
    )

    assert records["models"]["sha256"] == model_hash
    assert records["training_done"]["path"] == str(done.resolve())


def test_noninferiority_requires_every_frozen_seed() -> None:
    passing = [
        {"seed": seed, "kl_ratio_dpsae_to_mse_ci95": [0.98, 1.005]}
        for seed in (0, 1, 2)
    ]
    assert all_seed_noninferiority(passing, expected_seeds=[0, 1, 2], margin=1.01)

    passing[-1]["kl_ratio_dpsae_to_mse_ci95"][1] = 1.01
    assert not all_seed_noninferiority(passing, expected_seeds=[0, 1, 2], margin=1.01)
    with pytest.raises(ValueError, match="seed set"):
        all_seed_noninferiority(passing[:2], expected_seeds=[0, 1, 2], margin=1.01)


def test_identity_gate_enforces_both_frozen_tolerances() -> None:
    result = identity_gate(
        maximum=1e-6,
        total=2e-6,
        elements=100,
        max_tolerance=1e-5,
        mean_tolerance=1e-7,
    )
    assert result["passed"]

    with pytest.raises(RuntimeError, match="identity-hook"):
        identity_gate(
            maximum=1e-4,
            total=1e-4,
            elements=100,
            max_tolerance=1e-5,
            mean_tolerance=1e-7,
        )


def test_ioi_prompt_artifact_records_the_exact_generation_contract() -> None:
    config = load_config(DEFAULT_CONFIG)
    repository = {"revision": "test", "dirty": False, "status": []}

    first = build_ioi_prompt_payload(config, _SingleTokenNameTokenizer(), repository)
    second = build_ioi_prompt_payload(config, _SingleTokenNameTokenizer(), repository)

    assert first == second
    assert len(first["examples"]) == 2_048
    assert first["protocol"]["template_families"] == [11, 12, 13, 14]
    assert {row["order"] for row in first["examples"]} == {"BABA", "ABBA"}
    assert first["protocol"]["names"]


def test_ioi_bootstrap_resamples_aligned_prompts() -> None:
    mse = [
        {
            "prompt_index": float(index),
            "absolute_logit_difference_error": 2.0,
            "preferred_answer_agreement": 0.5,
            "accuracy": 0.25,
        }
        for index in range(4)
    ]
    dpsae = [
        {
            "prompt_index": float(index),
            "absolute_logit_difference_error": 1.0,
            "preferred_answer_agreement": 1.0,
            "accuracy": 0.5,
        }
        for index in range(4)
    ]

    result = bootstrap_ioi_pair(
        mse,
        dpsae,
        samples=100,
        seed=5,
        quantiles=[0.025, 0.975],
        chunk_size=13,
    )

    assert result[
        "absolute_logit_difference_error_dpsae_minus_mse_ci95"
    ] == pytest.approx([-1.0, -1.0])
    assert result["preferred_answer_agreement_dpsae_minus_mse_ci95"] == pytest.approx(
        [0.5, 0.5]
    )
    assert result["accuracy_dpsae_minus_mse_ci95"] == pytest.approx([0.25, 0.25])


def test_smoke_mode_cannot_dispatch_fresh_prepare(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(Path("experiments/exp09_frozen_network.py")),
            "prepare",
            "--smoke",
            "--allow-dirty",
        ],
    )
    with pytest.raises(ValueError, match="cannot dispatch"):
        main()


def test_runpod_launcher_explicitly_passes_resolved_environment_into_tmux() -> None:
    launcher = DEFAULT_CONFIG.parents[1] / "scripts/run_exp09_frozen_network_runpod.sh"
    source = launcher.read_text()
    start = source.index("  TMUX_ENVIRONMENT=(")
    stop = source.index("\n  )", start)
    environment_block = source[start:stop]
    required = {
        "DPSAE_EXP09_IN_TMUX",
        "DPSAE_PYTHON",
        "HF_HOME",
        "EXP09_CONFIG",
        "EXP09_OUTPUT",
        "EXP09_MODELS",
        "EXP09_CALIBRATION",
        "EXP09_TRAINING_DONE",
        "EXP09_CONFIRMATION_SUMMARY",
        "EXP09_GPU",
        "EXP09_GPU_MEMORY_FRACTION",
        "EXP09_MAXIMUM_PEAK_GPU_GIB",
        "EXP09_MINIMUM_FREE_GIB",
        "EXP09_LOCAL_FILES_ONLY",
        "EXP09_SMOKE_CACHE",
        "EXP09_SMOKE_SEQUENCES",
        "EXP09_SMOKE_IOI_EXAMPLES",
        "EXP09_OPEN_FRESH_RANGE",
    }

    assert all(f'"{name}=' in environment_block for name in required)
    assert "printf -v WORKER_COMMAND '%s %q'" in source
    assert 'tmux new-session -d -s "$SESSION" "$WORKER_COMMAND"' in source
