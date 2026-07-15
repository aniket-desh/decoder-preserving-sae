from scripts.summarize_exp08_confirmation import summarize_confirmation


def _evaluation(seed: int, reduction: float, nmse_ratio: float, repository: dict) -> dict:
    return {
        "complete": True,
        "repository": repository,
        "protocol": {"evaluation_seed": seed},
        "models": {
            f"mse_s{seed}": {
                "method": "mse",
                "seed": seed,
                "nmse": 1.0,
                "exact_decoder_distortion": 1.0,
            },
            f"dpsae_w0.03125_s{seed}": {
                "method": "dpsae",
                "seed": seed,
                "nmse": nmse_ratio,
                "exact_decoder_distortion": 1 - reduction,
            },
        },
        "paired_frontier": [
            {
                "baseline": f"mse_s{seed}",
                "candidate": f"dpsae_w0.03125_s{seed}",
                "decoder_weight": 0.03125,
                "nmse_ratio_to_mse": nmse_ratio,
                "nmse_change_percent": 100 * (nmse_ratio - 1),
                "exact_decoder_reduction": reduction,
                "exact_decoder_reduction_ci95": [reduction - 0.01, reduction + 0.01],
            }
        ],
    }


def _config() -> dict:
    return {
        "frontier": {
            "confirmation_seeds": [0, 1, 2],
            "confirmation_gate": {
                "maximum_nmse_ratio_every_seed": 1.01,
                "minimum_median_exact_decoder_reduction": 0.1,
                "require_positive_reduction_every_seed": True,
                "require_ci_excludes_zero_every_seed": True,
            },
        }
    }


def test_confirmation_summary_enforces_every_frozen_gate() -> None:
    repository = {"revision": "abc", "dirty": False, "status": []}
    evaluations = [
        _evaluation(0, 0.11, 1.0, repository),
        _evaluation(1, 0.12, 1.005, repository),
        _evaluation(2, 0.10, 0.999, repository),
    ]

    summary = summarize_confirmation(
        evaluations, _config(), selected_weight=0.03125, repository=repository
    )

    assert summary["gate_passed"]
    assert all(summary["gate_checks"].values())


def test_confirmation_summary_fails_if_one_seed_misses_nmse() -> None:
    repository = {"revision": "abc", "dirty": False, "status": []}
    evaluations = [
        _evaluation(0, 0.11, 1.0, repository),
        _evaluation(1, 0.12, 1.011, repository),
        _evaluation(2, 0.10, 0.999, repository),
    ]

    summary = summarize_confirmation(
        evaluations, _config(), selected_weight=0.03125, repository=repository
    )

    assert not summary["gate_passed"]
    assert not summary["gate_checks"]["nmse_ratio_every_seed"]
