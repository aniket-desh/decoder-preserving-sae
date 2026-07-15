import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "exp07_jumprelu_calibration",
    ROOT / "experiments/exp07_jumprelu_calibration.py",
)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


def test_isotonic_fit_and_blind_bracket_interpolation():
    assert runner.isotonic_non_decreasing([29.0, 31.0, 30.0, 35.0]) == [
        29.0,
        30.5,
        30.5,
        35.0,
    ]
    rows = [
        {"method": "mse", "multiplier": 16.0, "l0": 30.5, "fitted_l0": 30.5},
        {"method": "mse", "multiplier": 17.0, "l0": 33.5, "fitted_l0": 33.5},
    ]
    selected = runner.interpolate_bracket(rows, method="mse", target=32.0)
    assert selected["interpolated_multiplier"] == 16.5


def test_late_l0_trajectory_reads_only_health_fields(tmp_path):
    path = tmp_path / "training.jsonl"
    records = []
    for index, l0 in enumerate((32.0, 32.2, 32.4, 32.6), start=1):
        records.append(
            {
                "tokens_seen": index * 1_000_000,
                "models": {
                    "mse_s0": {
                        "l0": l0,
                        "dead": 3,
                        "threshold_min": 0.1,
                        "threshold_mean": 0.2,
                        "threshold_max": 0.3,
                        "nmse": 999.0,
                        "decoder": 999.0,
                    }
                },
            }
        )
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    result = runner.late_l0_trajectory(path, records=4)["mse_s0"]
    assert abs(result["late_half_shift"] - 0.4) < 1e-12
    assert abs(result["l0_slope_per_million_tokens"] - 0.2) < 1e-12
    assert result["finite_health"]
    assert "nmse" not in result and "decoder" not in result
