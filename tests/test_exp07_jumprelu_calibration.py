import importlib.util
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
