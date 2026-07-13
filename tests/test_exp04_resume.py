import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "experiments" / "exp04_ioi_mechanism.py"
SPEC = importlib.util.spec_from_file_location("exp04_ioi_mechanism", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
EXPERIMENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPERIMENT)
load_partial_analysis = EXPERIMENT.load_partial_analysis


def test_load_partial_analysis_accepts_completed_model_subset(tmp_path: Path):
    path = tmp_path / "analysis_confirmation.json"
    value = {"mse_s0": {"sparse_probe_curve": []}}
    path.write_text(json.dumps(value))

    assert load_partial_analysis(path, {"mse_s0", "dpsae_s0"}) == value


@pytest.mark.parametrize(
    "value",
    [
        {"unknown": {}},
        {"mse_s0": []},
    ],
)
def test_load_partial_analysis_rejects_incompatible_results(tmp_path: Path, value):
    path = tmp_path / "analysis_confirmation.json"
    path.write_text(json.dumps(value))

    with pytest.raises(ValueError, match="invalid partial analysis"):
        load_partial_analysis(path, {"mse_s0"})
