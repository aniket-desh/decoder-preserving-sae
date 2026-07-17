import importlib.util
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "record_arxiv_closure_cost", ROOT / "scripts/record_arxiv_closure_cost.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_build_ledger_is_explicitly_a_lower_bound(tmp_path):
    start = tmp_path / "setup.log"
    start.write_text("setup\n")
    os.utime(start, (1_000.0, 1_000.0))
    value = MODULE.build_ledger(
        retained_start=start,
        hourly_rate_usd=1.8,
        gpu_count=4,
        api_spend_usd=0.0,
        end_timestamp=4_600.0,
    )
    assert value["estimation_kind"] == "retained-window lower bound"
    assert value["elapsed_pod_hours"] == pytest.approx(1.0)
    assert value["allocated_a40_gpu_hours"] == pytest.approx(4.0)
    assert value["estimated_pod_charge_usd"] == pytest.approx(1.8)
    assert value["openai_api_spend_usd"] == 0.0


def test_build_ledger_rejects_an_end_before_start(tmp_path):
    start = tmp_path / "setup.log"
    start.write_text("setup\n")
    os.utime(start, (1_000.0, 1_000.0))
    with pytest.raises(ValueError, match="precedes"):
        MODULE.build_ledger(
            retained_start=start,
            hourly_rate_usd=1.8,
            gpu_count=4,
            api_spend_usd=0.0,
            end_timestamp=999.0,
        )
