from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from scripts import prepare_exp08_run as runner


def write_file(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)


def test_contract_binds_plot_code_and_source_baselines(tmp_path, monkeypatch) -> None:
    clean_root = tmp_path / "clean"
    source_root = tmp_path / "source"
    for relative in runner.CODE_PATHS:
        write_file(clean_root / relative, relative.encode())
    external = runner.external_paths(source_root)
    for name, path in external.items():
        write_file(path, name.encode())

    repository = {"revision": "revision", "dirty": False, "status": []}
    monkeypatch.setattr(runner, "ROOT", clean_root)
    monkeypatch.setattr(runner, "repository_state", lambda _: repository)

    contract = runner.build_contract(source_root)

    assert contract["source_artifact_tree"]["used_for_code"] is False
    assert {
        "scripts/finalize_exp08_candidates.sh",
        "scripts/plot_exp08_candidates.py",
        "scripts/run_exp08_gpu_runpod.sh",
        "scripts/run_exp08_synthetic_runpod.sh",
        "src/dpsae/plot_style.py",
    }.issubset(contract["code"])
    assert all((clean_root / path).is_file() for path in contract["code"])
    expected_baselines = {
        "static_baseline_evaluation",
        "structured_baseline_metrics",
        "structured_baseline_group_metrics",
        "structured_baseline_metadata",
        "structured_baseline_crossover",
    }
    assert expected_baselines.issubset(contract["external_inputs"])
    for name in expected_baselines:
        record = contract["external_inputs"][name]
        assert record["sha256"] == hashlib.sha256(name.encode()).hexdigest()


def test_contract_refuses_source_tree_as_clean_worktree(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(runner, "ROOT", tmp_path)
    with pytest.raises(ValueError, match="must differ"):
        runner.build_contract(tmp_path)
