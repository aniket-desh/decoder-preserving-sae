from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from fractions import Fraction
from pathlib import Path

import pytest

from dpsae import cpu_quota


def _write(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value)


def test_v2_fractional_quota_is_floored_before_four_way_split(tmp_path, monkeypatch):
    _write(tmp_path / "cpu.max", "3230000 100000\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    _write(proc_cgroup, "0::/\n")
    monkeypatch.setattr(cpu_quota, "affinity_cpu_count", lambda: 96)

    budget = cpu_quota.resolve_cpu_budget(
        4,
        cgroup_root=tmp_path,
        proc_cgroup=proc_cgroup,
    )

    assert budget.cgroup_quota_cores == 32.3
    assert budget.effective_cpu_count == 32
    assert budget.threads_per_worker == 8


def test_v2_uses_tightest_ancestor_and_accepts_unlimited_child(tmp_path, monkeypatch):
    _write(tmp_path / "cpu.max", "250000 100000\n")
    _write(tmp_path / "pod" / "worker" / "cpu.max", "max 100000\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    _write(proc_cgroup, "0::/pod/worker\n")
    monkeypatch.setattr(cpu_quota, "affinity_cpu_count", lambda: 12)

    budget = cpu_quota.resolve_cpu_budget(
        4,
        cgroup_root=tmp_path,
        proc_cgroup=proc_cgroup,
    )

    assert budget.cgroup_quota_cores == 2.5
    assert budget.effective_cpu_count == 2
    assert budget.threads_per_worker == 1


def test_v1_cpu_controller_quota_and_unlimited_parent(tmp_path, monkeypatch):
    _write(tmp_path / "cpu" / "cpu.cfs_quota_us", "-1\n")
    _write(tmp_path / "cpu" / "cpu.cfs_period_us", "100000\n")
    _write(tmp_path / "cpu" / "team" / "cpu.cfs_quota_us", "645000\n")
    _write(tmp_path / "cpu" / "team" / "cpu.cfs_period_us", "100000\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    _write(proc_cgroup, "2:cpu,cpuacct:/team\n")
    monkeypatch.setattr(cpu_quota, "affinity_cpu_count", lambda: 12)

    budget = cpu_quota.resolve_cpu_budget(
        4,
        cgroup_root=tmp_path,
        proc_cgroup=proc_cgroup,
    )

    assert cpu_quota.cgroup_cpu_quota(
        cgroup_root=tmp_path,
        proc_cgroup=proc_cgroup,
    ) == Fraction(129, 20)
    assert budget.effective_cpu_count == 6
    assert budget.threads_per_worker == 1


def test_runpod_namespaced_v1_mount_root_with_cpu_symlink(tmp_path, monkeypatch):
    controller = tmp_path / "cpu,cpuacct"
    _write(controller / "cpu.cfs_quota_us", "3230000\n")
    _write(controller / "cpu.cfs_period_us", "100000\n")
    (tmp_path / "cpu").symlink_to(controller.name)
    proc_cgroup = tmp_path / "proc-self-cgroup"
    _write(
        proc_cgroup,
        "7:cpu,cpuacct:/docker/container-id\n0::/docker/container-id\n",
    )
    monkeypatch.setattr(cpu_quota, "affinity_cpu_count", lambda: 96)

    budget = cpu_quota.resolve_cpu_budget(
        4,
        cgroup_root=tmp_path,
        proc_cgroup=proc_cgroup,
    )

    assert asdict(budget) == {
        "visible_cpu_count": 96,
        "cgroup_quota_cores": 32.3,
        "effective_cpu_count": 32,
        "worker_count": 4,
        "threads_per_worker": 8,
    }


def test_unlimited_quota_falls_back_to_affinity(tmp_path, monkeypatch):
    _write(tmp_path / "cpu.max", "max 100000\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    _write(proc_cgroup, "0::/\n")
    monkeypatch.setattr(cpu_quota, "affinity_cpu_count", lambda: 14)

    budget = cpu_quota.resolve_cpu_budget(
        4,
        cgroup_root=tmp_path,
        proc_cgroup=proc_cgroup,
    )

    assert asdict(budget) == {
        "visible_cpu_count": 14,
        "cgroup_quota_cores": None,
        "effective_cpu_count": 14,
        "worker_count": 4,
        "threads_per_worker": 3,
    }


@pytest.mark.parametrize("value", ["garbage", "max nope", "0 100000", "10 0"])
def test_malformed_v2_quota_fails_closed(tmp_path, value):
    quota_path = tmp_path / "cpu.max"
    _write(quota_path, value)

    with pytest.raises(cpu_quota.CpuQuotaError):
        cpu_quota.read_cgroup_v2_quota(quota_path)


def test_incomplete_v1_quota_pair_fails_closed(tmp_path):
    quota_path = tmp_path / "cpu.cfs_quota_us"
    _write(quota_path, "100000\n")

    with pytest.raises(cpu_quota.CpuQuotaError, match="incomplete"):
        cpu_quota.read_cgroup_v1_quota(quota_path, tmp_path / "cpu.cfs_period_us")


def test_cli_json_is_canonical_and_machine_readable(tmp_path, monkeypatch, capsys):
    _write(tmp_path / "cpu.max", "3230000 100000\n")
    proc_cgroup = tmp_path / "proc-self-cgroup"
    _write(proc_cgroup, "0::/\n")
    monkeypatch.setattr(cpu_quota, "affinity_cpu_count", lambda: 96)
    output_path = tmp_path / "artifacts" / "cpu_budget.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "cpu_quota",
            "json",
            "--workers",
            "4",
            "--cgroup-root",
            str(tmp_path),
            "--proc-cgroup",
            str(proc_cgroup),
            "--output",
            str(output_path),
        ],
    )

    assert cpu_quota.main() == 0
    output = capsys.readouterr().out.strip()
    assert output == json.dumps(json.loads(output), sort_keys=True)
    assert json.loads(output)["threads_per_worker"] == 8
    assert output_path.read_text() == output + "\n"


def test_cpu_budget_write_is_idempotent_and_rejects_drift(tmp_path):
    path = tmp_path / "nested" / "cpu_budget.json"
    budget = cpu_quota.CpuBudget(96, 32.3, 32, 4, 8)

    cpu_quota.write_cpu_budget(path, budget)
    first_stat = path.stat()
    cpu_quota.write_cpu_budget(path, budget)

    assert path.stat().st_ino == first_stat.st_ino
    assert not list(path.parent.glob(".cpu_budget.json.*.tmp"))
    with pytest.raises(cpu_quota.CpuQuotaError, match="disagrees"):
        cpu_quota.write_cpu_budget(path, cpu_quota.CpuBudget(96, 31.0, 31, 4, 7))


def test_cpu_budget_write_rejects_noncanonical_existing_json(tmp_path):
    path = tmp_path / "cpu_budget.json"
    budget = cpu_quota.CpuBudget(96, 32.3, 32, 4, 8)
    path.write_text(json.dumps(asdict(budget), indent=2) + "\n")

    with pytest.raises(cpu_quota.CpuQuotaError, match="canonical"):
        cpu_quota.write_cpu_budget(path, budget)


@pytest.mark.parametrize("kind", ["symlink", "directory"])
def test_cpu_budget_write_rejects_nonregular_existing_path(tmp_path, kind):
    path = tmp_path / "cpu_budget.json"
    budget = cpu_quota.CpuBudget(96, 32.3, 32, 4, 8)
    if kind == "symlink":
        target = tmp_path / "target.json"
        target.write_text(cpu_quota.canonical_cpu_budget_json(budget))
        path.symlink_to(target)
    else:
        path.mkdir()

    with pytest.raises(cpu_quota.CpuQuotaError, match="not a regular file"):
        cpu_quota.write_cpu_budget(path, budget)


def _concurrent_budget_writes(monkeypatch, path, budgets):
    original_link = os.link
    barrier = threading.Barrier(len(budgets))

    def synchronized_link(source, destination):
        barrier.wait(timeout=5)
        return original_link(source, destination)

    monkeypatch.setattr(cpu_quota.os, "link", synchronized_link)
    with ThreadPoolExecutor(max_workers=len(budgets)) as executor:
        futures = [executor.submit(cpu_quota.write_cpu_budget, path, budget) for budget in budgets]
    return [future.exception() for future in futures]


def test_concurrent_matching_cpu_budget_writers_are_idempotent(tmp_path, monkeypatch):
    path = tmp_path / "cpu_budget.json"
    budget = cpu_quota.CpuBudget(96, 32.3, 32, 4, 8)

    errors = _concurrent_budget_writes(monkeypatch, path, [budget, budget])

    assert errors == [None, None]
    assert path.read_text() == cpu_quota.canonical_cpu_budget_json(budget)
    assert not list(tmp_path.glob(".cpu_budget.json.*.tmp"))


def test_concurrent_conflicting_cpu_budget_writer_cannot_clobber(tmp_path, monkeypatch):
    path = tmp_path / "cpu_budget.json"
    first = cpu_quota.CpuBudget(96, 32.3, 32, 4, 8)
    second = cpu_quota.CpuBudget(96, 31.0, 31, 4, 7)

    errors = _concurrent_budget_writes(monkeypatch, path, [first, second])

    assert sum(error is None for error in errors) == 1
    assert sum(isinstance(error, cpu_quota.CpuQuotaError) for error in errors) == 1
    assert path.read_text() in {
        cpu_quota.canonical_cpu_budget_json(first),
        cpu_quota.canonical_cpu_budget_json(second),
    }
    assert not list(tmp_path.glob(".cpu_budget.json.*.tmp"))


@pytest.mark.parametrize(
    "name",
    ["run_exp10_timing_smoke_a40.sh", "run_exp10_concept_4xa40.sh"],
)
def test_exp10_launcher_records_and_uses_cgroup_cpu_budget(name):
    source = (Path(__file__).parents[1] / "scripts" / name).read_text()

    assert "nproc" not in source
    assert "-m dpsae.cpu_quota json --workers 4" in source
    assert '--output "$OUTPUT_ROOT/cpu_budget.json"' in source
    assert 'export LOKY_MAX_CPU_COUNT="$EFFECTIVE_CPUS"' in source
    for variable in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        assert f'export {variable}="$WORKER_THREADS"' in source


def test_concept_launcher_hashes_cpu_budget_source():
    source = (
        Path(__file__).parents[1] / "scripts" / "run_exp10_concept_4xa40.sh"
    ).read_text()

    assert '"$ROOT/src/dpsae/cpu_quota.py"' in source
