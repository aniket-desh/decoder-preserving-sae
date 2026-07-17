"""Resolve the CPU budget available to a process inside a cgroup.

Linux container runtimes commonly leave all host CPUs in the process affinity
mask while limiting aggregate CPU time through a cgroup quota.  Thread-pool
limits must respect both constraints to avoid oversubscribing the container.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import stat
import tempfile
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path


DEFAULT_CGROUP_ROOT = Path("/sys/fs/cgroup")
DEFAULT_PROC_CGROUP = Path("/proc/self/cgroup")


class CpuQuotaError(RuntimeError):
    """Raised when an existing cgroup CPU control file cannot be parsed."""


@dataclass(frozen=True)
class CpuBudget:
    visible_cpu_count: int
    cgroup_quota_cores: float | None
    effective_cpu_count: int
    worker_count: int
    threads_per_worker: int


def affinity_cpu_count() -> int:
    """Return CPUs visible through affinity, falling back to ``os.cpu_count``."""

    try:
        count = len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        count = os.cpu_count() or 1
    return max(1, int(count))


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text().strip()
    except FileNotFoundError:
        return None
    except (OSError, UnicodeError) as error:
        raise CpuQuotaError(f"cannot read cgroup CPU control file {path}: {error}") from error


def read_cgroup_v2_quota(path: Path) -> Fraction | None:
    """Read a cgroup-v2 ``cpu.max`` file as a CPU fraction."""

    value = _read_text(path)
    if value is None:
        return None
    fields = value.split()
    if len(fields) != 2:
        raise CpuQuotaError(f"malformed cgroup-v2 CPU quota in {path}: {value!r}")
    if fields[0] == "max":
        try:
            period = int(fields[1])
        except ValueError as error:
            raise CpuQuotaError(
                f"malformed cgroup-v2 CPU period in {path}: {fields[1]!r}"
            ) from error
        if period <= 0:
            raise CpuQuotaError(f"non-positive cgroup-v2 CPU period in {path}: {period}")
        return None
    try:
        quota = int(fields[0])
        period = int(fields[1])
    except ValueError as error:
        raise CpuQuotaError(f"malformed cgroup-v2 CPU quota in {path}: {value!r}") from error
    if quota <= 0 or period <= 0:
        raise CpuQuotaError(f"non-positive cgroup-v2 CPU quota in {path}: {value!r}")
    return Fraction(quota, period)


def read_cgroup_v1_quota(quota_path: Path, period_path: Path) -> Fraction | None:
    """Read cgroup-v1 CFS quota and period files as a CPU fraction."""

    quota_value = _read_text(quota_path)
    period_value = _read_text(period_path)
    if quota_value is None and period_value is None:
        return None
    if quota_value is None or period_value is None:
        raise CpuQuotaError(
            "incomplete cgroup-v1 CPU quota pair: "
            f"quota={quota_path}, period={period_path}"
        )
    try:
        quota = int(quota_value)
        period = int(period_value)
    except ValueError as error:
        raise CpuQuotaError(
            f"malformed cgroup-v1 CPU quota: quota={quota_value!r}, period={period_value!r}"
        ) from error
    if quota == -1 and period > 0:
        return None
    if quota <= 0 or period <= 0:
        raise CpuQuotaError(
            f"invalid cgroup-v1 CPU quota: quota={quota}, period={period}"
        )
    return Fraction(quota, period)


def _membership_paths(proc_cgroup: Path) -> tuple[str | None, list[str]]:
    """Return the v2 path and all v1 CPU-controller paths for this process."""

    value = _read_text(proc_cgroup)
    if value is None:
        return None, []
    v2_path: str | None = None
    v1_paths: list[str] = []
    parsed_lines = 0
    for line in value.splitlines():
        fields = line.split(":", maxsplit=2)
        if len(fields) != 3:
            raise CpuQuotaError(f"malformed cgroup membership line in {proc_cgroup}: {line!r}")
        parsed_lines += 1
        controllers = fields[1].split(",") if fields[1] else []
        relative_path = fields[2].lstrip("/")
        if not controllers:
            v2_path = relative_path
        elif "cpu" in controllers:
            v1_paths.append(relative_path)
    if not parsed_lines:
        raise CpuQuotaError(f"empty cgroup membership file: {proc_cgroup}")
    return v2_path, v1_paths


def _ancestors(path: Path, root: Path) -> list[Path]:
    """List ``path`` and its parents through ``root``, without escaping it."""

    resolved_root = root.resolve()
    resolved_path = path.resolve()
    try:
        relative = resolved_path.relative_to(resolved_root)
    except ValueError:
        raise CpuQuotaError(f"cgroup membership path escapes its mount: {path}") from None
    candidates = [resolved_root / relative]
    while candidates[-1] != resolved_root:
        candidates.append(candidates[-1].parent)
    return candidates


def cgroup_cpu_quota(
    *,
    cgroup_root: Path = DEFAULT_CGROUP_ROOT,
    proc_cgroup: Path = DEFAULT_PROC_CGROUP,
) -> Fraction | None:
    """Return the tightest finite CPU quota imposed by cgroup v1 or v2.

    All ancestors are checked because a child cgroup can report an unlimited
    quota while an ancestor still limits the process.
    """

    cgroup_root = Path(cgroup_root)
    v2_membership, v1_memberships = _membership_paths(Path(proc_cgroup))
    quotas: list[Fraction] = []

    v2_paths = [cgroup_root]
    if v2_membership is not None:
        v2_paths = _ancestors(cgroup_root / v2_membership, cgroup_root)
    for directory in v2_paths:
        quota = read_cgroup_v2_quota(directory / "cpu.max")
        if quota is not None:
            quotas.append(quota)

    v1_mounts = (cgroup_root / "cpu", cgroup_root / "cpu,cpuacct", cgroup_root)
    for mount in v1_mounts:
        memberships = v1_memberships or [""]
        for membership in memberships:
            for directory in _ancestors(mount / membership, mount):
                quota = read_cgroup_v1_quota(
                    directory / "cpu.cfs_quota_us",
                    directory / "cpu.cfs_period_us",
                )
                if quota is not None:
                    quotas.append(quota)

    return min(quotas) if quotas else None


def effective_cpu_count(
    *,
    cgroup_root: Path = DEFAULT_CGROUP_ROOT,
    proc_cgroup: Path = DEFAULT_PROC_CGROUP,
) -> int:
    """Return the integer CPU budget after affinity and cgroup limits."""

    visible = affinity_cpu_count()
    quota = cgroup_cpu_quota(cgroup_root=cgroup_root, proc_cgroup=proc_cgroup)
    if quota is None:
        return visible
    return max(1, min(visible, math.floor(quota)))


def threads_per_worker(
    worker_count: int,
    *,
    cgroup_root: Path = DEFAULT_CGROUP_ROOT,
    proc_cgroup: Path = DEFAULT_PROC_CGROUP,
) -> int:
    """Split the effective integer CPU budget evenly across workers."""

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    effective = effective_cpu_count(cgroup_root=cgroup_root, proc_cgroup=proc_cgroup)
    return max(1, effective // worker_count)


def resolve_cpu_budget(
    worker_count: int = 4,
    *,
    cgroup_root: Path = DEFAULT_CGROUP_ROOT,
    proc_cgroup: Path = DEFAULT_PROC_CGROUP,
) -> CpuBudget:
    """Resolve all CPU-budget values once for provenance and launch policy."""

    if worker_count <= 0:
        raise ValueError("worker_count must be positive")
    visible = affinity_cpu_count()
    quota = cgroup_cpu_quota(cgroup_root=cgroup_root, proc_cgroup=proc_cgroup)
    effective = visible if quota is None else max(1, min(visible, math.floor(quota)))
    return CpuBudget(
        visible_cpu_count=visible,
        cgroup_quota_cores=float(quota) if quota is not None else None,
        effective_cpu_count=effective,
        worker_count=worker_count,
        threads_per_worker=max(1, effective // worker_count),
    )


def canonical_cpu_budget_json(budget: CpuBudget) -> str:
    """Serialize a CPU budget in the canonical on-disk representation."""

    return json.dumps(asdict(budget), sort_keys=True) + "\n"


def _read_existing_cpu_budget(path: Path) -> str | None:
    """Read an existing regular budget file without following symlinks."""

    try:
        metadata = path.lstat()
    except FileNotFoundError:
        return None
    except OSError as error:
        raise CpuQuotaError(f"cannot inspect existing CPU budget {path}: {error}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise CpuQuotaError(f"existing CPU budget is not a regular file: {path}")

    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    try:
        descriptor = os.open(path, flags)
    except FileNotFoundError:
        return None
    except OSError as error:
        raise CpuQuotaError(f"cannot open existing CPU budget {path}: {error}") from error
    try:
        if not stat.S_ISREG(os.fstat(descriptor).st_mode):
            raise CpuQuotaError(f"existing CPU budget is not a regular file: {path}")
        with os.fdopen(descriptor, encoding="utf-8") as existing_file:
            descriptor = -1
            return existing_file.read()
    except (OSError, UnicodeError) as error:
        raise CpuQuotaError(f"cannot read existing CPU budget {path}: {error}") from error
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _verify_existing_cpu_budget(path: Path, payload: str) -> bool:
    """Verify a published budget, returning false only if it vanished."""

    existing = _read_existing_cpu_budget(path)
    if existing is None:
        return False
    if existing != payload:
        raise CpuQuotaError(
            f"existing CPU budget disagrees with the current canonical budget: {path}"
        )
    return True


def write_cpu_budget(path: Path, budget: CpuBudget) -> None:
    """Atomically create a canonical budget, or verify an existing one exactly."""

    path = Path(path)
    payload = canonical_cpu_budget_json(budget)
    if _verify_existing_cpu_budget(path, payload):
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        while True:
            try:
                os.link(temporary_name, path)
                break
            except FileExistsError:
                if _verify_existing_cpu_budget(path, payload):
                    break
            except OSError as error:
                raise CpuQuotaError(f"cannot publish CPU budget {path}: {error}") from error
    finally:
        if temporary_name is not None:
            Path(temporary_name).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("effective-cpus", "threads-per-worker", "json"),
        nargs="?",
        default="effective-cpus",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cgroup-root", type=Path, default=DEFAULT_CGROUP_ROOT)
    parser.add_argument("--proc-cgroup", type=Path, default=DEFAULT_PROC_CGROUP)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    try:
        budget = resolve_cpu_budget(
            args.workers,
            cgroup_root=args.cgroup_root,
            proc_cgroup=args.proc_cgroup,
        )
    except (CpuQuotaError, ValueError) as error:
        parser.error(str(error))
    if args.command == "json":
        if args.output is not None:
            try:
                write_cpu_budget(args.output, budget)
            except CpuQuotaError as error:
                parser.error(str(error))
        print(canonical_cpu_budget_json(budget), end="")
        return 0
    if args.output is not None:
        parser.error("--output is supported only by the json command")
    value = (
        budget.effective_cpu_count
        if args.command == "effective-cpus"
        else budget.threads_per_worker
    )
    print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
