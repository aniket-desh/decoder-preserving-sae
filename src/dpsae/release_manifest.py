"""Hash-complete, result-blind manifests for the arXiv experiment closure.

The scanner deliberately treats artifacts as opaque bytes.  It records every
regular file and directory below each configured root, rejects links and
partial-file suffixes, and never parses scientific result payloads.  This lets
the release audit include failed Exp10 attempts and monitor/control provenance
without accidentally opening a sealed or partial result.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = 1
RELEASE_NAME = "decoder_preserving_sae_arxiv_experiment_closure"


def canonical_digest(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _relative_path(path: Path, root: Path, *, label: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its configured anchor: {path}") from error
    if relative == Path("."):
        return "."
    return relative.as_posix()


def _resolve_below(anchor: Path, relative: str, *, label: str) -> Path:
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise ValueError(f"{label} must be a relative path without '..': {relative!r}")
    anchor = anchor.resolve()
    path = (anchor / raw).resolve()
    _relative_path(path, anchor, label=label)
    return path


def sha256_stable_file(path: Path, *, chunk_bytes: int = 8 * 1024 * 1024) -> str:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"release inputs must be regular non-symlink files: {path}")
    before = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_bytes):
            digest.update(chunk)
    after = path.stat()
    identity_before = (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns)
    identity_after = (after.st_dev, after.st_ino, after.st_size, after.st_mtime_ns)
    if identity_before != identity_after:
        raise RuntimeError(f"file changed while it was being hashed: {path}")
    return digest.hexdigest()


def file_record(path: Path, root: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": _relative_path(path, root, label="file"),
        "bytes": int(stat.st_size),
        "sha256": sha256_stable_file(path),
    }


def _walk_tree(root: Path) -> tuple[list[Path], list[Path]]:
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"artifact root must be a non-symlink directory: {root}")
    directories = [root]
    files: list[Path] = []
    for current, names, filenames in os.walk(root, followlinks=False):
        current_path = Path(current)
        names.sort()
        filenames.sort()
        for name in names:
            path = current_path / name
            if path.is_symlink() or not path.is_dir():
                raise ValueError(f"artifact trees may not contain linked/special dirs: {path}")
            directories.append(path)
        for name in filenames:
            path = current_path / name
            if path.is_symlink() or not path.is_file():
                raise ValueError(f"artifact trees may not contain linked/special files: {path}")
            if name.endswith((".tmp", ".part", ".partial")):
                raise RuntimeError(f"unfinished artifact is present at release freeze: {path}")
            files.append(path)
    return sorted(set(directories)), sorted(files)


def scan_tree(root: Path, anchor: Path) -> dict[str, Any]:
    root = root.resolve()
    anchor = anchor.resolve()
    root_relative = _relative_path(root, anchor, label="artifact root")
    directories, files = _walk_tree(root)
    directory_records = [
        _relative_path(path, root, label="artifact directory") for path in directories
    ]
    file_records = [file_record(path, root) for path in files]
    tree_identity = {
        "directories": directory_records,
        "files": file_records,
    }
    return {
        "anchor_relative_root": root_relative,
        "directory_count": len(directory_records),
        "file_count": len(file_records),
        "total_bytes": sum(int(record["bytes"]) for record in file_records),
        "tree_sha256": canonical_digest(tree_identity),
        **tree_identity,
    }


def _subroot_summaries(root: Path) -> list[dict[str, Any]]:
    summaries = []
    for child in sorted(root.iterdir()):
        if child.is_symlink():
            raise ValueError(f"collection roots may not contain symlinks: {child}")
        if not child.is_dir():
            continue
        tree = scan_tree(child, root)
        summaries.append(
            {
                "name": child.name,
                "directory_count": tree["directory_count"],
                "file_count": tree["file_count"],
                "total_bytes": tree["total_bytes"],
                "tree_sha256": tree["tree_sha256"],
            }
        )
    return summaries


def scan_artifact_group(
    spec: Mapping[str, Any], *, repository_root: Path, run_root: Path
) -> dict[str, Any]:
    group_id = str(spec["id"])
    anchor_name = str(spec["anchor"])
    if anchor_name not in {"repository", "run"}:
        raise ValueError(f"artifact group {group_id!r} has unknown anchor {anchor_name!r}")
    anchor = repository_root if anchor_name == "repository" else run_root
    path = _resolve_below(anchor, str(spec["path"]), label=f"group {group_id}")
    required = bool(spec["required"])
    kind = str(spec.get("kind", "tree"))
    if kind not in {"tree", "collection"}:
        raise ValueError(f"artifact group {group_id!r} has unknown kind {kind!r}")
    base = {
        "id": group_id,
        "anchor": anchor_name,
        "configured_path": str(spec["path"]),
        "required": required,
        "kind": kind,
        "purpose": str(spec["purpose"]),
    }
    if not path.exists():
        if required:
            raise FileNotFoundError(f"required artifact group is absent: {group_id}: {path}")
        return {**base, "present": False, "tree": None, "subroots": []}
    tree = scan_tree(path, anchor)
    subroots = _subroot_summaries(path) if kind == "collection" else []
    return {**base, "present": True, "tree": tree, "subroots": subroots}


def _validate_run_root_coverage(policy: Mapping[str, Any], run_root: Path) -> None:
    """Require one non-overlapping run group for every top-level run entry."""

    if run_root.is_symlink() or not run_root.is_dir():
        raise ValueError(f"run root must be a non-symlink directory: {run_root}")
    run_root = run_root.resolve()

    claims: list[tuple[str, Path]] = []
    for spec in policy["artifact_groups"]:
        if str(spec["anchor"]) != "run":
            continue
        group_id = str(spec["id"])
        path = _resolve_below(run_root, str(spec["path"]), label=f"group {group_id}")
        claims.append((group_id, path.relative_to(run_root)))

    root_claims = [group_id for group_id, relative in claims if relative == Path(".")]
    if len(root_claims) > 1 or (root_claims and len(claims) > 1):
        raise ValueError("overlapping top-level run artifact-group paths")
    if root_claims:
        return

    owners: dict[str, str] = {}
    for group_id, relative in claims:
        top_level = relative.parts[0]
        if top_level in owners:
            raise ValueError(
                "overlapping top-level run artifact-group paths: "
                f"{top_level!r} is claimed by more than one group"
            )
        owners[top_level] = group_id

    for group_id, relative in claims:
        if len(relative.parts) != 1:
            raise ValueError(
                "run-anchored artifact groups must claim a complete top-level entry: "
                f"{group_id!r} uses {relative.as_posix()!r}"
            )

    unexpected = sorted(entry.name for entry in run_root.iterdir() if entry.name not in owners)
    if unexpected:
        raise ValueError(f"unexpected top-level run-root entries: {unexpected}")


def _scan_artifact_groups(
    policy: Mapping[str, Any], *, repository_root: Path, run_root: Path
) -> list[dict[str, Any]]:
    _validate_run_root_coverage(policy, run_root)
    return [
        scan_artifact_group(
            spec, repository_root=repository_root, run_root=run_root
        )
        for spec in policy["artifact_groups"]
    ]


def _source_records(policy: Mapping[str, Any], repository_root: Path) -> list[dict[str, Any]]:
    records = []
    seen: set[str] = set()
    for spec in policy["source_files"]:
        relative = str(spec["path"])
        required = bool(spec["required"])
        if relative in seen:
            raise ValueError(f"duplicate source path in release policy: {relative}")
        seen.add(relative)
        path = _resolve_below(repository_root, relative, label="source file")
        if not path.exists():
            if required:
                raise FileNotFoundError(f"required release source is absent: {path}")
            records.append({"path": relative, "required": False, "present": False})
            continue
        record = file_record(path, repository_root)
        records.append({**record, "required": required, "present": True})
    return records


def repository_record(root: Path) -> dict[str, Any]:
    def command(arguments: Sequence[str]) -> str | None:
        try:
            return subprocess.check_output(
                list(arguments), cwd=root, text=True, stderr=subprocess.DEVNULL
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    status = command(("git", "status", "--porcelain=v1", "--untracked-files=all"))
    return {
        "revision": command(("git", "rev-parse", "HEAD")),
        "dirty": bool(status),
        "status": [] if not status else status.splitlines(),
    }


def load_policy(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"malformed release policy: {path}: {error}") from error
    if not isinstance(value, dict) or value.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported release-policy schema")
    if value.get("release") != RELEASE_NAME:
        raise ValueError("release-policy identity drift")
    groups = value.get("artifact_groups")
    sources = value.get("source_files")
    if not isinstance(groups, list) or not groups:
        raise ValueError("release policy must define artifact groups")
    if not isinstance(sources, list) or not sources:
        raise ValueError("release policy must define source files")
    group_ids = [str(group.get("id")) for group in groups if isinstance(group, dict)]
    if len(group_ids) != len(groups) or len(set(group_ids)) != len(group_ids):
        raise ValueError("release artifact-group IDs must be unique objects")
    return value


def _manifest_without_digest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in manifest.items() if key != "manifest_sha256"}


def build_manifest(
    *,
    policy_path: Path,
    repository_root: Path,
    run_root: Path,
    require_clean_repository: bool = True,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    run_root = run_root.resolve()
    policy_path = policy_path.resolve()
    policy = load_policy(policy_path)
    repository = repository_record(repository_root)
    if require_clean_repository and repository["dirty"]:
        raise RuntimeError("release freeze requires a clean repository worktree")
    policy_record = file_record(policy_path, repository_root)
    sources = _source_records(policy, repository_root)
    groups = _scan_artifact_groups(
        policy, repository_root=repository_root, run_root=run_root
    )
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "release": RELEASE_NAME,
        "inventory_complete": True,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": repository,
        "anchors": {
            "repository": str(repository_root),
            "run": str(run_root),
        },
        "policy": policy_record,
        "source_files": sources,
        "source_set_sha256": canonical_digest(sources),
        "artifact_groups": groups,
        "artifact_set_sha256": canonical_digest(groups),
        "result_payloads_parsed": False,
    }
    manifest["manifest_sha256"] = canonical_digest(manifest)
    return manifest


def audit_manifest(
    manifest: Mapping[str, Any],
    *,
    policy_path: Path,
    repository_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported release-manifest schema")
    if manifest.get("release") != RELEASE_NAME or manifest.get("inventory_complete") is not True:
        raise ValueError("release-manifest identity or completion drift")
    if manifest.get("result_payloads_parsed") is not False:
        raise ValueError("release manifest crossed the result-blind boundary")
    observed_digest = canonical_digest(_manifest_without_digest(manifest))
    if manifest.get("manifest_sha256") != observed_digest:
        raise ValueError("release-manifest self digest mismatch")

    repository_root = repository_root.resolve()
    run_root = run_root.resolve()
    policy_path = policy_path.resolve()
    policy = load_policy(policy_path)
    repository = repository_record(repository_root)
    if manifest.get("repository") != repository:
        raise ValueError("release repository identity changed after manifest creation")
    if manifest.get("policy") != file_record(policy_path, repository_root):
        raise ValueError("release policy changed after manifest creation")
    sources = _source_records(policy, repository_root)
    if manifest.get("source_files") != sources:
        raise ValueError("release source set changed after manifest creation")
    if manifest.get("source_set_sha256") != canonical_digest(sources):
        raise ValueError("release source-set digest mismatch")
    groups = _scan_artifact_groups(
        policy, repository_root=repository_root, run_root=run_root
    )
    if manifest.get("artifact_groups") != groups:
        raise ValueError("release artifact set changed after manifest creation")
    if manifest.get("artifact_set_sha256") != canonical_digest(groups):
        raise ValueError("release artifact-set digest mismatch")
    return {
        "schema_version": 1,
        "release": RELEASE_NAME,
        "complete": True,
        "manifest_sha256": observed_digest,
        "source_file_count": sum(bool(row.get("present")) for row in sources),
        "artifact_group_count": len(groups),
        "artifact_file_count": sum(
            int(group["tree"]["file_count"])
            for group in groups
            if group["present"]
        ),
        "result_payloads_parsed": False,
    }


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
