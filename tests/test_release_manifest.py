import json
import subprocess
import sys
from pathlib import Path

import pytest

from dpsae import release_manifest as release


def _policy(tmp_path):
    repository = tmp_path / "repository"
    run = tmp_path / "run"
    repository.mkdir()
    run.mkdir()
    (repository / "source.py").write_text("value = 1\n")
    (repository / "required").mkdir()
    (repository / "required/result.json").write_text('{"sealed": true}\n')
    (run / "attempts/failed-v1").mkdir(parents=True)
    (run / "attempts/failed-v1/provenance.json").write_text("{}\n")
    (run / "control").mkdir()
    (run / "control/status.json").write_text("{}\n")
    policy = {
        "schema_version": 1,
        "release": release.RELEASE_NAME,
        "artifact_groups": [
            {
                "id": "required",
                "anchor": "repository",
                "path": "required",
                "required": True,
                "kind": "tree",
                "purpose": "required results",
            },
            {
                "id": "attempts",
                "anchor": "run",
                "path": "attempts",
                "required": True,
                "kind": "collection",
                "purpose": "failed and final attempts",
            },
            {
                "id": "control",
                "anchor": "run",
                "path": "control",
                "required": True,
                "kind": "tree",
                "purpose": "control state",
            },
            {
                "id": "optional",
                "anchor": "run",
                "path": "optional",
                "required": False,
                "kind": "tree",
                "purpose": "conditional stage",
            },
        ],
        "source_files": [{"path": "source.py", "required": True}],
    }
    policy_path = repository / "policy.json"
    policy_path.write_text(json.dumps(policy))
    return repository, run, policy_path


def test_manifest_is_hash_complete_and_records_absent_optional_groups(tmp_path, monkeypatch):
    repository, run, policy = _policy(tmp_path)
    monkeypatch.setattr(
        release,
        "repository_record",
        lambda _root: {"revision": "abc123", "dirty": False, "status": []},
    )

    manifest = release.build_manifest(
        policy_path=policy, repository_root=repository, run_root=run
    )

    by_id = {group["id"]: group for group in manifest["artifact_groups"]}
    assert by_id["optional"]["present"] is False
    assert by_id["attempts"]["subroots"] == [
        {
            "name": "failed-v1",
            "directory_count": 1,
            "file_count": 1,
            "total_bytes": 3,
            "tree_sha256": by_id["attempts"]["subroots"][0]["tree_sha256"],
        }
    ]
    assert manifest["result_payloads_parsed"] is False
    assert manifest["manifest_sha256"] == release.canonical_digest(
        {key: value for key, value in manifest.items() if key != "manifest_sha256"}
    )

    report = release.audit_manifest(
        manifest,
        policy_path=policy,
        repository_root=repository,
        run_root=run,
    )
    assert report["complete"] is True
    assert report["artifact_file_count"] == 3


def test_audit_rejects_changed_added_and_removed_artifacts(tmp_path, monkeypatch):
    repository, run, policy = _policy(tmp_path)
    monkeypatch.setattr(
        release,
        "repository_record",
        lambda _root: {"revision": "abc123", "dirty": False, "status": []},
    )
    manifest = release.build_manifest(
        policy_path=policy, repository_root=repository, run_root=run
    )
    (run / "control/status.json").write_text('{"changed": true}\n')
    with pytest.raises(ValueError, match="artifact set changed"):
        release.audit_manifest(
            manifest,
            policy_path=policy,
            repository_root=repository,
            run_root=run,
        )

    (run / "control/status.json").write_text("{}\n")
    (run / "control/extra.json").write_text("{}\n")
    with pytest.raises(ValueError, match="artifact set changed"):
        release.audit_manifest(
            manifest,
            policy_path=policy,
            repository_root=repository,
            run_root=run,
        )


def test_required_roots_links_and_partial_files_fail_closed(tmp_path, monkeypatch):
    repository, run, policy = _policy(tmp_path)
    monkeypatch.setattr(
        release,
        "repository_record",
        lambda _root: {"revision": "abc123", "dirty": False, "status": []},
    )
    (run / "control/inflight.json.tmp").write_text("{}")
    with pytest.raises(RuntimeError, match="unfinished artifact"):
        release.build_manifest(
            policy_path=policy, repository_root=repository, run_root=run
        )
    (run / "control/inflight.json.tmp").unlink()
    (run / "control/link").symlink_to(run / "control/status.json")
    with pytest.raises(ValueError, match="linked/special files"):
        release.build_manifest(
            policy_path=policy, repository_root=repository, run_root=run
        )


def test_missing_required_group_fails_but_optional_source_is_explicit(tmp_path, monkeypatch):
    repository, run, policy = _policy(tmp_path)
    monkeypatch.setattr(
        release,
        "repository_record",
        lambda _root: {"revision": "abc123", "dirty": False, "status": []},
    )
    loaded = json.loads(policy.read_text())
    loaded["source_files"].append({"path": "future.py", "required": False})
    policy.write_text(json.dumps(loaded))
    manifest = release.build_manifest(
        policy_path=policy, repository_root=repository, run_root=run
    )
    assert manifest["source_files"][-1] == {
        "path": "future.py",
        "required": False,
        "present": False,
    }
    for path in (repository / "required").iterdir():
        path.unlink()
    (repository / "required").rmdir()
    with pytest.raises(FileNotFoundError, match="required artifact group"):
        release.build_manifest(
            policy_path=policy, repository_root=repository, run_root=run
        )


def test_build_and_audit_cli_round_trip(tmp_path):
    repository, run, policy = _policy(tmp_path)
    manifest = tmp_path / "release.json"
    report = tmp_path / "audit.json"
    script = Path(__file__).resolve().parents[1] / "scripts/finalize_arxiv_experiment_closure.py"

    subprocess.run(
        [
            sys.executable,
            str(script),
            "build",
            "--policy",
            str(policy),
            "--repository-root",
            str(repository),
            "--run-root",
            str(run),
            "--manifest",
            str(manifest),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        [
            sys.executable,
            str(script),
            "audit",
            "--policy",
            str(policy),
            "--repository-root",
            str(repository),
            "--run-root",
            str(run),
            "--manifest",
            str(manifest),
            "--report",
            str(report),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(report.read_text())["complete"] is True
