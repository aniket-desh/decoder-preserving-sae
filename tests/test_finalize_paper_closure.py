import tarfile

from scripts import finalize_paper_closure as finalize


def test_artifact_files_excludes_rolling_checkpoints_and_temporary_files(tmp_path):
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    (artifact / "result.json").write_text("{}")
    (artifact / "checkpoint.pt").write_bytes(b"checkpoint")
    (artifact / "partial.json.tmp").write_text("partial")
    backup = artifact / ".hf_backup"
    backup.mkdir()
    (backup / "marker").touch()

    assert finalize.artifact_files([artifact]) == [artifact / "result.json"]


def test_file_record_is_relative_and_hashes_exact_bytes(tmp_path):
    path = tmp_path / "payload.bin"
    path.write_bytes(b"decoder preserving")

    record = finalize.file_record(path, tmp_path)

    assert record["path"] == "payload.bin"
    assert record["bytes"] == len(b"decoder preserving")
    assert record["sha256"] == finalize.sha256_file(path)


def test_code_bundle_contains_normalized_exact_inputs(tmp_path, monkeypatch):
    (tmp_path / "src").mkdir()
    (tmp_path / "src/example.py").write_text("value = 1\n")
    (tmp_path / "uv.lock").write_text("version = 1\n")
    monkeypatch.setattr(finalize, "CODE_PATHS", ("src/example.py",))
    output = tmp_path / "artifacts/code_bundle.tar"

    finalize.write_code_bundle(tmp_path, output)

    with tarfile.open(output) as archive:
        assert archive.getnames() == ["src/example.py", "uv.lock"]
        info = archive.getmember("src/example.py")
        assert info.mtime == 0
        assert archive.extractfile(info).read() == b"value = 1\n"
