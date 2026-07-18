import importlib.util
import json
from pathlib import Path

import pytest

from dpsae.release_manifest import canonical_digest, sha256_stable_file


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "render_arxiv_closure_figures", ROOT / "scripts/render_arxiv_closure_figures.py"
)
RENDERER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(RENDERER)


def _payload_manifest(path: Path, payload: Path, release_sha256: str) -> None:
    value = {
        "schema_version": 1,
        "complete": True,
        "core_release_manifest_sha256": release_sha256,
        "outputs": [
            {
                "role": "closure_payload",
                "path": str(payload.resolve()),
                "bytes": payload.stat().st_size,
                "sha256": sha256_stable_file(payload),
            }
        ],
    }
    value["build_manifest_sha256"] = canonical_digest(value)
    path.write_text(json.dumps(value) + "\n")


def test_renderer_requires_payload_output_bound_to_core_release(tmp_path):
    payload = tmp_path / "closure_payload.json"
    payload.write_text('{"complete": true}\n')
    manifest = tmp_path / "closure_payload_manifest.json"
    release = {"manifest_sha256": "a" * 64}
    _payload_manifest(manifest, payload, release["manifest_sha256"])

    observed = RENDERER._validate_payload_manifest(manifest, payload.resolve(), release)
    assert observed["complete"] is True

    payload.write_text('{"complete": false}\n')
    with pytest.raises(ValueError, match="changed after"):
        RENDERER._validate_payload_manifest(manifest, payload.resolve(), release)


def test_renderer_allows_hash_identical_payload_relocation(tmp_path):
    source = tmp_path / "source" / "closure_payload.json"
    source.parent.mkdir()
    source.write_text('{"complete": true}\n')
    manifest = tmp_path / "closure_payload_manifest.json"
    release = {"manifest_sha256": "a" * 64}
    _payload_manifest(manifest, source, release["manifest_sha256"])

    relocated = tmp_path / "relocated" / "closure_payload.json"
    relocated.parent.mkdir()
    relocated.write_bytes(source.read_bytes())

    observed = RENDERER._validate_payload_manifest(
        manifest, relocated.resolve(), release
    )
    assert observed["complete"] is True


def test_renderer_rejects_payload_manifest_for_another_core_release(tmp_path):
    payload = tmp_path / "closure_payload.json"
    payload.write_text('{"complete": true}\n')
    manifest = tmp_path / "closure_payload_manifest.json"
    _payload_manifest(manifest, payload, "b" * 64)

    with pytest.raises(ValueError, match="another core release"):
        RENDERER._validate_payload_manifest(
            manifest, payload.resolve(), {"manifest_sha256": "a" * 64}
        )
