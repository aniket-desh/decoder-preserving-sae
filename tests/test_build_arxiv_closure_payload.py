import csv
import importlib.util
import json
from pathlib import Path

from dpsae import release_manifest as release


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "build_arxiv_closure_payload", ROOT / "scripts/build_arxiv_closure_payload.py"
)
BUILDER = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(BUILDER)


def _write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value) + "\n")


def _absolute_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": release.sha256_stable_file(path),
    }


def _fixture(tmp_path: Path):
    repository, run = tmp_path / "repository", tmp_path / "run"
    repository.mkdir()
    run.mkdir()
    (repository / "source.py").write_text("value = 1\n")
    config = repository / "configs/exp10_concept_discovery.json"
    _write(
        config,
        {
            "benchmark": {
                "datasets": ["task_a", "task_b"],
                "probe_seeds": list(range(10)),
                "ks": [1, 2, 5],
                "family_by_dataset": {"task_a": "family_a", "task_b": "family_b"},
            },
            "statistics": {
                "bootstrap_samples": 50,
                "bootstrap_seed": 19,
                "confidence_level": 0.95,
            },
        },
    )

    exp09 = run / "exp09_frozen_network"
    natural = exp09 / "natural_results.json"
    _write(
        natural,
        {
            "complete": True,
            "confirmatory": True,
            "protocol": {"noninferiority_margin": 1.01},
            "paired": [
                {
                    "seed": seed,
                    "kl_ratio_dpsae_to_mse": 0.9 + seed * 0.01,
                    "kl_ratio_dpsae_to_mse_ci95": [0.85, 0.95 + seed * 0.01],
                }
                for seed in (0, 1, 2)
            ],
        },
    )
    _write(
        exp09 / "completion_manifest.json",
        {
            "complete": True,
            "confirmatory": True,
            "natural_noninferiority_passed": True,
            "inputs": {"natural_results": _absolute_record(natural)},
        },
    )
    _write(exp09 / "smoke/natural_results.json", {"complete": False})
    _write(exp09 / "smoke/completion_manifest.json", {"complete": False})

    attempt = run / "exp10_concept_discovery" / "final-attempt"
    report = attempt / "advancement_report.json"
    _write(
        report,
        {
            "complete": True,
            "checks": {"complete_matrix": True},
            "advance_fresh_confirmation": False,
            "primary": {
                "family_block_interval": {
                    "estimate": -0.01,
                    "lower": -0.02,
                    "upper": 0.0,
                }
            },
            "companion_task_metrics": {
                dataset: {
                    "original_residual": {"test_auc": 0.8 + offset},
                    "methods": {
                        method: {
                            "reconstruction": {"test_auc": 0.7 + offset + method_offset},
                            "full_code": {"test_auc": 0.75 + offset + method_offset},
                        }
                        for method, method_offset in (("mse", 0.0), ("dpsae", 0.01))
                    },
                }
                for dataset, offset in (("task_a", 0.0), ("task_b", 0.02))
            },
        },
    )
    artifact_manifest = attempt / "artifact_manifest_final.jsonl"
    artifact_records = [
        {
            "path": "advancement_report.json",
            "kind": "advancement_report",
            "bytes": report.stat().st_size,
            "sha256": release.sha256_stable_file(report),
        }
    ]
    for method, method_offset in (("mse", 0.0), ("dpsae", 0.01)):
        for seed in range(10):
            result = attempt / "jobs" / method / f"seed_{seed}" / "result_eval_results.json"
            _write(
                result,
                {
                    "eval_result_details": {
                        dataset: {
                            "sae_metrics_by_k": {
                                str(k): {"test_auc": 0.6 + offset + method_offset + 0.001 * k}
                                for k in (1, 2, 5)
                            }
                        }
                        for dataset, offset in (("task_a", 0.0), ("task_b", 0.02))
                    }
                },
            )
            artifact_records.append(
                {
                    "path": str(result.relative_to(attempt)),
                    "kind": "saebench_result",
                    "method": method,
                    "probe_seed": seed,
                    "bytes": result.stat().st_size,
                    "sha256": release.sha256_stable_file(result),
                }
            )
    artifact_manifest.write_text(
        "".join(json.dumps(row) + "\n" for row in artifact_records)
    )
    _write(
        attempt / "artifact_audit_final.json",
        {
            "complete": True,
            "passed": True,
            "phase": "final",
            "expected_counts": {"advancement_report": 1},
            "observed_counts": {"advancement_report": 1},
            "manifest_path": str(artifact_manifest.resolve()),
            "manifest_sha256": release.sha256_stable_file(artifact_manifest),
        },
    )

    exp11 = run / "exp11_static_matched_nmse"
    _write(
        exp11 / "summary.json",
        {
            "complete": True,
            "screen": {
                "advance": False,
                "status": "no_matching_candidate",
                "rule": {"target_nmse_ratio": 1.07, "matching_tolerance": 0.01},
                "dpsae_anchor": {"nmse_ratio": 1.05, "decoder_reduction": 0.2},
                "candidates": [
                    {
                        "spec": {"loss_weight": beta},
                        "nmse_ratio": 1.10 + index * 0.01,
                        "decoder_reduction": 0.05 + index * 0.01,
                    }
                    for index, beta in enumerate((2, 4, 8, 16, 32))
                ],
                "selected": None,
            },
            "confirmation": {"status": "not_run_by_predeclared_gate"},
        },
    )

    policy = {
        "schema_version": 1,
        "release": release.RELEASE_NAME,
        "artifact_groups": [
            {
                "id": name,
                "anchor": "run",
                "path": path,
                "required": required,
                "kind": kind,
                "purpose": name,
            }
            for name, path, required, kind in (
                ("exp09_frozen_network", "exp09_frozen_network", True, "tree"),
                ("exp10_concept_discovery", "exp10_concept_discovery", True, "collection"),
                ("exp11_static_matched_nmse", "exp11_static_matched_nmse", True, "tree"),
                ("exp12_fresh_confirmation", "exp12", False, "tree"),
                ("exp13_concept_confirmation", "exp13", False, "tree"),
                ("exp10_autointerp", "autointerp", False, "collection"),
            )
        ],
        "source_files": [
            {"path": "source.py", "required": True},
            {"path": "configs/exp10_concept_discovery.json", "required": True},
        ],
    }
    policy_path = repository / "policy.json"
    _write(policy_path, policy)
    manifest = release.build_manifest(
        policy_path=policy_path,
        repository_root=repository,
        run_root=run,
        require_clean_repository=False,
    )
    manifest_path = tmp_path / "core_release.json"
    audit_path = tmp_path / "core_audit.json"
    _write(manifest_path, manifest)
    _write(
        audit_path,
        release.audit_manifest(
            manifest,
            policy_path=policy_path,
            repository_root=repository,
            run_root=run,
        ),
    )
    return repository, run, policy_path, manifest, manifest_path, audit_path


def test_builder_emits_hashed_null_payload_outside_core_and_core_reaudits(tmp_path):
    repository, run, policy, manifest, manifest_path, audit_path = _fixture(tmp_path)
    payload_path, csv_path, provenance_path = BUILDER.build(
        manifest_path, audit_path, tmp_path / "publication"
    )

    payload = json.loads(payload_path.read_text())
    assert payload["figures"]["concept_ladder"]["available"] is True
    assert len(payload["figures"]["concept_ladder"]["records"]) == 12
    assert payload["figures"]["frozen_network_noninferiority"]["available"] is True
    assert payload["figures"]["static_nmse_control"]["available"] is True
    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["experiment"]: row["status"] for row in rows} == {
        "Exp09": "passed",
        "Exp10": "failed",
        "Exp11": "no_matching_candidate",
        "Exp12": "not-run",
        "Exp13": "not-run",
        "Autointerp": "not-run",
    }
    provenance = json.loads(provenance_path.read_text())
    assert provenance["core_release_manifest_sha256"] == manifest["manifest_sha256"]
    assert all(len(row["sha256"]) == 64 for row in provenance["sources"])

    # Publication products are siblings, so they do not mutate the frozen core tree.
    assert release.audit_manifest(
        manifest,
        policy_path=policy,
        repository_root=repository,
        run_root=run,
    )["complete"] is True


def test_policy_excludes_post_audit_outputs_and_unrun_branch_sources():
    policy = json.loads((ROOT / "configs/arxiv_release_closure.json").read_text())
    assert "release_plot_inputs" not in {row["id"] for row in policy["artifact_groups"]}
    assert "exp13_concept_confirmation" in {row["id"] for row in policy["artifact_groups"]}
    sources = {row["path"] for row in policy["source_files"]}
    assert all((ROOT / path).is_file() for path in sources)
    assert not any("exp12" in path or "exp13" in path or "autointerp" in path for path in sources)
    assert {
        "experiments/exp09_frozen_network.py",
        "experiments/exp10_concept_discovery.py",
        "experiments/exp11_static_matched_nmse.py",
        "scripts/audit_exp10_artifacts.py",
    } <= sources
