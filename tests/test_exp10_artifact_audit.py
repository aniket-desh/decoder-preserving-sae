from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]


def _load_auditor() -> ModuleType:
    path = ROOT / "scripts/audit_exp10_artifacts.py"
    spec = importlib.util.spec_from_file_location("audit_exp10_artifacts", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


AUDIT = _load_auditor()


def _json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _torch(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(value, path)


def _metrics() -> dict[str, float]:
    return {"test_f1": 0.7, "test_acc": 0.7, "test_auc": 0.8, "val_auc": 0.75}


def _outputs(count: int) -> dict[str, torch.Tensor]:
    return {
        "decision_score": torch.linspace(-1, 1, count),
        "prediction": torch.tensor([index % 2 for index in range(count)]),
    }


def _classifier(width: int) -> dict[str, Any]:
    return {
        "coefficient": torch.zeros(1, width),
        "intercept": torch.zeros(1),
        "classes": torch.tensor([0, 1]),
        "C": 1.0,
    }


def _build_tree(tmp_path: Path) -> tuple[Path, Path]:
    output = tmp_path / "run"
    config_path = tmp_path / "config.json"
    datasets = ["task_a"]
    seeds = [11]
    config = {
        "model": {
            "transformer_lens_name": "tiny-model",
            "hook_name": "blocks.0.hook_resid_post",
            "d_model": 3,
        },
        "pilot_checkpoint": {"checkpoint_id": "tiny-checkpoint", "dictionary_size": 4},
        "benchmark": {
            "datasets": datasets,
            "dataset_manifest_sha256": AUDIT.canonical_digest(datasets),
            "family_by_dataset": {"task_a": "family_a"},
            "companion_full_code_matrix_format": "scipy_csr_exact_values",
            "probe_seeds": seeds,
            "ks": [1, 2],
        },
        "statistics": {"primary_k": 2},
        "runtime": {
            "worker_count": 2,
            "sparse_worker_shards": [
                {"method": "mse", "probe_seeds": seeds},
                {"method": "dpsae", "probe_seeds": seeds},
            ],
            "companion_seed_shards": [seeds, []],
            "timing_smoke": {
                "probe_seed": 99,
                "task_count": 1,
                "require_passed_report_before_workers": True,
            },
        },
    }
    _json(config_path, config)
    digest = AUDIT.canonical_digest(config)
    artifact_hashes = {
        "models_sha256": "1" * 64,
        "calibration_sha256": "2" * 64,
        "evaluation_sha256": "3" * 64,
    }
    _json(
        output / "resolved_config.json",
        {
            "schema_version": 1,
            "config_digest": digest,
            "config_sha256": AUDIT.file_sha256(config_path),
            "artifact_hashes": artifact_hashes,
        },
    )
    _json(
        output / "eligibility.json",
        {"schema_version": 1, "passed": True, "artifact_hashes": artifact_hashes},
    )
    cache_path = output / "cache" / "task_a.pt"
    _torch(cache_path, torch.zeros(4, 3))
    _json(
        output / "cache_ready.json",
        {
            "schema_version": 1,
            "complete": True,
            "config_digest": digest,
            "dataset_manifest_sha256": config["benchmark"]["dataset_manifest_sha256"],
            "files": {
                "task_a": {
                    "path": str(cache_path.resolve()),
                    "shape": [4, 3],
                    "bytes": cache_path.stat().st_size,
                    "sha256": AUDIT.file_sha256(cache_path),
                }
            },
        },
    )
    _json(
        output / "timing_smoke.json",
        {
            "schema_version": 2,
            "complete": True,
            "passed": True,
            "config_digest": digest,
            "probe_seed": 99,
            "task_count": 1,
            "saved_concept_metric_count": 0,
            "companion_full_code_matrix_format": "scipy_csr_exact_values",
        },
    )
    split = AUDIT._expected_split(config, "task_a", 11, 4)
    labels = torch.tensor([0, 1, 0, 1])

    for method in AUDIT.METHODS:
        job = output / "jobs/tiny-checkpoint" / method / "seed_11"
        raw_path = (
            job
            / "raw/release_custom_sae/sae_probes_tiny-model/normal_setting"
            / "task_a_blocks.0.hook_resid_post_l1.json"
        )
        raw_rows = []
        provenance_rows = []
        predictions = {}
        for k in config["benchmark"]["ks"]:
            ids = list(range(k))
            raw_rows.append(
                {
                    **_metrics(),
                    "k": k,
                    "dataset": "task_a",
                    "hook_name": config["model"]["hook_name"],
                    "reg_type": "l1",
                    "binarize": False,
                    "indices": ids,
                }
            )
            provenance_rows.append(
                {
                    "k": k,
                    "metrics": _metrics(),
                    "feature_ids": ids,
                    "feature_weights": [
                        {"feature_id": feature_id, "weight": 0.1} for feature_id in ids
                    ],
                    "intercept": 0.0,
                    "regularization_C": 1.0,
                }
            )
            predictions[str(k)] = _outputs(4)
        _json(raw_path, raw_rows)
        prediction_path = job / "predictions/task_a.pt"
        _torch(
            prediction_path,
            {
                "schema_version": 1,
                "config_digest": digest,
                "checkpoint_id": "tiny-checkpoint",
                "method": method,
                "probe_seed": 11,
                "dataset": "task_a",
                **split,
                "label": labels,
                "by_k": predictions,
                "decision_score_semantics": "test",
            },
        )
        provenance_path = job / "provenance/task_a.json"
        _json(
            provenance_path,
            {
                "schema_version": 2,
                "config_digest": digest,
                "checkpoint_id": "tiny-checkpoint",
                "method": method,
                "probe_seed": 11,
                "dataset": "task_a",
                "family": "family_a",
                "raw_result_sha256": AUDIT.file_sha256(raw_path),
                "heldout_split_id": split["split_id"],
                "heldout_example_count": 4,
                "heldout_example_id_policy": split["example_id_policy"],
                "heldout_predictions_sha256": AUDIT.file_sha256(prediction_path),
                "rows": provenance_rows,
            },
        )
        aggregate = job / "saebench_output/result_eval_results.json"
        _json(aggregate, {"method": method})
        _json(
            job / "done.json",
            {
                "schema_version": 1,
                "complete": True,
                "config_digest": digest,
                "artifact_hashes": artifact_hashes,
                "method": method,
                "probe_seed": 11,
                "dataset_count": 1,
                "dataset_manifest_sha256": config["benchmark"]["dataset_manifest_sha256"],
                "saebench_result_sha256": AUDIT.file_sha256(aggregate),
                "provenance_hashes": {"task_a": AUDIT.file_sha256(provenance_path)},
            },
        )

    companion = output / "companion/tiny-checkpoint/seed_11"
    weight_path = companion / "weights/task_a.pt"
    heldout_methods = {
        method: {representation: _outputs(4) for representation in ("full_code", "reconstruction")}
        for method in AUDIT.METHODS
    }
    _torch(
        weight_path,
        {
            "schema_version": 2,
            "config_digest": digest,
            "checkpoint_id": "tiny-checkpoint",
            "probe_seed": 11,
            "dataset": "task_a",
            "heldout": {
                **split,
                "label": labels,
                "decision_score_semantics": "test",
                "original_residual": _outputs(4),
                "methods": heldout_methods,
            },
            "original_residual": _classifier(3),
            **{
                method: {"full_code": _classifier(4), "reconstruction": _classifier(3)}
                for method in AUDIT.METHODS
            },
        },
    )
    metric_methods = {
        method: {representation: _metrics() for representation in ("full_code", "reconstruction")}
        for method in AUDIT.METHODS
    }
    metric_path = companion / "metrics/task_a.json"
    _json(
        metric_path,
        {
            "schema_version": 2,
            "config_digest": digest,
            "checkpoint_id": "tiny-checkpoint",
            "probe_seed": 11,
            "dataset": "task_a",
            "family": "family_a",
            "num_train": 4,
            "regularization": "sae_probes_find_best_reg_l2",
            "full_code_matrix_format": "scipy_csr_exact_values",
            "heldout_split_id": split["split_id"],
            "heldout_example_count": 4,
            "heldout_example_id_policy": split["example_id_policy"],
            "metrics": {"original_residual": _metrics(), "methods": metric_methods},
            "weights_sha256": AUDIT.file_sha256(weight_path),
        },
    )
    _json(
        companion / "done.json",
        {
            "schema_version": 1,
            "complete": True,
            "config_digest": digest,
            "artifact_hashes": artifact_hashes,
            "probe_seed": 11,
            "dataset_count": 1,
            "dataset_manifest_sha256": config["benchmark"]["dataset_manifest_sha256"],
            "dataset_hashes": {"task_a": AUDIT.file_sha256(metric_path)},
        },
    )

    timing_hash = AUDIT.file_sha256(output / "timing_smoke.json")
    for index, method in enumerate(AUDIT.METHODS):
        companion_seeds = [11] if index == 0 else []
        _json(
            output / "workers" / f"worker_{index}.json",
            {
                "schema_version": 1,
                "complete": True,
                "config_digest": digest,
                "worker_index": index,
                "method": method,
                "probe_seeds": [11],
                "companion_seeds": companion_seeds,
                "sparse_job_count": 1,
                "companion_job_count": len(companion_seeds),
                "timing_smoke_sha256": timing_hash,
            },
        )

    candidate_path = output / "candidate_associations.jsonl"
    candidate_path.write_text("")
    checks = {"complete_matrix": True}
    _json(
        output / "candidate_manifest.json",
        {
            "schema_version": 1,
            "config_digest": digest,
            "candidate_count": 0,
            "candidate_jsonl_sha256": AUDIT.file_sha256(candidate_path),
            "pilot_gate": {"passed": True, "checks": checks},
            "autointerp_eligible": False,
        },
    )
    _json(
        output / "advancement_report.json",
        {
            "schema_version": 1,
            "complete": True,
            "config_digest": digest,
            "artifact_hashes": artifact_hashes,
            "primary": {"k": 2},
            "checks": checks,
            "advance_fresh_confirmation": True,
            "advance_autointerp": False,
            "task_metrics": {"task_a": {"mse": {}, "dpsae": {}}},
            "companion_task_metrics": {"task_a": {}},
            "candidate_manifest_sha256": AUDIT.file_sha256(output / "candidate_manifest.json"),
        },
    )
    return config_path, output


def test_preaggregate_and_final_audits_write_exact_manifests(tmp_path: Path) -> None:
    config, output = _build_tree(tmp_path)
    pre = AUDIT.audit_artifacts(config_path=config, output_root=output, phase="pre-aggregate")
    final = AUDIT.audit_artifacts(config_path=config, output_root=output, phase="final")

    assert pre["passed"] is True
    assert pre["observed_counts"]["sparse_provenance"] == 2
    assert pre["observed_counts"]["companion_weights"] == 1
    assert pre["heldout_split_alignment_count"] == 1
    assert final["manifest_entry_count"] == pre["manifest_entry_count"] + 3
    assert (output / "artifact_audit_pre_aggregate.json").is_file()
    assert (output / "artifact_manifest_final.jsonl").is_file()


def test_audit_rejects_hash_tampering(tmp_path: Path) -> None:
    config, output = _build_tree(tmp_path)
    prediction = output / "jobs/tiny-checkpoint/mse/seed_11/predictions/task_a.pt"
    _torch(prediction, {"tampered": True})

    with pytest.raises(AUDIT.AuditError, match="prediction hash drift"):
        AUDIT.audit_artifacts(config_path=config, output_root=output, phase="pre-aggregate")


def test_audit_rejects_extra_artifacts(tmp_path: Path) -> None:
    config, output = _build_tree(tmp_path)
    _json(
        output / "jobs/tiny-checkpoint/mse/seed_11/provenance/extra.json",
        {},
    )

    with pytest.raises(AUDIT.AuditError, match="file set drift"):
        AUDIT.audit_artifacts(config_path=config, output_root=output, phase="pre-aggregate")


def test_audit_rejects_cross_representation_split_misalignment(tmp_path: Path) -> None:
    config, output = _build_tree(tmp_path)
    weight_path = output / "companion/tiny-checkpoint/seed_11/weights/task_a.pt"
    weights = torch.load(weight_path, map_location="cpu", weights_only=True)
    weights["heldout"]["split_id"] = "wrong-split"
    _torch(weight_path, weights)
    metric_path = output / "companion/tiny-checkpoint/seed_11/metrics/task_a.json"
    metrics = json.loads(metric_path.read_text())
    metrics["weights_sha256"] = AUDIT.file_sha256(weight_path)
    _json(metric_path, metrics)
    done_path = output / "companion/tiny-checkpoint/seed_11/done.json"
    done = json.loads(done_path.read_text())
    done["dataset_hashes"]["task_a"] = AUDIT.file_sha256(metric_path)
    _json(done_path, done)

    with pytest.raises(AUDIT.AuditError, match="held-out split mismatch"):
        AUDIT.audit_artifacts(config_path=config, output_root=output, phase="pre-aggregate")
