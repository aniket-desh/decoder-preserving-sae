import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments import exp13_concept_confirmation as runner


def test_checked_in_contract_freezes_exact_matrix_and_blind_gate() -> None:
    config = runner.load_config()
    template = runner.shard_template()

    assert config["status"] == "frozen_pending_exp12_authorization_and_user_approval"
    assert runner.canonical_digest(template) == config["runtime"][
        "shard_template_sha256"
    ]
    assert [len(worker["sparse"]) for worker in template["workers"]] == [15] * 4
    assert [len(worker["companion"]) for worker in template["workers"]] == [
        8,
        8,
        7,
        7,
    ]
    assert config["runtime"]["cache_regeneration_forbidden"] is True
    assert config["statistics"]["family_tests"]["gate_forming"] is False
    assert config["candidates"]["never_quota_fill"] is True


def test_realized_shards_cover_every_identity_once() -> None:
    pairs = [11, 22, 33]
    seeds = list(range(100, 110))
    shards = runner.realized_shards(pairs, seeds)
    sparse = [
        (row["pair_seed"], row["method"], row["probe_seed"])
        for worker in shards["workers"]
        for row in worker["sparse"]
    ]
    companion = [
        (row["pair_seed"], row["probe_seed"])
        for worker in shards["workers"]
        for row in worker["companion"]
    ]

    assert Counter(sparse) == Counter(
        (pair, method, seed)
        for pair in pairs
        for method in runner.METHODS
        for seed in seeds
    )
    assert Counter(companion) == Counter(
        (pair, seed) for pair in pairs for seed in seeds
    )


def _timing(projected_hours: float) -> dict:
    return {
        "schema_version": 7,
        "complete": True,
        "passed": True,
        "names_and_concept_results_suppressed": True,
        "saved_concept_metric_count": 0,
        "projection": {"projected_pod_hours": projected_hours},
    }


def test_runtime_projection_multiplies_complete_pilot_and_fails_closed() -> None:
    config = runner.load_config()
    passing = runner.project_runtime(config, _timing(2.3968))
    failing = runner.project_runtime(config, _timing(2.6))

    assert passing["projected_pod_hours"] == pytest.approx(7.1904)
    assert passing["passed"] is True
    assert failing["projected_pod_hours"] == pytest.approx(7.8)
    assert failing["passed"] is False
    invalid = _timing(2.0)
    invalid["saved_concept_metric_count"] = 1
    with pytest.raises(RuntimeError, match="schema-v7"):
        runner.project_runtime(config, invalid)


def test_confirmation_checks_require_every_prespecified_gate() -> None:
    checks = runner.confirmation_checks(
        pair_macros={11: 0.004, 22: 0.006, 33: 0.008},
        pooled_interval={"lower": 0.001},
        complete_matrix=True,
        matched_gates=True,
        minimum_median=0.005,
    )
    assert all(checks.values())

    failed = runner.confirmation_checks(
        pair_macros={11: -0.001, 22: 0.006, 33: 0.008},
        pooled_interval={"lower": 0.001},
        complete_matrix=True,
        matched_gates=True,
        minimum_median=0.005,
    )
    assert failed["all_pair_macros_positive"] is False
    assert failed["median_pair_macro"] is True


def test_confirmation_gate_exports_pair_identities_for_autointerp() -> None:
    contract = {
        "pair_seeds": [11, 22, 33],
        "selected_requested_snapshot_tokens": 250_000_000,
    }
    contexts = {
        pair: {
            "config": {"pilot_checkpoint": {"checkpoint_id": f"checkpoint_{pair}"}}
        }
        for pair in contract["pair_seeds"]
    }
    gate = runner.confirmation_gate_record(
        contract=contract,
        contexts=contexts,
        checks={"complete_matrix": True},
        passed=True,
    )

    assert gate["passed"] is True
    assert gate["checkpoint_count"] == 3
    assert gate["pair_seeds"] == [11, 22, 33]
    assert gate["checkpoint_ids"]["22"] == "checkpoint_22"


def test_family_bootstrap_is_paired_stratified_centered_and_holm_reportable() -> None:
    labels = np.asarray([0] * 12 + [1] * 12)
    records = [
            {
                "label": labels,
                "mse_score": np.zeros(len(labels)),
            "dpsae_score": np.asarray([0.0] * 12 + [1.0] * 12),
        }
        for _ in range(3)
    ]
    first = runner.centered_paired_stratified_bootstrap_pvalue(
        records, samples=199, seed=42
    )
    second = runner.centered_paired_stratified_bootstrap_pvalue(
        records, samples=199, seed=42
    )

    assert first == second
    assert first["estimate"] > 0
    assert 0 < first["p_value_one_sided_centered"] <= 1
    assert runner.holm_adjust({"a": 0.01, "b": 0.04, "c": 0.03}) == {
        "a": pytest.approx(0.03),
        "b": pytest.approx(0.06),
        "c": pytest.approx(0.06),
    }


def test_vectorized_stratified_auc_counts_preserve_ties_exactly() -> None:
    draws = runner._auc_draws_from_stratified_counts(
        scores=np.asarray([0.0, 0.5, 0.5, 1.0]),
        negative_indices=np.asarray([0, 1]),
        positive_indices=np.asarray([2, 3]),
        negative_counts=np.asarray([[1, 1], [0, 2]]),
        positive_counts=np.asarray([[1, 1], [2, 0]]),
    )

    assert draws.tolist() == pytest.approx([0.875, 0.5])


def test_family_prediction_loader_uses_exact_exp10_heldout_schema(
    tmp_path: Path,
) -> None:
    contract = {"probe_seeds": [101]}
    contexts = {}
    for pair in (11, 22, 33):
        root = tmp_path / f"pair_{pair}"
        checkpoint_id = f"checkpoint_{pair}"
        contexts[pair] = {
            "root": root,
            "config": {
                "pilot_checkpoint": {"checkpoint_id": checkpoint_id},
                "benchmark": {
                    "datasets": ["task"],
                    "family_by_dataset": {"task": "synthetic"},
                },
            },
        }
        for method, scores in (
            ("mse", [0.1, 0.2, 0.3, 0.4]),
            ("dpsae", [0.0, 0.1, 0.9, 1.0]),
        ):
            job = root / "jobs" / checkpoint_id / method / "seed_101"
            (job / "predictions").mkdir(parents=True, exist_ok=True)
            (job / "provenance").mkdir(parents=True, exist_ok=True)
            prediction_path = job / "predictions/task.pt"
            torch.save(
                {
                    "split": "test",
                    "split_id": "frozen-split",
                    "example_id_policy": "positional",
                    "example_ids": ["a", "b", "c", "d"],
                    "label": torch.tensor([0, 0, 1, 1]),
                    "by_k": {"5": {"decision_score": torch.tensor(scores)}},
                },
                prediction_path,
            )
            (job / "provenance/task.json").write_text(
                json.dumps(
                    {
                        "heldout_predictions_sha256": runner.file_sha256(
                            prediction_path
                        )
                    }
                )
            )

    records = runner._family_prediction_records(
        contract=contract, contexts=contexts, family="synthetic"
    )

    assert len(records) == 3
    assert records[0]["label"].tolist() == [0, 0, 1, 1]
    assert records[0]["dpsae_score"].tolist() == pytest.approx([0.0, 0.1, 0.9, 1.0])


def _candidate_fixture(tmp_path: Path):
    config = copy.deepcopy(runner.load_config())
    config["candidates"]["maximum_per_method"] = 2
    seeds = list(range(10))
    contract = {
        "pair_seeds": [11, 22, 33],
        "probe_seeds": seeds,
        "pairs": [
            {
                "pair_seed": pair,
                **{
                    name: {"path": f"/{pair}/{name}", "bytes": pair, "sha256": str(pair)}
                    for name in (
                        "models",
                        "calibration",
                        "evaluation",
                        "source_snapshot_manifest",
                        "source_models",
                    )
                },
            }
            for pair in (11, 22, 33)
        ],
    }
    contexts = {}
    sparse_by_pair = {}
    for pair in contract["pair_seeds"]:
        pair_root = tmp_path / f"pair_{pair}"
        pair_config = {
            "pilot_checkpoint": {"checkpoint_id": f"checkpoint_{pair}"},
            "benchmark": {"family_by_dataset": {"task": "synthetic"}},
        }
        contexts[pair] = {"config": pair_config, "root": pair_root}
        sparse = {}
        for method in runner.METHODS:
            for seed in seeds:
                root = (
                    pair_root
                    / "jobs"
                    / f"checkpoint_{pair}"
                    / method
                    / f"seed_{seed}"
                )
                (root / "provenance").mkdir(parents=True, exist_ok=True)
                (root / "predictions").mkdir(parents=True, exist_ok=True)
                (root / "provenance/task.json").write_text(
                    json.dumps({"pair": pair, "method": method, "seed": seed})
                )
                (root / "predictions/task.pt").write_bytes(
                    f"{pair}-{method}-{seed}".encode()
                )
                features = [{"feature_id": 7, "weight": 0.5}]
                if seed < 5:
                    features.append({"feature_id": 8, "weight": 10.0})
                sparse[(method, seed, "task")] = {
                    "rows": [{"k": 5, "feature_weights": features}]
                }
        sparse_by_pair[pair] = sparse
    deltas = {pair: {"task": 0.01} for pair in contract["pair_seeds"]}
    return config, contract, contexts, sparse_by_pair, deltas


def test_candidate_promotion_is_checkpoint_local_recurrent_equal_and_hash_bound(
    tmp_path: Path,
) -> None:
    config, contract, contexts, sparse, deltas = _candidate_fixture(tmp_path)
    selected, positive = runner._candidate_pool(
        config=config,
        contract=contract,
        contexts=contexts,
        pair_task_deltas=deltas,
        sparse_by_pair=sparse,
    )

    assert positive == ["task"]
    assert Counter(row["method"] for row in selected) == {"mse": 2, "dpsae": 2}
    assert all(row["feature_id"] == 7 for row in selected)
    assert all(row["autointerp_eligible"] is True for row in selected)
    assert all(row["probe_seed_count"] == 10 for row in selected)
    assert len({(row["pair_seed"], row["feature_id"]) for row in selected}) == 2
    for row in selected:
        assert len(row["contributing_artifacts"]) == 10
        assert set(row["checkpoint_artifacts"]) == {
            "models",
            "calibration",
            "evaluation",
            "source_snapshot_manifest",
            "source_models",
        }
        assert all(
            set(record["provenance"]) == {"path", "bytes", "sha256"}
            and set(record["predictions"]) == {"path", "bytes", "sha256"}
            for record in row["contributing_artifacts"]
        )


def test_candidate_promotion_rejects_a_task_negative_in_any_pair(tmp_path: Path) -> None:
    config, contract, contexts, sparse, deltas = _candidate_fixture(tmp_path)
    deltas[22]["task"] = -0.001
    selected, positive = runner._candidate_pool(
        config=config,
        contract=contract,
        contexts=contexts,
        pair_task_deltas=deltas,
        sparse_by_pair=sparse,
    )

    assert selected == []
    assert positive == []


def test_launcher_keeps_all_long_stages_in_named_tmux_sessions() -> None:
    text = (runner.ROOT / "scripts/run_exp13_concept_confirmation_4xa40.sh").read_text()

    assert "EXP13_USER_APPROVED" in text
    assert "EXP13_INSIDE_TMUX" in text
    assert 'SESSION_PREFIX-gpu$index' in text
    assert '"$SESSION_PREFIX-finalize"' in text
    assert "remain-on-exit on" in text
    assert "finalize --wait-seconds" in text
    assert "retain-failure --stage worker" in text
    assert "retain-failure --stage finalizer" in text
    assert "prepare-cache" not in text
    assert "generate_model_activations" not in text


def test_early_worker_bootstrap_failure_is_retained_and_aborts_fleet(
    tmp_path: Path,
) -> None:
    config = runner.load_config()
    runner.retain_entrypoint_failure(
        config=config,
        output_root=tmp_path,
        command="run-worker",
        error=RuntimeError("bootstrap drift"),
        worker_index=2,
    )

    abort = json.loads((tmp_path / "abort_requested.json").read_text())
    status = json.loads((tmp_path / "worker_status/worker_2.json").read_text())
    assert abort["abort_requested"] is True
    assert status["state"] == "failed"
    assert status["error"] == "bootstrap drift"


def test_freeze_entrypoint_retains_fail_closed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail_freeze(**_kwargs):
        raise RuntimeError("authorization missing")

    monkeypatch.setattr(runner, "freeze_run", fail_freeze)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="authorization missing"):
        runner.main(
            [
                "--output-root",
                str(output),
                "freeze",
                "--base-config",
                "base.json",
                "--exp12-config",
                "exp12.json",
                "--exp12-root",
                "exp12",
                "--pilot-root",
                "pilot",
                "--pilot-audit",
                "audit.json",
                "--source-cache-ready",
                "cache.json",
                "--model-cache",
                "cache",
                "--saebench-root",
                "saebench",
            ]
        )

    failure = json.loads((output / "freeze_failed.json").read_text())
    assert failure["failed"] is True
    assert failure["error"] == "authorization missing"
