import importlib.util
import json
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "audit_exp04_results.py"
SPEC = importlib.util.spec_from_file_location("audit_exp04_results", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
expected_models = AUDIT.expected_models
run_audit = AUDIT.run_audit


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


def build_complete_artifacts(root: Path) -> None:
    config = {
        "model_name": "test",
        "layer": 1,
        "sae": {"dictionary_size": 8},
        "training": {
            "decoder_weight_multipliers": [0.25],
            "confirmation_seeds": [0],
            "robustness_seeds": [0],
            "screen_tokens": 10,
            "confirmation_tokens": 20,
            "robustness_tokens": 15,
        },
        "ioi": {"feature_counts": [1, 2]},
    }
    write_json(root / "resolved_config.json", config)
    write_json(root / "screening_selection.json", {"selected_decoder_weight": 0.25})
    budgets = {
        "screen": 10,
        "confirmation": 20,
        "robustness16": 15,
        "robustness64": 15,
    }
    for stage, budget in budgets.items():
        names = expected_models(config, stage)
        validation = {
            name: {"nmse": 0.1, "decoder": 0.2, "l0": 4.0, "dead": 0}
            for name in names
        }
        write_json(
            root / stage / "done.json",
            {"stage": stage, "tokens_seen": budget, "validation": validation},
        )
        write_json(root / stage / "validation.json", validation)
        (root / stage / "models.pt").write_bytes(b"models")

    analysis = {"protocol": {"model": "test", "layer": 1}}
    for stage in ("confirmation", "robustness16", "robustness64"):
        stage_result = {}
        for name in expected_models(config, stage):
            row = {
                "sparse_probe_curve": [
                    {"features": count, "accuracy": 0.75, "auc": 0.8}
                    for count in config["ioi"]["feature_counts"]
                ],
                "original_dense_probe": {"accuracy": 0.8, "auc": 0.85},
                "reconstruction_dense_probe": {"accuracy": 0.78, "auc": 0.82},
                "features_to_80pct_dense": 2,
                "ranked_features": [0, 1],
                "matched_random_features": [2, 3],
            }
            if stage == "confirmation":
                row["causal_frontier"] = [
                    {
                        "features": count,
                        "full_logit_difference": 1.0,
                        "ablation_effect": 0.2,
                        "abc_patch_effect": 0.1,
                        "random_patch_effect": 0.01,
                        "ablation_relative_activation_change": 0.02,
                        "abc_patch_relative_activation_change": 0.03,
                        "random_patch_relative_activation_change": 0.02,
                    }
                    for count in config["ioi"]["feature_counts"]
                ]
                row["collateral_frontier"] = [
                    {
                        "features": count,
                        "collateral_kl": 0.01,
                        "original_cross_entropy": 3.0,
                        "reconstruction_cross_entropy": 3.1,
                    }
                    for count in config["ioi"]["feature_counts"]
                ]
            stage_result[name] = row
        analysis[stage] = stage_result
    write_json(root / "analysis.json", analysis)
    (root / "ioi_state_activations.pt").write_bytes(b"states")
    figures = root / "figures"
    figures.mkdir()
    (figures / "exp04_headline.pdf").write_bytes(b"pdf")
    (figures / "exp04_headline.png").write_bytes(b"png")
    (figures / "exp04_feature_traces.pdf").write_bytes(b"pdf")
    (figures / "exp04_feature_traces.png").write_bytes(b"png")


def test_complete_exp04_artifact_tree_passes(tmp_path: Path):
    build_complete_artifacts(tmp_path)

    report = run_audit(tmp_path)

    assert report["status"] == "passed"
    assert report["errors"] == []


def test_missing_exp04_model_is_reported(tmp_path: Path):
    build_complete_artifacts(tmp_path)
    validation_path = tmp_path / "confirmation" / "validation.json"
    validation = json.loads(validation_path.read_text())
    validation.pop("dpsae_s0")
    write_json(validation_path, validation)

    report = run_audit(tmp_path)

    assert report["status"] == "failed"
    assert any("missing validation models" in error for error in report["errors"])


def test_probe_metric_allows_float32_roundoff_at_one(tmp_path: Path):
    build_complete_artifacts(tmp_path)
    analysis_path = tmp_path / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    analysis["confirmation"]["mse_s0"]["sparse_probe_curve"][0]["auc"] = (
        1.0 + 2.0**-22
    )
    write_json(analysis_path, analysis)

    report = run_audit(tmp_path)

    assert report["status"] == "passed"


def test_probe_metric_rejects_material_probability_overshoot(tmp_path: Path):
    build_complete_artifacts(tmp_path)
    analysis_path = tmp_path / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    analysis["confirmation"]["mse_s0"]["sparse_probe_curve"][0]["auc"] = 1.001
    write_json(analysis_path, analysis)

    report = run_audit(tmp_path)

    assert report["status"] == "failed"
    assert any("invalid auc" in error for error in report["errors"])
