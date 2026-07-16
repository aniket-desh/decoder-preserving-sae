import copy
import json
from pathlib import Path

import pytest

from dpsae.exp10_autointerp import (
    file_sha256,
    finalize_batch,
    prepare_batch,
    read_jsonl,
    validate_label,
)
from experiments import exp10_concept_discovery as runner
from experiments.exp10_context_mining import mine_contexts


def write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, values) -> None:
    path.write_text("".join(json.dumps(value, sort_keys=True) + "\n" for value in values))


def confirmed_candidates(tmp_path: Path, count_per_method: int = 1):
    candidates = []
    for method in ("mse", "dpsae"):
        for index in range(count_per_method):
            candidates.append(
                {
                    "candidate_id": f"candidate_{method}_{index}",
                    "method": method,
                    "dataset": f"task_{index}",
                    "feature_id": index,
                    "probe_seed_frequency": 1.0,
                    "mean_absolute_weight": 0.5,
                    "autointerp_eligible": True,
                }
            )
    candidate_path = tmp_path / "candidates.jsonl"
    write_jsonl(candidate_path, candidates)
    manifest = {
        "confirmation_gate": {"passed": True, "checkpoint_count": 3},
        "autointerp_eligible": True,
        "candidate_jsonl_sha256": file_sha256(candidate_path),
    }
    manifest_path = tmp_path / "candidate_manifest.json"
    write_json(manifest_path, manifest)
    return manifest_path, candidate_path, candidates


def activation_rows(candidate_id: str):
    rows = []
    for index in range(12):
        rows.append(
            {
                "candidate_id": candidate_id,
                "context_id": f"{candidate_id}_dp_{index}",
                "split": "discovery",
                "text": f"positive discovery example {index}",
                "activation": float(12 - index),
                "tokens": ["positive", str(index)],
            }
        )
    for index in range(4):
        rows.append(
            {
                "candidate_id": candidate_id,
                "context_id": f"{candidate_id}_dn_{index}",
                "split": "discovery",
                "text": f"negative discovery example {index}",
                "activation": 0.0,
                "tokens": ["positive", "negative", str(index)],
            }
        )
    for index in range(2):
        rows.append(
            {
                "candidate_id": candidate_id,
                "context_id": f"{candidate_id}_hp_{index}",
                "split": "heldout",
                "text": f"positive heldout example {index}",
                "activation": 2.0 - index,
            }
        )
        rows.append(
            {
                "candidate_id": candidate_id,
                "context_id": f"{candidate_id}_hn_{index}",
                "split": "heldout",
                "text": f"negative heldout example {index}",
                "activation": 0.0,
            }
        )
    return rows


def small_context_config():
    config = copy.deepcopy(runner.load_config())
    config["context_mining"].update(
        discovery_high=1,
        discovery_middle=1,
        discovery_near_miss_negative=1,
        heldout_positive=1,
        heldout_negative=1,
        independent_relabel_fraction=0.5,
    )
    config["candidates"]["target_unique_maximum"] = 2
    return config


def mine_small_contexts(tmp_path: Path):
    config = small_context_config()
    manifest_path, candidates_path, candidates = confirmed_candidates(tmp_path)
    activations_path = tmp_path / "activations.jsonl"
    write_jsonl(
        activations_path,
        [row for candidate in candidates for row in activation_rows(candidate["candidate_id"])],
    )
    context_root = tmp_path / "contexts"
    result = mine_contexts(
        config=config,
        candidate_manifest_path=manifest_path,
        candidates_path=candidates_path,
        activation_jsonl=activations_path,
        output_root=context_root,
    )
    return config, manifest_path, candidates_path, context_root, result


def test_context_mining_is_disjoint_and_confirmation_gated(tmp_path: Path):
    config, _, _, context_root, result = mine_small_contexts(tmp_path)

    assert result["complete"]
    for candidate in read_jsonl(context_root / "candidate_contexts.jsonl"):
        ids = [
            item["context_id"]
            for group in candidate["contexts"].values()
            for values in group.values()
            for item in values
        ]
        assert len(ids) == len(set(ids))
    assert config["context_mining"]["independent_relabel_fraction"] == 0.5


def test_prepare_batch_is_blinded_stable_and_under_cost_cap(tmp_path: Path):
    config, candidate_manifest, candidates, context_root, _ = mine_small_contexts(tmp_path)
    batch_root = tmp_path / "batch"

    first = prepare_batch(
        config=config,
        candidate_manifest_path=candidate_manifest,
        candidates_path=candidates,
        context_manifest_path=context_root / "context_manifest.json",
        contexts_path=context_root / "candidate_contexts.jsonl",
        output_root=batch_root,
    )
    second = prepare_batch(
        config=config,
        candidate_manifest_path=candidate_manifest,
        candidates_path=candidates,
        context_manifest_path=context_root / "context_manifest.json",
        contexts_path=context_root / "candidate_contexts.jsonl",
        output_root=batch_root,
    )

    assert first == second
    assert first["request_count"] == 3
    assert first["cost_preflight"]["conservative_total_usd"] < 10
    requests = read_jsonl(batch_root / "batch_requests.jsonl")
    assert len({request["custom_id"] for request in requests}) == 3
    serialized = json.dumps(requests)
    assert "candidate_mse" not in serialized
    assert "candidate_dpsae" not in serialized
    assert all(request["body"]["model"] == "gpt-5.4-mini-2026-03-17" for request in requests)
    assert all(request["body"]["reasoning"] == {"effort": "low"} for request in requests)


def test_prepare_batch_rejects_planned_cost_above_hard_cap(tmp_path: Path):
    config, candidate_manifest, candidates, context_root, _ = mine_small_contexts(tmp_path)
    config["autointerp"]["hard_planned_cost_usd"] = 0.000001

    with pytest.raises(RuntimeError, match="exceeds hard"):
        prepare_batch(
            config=config,
            candidate_manifest_path=candidate_manifest,
            candidates_path=candidates,
            context_manifest_path=context_root / "context_manifest.json",
            contexts_path=context_root / "candidate_contexts.jsonl",
            output_root=tmp_path / "batch",
        )


def valid_label():
    return {
        "short_label": "hedged prediction",
        "description": "Qualifies a prediction with epistemic uncertainty.",
        "positive_evidence": ["might happen"],
        "counterevidence": ["certain factual statement"],
        "specificity": "medium",
        "polysemantic": False,
        "alternative_labels": ["uncertain forecast"],
    }


def test_finalize_rejects_duplicate_ids_and_validates_schema(tmp_path: Path):
    config, candidate_manifest, candidates, context_root, _ = mine_small_contexts(tmp_path)
    batch_root = tmp_path / "batch"
    prepare_batch(
        config=config,
        candidate_manifest_path=candidate_manifest,
        candidates_path=candidates,
        context_manifest_path=context_root / "context_manifest.json",
        contexts_path=context_root / "candidate_contexts.jsonl",
        output_root=batch_root,
    )
    requests = read_jsonl(batch_root / "batch_requests.jsonl")
    outputs = []
    for index, request in enumerate(requests):
        outputs.append(
            {
                "custom_id": request["custom_id"],
                "error": None,
                "response": {
                    "status_code": 200,
                    "request_id": f"req_{index}",
                    "body": {
                        "id": f"resp_{index}",
                        "model": config["autointerp"]["primary_model"],
                        "usage": {"input_tokens": 100, "output_tokens": 50},
                        "output": [
                            {
                                "type": "message",
                                "content": [
                                    {"type": "output_text", "text": json.dumps(valid_label())}
                                ],
                            }
                        ],
                    },
                },
            }
        )
    output_path = tmp_path / "batch_output.jsonl"
    write_jsonl(output_path, outputs)

    final = finalize_batch(
        config=config,
        manifest_path=batch_root / "batch_manifest.json",
        request_path=batch_root / "batch_requests.jsonl",
        mapping_path=batch_root / "batch_mapping.jsonl",
        batch_output_path=output_path,
        output_root=tmp_path / "labels",
    )
    assert final["complete"]
    assert final["request_count"] == 3

    write_jsonl(output_path, outputs + [outputs[0]])
    with pytest.raises(RuntimeError, match="duplicates"):
        finalize_batch(
            config=config,
            manifest_path=batch_root / "batch_manifest.json",
            request_path=batch_root / "batch_requests.jsonl",
            mapping_path=batch_root / "batch_mapping.jsonl",
            batch_output_path=output_path,
            output_root=tmp_path / "labels2",
        )

    invalid = valid_label()
    invalid["unexpected"] = True
    with pytest.raises(ValueError, match="missing or additional"):
        validate_label(invalid, config["autointerp"]["output_schema"])
