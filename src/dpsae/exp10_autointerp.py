"""Fail-closed OpenAI Batch preparation and result validation for exp10."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Any, Iterable, Mapping


ROOT = Path(__file__).resolve().parents[2]


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def canonical_digest(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    values = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number} is not a JSON object")
        values.append(value)
    return values


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def atomic_jsonl(path: Path, values: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as handle:
        for value in values:
            handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
    temporary.replace(path)


def repository_provenance() -> dict[str, Any]:
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True
    ).strip()
    dirty = bool(
        subprocess.check_output(
            ["git", "-C", str(ROOT), "status", "--porcelain"], text=True
        ).strip()
    )
    module_path = Path(__file__).resolve()
    return {
        "repository_revision": revision,
        "repository_dirty": dirty,
        "autointerp_source": str(module_path.relative_to(ROOT)),
        "autointerp_source_sha256": file_sha256(module_path),
    }


def validate_config_schema(config: Mapping[str, Any]) -> None:
    autointerp = config["autointerp"]
    schema = autointerp["output_schema"]
    required = {
        "short_label",
        "description",
        "positive_evidence",
        "counterevidence",
        "specificity",
        "polysemantic",
        "alternative_labels",
    }
    if schema.get("type") != "object" or schema.get("additionalProperties") is not False:
        raise ValueError("autointerp output schema must be a strict object")
    if set(schema.get("properties", {})) != required or set(schema.get("required", [])) != required:
        raise ValueError("autointerp output schema fields drifted")
    if autointerp["endpoint"] != "/v1/responses":
        raise ValueError("exp10 uses the Responses Batch endpoint")
    if autointerp["primary_reasoning_effort"] != "low":
        raise ValueError("exp10 primary reasoning effort is frozen to low")


def _validate_string(value: Any, schema: Mapping[str, Any], path: str) -> None:
    if not isinstance(value, str):
        raise ValueError(f"{path} must be a string")
    length = len(value)
    if length < int(schema.get("minLength", 0)) or length > int(
        schema.get("maxLength", 2**31)
    ):
        raise ValueError(f"{path} length violates the strict schema")


def validate_label(payload: Mapping[str, Any], schema: Mapping[str, Any]) -> None:
    properties = schema["properties"]
    if set(payload) != set(schema["required"]):
        raise ValueError("label output has missing or additional fields")
    for key in ("short_label", "description"):
        _validate_string(payload[key], properties[key], key)
    for key in ("positive_evidence", "counterevidence", "alternative_labels"):
        value = payload[key]
        item_schema = properties[key]
        if not isinstance(value, list):
            raise ValueError(f"{key} must be an array")
        if len(value) < int(item_schema.get("minItems", 0)) or len(value) > int(
            item_schema.get("maxItems", 2**31)
        ):
            raise ValueError(f"{key} item count violates the strict schema")
        for index, item in enumerate(value):
            _validate_string(item, item_schema["items"], f"{key}[{index}]")
    if payload["specificity"] not in properties["specificity"]["enum"]:
        raise ValueError("specificity is outside the frozen enum")
    if not isinstance(payload["polysemantic"], bool):
        raise ValueError("polysemantic must be boolean")


def estimate_input_tokens(value: Any) -> int:
    """Conservative dependency-free preflight estimator frozen in the config."""

    raw = canonical_json(value).encode()
    whitespace_tokens = len(canonical_json(value).split())
    return max(math.ceil(len(raw) / 3), 2 * whitespace_tokens) + 256


def request_cost_usd(
    *,
    model: str,
    input_tokens: int,
    output_tokens: int,
    config: Mapping[str, Any],
) -> float:
    autointerp = config["autointerp"]
    prices = autointerp["standard_prices_per_million"].get(model)
    if not isinstance(prices, Mapping):
        raise ValueError(f"no frozen price for model {model}")
    standard = (
        input_tokens * float(prices["input"])
        + output_tokens * float(prices["output"])
    ) / 1_000_000
    return standard * float(autointerp["batch_discount"])


def _balanced_candidates(
    candidates: list[dict[str, Any]], config: Mapping[str, Any]
) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {"mse": [], "dpsae": []}
    for candidate in candidates:
        method = candidate.get("method")
        if method not in grouped:
            raise ValueError(f"candidate has unknown method {method!r}")
        if candidate.get("autointerp_eligible") is not True:
            raise RuntimeError("candidate file contains a non-confirmed association")
        grouped[method].append(candidate)
    for method in grouped:
        grouped[method].sort(
            key=lambda item: (
                -float(item["probe_seed_frequency"]),
                -float(item["mean_absolute_weight"]),
                str(item["candidate_id"]),
            )
        )
    maximum = int(config["candidates"]["target_unique_maximum"])
    per_method = min(len(grouped["mse"]), len(grouped["dpsae"]), maximum // 2)
    if per_method == 0:
        raise RuntimeError("no method-balanced confirmed candidates are available")
    selected = grouped["mse"][:per_method] + grouped["dpsae"][:per_method]
    return sorted(selected, key=lambda item: str(item["candidate_id"]))


def _flatten_context_group(group: Mapping[str, list[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    result = []
    for kind in sorted(group):
        for context in group[kind]:
            result.append(
                {
                    "kind": kind,
                    "context_id": context["context_id"],
                    "text": context["text"],
                    "activation": context["activation"],
                    **(
                        {"active_token": context["active_token"]}
                        if "active_token" in context
                        else {}
                    ),
                }
            )
    return result


SYSTEM_PROMPT = (
    "You label one hidden feature from blinded activating and non-activating text contexts. "
    "Infer the narrow intersection across positives, use negatives to reject correlated context, "
    "and state uncertainty through specificity and polysemanticity. Do not infer the model, method, "
    "feature rank, benchmark task, or experimental outcome. Return only the requested JSON object."
)


def _user_prompt(contexts: list[dict[str, Any]]) -> str:
    lines = [
        "Explain the latent feature represented by these blinded contexts.",
        "Activation values are comparable only within this feature.",
    ]
    for index, context in enumerate(contexts, 1):
        token = f"; active token={context['active_token']!r}" if "active_token" in context else ""
        lines.append(
            f"[{index:02d}] {context['kind']}; activation={float(context['activation']):.6g}{token}\n"
            f"{context['text']}"
        )
    return "\n\n".join(lines)


def _opaque_feature_id(candidate_id: str, config: Mapping[str, Any]) -> str:
    salt = config["autointerp"]["stable_id_salt"]
    return "feature_" + hashlib.sha256(f"{salt}\0{candidate_id}".encode()).hexdigest()[:24]


def _request(
    *,
    custom_id: str,
    contexts: list[dict[str, Any]],
    config: Mapping[str, Any],
) -> tuple[dict[str, Any], int]:
    autointerp = config["autointerp"]
    input_value = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(contexts)},
    ]
    tokens = estimate_input_tokens(input_value)
    maximum = int(autointerp["maximum_input_tokens_per_primary_request"])
    if tokens > maximum:
        raise RuntimeError(
            f"request {custom_id} preflight is {tokens} tokens, above frozen maximum {maximum}"
        )
    body = {
        "model": autointerp["primary_model"],
        "reasoning": {"effort": autointerp["primary_reasoning_effort"]},
        "max_output_tokens": int(autointerp["maximum_output_tokens_per_primary_request"]),
        "input": input_value,
        "text": {
            "format": {
                "type": "json_schema",
                "name": autointerp["output_schema_name"],
                "strict": True,
                "schema": autointerp["output_schema"],
            }
        },
    }
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": autointerp["endpoint"],
        "body": body,
    }, tokens


def _planned_cost(
    *,
    candidate_count: int,
    request_count: int,
    estimated_primary_input_tokens: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    autointerp = config["autointerp"]
    primary_model = autointerp["primary_model"]
    maximum_input = int(autointerp["maximum_input_tokens_per_primary_request"])
    maximum_output = int(autointerp["maximum_output_tokens_per_primary_request"])
    primary_estimated = request_cost_usd(
        model=primary_model,
        input_tokens=estimated_primary_input_tokens,
        output_tokens=request_count * maximum_output,
        config=config,
    )
    primary_maximum = request_count * request_cost_usd(
        model=primary_model,
        input_tokens=maximum_input,
        output_tokens=maximum_output,
        config=config,
    )
    followups = {}
    followup_total = 0.0
    for name, plan in autointerp["planned_followup_maxima"].items():
        calls = math.ceil(candidate_count * float(plan["calls_per_candidate"]))
        cost = calls * request_cost_usd(
            model=plan["model"],
            input_tokens=int(plan["input_tokens"]),
            output_tokens=int(plan["output_tokens"]),
            config=config,
        )
        followups[name] = {"calls": calls, "maximum_cost_usd": cost}
        followup_total += cost
    conservative = primary_maximum + followup_total
    result = {
        "candidate_count": candidate_count,
        "primary_request_count": request_count,
        "estimated_primary_input_tokens": estimated_primary_input_tokens,
        "estimated_primary_cost_usd": primary_estimated,
        "maximum_primary_cost_usd": primary_maximum,
        "followups": followups,
        "conservative_total_usd": conservative,
        "hard_cap_usd": float(autointerp["hard_planned_cost_usd"]),
    }
    if conservative > result["hard_cap_usd"]:
        raise RuntimeError(
            f"planned Batch cost ${conservative:.2f} exceeds hard ${result['hard_cap_usd']:.2f} cap"
        )
    return result


def prepare_batch(
    *,
    config: Mapping[str, Any],
    candidate_manifest_path: Path,
    candidates_path: Path,
    context_manifest_path: Path,
    contexts_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    validate_config_schema(config)
    candidate_manifest = read_json(candidate_manifest_path)
    if (
        candidate_manifest.get("confirmation_gate", {}).get("passed") is not True
        or candidate_manifest.get("autointerp_eligible") is not True
    ):
        raise RuntimeError("Batch preparation requires the fresh-confirmation gate")
    if candidate_manifest.get("candidate_jsonl_sha256") != file_sha256(candidates_path):
        raise RuntimeError("candidate JSONL changed after confirmation")
    context_manifest = read_json(context_manifest_path)
    if context_manifest.get("confirmation_gate", {}).get("passed") is not True:
        raise RuntimeError("context manifest is not confirmation-gated")
    if context_manifest.get("contexts_jsonl_sha256") != file_sha256(contexts_path):
        raise RuntimeError("mined contexts changed after their manifest")
    if context_manifest.get("candidate_jsonl_sha256") != file_sha256(candidates_path):
        raise RuntimeError("contexts were mined for another candidate file")

    candidates = _balanced_candidates(read_jsonl(candidates_path), config)
    contexts = {row["candidate_id"]: row for row in read_jsonl(contexts_path)}
    if not set(item["candidate_id"] for item in candidates).issubset(contexts):
        raise RuntimeError("some selected candidates have no mined contexts")
    relabel_count = math.ceil(
        len(candidates) * float(config["context_mining"]["independent_relabel_fraction"])
    )
    relabel_ids = {
        item["candidate_id"]
        for item in sorted(
            candidates,
            key=lambda item: hashlib.sha256(
                f"{config['context_mining']['sampling_seed']}:{item['candidate_id']}".encode()
            ).hexdigest(),
        )[:relabel_count]
    }
    requests = []
    mapping = []
    total_estimated_input = 0
    for candidate in candidates:
        candidate_id = candidate["candidate_id"]
        opaque = _opaque_feature_id(candidate_id, config)
        variants = [("primary", "discovery_primary")]
        if candidate_id in relabel_ids:
            variants.append(("relabel", "discovery_relabel"))
        for variant, group_name in variants:
            selected_contexts = _flatten_context_group(contexts[candidate_id]["contexts"][group_name])
            custom_id = f"{opaque}_{'p' if variant == 'primary' else 'r'}"
            request, estimated_tokens = _request(
                custom_id=custom_id, contexts=selected_contexts, config=config
            )
            requests.append(request)
            total_estimated_input += estimated_tokens
            mapping.append(
                {
                    "custom_id": custom_id,
                    "opaque_feature_id": opaque,
                    "candidate_id": candidate_id,
                    "variant": variant,
                    "context_group": group_name,
                    "context_ids": [value["context_id"] for value in selected_contexts],
                    "estimated_input_tokens": estimated_tokens,
                    "source": candidate,
                }
            )
    custom_ids = [request["custom_id"] for request in requests]
    if len(custom_ids) != len(set(custom_ids)):
        raise RuntimeError("stable Batch custom IDs collided")
    costs = _planned_cost(
        candidate_count=len(candidates),
        request_count=len(requests),
        estimated_primary_input_tokens=total_estimated_input,
        config=config,
    )
    request_path = output_root / "batch_requests.jsonl"
    mapping_path = output_root / "batch_mapping.jsonl"
    atomic_jsonl(request_path, requests)
    atomic_jsonl(mapping_path, mapping)
    manifest = {
        "schema_version": 1,
        "complete": True,
        "confirmation_gate": candidate_manifest["confirmation_gate"],
        "config_digest": canonical_digest(config),
        "output_schema_sha256": canonical_digest(config["autointerp"]["output_schema"]),
        "model": config["autointerp"]["primary_model"],
        "reasoning_effort": config["autointerp"]["primary_reasoning_effort"],
        "endpoint": config["autointerp"]["endpoint"],
        "candidate_manifest_sha256": file_sha256(candidate_manifest_path),
        "context_manifest_sha256": file_sha256(context_manifest_path),
        "request_jsonl_sha256": file_sha256(request_path),
        "mapping_jsonl_sha256": file_sha256(mapping_path),
        "candidate_count": len(candidates),
        "request_count": len(requests),
        "relabel_count": relabel_count,
        "cost_preflight": costs,
        "repository_provenance": repository_provenance(),
    }
    atomic_json(output_root / "batch_manifest.json", manifest)
    return manifest


def submit_batch(
    *,
    config: Mapping[str, Any],
    manifest_path: Path,
    request_path: Path,
    state_path: Path,
) -> dict[str, Any]:
    """Submit once; existing matching state is returned without another API call."""

    validate_config_schema(config)
    manifest = read_json(manifest_path)
    if manifest.get("confirmation_gate", {}).get("passed") is not True:
        raise RuntimeError("refusing to submit an unconfirmed autointerp batch")
    if manifest.get("request_jsonl_sha256") != file_sha256(request_path):
        raise RuntimeError("Batch request file changed after preflight")
    cost = float(manifest["cost_preflight"]["conservative_total_usd"])
    hard_cap = float(config["autointerp"]["hard_planned_cost_usd"])
    if cost > hard_cap:
        raise RuntimeError("preflight cost exceeds the hard cap")
    if state_path.exists():
        state = read_json(state_path)
        if (
            state.get("request_jsonl_sha256") != file_sha256(request_path)
            or state.get("manifest_sha256") != file_sha256(manifest_path)
        ):
            raise RuntimeError("existing Batch state belongs to another request")
        return state

    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("install the exp10 autointerp optional dependency") from error
    client = OpenAI()
    with request_path.open("rb") as handle:
        input_file = client.files.create(file=handle, purpose="batch")
    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint=config["autointerp"]["endpoint"],
        completion_window=config["autointerp"]["completion_window"],
        metadata={
            "experiment": config["experiment_id"],
            "request_sha256": file_sha256(request_path)[:32],
        },
    )
    state = {
        "schema_version": 1,
        "submitted": True,
        "manifest_sha256": file_sha256(manifest_path),
        "request_jsonl_sha256": file_sha256(request_path),
        "input_file_id": input_file.id,
        "batch_id": batch.id,
        "endpoint": config["autointerp"]["endpoint"],
        "completion_window": config["autointerp"]["completion_window"],
        "status": batch.status,
    }
    atomic_json(state_path, state)
    return state


def _extract_output_text(body: Mapping[str, Any]) -> str:
    texts = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text" and isinstance(content.get("text"), str):
                texts.append(content["text"])
    if len(texts) != 1:
        raise ValueError(f"expected one structured output_text block, observed {len(texts)}")
    return texts[0]


def finalize_batch(
    *,
    config: Mapping[str, Any],
    manifest_path: Path,
    request_path: Path,
    mapping_path: Path,
    batch_output_path: Path,
    output_root: Path,
) -> dict[str, Any]:
    validate_config_schema(config)
    manifest = read_json(manifest_path)
    if manifest.get("request_jsonl_sha256") != file_sha256(request_path):
        raise RuntimeError("request JSONL changed after submission")
    if manifest.get("mapping_jsonl_sha256") != file_sha256(mapping_path):
        raise RuntimeError("Batch mapping changed after submission")
    requests = read_jsonl(request_path)
    mapping = {row["custom_id"]: row for row in read_jsonl(mapping_path)}
    expected_ids = [row["custom_id"] for row in requests]
    if len(mapping) != len(expected_ids) or set(mapping) != set(expected_ids):
        raise RuntimeError("Batch mapping is missing or duplicating request IDs")
    output_rows = read_jsonl(batch_output_path)
    observed_ids = [row.get("custom_id") for row in output_rows]
    duplicates = sorted({value for value in observed_ids if observed_ids.count(value) > 1})
    missing = sorted(set(expected_ids).difference(observed_ids))
    extra = sorted(set(observed_ids).difference(expected_ids))
    if duplicates or missing or extra:
        raise RuntimeError(
            f"Batch ID mismatch: duplicates={duplicates}, missing={missing}, extra={extra}"
        )
    by_id = {row["custom_id"]: row for row in output_rows}
    results = []
    total_input_tokens = 0
    total_output_tokens = 0
    for custom_id in expected_ids:
        row = by_id[custom_id]
        if row.get("error") is not None:
            raise RuntimeError(f"Batch request {custom_id} failed: {row['error']}")
        response = row.get("response")
        if not isinstance(response, Mapping) or response.get("status_code") != 200:
            raise RuntimeError(f"Batch request {custom_id} has no HTTP 200 response")
        body = response.get("body")
        if not isinstance(body, Mapping):
            raise RuntimeError(f"Batch request {custom_id} has no response body")
        if body.get("model") != config["autointerp"]["primary_model"]:
            raise RuntimeError(f"Batch response model drift for {custom_id}")
        text = _extract_output_text(body)
        label = json.loads(text)
        if not isinstance(label, Mapping):
            raise ValueError(f"Batch response {custom_id} is not a JSON object")
        validate_label(label, config["autointerp"]["output_schema"])
        usage = body.get("usage", {})
        input_tokens = int(usage.get("input_tokens", 0))
        output_tokens = int(usage.get("output_tokens", 0))
        if input_tokens <= 0 or output_tokens <= 0:
            raise RuntimeError(f"Batch response {custom_id} lacks positive token usage")
        total_input_tokens += input_tokens
        total_output_tokens += output_tokens
        results.append(
            {
                "schema_version": 1,
                **mapping[custom_id],
                "request_id": response.get("request_id"),
                "response_id": body.get("id"),
                "model": body.get("model"),
                "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
                "label": dict(label),
            }
        )
    actual_cost = request_cost_usd(
        model=config["autointerp"]["primary_model"],
        input_tokens=total_input_tokens,
        output_tokens=total_output_tokens,
        config=config,
    )
    if actual_cost > float(config["autointerp"]["hard_planned_cost_usd"]):
        raise RuntimeError("actual primary Batch cost exceeds the hard experiment cap")
    result_path = output_root / "labels.jsonl"
    atomic_jsonl(result_path, results)
    grouped: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        grouped.setdefault(result["candidate_id"], []).append(result)
    relabeled = [values for values in grouped.values() if len(values) == 2]
    exact_short_label_matches = sum(
        values[0]["label"]["short_label"].casefold()
        == values[1]["label"]["short_label"].casefold()
        for values in relabeled
    )
    final_manifest = {
        "schema_version": 1,
        "complete": True,
        "config_digest": canonical_digest(config),
        "batch_manifest_sha256": file_sha256(manifest_path),
        "batch_output_sha256": file_sha256(batch_output_path),
        "labels_jsonl_sha256": file_sha256(result_path),
        "request_count": len(results),
        "candidate_count": len(grouped),
        "usage": {
            "input_tokens": total_input_tokens,
            "output_tokens": total_output_tokens,
            "primary_batch_cost_usd": actual_cost,
        },
        "relabel_stability": {
            "candidate_count": len(relabeled),
            "exact_short_label_match_rate": (
                exact_short_label_matches / len(relabeled) if relabeled else None
            ),
        },
        "repository_provenance": manifest["repository_provenance"],
    }
    atomic_json(output_root / "labels_manifest.json", final_manifest)
    return final_manifest
