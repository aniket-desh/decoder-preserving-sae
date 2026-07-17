"""Fail-closed OpenAI Batch preparation and result validation for exp10."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
import time
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


def _spend_entries(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    preflight = manifest["cost_preflight"]
    entries: dict[str, dict[str, Any]] = {
        "primary_labels": {
            "model": manifest["model"],
            "planned_maximum_usd": float(preflight["maximum_primary_cost_usd"]),
            "actual_usd": None,
        }
    }
    for name, value in preflight["followups"].items():
        entries[name] = {
            "calls": int(value["calls"]),
            "planned_maximum_usd": float(value["maximum_cost_usd"]),
            "actual_usd": None,
        }
    return entries


def _spend_totals(entries: Mapping[str, Mapping[str, Any]]) -> dict[str, float]:
    actual = math.fsum(
        float(entry["actual_usd"])
        for entry in entries.values()
        if entry.get("actual_usd") is not None
    )
    reserved = math.fsum(
        float(entry["planned_maximum_usd"])
        for entry in entries.values()
        if entry.get("actual_usd") is None
    )
    return {
        "actual_usd": actual,
        "outstanding_reserved_usd": reserved,
        "worst_case_total_usd": actual + reserved,
    }


def _validate_spend_ledger(
    *,
    ledger: Mapping[str, Any],
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
) -> None:
    if ledger.get("schema_version") != 1:
        raise RuntimeError("autointerp spend ledger schema drift")
    if not isinstance(ledger.get("complete"), bool):
        raise RuntimeError("autointerp spend ledger completion field drift")
    if ledger.get("config_digest") != canonical_digest(config):
        raise RuntimeError("autointerp spend ledger config drift")
    if ledger.get("batch_manifest_sha256") != file_sha256(manifest_path):
        raise RuntimeError("autointerp spend ledger belongs to another Batch manifest")
    hard_cap = float(config["autointerp"]["hard_planned_cost_usd"])
    if float(ledger.get("hard_cap_usd", -1)) != hard_cap:
        raise RuntimeError("autointerp spend ledger hard cap drift")
    entries = ledger.get("entries")
    expected = _spend_entries(manifest)
    if not isinstance(entries, Mapping) or set(entries) != set(expected):
        raise RuntimeError("autointerp spend ledger entry schema drift")
    for name, expected_entry in expected.items():
        entry = entries[name]
        if not isinstance(entry, Mapping):
            raise RuntimeError(f"autointerp spend ledger entry {name} is not an object")
        if set(entry) != set(expected_entry):
            raise RuntimeError(f"autointerp spend ledger fields drift for {name}")
        if float(entry.get("planned_maximum_usd", -1)) != float(
            expected_entry["planned_maximum_usd"]
        ):
            raise RuntimeError(f"autointerp spend ledger reservation drift for {name}")
        for identity in set(expected_entry).difference(
            {"planned_maximum_usd", "actual_usd"}
        ):
            if entry.get(identity) != expected_entry[identity]:
                raise RuntimeError(f"autointerp spend ledger identity drift for {name}")
        actual = entry.get("actual_usd")
        if actual is not None:
            actual = float(actual)
            if not math.isfinite(actual) or actual < 0:
                raise RuntimeError(f"autointerp spend ledger actual is invalid for {name}")
            if actual > float(expected_entry["planned_maximum_usd"]) + 1e-12:
                raise RuntimeError(f"autointerp spend exceeded its reservation for {name}")
    totals = _spend_totals(entries)
    recorded = ledger.get("totals")
    if not isinstance(recorded, Mapping) or any(
        not math.isclose(float(recorded.get(key, -1)), value, rel_tol=0, abs_tol=1e-12)
        for key, value in totals.items()
    ):
        raise RuntimeError("autointerp spend ledger totals drift")
    if totals["worst_case_total_usd"] > hard_cap + 1e-12:
        raise RuntimeError("autointerp aggregate spend exceeds the hard experiment cap")


def initialize_spend_ledger(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    ledger_path: Path,
) -> dict[str, Any]:
    entries = _spend_entries(manifest)
    ledger = {
        "schema_version": 1,
        "complete": False,
        "config_digest": canonical_digest(config),
        "batch_manifest_sha256": file_sha256(manifest_path),
        "hard_cap_usd": float(config["autointerp"]["hard_planned_cost_usd"]),
        "entries": entries,
        "totals": _spend_totals(entries),
    }
    _validate_spend_ledger(
        ledger=ledger,
        config=config,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    if ledger_path.exists():
        existing = read_json(ledger_path)
        _validate_spend_ledger(
            ledger=existing,
            config=config,
            manifest=manifest,
            manifest_path=manifest_path,
        )
        if any(entry.get("actual_usd") is not None for entry in existing["entries"].values()):
            return existing
    atomic_json(ledger_path, ledger)
    return ledger


def record_primary_batch_spend(
    *,
    config: Mapping[str, Any],
    manifest: Mapping[str, Any],
    manifest_path: Path,
    ledger_path: Path,
    actual_usd: float,
) -> dict[str, Any]:
    ledger = read_json(ledger_path)
    _validate_spend_ledger(
        ledger=ledger,
        config=config,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    prior = ledger["entries"]["primary_labels"].get("actual_usd")
    if prior is not None and not math.isclose(
        float(prior), float(actual_usd), rel_tol=0, abs_tol=1e-12
    ):
        raise RuntimeError("primary Batch spend changed after it was recorded")
    ledger["entries"]["primary_labels"]["actual_usd"] = float(actual_usd)
    ledger["totals"] = _spend_totals(ledger["entries"])
    _validate_spend_ledger(
        ledger=ledger,
        config=config,
        manifest=manifest,
        manifest_path=manifest_path,
    )
    atomic_json(ledger_path, ledger)
    return ledger


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
    manifest_path = output_root / "batch_manifest.json"
    atomic_json(manifest_path, manifest)
    initialize_spend_ledger(
        config=config,
        manifest=manifest,
        manifest_path=manifest_path,
        ledger_path=output_root / "spend_ledger.json",
    )
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
    ledger_path = manifest_path.parent / "spend_ledger.json"
    if not ledger_path.is_file():
        raise RuntimeError("autointerp spend ledger is missing")
    _validate_spend_ledger(
        ledger=read_json(ledger_path),
        config=config,
        manifest=manifest,
        manifest_path=manifest_path,
    )
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


def validate_batch_output(
    *,
    config: Mapping[str, Any],
    request_path: Path,
    mapping_path: Path,
    batch_output_path: Path,
) -> dict[str, Any]:
    """Validate every result before a downloaded Batch artifact is accepted."""

    requests = read_jsonl(request_path)
    mapping_rows = read_jsonl(mapping_path)
    if any(not isinstance(row.get("custom_id"), str) or not row["custom_id"] for row in requests):
        raise RuntimeError("Batch request JSONL contains an invalid custom ID")
    if any(
        not isinstance(row.get("custom_id"), str) or not row["custom_id"]
        for row in mapping_rows
    ):
        raise RuntimeError("Batch mapping contains an invalid custom ID")
    mapping = {row["custom_id"]: row for row in mapping_rows}
    expected_ids = [row["custom_id"] for row in requests]
    if len(expected_ids) != len(set(expected_ids)):
        raise RuntimeError("Batch request JSONL contains duplicate custom IDs")
    if len(mapping) != len(mapping_rows) or len(mapping) != len(expected_ids) or set(mapping) != set(
        expected_ids
    ):
        raise RuntimeError("Batch mapping is missing or duplicating request IDs")
    output_rows = read_jsonl(batch_output_path)
    observed_ids = [row.get("custom_id") for row in output_rows]
    if any(not isinstance(value, str) or not value for value in observed_ids):
        raise RuntimeError("Batch output contains an invalid custom ID")
    counts: dict[Any, int] = {}
    for value in observed_ids:
        counts[value] = counts.get(value, 0) + 1
    duplicates = sorted(str(value) for value, count in counts.items() if count > 1)
    missing = sorted(set(expected_ids).difference(observed_ids))
    extra = sorted(str(value) for value in set(observed_ids).difference(expected_ids))
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
    return {
        "results": results,
        "request_count": len(results),
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "primary_batch_cost_usd": actual_cost,
    }


def _api_value(value: Any, name: str) -> Any:
    if isinstance(value, Mapping):
        return value.get(name)
    return getattr(value, name, None)


def _request_counts(value: Any) -> dict[str, int] | None:
    if value is None:
        return None
    result = {}
    for name in ("total", "completed", "failed"):
        item = _api_value(value, name)
        if item is None:
            return None
        result[name] = int(item)
    return result


def _downloaded_bytes(value: Any) -> bytes:
    text = getattr(value, "text", None)
    if callable(text):
        text = text()
    if isinstance(text, str):
        return text.encode()
    content = getattr(value, "content", None)
    if isinstance(content, bytes):
        return content
    read = getattr(value, "read", None)
    if callable(read):
        payload = read()
        if isinstance(payload, bytes):
            return payload
        if isinstance(payload, str):
            return payload.encode()
    raise RuntimeError("OpenAI Files response did not contain text or bytes")


def _validate_batch_state(
    *,
    state: Mapping[str, Any],
    config: Mapping[str, Any],
    manifest_path: Path,
    request_path: Path,
) -> None:
    if state.get("schema_version") != 1 or state.get("submitted") is not True:
        raise RuntimeError("autointerp Batch state is not a submitted schema-v1 job")
    if state.get("manifest_sha256") != file_sha256(manifest_path):
        raise RuntimeError("autointerp Batch state belongs to another manifest")
    if state.get("request_jsonl_sha256") != file_sha256(request_path):
        raise RuntimeError("autointerp Batch state belongs to another request JSONL")
    if state.get("endpoint") != config["autointerp"]["endpoint"]:
        raise RuntimeError("autointerp Batch state endpoint drift")
    if state.get("completion_window") != config["autointerp"]["completion_window"]:
        raise RuntimeError("autointerp Batch state completion-window drift")
    for name in ("batch_id", "input_file_id"):
        if not isinstance(state.get(name), str) or not state[name]:
            raise RuntimeError(f"autointerp Batch state lacks {name}")


def poll_and_download_batch(
    *,
    config: Mapping[str, Any],
    manifest_path: Path,
    request_path: Path,
    mapping_path: Path,
    state_path: Path,
    batch_output_path: Path,
    poll_seconds: float = 60,
    timeout_seconds: float = 26 * 3600,
    max_consecutive_poll_errors: int = 5,
    wait: bool = True,
    client: Any | None = None,
) -> dict[str, Any]:
    """Resume one submitted Batch until its validated output is durable locally."""

    validate_config_schema(config)
    if (
        not math.isfinite(poll_seconds)
        or not math.isfinite(timeout_seconds)
        or poll_seconds < 0
        or timeout_seconds <= 0
        or max_consecutive_poll_errors <= 0
    ):
        raise ValueError("Batch polling intervals must be nonnegative and finite")
    manifest = read_json(manifest_path)
    if manifest.get("confirmation_gate", {}).get("passed") is not True:
        raise RuntimeError("refusing to poll an unconfirmed autointerp batch")
    if manifest.get("request_jsonl_sha256") != file_sha256(request_path):
        raise RuntimeError("Batch request file changed after submission")
    if manifest.get("mapping_jsonl_sha256") != file_sha256(mapping_path):
        raise RuntimeError("Batch mapping changed after submission")
    if not state_path.is_file():
        raise RuntimeError("submitted Batch state is missing; polling never submits a job")
    state = read_json(state_path)
    _validate_batch_state(
        state=state,
        config=config,
        manifest_path=manifest_path,
        request_path=request_path,
    )
    ledger_path = manifest_path.parent / "spend_ledger.json"
    if not ledger_path.is_file():
        raise RuntimeError("autointerp spend ledger is missing")
    _validate_spend_ledger(
        ledger=read_json(ledger_path),
        config=config,
        manifest=manifest,
        manifest_path=manifest_path,
    )

    def accept_local_output() -> dict[str, Any]:
        if state.get("status") != "completed" or not state.get("output_file_id"):
            raise RuntimeError("downloaded Batch output lacks completed remote provenance")
        validated = validate_batch_output(
            config=config,
            request_path=request_path,
            mapping_path=mapping_path,
            batch_output_path=batch_output_path,
        )
        digest = file_sha256(batch_output_path)
        prior_digest = state.get("batch_output_sha256")
        if prior_digest is not None and prior_digest != digest:
            raise RuntimeError("downloaded Batch output changed after validation")
        record_primary_batch_spend(
            config=config,
            manifest=manifest,
            manifest_path=manifest_path,
            ledger_path=ledger_path,
            actual_usd=float(validated["primary_batch_cost_usd"]),
        )
        state.update(
            {
                "downloaded": True,
                "validated": True,
                "batch_output_path": str(batch_output_path.resolve()),
                "batch_output_sha256": digest,
                "batch_output_bytes": batch_output_path.stat().st_size,
                "validated_request_count": int(validated["request_count"]),
                "usage": {
                    "input_tokens": int(validated["input_tokens"]),
                    "output_tokens": int(validated["output_tokens"]),
                    "primary_batch_cost_usd": float(validated["primary_batch_cost_usd"]),
                },
            }
        )
        atomic_json(state_path, state)
        return state

    if state.get("downloaded") is True:
        if not batch_output_path.is_file():
            raise RuntimeError("Batch state records a download but the output file is missing")
        return accept_local_output()

    if client is None:
        try:
            from openai import OpenAI
        except ImportError as error:
            raise RuntimeError("install the exp10 autointerp optional dependency") from error
        client = OpenAI()

    active = {"validating", "in_progress", "finalizing", "cancelling"}
    failed = {"failed", "expired", "cancelled"}
    started = time.monotonic()
    consecutive_poll_errors = 0
    while True:
        try:
            batch = client.batches.retrieve(state["batch_id"])
        except Exception as error:
            consecutive_poll_errors += 1
            state.update(
                {
                    "poll_error_count": int(state.get("poll_error_count", 0)) + 1,
                    "consecutive_poll_errors": consecutive_poll_errors,
                    "last_poll_error_type": type(error).__name__,
                    "last_polled_unix_seconds": time.time(),
                }
            )
            atomic_json(state_path, state)
            if (
                not wait
                or consecutive_poll_errors >= max_consecutive_poll_errors
                or time.monotonic() - started >= timeout_seconds
            ):
                raise RuntimeError(
                    f"OpenAI Batch polling failed {consecutive_poll_errors} consecutive times"
                ) from error
            time.sleep(poll_seconds)
            continue
        consecutive_poll_errors = 0
        identities = {
            "batch_id": _api_value(batch, "id"),
            "input_file_id": _api_value(batch, "input_file_id"),
            "endpoint": _api_value(batch, "endpoint"),
            "completion_window": _api_value(batch, "completion_window"),
        }
        for name, observed in identities.items():
            if observed != state[name]:
                raise RuntimeError(f"retrieved Batch {name} drift")
        status = _api_value(batch, "status")
        if status not in active | failed | {"completed"}:
            raise RuntimeError(f"retrieved Batch has unknown status {status!r}")
        counts = _request_counts(_api_value(batch, "request_counts"))
        state.update(
            {
                "status": status,
                "output_file_id": _api_value(batch, "output_file_id"),
                "error_file_id": _api_value(batch, "error_file_id"),
                "request_counts": counts,
                "consecutive_poll_errors": 0,
                "last_polled_unix_seconds": time.time(),
            }
        )
        atomic_json(state_path, state)
        if status in failed:
            raise RuntimeError(f"OpenAI Batch reached terminal status {status}")
        if status == "completed":
            expected = int(manifest["request_count"])
            if counts is None or counts != {"total": expected, "completed": expected, "failed": 0}:
                raise RuntimeError(f"completed Batch request counts failed closed: {counts}")
            if not isinstance(state.get("output_file_id"), str) or not state["output_file_id"]:
                raise RuntimeError("completed Batch has no output file ID")
            if state.get("error_file_id") is not None:
                raise RuntimeError("completed Batch unexpectedly has an error file")
            if not batch_output_path.is_file():
                response = client.files.content(state["output_file_id"])
                temporary = batch_output_path.with_suffix(batch_output_path.suffix + ".download")
                temporary.parent.mkdir(parents=True, exist_ok=True)
                temporary.write_bytes(_downloaded_bytes(response))
                try:
                    validate_batch_output(
                        config=config,
                        request_path=request_path,
                        mapping_path=mapping_path,
                        batch_output_path=temporary,
                    )
                except Exception:
                    temporary.unlink(missing_ok=True)
                    raise
                temporary.replace(batch_output_path)
            return accept_local_output()
        if not wait:
            return state
        if time.monotonic() - started >= timeout_seconds:
            raise TimeoutError(f"OpenAI Batch remained {status} past the polling timeout")
        time.sleep(poll_seconds)


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
    validated = validate_batch_output(
        config=config,
        request_path=request_path,
        mapping_path=mapping_path,
        batch_output_path=batch_output_path,
    )
    results = validated["results"]
    total_input_tokens = int(validated["input_tokens"])
    total_output_tokens = int(validated["output_tokens"])
    actual_cost = float(validated["primary_batch_cost_usd"])
    record_primary_batch_spend(
        config=config,
        manifest=manifest,
        manifest_path=manifest_path,
        ledger_path=manifest_path.parent / "spend_ledger.json",
        actual_usd=actual_cost,
    )
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
