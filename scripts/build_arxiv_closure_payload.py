#!/usr/bin/env python3
"""Build the null-path closure payload from a hash-audited core release."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
for value in (ROOT, ROOT / "src"):
    if str(value) not in sys.path:
        sys.path.insert(0, str(value))

from dpsae.exp10_statistics import family_block_bootstrap  # noqa: E402
from dpsae.release_manifest import (  # noqa: E402
    atomic_json,
    canonical_digest,
    sha256_stable_file,
)

SUMMARY_FIELDS = (
    "experiment",
    "endpoint",
    "estimate",
    "ci_low",
    "ci_high",
    "gate",
    "status",
    "scope",
)


def _validate_payload(payload: Mapping[str, Any]) -> None:
    if payload.get("schema_version") != 1 or payload.get("complete") is not True:
        raise ValueError("incomplete closure plotting payload")
    digest = payload.get("release_manifest_sha256")
    if not isinstance(digest, str) or len(digest) != 64:
        raise ValueError("closure plotting payload is not release-bound")
    concept = payload["figures"]["concept_ladder"]
    keys = {(row["method"], row["stage"]) for row in concept["records"]}
    expected = {
        (method, stage)
        for method in ("mse", "dpsae")
        for stage in ("residual", "reconstruction", "full_code", "k5", "k2", "k1")
    }
    if concept.get("available") is not True or keys != expected:
        raise ValueError("concept ladder is not a complete 2x6 grid")
    if any(set(row) != set(SUMMARY_FIELDS) for row in payload["summary_table"]["rows"]):
        raise ValueError("closure summary row schema drift")


def _write_summary(rows: list[Mapping[str, Any]], path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _record(path: Path, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_stable_file(path),
    }


class AuditedFiles:
    def __init__(self, release: Mapping[str, Any]) -> None:
        self.release = release
        self.groups = {row["id"]: row for row in release["artifact_groups"]}
        self.paths: dict[Path, Mapping[str, Any]] = {}
        anchors = {key: Path(value).resolve() for key, value in release["anchors"].items()}
        for group in release["artifact_groups"]:
            if not group.get("present"):
                continue
            anchor = anchors[group["anchor"]]
            root = (anchor / group["tree"]["anchor_relative_root"]).resolve()
            for row in group["tree"]["files"]:
                self.paths[(root / row["path"]).resolve()] = row

    def source(self, relative: str) -> Path:
        matches = [row for row in self.release["source_files"] if row.get("path") == relative]
        if len(matches) != 1 or matches[0].get("present") is not True:
            raise ValueError(f"audited release does not contain source {relative}")
        path = (Path(self.release["anchors"]["repository"]) / relative).resolve()
        row = matches[0]
        if path.stat().st_size != row["bytes"] or sha256_stable_file(path) != row["sha256"]:
            raise ValueError(f"audited source changed after release freeze: {relative}")
        return path

    def matches(self, group_id: str, name: str) -> list[Path]:
        group = self.groups[group_id]
        if not group.get("present"):
            return []
        anchor = Path(self.release["anchors"][group["anchor"]]).resolve()
        root = (anchor / group["tree"]["anchor_relative_root"]).resolve()
        return [
            (root / row["path"]).resolve()
            for row in group["tree"]["files"]
            if Path(row["path"]).name == name
        ]

    def unique(self, group_id: str, name: str) -> Path:
        paths = self.matches(group_id, name)
        if len(paths) != 1:
            raise ValueError(f"expected one audited {group_id}/{name}, found {len(paths)}")
        return self.verify(paths[0])

    def relative(self, group_id: str, relative: str) -> Path:
        group = self.groups[group_id]
        if not group.get("present"):
            raise ValueError(f"audited release does not contain group {group_id}")
        anchor = Path(self.release["anchors"][group["anchor"]]).resolve()
        root = (anchor / group["tree"]["anchor_relative_root"]).resolve()
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as error:
            raise ValueError(f"relative result escapes audited group {group_id}") from error
        return self.verify(path)

    def verify(self, path: Path) -> Path:
        path = path.resolve()
        row = self.paths.get(path)
        if row is None:
            raise ValueError(f"result is outside the audited release inventory: {path}")
        if path.stat().st_size != row["bytes"] or sha256_stable_file(path) != row["sha256"]:
            raise ValueError(f"audited result changed after release freeze: {path}")
        return path


def _validate_release(manifest_path: Path, audit_path: Path) -> tuple[dict, dict]:
    release = _json(manifest_path)
    observed = canonical_digest(
        {key: value for key, value in release.items() if key != "manifest_sha256"}
    )
    if (
        release.get("inventory_complete") is not True
        or release.get("result_payloads_parsed") is not False
        or release.get("manifest_sha256") != observed
    ):
        raise ValueError("core release manifest is incomplete, result-bearing, or corrupt")
    audit = _json(audit_path)
    if (
        audit.get("complete") is not True
        or audit.get("result_payloads_parsed") is not False
        or audit.get("manifest_sha256") != observed
    ):
        raise ValueError("release audit does not authorize this core manifest")
    return release, audit


def _follow_record(
    files: AuditedFiles, record: Mapping[str, Any], expected: Path, role: str
) -> tuple[Path, dict[str, Any]]:
    path = files.verify(expected)
    if record.get("sha256") != sha256_stable_file(path) or record.get("bytes") != path.stat().st_size:
        raise ValueError(f"{role} disagrees with its completion record")
    return path, _record(path, role)


def _exp09(files: AuditedFiles) -> tuple[dict, dict, list[dict]]:
    completion_path = files.relative(
        "exp09_frozen_network", "completion_manifest.json"
    )
    completion = _json(completion_path)
    if completion.get("complete") is not True or completion.get("confirmatory") is not True:
        raise ValueError("Exp09 completion manifest is not confirmatory and complete")
    natural_path = files.relative("exp09_frozen_network", "natural_results.json")
    natural_path, source = _follow_record(
        files, completion["inputs"]["natural_results"], natural_path, "exp09_natural_results"
    )
    natural = _json(natural_path)
    if natural.get("complete") is not True or natural.get("confirmatory") is not True:
        raise ValueError("Exp09 natural result is not final")
    margin = float(natural["protocol"]["noninferiority_margin"])
    rows = []
    for row in natural["paired"]:
        low, high = map(float, row["kl_ratio_dpsae_to_mse_ci95"])
        rows.append(
            {
                "seed": int(row["seed"]),
                "estimate": float(row["kl_ratio_dpsae_to_mse"]),
                "ci_low": low,
                "ci_high": high,
            }
        )
    passed = bool(completion.get("natural_noninferiority_passed"))
    summary = {
        "experiment": "Exp09",
        "endpoint": "DPSAE/MSE frozen-network KL ratio",
        "estimate": f"{sum(row['estimate'] for row in rows) / len(rows):.6f}",
        "ci_low": "NA",
        "ci_high": f"{max(row['ci_high'] for row in rows):.6f}",
        "gate": f"all seedwise CI highs < {margin:g}",
        "status": "passed" if passed else "failed",
        "scope": "Seedwise confirmatory noninferiority; displayed estimate is the seed mean.",
    }
    return {"available": True, "noninferiority_margin": margin, "records": rows}, summary, [
        _record(completion_path, "exp09_completion_manifest"),
        source,
    ]


def _metric(value: Mapping[str, Any]) -> float:
    metrics = value.get("metrics", value)
    return float(metrics["test_auc"])


def _saebench_rows(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        if len(payload) != 1 or not isinstance(payload[0], Mapping):
            raise ValueError(f"unexpected SAEBench aggregate list schema: {path}")
        payload = payload[0]
    if not isinstance(payload, Mapping):
        raise ValueError(f"unexpected SAEBench aggregate schema: {path}")
    details = payload.get("eval_result_details")
    rows: dict[str, Mapping[str, Any]] = {}
    if isinstance(details, Mapping):
        iterable = details.items()
    elif isinstance(details, list):
        iterable = ((None, row) for row in details)
    else:
        raise ValueError(f"SAEBench aggregate lacks eval_result_details: {path}")
    for key, row in iterable:
        if not isinstance(row, Mapping):
            raise ValueError(f"invalid SAEBench result detail: {path}")
        dataset = row.get("dataset") or row.get("dataset_name") or key
        if not isinstance(dataset, str) or not dataset or dataset in rows:
            raise ValueError(f"missing or duplicate SAEBench dataset identity: {path}")
        rows[dataset] = row
    return rows


def _concept_ladder(
    files: AuditedFiles,
    artifact_records: list[Mapping[str, Any]],
    output_root: Path,
    report: Mapping[str, Any],
) -> tuple[dict, list[dict]]:
    config_path = files.source("configs/exp10_concept_discovery.json")
    config = _json(config_path)
    datasets = [str(value) for value in config["benchmark"]["datasets"]]
    methods = ("mse", "dpsae")
    seeds = [int(value) for value in config["benchmark"]["probe_seeds"]]
    ks = [int(value) for value in config["benchmark"]["ks"]]
    if ks != [1, 2, 5]:
        raise ValueError("Exp10 concept-ladder k grid drift")
    saebench = [row for row in artifact_records if row.get("kind") == "saebench_result"]
    if len(saebench) != len(methods) * len(seeds):
        raise ValueError("passed Exp10 audit lacks the complete 20-file SAEBench fleet")
    values = {
        method: {k: {dataset: [] for dataset in datasets} for k in ks}
        for method in methods
    }
    source_records = [_record(config_path, "exp10_frozen_config")]
    identities: set[tuple[str, int]] = set()
    for row in saebench:
        method, seed = str(row.get("method")), int(row.get("probe_seed"))
        if method not in methods or seed not in seeds or (method, seed) in identities:
            raise ValueError("Exp10 SAEBench method/seed identity drift")
        identities.add((method, seed))
        path = files.verify(output_root / row["path"])
        if sha256_stable_file(path) != row["sha256"]:
            raise ValueError("Exp10 SAEBench result changed after final audit")
        details = _saebench_rows(path)
        if set(details) != set(datasets):
            raise ValueError("Exp10 SAEBench dataset coverage drift")
        for dataset, detail in details.items():
            by_k = detail.get("sae_metrics_by_k")
            if not isinstance(by_k, Mapping):
                raise ValueError("Exp10 SAEBench detail lacks sae_metrics_by_k")
            for k in ks:
                metric = by_k.get(str(k), by_k.get(k))
                if not isinstance(metric, Mapping):
                    raise ValueError(f"Exp10 SAEBench detail lacks k={k}")
                values[method][k][dataset].append(_metric(metric))
        source_records.append(_record(path, f"exp10_saebench_{method}_{seed}"))
    family = config["benchmark"]["family_by_dataset"]
    stats = config["statistics"]

    task_values: dict[str, dict[str, dict[str, float]]] = {method: {} for method in methods}
    for method in methods:
        companion = report["companion_task_metrics"]
        task_values[method]["residual"] = {
            dataset: float(companion[dataset]["original_residual"]["test_auc"])
            for dataset in datasets
        }
        for stage in ("reconstruction", "full_code"):
            task_values[method][stage] = {
                dataset: float(companion[dataset]["methods"][method][stage]["test_auc"])
                for dataset in datasets
            }
        for k in ks:
            stage = f"k{k}"
            task_values[method][stage] = {}
            for dataset in datasets:
                observed = values[method][k][dataset]
                if len(observed) != len(seeds):
                    raise ValueError("Exp10 SAEBench probe-seed coverage drift")
                task_values[method][stage][dataset] = sum(observed) / len(observed)

    records = []
    for method in methods:
        for stage in ("residual", "reconstruction", "full_code", "k5", "k2", "k1"):
            interval = family_block_bootstrap(
                task_values[method][stage],
                family,
                samples=int(stats["bootstrap_samples"]),
                seed=int(stats["bootstrap_seed"]),
                confidence_level=float(stats["confidence_level"]),
            )
            records.append(
                {
                    "method": method,
                    "stage": stage,
                    "estimate": interval["estimate"],
                    "ci_low": interval["lower"],
                    "ci_high": interval["upper"],
                }
            )
    return {"available": True, "records": records}, source_records


def _exp10(files: AuditedFiles) -> tuple[dict, dict, list[dict]]:
    passing: list[tuple[Path, dict]] = []
    for path in files.matches("exp10_concept_discovery", "artifact_audit_final.json"):
        files.verify(path)
        audit = _json(path)
        if (
            audit.get("complete") is True
            and audit.get("passed") is True
            and audit.get("phase") == "final"
            and audit.get("expected_counts") == audit.get("observed_counts")
        ):
            passing.append((path, audit))
    if len(passing) != 1:
        raise ValueError(f"expected one passed Exp10 final audit, found {len(passing)}")
    audit_path, audit = passing[0]
    manifest_path = files.verify(audit_path.parent / Path(audit["manifest_path"]).name)
    if sha256_stable_file(manifest_path) != audit["manifest_sha256"]:
        raise ValueError("Exp10 artifact manifest does not match its passed final audit")
    records = [json.loads(line) for line in manifest_path.read_text().splitlines() if line]
    reports = [row for row in records if row.get("kind") == "advancement_report"]
    if len(reports) != 1:
        raise ValueError("passed Exp10 audit does not identify one advancement report")
    report_record = reports[0]
    report_path = files.verify(audit_path.parent / report_record["path"])
    if sha256_stable_file(report_path) != report_record["sha256"]:
        raise ValueError("Exp10 advancement report changed after final audit")
    report = _json(report_path)
    if report.get("complete") is not True or report.get("checks", {}).get("complete_matrix") is not True:
        raise ValueError("Exp10 advancement report is incomplete")
    if report.get("advance_fresh_confirmation") is not False:
        raise ValueError("this builder is only for the completed Exp10 null path")
    concept, concept_sources = _concept_ladder(files, records, audit_path.parent, report)
    interval = report["primary"]["family_block_interval"]
    summary = {
        "experiment": "Exp10",
        "endpoint": "DPSAE-MSE sparse-probe AUROC",
        "estimate": f"{float(interval['estimate']):.6f}",
        "ci_low": f"{float(interval['lower']):.6f}",
        "ci_high": f"{float(interval['upper']):.6f}",
        "gate": "all frozen pilot advancement checks",
        "status": "failed",
        "scope": "Pilot result; fresh confirmation and autointerp were not opened.",
    }
    return concept, summary, [
        _record(audit_path, "exp10_final_audit"),
        _record(manifest_path, "exp10_final_artifact_manifest"),
        _record(report_path, "exp10_advancement_report"),
        *concept_sources,
    ]


def _exp11(files: AuditedFiles) -> tuple[dict, dict, list[dict]]:
    path = files.relative("exp11_static_matched_nmse", "summary.json")
    value = _json(path)
    screen, confirmation = value.get("screen", {}), value.get("confirmation", {})
    if (
        value.get("complete") is not True
        or screen.get("advance") is not False
        or confirmation.get("status") != "not_run_by_predeclared_gate"
    ):
        raise ValueError("this builder requires the finalized Exp11 null path")
    rule = screen["rule"]
    target, tolerance = float(rule["target_nmse_ratio"]), float(rule["matching_tolerance"])
    records = [
        {
            "candidate": "MSE",
            "method": "mse",
            "nmse_ratio": 1.0,
            "decoder_reduction": 0.0,
            "selected": False,
        },
        {
            "candidate": "DPSAE",
            "method": "dpsae",
            "nmse_ratio": float(screen["dpsae_anchor"]["nmse_ratio"]),
            "decoder_reduction": float(screen["dpsae_anchor"]["decoder_reduction"]),
            "selected": False,
        },
    ]
    for candidate in screen["candidates"]:
        beta = float(candidate["spec"]["loss_weight"])
        records.append(
            {
                "candidate": f"spectral-{beta:g}",
                "method": "spectral",
                "nmse_ratio": float(candidate["nmse_ratio"]),
                "decoder_reduction": float(candidate["decoder_reduction"]),
                "selected": candidate == screen.get("selected"),
            }
        )
    summary = {
        "experiment": "Exp11",
        "endpoint": "matched-NMSE spectral control",
        "estimate": "NA",
        "ci_low": "NA",
        "ci_high": "NA",
        "gate": f"NMSE ratio in [{target - tolerance:g}, {target + tolerance:g}]",
        "status": screen.get("status", "failed"),
        "scope": "No spectral candidate matched the frozen screen; confirmation was skipped.",
    }
    return {
        "available": True,
        "target_nmse_low": target - tolerance,
        "target_nmse_high": target + tolerance,
        "records": records,
    }, summary, [_record(path, "exp11_final_summary")]


def _not_run(experiment: str, gate: str, scope: str) -> dict[str, str]:
    return {
        "experiment": experiment,
        "endpoint": "gated follow-up",
        "estimate": "NA",
        "ci_low": "NA",
        "ci_high": "NA",
        "gate": gate,
        "status": "not-run",
        "scope": scope,
    }


def build(manifest_path: Path, audit_path: Path, output_dir: Path) -> tuple[Path, Path, Path]:
    release, release_audit = _validate_release(manifest_path, audit_path)
    run_root = Path(release["anchors"]["run"]).resolve()
    output_dir = output_dir.resolve()
    if output_dir == run_root or output_dir.is_relative_to(run_root):
        raise ValueError("publication outputs must be outside the audited core run root")
    files = AuditedFiles(release)
    for group_id in ("exp12_fresh_confirmation", "exp13_concept_confirmation", "exp10_autointerp"):
        group = files.groups[group_id]
        if group.get("present") and group.get("tree", {}).get("file_count", 0):
            raise ValueError(f"{group_id} is present; this is not the completed null path")

    frozen, exp09, sources09 = _exp09(files)
    concept, exp10, sources10 = _exp10(files)
    static, exp11, sources11 = _exp11(files)
    rows = [
        exp09,
        exp10,
        exp11,
        _not_run("Exp12", "Exp10 pilot advancement", "Skipped by the frozen Exp10 gate."),
        _not_run("Exp13", "Exp10 pilot advancement", "Skipped by the frozen Exp10 gate."),
        _not_run("Autointerp", "fresh confirmation", "No API labeling was run on pilot candidates."),
    ]
    payload = {
        "schema_version": 1,
        "complete": True,
        "release_manifest_sha256": release["manifest_sha256"],
        "figures": {
            "concept_ladder": concept,
            "frozen_network_noninferiority": frozen,
            "static_nmse_control": static,
        },
        "summary_table": {"available": True, "rows": rows},
    }
    _validate_payload(payload)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload_path = output_dir / "closure_payload.json"
    csv_path = output_dir / "closure_summary.csv"
    atomic_json(payload_path, payload)
    _write_summary(rows, csv_path)
    sources = [
        _record(manifest_path, "core_release_manifest"),
        _record(audit_path, "core_release_audit"),
        *sources09,
        *sources10,
        *sources11,
    ]
    build_manifest = {
        "schema_version": 1,
        "complete": True,
        "core_release_manifest_sha256": release["manifest_sha256"],
        "core_release_audit": release_audit,
        "sources": sources,
        "outputs": [_record(payload_path, "closure_payload"), _record(csv_path, "closure_summary")],
        "builder": _record(Path(__file__), "payload_builder"),
    }
    build_manifest["build_manifest_sha256"] = canonical_digest(build_manifest)
    build_path = output_dir / "closure_payload_manifest.json"
    atomic_json(build_path, build_manifest)
    return payload_path, csv_path, build_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release-manifest", type=Path, required=True)
    parser.add_argument("--release-audit-report", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    _, _, build_path = build(args.release_manifest, args.release_audit_report, args.output_dir)
    print(build_path)


if __name__ == "__main__":
    main()
