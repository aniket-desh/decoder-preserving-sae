# Reproducibility guide

Reproduction has three layers: versioned code and configs in Git, large immutable inputs and outputs in the external artifact bundle, and compact paper-facing summaries derived from the audited bundle. Keeping these layers separate makes the public repository readable without weakening provenance.

## What is in Git

The reusable implementation lives in `src/dpsae/`. Scientific entry points live in `experiments/`, and every retained runner has an explicit config or artifact contract. `results/paper_audits/` contains small machine-readable summaries for appendix checks; it is not a substitute for the per-example and checkpoint artifacts.

The public tree intentionally excludes:

- model checkpoints, optimizer state, activation caches, tokenized corpora, and per-example outputs;
- cloud-provider backup, monitoring, tmux, and retry helpers;
- conditional fresh-Pythia, concept-confirmation, context-mining, and API-labeling implementations, because the preregistered concept gate blocked those stages before they ran.

The immutable closure bundle retains the exact experiment-time source archive and manifests. The cleaned public revision therefore remains easy to navigate, while the archived revision still supports byte-level audit of the executed fleet.

## Environment

```bash
uv sync --extra dev --extra experiments
```

The `experiments` extra adds Hugging Face, scikit-learn, and joblib dependencies used by the language-model and probe workflows. The concept pilot additionally requires a separate SAEBench checkout pinned to commit `8042bb3828c6340da8d12062324e92b2077c571c`; the benchmark is not vendored or silently upgraded.

Run the complete retained test and lint surface with:

```bash
bash scripts/check.sh
```

## Reproduction levels

1. **Local controlled experiments.** `scripts/run_exp01_local.sh` and `scripts/run_exp02_local.sh` regenerate the self-contained synthetic studies. The estimator runner can be invoked with `scripts/run_exp03_local.sh`; its activation-backed cells require a local GPT-2 cache, while its synthetic cells do not.
2. **Language-model smoke tests.** Each language runner exposes smoke, validation, or status stages that operate on already-opened data. These verify checkpoint loading, activation hooks, reconstruction parity, artifact schemas, and memory bounds without opening a fresh confirmatory split.
3. **Full paper reproduction.** Restore the exact external inputs, validate every hash, and run the paper-facing GPU and boundary-test runners. These commands intentionally refuse dirty revisions or mismatched hardware and do not download or substitute checkpoints implicitly.

The mapping from retained entry points to configs and scientific status is in `experiments/README.md`.

## Expected external inputs

The full GPT-2 workflow expects the following classes of inputs under `artifacts/` or paths passed explicitly on the command line:

- a tokenized FineWeb cache with absolute token-range metadata;
- GPT-2 activation calibration statistics;
- paired MSE/DPSAE checkpoints and their training completion records;
- static-control calibration and baseline evaluation records;
- checkpoint, config, corpus, and repository SHA-256 manifests.

The exact default paths are declared by `configs/paper_closure.json`, `configs/exp09_frozen_network.json`, and the retained runners. Do not reconstruct a missing path by guessing: restore the record named by the artifact manifest and verify its bytes first.

## Auditing a closure bundle

Given a restored run tree, build and audit its result-blind inventory before parsing outcomes:

```bash
PYTHONPATH=src python3 scripts/finalize_arxiv_experiment_closure.py build \
  --run-root /path/to/run \
  --manifest /path/to/core_release.json

PYTHONPATH=src python3 scripts/finalize_arxiv_experiment_closure.py audit \
  --run-root /path/to/run \
  --manifest /path/to/core_release.json \
  --report /path/to/core_release_audit.json
```

Then derive the publication payload outside the immutable run tree and render it:

```bash
PYTHONPATH=src python3 scripts/build_arxiv_closure_payload.py \
  --release-manifest /path/to/core_release.json \
  --release-audit-report /path/to/core_release_audit.json \
  --output-dir /path/to/publication_payload

PYTHONPATH=src python3 scripts/render_arxiv_closure_figures.py \
  --payload /path/to/publication_payload/closure_payload.json \
  --payload-manifest /path/to/publication_payload/closure_payload_manifest.json \
  --release-manifest /path/to/core_release.json \
  --output-dir /path/to/rendered_figures
```

The builder rejects a non-null closure path, missing or changed source artifacts, an incomplete concept matrix, or a release manifest that does not re-audit. The renderer independently checks payload, manifest, source, style, and font identities before writing figures.

## Interpreting the null path

The final concept pilot was eligible at its matched operating point but failed its frozen advancement rule. Fresh Pythia training, three-pair concept confirmation, context mining, and API feature labeling were therefore not run. Their absence is a result of the protocol, not missing public code. The retained Exp10 runner, artifact auditor, config, and statistics module reproduce the pilot and its negative advancement decision.
