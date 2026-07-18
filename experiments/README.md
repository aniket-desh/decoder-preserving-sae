# Experiment entry points

The experiment directory contains scientific runners rather than infrastructure. Cloud allocation, backup, and monitoring helpers are intentionally kept out of the public surface; the retained shell scripts only compose experiment stages needed to reproduce a reported result.

## Paper-facing workflows

| Entry point | Config | Role |
| --- | --- | --- |
| `exp01_isotropic_spectral.py` | `configs/exp01_isotropic_spectral.json` | Rank-relaxation and sparse isotropic spectral mechanics. |
| `exp02_structured_prior.py` | `configs/exp02_structured_prior.json` | Controlled structured-prior result across paired seeds. |
| `exp02_prior_weight_sweep.py` | `configs/exp02_structured_prior.json` | Task-prior sensitivity sweep used in the appendix. |
| `exp03_estimator_scaling.py` | `configs/exp03_estimator_scaling.json` | Estimator accuracy, ridge calibration, and systems checks. |
| `paper_closure.py` | `configs/paper_closure.json` and `configs/exp04b_confirmatory.json` | GPT-2 gamma selection, paired training, and exact held-out readout evaluation. |
| `exp08_language_evidence.py` | `configs/paper_closure.json` | Robustness, frozen-fidelity diagnostic, and objective-overhead summaries for the selected checkpoints. |
| `exp09_frozen_network.py` | `configs/exp09_frozen_network.json` | Confirmatory natural-text output-KL noninferiority and secondary IOI endpoints. |
| `exp10_concept_discovery.py` | `configs/exp10_concept_discovery.json` | Matched Pythia sparse-concept pilot and frozen advancement decision. |
| `exp11_static_matched_nmse.py` | `configs/exp11_static_matched_nmse.json` | Preregistered matched-NMSE static spectral feasibility screen. |

The main self-contained entry points accept `--smoke` or can be inspected with `--help`. The language workflows are staged because cache preparation, model training, evaluation, and finalization have different artifact and fresh-data boundaries.

## Appendix and negative-result controls

- `exp04_ioi_mechanism.py` and the `exp04b_*` runners contain the IOI concentration, attribution, and confirmatory controls.
- `exp05_decoder_advantage_discovery.py`, `exp05_semantic_recurrence.py`, and `exp05_finalize_semantic_review.py` retain the open-ended semantic discovery branch and its failed transfer audit.
- `exp06_generality.py` contains the single-seed layer/model screens; those screens are diagnostic rather than a cross-model claim.
- `exp07_advantage_spectrum.py`, `exp07_gradient_fidelity.py`, and `exp07_jumprelu_calibration.py` contain the task-spectrum, estimator-gradient, and failed JumpReLU feasibility checks.
- `plot_exp04b.py` is the retained plotter for the corresponding appendix artifacts.

These runners stay in the release because the manuscript reports their null, mixed, or feasibility outcomes. The public tree omits conditional stages that the Exp10 gate never authorized.

## Runner composition

The small top-level runner set maps directly onto the paper:

- `scripts/run_exp01_local.sh`, `scripts/run_exp02_local.sh`, and `scripts/run_exp03_local.sh` run the local controlled studies.
- `scripts/run_exp08_gpu_runpod.sh` and `scripts/run_exp08_synthetic_runpod.sh` reproduce the paper-facing GPT-2 and structured-prior closure fleets from immutable inputs.
- `scripts/run_exp09_frozen_network_runpod.sh` runs the fresh frozen-network boundary test.
- `scripts/run_exp10_timing_smoke_a40.sh`, `scripts/run_exp10_concept_4xa40.sh`, and `scripts/run_steps1_4_autonomous_runpod.sh` preserve the exact resource-gated concept pilot.
- `scripts/run_exp11_static_matched_nmse_runpod.sh` runs the static-control screen and its conditional finalization.

Every other retained file under `scripts/` validates, aggregates, or renders artifacts produced by these workflows.
