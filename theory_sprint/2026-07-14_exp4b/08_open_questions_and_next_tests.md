# Open questions and next tests

The immediate priority is to test the missing bridge from refittable finite-group information to frozen-model behavior without changing the SAE fleet. Mechanism attribution comes next; Fisher and manifolds remain follow-up work.

## Single highest-value next experiment

Run an **evaluation-only, valid-target, equal-effect IOI audit on the existing checkpoints**.

1. Predefine a continuous target that is linear by construction at the audited activation, for example \(y=Xw\) for a frozen direction \(w\) chosen without the final split. Use the 3,072/1,024/2,048 ranking/selection/test partition already frozen by Experiment 4b.
2. Require the original dense activation to pass a preregistered test gate, such as held-out \(R^2\ge0.8\), after choosing ridge only on the 1,024-example selection split. If the gate fails, stop before comparing SAEs.
3. Rank SAE features on the ranking split, select every hyperparameter and feature-count interpolation on the selection split, and report the continuous target once on test.
4. For the causal comparison, interpolate each method to the same held-out IOI effect before comparing natural-text collateral KL. If common support is absent, report that rather than extrapolating.

This one protocol is highest value because it directly tests the strongest failed inference in the current paper—that refittable decoder preservation helps a frozen circuit—while reusing all checkpoints and avoiding another training fleet. It also fixes both defects in the current IOI analysis: the continuous target did not pass its dense gate, and collateral KL was compared at unequal IOI effects.

Expected outcomes are discriminating. Better target preservation with lower equal-effect KL would justify a narrow empirical bridge, though not a universal theorem. Better target preservation without better equal-effect KL would support the theory's refitted-versus-frozen separation. No target improvement would keep the contribution entirely at the task-agnostic operator level.

## Evaluation-only checks on existing checkpoints

| Priority | Check | Information gained | Cost and guardrail |
| ---: | --- | --- | --- |
| 1 | Valid-target, equal-effect audit above | Tests the only missing bridge needed for a stronger functional claim | Existing caches/checkpoints; no training; stop if dense gate or common support fails |
| 2 | Exact-versus-sampled decoder-gradient audit at \(m\in\{1,4,8,16,32\}\) | Measures cosine, norm bias, and variance of the actual normalized self-ratio gradient against identity targets | A few frozen training-shaped batches; cap memory by processing one group/probe bank at a time |
| 3 | Retry the cache-only dense ridge sweep that the sprint's SSH diagnostic could not run | Determines whether the current \(R^2=-3.0968\) is specific to ridge 0.01 | CPU or <2 GB GPU; hard connection timeout; no new cache or model load |
| 4 | Remove the \(n=64\) 128-group cap or match all group sizes to the same token subset | Removes the only confound in the 35%/24%/13% trend | Exact evaluation only; stream groups to avoid memory growth |
| 5 | Report ratio-of-sums, mean-of-group-ratios, and mean-of-batch-ratios on frozen models | Measures whether estimator weighting matters numerically in training-shaped batches | Existing activations; no optimization |

## Short retraining baselines

| Priority | Baseline | Question answered | Minimal design |
| ---: | --- | --- | --- |
| 1 | Same decoder term with BatchTopK, tokenwise TopK, and JumpReLU | Is the gain objective-level or tied to global active-set competition? | One matched \(k=32\) seed for screening, then confirm only a real interaction; identical tokens and exact holdout |
| 2 | Train DPSAE at \(n=64,128,256\) with matched token budgets | Does evaluation group sensitivity change the learned representation, or only the metric? | Reuse selected \(\gamma\); match exact evaluation tokens and group counts |
| 3 | Small \(\gamma\) frontier for MSE/DPSAE and the strongest static baseline | Is the 24% gain still distinct at matched NMSE? | Few weights, seed 0 selection, untouched confirmation seeds only after a frontier separation appears |

These retraining jobs should remain bounded: estimate checkpoint and activation-cache size first, retain only the best and final checkpoints, stream data, and stop a candidate on the preregistered NMSE/decoder frontier. The supplied RTX 6000 Pro is ample for these existing GPT-2-small configurations, but storage should be checked before launching a fleet.

## Main-track stretch work

- Replicate the exact identity-target result at a second GPT-2 layer and one non-GPT-2 model. This is the minimum evidence needed to move from repeatability to architectural generality.
- Characterize sparse allocation empirically by measuring which sample-space eigenmodes each active-set pattern preserves, then test whether a simple statistic predicts the DPSAE–MSE gap across groups. A theorem should follow a stable empirical law, not precede it.
- Solve or approximate the noncommuting weighted PSD-contraction problem only if a structured task prior becomes empirically useful. It is mathematically clean but currently detached from the primary isotropic result.

## Follow-up-paper directions

- A Fisher-pullback SAE objective would target frozen local output behavior and should be compared directly with output-KL end-to-end SAEs. It is a different paper unless the valid-target audit establishes a bridge.
- Activation-manifold or block-sparse geometry becomes justified only if linear targets pass while curved intervention paths show a residual failure. Current 4b results supply no such evidence.
- A population theory for token groups requires an explicit stochastic process, sequence dependence assumptions, and an asymptotic regime with controlled \(n/d\). The present finite-group result should not be stretched into that claim.

## Gaps that could still alter the central statement

The representation-level 24% result is robust to fresh exact targets, but it is still conditional on one model, layer, corpus family, SAE architecture, and selected objective weight. A traceability gap also remains: the final IOI artifact does not record its exact code revision, and the complete small baseline/IOI JSONs were remote-only at sprint start. Neither issue invalidates the audited finite-group result, but both must be fixed before a fully reproducible paper release.
