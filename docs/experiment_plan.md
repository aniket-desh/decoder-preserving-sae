# Experiment plan

**Status:** canonical execution contract for the remaining arXiv experiments and the subsequent main-track hardening pass, updated July 16, 2026.

This document records the decisions, gates, costs, and execution order that must stay fixed while the remaining experiments run. `docs/experiments_todo.md` remains the theory-to-evidence map and `docs/theory_todo.md` remains the theorem-order checklist, but this file is the operational reference for concept discovery, automated feature descriptions, frozen-network confirmation, evidence scaling, and RunPod execution.

Any change to a frozen choice below must be recorded as a dated amendment before opening the affected result. The amendment must say what changed, why it changed, which artifacts or metrics had already been inspected, and whether the change makes the affected analysis exploratory. Silent deviations, post-hoc threshold changes, and extending a grid after seeing the target metric are not allowed.

Quick map:

- Sections 1--3 record the claim, completed evidence, negative evidence, and evaluation hierarchy.
- Section 4 is the Pythia/SAEBench concept-discovery protocol and advancement gate.
- Section 5 is the GPT-5.4 feature-description, validation, provenance, and cost protocol.
- Section 6 is the fresh GPT-2 frozen-network confirmation.
- Section 7 separates arXiv closure from optional main-track evidence scaling.
- Sections 8--9 define the three- or four-A40 RunPod topology and cost envelope.
- Sections 10--15 define artifacts, paper integration, planned figures and retained raw fields, execution order, unresolved freeze items, and dated decisions.

## 1. Scientific scope and completed evidence

The paper's narrow claim is that adding Harvey-style decoder disagreement to a matched sparse-autoencoder objective can preserve held-out linearly decodable information at fixed sparsity. The activation-manifold extension remains out of scope unless the core experiments show that curvature is necessary.

The key theoretical correction is that isotropic decoder preservation does not identify semantically important low-variance directions under the rank-constrained relaxation. It has the same singular-direction ordering as MSE with a ridge-saturated distortion. Low-variance task selection is therefore tested separately using a structured prior, while the isotropic experiment studies spectral saturation under a genuinely sparse, nonorthogonal bottleneck.

The main-text evidence is organized around the phenomenon rather than the original five-experiment chronology. The conceptual construction establishes that MSE and sparsity leave taskwise fidelity underdetermined; the rank relaxation explains the isotropic spectral limit; the structured task-prior generator tests task-selective sparse allocation; and the GPT-2 experiment tests held-out readout fidelity at matched reconstruction quality. Estimator audits, architecture checks, gradient diagnostics, and the failed IOI, open-ended semantic, and JumpReLU branches are controls or appendix evidence.

The locked positive results are:

- The structured-prior synthetic reduces median held-out protected-task distortion by 25.3% versus paired MSE across 10/10 seeds without an observed NMSE penalty.
- The clean GPT-2-small block-8 confirmation freezes $\gamma=0.03125$ on a disjoint selection range, then reduces exact decoder distortion by 10.61%, 11.38%, and 10.84% across three 100M-token paired seeds while matching or slightly improving NMSE.
- A one-seed tokenwise-TopK control retains exactly 32 activations per token and yields an 11.50% decoder-distortion reduction at a 0.81% NMSE cost.
- The evaluation-geometry audit, task-spectrum recomputation, static controls, and isolated objective-overhead measurements are complete.

These results establish protected refittable linear-readout fidelity. They do not establish that DPSAE learns better standard concepts or that the original frozen downstream network processes DPSAE reconstructions at least as faithfully as matched MSE reconstructions. Those are the two required external closure experiments.

## 2. Existing negative and diagnostic evidence

The completed JumpReLU validity study uses one-time first-batch quantile initialization, FP32 scores and threshold gradients, and rectangular-kernel pseudo-gradients. It trains paired objectives for 25,001,984 tokens at shared sparsity-loss weights 2, 4, 8, and 16. The decoder-blind selector reads only held-out inference $L_0$, late-window drift, finite state, threshold health, dead-feature count, and run provenance. A weight can advance only if both $L_0$ confidence intervals lie in $[30.4,33.6]$, the paired mismatch is at most 1.6 activations, late drift is acceptable, and the maximum dead fraction is at most 10%.

No JumpReLU setting advances. The DPSAE/MSE $L_0$ pairs are $37.73/38.55$, $37.38/36.93$, $36.10/36.11$, and $35.88/35.61$ as the weight increases, and maximum dead fractions are 48.5--54.4%. NMSE, decoder distortion, and language-model loss remain sealed, so this is a controller-feasibility failure rather than a negative DPSAE effect estimate. The grid must not be extended after seeing the full-horizon result.

The single-seed GPT-2 block-4 screen passes its decoder-distortion gate, while the Pythia-160M-deduped block-8 screen reaches a 9.13% reduction and misses the frozen 10% gate. Both remain diagnostic rather than supporting a cross-layer or cross-model generality claim. The IOI feature-concentration and causal-specificity experiments, open-ended semantic recurrence experiment, and JumpReLU branch also failed their positive gates and must be reported as scope-defining negative evidence rather than hidden or rehabilitated post hoc.

The completed frozen-network diagnostic uses the three clean GPT-2 checkpoint pairs on the once-opened 195M--200M FineWeb range. It measures output KL, next-token loss recovered, cross-entropy increase, top-1 agreement, next-token accuracy, same-split NMSE, and inference $L_0$. DPSAE has lower point-estimate output KL in all three seeds, with intervals excluding zero for seeds 1 and 2, but the run did not predeclare a primary metric, noninferiority margin, or task-level endpoint. It is review-only and may inform sample size, not confirmatory inference.

The matched-NMSE static spectral protocol is now frozen in `configs/exp11_static_matched_nmse.json`. Its one-seed, 25M-token screen compares spectral coefficients $\beta\in\{2,4,8,16,32\}$ against freshly co-trained MSE and $\gamma=0.25$ DPSAE anchors, targets an NMSE ratio of $1.07\pm0.01$, and breaks equally close matches toward the smaller coefficient. It advances to three paired 100M-token seeds only if the selected spectral point's decoder reduction is within two percentage points of the co-trained DPSAE anchor. The archived $\gamma=0.03125$ fleet is hash-checked for provenance but is not mislabeled as the high-weight comparison.

## 3. Evaluation hierarchy

The remaining experiments must keep three questions separate:

1. **Information preservation:** Can a dense probe recover a labeled concept from the full reconstruction or full sparse code?
2. **Feature concentration:** Can the same concept be recovered from only one, two, or five SAE features?
3. **Frozen-network compatibility:** Does the original downstream network behave similarly when the activation is replaced by the full SAE reconstruction?

An improvement at one level is not evidence for the others. Automated feature descriptions support a replicated feature-concentration result; they cannot rescue a null sparse-probe result. Frozen-network output KL tests the original downstream computation; it is not another estimator of refittable decoder distortion.

## 4. Experiment A: standard concept discovery

### 4.1 Pilot model and checkpoint status

The pilot reuses the existing matched MSE/DPSAE pair on `EleutherAI/pythia-160m-deduped`, residual stream after block 8, width 16,384, target $L_0=32$, and frozen decoder weight $\gamma=0.03125$. Both SAEs were trained for 25M tokens. This pair is an evaluation-only pilot and can never count as one of the fresh confirmatory training seeds after it has been used to decide whether the experiment advances.

Before evaluation, retrieve the exact checkpoint payloads and provenance from the private paper-closure backup, then verify model revision, activation site, calibration statistics, width, sparsity mechanism, decoder orientation, and all checkpoint hashes. The adapter must reproduce the repository's native code, reconstruction, NMSE, and $L_0$ within declared floating-point tolerances before any benchmark result is inspected.

### 4.2 Benchmark freeze

The standardized sparse-probe evaluation is pinned to SAEBench commit `8042bb3828c6340da8d12062324e92b2077c571c`, using the current `sparse_probing_sae_probes` interface. The run uses:

- the exact `sae-probes` dataset list exported by that commit, with the list and its hash saved in the run manifest;
- the `normal` benchmark setting;
- L1 feature selection;
- $k\in\{1,2,5\}$;
- non-binarized latent activations;
- the full residual-stream logistic-regression baseline, evaluated once per seed and task by the companion evaluator rather than duplicated inside the SAEBench wrapper;
- ten preregistered probe seeds, with seed-specific output directories;
- a frozen missing-result rule and dataset-family map recorded in the config before the pilot starts.

The pinned SAEBench wrapper supplies the primary $k$-sparse SAE probes, but its optional original-residual baseline is disabled because the companion evaluator retains the identical unfiltered L2 logistic-regression baseline once per seed and task. The companion evaluator must use exactly the same examples, splits, targets, preprocessing, and probe implementation to evaluate:

- original residual activations;
- the full MSE and DPSAE reconstructions;
- the full MSE and DPSAE sparse codes;
- the $k=1,2,5$ selected-feature representations.

The shared model-activation cache must be generated once, hashed, and opened read-only by all evaluation processes. MSE and DPSAE must never receive different examples, splits, or cached base activations.

The runtime contract is also frozen before outcomes are opened. A static call-graph audit counted 21,470 `find_best_reg` calls in the unoptimized plan; disabling the duplicated SAEBench residual baseline removes 2,260 calls and leaves 19,210 without changing any task, seed, $k$, representation, or regularization choice. The remaining duplication is intentional: the SAEBench result remains the primary standardized artifact, while an exact refit records held-out predictions and weights for provenance. Four workers receive immutable sparse shards of five method-seed jobs each and companion shards of 3/3/2/2 seeds, run every sparse job before their companion jobs, and load the two adapters only once per process. Each process receives `floor(effective CPU count / 4)` BLAS/OpenMP threads, where effective CPU count is the smaller of the process-affinity count and the floored finite cgroup CPU quota.

Before the paid concept fleet starts, one A40 must generate the shared cache, after which four timing processes must reproduce the fleet's CPU/GPU topology with nonreport seed `2027071799`. Each process validates the sealed environment, loads both adapters, and writes a ready artifact before an atomic barrier releases all four onto the same eight tasks. The tasks are selected using dataset size and frozen manifest order only, with two from each size quartile; worker reports may contain only opaque slots, train/test sizes, stage times, peak RSS/GPU memory, cgroup CPU-stat deltas, and timing provenance, never dataset names or concept-facing outputs. The finalizer rejects missing or duplicate workers, resource-identity drift, unequal task selections, excessive start skew, and isolated or pooled timing schemas.

Full BatchTopK codes are passed to sklearn as exact-value SciPy CSR matrices, exposing their structural zeros without changing any value. All five companion L2 representations--the original residual, two reconstructions, and two full codes--preserve upstream `find_best_reg`'s ten C values, CV splits, fresh cold `LogisticRegression` per C and fold, `lbfgs` objective, seed, tolerance, iteration limit, original-order `np.argmax` tie rule, and cold selected-C refit on the exact seeded full-train shuffle. Their 50 independent validation candidates are dispatched through one retained representation-major process pool with at most eight jobs per worker, regrouped before selection, and followed by five independent selected-C refits through that pool; nested numerical-library threads are capped at one. This moves the original-residual and reconstruction fits from the eight-thread parent into one-thread loky children, so production validation must compare every selected C, coefficient, intercept, class, decision score, prediction, and metric against the pinned upstream function before timing may begin. The local independent reference retains the coefficient-path counterexample and counts exactly 55 fresh estimators for the common one-fold case. The simultaneous smoke therefore exercises up to 32 cold-C processes under the pod's 32.3-core cgroup quota without repeatedly draining the pool between short grids. Each timing worker is projected separately using its own sparse method and frozen 3/3/2/2 companion-seed count. Its `parent_peak_rss_mib` field is explicitly parent-only; the retained resource watcher records pod storage, cgroup memory, OOM counters, and GPU memory while loky children are alive. The authoritative wall-time projection adds the measured cold-cache duration and the maximum pre-barrier initialization once to the slowest worker's p95 workload with 30% headroom; rows are never pooled or averaged across workers. Concept workers are gated on a passed schema-v7 report at or below three pod-hours, and every superseded or isolated runtime schema is rejected even if its other fields pass.

If the immutable 113-task cache was generated before the final runtime config and its manifest lacks an in-process generation timer, do not regenerate it or time a warm validation pass as though it were cold generation. `record-cold-cache-timing` must deterministically build an external provenance JSON from the original `cache_ready.json`, model-cache path, exact start/end Unix seconds, and an independently supplied expected duration. The command computes the source-manifest SHA-256 and canonical digest of all recorded cache-file hashes, rejects any elapsed-time mismatch, and lets `timing-preflight` adopt that exact manifest into a fresh output root. The smoke report records the external provenance path, SHA-256, 1,506-second duration, source-cache hash, model-cache path, and file-hash digest, and its projection uses the 1,506 seconds rather than the warm adoption time.

### 4.3 Checkpoint eligibility

The pilot is scientifically interpretable only if the same evaluation split confirms matched operating quality:

- DPSAE/MSE activation-NMSE ratio at most 1.01;
- inference $L_0$ difference within 5% of the target and no material method mismatch;
- finite activations and reconstructions;
- no adapter-dependent decoder renormalization that changes native model behavior;
- no benchmark-specific tuning based on concept labels.

If the existing Pythia pair fails the matched-quality gate, its concept result is diagnostic. Any replacement gamma must be selected on a disjoint natural-text range using only NMSE, $L_0$, and decoder-distortion criteria, then frozen before concept data are opened.

### 4.4 Primary and secondary estimands

For benchmark task $j$, let $A^{m}_{j,k}$ be held-out AUROC for method $m\in\{\mathrm{MSE},\mathrm{DPSAE}\}$ using $k$ selected SAE features. The pilot and confirmation primary statistic is the macro-average paired $k=5$ difference

$$
\Delta_5 = \frac{1}{J}\sum_{j=1}^{J}
\left(A^{\mathrm{DPSAE}}_{j,5}-A^{\mathrm{MSE}}_{j,5}\right).
$$

Average the ten probe seeds within each task first, then form the macro-average. The confidence interval must resample the frozen concept-family blocks rather than treat closely related task variants as independent. Probe-reseed variability is reported separately from task-family variability.

Secondary metrics are:

- macro AUROC differences at $k=1$ and $k=2$;
- held-out accuracy and F1 at $k=1,2,5$;
- original-activation, full-reconstruction, and full-code probe performance;
- per-task and per-family paired differences;
- the excess sparse gain

$$
\Delta^{\mathrm{excess}}_5 =
\Delta^{\mathrm{sparse}}_5-\Delta^{\mathrm{full\ code}},
$$

which asks whether a sparse-feature gain exceeds any general improvement in information retained by the full code.

The primary aggregate includes every successfully completed dataset in the frozen suite under the preregistered missingness rule. Any original-activation recoverability filter used for secondary analysis must be fixed before MSE/DPSAE results are opened and must depend only on the original activations.

### 4.5 Pilot advancement gate

The pilot advances only if all checkpoint-eligibility conditions pass and:

1. the family-block-bootstrap 95% lower bound for $\Delta_5$ is above zero;
2. the point estimate exceeds 0.005 AUROC;
3. the point estimate exceeds twice the probe-reseed standard error;
4. the sign is not produced by one benchmark family while the remaining families are null or negative;
5. adapter and cache audits pass without benchmark-specific repair.

A pilot that does not advance closes the standard-concept question as a clean null at this checkpoint. Do not tune gamma, feature count, task subsets, regularization, or prompt labels against a failed pilot.

### 4.6 Training maturity and fresh confirmation

The 25M-token pair is adequate for screening but is not automatically adequate for a paper-facing concept claim. If the pilot advances, train three fresh paired Pythia seeds at a common, concept-blind maturity budget. Save checkpoints at 25M, 50M, 100M, 250M, and, if supported by genuinely fresh activation streaming, 500M tokens.

Before training, freeze a common final checkpoint rule based only on held-out natural-text NMSE, exact decoder distortion, $L_0$, dead-feature rate, and whether those training diagnostics have plateaued. Do not use concept scores or automated labels to choose the training duration. Do not create a nominal 500M-token run by cycling a small activation cache enough times to overfit it; record unique corpus exposure and cache reuse separately.

The existing pilot pair is excluded from confirmation. The confirmatory claim requires:

- positive $\Delta_5$ for all three fresh checkpoint pairs;
- median seedwise $\Delta_5\geq0.005$;
- a pooled family-block interval excluding zero under the frozen aggregation;
- the same matched-NMSE and matched-$L_0$ gates in every seed;
- Holm correction across predeclared concept families;
- individual concepts treated as descriptive or controlled with a declared FDR procedure.

The trained checkpoint pair is the replication unit for method generality; probe examples and probe reseeds are not additional SAE-training replicates.

Every fresh Pythia pair must also receive the paper's exact decoder-distortion evaluation and the frozen-network natural-text evaluation. This converts a successful Pythia confirmation into a genuine second-model replication of the central phenomenon rather than a concept-only side experiment.

### 4.7 Optional sparsity bracket

The confirmatory setting remains width 16,384 and $L_0=32$. For main-track robustness, one paired seed may bracket it at target $L_0\in\{16,64\}$ using selection rules frozen before training. These are operating-point sensitivity checks, not a new scaling-law claim, and they do not replace the three $L_0=32$ seed pairs.

## 5. Automated feature descriptions

### 5.1 Role and candidate selection

Automated descriptions are supporting evidence and run only for feature-task associations whose $k$-sparse advantage replicates across fresh checkpoint pairs. Candidate features are selected deterministically from the frozen sparse-probe weights, with equal MSE and DPSAE candidate budgets. Feature IDs that recur across tasks are deduplicated without changing their selection provenance.

The labeler is blinded to method, seed, feature rank, benchmark outcome, and which method won. Random opaque IDs replace semantic checkpoint names in prompts.

### 5.2 Context construction

Each feature receives a discovery context set containing approximately:

- 20 high-activation contexts stratified across documents and high-activation quantiles;
- 10 middle-quantile positive contexts so the explanation is not defined only by extreme lexical examples;
- 10 near-miss or token-matched negative contexts;
- concise activation traces and token positions without revealing method identity.

A disjoint held-out context set is archived before labeling and never shown to the explanation model. Context sampling, truncation, document deduplication, activation thresholds, and negative matching must be versioned and hashed.

### 5.3 Labeling models and schema

The primary explanation model is `gpt-5.4-mini-2026-03-17` with reasoning effort `low` and structured output. The schema contains:

```json
{
  "short_label": "expressing uncertainty about a prediction",
  "description": "Activates when the speaker qualifies or weakens a prediction...",
  "positive_evidence": ["might", "probably", "unclear whether"],
  "counterevidence": ["does not generally activate on factual uncertainty questions"],
  "specificity": "medium",
  "polysemantic": false,
  "alternative_labels": ["hedged prediction", "epistemic qualification"]
}
```

The model may return a self-assessed confidence or coherence field for triage, but that field is not scientific evidence. Label validity comes from held-out scoring.

`gpt-5.4-nano-2026-03-17` may normalize labels, map them into a fixed ontology, shorten labels, deduplicate near-synonyms, and perform elementary label-context entailment. It is not the primary explanation model. Ambiguous, unstable, or polysemantic features are adjudicated by `gpt-5.4-2026-03-05` or blinded human review. GPT-5.6 Luna is not the escalation model because it is positioned as a cost-sensitive, approximately nano-tier option.

Twenty percent of candidates receive a second primary-label call with an independently resampled discovery context set. This estimates explanation stability rather than trusting one prompt realization.

### 5.4 Held-out validity

Report:

- discrimination between held-out activating and matched-negative contexts;
- simulation performance when a separate evaluator predicts activation from the explanation;
- label stability across the two context resamples;
- specificity and polysemanticity under a frozen rubric;
- blinded human preference or correctness judgments on a 100--300-feature validation set.

The evaluator must not see method identity or the original probe advantage. If the same model family generates and scores explanations, report that dependence and include human calibration on the validation subset.

### 5.5 API provenance and cost

Archive raw request and response JSONL, request IDs, exact model snapshots, reasoning effort, prompt and schema hashes, feature/context IDs, token counts, retries, and code revision. API keys remain in environment secrets and never enter artifacts or Git.

At 4,000 input tokens and 150 output tokens, one GPT-5.4 mini explanation costs approximately $0.003675 at standard rates or $0.0018375 through the Batch API. Planning examples are:

| Primary-label calls | Standard | Batch |
|---:|---:|---:|
| 300 | $1.10 | $0.56 |
| 600 | $2.21 | $1.10 |
| 1,200 | $4.41 | $2.21 |
| 32,768 | $120.42 | $60.21 |
| 98,304 | $361.27 | $180.63 |

The intended plan is 300--600 unique replicated candidates with resampling only where declared, keeping primary labeling plus nano cleanup and hard-case adjudication below a $10 planning envelope. Labeling every latent is affordable in token terms but scientifically wasteful because it creates an unreviewable annotation mass.

Generate and archive feature contexts on GPU, terminate the GPU pod, then submit asynchronous Batch API jobs from local or CPU compute. Do not pay for idle GPUs during the API's turnaround window.

Pricing references checked July 16, 2026:

- https://developers.openai.com/api/docs/models/gpt-5.4-mini
- https://developers.openai.com/api/docs/models/gpt-5.4-nano
- https://developers.openai.com/api/docs/models/gpt-5.4
- https://developers.openai.com/api/docs/models/gpt-5.6-luna
- https://developers.openai.com/api/docs/pricing
- https://developers.openai.com/api/docs/guides/batch

## 6. Experiment B: confirmatory frozen-network compatibility

### 6.1 Checkpoints, data, and controls

Reuse the three fresh GPT-2-small block-8 MSE/DPSAE confirmation pairs at $\gamma=0.03125$. No new SAE training is needed.

Prepare a fresh immutable FineWeb range beginning at absolute token 200M. The intended reserved range is 200M--210M, from which the confirmatory natural-text sample contains 2,048 deterministic length-256 sequences. The old 195M--200M diagnostic informs sample size only. Smoke tests must use old or synthetic data, and the fresh range is opened once after the implementation, metric definitions, margin, seeds, and artifact schema pass review.

The frozen downstream model receives these conditions:

- original forward pass with no hook;
- identity-hook plumbing control;
- mean-activation ablation;
- full MSE-SAE reconstruction for each seed;
- full DPSAE reconstruction for each paired seed.

All downstream weights remain fixed. Record same-split activation NMSE and inference $L_0$ so output differences cannot be separated from operating quality.

### 6.2 Primary estimand and noninferiority margin

For seed pair $s$, define

$$
R_s =
\frac{\operatorname{KL}(p_{\mathrm{orig}}\Vert p_{\mathrm{DPSAE},s})}
     {\operatorname{KL}(p_{\mathrm{orig}}\Vert p_{\mathrm{MSE},s})}.
$$

The primary claim is noninferiority at margin 1.01: DPSAE may increase output divergence by no more than 1% relative to its paired MSE SAE. The margin reuses the paper's existing 1% matched-quality tolerance and is fixed independently of the favorable KL point estimates on the exploratory split.

Use 10,000 sequence-level paired bootstrap resamples. Noninferiority requires the upper 95% confidence bound for $R_s$ to be below 1.01 in all three checkpoint pairs. A point estimate or interval below 1 is a secondary superiority result; it is not required for closure.

The sequence is the resampling unit within a trained pair. Report seedwise intervals rather than treating the three training seeds as a large population. A hierarchical or median-across-seeds summary may be descriptive, but it must not obscure a seed that fails noninferiority.

### 6.3 Secondary natural-text metrics

Report paired DPSAE-minus-MSE differences for:

- cross-entropy increase relative to the original model;
- next-token loss recovered relative to mean ablation;
- top-1 agreement with the original model;
- next-token accuracy;
- activation-NMSE ratio;
- inference-$L_0$ difference.

These metrics remain secondary because output KL is smoother and directly measures compatibility with the original predictive distribution.

### 6.4 Task-level endpoint

Reuse the repository's IOI generation and scoring infrastructure on a newly generated, frozen prompt set with recorded generator seed, names, templates, and counterfactual construction. The task-level compatibility endpoints are:

- absolute error in correct-versus-incorrect-name logit difference relative to the original model;
- agreement with the original model's preferred answer;
- ordinary IOI accuracy as a secondary behavioral metric.

Use a paired prompt-level bootstrap. This is a full-reconstruction compatibility test and does not reopen the failed claim that a small DPSAE feature set isolates the IOI mechanism more causally.

### 6.5 Interpretation

- If all three pairs pass KL noninferiority, the paper may say that the decoder-fidelity gain does not produce more than the declared 1% relative output-KL degradation under the tested frozen network.
- If intervals also favor DPSAE, report frozen-network superiority as a secondary empirical result without claiming the refittable-readout objective directly optimized it.
- If any pair fails, report the trade-off honestly: the objective improves refittable readouts but is not confirmed compatible with the frozen downstream computation at the declared margin.
- Mixed task-level outcomes remain mixed; do not replace the natural-text primary with a favorable task result.

## 7. Evidence scale for arXiv and main-track review

The 2026 Mechanistic Interpretability Workshop spotlights do not establish a universal three-model requirement. They scale along the axis named by the claim:

- Predictive Concept Decoders uses one Llama-3.1-8B-Instruct subject model, then varies data budget and downstream task because scaling and application breadth are the claim.
- The variant-specific crosscoder study uses one Qwen3-4B/base-vs-GRPO pair, three seeds, capacity sweeps, twelve base-vs-base controls, and a magnitude-matched causal ablation because pairing artifacts and task causality are the threats.
- The RL-induced tool-use study uses one Qwen2.5-3B model pair and one task despite training 48 crosscoders; its breadth is a hyperparameter search, not cross-model confirmation.
- The Dark Subspace study spans several architectures, but its controlled Pythia cell is the main causal evidence and its cross-architecture cells are explicitly single-seed and exploratory.
- Size Doesn't Matter tests several Qwen sizes, Gemma, Pythia, layers, selectors, sparsities, and three training seeds because it recommends a new default SAE scoring geometry and claims that mechanism generalizes.
- The Geometric Wall analyzes 844 public Gemma Scope checkpoints because layers, widths, and sparsities are the statistical units of a layerwise scaling-law claim; the authors did not train 844 new SAEs.

The transferable standard is that every obvious alternative explanation receives a targeted control and the appendix records enough detail to audit the result. Raw host-model count is not the objective.

### 7.1 arXiv evidence package

The arXiv version should contain:

1. the completed theory, synthetic generator, three-seed GPT-2 confirmation, task spectrum, estimator and geometry audits, architecture control, and static controls;
2. the standardized Pythia concept pilot and, only if it advances, three fresh mature paired seeds;
3. automated descriptions only for replicated concept findings;
4. confirmatory frozen-network natural-text and task-level results;
5. the matched-NMSE static spectral screen or softened paper wording that no longer overstates the existing control;
6. an explicit negative-results and scope audit.

A clean null concept result or failed frozen-network noninferiority result does not block arXiv. It changes the conclusion and prevents a stronger feature or behavior claim.

### 7.2 main-track hardening

If fresh Pythia pairs reproduce the central decoder result, the paper has two base models with paired training replication. A third model is then useful insurance rather than a prerequisite for the narrow claim.

If Pythia remains only the diagnostic 9.13% screen and the positive central result remains GPT-2 block 8 alone, add one modern, benchmark-compatible model before main-track submission. The low-engineering choice is Gemma 2 2B at the SAEBench-supported residual-stream layer, using width 16,384, target $L_0=32$, a concept-blind gamma selection, three paired seeds, and a mature training budget. Evaluate exact decoder distortion, matched NMSE/$L_0$, frozen output KL, and the same concept suite on every pair. One-seed $L_0=16$ and $64$ controls may bracket the central operating point.

Qwen3 1.7B or 4B is a more current alternative but requires additional activation-hook and benchmark-adapter engineering. Prefer Gemma unless a short integration audit shows that Qwen support is comparably reliable.

Stop after one genuinely different modern family. Do not add shallow one-seed rows across many host models, layers, or widths unless the paper's claim changes to architecture-wide or scaling-law generality.

Workshop evidence references:

- https://mechinterpworkshop.com/posters/
- https://arxiv.org/abs/2512.15712
- https://mechinterpworkshop.com/poster-pdfs/98.pdf
- https://mechinterpworkshop.com/poster-pdfs/288.pdf
- https://mechinterpworkshop.com/poster-pdfs/546.pdf
- https://arxiv.org/abs/2606.15054
- https://arxiv.org/abs/2605.09887

## 8. RunPod topology and parallel execution

### 8.1 Intended hardware

The intended compute allocation is one RunPod machine or coordinated pod allocation with three or four NVIDIA A40 48GB GPUs. No experiment requires distributed model training. Use one isolated process per GPU, explicit `CUDA_VISIBLE_DEVICES`, named `tmux` sessions, and stage-specific log files.

Request at least 64GB host RAM, preferably 128GB when available, and enough CPU workers to feed four independent GPU processes. Use a shared 200--250GB network volume so model weights, immutable input caches, and environment files are downloaded once. All processes may read shared inputs, but only one designated process may create or mutate a cache. Every stage writes to its own output directory and atomically promotes a completion manifest.

Do not duplicate a 100GB SAEBench activation cache per GPU. Generate it once, validate and hash it, then mount or open it read-only.

### 8.2 Parallel stage map

**Preparation, before paid GPU time**

- Freeze and review configs, dataset lists, seeds, margins, missingness rules, and artifact schemas.
- Fetch and hash existing GPT-2 and Pythia checkpoints.
- Build CPU/synthetic adapter tests and smoke-test commands.
- Prepare one environment lock and one run manifest template.
- Assign non-overlapping output roots and `tmux` session names.

**Concept timing preflight**

Run `scripts/run_exp10_timing_smoke_a40.sh` in a named coordinator `tmux` session before launching the concept fleet. It verifies the sealed config and source hashes, prepares the cache through a single writer, starts one retained timing child per A40, waits for all four ready artifacts, and releases their identical opaque task sets through one atomic barrier. A finalizer writes the sole authoritative report only after validating all four worker reports, runtime identities, task selections, start skew, and cgroup CPU-stat deltas. Stop and revise the operational plan without opening concept results if the slowest frozen worker projection plus 30% headroom exceeds three pod-hours; do not change scientific tasks, seeds, $k$, or controls to make the timing gate pass.

**Concept pilot wave after the timing gate**

| GPU | Frozen sparse shard | Companion shard, run after sparse work |
|---:|---|---|
| 0 | MSE seeds `2027071701`--`2027071705` | seeds `2027071701`--`2027071703` |
| 1 | MSE seeds `2027071706`--`2027071710` | seeds `2027071704`--`2027071706` |
| 2 | DPSAE seeds `2027071701`--`2027071705` | seeds `2027071707`--`2027071708` |
| 3 | DPSAE seeds `2027071706`--`2027071710` | seeds `2027071709`--`2027071710` |

All four workers open the completed activation cache read-only and reuse one cached MSE adapter plus one cached DPSAE adapter for the process lifetime. The 3/3/2/2 companion split distributes its ten seed-level sweeps across the fleet instead of leaving the entire companion tail on one worker. Frozen-network confirmation and the matched-NMSE spectral screen remain independent stages, but on a four-A40 allocation they run before or after this all-GPU concept wave rather than competing with it.

**Fresh Pythia confirmation, only after pilot advancement**

| GPU | Work |
|---:|---|
| 0 | Fresh seed-0 MSE/DPSAE pair |
| 1 | Fresh seed-1 MSE/DPSAE pair |
| 2 | Fresh seed-2 MSE/DPSAE pair |
| 3 | Checkpoint evaluation, optional sparsity bracket, or completed-stage artifact validation |

Keep each MSE/DPSAE seed pair on the same GPU and in the same paired training process so it shares initialization rules, activation stream, minibatch order, and logging. Parallelize across seed pairs, not within a pair. With only three A40s, run the three seed pairs concurrently and evaluate checkpoints as GPUs finish.

**Labeling wave**

- Extract and archive selected feature contexts while the GPU allocation is still active.
- Validate that every context maps to a frozen checkpoint, benchmark task, token position, and activation value.
- Upload bulky caches and checkpoints to the designated private artifact store.
- Terminate the GPU allocation.
- Submit GPT-5.4 mini Batch jobs from local or CPU compute.

Suggested session names are `dpsae-concept-timing`, `dpsae-concept-pilot`, `dpsae-frozen-confirm`, `dpsae-spectral-screen`, `dpsae-concept-confirm-s0`, `dpsae-concept-confirm-s1`, and `dpsae-concept-confirm-s2`. Every long command must tee to a stable log path inside its stage output directory. Keep `dpsae-resource-watch` alive for the whole allocation using `scripts/watch_arxiv_closure_hardware.sh`; it records all four GPUs, host RAM, `/workspace` capacity, kernel OOM-event availability/count, load, and active tmux sessions every 30 seconds. Warnings begin at 44,000 MiB GPU memory, 88 C, 12 GiB available host RAM, or 25 GiB free storage, with storage marked critical at 10 GiB.

### 8.3 Failure isolation

One GPU process failing must not corrupt shared state or stop independent stages. Each stage records `running`, `complete`, or `failed` status atomically; `complete` requires artifact validation, not process exit alone. Resume commands must be idempotent and must verify config, code, input, and checkpoint hashes before reusing partial outputs.

Do not overwrite an opened confirmatory artifact. A corrected run receives a new run ID and records why the prior artifact was invalid.

## 9. Cost envelope

RunPod prices are time-sensitive; the deploy console is authoritative. The active July 16, 2026 allocation is four A40 48GB GPUs plus a 200GB network volume at $1.80 per wall-clock hour, or $0.45 per allocated GPU-hour. Parallel allocation changes wall time, not the underlying amount of computation, and may add coordination overhead.

| Allocation | Hourly GPU cost | 12 hours | 24 hours |
|---|---:|---:|---:|
| 1x A40 | $0.45 | $5.40 | $10.80 |
| 3x A40 | $1.35 | $16.20 | $32.40 |
| 4x A40 | $1.80 | $21.60 | $43.20 |

Earlier estimates assumed a 25M-token concept confirmation and put the concept pilot at 6--12 A40 GPU-hours, fresh confirmation at another 6--12 GPU-hours, and frozen-network confirmation at 2--4 GPU-hours. The maturity plan can extend Pythia training to 100M--500M tokens, so those confirmation numbers are now lower bounds. Reserve 30--60 A40 GPU-hours for a mature three-pair Pythia confirmation until a short timing benchmark gives a better estimate.

The concept pilot now has a stricter empirical spend gate: the blind smoke must project at most three hours on the active four-A40 pod after 30% headroom, corresponding to at most $5.40 for the projected fleet workload at $1.80 per pod-hour, plus the one-GPU smoke and any cache-preparation time not already included in its measured cache term. This replaces the earlier 6--12 GPU-hour guess once the smoke artifact exists; it does not authorize an expanded benchmark.

A practical arXiv planning envelope is 50--100 total A40 GPU-hours, or $22.50--$45 in GPU charges at the active price, plus storage and API use. Four A40s compress that to roughly 12.5--25 fully utilized wall-clock hours. Setup delays, idle synchronization, cache generation, and retries can increase the billed wall time, so a 24-hour four-A40 reservation costs $43.20 before storage.

A 200GB network volume costs approximately $14 per month at the checked $0.07/GB-month rate. Do not leave stopped local Pod volumes accumulating higher charges; upload versioned checkpoints and bulky results, verify the backup, and delete the Pod when the run is complete.

The targeted GPT-5.4 labeling plan should remain below $10. A one-month 200GB volume, 24 hours on the active four-A40 allocation, and the labeling envelope total $67.20 before retry allowance. The optional modern-model main-track replication is outside this arXiv envelope and must receive a 2M-token timing/memory benchmark before its own budget is frozen.

Pricing references checked July 16, 2026:

- https://www.runpod.io/pricing
- https://docs.runpod.io/pods/pricing

## 10. Provenance and artifact contract

Every reported number must resolve to:

- a versioned config and its SHA-256 hash;
- repository revision and dirty-worktree status;
- exact model revision and checkpoint hash;
- seed and paired-seed mapping;
- corpus, absolute token interval, dataset version, and split hash;
- machine, GPU model, driver, CUDA, PyTorch, and dependency-lock versions;
- machine-readable per-example, per-sequence, or per-task output;
- aggregation script and bootstrap seed;
- wall time, GPU-hours, peak memory, and token/API cost where relevant.

Generated checkpoints and bulky activation caches stay out of Git unless explicitly promoted. During execution, immutable inputs live under `/workspace/dpsae-restored/` and run outputs under `/workspace/dpsae-runs/20260716/<experiment>/<run_id>/`; no launcher may overwrite a completed run ID. Back up the retained run tree to the private Hugging Face repository `aniketdesh/decoder-preserving-sae-paper-closure-20260714` with a manifest that records relative path, size, SHA-256 checksum, producing run ID, and upload verification timestamp. Keep checkpoints, per-example records, per-sequence sufficient statistics, context traces, summaries, configs, environment locks, and logs through paper review; disposable activation caches may be removed only after the manifest identifies their deterministic regeneration inputs. Paper tables and figures must read versioned machine-readable summaries rather than manually transcribed values.

For multi-GPU runs, the root manifest records the allocation and one child manifest per GPU/stage. A final merger checks that all child configs share the intended benchmark revision, data split, objective settings, and paired seed set before producing a paper-facing summary.

## 11. Appendix and paper integration

The paper appendix should function as an audit trail rather than a warehouse of post-hoc results. Add:

- a master experiment ledger covering every model, layer, SAE architecture, width, sparsity, gamma, token budget, seed, data range, machine, code revision, checkpoint hash, status, and artifact path;
- full paired-seed training curves and checkpoint trajectories;
- the complete gamma frontier, rejected candidates, and frozen selection rule;
- per-task concept results, probe-reseed variability, family aggregation, multiplicity correction, full-code and reconstruction baselines, and every advancement decision;
- exact autointerp prompts, schemas, snapshots, context rules, judge calibration, and API accounting;
- the frozen-network margin rationale, sample-size calculation, all seedwise intervals, task-level results, and control conditions;
- a failure-and-scope section for IOI, semantic recurrence, JumpReLU, cross-layer/model screens, and static controls;
- a reproducibility statement in the main paper that points to configs, code, appendices, and artifact manifests.

Unlimited appendix pages do not move load-bearing evidence out of the main paper. Current ICLR guidance says reviewers are not required to read supplementary material, so the nine-page main text must still state the primary estimands, advancement logic, replication count, matched-quality controls, and the result that changes the conclusion.

Review guidance references checked July 16, 2026:

- https://iclr.cc/Conferences/2026/AuthorGuide
- https://iclr.cc/Conferences/2026/ReviewerGuide

When implementation choices change, update the method section and this plan in the same change so the manuscript cannot drift from the code. After results stabilize, remove the concept and frozen-network WIP markers, resolve the stale cross-model WIP, compile from `paper/`, resolve citations, and visually inspect every rendered page.

## 12. Planned figures and required raw-data retention

The useful lesson from MP-SAE, Matryoshka, Temporal SAE, Priors in Time, SAEBench, and Goodfire's feature analyses is to visualize the exact phenomenology the method claims to change. For this paper that means paired concept effects, the information-to-concentration ladder, token-level evidence for any feature labels, concept-blind training maturity, and frozen-network noninferiority. It does not mean importing their UMAPs, hierarchy heatmaps, modality histograms, steering sweeps, or broad sparsity frontiers: those plots test geometry, hierarchy, causal intervention, or scaling claims outside this frozen protocol.

Figure typography is frozen to the vendored **D-DIN** family under `src/dpsae/fonts/d-din/`. Regular, Italic, and Bold form the ordinary hierarchy across Matplotlib, TikZ, and PGFPlots; Condensed is allowed only for an irreducible short categorical or diagram label after wording and layout have been improved; Expanded is restricted to sparse display accents and never appears on axes, ticks, legends, or ordinary annotations. Final renders must resolve ordinary Latin text to the pinned repository files and embed the font in the vector PDF; a host-installed DIN substitute or silent fallback invalidates the release render.

The panel plan below is outcome-invariant. A failed pilot or noninferiority test gets the same effect plot with its null or margin visible; the destination changes, not the plot definition. All paper figures must be generated from the retained records listed here rather than from aggregate console output.

| Panel | Claim and visual encoding | Aggregation and uncertainty | Raw data or snapshots that must be retained now | Priority |
|---|---|---|---|---|
| **A. Concept information-to-concentration ladder** | Separates general information retention from concentration into a few features. The x-axis is the frozen representation restriction: original residual, full reconstruction, full code, then selected $k=5,2,1$ features; the y-axis is held-out AUROC. MSE and DPSAE are paired lines, with the shared original-residual result shown once. | Average ten probe seeds within task, then macro-average using the frozen family blocks; show family-block-bootstrap 95% intervals. Show probe-reseed spread as light points or a separate inset, not as extra task replicates. | For every example: immutable example/split ID, task and family, target, method, training-pair seed, probe seed, representation type, $k$, held-out decision score and prediction. Also retain selected feature IDs, ranks, signed probe weights, regularization choice, missingness reason, checkpoint hash, and original/reconstruction/full-code operating metrics. | **Main** only if fresh confirmation advances; otherwise the complete pilot ladder is **appendix** evidence for the null. |
| **B. Per-concept paired effects and excess sparse gain** | Establishes whether $\Delta_5$ is broad rather than family-driven and whether it exceeds the full-code gain. Use a family-grouped forest plot of taskwise DPSAE-minus-MSE $\Delta_5$, plus a companion scatter with $\Delta^{\mathrm{full\ code}}$ on x, $\Delta_5$ on y, and the equality line marking zero excess gain. | Task points are means over probe seeds; thin intervals show probe-reseed uncertainty only. The macro diamond uses the family-block bootstrap, and family summaries carry the frozen Holm correction. Do not pool examples across tasks. | Retain Panel A records plus the frozen family map, bootstrap resample indices/seeds, family-level hypotheses, unadjusted and adjusted $p$-values, and every excluded/missing task with its preregistered reason. | **Main** beside Panel A if confirmed; otherwise **appendix**, with the gate result stated in text. |
| **C. Validated feature activation cards** | Gives qualitative evidence that replicated sparse-probe features activate on the labeled pattern rather than a salient token. Each card shows the blinded label, probe rank/weight, and token-aligned activation heatmaps or traces for deterministically sampled high-, middle-, and near-miss/negative contexts. MSE and DPSAE examples may be paired by benchmark task, but features must not be presented as one-to-one matches unless a separate matching rule was frozen. | No population inference is attached to a card. Examples come only from the archived sampling strata; the caption reports the selection rule and held-out validation score. A small main-text set is chosen by a method-blind deterministic rule, with the full gallery in the appendix. | Retain exact text, token IDs, decoded-token strings, character offsets, document/source hash, context window boundaries, activation for the selected feature at every token, target-token position, activation quantile/stratum, discovery-versus-held-out flag, matched-negative ID and rule, feature ID, checkpoint/seed, probe provenance, and raw label request/response IDs. Summary-only top examples are insufficient. | **Main** only for replicated candidates whose held-out label validation passes; otherwise omit from the main text and retain any attempted gallery in the **appendix/artifact**. |
| **D. Autointerp validity distribution** | Demonstrates that descriptions predict held-out activations and are stable across context resamples. Plot ECDFs or paired distributions of held-out discrimination and simulation scores, with separate markers for the human-reviewed subset and second-description stability. Method identity remains hidden during scoring. | Bootstrap over features within training-pair seed and report seedwise distributions; use Wilson or exact binomial intervals for human correctness/preference. Do not use model self-confidence as an error bar or treat evaluator calls as independent scientific replicates. | Retain per held-out context: feature/context ID, true activation value and active/negative label, evaluator prediction/probability, explanation-call ID, evaluator snapshot/prompt hash, nearest-neighbor or matched-negative provenance, resampled-description alignment, and raw blinded human rubric responses/adjudication. | **Appendix** validation; at most a compact inset accompanies Panel C in the main text. |
| **E. Concept-blind training maturity** | Shows that the fresh Pythia checkpoint was chosen because operating metrics matured, not because concept scores looked favorable. Plot tokens and unique corpus exposure on x against held-out NMSE, exact decoder distortion, $L_0$, and dead-feature fraction in aligned facets; show paired methods, raw seed trajectories, and vertical checkpoint markers. | Show all three seed lines and a median/range summary; evaluation-window intervals use the natural-text sampling unit. Plot recorded windows without a smoothing curve that implies extra observations. Concept scores must not appear until the duration rule is frozen; any later checkpointwise concept trajectory is explicitly exploratory and appendix-only. | Preserve checkpoints and optimizer/scheduler state at 25M, 50M, 100M, 250M, and eligible 500M tokens; retain every held-out metric window with token count, unique-token exposure, absolute corpus interval, cache epoch/reuse count, wall time, and seed. Keep the final stop-rule decision record even if training stops before the largest snapshot. | **Appendix**, with a small matched-quality/maturity inset in the **main** concept figure if space permits. |
| **F. Frozen-network noninferiority forest** | Directly answers the frozen-network question. Plot $R_s=\mathrm{KL}_{\mathrm{DPSAE}}/\mathrm{KL}_{\mathrm{MSE}}$ for each of the three trained pairs with paired-bootstrap 95% intervals; vertical lines at 1.00 and the 1.01 noninferiority margin make superiority and failure visually distinct. | Use the sequence-level paired bootstrap specified in Section 6 and show every seed separately. A descriptive across-seed marker may be added but cannot hide a failing seed. | For every frozen sequence and condition (`original`, identity hook, mean ablation, paired MSE, paired DPSAE), retain sequence/index hash, valid-token count, summed original-to-condition KL, summed cross-entropy, original and condition top-1 agreement count, next-token correct count, activation-NMSE numerator/denominator, $L_0$ sum/count, checkpoint pair, and bootstrap seed. These per-sequence sufficient statistics avoid retaining full-vocabulary logits while preserving exact aggregation. | **Main regardless of pass or fail**, because the outcome changes the paper's conclusion. |
| **G. Frozen IOI task compatibility** | Tests the same full reconstructions on the preregistered task without reopening the failed sparse-feature IOI claim. Plot paired prompt-level absolute error in the correct-minus-incorrect name logit difference for MSE and DPSAE, with original-preference agreement and ordinary accuracy in a compact aligned panel. | Use the paired prompt bootstrap, show training-pair seeds separately, and report every frozen template/name family. Natural-text KL remains primary even if IOI is more favorable. | Retain prompt ID and text hash, generator seed, template/name/counterfactual IDs, correct and incorrect token IDs, original/MSE/DPSAE logits for those tokens, original preference, reconstructed preference, correctness, and prompt-level absolute errors for every checkpoint pair. | **Appendix** by default; promote a compact secondary panel only if it agrees with, and materially clarifies, Panel F. |
| **H. Gate-aware null and negative displays** | Prevents positive-only visualization. Render Panels A--B for the concept pilot and Panel F for frozen compatibility before routing them to main or appendix. Existing heterogeneous failures remain an evidence-ledger table with their native estimand, frozen threshold, status, and artifact link; do not combine incomparable effects into a standardized summary plot. | Use each experiment's already frozen uncertainty and gate. Show zero, noninferiority margins, or feasibility bands directly and label diagnostic versus confirmatory status. | Retain the complete panel inputs even when a gate fails, plus gate version, decision timestamp, opened metrics, status, invalidation reason, and artifact hash. Do not discard feature contexts merely because their label is unattractive; quarantine invalid runs under a new run ID. | The load-bearing concept and frozen outcome appears in **main** prose either way; complete null plots and the negative-results ledger belong in the **appendix** unless a failure is itself the conclusion-changing result. |

The additional collection burden relative to Sections 4--6 is small but time-sensitive: save held-out per-example probe scores rather than task aggregates, full token-aligned traces for every candidate context rather than only the peak token, per-sequence frozen-network sufficient statistics rather than only bootstrap summaries, and optimizer-bearing maturity snapshots with unique-corpus exposure. These are the fields that cannot be reconstructed cheaply after the GPU pod is released.

Plot-design sources reviewed July 16, 2026:

- MP-SAE uses controlled dictionary-alignment heatmaps, sparsity curves, co-activation structure, and multimodal feature distributions: https://arxiv.org/abs/2506.03093
- Matryoshka uses activation-hole examples, sparse-probe curves, and training/scaling trajectories: https://arxiv.org/abs/2503.17547
- Temporal SAE uses probe performance across feature budgets and token-aligned top-feature traces: https://arxiv.org/abs/2511.05541
- Priors in Time uses sequence trajectories and similarity maps to study temporal structure: https://arxiv.org/abs/2511.01836
- The sparse-probing case study motivates strong full-activation baselines and per-dataset paired results: https://arxiv.org/abs/2502.16681
- SAEBench and its 2026 reliability audit motivate $k$-curves, checkpoint trajectories, and explicit separation of probe-reseed noise from SAE-seed variation: https://arxiv.org/abs/2503.09532 and https://arxiv.org/abs/2605.18229
- Goodfire's Llama feature work motivates activation-stratified contexts, matched distractors, and held-out explanation scoring, while its map and geometry work illustrates plot forms deliberately excluded from this narrow claim: https://www.goodfire.ai/research/understanding-and-steering-llama-3, https://www.goodfire.ai/research/mapping-latent-spaces-llama, and https://arxiv.org/abs/2604.28119

## 13. Execution gates and order

1. **Freeze configs and manifests.** Finish every item marked “to freeze” below before paid runs.
2. **Validate adapters.** Native and benchmark code/reconstruction/NMSE/$L_0$ must agree before concept scores are opened.
3. **Pass the blind concept timing gate.** Generate and hash the shared cache, time the same eight size-stratified opaque tasks on four synchronized workers with seed `2027071799`, and require the slowest frozen-shard p95 projection plus 30% headroom to remain at or below three pod-hours.
4. **Run the independent first-wave stages.** Dedicate all four GPUs to the frozen concept shards for its wave; frozen-network confirmation and the matched-NMSE spectral screen may run in separate waves in either order.
5. **Audit, aggregate, then audit again.** A pre-aggregation artifact audit must prove exact shard completion and hash consistency; the final audit validates the aggregate before any concept outcome is interpreted.
6. **Apply the concept pilot gate once.** Record pass or fail without extending the evaluation grid.
7. **Train fresh Pythia pairs only after advancement.** Choose maturity without concept labels, then run all three pairs concurrently.
8. **Run confirmatory concept evaluation once.** Apply the frozen seedwise, aggregate, and multiplicity rules.
9. **Describe only replicated candidates.** Archive contexts, shut down GPUs, then call the Batch API.
10. **Integrate all outcomes, including nulls.** Update docs, methods, results, limitations, compute, and artifact manifests together.
11. **Decide main-track generality after arXiv closure.** Add one modern family only if it materially addresses the remaining single-model criticism.

## 14. Freeze checklist

Already fixed by this document:

- the separation between information preservation, sparse feature concentration, and frozen-network compatibility;
- Pythia-160M-deduped block 8, width 16,384, target $L_0=32$ for the concept pilot;
- SAEBench commit, `normal` setting, L1 selection, non-binarized latents, $k=1,2,5$, and ten probe seeds;
- macro paired $k=5$ AUROC as the concept primary;
- the concept pilot advancement thresholds;
- exclusion of the opened pilot pair from fresh confirmation;
- GPT-5.4 mini as primary labeler, nano as cleanup, and full GPT-5.4 or humans for adjudication;
- GPT-2 frozen output-KL ratio as primary, 1.01 noninferiority margin, sequence bootstrap, fresh 200M--210M range, and all-three-seed requirement;
- three- or four-A40 non-distributed execution, with one paired seed per GPU during confirmation;
- no GPU allocation kept alive during asynchronous labeling.

Must be frozen in machine-readable config before the affected run:

- Pythia maturity stop rule, unique-corpus exposure, and final maximum token budget;
- any Gemma or Qwen main-track replication config, which remains outside the arXiv frozen scope.

Now frozen in the experiment configs before opening outcomes:

- the exact 113-dataset `sae-probes` list and hashes, family blocks, fail-closed missingness rule, regularization choices, ten probe seeds, family bootstrap seed, and multiplicity rules;
- adapter parity tolerances, stored normalization convention, native threshold rule, and prohibition on decoder renormalization;
- GPT-2 checkpoint hashes, frozen-network sequence and IOI generator seeds, sample sizes, bootstrap seeds, identity tolerances, and all-three-seed noninferiority rule;
- deterministic nonoverlapping FineWeb sequence-selection algorithm and seed within 200M--210M; the resulting integer indices and cache hash become immutable preparation artifacts before inference begins;
- the matched-NMSE spectral coefficient grid, target, tolerance, tie break, and conditional three-seed advancement rule;
- the active four-A40/$1.80-per-hour/200GB allocation and the run, backup, checksum, and retention layout in Section 10.

## 15. Dated decision log

**July 15, 2026.** The experiment-and-figure closure suite completed the task-prior strength sweep, clean gamma sweep, three-seed GPT-2 confirmation, evaluation-geometry robustness, task-spectrum recomputation, isolated objective overhead, and review-only frozen-model diagnostic. Standard concept discovery and confirmatory frozen-network evaluation remained blocked pending external compute.

**July 16, 2026.** The remaining work was split into a Pythia/SAEBench concept-discovery study and a GPT-2 frozen-network confirmation. The concept hierarchy, pilot advancement rule, autointerp model hierarchy, API budget, output-KL noninferiority design, fresh frozen range, task-level endpoint, and single-A40 cost estimates were recorded.

**July 16, 2026, evidence-scale review.** A review of the 2026 Mechanistic Interpretability Workshop spotlights found that experimental breadth follows the claim's inferential axis rather than a universal host-model count. The arXiv plan keeps the two closure experiments primary; a successful fresh Pythia fleet becomes the second-model replication, while one modern Gemma-family replication is reserved as main-track hardening if the central positive evidence otherwise remains GPT-2-only. The appendix will expose the existing experiment fleet, all negative gates, prompts, splits, compute, and provenance.

**July 16, 2026, compute plan.** The intended RunPod allocation changed from one A40 to three or four A40s to reduce wall time. Each confirmatory seed pair remains co-located on one GPU, the three pairs run concurrently, and a fourth GPU runs independent evaluation or controls. Total GPU-hours are not assumed to fall merely because the work is parallelized.

**July 16, 2026, plot and retention review.** MP-SAE, Matryoshka, Temporal SAE, Priors in Time, SAEBench/sparse-probing, and Goodfire feature-analysis papers were reviewed for plot forms before the paid closure run. The plan adopts paired $k$-probe curves, per-concept effect forests, full-code-versus-sparse-code excess-gain views, token-aligned validated feature cards, concept-blind maturity trajectories, and seedwise frozen-network noninferiority forests. Geometry embeddings, hierarchy/modality plots, steering sweeps, and broad SAE sparsity frontiers were rejected because the frozen experiments do not support those claims. Per-example probe outputs, per-token feature traces, per-sequence frozen sufficient statistics, and maturity snapshots are now required retention fields so the necessary panels remain constructible after compute is released.

**July 16, 2026, execution freeze.** The active pod was provisioned as four A40 48GB GPUs with a 200GB network volume at $1.80 per hour. Exact GPT-2 and Pythia checkpoint bundles, calibration artifacts, prior split caches, and provenance manifests were restored from the private Hugging Face archives and verified by SHA-256 before use. The concept, frozen-network, and matched-NMSE static configs now freeze every outcome-facing pilot choice; fresh Pythia duration remains conditional and must be frozen from concept-blind maturity metrics only if the pilot advances.

**July 16, 2026, concept runtime audit.** A static audit found 21,470 `find_best_reg` calls in the concept plan, of which 2,260 came from a residual-stream baseline duplicated between SAEBench and the companion evaluator. The wrapper copy was disabled while the exact companion baseline was retained, reducing the plan to 19,210 calls without changing any scientific endpoint. The four-GPU schedule was frozen to 5/5/5/5 sparse method-seed shards followed by 3/3/2/2 companion-seed shards, with adapters cached once per worker and BLAS/OpenMP threads capped at one quarter of the visible CPU affinity. A blind eight-task size-quartile smoke with nonreport seed `2027071799` must project the complete fleet at or below three pod-hours after 30% headroom before workers may start.

**July 16, 2026, dense timing stop and exact-CSR fallback.** The first blind timing report projected 13.975 pod-hours after the frozen 30% headroom, so the launcher stopped before opening any concept outcomes or starting the fleet. Its stage-only profile isolated the cost to the dense companion full-code L2 probe (`109.05` seconds per task at p95), while each sparse-probe stage was approximately one second. Those full codes have width 16,384 but BatchTopK support around 32, so the operational fallback converts their exact nonzero values and indices to canonical SciPy CSR before calling the same `find_best_reg`; every scientific task, seed, split, representation, regularization grid, solver, and artifact remains unchanged. A first CSR smoke stopped before its first fit because upstream `sae-probes` calls `len(X_train)`, which SciPy leaves ambiguous; a CSR subclass now returns the unambiguous row count while preserving standard CSR storage and operations. The optimized smoke uses schema v2, and the gate plus artifact auditor reject the failed dense schema-v1 report so it cannot accidentally authorize the fleet.

**July 16, 2026, cold-CSR timing stop and exact parallel fallback.** The complete schema-v2 CSR smoke still failed closed at 9.067 projected pod-hours: companion p95 was `68.83` seconds per task and the worst two-method full-code block took `75.28` seconds. A proposed coefficient-path reuse optimization was then rejected before deployment because a deterministic sklearn-1.7.2 counterexample changed the selected C despite identical mathematical objectives. The replacement performs every full-code C/fold fit cold and independently, preserves upstream C order and tie resolution, and only schedules the ten C candidates concurrently; original residual, reconstruction, and sparse L1 paths remain pinned upstream calls. A regression fixture freezes the counterexample at the third C-grid element (approximately `599.48425`) alongside dense and sparse sequential-versus-parallel output checks. The next blind report is schema v4, and the gate plus artifact auditor reject all earlier timing schemas.

**July 16, 2026, six-job timing stop and ten-job fallback.** The schema-v4 exact cold-C smoke remained fail-closed at 3.3199958 projected pod-hours, with companion p95 `21.9479` seconds and worst full-code stage `19.3504` seconds. No estimator, split, C ordering, tie rule, or artifact changed: schema v5 only raises the per-worker process fan-out from six to ten so all ten independent cold C candidates run in one scheduling wave. Across four concept workers this reserves at most 40 of the available 96 CPU cores, leaving headroom for orchestration and the sequential upstream probes. The gate and artifact auditor reject schema v4 even if its other identities match.

**July 16, 2026, exact ten-job timing pass and fleet release.** The blind schema-v5 smoke at repository revision `5d5db9756ade0701fa8991511196d3c30621e53f` passed the frozen gate at 2.7184148 projected pod-hours, including the hash-bound 1,506-second cold-cache measurement and 30% headroom. Companion p95 fell to `17.3626` seconds, total task p95 was `18.7450` seconds, and the report retained zero concept metrics with dataset names suppressed. After the supervisor independently matched the schema, config digest, source hashes, seed, task count, matrix format, exact optimization identity, and ten-job count, it released the four frozen workers plus the integrity-audit finalizer in tmux against the fresh `pythia160m-block8-s0-pilot-v6-cold-c10` output root.

**July 16, 2026, cgroup-quota stop and schema-v6 fallback.** Live throughput falsified the schema-v5 projection before any concept values were opened: the container exposed 96 CPUs through affinity and `nproc`, but its finite cgroup quota was only 32.3 cores. The isolated timing worker stayed below that quota at 24 threads, while the fleet launched four 24-thread parents plus four ten-process cold-C pools; all sampled cgroup periods were throttled, so the linear four-worker projection was invalid. The supervisor was deliberately tripped fail-closed at `03:47:11Z`, after all 20 sparse jobs and 134 of 1,130 companion dataset-seed artifacts had been atomically written; the attempt ran from `01:59:40Z` to `03:47:20Z`, consuming approximately 1.79 pod-hours or $3.23 at $1.80 per hour. Those partial artifacts remain provenance for the failed attempt and are not reused in the fresh run. Schema v6 resolves the effective CPU count as `min(affinity, floor(finite cgroup quota))`, records the raw 32.3-core quota and derived 32/8 effective/worker counts, and limits each worker to eight numerical-library threads and eight independent cold-C processes. Its replacement timing smoke synchronizes four processes on the same opaque tasks, retains one report per process plus cgroup throttling deltas, and projects the slowest exact shard rather than extrapolating or pooling an isolated measurement. A new blind report and output root are required before another fleet release; estimators, splits, C grid and ordering, tie resolution, seeded shuffles, and final cold refits are unchanged.

**July 16, 2026, synchronized cgroup timing stop and all-representation batching fallback.** The schema-v6 four-worker smoke reproduced the live 32.3-core contention and failed closed at `4.3659171` projected pod-hours, including the 1,506-second cache term, 50.27-second maximum initialization, and 30% headroom. The slow 3-seed workers measured companion p95 values of `30.6549` and `30.2813` seconds; on their hardest opaque tasks, full-code fitting used approximately 19 seconds while the original-residual and reconstruction controls contributed another 12--17 seconds. Sparse stages remained approximately one second. The supervisor halted before creating any concept worker or finalizer, and all four reports retained zero concept metrics. Schema v7 changes scheduling and numerical-thread placement only: the exact 50 cold validation candidates for all five L2 representations enter one retained eight-process queue, are regrouped in representation-major and original-C order, and receive five fresh selected-C shuffled refits in one-thread children. It changes no example, split, representation, estimator, solver, hyperparameter, seed, tie rule, or reported artifact, but the production topology must match pinned upstream coefficients, intercepts, classes, scores, predictions, and metrics within the frozen tolerance before timing may begin. A fresh synchronized blind report and fresh output root remain mandatory before fleet release.

**July 16, 2026, figure typography freeze.** D-DIN was frozen as the paper-wide figure typeface and vendored with its upstream licenses and pinned SHA-256 values. Regular, Italic, and Bold are the ordinary hierarchy; Condensed is reserved for irreducible short labels and Expanded for sparse display accents. The plotting layer registers the repository files directly, uses an ASCII minus for D-DIN-compatible negative ticks, and requires embedded D-DIN fonts in release PDFs.

**July 16, 2026, unattended approval boundary.** The pod-resident supervisor may finish the blind timing gate, wait for the matched-NMSE control to release GPU 3, launch the four frozen concept workers, run the mandatory pre-aggregation and final integrity audits, and record the concept advancement decision without a live SSH connection. It must stop after that decision: fresh confirmation, any OpenAI API call, the cross-experiment release audit, and Hugging Face backup require renewed user approval. A four-hour ceiling bounds the concept fleet after launch; exceeding it records a failure instead of silently continuing an unexpectedly expensive run.

**July 16, 2026, renewed unattended completion approval.** The user subsequently authorized continuing through every predeclared advancement gate while they are offline, including any gated fresh confirmation, GPT feature labeling within the funded API budget, cross-experiment release audit, checkpoint verification, and Hugging Face backup. This authorization does not loosen scientific, privacy, integrity, or spend gates: operational bugs may be fixed without opening sealed outcomes or changing frozen tasks, seeds, splits, estimators, endpoints, or decision rules, and the manuscript prose remains unchanged until the user reviews the completed evidence.
