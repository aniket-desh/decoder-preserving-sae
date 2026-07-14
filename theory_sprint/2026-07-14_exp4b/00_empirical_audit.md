# Experiment 4b empirical audit

## Verdict

Experiment 4b establishes a narrow representation-level result: on one GPT-2-small activation site and one BatchTopK architecture, the trained DPSAE objective reduces fresh exact in-group ridge-hat distortion by about 24% relative to paired MSE SAEs, with about a 7% NMSE cost. The sign replicates across three paired initializations, three sparsities, three ridge settings, and three group constructions. The frozen IOI comparison does not establish better causal specificity, the proposed continuous target fails its dense-activation validation under the fixed probe protocol, and the experiment supplies no positive reason to promote Fisher or activation-manifold geometry into the paper.

The most consequential qualification is that group size changes the observed DPSAE advantage from roughly 24% at 128 tokens to 13% at 256 tokens on the same 16,384-token exact set even after recalibrating ridge. The 64-token condition is about 35%, but the evaluator's 128-group cap subsamples 8,192 tokens there, so that endpoint also changes the evaluated token subset. The clean 128-versus-256 comparison is enough to show that the objective is a family of group-dependent geometries, not an empirically grouping-independent estimate of a unique corpus-level operator.

## Repository state and provenance

- The local worktree was clean at the start of the sprint at `b5544fc28afb77c727ec416df6a1b41587e82fe6` on `main`.
- The full raw 4b artifacts live on the supplied RunPod under `/workspace/decoder-preserving-sae/artifacts/exp04b_confirmatory/`. Only the two source-fleet natural-text JSONs and natural-text figure were present locally at the start.
- Both natural evaluation JSONs record code revision `38b4c3ba08ae9e29ba3f026c78bdc815183470c9`, dirty `false`. The current code differs in one IOI exposure-summary lookup fix, stricter cache filtering, plotting, and an audit-axis plotting fix; the natural evaluation implementation is unchanged.
- The final IOI JSON does not record a Git revision. Its successful exposure-summary construction requires commit `d9854766748d76441a9625a9b7bf62d093cd1eb4` or later, because the previous `_selected_row` implementation could not read exposure rows. The exact IOI analysis revision is therefore **not traceable from the artifact**, although the raw values and current implementation are auditable.
- On the RunPod at inspection time, the checkout was at `b5544fc...` with one unrelated untracked `uv.lock`; raw artifacts are ignored by Git.

### Raw artifact inventory

| Artifact | Availability | SHA-256 | Role |
| --- | --- | --- | --- |
| `natural_evaluation_source.json` | local and RunPod | `ac074b3c3f259cdb781621eb03e5066c980c14b8b4b3dc6169c1bdf831d6e92b` | source MSE/DPSAE/whitening, all sparsities, sampled and primary exact metrics |
| `natural_exact_audit_source.json` | local and RunPod | `c09069c74a43a5534a5d31baede048b5aa8ceef636f96f0c88f492e7e85cac92` | 168 source-fleet exact one-factor rows |
| `natural_evaluation_baseline.json` | RunPod | `26c4680d14989bd35d25a5ba9727de80d6cf7742614da79df0f8b4b7d881dd59` | new 100M-token MSE/DPSAE/whitening/spectral confirmation fleet |
| `natural_exact_audit_baseline.json` | RunPod | `1b76f68c0d23627e395848b3da1e85b6f07e2c8818ee6536e0ea21c200c7f07a` | exact baseline-fleet one-factor rows |
| `baseline_selection.json` | RunPod | `0f36cde099f144cec9fa7754b64adbc3304b0cef0e651fa67c40a5ade2e7c783` | static-baseline screening rule and selected weights |
| `ioi_feature_count_selection.json` | RunPod | `2c7115835d837ab22669959e754d45584736f1a4bd135c685da685cd842948a1` | frozen global feature count |
| `ioi_confirmatory.json` | RunPod | `85f9c04a44586d3f1dfbe5a940beae0f52c5236a913f65a929b6d2e3d10760c6` | final frozen IOI results and paired bootstraps |
| `resolved_config.json` | RunPod | `d4ea61b7b1193886b7edc60e61a4431b2f15aabe573c3de0a58cf2610264455c` | resolved 4b configuration |

All quantitative baseline and IOI rows below were queried directly from the remote JSON rather than inferred from figures.

## Exact implemented protocol

### Representation and SAE

The source is `openai-community/gpt2`, with 768-dimensional `resid_post` activations after one-based block 8 (`hidden_states[8]`). A fixed mean vector and one global RMS scalar estimated on 65,536 calibration tokens normalize every later activation; geometry groups are not centered independently. The SAE is an untied, nonnegative BatchTopK model with 16,384 latents, ReLU preactivations, unit-normalized decoder rows, and a learned decoder bias. During each 2,048-token training batch, BatchTopK retains exactly `batch_tokens * k` entries globally. Evaluation replaces global TopK with the learned activation threshold, so inference L0 is close to but not exactly `k`.

The k=32 source and baseline confirmation fleets train for 100,001,792 tokens. The k=16 and k=64 robustness fleets train for 50,001,920 tokens each, so sparsity comparisons establish paired within-k robustness rather than equal-budget comparisons across k. All compared methods within a fleet see the same activation batches, optimizer schedule, initialization seed, and random target sequence.

### Training objectives

For each group \(X_g\in\mathbb R^{128\times768}\), the code uses

\[
K_\lambda(X_g)=X_g(X_g^\top X_g+128\lambda I)^{-1}X_g^\top,
\]

with \(\lambda=1.6049035191535947\), calibrated to \(\operatorname{tr}K/n=0.25\). Each step draws 16 Gaussian columns per group and then normalizes **each column in each group** to sample RMS one. Thus the actual probe law is uniform direction with fixed radius \(\sqrt n\), not an unnormalized Gaussian. The DPSAE training term is one ratio of global sums,

\[
\widehat L_{\rm dec}=
\frac{\sum_{g,j}\lVert(K_\lambda(X_g)-K_\lambda(\widehat X_g))y_{gj}\rVert_2^2}
{\sum_{g,j}\lVert K_\lambda(X_g)y_{gj}\rVert_2^2},
\]

not a mean of per-group ratios. The original solve and denominator are under `no_grad`; only the reconstructed representation's solve receives gradients. SAE encoding/decoding uses BF16 autocast on CUDA, then the reconstruction and all ridge solves are FP32. The selected objective is

\[
L_{\rm DPSAE}=L_{\rm NMSE}+0.25\widehat L_{\rm dec}+\tfrac1{32}L_{\rm AuxK}.
\]

The static spectral control uses calibration covariance \(C\) and

\[
M_\lambda=C(C+\lambda I)^{-2},\qquad
L=L_{\rm NMSE}+\beta\frac{\lVert(X-\widehat X)M_\lambda^{1/2}\rVert_F^2}
{\lVert XM_\lambda^{1/2}\rVert_F^2}.
\]

Seed-0 screening over \(\beta\in\{0.125,0.25,0.5,1\}\) used 25M training tokens, then selected on the untouched 180M--185M FineWeb tail under a `1.10 * MSE NMSE` cap. All candidates qualified; whitening selected \(\beta=0.5\) and spectral selected \(\beta=1\). Confirmation used seeds 0, 1, 2, a new data-order seed `1793502167`, a new probe-sequence seed `1605706622`, 100,001,792 tokens, and the untouched 185M--190M tail.

Initialization seeds are the reported model seeds 0/1/2. Data-order and training-probe randomness are common within each paired fleet and separated by deterministic stream seeds. Natural evaluation uses probe seed `2027071301`, selection seed `2027071302`, and test/bootstrap seed `2027071303`. IOI analysis and bootstrap use `2027071310` with fixed offsets for collateral KL. The raw artifacts do not separately record a stochastic evaluation seed beyond these protocol seeds.

### Natural-text estimands

The primary sampled report uses 65,536 normalized activations, contiguous groups of 128, and 16 fresh normalized probes. The exact report uses 16,384 tokens in 128 groups and identity targets, so each group reports

\[
\frac{\lVert K_\lambda(X_g)-K_\lambda(\widehat X_g)\rVert_F^2}
{\lVert K_\lambda(X_g)\rVert_F^2}.
\]

The headline aggregates as a ratio of sums across groups. Paired reductions use the same groups and bootstrap 10,000 group resamples. These confidence intervals quantify finite-group uncertainty conditional on the trained models; they do not quantify variation over initialization, corpora, model families, layers, or SAE architectures.

The one-factor exact audit changes ridge fraction, group size with recalibrated ridge, or grouping. `contiguous` preserves flattened sequence order, `shuffled` randomly mixes evaluated tokens after the LM forward pass, and `document_balanced` round-robins token positions across EOS-inferred documents/sequences. The latter is an evaluation grouping policy, not a true document-indexed training objective. `max_groups=128` means n=128 uses all 16,384 exact tokens, n=256 uses all 16,384 in 64 groups, and n=64 randomly selects 128 of 256 possible groups and therefore only 8,192 tokens.

### IOI protocol

The original 4,096-example discovery set is split into 3,072 ranking examples and 1,024 selection examples; the untouched original 2,048-example validation split is used as final test. Code and artifacts call the internal selection rows `validation`, but they do not overlap the final split. Duplicate-state features are ranked by the absolute standardized clean-versus-ABC S2 code difference.

One global count is chosen across the six MSE and DPSAE seed models by maximum median S2 zero-ablation effect subject to maximum natural-text collateral KL at most 0.06. The frozen count is 64, with median validation effect 2.4181 and maximum validation KL 0.03428. Whitening and spectral do not influence selection.

The final intervention zeroes the 64 ranked SAE coordinates only at S2. The collateral control zeroes the same coordinates at one natural-text position per sequence, draws its END-minus-intervention lag from the IOI distribution, and reads KL at the final token. Final uncertainty uses 10,000 paired bootstrap resamples over 2,048 IOI examples or 256 natural sequences. Exposure matching interpolates the MSE KL curve at the candidate's firing, mass, or decoded-energy exposure only when the curves share support; seed 0 has no common support for any of the three exposure coordinates.

The continuous target is original-model correct-IO minus subject logit difference. Features are ranked on the 3,072 ranking examples by absolute univariate correlation, a fixed ridge of 0.01 is fit on ranked prefixes, the same already-frozen count 64 is used on final test, and no ridge hyperparameter is selected on the 1,024-example intermediate split. The dense activation uses the same fixed ridge protocol.

## Recomputed headline tables

`checks/recompute_exp4b_tables.py` recomputes exact reductions as ratios of raw per-group numerator sums and asserts agreement with every stored source-fleet reduction to `1e-10`.

### New k=32 confirmation fleet

| Method vs paired MSE | Seed | Exact distortion change | 95% paired group CI | NMSE change |
| --- | ---: | ---: | ---: | ---: |
| DPSAE | 0 | -24.39% | [-24.96%, -23.85%] | +6.91% |
| DPSAE | 1 | -24.37% | [-24.90%, -23.84%] | +7.09% |
| DPSAE | 2 | -24.23% | [-24.77%, -23.70%] | +7.11% |
| Whitening | 0 | +12.57% | [+11.75%, +13.40%] | +5.75% |
| Whitening | 1 | +12.56% | [+11.83%, +13.30%] | +5.53% |
| Whitening | 2 | +12.16% | [+11.45%, +12.86%] | +5.47% |
| Static spectral | 0 | -0.37% | [-1.00%, +0.28%] | +0.91% |
| Static spectral | 1 | -0.91% | [-1.45%, -0.39%] | +0.92% |
| Static spectral | 2 | -0.88% | [-1.50%, -0.25%] | +0.85% |

The result rules out these two selected controls as quantitative explanations of the 24% effect. It does not rule out all static covariance objectives, because only one theorem-derived omission-cost surrogate and one whitening construction were tuned, nor does it provide a matched-NMSE Pareto curve.

### Source-fleet sparsity and grouping audit

| Condition | Seed-0/1/2 exact DPSAE reductions vs paired MSE |
| --- | --- |
| k=16, base geometry | 29.41%, 29.83%, 29.93% |
| k=32, base geometry | 23.66%, 23.68%, 24.38% |
| k=64, base geometry | 22.58%, 22.50%, 23.44% |
| k=32, group 64, recalibrated ridge | 34.24%, 34.57%, 34.83% |
| k=32, group 256, recalibrated ridge | 12.94%, 12.85%, 13.63% |
| k=32, shuffled group 128 | 22.97%, 22.79%, 23.23% |
| k=32, document-balanced group 128 | 23.09%, 22.70%, 23.35% |
| k=32, ridge 0.382 | 24.49%, 24.67%, 25.24% |
| k=32, ridge 4.681 | 24.92%, 24.83%, 25.47% |

Sign robustness is strong. Magnitude robustness is false for group size: the unconfounded full-token comparison falls from about 24% at n=128 to 13% at n=256. The n=64 endpoint is additionally confounded by the evaluator's group cap, but this does not affect the conclusion that each group size defines a different sample-space ridge estimand.

### Frozen IOI test at 64 features

| Method vs paired MSE | Seed | IOI-effect difference (95% CI) | Collateral-KL difference (95% CI) | Continuous-target R2 difference |
| --- | ---: | ---: | ---: | ---: |
| DPSAE | 0 | +0.200 [+0.169, +0.231] | -0.00077 [-0.00388, +0.00233] | -0.256 |
| DPSAE | 1 | -0.481 [-0.532, -0.429] | -0.00962 [-0.01339, -0.00597] | -1.903 |
| DPSAE | 2 | -1.086 [-1.136, -1.035] | -0.01369 [-0.01772, -0.01001] | +0.013 |
| Whitening | 0 | +0.482 [+0.437, +0.528] | +0.00908 [+0.00570, +0.01251] | +0.047 |
| Whitening | 1 | +0.429 [+0.381, +0.476] | -0.00151 [-0.00557, +0.00247] | +0.162 |
| Whitening | 2 | -1.269 [-1.320, -1.217] | -0.01470 [-0.01849, -0.01115] | -0.024 |
| Static spectral | 0 | -0.127 [-0.157, -0.096] | -0.00313 [-0.00645, +0.00021] | +0.029 |
| Static spectral | 1 | -0.161 [-0.209, -0.111] | -0.00778 [-0.01165, -0.00398] | +0.150 |
| Static spectral | 2 | -0.728 [-0.777, -0.677] | -0.01310 [-0.01704, -0.00932] | -0.203 |

DPSAE's lower KL in seeds 1 and 2 comes with substantially lower IOI effect, so this is an effect-versus-damage movement rather than evidence of causal specificity. Exposure interpolation also fails to give a consistent rescue: seed 0 has no common support; in seed 1 DPSAE is better only when matched on firing frequency and slightly worse on activation mass and decoded energy; seed 2 favors DPSAE on all three but still at much lower IOI effect.

Every 64-feature sparse-code R2 is negative. The original dense activation gives R2 = -3.0968 and every full SAE reconstruction is also negative under the fixed ridge-0.01 protocol. This invalidates the intended architecture comparison, but it does not prove that the target is intrinsically non-linearly-decodable: ridge was not selected on the intermediate split, and the result may reflect regularization or split shift as well as target mismatch.

## Claim ledger

| Claim | Exact evidence | Status | Important caveat | Theoretical implication |
| --- | --- | --- | --- | --- |
| DPSAE robustly reduces truly held-out in-group decoder distortion. | Fresh 185M--190M tail; exact identity metric; three confirmation seeds each improve 24.23--24.39% with paired group CIs excluding zero. | established by this experiment | Conditional on GPT-2 block 8, BatchTopK, the selected weight, and groupwise transductive ridge operators; bootstrap is not over training seeds. | The paper can center prediction-operator preservation as an empirical representation result. |
| The gain is beyond the tuned whitening and theorem-derived static spectral controls tested here. | Whitening worsens exact distortion 12.16--12.57%; spectral improves 0.37--0.91%; DPSAE improves about 24%. | established by this experiment | The experiment tests two static controls, not every covariance metric, and does not trace a matched-NMSE frontier. | The fixed omission-cost metric is insufficient; the reconstruction-dependent hat operator or sparse optimization may matter. |
| DPSAE preserves a harder held-out continuous IOI target. | All sparse R2 values are negative; dense original R2 is -3.0968 under the same protocol. | not resolved | The validity gate fails before comparing architectures; ridge 0.01 was fixed rather than selected. | No theorem-to-task bridge can be claimed from this target. |
| DPSAE has better causal specificity under a matched intervention and natural exposure. | Only seed 0 has the desired sign pattern, with KL CI crossing zero; seeds 1/2 lose IOI effect while lowering KL; exposure matching is mixed or unsupported. | contradicted | The operator and lag are matched, but effect magnitude is not; equal-effect matching was not performed. | Decoder geometry does not by itself imply frozen-model causal specificity. |
| The result depends on contiguous geometry groups. | At group size 128, shuffled and document-balanced reductions remain 22.7--23.4%, close to contiguous. | contradicted | This addresses evaluation grouping of fixed models, not retraining under different group policies. | Sequence contiguity is not necessary for the observed evaluation advantage at n=128. |
| The objective estimates a grouping-independent corpus geometry. | On the same full 16,384 tokens, recalibrated group size changes k=32 reduction from 23.66--24.38% (n=128) to 12.85--13.63% (n=256). | contradicted | The n=64 endpoint is additionally confounded by a 128-group cap and is not needed for this conclusion. | Theory must define a distribution over groups; no unique population operator follows without extra assumptions. |
| The effect is specific to BatchTopK. | Only BatchTopK is trained in 4b. | not resolved | TopK/JumpReLU or a rank-matched non-sparse model was not tested. | Sparse-nonlinear theory can identify mechanisms but cannot attribute causality to BatchTopK. |
| Three paired seeds establish generality. | Three initializations share data/probe streams and all reproduce the representation effect. | contradicted | Seeds show optimization repeatability, not variation across corpora, layers, models, architectures, or task families. | Theorems should be conditional and the empirical claim should say replicate, not general. |
| Experiment 4b motivates Fisher-pullback or activation-manifold theory for this paper. | Only the representation gate passes; causal specificity and harder-target gates fail. | contradicted | No manifold metric was measured. | Keep Fisher geometry as a distinction or follow-up, not a core contribution. |
| The spectral theorem explains the full BatchTopK result. | Static spectral control recovers at most 0.91% versus DPSAE's 24%; rank relaxation preserves the same singular subspace as MSE. | contradicted | The control is a local/static surrogate, so it does not identify which nonlinear mechanism supplies the gap. | The theorem is a boundary result; sparse active sets, nonorthogonality, and reconstruction-dependent geometry remain candidates. |

## Discrepancies and missing audit material

1. The checked-in summary says the full 4b suite is complete, but the complete baseline and IOI JSONs were absent locally and existed only on the RunPod at sprint start. Their hashes and queried rows are recorded above; the small raw artifacts should be copied to durable project storage after the sprint.
2. The IOI artifact has no exact code revision. It is structurally known to require `d985476` or later, but the paper's traceability requirement is not fully met.
3. The prose phrase “the target is not linearly recoverable” is too broad. The defensible statement is “the dense activation fails the frozen ridge-0.01 recoverability gate on the final split.”
4. The prose says geometry robustness shows the result is not an artifact of one group size. The sign claim is correct, but the magnitude varies by nearly a factor of three, so group-size dependence must remain visible.
5. The paper cannot say the gain “depends on” the dynamic decoder objective in a causal-exclusion sense. It can say the two preregistered static controls do not match it.
6. A bounded cache-only diagnostic was planned to select the dense continuous-target ridge on the 1,024-example intermediate split. The initial read-only SSH inspection hung and was aborted before any remote computation ran, so the fixed-ridge caveat remains unresolved and no new distribution-shift claim is justified.

## Empirical implications for the theory sprint

The central theory should explain exactly what finite-group prediction-operator distortion guarantees and why it remains transductive. The isotropic rank theorem should be presented as a negative boundary: it predicts no subspace reordering, and 4b shows that its static omission-cost surrogate explains little of the sparse nonlinear result. This surrogate is not the Fréchet first-order metric of the map from a representation to its ridge hat matrix. The group-size sensitivity makes any population-consistency theorem optional and assumption-heavy; a clean statement that group construction defines the estimand is more honest.

The highest-value immediate experiment is evaluation-only: replace the failed IOI target with a direction that is linear at the chosen activation by construction, freeze a dense recoverability gate, then match each method to the same held-out IOI effect before comparing collateral KL. That directly tests the missing bridge from refittable decoder preservation to frozen-model behavior without retraining SAEs.
