# Theory claim ledger

“Proved” means proved for the stated finite matrices. “Verified” means the algebra is accompanied by a deterministic numerical check. “Empirical” means supported only by Experiment 4b. None of these labels implies generalization beyond the audited model, layer, architecture, or corpus.

| Result | Status | Assumptions | Proof location | Counterexample audit | Empirical relevance | Novelty risk |
| --- | --- | --- | --- | --- | --- | --- |
| T1: exact average refitted-task disagreement | proved | Fixed group, positive ridge, task second moment \(\Sigma\) | `05_final_theory.md`, Theorem 1 | Singular support and relative targets checked | Direct interpretation of the exact primary metric | Known Harvey/GULP result specialized to DPSAE |
| T2: absolute ellipsoidal worst-case bound | proved | Tasks in \(\operatorname{range}\Sigma\), absolute error | `05_final_theory.md`, Theorem 1 | Relative-error family diverges | States the strongest uniform guarantee | Standard norm inequality |
| T3: zero distance iff equal row Grams | proved | \(\lambda>0,\Sigma\succ0\) | `05_final_theory.md`, Proposition 2 | Singular \(\Sigma\) and \(\lambda=0\) both break it | Defines the exact quotient geometry | Standard ridge-hat algebra |
| T4: activation MSE upper-bounds decoder distance | proved | Equal feature widths and finite activation operator norms | `05_final_theory.md`, Proposition 3 | Orthogonal rotation kills converse | Explains MSE as an anchor, not the gain | Standard perturbation lemma |
| T5: isotropic rank-\(r\) tail \(\sum_{i>r}q_i^2\) | proved | Arbitrary rank-\(r\) representation, not sparse SAE feasibility | `05_final_theory.md`, Theorem 4 | Rank deficiency, ties, \(n<d\), \(n>d\), ridge limits checked | Negative boundary for the mechanism | Specialized synthesis; uncertain standalone novelty |
| T6: static control equals full-mode omission price | proved | PCA-aligned mode deletion | `05_final_theory.md`, Corollary 5 | Sign flip and infinitesimal perturbation break general equivalence | Interprets the failed spectral baseline | Project-derived |
| T7: static control is the Fréchet metric | refuted | False even in one dimension | `03_failed_routes_and_counterexamples.md`, section 5 | Exact differential has different spectral coefficient | Prevents overreading the static baseline | None; false claim |
| T8: commuting prior ranks \(\omega_iq_i^2\) | proved | \(K\Sigma=\Sigma K\), sufficient candidate width | `05_final_theory.md`, Theorem 6 | Singular weights and score ties checked | Defines a mathematically possible future extension | Weighted approximation is known in spirit |
| T9: arbitrary prior admits the same mode ranking | refuted | Noncommuting full-rank \(\Sigma\) | `03_failed_routes_and_counterexamples.md`, section 6 | Rotated valid hat costs \(0.00510<0.022\) | Not an explanation of isotropic 4b | None; false claim |
| T10: normalized-sphere numerator moments | proved | Independent radius-\(\sqrt n\) probes | `05_final_theory.md`, section 4 | Monte Carlo variance check | Corrects Experiment 3 transfer | Classical fourth-moment identity |
| T11: finite self-normalized ratio is unbiased | refuted | False with one group/probe; bias sign unrestricted | `03_failed_routes_and_counterexamples.md`, section 1 | Two attainable diagonal-hat examples | Separates training surrogate from exact evaluation | Classical ratio-estimator issue |
| T12: SGD targets the exact trace ratio | refuted | It targets \(\mathbb E[Z/D]\) under differentiation/interchange conditions | `05_final_theory.md`, section 4 | Random-denominator examples | Motivates gradient-fidelity audit | Standard stochastic-objective distinction |
| T13: fixed-law group-sum population limit | proved | Iid integrable groups, fixed \(n,P_n\), positive reference mean | `05_final_theory.md`, section 5 | Minibatch expectation differs from ratio of expectations | Makes grouping part of the method | Standard law of large numbers |
| T14: grouping-independent corpus geometry | refuted | Same rows and \(n\) can still differ by partition | `03_failed_routes_and_counterexamples.md`, section 3 | Exact 0-versus-2 regrouping example | Matches 24%-versus-13% group-size result | Project-specific counterexample |
| T15: fixed BatchTopK cell is affine with local rank bound | proved | Strict active ordering, away from ReLU/TopK boundaries | `05_final_theory.md`, section 6 | Positive ties give discontinuity | Only modest sparse mechanism statement | Standard piecewise-linear fact |
| T16: BatchTopK \(k\) equals matrix rank | refuted | Different rows may select different atoms | `03_failed_routes_and_counterexamples.md`, section 4 | High-rank sparse and sparse-infeasible low-rank examples | Blocks direct transfer of Theorem 4 | Project-specific synthesis |
| T17: ridge geometry and frozen/Fisher geometry control each other | refuted both ways | No alignment assumptions | `05_final_theory.md`, section 7 | Feature swap and downstream-null deletion | Explains mixed IOI result | Standard distinction; low novelty |
| E1: fresh exact 4b representation gain | established by this experiment | GPT-2 small block 8, BatchTopK, selected objective, one corpus tail | `00_empirical_audit.md` | Three paired seeds; exact identity targets | Central empirical result | Empirical combination appears project-specific |
| E2: tested static controls explain the gain | contradicted | Only selected whitening and spectral controls | `00_empirical_audit.md` | Spectral <1%, whitening worse, DPSAE ~24% | Localizes an unresolved sparse nonlinear gap | No universal exclusion |
| E3: advantage is group-size invariant | contradicted | Same 16,384 tokens for \(n=128,256\) | `00_empirical_audit.md` | \(n=64\) endpoint separately confounded | Forces finite-group wording | No novelty claim |
| E4: better frozen causal specificity | not resolved | IOI effect not matched across methods | `00_empirical_audit.md` | Seeds move along effect–KL tradeoff | Keeps Fisher outside core | No novelty claim |
| E5: harder continuous target is better preserved | not resolved | Dense activation fails fixed-ridge gate | `00_empirical_audit.md` | All sparse and dense \(R^2\) negative | Cannot support task-general claim | No novelty claim |

## Claims suitable for the main paper

The main paper can safely use T1–T3 as the interpretation of the objective, T6 as the rank-relaxed boundary theorem, and E1–E3 as the central experimental story. T11–T15 belong in an estimator/grouping appendix because they prevent the method from being described as an unbiased, grouping-free population metric. T16–T19 are limitations and mechanism exclusions, not positive explanations of the 24% effect.

## Claims that require new evidence

- A BatchTopK-specific mechanism requires an architecture ablation with the same decoder term.
- Frozen functional preservation requires a target that passes a dense recoverability gate and an equal-effect intervention comparison.
- Any matched-NMSE superiority claim requires a frontier rather than one selected objective weight.
- Any population generality claim requires more layers, models, corpora, and architectures.
- Any Fisher or activation-manifold claim requires its geometry to be measured and compared directly; Experiment 4b does not supply that evidence.
