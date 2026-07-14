# Paper narrative after Experiment 4b

## The paper this result supports

Standard SAEs minimize coordinate reconstruction error, but two activation tables with similar MSE can support different families of refitted linear decoders. DPSAE adds a task-agnostic ridge prediction-operator penalty: within each finite activation group, it compares the predictions obtained after fitting the same target from the original activations and from the SAE reconstruction. Experiment 4b shows that this objective changes the reconstruction frontier in the metric it directly targets.

The headline is narrow and reproducible. On GPT-2 small block 8 with a 16,384-latent BatchTopK SAE at \(k=32\), DPSAE reduces exact held-out finite-group ridge-hat distortion by 24.23–24.39% relative to paired MSE SAEs across three new seeds, at a 6.91–7.11% NMSE cost. A tuned whitening control worsens decoder distortion by roughly 12%, and a theorem-derived static spectral control improves it by less than 1%. Robustness fleets preserve the sign across \(k\in\{16,32,64\}\), ridge settings, and tested group constructions.

The theory explains exactly what this metric buys. Its unnormalized value is the expected squared disagreement of separately refitted ridge predictions over a declared distribution of in-group targets, and it upper-bounds worst-case absolute disagreement on the corresponding target ellipsoid. With full task support and positive ridge, zero distance means equal row Gram matrices. This is a representation-level guarantee on the observed group, not a frozen-network or causal guarantee.

## Why the rank theorem belongs in the paper

Under an isotropic task prior and an unconstrained rank-\(r\) reconstruction, decoder preservation retains the same left singular directions as PCA. The only change is the omission cost, from \(\sigma_i^2\) to

\[
q_i^2=\left(\frac{\sigma_i^2}{\sigma_i^2+n\lambda}\right)^2.
\]

That theorem is useful because it rules out the easiest story. A static residual weighted to reproduce those omission costs captures less than 1% improvement, so the 24% DPSAE effect is not quantitatively explained by simply reweighting PCA modes. The unresolved mechanism lies in the difference between the relaxation and the actual shared overcomplete sparse model: active-set allocation, nonorthogonality, reconstruction-dependent row Grams, or optimization.

## What Experiment 4b removes from the main claim

The frozen IOI evaluation does not show causal specificity. Two DPSAE seeds lower collateral KL while also lowering the IOI effect substantially, and exposure matching is inconsistent. The continuous target also fails its dense-activation validity gate under the fixed ridge protocol. Those results are scientifically useful because they separate refittable linear information from frozen-model use, but they cannot support a functional-preservation claim.

Group size must also be visible in the method. On the same 16,384 exact tokens, the paired advantage falls from about 24% at groups of 128 to about 13% at groups of 256 even after ridge recalibration. The objective is therefore a distribution over finite-group operators rather than an unbiased view of one grouping-independent corpus operator. The 64-token endpoint is additionally confounded by an evaluation cap and should not carry the argument.

## Recommended claim hierarchy

1. **Primary empirical claim.** DPSAE improves exact held-out finite-group ridge prediction-operator preservation at fixed sparsity, beyond paired MSE and the two tested static controls.
2. **Primary theoretical claim.** The objective exactly controls average refitted ridge-task disagreement and admits a sharp isotropic rank-relaxation theorem.
3. **Mechanism result.** The rank theorem and its static omission-cost control are insufficient to explain the sparse nonlinear gain.
4. **Estimator qualification.** Normalized finite probes give a self-normalized stochastic surrogate, while the exact metric is used for confirmation.
5. **Scope boundary.** Refitted decoder preservation is distinct from frozen output, Fisher, causal, and activation-manifold geometry.

## Main text versus appendix

The main text should contain Theorem 1's average-task identity and absolute ellipsoidal bound, the isotropic rank theorem, and the one-paragraph interpretation of the static omission-cost control. Those results directly interpret the objective and the main comparison figure. The row-Gram zero set and activation-MSE bound can be short propositions or remarks in the main text if space allows.

The appendix should contain the full projection proofs, tie/rank/ridge edge cases, normalized-sphere moment and ratio-bias analysis, grouping counterexample and population limit, commuting structured-prior theorem, noncommuting counterexample, and BatchTopK/Fisher separations. They are essential for correctness but do not all advance the central empirical story at first reading.

Activation-manifold geometry belongs in follow-up work, not the main text or a theory appendix. Fisher geometry belongs in the discussion as the precise frozen-model geometry that DPSAE does not control, together with the two-way counterexamples and the failed IOI gate.

## When the method should and should not help

DPSAE should help when the desired property is robust access by a family of ridge-regularized linear decoders that may be refit on reconstruction, and when the deployment groups resemble the groups used to define the objective. It is especially plausible when coordinate MSE allocates sparse capacity poorly relative to the saturated ridge-hat spectrum.

It should not be expected to help when success depends on exact compatibility with frozen downstream weights, on target directions outside the chosen task covariance, on group scales unlike training, or on nonlinear/manifold information invisible to the protected linear task family. The MSE anchor also remains necessary because decoder geometry is invariant to feature rotations that a frozen model may not tolerate.

## Suggested figure logic

The main result figure should put exact decoder distortion and NMSE together for MSE, DPSAE, whitening, and static spectral controls, with paired seeds visible. A second panel should show the \(n=128\) versus \(n=256\) magnitude change and label the metric as finite-group. The IOI and continuous-target results belong in a scope/limitations figure or appendix, because they falsify a stronger interpretation rather than support the primary claim.

## Safe wording

Use “preserves held-out finite-group ridge-decodable information under refitting” rather than “preserves downstream behavior” or “preserves information used by the model.” Use “the two tested static controls do not explain the gain” rather than “dynamic geometry is necessary.” Use “replicates across three paired initializations” rather than “generalizes,” because no new model, layer, corpus family, or SAE architecture was tested.

The most defensible contribution is the combination of an objective, an exact finite-group interpretation, a boundary theorem that rejects a tempting linear explanation, and a confirmation experiment that survives fresh seeds and exact targets. The paper becomes weaker if it adds Fisher or activation-manifold geometry before a functional gate passes.
