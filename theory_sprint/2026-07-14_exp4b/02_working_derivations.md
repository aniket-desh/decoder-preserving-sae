# Working derivations

This file records the path from the Experiment 4b object to the promoted theorem set. It keeps useful intermediate identities and the points where the initial interpretation changed. Complete statements and proofs are in `05_final_theory.md`; falsified conjectures are in `03_failed_routes_and_counterexamples.md`.

## 1. Start from the implemented finite-group operator

For \(X\in\mathbb R^{n\times d}\), reconstruction \(Z\), and \(\tau=n\lambda>0\),

\[
K_X=X(X^\top X+\tau I)^{-1}X^\top
=XX^\top(XX^\top+\tau I)^{-1}.
\]

The row-Gram form immediately gives

\[
K_X=I-\tau(XX^\top+\tau I)^{-1},
\qquad
XX^\top=\tau K_X(I-K_X)^{-1}.
\]

This settled two early questions. Positive-ridge hats retain every nonzero row-Gram eigenvalue through an injective scalar transform, and equality of hats is exactly equality of row Grams. At zero ridge this fails because only the column-space projector remains.

For task second moment \(\Sigma\succeq0\), set

\[
A=K_X-K_Z,
\qquad
D_\Sigma^2=\operatorname{tr}(A\Sigma A^\top)
=\|A\Sigma^{1/2}\|_F^2.
\]

The trace identity gives \(D_\Sigma^2=\mathbb E\|Ay\|^2\) whenever \(\mathbb E[yy^\top]=\Sigma\). Writing \(y=\Sigma^{1/2}u\) also gives the ellipsoidal bound

\[
\|Ay\|^2
\le D_\Sigma^2,y^\top\Sigma^\dagger y,
\qquad y\in\operatorname{range}\Sigma.
\]

The exact worst case is the operator norm squared. The Frobenius objective is an average identity and a worst-case upper bound, not a worst-case equality.

## 2. The first failed upgrade: relative control

Normalizing by \(\|K_X\Sigma^{1/2}\|_F^2\) gives a relative average energy. It does not control

\[
\sup_y\frac{\|(K_X-K_Z)y\|^2}{\|K_Xy\|^2}
\]

unless the reference prediction is bounded away from zero on the protected task set. This forced the final theorem to use absolute ellipsoidal control. The two-dimensional family in `03_failed_routes_and_counterexamples.md` makes the trace ratio vanish while one target's relative error diverges.

## 3. How MSE relates to the operator

With \(G=XX^\top\), \(H=ZZ^\top\),

\[
K_X-K_Z
=\tau(H+\tau I)^{-1}(G-H)(G+\tau I)^{-1}.
\]

Both resolvents have operator norm at most \(1/\tau\), hence

\[
\|K_X-K_Z\|_F
\le\frac1{n\lambda}\|XX^\top-ZZ^\top\|_F.
\]

If the feature widths agree,

\[
XX^\top-ZZ^\top=(X-Z)X^\top+Z(X-Z)^\top,
\]

which yields the activation-error bound promoted as Proposition 3. The sign or rotation construction \(Z=XQ\) kills every converse. This explains why MSE is a sensible anchor but cannot be treated as equivalent to prediction-operator preservation.

## 4. Rank relaxation and the static control

For compact SVD \(X=U\operatorname{diag}(\sigma_i)V^\top\),

\[
K_X=U\operatorname{diag}(q_i)U^\top,
\qquad
q_i=\frac{\sigma_i^2}{\sigma_i^2+n\lambda}.
\]

Any rank-\(r\) reconstruction has a hat of rank at most \(r\), so ordinary low-rank approximation gives the lower bound \(\sum_{i>r}q_i^2\). Truncated SVD attains it. Because \(q_i\) increases strictly with \(\sigma_i\), isotropic decoder preservation does not reorder PCA modes.

The implemented static operator

\[
M_\lambda=C(C+\lambda I)^{-2},
\qquad C=X^\top X/n,
\]

assigns a deleted singular component the cost

\[
\sigma_i^2v_i^\top M_\lambda v_i=nq_i^2.
\]

This is exactly the rank theorem's full-mode omission cost up to a shared factor. The initial idea that it was the Fréchet metric was wrong. The true differential is

\[
\dot K
=\tau(G+\tau I)^{-1}
(X\dot X^\top+\dot X X^\top)
(G+\tau I)^{-1},
\]

which weights infinitesimal singular-value perturbations differently.

## 5. Structured priors

When \(K_X\) and \(\Sigma\) commute, write their shared eigenvalues as \(q_i\) and \(\omega_i\). Then

\[
K_X\Sigma^{1/2}
=U\operatorname{diag}(q_i\sqrt{\omega_i})U^\top.
\]

The rank-\(r\) lower bound keeps the largest \(\omega_iq_i^2\), and a diagonal ridge hat attains it. This gives an exact way for a task prior to reorder retained modes.

The extension fails without commutation. Mapping the best unconstrained approximation of \(K_X\Sigma^{1/2}\) back through \(\Sigma^{-1/2}\) generally produces a nonsymmetric matrix, which is not a ridge hat. The exact problem is

\[
\min_{\substack{0\preceq B\prec I\\\operatorname{rank}B\le r}}
\operatorname{tr}[(K_X-B)\Sigma(K_X-B)],
\]

and a rational two-dimensional example shows a rotated feasible hat beating both source eigenmodes.

## 6. The probe law changed the estimator analysis

Experiment 4b does not use unnormalized Gaussian targets. It maps \(g\sim\mathcal N(0,I_n)\) to

\[
y=\sqrt n\,g/\|g\|,
\]

which is uniform on a fixed-radius sphere. For symmetric \(B,N\),

\[
\mathbb E[y^\top By]=\operatorname{tr}B,
\]

\[
\operatorname{Cov}(y^\top By,y^\top Ny)
=\frac{2}{n+2}
\left(n\operatorname{tr}(BN)-\operatorname{tr}B\operatorname{tr}N\right).
\]

The radial-normalization subtraction term means Experiment 3's Gaussian variance formula is only an approximation for Experiment 4b.

More importantly, training computes one random ratio over every group and probe. The two sums are unbiased trace estimators, but their ratio is not unbiased. The sign of the bias depends on numerator–denominator covariance, and valid diagonal ridge hats give either sign. Consequently SGD targets \(\nabla\mathbb E[Z/D]\), not the gradient of the exact ratio of traces. Fresh identity-target evaluation is what makes the empirical headline robust to this distinction.

## 7. Group construction is mathematical, not clerical

For a fixed group law \(P_n\), a ratio over iid group sums converges to

\[
\mathcal R_{n,P_n}
=\frac{\mathbb E_{P_n}N_n}{\mathbb E_{P_n}R_n},
\]

not the mean of groupwise ratios. Changing the partition changes the block-diagonal collection of sample-space operators. A four-row example changes exact relative loss from 0 to 2 under two partitions with the same \(n\), and Experiment 4b changes from about 24% to 13% between \(n=128\) and \(n=256\) on the same token set. There is no grouping-free population object without an explicit asymptotic model.

## 8. Why the sparse mechanism remained open

On a fixed BatchTopK active cell, token \(i\)'s Jacobian is

\[
J_i=\sum_{\ell\in S_i}d_\ell(w^e_\ell)^\top,
\qquad
\operatorname{rank}J_i\le|S_i|.
\]

This is a local necessary capacity statement. It does not identify the global support allocation because supports change at ReLU and TopK boundaries and compete across the full batch. A one-sparse reconstruction can have high matrix rank when different rows select different atoms, while nonnegative ray constraints can exclude low-rank tables. The rank theorem therefore stays a boundary result.

## 9. Fisher decision

Ridge hats describe decoders that are refit after reconstruction. The Fisher pullback describes local sensitivity of a frozen downstream network. A feature rotation can keep ridge distance at zero while changing frozen logits arbitrarily; deleting a frozen-network-ignored coordinate can have zero Fisher cost while changing the ridge hat. The geometries are incomparable without additional assumptions, and Experiment 4b's mixed IOI result supplies no reason to add those assumptions now.

The final theory therefore keeps Fisher as a contrast and future measurement rather than extending the main claim.
