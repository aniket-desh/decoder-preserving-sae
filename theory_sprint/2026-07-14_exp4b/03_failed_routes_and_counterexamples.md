# Failed routes and counterexamples

The sprint treated failed claims as results because they determine the paper's safe scope. Every construction below is exact unless a numerical value is explicitly attributed to `checks/verify_theory.py`.

## 1. “Finite normalized probes are an unbiased exact trace-ratio estimator”

Let

\[
K=\operatorname{diag}(0.8,0.2),\qquad
\widehat K=\operatorname{diag}(0.4,0.2).
\]

With one radius-\(\sqrt2\) spherical probe, the exact ratio of traces is \(0.235294\ldots\), while the expected per-probe ratio is \(0.2\). Replacing \(\widehat K\) by \(\operatorname{diag}(0.8,0)\) makes the exact ratio \(0.0588235\ldots\) while the expected probe ratio remains \(0.2\). The bias can therefore have either sign. Experiment 4b's fresh exact identity evaluation protects the headline result from this failure, but SGD still optimizes an expected self-normalized finite-probe objective rather than the exact trace ratio.

## 2. “The relative trace loss gives uniform relative task control”

Take

\[
K_\epsilon=\operatorname{diag}(1/2,\epsilon),\qquad
\widehat K_\epsilon=\operatorname{diag}(1/2,\epsilon+\sqrt\epsilon).
\]

The isotropic global relative trace tends to zero as \(4\epsilon\), but for target \(e_2\) the squared error relative to \(\|K_\epsilon e_2\|^2\) diverges as \(1/\epsilon\). The surviving theorem controls absolute error on a task ellipsoid; a relative per-task claim needs a lower bound on the reference prediction.

## 3. “The decoder objective defines one grouping-independent corpus geometry”

Use four scalar-feature rows

\[
X=(1,1,1,1)^\top,\qquad
\widehat X=(1,1,-1,-1)^\top
\]

and groups of two. Grouping rows as \(\{1,2\},\{3,4\}\) gives zero loss because each reconstructed group differs only by a sign. Grouping them as \(\{1,3\},\{2,4\}\) makes original and reconstructed group directions orthogonal and gives exact relative loss 2 for every positive ridge. The same rows and group size can therefore define incompatible operators solely through partition choice.

## 4. “BatchTopK sparsity \(k\) is the rank \(r\) in the spectral theorem”

With two rows, two atoms, \(k=1\), code matrix \(I_2\), and decoder \(I_2\), every row is one-sparse while the reconstructed matrix has rank two. Conversely, with \(p\) nonnegative one-sparse atoms and a free bias, reconstructions lie on a union of \(p\) rays; \(2p+1\) planar points in general position have matrix rank at most two but cannot all lie on those rays. Neither implication between sparse feasibility and low matrix rank holds.

## 5. “The static spectral loss is the Fréchet metric of the ridge-hat map”

For \(G=XX^\top\), \(\tau=n\lambda\), and perturbation \(\dot X\),

\[
\dot K=\tau(G+\tau I)^{-1}
(X\dot X^\top+\dot X X^\top)
(G+\tau I)^{-1}.
\]

Along a singular-value perturbation, the squared local coefficient is proportional to \(4\tau^2\sigma_i^2/(\sigma_i^2+\tau)^4\). The implemented static feature weight instead gives \(n\sigma_i^2/(\sigma_i^2+\tau)^2\). It exactly prices deletion of a whole singular mode, up to a common factor, but it is not a local differential metric. Its failure in Experiment 4b rules out that omission-cost control, not every static quadratic approximation.

## 6. “The non-isotropic theorem is always a weighted eigenmode ranking”

For

\[
K=\operatorname{diag}(0.9,0.2),\qquad
\Sigma=\begin{pmatrix}1&0.8\\0.8&1\end{pmatrix},
\]

the best rank-one coordinate-aligned hat has objective \(0.04\). Direct one-dimensional optimization finds a rotated direction at about \(8.95^\circ\), eigenvalue about \(0.8173\), and objective about \(0.01772\). Mode ranking is exact in the commuting case only; the noncommuting problem is a weighted positive-semidefinite low-rank approximation.

## 7. “Small ridge-hat distance preserves a frozen downstream network”

Let \(X=I_2\) and \(\widehat X=XQ\), where \(Q\) swaps feature coordinates. The row Grams and ridge hats are identical, but a frozen weight \(w=Me_1\) changes predictions from \((M,0)^\top\) to \((0,M)^\top\). The disagreement is unbounded with \(M\) despite zero decoder distance. This is the sharpest failed claim because it blocks the tempting jump from the confirmed representation metric to causal or output preservation.

## 8. “Fisher/output preservation implies ridge-task preservation”

Let a frozen downstream map read only coordinate one, take \(X=I_2\), and \(\widehat X=\operatorname{diag}(1,0)\). Frozen outputs agree and the reconstruction error lies in a Fisher-null direction, but one ridge-hat eigenvalue disappears, giving positive decoder distance. Fisher and refittable ridge geometry are incomparable without additional alignment assumptions.

## 9. “Small decoder distance implies small activation MSE, or conversely”

Any nontrivial orthogonal feature rotation \(\widehat X=XQ\) preserves the row Gram exactly and therefore has zero decoder distance, while its coordinate MSE can be large. The resolvent argument supplies only the other direction: bounded MSE plus bounded activation operator norms upper-bounds decoder distance. This is why the MSE anchor remains necessary even though it cannot explain the DPSAE advantage.

## 10. “Experiment 4b establishes causal specificity or a harder target”

The claim fails empirically before theory enters. DPSAE lowers collateral KL in seeds 1 and 2 while also substantially lowering the IOI effect, and its continuous-target sparse \(R^2\) values are negative. The original dense activation also has negative \(R^2\) under fixed ridge \(0.01\), so the target protocol fails its validity gate. The safe result is representation-level ridge-hat preservation only.
