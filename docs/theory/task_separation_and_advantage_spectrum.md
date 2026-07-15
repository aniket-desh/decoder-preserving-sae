# Equal-reconstruction separation and taskwise decoder-advantage spectra

**Status:** theorem-ready theory note; this is not manuscript prose.  All
statements below concern the exact identity-target audit, not the stochastic
training estimator.

## 1. Fixed conventions

Raw activations are transformed once using calibration-set statistics,

\[
x_{\mathrm{normalized}}=\frac{x_{\mathrm{raw}}-\mu_{\mathrm{cal}}}
{s_{\mathrm{cal}}},
\]

where \(\mu_{\mathrm{cal}}\) is a featurewise mean and
\(s_{\mathrm{cal}}>0\) is one scalar root-mean-square scale.  The matrices
below are already in these normalized coordinates.  Geometry groups are not
centered or rescaled again, and the regression has no separately fitted
intercept.

For a group \(X\in\mathbb R^{n\times d}\) and the average-loss ridge
objective

\[
\frac1n\lVert Xw-y\rVert_2^2+\lambda\lVert w\rVert_2^2,
\qquad \lambda>0,
\]

define

\[
K_\lambda(X)
=X(X^\top X+n\lambda I_d)^{-1}X^\top.
\]

Thus the matrix regularizer is \(\rho=n\lambda\).  Every comparison uses the
same \(\lambda\), preprocessing, samples, and row order for the source and all
reconstructions in a group.  A different calibrated ridge may be used for a
different group size, but not for different methods within the same
comparison.

## 2. Equal MSE and equal sparsity do not order task fidelity

### Proposition 1 (two-sample task-separation witness)

Fix \(\lambda>0\), put \(n=d=2\) and \(\rho=2\lambda\), and choose
\(a,b,\delta>0\).  Let

\[
X=\begin{pmatrix}a&0\\0&b\end{pmatrix},\qquad
Z_1=\begin{pmatrix}a+\delta&0\\0&b\end{pmatrix},\qquad
Z_2=\begin{pmatrix}a&0\\0&b+\delta\end{pmatrix}.
\]

Both reconstructions can be realized by a nonnegative sparse autoencoder with
zero decoder bias, unit-norm identity decoder \(D=I_2\), and codes
\(C_i=Z_i\).  Each sample has exactly one active latent, so both models have
average \(L_0=1\).  They also have identical reconstruction error,

\[
\lVert X-Z_1\rVert_F^2=\lVert X-Z_2\rVert_F^2=\delta^2,
\]

and hence identical MSE and NMSE under any common element count and source
normalization.  Nevertheless, neither reconstruction preserves every ridge
readout at least as well as the other.

#### Proof

For \(t\ge 0\), define

\[
\kappa_\rho(t)=\frac{t^2}{t^2+\rho}.
\]

Direct substitution gives

\[
K_\lambda(X)=\operatorname{diag}(\kappa_\rho(a),\kappa_\rho(b)),
\]

\[
K_\lambda(Z_1)=\operatorname{diag}(\kappa_\rho(a+\delta),\kappa_\rho(b)),
\quad
K_\lambda(Z_2)=\operatorname{diag}(\kappa_\rho(a),\kappa_\rho(b+\delta)).
\]

Let \(A_i=K_\lambda(X)-K_\lambda(Z_i)\) and
\(\eta_\rho(t)=\kappa_\rho(t+\delta)-\kappa_\rho(t)\).  Since
\(\kappa_\rho\) is strictly increasing on \((0,\infty)\),
\(\eta_\rho(a),\eta_\rho(b)>0\), and

\[
A_1=\operatorname{diag}(-\eta_\rho(a),0),\qquad
A_2=\operatorname{diag}(0,-\eta_\rho(b)).
\]

For the two unit sample-space tasks \(e_1,e_2\),

\[
\lVert A_1e_1\rVert_2^2=\eta_\rho(a)^2>0,
\qquad
\lVert A_2e_1\rVert_2^2=0,
\]

whereas

\[
\lVert A_1e_2\rVert_2^2=0,
\qquad
\lVert A_2e_2\rVert_2^2=\eta_\rho(b)^2>0.
\]

Thus \(Z_2\) is strictly better for \(e_1\) and \(Z_1\) is strictly better
for \(e_2\), despite exact MSE and sparsity matching. \(\square\)

The construction is stated after the repository's fixed preprocessing.  This
is admissible because that preprocessing imposes no zero-mean or unit-RMS
constraint on each individual geometry group.

### Corollary 1 (equal aggregate distortion can hide complete task exchange)

If \(a=b\), then

\[
\lVert A_1\rVert_F^2=\lVert A_2\rVert_F^2=\eta_\rho(a)^2.
\]

The models therefore match in MSE, \(L_0\), and exact isotropic decoder
distortion, yet each is exact on a task where the other has positive error.
An aggregate decoder score does not identify which tasks carry that error.

### Corollary 2 (positive aggregate advantage need not imply taskwise dominance)

Label \(Z_1\) as the baseline reconstruction and \(Z_2\) as the candidate.
Their task-advantage operator is

\[
Q=A_1^\top A_1-A_2^\top A_2
=\operatorname{diag}(\eta_\rho(a)^2,-\eta_\rho(b)^2).
\]

For fixed \(a,\delta,\rho>0\), \(\eta_\rho(a)>0\) while
\(\eta_\rho(b)\to0\) as \(b\to\infty\).  Hence one can choose \(b\) so
that \(\operatorname{tr}Q>0\) while \(\lambda_{\min}(Q)<0\).  The candidate
then has lower aggregate decoder error but is strictly worse on task \(e_2\).

For a concrete check, take \(\rho=1\), \(a=1\), \(b=2\), and
\(\delta=0.1\).  Then

\[
Q\approx\operatorname{diag}(2.2573\times10^{-3},
-2.2974\times10^{-4}),
\]

so the trace is positive and the operator is indefinite.

## 3. Exact taskwise advantage identities

### Theorem 2 (advantage spectrum for one geometry group)

Fix a source group \(X\) and two reconstructions \(Z_M,Z_D\), evaluated with
the conventions in Section 1.  Define

\[
A_M=K_\lambda(X)-K_\lambda(Z_M),\qquad
A_D=K_\lambda(X)-K_\lambda(Z_D),
\]

and

\[
Q=A_M^\top A_M-A_D^\top A_D.
\]

Then:

1. For every target \(y\in\mathbb R^n\),
   \[
   y^\top Qy
   =\lVert A_My\rVert_2^2-\lVert A_Dy\rVert_2^2.
   \]
   Positive values mean that \(D\) has smaller absolute prediction error for
   that target.
2. For unit-norm targets,
   \[
   \max_{\lVert y\rVert=1}y^\top Qy=\lambda_{\max}(Q),\qquad
   \min_{\lVert y\rVert=1}y^\top Qy=\lambda_{\min}(Q).
   \]
   Maximizers and minimizers are the corresponding eigenspaces.
3. The candidate weakly dominates the baseline on every target if and only if
   \(Q\succeq0\).  It is strictly better on every nonzero target if and only
   if \(Q\succ0\).
4. The trace is the isotropic average advantage:
   \[
   \operatorname{tr}Q
   =\lVert A_M\rVert_F^2-\lVert A_D\rVert_F^2
   =\mathbb E_{g\sim\mathcal N(0,I_n)}[g^\top Qg].
   \]
   If \(u\) is uniform on the unit sphere, then
   \(\mathbb E[u^\top Qu]=\operatorname{tr}Q/n\).

#### Proof

The first identity follows by expanding
\(\lVert A_my\rVert_2^2=y^\top A_m^\top A_my\) and subtracting.  The matrix
\(Q\) is real symmetric, so the two extremal identities are the Rayleigh--Ritz
theorem.  The equivalence between universal nonnegative quadratic form and
positive semidefiniteness proves the dominance statement.  Finally,

\[
\operatorname{tr}(A_m^\top A_m)=\lVert A_m\rVert_F^2,
\]

and for any zero-mean isotropic random vector with second moment \(I_n\),

\[
\mathbb E[g^\top Qg]
=\operatorname{tr}(Q\mathbb E[gg^\top])
=\operatorname{tr}Q.
\]

For a uniform unit vector, \(\mathbb E[uu^\top]=I_n/n\). \(\square\)

### Theorem 3 (exact reconciliation with the ratio-of-sums headline)

Let groups be indexed by \(g=1,\ldots,G\).  Group sizes and calibrated ridges
may vary with \(g\), provided both methods use the same samples and ridge
within each group.  Define

\[
N_{m,g}=\lVert A_{m,g}\rVert_F^2,
\qquad
S_g=\lVert K_{\lambda_g}(X_g)\rVert_F^2,
\qquad m\in\{M,D\},
\]

and assume \(S=\sum_gS_g>0\).  The exact headline distortion for method \(m\)
is

\[
R_m=\frac{\sum_gN_{m,g}}{\sum_gS_g}.
\]

For \(Q_g=A_{M,g}^\top A_{M,g}-A_{D,g}^\top A_{D,g}\),

\[
\boxed{
R_M-R_D=\frac{\sum_g\operatorname{tr}Q_g}{\sum_gS_g}.
}
\]

If \(N_M=\sum_gN_{M,g}>0\), the paired reduction reported by the exact audit
is

\[
\boxed{
1-\frac{\sum_gN_{D,g}}{\sum_gN_{M,g}}
=\frac{\sum_g\operatorname{tr}Q_g}{\sum_gN_{M,g}}
=\frac{R_M-R_D}{R_M}.
}
\]

#### Proof

Theorem 2 gives

\[
\operatorname{tr}Q_g=N_{M,g}-N_{D,g}.
\]

Summing over groups and dividing by the common source denominator \(S\) gives
the first identity.  Dividing the same summed difference by the positive
baseline numerator \(N_M\) gives the second. \(\square\)

Consequently, a positive paired headline reduction is equivalent to positive
**total** trace.  It does not imply positive trace in every group, positive
advantage on every task, or \(Q_g\succeq0\).

### Useful sanity bounds and invariances

Every ridge hat matrix is a symmetric positive-semidefinite contraction.  Its
nonzero eigenvalues are

\[
\frac{\sigma_i(X)^2}{\sigma_i(X)^2+n\lambda}\in[0,1).
\]

Therefore \(\lVert A_m\rVert_{\mathrm{op}}\le1\),
\(0\preceq A_m^\top A_m\preceq I\), and every eigenvalue of \(Q\) lies in
\([-1,1]\).  A simultaneous permutation of the source and both
reconstructions conjugates \(Q\) by the same permutation, preserving its
eigenvalues and trace.

These statements are for absolute task error at fixed \(\lVert y\rVert_2\).
If one instead normalizes each task by its source prediction energy, the
quantity is

\[
\frac{y^\top Qy}{y^\top K_\lambda(X)^2y},
\]

when the denominator is positive.  Its extrema form a generalized eigenvalue
problem, not the ordinary eigendecomposition of \(Q\), and the ratio is
undefined on the nullspace of \(K_\lambda(X)\).  The repository's exact
ratio-of-sums headline does not average these per-task ratios.

## 4. Adversarial audit cases and claim gates

1. **Trace reconciliation is a hard correctness check.**  The spectrum audit
   must reproduce
   \(\sum_g\operatorname{tr}Q_g=\sum_gN_{M,g}-\sum_gN_{D,g}\) in the same
   groups, ridge, preprocessing, and checkpoints.  Failure beyond a
   dtype-scaled numerical tolerance invalidates the audit before any spectrum
   is interpreted.
2. **The current positive exact reduction predicts positive total trace.**  A
   nonpositive total trace for any claimed-positive paired seed falsifies
   artifact alignment or sign conventions; it is not a new empirical result.
3. **The allocation hypothesis predicts indefiniteness.**  Material positive
   and negative eigenvalues would show that the objectives exchange fidelity
   across tasks.  If every \(Q_g\) is positive semidefinite within numerical
   tolerance, the stronger uniform-dominance description is accurate and the
   separation witness remains only a logical possibility.  If every spectrum
   is numerically rank one or its sign pattern changes arbitrarily across
   seeds, do not claim a stable task-allocation mechanism.
4. **Positive trace can be an outlier effect.**  Report groupwise traces and
   cumulative positive-trace concentration.  If one group supplies more than
   half of the total positive trace, describe the headline as concentrated and
   do not infer broad groupwise preservation.
5. **Repeated or nearly repeated eigenvalues do not identify individual
   eigentasks.**  When an extreme eigengap is below the perturbation scale,
   report the invariant extreme subspace or projector.  Sign-canonicalizing one
   unstable eigenvector does not make it reproducible.
6. **Small source energy makes task-relative claims unstable.**  Always report
   \(\lVert K_Xy\rVert^2\) for selected directions.  A large absolute
   advantage on a near-null source task cannot support a relative-readout
   claim.
7. **Grouping is part of the estimand.**  Eigenvectors from different groups
   live in different sample coordinates and cannot be averaged.  Only their
   spectra, traces, or values of a separately specified transferable target
   constructor are directly comparable.
8. **The exact spectrum does not validate the training gradient.**  None of
   these identities replace the implemented fixed-radius, self-normalized
   sampled objective by the identity-target ratio.  Stochastic-gradient
   fidelity requires a separate audit.

## 5. Compact claim-ledger entries

| Claim ID | Theorem-ready claim | Assumptions | Empirical prediction or check | Falsifier / scope boundary | Status |
|---|---|---|---|---|---|
| TH-SEP-01 | Equal reconstruction MSE and equal code \(L_0\) do not order fixed-ridge task errors; this remains true for nonnegative one-active-coordinate codes and a unit identity decoder. | Fixed global preprocessing, \(\lambda>0\), no group centering, refitted ridge tasks. | Matched NMSE/L0 need not make the empirical advantage operator PSD. | This is an existence theorem, so PSD empirical spectra limit its mechanistic relevance but do not falsify it. | Proved above. |
| TH-SPEC-01 | The eigenvalues of \(Q=A_M^\top A_M-A_D^\top A_D\) are the extremal absolute task-error advantages, and \(Q\succeq0\) exactly characterizes weak dominance over all tasks. | Same source rows, ridge, preprocessing, and target norm. | Positive and negative eigenvalues certify taskwise exchange; PSD certifies uniform weak dominance within the audited group. | No semantic, held-out, frozen-network, or per-task-relative conclusion follows from an eigenvalue alone. | Proved above. |
| TH-SPEC-02 | Summed spectral trace exactly reconciles with the exact ratio-of-sums distortion difference and paired reduction. | Identity targets; common source denominator; baseline numerator positive for paired reduction. | Trace and stored numerator differences must agree numerically for every seed and audit slice. | Does not apply to the sampled self-normalized training gradient or to a mean of groupwise ratios. | Proved above. |
| TH-CTRL-01 | A JumpReLU decoder-effect comparison is identified only after a decoder-blind controller setting passes the frozen sparsity and health gates at full horizon. | Shared grid and training budget; no decoder, NMSE, or language-model outcome access during selection. | Both models must satisfy the common \(L_0\) band, pairwise mismatch, drift, finite-state, provenance, and dead-feature gates. | Failure is evidence about controller feasibility, not the sign of a DPSAE effect. | Evaluated; no setting passed. |

## 6. JumpReLU validity-gate outcome

The final study trains both objectives for 25,001,984 tokens at the same
predeclared sparsity-loss weights \(2,4,8,16\). Selection reads only held-out
\(L_0\), late-window drift, finite state, threshold health, dead features, and
run provenance. Decoder distortion, NMSE, and language-model loss remain
sealed. A setting advances only if both \(L_0\) confidence intervals lie in
\([30.4,33.6]\), the inter-method gap is at most \(1.6\), late drift passes,
and the maximum dead fraction is at most \(10\%\).

No weight passes. The DPSAE/MSE \(L_0\) pairs are \(37.73/38.55\),
\(37.38/36.93\), \(36.10/36.11\), and \(35.88/35.61\), while maximum dead
fractions range from \(48.5\%\) to \(54.4\%\). The attempted JumpReLU
comparison is therefore unidentified. This is a negative
controller-feasibility result, not evidence that DPSAE helps or hurts under
JumpReLU, and the grid is not extended after inspection.
