# Final theory after Experiment 4b

## Scope and central claim

The defensible theory is finite-group and transductive. For one activation group, DPSAE preserves the prediction operator obtained by fitting a ridge decoder on the original representation and refitting it on the reconstruction. This gives an exact guarantee for a stated family of in-group linear targets. It does not guarantee frozen-network behavior, causal specificity, new-row generalization, or a grouping-independent population geometry.

Experiment 4b makes this narrow object worth centering: the exact identity-target metric improves by 24.23–24.39% across three new paired seeds, while the two selected static controls do not approach that gain. The group-size dependence and mixed IOI results make every broader interpretation conditional.

## 1. Finite-group prediction-operator geometry

Let \(X\in\mathbb R^{n\times d}\) be an activation group, \(Z\in\mathbb R^{n\times p}\) a reconstruction, \(\lambda>0\) the average-loss ridge coefficient, and \(\tau=n\lambda\). Define

\[
G_X=XX^\top,
\qquad
K_X=K_\lambda(X)
=X(X^\top X+\tau I)^{-1}X^\top
=G_X(G_X+\tau I)^{-1}.
\]

The same definitions apply to \(Z\). The matrix \(K_X\) maps any target vector \(y\in\mathbb R^n\) to the in-sample predictions of a ridge decoder fit on \((X,y)\). For a positive-semidefinite task second moment \(\Sigma\), define

\[
D_\Sigma^2(X,Z)
=\|(K_X-K_Z)\Sigma^{1/2}\|_F^2.
\]

### Theorem 1: average and worst-case task control

If \(\mathbb E[yy^\top]=\Sigma\), then

\[
\mathbb E\|K_Xy-K_Zy\|_2^2=D_\Sigma^2(X,Z).
\]

Moreover,

\[
\sup_{\substack{y\in\operatorname{range}\Sigma\\
y^\top\Sigma^\dagger y\le1}}
\|(K_X-K_Z)y\|_2^2
=\|(K_X-K_Z)\Sigma^{1/2}\|_{\mathrm{op}}^2
\le D_\Sigma^2(X,Z).
\]

**Proof.** The expectation is the trace identity

\[
\mathbb E\|Ay\|^2
=\operatorname{tr}(A\mathbb E[yy^\top]A^\top)
=\|A\Sigma^{1/2}\|_F^2,
\qquad A=K_X-K_Z.
\]

Every point in the displayed ellipsoid is \(y=\Sigma^{1/2}u\) for some \(\|u\|\le1\). Maximizing gives the squared operator norm, which is at most the squared Frobenius norm. \(\square\)

The result is an absolute-error statement. The relative objective

\[
\frac{D_\Sigma^2(X,Z)}{\|K_X\Sigma^{1/2}\|_F^2}
\]

normalizes average energy, but it gives no uniform prediction-relative guarantee for targets on which \(K_Xy\) is small.

### Proposition 2: exact zero set

For \(\lambda>0\),

\[
K_X=I-\tau(G_X+\tau I)^{-1},
\qquad
G_X=\tau K_X(I-K_X)^{-1}.
\]

Thus the row-Gram-to-hat map is injective. If \(\Sigma\succ0\),

\[
D_\Sigma(X,Z)=0
\quad\Longleftrightarrow\quad
K_X=K_Z
\quad\Longleftrightarrow\quad
XX^\top=ZZ^\top.
\]

If \(\Sigma\) is singular, zero distance means only \((K_X-K_Z)\Sigma^{1/2}=0\). The positive-ridge condition matters: at \(\lambda=0\), hat matrices reduce to projections and lose singular-value information.

**Proof.** Starting from \(K_X=G_X(G_X+\tau I)^{-1}\), insert \(G_X=(G_X+\tau I)-\tau I\):

\[
K_X
=[(G_X+\tau I)-\tau I](G_X+\tau I)^{-1}
=I-\tau(G_X+\tau I)^{-1}.
\]

Every eigenvalue of \(K_X\) is \(g/(g+\tau)\in[0,1)\) for an eigenvalue \(g\ge0\) of \(G_X\), so \(I-K_X\) is invertible. From

\[
K_X(G_X+\tau I)=G_X
\]

we get \(\tau K_X=(I-K_X)G_X\). Because \(K_X\) commutes with \(G_X\), this rearranges to

\[
G_X=\tau K_X(I-K_X)^{-1}.
\]

Thus equal hats give equal row Grams, and the forward formula gives the converse. Finally,

\[
D_\Sigma=0
\Longleftrightarrow
(K_X-K_Z)\Sigma^{1/2}=0.
\]

If \(\Sigma\succ0\), \(\Sigma^{1/2}\) is invertible, so this is equivalent to \(K_X=K_Z\). If \(\Sigma\) is singular, it requires equality only on \(\operatorname{range}\Sigma\). \(\square\)

### Proposition 3: activation reconstruction is a one-way control

The resolvent identity gives

\[
K_X-K_Z
=\tau(G_Z+\tau I)^{-1}(G_X-G_Z)(G_X+\tau I)^{-1}.
\]

Therefore

\[
\|K_X-K_Z\|_F
\le \frac1{n\lambda}\|XX^\top-ZZ^\top\|_F.
\]

When \(X\) and \(Z\) have the same width,

\[
D_\Sigma^2(X,Z)
\le
\|\Sigma\|_{\mathrm{op}}
\frac{(\|X\|_{\mathrm{op}}+\|Z\|_{\mathrm{op}})^2}{(n\lambda)^2}
\|X-Z\|_F^2.
\]

The converse is false because \(Z=XQ\) for an orthogonal feature rotation preserves \(XX^\top\) exactly while its coordinate MSE may be large. MSE is therefore a sufficient but potentially loose anchor; it is not equivalent to decoder preservation.

**Proof.** Since \(K_X=I-\tau(G_X+\tau I)^{-1}\) and likewise for \(Z\), the inverse-difference identity

\[
A^{-1}-B^{-1}=A^{-1}(B-A)B^{-1}
\]

with \(A=G_Z+\tau I\), \(B=G_X+\tau I\) gives

\[
K_X-K_Z
=\tau(G_Z+\tau I)^{-1}(G_X-G_Z)(G_X+\tau I)^{-1}.
\]

Both row Grams are positive semidefinite, hence both inverse factors have operator norm at most \(1/\tau\). Submultiplicativity yields

\[
\|K_X-K_Z\|_F
\le \tau\cdot\frac1\tau\cdot\|G_X-G_Z\|_F\cdot\frac1\tau
=\frac1\tau\|G_X-G_Z\|_F.
\]

For \(E=X-Z\), direct expansion gives

\[
G_X-G_Z=XX^\top-ZZ^\top=EX^\top+ZE^\top.
\]

Using \(\|AB\|_F\le\|A\|_F\|B\|_{\mathrm{op}}\),

\[
\|G_X-G_Z\|_F
\le(\|X\|_{\mathrm{op}}+\|Z\|_{\mathrm{op}})\|E\|_F.
\]

Finally,

\[
D_\Sigma^2
=\|(K_X-K_Z)\Sigma^{1/2}\|_F^2
\le\|K_X-K_Z\|_F^2\|\Sigma\|_{\mathrm{op}},
\]

which gives the displayed result after substituting \(\tau=n\lambda\). \(\square\)

## 2. The exact rank-relaxed boundary

Let the compact singular value decomposition be

\[
X=U\operatorname{diag}(\sigma_1,\ldots,\sigma_s)V^\top,
\qquad
\sigma_1\ge\cdots\ge\sigma_s>0,
\]

and define

\[
q_i=\frac{\sigma_i^2}{\sigma_i^2+n\lambda}.
\]

Then \(K_X=U\operatorname{diag}(q_i)U^\top\).

### Theorem 4: isotropic rank-\(r\) optimum

For \(0\le r\le s\),

\[
\min_{\operatorname{rank}Z\le r}
\|K_X-K_Z\|_F^2
=\sum_{i>r}q_i^2.
\]

A truncated representation

\[
X_r=U_r\operatorname{diag}(\sigma_1,\ldots,\sigma_r)V_r^\top
\]

attains the minimum.

**Proof.** Every feasible \(B=K_Z\) has rank at most \(r\). Let \(P\) be the orthogonal projector onto \(\operatorname{col}B\), so \(PB=B\). Decompose

\[
K_X-B=(I-P)K_X+P(K_X-B).
\]

The summands are Frobenius-orthogonal because \((I-P)P=0\), hence

\[
\|K_X-B\|_F^2
=\|(I-P)K_X\|_F^2+\|P(K_X-B)\|_F^2
\ge\|(I-P)K_X\|_F^2.
\]

Writing \(a_i=u_i^\top Pu_i=\|Pu_i\|^2\),

\[
\|(I-P)K_X\|_F^2
=\operatorname{tr}[K_X(I-P)K_X]
=\sum_{i=1}^s q_i^2(1-a_i).
\]

The projector constraints give \(0\le a_i\le1\) and

\[
\sum_{i=1}^s a_i
\le\operatorname{tr}P
=\operatorname{rank}P
\le r.
\]

Because \(q_1^2\ge\cdots\ge q_s^2\), moving any available weight from a lower-scored \(a_j\) to a higher-scored unsaturated \(a_i\) cannot decrease \(\sum_iq_i^2a_i\). Its maximum under these constraints is therefore \(\sum_{i=1}^rq_i^2\). Consequently

\[
\|K_X-B\|_F^2
\ge\sum_{i=r+1}^s q_i^2.
\]

For the truncated representation \(X_r\), direct substitution into the hat formula gives

\[
K_{X_r}=\sum_{i=1}^r q_iu_iu_i^\top,
\]

so its squared error is exactly the tail sum. \(\square\)

Because \(q_i\) is strictly increasing in \(\sigma_i\), isotropic ridge-decoder preservation and rank-constrained MSE retain the same singular directions. They differ only in omission price: MSE pays \(\sigma_i^2\), while decoder distance pays the saturated cost \(q_i^2\). If \(q_r>q_{r+1}\), the optimal sample-space hat and row Gram are unique; ties permit rotations inside the tied left-singular subspace. Feature-space factorizations remain nonunique.

This theorem is a negative boundary result for DPSAE. BatchTopK's \(k\) is not matrix rank, and a single shared sparse dictionary across groups is not the feasible set above. The theorem cannot explain the 24% empirical gain by itself.

The proof treats \(n<d\) and \(n>d\) identically because it uses only the compact rank \(s\le\min(n,d)\). If \(r\ge s\), the minimum is zero; if \(X=0\), every \(q_i\) list is empty and \(Z=0\) attains zero. A strict cutoff \(q_r>q_{r+1}\) makes the optimal sample-space hat unique. A tie permits any required-dimensional subspace inside the tied left-singular eigenspace, while right-orthogonal feature factorizations remain nonunique.

The ridge limits at fixed \(X\) are

\[
n\lambda\downarrow0:
q_i\to1,
\qquad
\min\|K_X-K_Z\|_F^2\to s-r,
\]

and

\[
n\lambda\to\infty:
q_i^2
=\frac{\sigma_i^4}{(n\lambda)^2}+O((n\lambda)^{-3}).
\]

The small-ridge limit is a column-space projector result, not row-Gram equality at \(\lambda=0\). The large-ridge objective vanishes without rescaling and becomes a Schatten-4 tail after multiplication by \((n\lambda)^2\).

### Corollary 5: what the static spectral control prices

Let \(C=X^\top X/n\) and

\[
M_\lambda=C(C+\lambda I)^{-2}.
\]

Deleting singular component \(\sigma_i u_iv_i^\top\) incurs static residual cost

\[
\sigma_i^2v_i^\top M_\lambda v_i=nq_i^2.
\]

The static control therefore reproduces the rank theorem's full-mode omission costs up to a common factor. It is not the local Fréchet metric. For perturbation \(\dot X\),

\[
\dot K
=\tau(G_X+\tau I)^{-1}
(X\dot X^\top+\dot X X^\top)
(G_X+\tau I)^{-1},
\]

whose weighting of infinitesimal singular-value changes has a different spectral dependence. Experiment 4b shows that this omission-cost control recovers less than 1% improvement where DPSAE recovers about 24%; it does not eliminate every possible static quadratic control.

**Proof.** Since \(Cv_i=(\sigma_i^2/n)v_i\),

\[
M_\lambda v_i
=\frac{\sigma_i^2/n}{(\sigma_i^2/n+\lambda)^2}v_i
=\frac{n\sigma_i^2}{(\sigma_i^2+n\lambda)^2}v_i.
\]

The deleted residual has Frobenius energy \(\sigma_i^2\) along \(v_i\), so its weighted cost is

\[
\sigma_i^2\frac{n\sigma_i^2}{(\sigma_i^2+n\lambda)^2}
=n\left(\frac{\sigma_i^2}{\sigma_i^2+n\lambda}\right)^2
=nq_i^2.
\]

For the differential, differentiate \(K=I-\tau(G+\tau I)^{-1}\) and use \(d(A^{-1})=-A^{-1}(dA)A^{-1}\):

\[
dK=\tau(G+\tau I)^{-1}(dG)(G+\tau I)^{-1},
\qquad
dG=X(dX)^\top+(dX)X^\top.
\]

This depends on two sample-space resolvents and both Gram perturbation terms, so it is not the fixed feature-space residual form above. \(\square\)

## 3. Structured task priors

### Theorem 6: commuting priors reorder modes

Suppose \(K_X\) and \(\Sigma\) share eigenvectors \(u_i\), with eigenvalues \(q_i\) and \(\omega_i\ge0\). Then

\[
\min_{\operatorname{rank}Z\le r}D_\Sigma^2(X,Z)
=\sum_{i\notin S_r}\omega_iq_i^2,
\]

where \(S_r\) contains indices of the \(r\) largest scores \(\omega_iq_i^2\). An optimal hat keeps \(q_i u_i u_i^\top\) for \(i\in S_r\).

**Proof.** Any \(N=K_Z\Sigma^{1/2}\) has rank at most \(r\). In the shared basis, the target matrix is

\[
K_X\Sigma^{1/2}
=U\operatorname{diag}(q_i\sqrt{\omega_i})U^\top
\]

whose squared singular values are \(a_i^2=\omega_iq_i^2\). Let \(P\) project onto \(\operatorname{col}N\). The same orthogonal decomposition used in Theorem 4 gives

\[
\|K_X\Sigma^{1/2}-N\|_F^2
\ge\|(I-P)K_X\Sigma^{1/2}\|_F^2
=\sum_i a_i^2(1-u_i^\top Pu_i).
\]

Again \(0\le u_i^\top Pu_i\le1\) and their sum is at most \(r\), so the lower bound is the sum of all but the \(r\) largest \(a_i^2=\omega_iq_i^2\). Choose the diagonal hat

\[
B=\sum_{i\in S_r}q_iu_iu_i^\top.
\]

It is positive semidefinite, has rank at most \(r\), and has eigenvalues below one. It is attainable by the row Gram \(\tau B(I-B)^{-1}\) whenever the reconstruction width is at least its rank. Its error is exactly the lower bound. \(\square\)

The commutation condition is essential. With

\[
K=\operatorname{diag}(4/5,1/5),
\qquad
\Sigma=\begin{pmatrix}11/20&1/2\\1/2&11/20\end{pmatrix},
\]

the best coordinate-aligned rank-one hat costs \(11/500=0.022\). The valid rotated hat

\[
v=(4,1)^\top/\sqrt{17},
\qquad
B=(61/89)vv^\top
\]

costs \(964/189125\approx0.00510\). For noncommuting priors, the exact problem is a constrained weighted positive-semidefinite low-rank approximation, not a scalar eigenmode ranking.

Experiment 4b used the isotropic target prior in its primary metric. This structured-prior theorem is a clean future extension, not an explanation of the current result.

## 4. The implemented stochastic objective

The training code draws \(g\sim\mathcal N(0,I_n)\) and normalizes each probe to

\[
y=\sqrt n\,g/\|g\|,
\]

so \(y\) is uniform on the radius-\(\sqrt n\) sphere. For a symmetric matrix \(B\),

\[
\mathbb E[y^\top By]=\operatorname{tr}B,
\]

\[
\operatorname{Var}(y^\top By)
=\frac{2}{n+2}
\left(n\operatorname{tr}(B^2)-(\operatorname{tr}B)^2\right).
\]

For positive-semidefinite \(B\), \(m\) probes, and effective rank

\[
r_{\mathrm{eff}}(B)=\frac{(\operatorname{tr}B)^2}{\operatorname{tr}(B^2)},
\]

the relative variance of their mean is

\[
\frac{2}{m(n+2)}
\left(\frac n{r_{\mathrm{eff}}(B)}-1\right).
\]

This is not the unnormalized Gaussian law previously checked in Experiment 3.

For groups \(g=1,\ldots,G\), let

\[
B_g=(K_{X_g}-K_{Z_g})^2,
\qquad
C_g=K_{X_g}^2.
\]

The implemented term is one self-normalized ratio

\[
\widehat R_m
=\frac{Z_m}{D_m}
=\frac{\sum_{g,j}y_{gj}^\top B_gy_{gj}}
{\sum_{g,j}y_{gj}^\top C_gy_{gj}},
\]

whereas the exact identity-target quantity is

\[
R_{\mathrm{tr}}
=\frac{\sum_g\operatorname{tr}B_g}
{\sum_g\operatorname{tr}C_g}.
\]

In Experiment 4b training, \(n=128\), \(m=16\), and a 2,048-token batch produces 16 contiguous groups, hence 256 independently normalized target columns. The code clamps the one global denominator below at \(10^{-12}\), rather than adding an epsilon to each group. It detaches \(K_{X_g}\) and the denominator. Because the original activations are fixed with respect to SAE parameters, that detachment does not change the mathematical SAE gradient; it avoids building an unnecessary graph. The clamp changes the objective in rank-deficient denominator pathologies but is inactive for the audited full-row-rank groups.

The numerator and denominator are separately unbiased for their trace totals, but

\[
\mathbb E[Z_m/D_m]\ne \mathbb EZ_m/\mathbb ED_m
\]

in general. A second-order expansion, with \(\rho=\mathbb EZ_m/\mathbb ED_m\), gives

\[
\mathbb E\widehat R_m
=\rho+
\frac{\rho\operatorname{Var}(D_m)-\operatorname{Cov}(Z_m,D_m)}
{(\mathbb ED_m)^2}
+O((Gm)^{-2}),
\]

under denominator concentration. The leading bias can have either sign. Conditioned on probes, SGD uses \(\nabla Z_m/D_m\), so its expectation is the gradient of \(\mathbb E[Z_m/D_m]\), not the gradient of \(R_{\mathrm{tr}}\).

This estimator gap does not explain away Experiment 4b: the primary held-out result uses identity targets and recomputes the exact ratio. It does mean the paper must distinguish the stochastic training surrogate from the exact evaluation object.

## 5. Group law is part of the estimand

For fixed group size \(n\) and a specified group distribution \(P_n\), define exact group numerator \(N_n\) and reference energy \(R_n\). If iid groups are integrable and \(\mathbb E_{P_n}R_n>0\), then

\[
\frac{\sum_{g=1}^G N_{n,g}}
{\sum_{g=1}^G R_{n,g}}
\xrightarrow{\mathrm{a.s.}}
\mathcal R_{n,P_n}
=\frac{\mathbb E_{P_n}N_n}{\mathbb E_{P_n}R_n}.
\]

This is not \(\mathbb E[N_n/R_n]\), and changing the partition or \(n\) changes the operator blocks themselves. No unique corpus-level ridge operator follows without a declared asymptotic regime and dependence assumptions.

The iid statement is a conditional population route, not a description of the actual training batches. Experiment 4b groups contiguous half-sequences, and BatchTopK allocates one global support budget before those rows are reshaped into groups, so a group's reconstruction depends on scores elsewhere in the 2,048-token batch. Sequence dependence and global support competition both violate a naive independent-group model.

The distinction is structural, not technical. Four scalar rows can have exact relative loss 0 under one size-two partition and 2 under another. Empirically, Experiment 4b's full-token reduction falls from about 24% at \(n=128\) to about 13% at \(n=256\), despite ridge recalibration. The method should therefore be defined as preserving a distribution of finite-group operators.

## 6. What can be said about BatchTopK

On an open cell with fixed ReLU signs and a strict BatchTopK ordering, token \(i\)'s reconstruction is affine:

\[
\widehat x_i=b_d+
\sum_{\ell\in S_i}
\left((x_i-b_d)^\top w^e_\ell+b^e_\ell\right)d_\ell.
\]

Its within-cell Jacobian is

\[
J_i=\sum_{\ell\in S_i}d_\ell(w^e_\ell)^\top,
\qquad
\operatorname{rank}J_i\le |S_i|.
\]

This yields one necessary local statement: an exactly preserved tangent subspace for token \(i\) cannot exceed its active count on that cell. It does not yield a global allocation theorem. BatchTopK constrains the total number of active row-latent pairs, different rows may use different atoms, and the reconstruction can jump at positive selection ties. A one-sparse reconstruction matrix can have rank as large as \(\min(n,d)\), while some rank-two tables cannot be represented by a fixed union of nonnegative one-sparse rays.

The observed gap between DPSAE and the static rank-derived control is therefore consistent with sparse active-set allocation, overcompleteness, nonorthogonality, reconstruction-dependent geometry, or optimization. Experiment 4b does not identify which mechanism is causal.

## 7. Fisher and frozen output geometry

Ridge-hat preservation is invariant to right-orthogonal feature rotations because it permits the linear decoder to be refit. A frozen downstream model is coordinate-sensitive. The two geometries are incomparable:

- With \(X=I_2\) and \(Z=XQ\) for a feature swap, \(K_X=K_Z\), but a frozen weight \(Me_1\) changes outputs by an amount growing with \(M\).
- If a frozen network reads only coordinate one, changing or deleting coordinate two can have zero output and Fisher cost while changing the ridge hat.

Locally, frozen output KL has the form

\[
\operatorname{KL}(p(h)\|p(h+\delta))
=\tfrac12\delta^\top J_f(h)^\top F_{\mathrm{out}}(h)J_f(h)\delta
+O(\|\delta\|^3),
\]

which is token-specific, coordinate-sensitive, and tied to the frozen downstream Jacobian. The DPSAE metric is groupwise, invariant to feature rotations, and tied to refitted ridge tasks. Neither controls the other without extra assumptions.

The theory decision is therefore to keep Fisher as a comparison and future experiment, not as the main-paper explanation. Experiment 4b's mixed IOI result supports that separation.

## What the theory does not imply

- Small decoder distance does not imply small frozen-logit change, output KL, causal specificity, or Fisher distance. The feature-swap example gives zero decoder distance with arbitrarily large frozen change.
- Small Fisher or frozen-output distance does not imply decoder preservation. A downstream-null activation direction can remain ridge-relevant.
- The relative trace ratio does not control every task relatively. It controls relative average energy only.
- The finite normalized-probe loss is not an unbiased estimate of the exact trace ratio, and its stochastic gradient does not target that exact ratio.
- The finite-group objective is not a unique corpus-level geometry. Group size and partition are part of the estimand.
- The isotropic rank theorem is not a theorem about BatchTopK feature allocation, feature discovery, or semantic importance.
- Failure of the selected whitening and spectral controls does not rule out every static covariance objective or prove that reconstruction dependence is necessary.
- Three paired seeds establish repeatability at one GPT-2 site, not generality across models, layers, corpora, architectures, or tasks.
- No Experiment 4b result supports an activation-manifold or Fisher-pullback extension as a current central contribution.

## Mapping the theorems to Experiment 4b

| Result | What it explains in Experiment 4b | What 4b does not test |
| --- | --- | --- |
| Theorem 1 | Gives the exact interpretation of the identity-target primary metric as average refitted ridge-prediction disagreement and an absolute ellipsoidal bound. | New-row generalization, frozen transformer behavior, and per-task relative error. |
| Proposition 2 | Shows that the isotropic exact metric compares row Gram matrices through an injective ridge transform. | Whether a low but nonzero distance preserves any named semantic variable. |
| Proposition 3 | Justifies retaining the MSE anchor and shows why MSE can control but need not match decoder geometry. | A quantitative prediction of the observed 24%-versus-7% frontier. |
| Theorem 4 | Proves that a rank-only isotropic relaxation cannot reorder PCA directions. | The shared nonnegative overcomplete BatchTopK feasible set. |
| Corollary 5 | Identifies the tested spectral baseline as the rank theorem's full-mode omission pricing. Its <1% gain rejects that simple quantitative explanation. | Every static metric or a matched-NMSE static frontier. |
| Theorem 6 | Shows how a commuting structured task prior could reorder modes. | The primary 4b metric is isotropic, so this theorem is prospective. |
| Probe analysis | Explains the exact difference between normalized stochastic training and identity-target confirmation. | Direct gradient fidelity at 16 probes, which remains unmeasured. |
| Group-law analysis | Predicts that changing finite groups can change the estimand, matching the 24% at \(n=128\) versus 13% at \(n=256\). | Training under alternative group laws and any asymptotic token-process model. |
| Fisher separation | Predicts that representation improvement need not produce better IOI specificity, matching the mixed frozen intervention result. | Whether an explicit alignment regularizer can bridge the two geometries. |

## The paper-level result

The final theory story is intentionally narrower than the initial ambition:

1. DPSAE directly regularizes a finite-group ridge prediction operator, which exactly controls average refitted linear-task disagreement and upper-bounds absolute worst-case disagreement on a declared task ellipsoid.
2. In an isotropic rank relaxation, this geometry cannot reorder singular directions, so the rank theorem is a boundary rather than an explanation of sparse SAE behavior.
3. The theorem-derived static omission-cost control captures little of the confirmed empirical gain, which localizes the unresolved mechanism to the sparse nonlinear, shared-dictionary, or reconstruction-dependent setting.
4. The stochastic normalized-probe ratio and the group construction are part of the method, not implementation-neutral estimators of a unique population quantity.
5. Frozen output, causal, Fisher, and activation-manifold claims require separate evidence and remain outside the core result.

The unresolved mathematical problem is to characterize which finite-group ridge modes a shared overcomplete nonnegative sparse dictionary preserves under global support competition. Experiment 4b shows that the phenomenon is real enough to study, but it does not yet identify that allocation rule.
