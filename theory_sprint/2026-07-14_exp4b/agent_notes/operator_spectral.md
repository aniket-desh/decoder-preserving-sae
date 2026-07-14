# Operator and spectral track: independent derivation and adversarial audit

I read `00_empirical_audit.md` before doing the derivation below. The implementation-matched object for one geometry group is, with (X,Z\in\mathbb R^{n\times d}), average-loss ridge parameter λ>0, and τ:=nλ,

\[
G_X:=XX^\top,
\qquad
K_X:=K_\lambda(X)
=X(X^\top X+\tau I_d)^{-1}X^\top
=G_X(G_X+\tau I_n)^{-1}.
\]

The last equality follows from the compact SVD and is valid whether (n<d), (n=d), or (n>d), including when (X) is rank deficient. Write Δ:=K_X-K_Z. For a task second-moment matrix (T\succeq0), the exact unnormalized task loss is

\[
D_T^2(X,Z):=\operatorname{tr}(\Delta T\Delta^\top)
=\|\Delta T^{1/2}\|_F^2.
\]

`decoder_distance(..., reduction="sum")` computes this quantity; its `mean` reduction divides it by (n^2). Experiment 4b's exact identity-target metric instead reports a ratio of sums, (\sum_g\|\Delta_g\|_F^2/\sum_g\|K_{X_g}\|_F^2). The training path in `TrainingFleet.train_batch` likewise takes one ratio after summing over all groups and probes. This is different from the unused helper `batched_sampled_decoder_loss`, which averages a separate ratio for each group.

## 1. What small decoder distortion guarantees

### 1.1 Exact average-case identity

Let (y\in\mathbb R^n) be any random target with finite second moment (\mathbb E[yy^\top]=T); zero mean is unnecessary if (T) is understood as the second moment rather than the covariance. Then

\[
\begin{aligned}
\mathbb E\|K_Xy-K_Zy\|_2^2
&=\mathbb E\,y^\top\Delta^\top\Delta y\\
&=\mathbb E\,\operatorname{tr}(\Delta^\top\Delta yy^\top)\\
&=\operatorname{tr}(\Delta^\top\Delta T)\\
&=\operatorname{tr}(\Delta T\Delta^\top)\\
&=D_T^2(X,Z).
\end{aligned}
\]

Thus the unnormalized exact loss is precisely an average squared disagreement over a declared task law. This is an in-group, in-sample statement: (K_Xy) is the fitted ridge prediction on the same (n) rows used to form (K_X). It contains no out-of-group or population generalization guarantee.

For Experiment 4b's normalized Gaussian probes, each column is uniform on the sphere of radius (\sqrt n), so (\mathbb E[yy^\top]=I_n). Each numerator and denominator trace estimate is individually unbiased for the corresponding isotropic Frobenius energy, but their finite-probe ratio is not generally an unbiased estimate of the trace ratio.

### 1.2 Exact ellipsoidal worst case, including singular (T)

Let (k=\operatorname{rank}(T)) and define the possibly degenerate ellipsoid

\[
\mathcal E_T
:=\{y\in\operatorname{range}(T):y^\top T^\dagger y\le1\}
=\{T^{1/2}u:\|u\|_2\le1\}.
\]

The equality of the two definitions follows by taking the minimum-norm preimage (u=T^{\dagger/2}y). Then

\[
\sup_{y\in\mathcal E_T}\|\Delta y\|_2^2
=\sup_{\|u\|\le1}\|\Delta T^{1/2}u\|_2^2
=\|\Delta T^{1/2}\|_{\mathrm{op}}^2.
\]

If this worst-case value is (W_T^2), the singular values of (\Delta T^{1/2}) give

\[
W_T^2\le D_T^2\le kW_T^2.
\]

So (D_T\le\varepsilon) gives the uniform absolute guarantee (\|\Delta y\|\le\varepsilon) for every (y\in\mathcal E_T). Conversely, a uniform bound on that ellipsoid controls (D_T) only with the unavoidable factor (\sqrt{k}). If (u) is uniform on the unit sphere in (\operatorname{range}(T)), then (y=\sqrt{k}T^{1/2}u) has second moment (T), making the average/worst-case normalization explicit.

### 1.3 What the relative denominator does and does not do

Define the population trace ratio, when its denominator is positive,

\[
R_T(X,Z)
:=\frac{\operatorname{tr}(\Delta T\Delta)}
{\operatorname{tr}(K_XTK_X)}
=\frac{\|\Delta T^{1/2}\|_F^2}
{\|K_XT^{1/2}\|_F^2}.
\]

This is relative *average energy*. It is not a worst-case relative guarantee. To see the exact worst-case object, put

\[
A=T^{1/2}\Delta^2T^{1/2},
\qquad
B=T^{1/2}K_X^2T^{1/2}.
\]

For tasks (y=T^{1/2}u), the directional relative error is (u^\top Au/u^\top Bu). Its supremum is infinite if some (u\in\ker B) has (u^\top Au>0). If (\ker B\subseteq\ker A), the finite supremum is the largest generalized eigenvalue of ((A,B)) on (\operatorname{range}(B)), equivalently

\[
\lambda_{\max}\!\left(B^{\dagger/2}AB^{\dagger/2}\right).
\]

The trace ratio is at most this generalized worst case, and it can be arbitrarily smaller.

An explicit valid-hat counterexample makes the gap sharp. Let

\[
K_X=\operatorname{diag}(1/2,\eta^2),
\qquad
K_Z=\operatorname{diag}(1/2,\eta^2+\eta),
\qquad T=I_2,
\]

with (0<\eta<1/3). Every diagonal entry lies in ([0,1)), so both matrices are ridge hats for finite representations by the attainability construction in Section 4.3. Then

\[
R_I=\frac{\eta^2}{1/4+\eta^4}\longrightarrow0,
\]

but for (y=e_2),

\[
\frac{\|\Delta y\|^2}{\|K_Xy\|^2}
=\frac{\eta^2}{\eta^4}=\eta^{-2}\longrightarrow\infty.
\]

If (\|K_XT^{1/2}\|_F=0), the mathematical relative quantity is undefined. The code's (10^{-12}) clamp makes a finite numerical loss, but it no longer has a scale-free relative interpretation.

Finally, the implemented finite-probe training loss is a random ratio, not (R_T):

\[
\widehat R
=\frac{\sum_{g,j}\|\Delta_gy_{gj}\|^2}
{\sum_{g,j}\|K_{X_g}y_{gj}\|^2}.
\]

Even though both sums are unbiased trace estimators under normalized-sphere probes, (\mathbb E[\widehat R]\ne\mathbb E[N]/\mathbb E[D]) in general. Stop-gradient through (K_X) and the denominator does not change the forward value; because (X) is a fixed reference, it correctly treats the denominator as constant with respect to (Z).

## 2. Zero set, row-Gram equivalence, and invariances

### 2.1 The ridge hat is an injective function of the row Gram

Since (G_X\succeq0),

\[
K_X=G_X(G_X+\tau I)^{-1}
=I-\tau(G_X+\tau I)^{-1}.
\]

Every eigenvalue of (K_X) lies in ([0,1)), so (I-K_X) is invertible. Rearranging gives the inverse map

\[
I-K_X=\tau(G_X+\tau I)^{-1},
\qquad
G_X=\tau K_X(I-K_X)^{-1}.
\]

Therefore, for the same strictly positive ridge,

\[
K_X=K_Z
\quad\Longleftrightarrow\quad
XX^\top=ZZ^\top.
\]

This equivalence fails at zero ridge: as (\tau\downarrow0), (K_X) approaches only the projector onto (\operatorname{col}(X)), losing all nonzero singular-value information.

### 2.2 Full-rank and singular task priors

Because (D_T^2=\|\Delta T^{1/2}\|_F^2),

\[
D_T=0
\quad\Longleftrightarrow\quad
\Delta T^{1/2}=0
\quad\Longleftrightarrow\quad
\Delta v=0\ \text{for every }v\in\operatorname{range}(T).
\]

If (T\succ0), this is equivalent to (K_X=K_Z), hence to (XX^\top=ZZ^\top). If (T) is singular, it only requires the two hats to agree on the task-supported subspace. Symmetry of Δ also implies agreement of the corresponding rows, but it does not imply equality of projected row Grams because the inverse map (K\mapsto\tau K(I-K)^{-1}) is nonlinear and depends on the unobserved block.

For example, with τ=1,

\[
T=\operatorname{diag}(1,0),\quad
K_X=\operatorname{diag}(1/2,1/3),\quad
K_Z=\operatorname{diag}(1/2,2/3),
\]

we have (D_T=0), but the row Grams are respectively

\[
G_X=\operatorname{diag}(1,1/2),
\qquad
G_Z=\operatorname{diag}(1,2).
\]

One realization is (X=\operatorname{diag}(1,1/\sqrt2)), (Z=\operatorname{diag}(1,\sqrt2)).

### 2.3 Exact quotient for equal feature dimension

Assume (X,Z\in\mathbb R^{n\times d}). If (XX^\top=ZZ^\top=:G) has rank (s>0), take (G=U\Sigma^2U^\top) with (U\in\mathbb R^{n\times s}) orthonormal and Σ positive diagonal. Define

\[
V_X=X^\top U\Sigma^{-1},
\qquad
V_Z=Z^\top U\Sigma^{-1}.
\]

Then

\[
V_X^\top V_X
=\Sigma^{-1}U^\top XX^\top U\Sigma^{-1}=I_s,
\]

and similarly (V_Z^\top V_Z=I_s). Also (X=U\Sigma V_X^\top) and (Z=U\Sigma V_Z^\top). Extend (V_X,V_Z) to orthogonal bases of (\mathbb R^d), and choose an orthogonal (Q\in O(d)) such that (V_X^\top Q=V_Z^\top). Then

\[
XQ=U\Sigma V_X^\top Q=U\Sigma V_Z^\top=Z.
\]

The rank-zero case is trivial. Consequently, under full task support and λ>0, decoder distance is a metric on row-Gram matrices, or equivalently a pseudometric on representations whose zero classes are precisely right-orthogonal orbits (\{XQ:Q\in O(d)\}).

It is not invariant to generic feature rescaling or to generic invertible feature maps at positive ridge. It is invariant to a right orthogonal rotation, including signs and permutations. Since Experiment 4b does not center inside a group, feature translations are also not invariances. A frozen downstream weight matrix is generally not preserved by (X\mapsto XQ) unless its weights are counter-rotated, so zero refittable-decoder distance does not imply frozen-model compatibility.

## 3. Activation reconstruction controls decoder distortion in one direction only

Let (E=X-Z), (G=XX^\top), and (H=ZZ^\top). The resolvent identity gives

\[
\begin{aligned}
K_X-K_Z
&=\tau\big[(H+\tau I)^{-1}-(G+\tau I)^{-1}\big]\\
&=\tau(H+\tau I)^{-1}(G-H)(G+\tau I)^{-1}.
\end{aligned}
\]

Both resolvents have operator norm at most (1/\tau). Moreover,

\[
G-H=(X-Z)X^\top+Z(X-Z)^\top=EX^\top+ZE^\top,
\]

and hence

\[
\|G-H\|_F
\le\|E\|_F\|X\|_{\mathrm{op}}
+\|Z\|_{\mathrm{op}}\|E\|_F.
\]

Combining the inequalities,

\[
\boxed{
\|K_X-K_Z\|_F
\le
\frac{\|X\|_{\mathrm{op}}+\|Z\|_{\mathrm{op}}}{n\lambda}
\|X-Z\|_F.
}
\]

Therefore

\[
\boxed{
D_T^2(X,Z)
\le
\|T\|_{\mathrm{op}}
\left(
\frac{\|X\|_{\mathrm{op}}+\|Z\|_{\mathrm{op}}}{n\lambda}
\right)^2
\|X-Z\|_F^2.
}
\]

Under the explicit norm assumption (\|X\|_{\mathrm{op}},\|Z\|_{\mathrm{op}}\le B), replace the numerator by (2B). A relative version requires the additional nondegeneracy assumption (\operatorname{tr}(K_XTK_X)\ge c>0), after which the right side is divided by (c). There is no uniform relative bound without such an assumption.

This global bound is deliberately modest. It blows up like (1/(n\lambda)), retains the activation spectral norms, ignores the active-set and overcomplete-dictionary constraints, and supplies no converse. It establishes continuity, not a mechanism capable of explaining the 24% Exp4b gain at a 7% NMSE cost.

The converse fails even at zero decoder distance. For any nonzero (X), let (Z=-X). Then (ZZ^\top=XX^\top), so (D_T(X,Z)=0) for every (T), while

\[
\|X-Z\|_F^2=4\|X\|_F^2>0.
\]

Scaling (X) makes this activation error arbitrarily large. This is also a concrete frozen-coordinate failure: a fixed downstream linear map generally changes sign.

## 4. Complete isotropic rank-constrained theorem

### 4.1 Statement covering all shapes and rank deficiencies

Let (X\in\mathbb R^{n\times d}) have rank (s\le\min(n,d)) and compact SVD

\[
X=U\operatorname{diag}(\sigma_1,\ldots,\sigma_s)V^\top,
\qquad
\sigma_1\ge\cdots\ge\sigma_s>0.
\]

For τ=nλ>0 define

\[
q_i:=\frac{\sigma_i^2}{\sigma_i^2+\tau}\in(0,1).
\]

Let (r\) be any integer with (0\le r\le\min(n,d)), and put (k=\min(r,s)). Then

\[
\boxed{
\min_{Z\in\mathbb R^{n\times d}:\operatorname{rank}(Z)\le r}
\|K_X-K_Z\|_F^2
=\sum_{i=k+1}^s q_i^2.
}
\]

One minimizer is the usual truncated SVD (X_k=U_k\Sigma_kV_k^\top). Thus isotropic decoder preservation and PCA choose the same sample-space singular directions in this relaxation, although decoder omission costs are (q_i^2), which saturate at one, rather than σ_i².

### 4.2 Spectral form of the source hat

Using (X=U\Sigma V^\top), decompose the feature identity as (I_d=VV^\top+(I_d-VV^\top)). Then

\[
(X^\top X+\tau I_d)^{-1}
=V(\Sigma^2+\tau I_s)^{-1}V^\top
+\tau^{-1}(I_d-VV^\top).
\]

Substitution gives

\[
\begin{aligned}
K_X
&=U\Sigma V^\top
\left[V(\Sigma^2+\tau I)^{-1}V^\top
+\tau^{-1}(I-VV^\top)\right]
V\Sigma U^\top\\
&=U\Sigma(\Sigma^2+\tau I)^{-1}\Sigma U^\top\\
&=U\operatorname{diag}(q_1,\ldots,q_s)U^\top.
\end{aligned}
\]

The cross term vanishes because ((I-VV^\top)V=0). The scalar map (q(\sigma)=\sigma^2/(\sigma^2+\tau)) has derivative (2\sigma\tau/(\sigma^2+\tau)^2>0), so it preserves the ordering of positive singular values and exactly preserves their ties.

### 4.3 Feasible ridge hats and attainability

If (Z=A\operatorname{diag}(\zeta_1,\ldots,\zeta_t)B^\top) is a compact SVD, then

\[
K_Z=A\operatorname{diag}
\left(\frac{\zeta_j^2}{\zeta_j^2+\tau}\right)A^\top.
\]

Hence every feasible hat is symmetric positive semidefinite, has rank at most (r), and has all nonzero eigenvalues strictly below one. Conversely, if

\[
M=A\operatorname{diag}(m_1,\ldots,m_t)A^\top,
\quad 0<m_j<1,
\quad t\le\min(r,d),
\]

choose any (B\in\mathbb R^{d\times t}) with orthonormal columns and set

\[
\zeta_j=\sqrt{\frac{\tau m_j}{1-m_j}},
\qquad
Z=A\operatorname{diag}(\zeta_j)B^\top.
\]

Then (\operatorname{rank}(Z)=t) and (K_Z=M). Thus finite ridge representations realize exactly the finite-rank PSD contractions with eigenvalues in ([0,1)).

### 4.4 Rank-approximation lower bound with no shape assumption

Let (M) be any matrix of rank at most (r), and let (P) be the orthogonal projector onto its column space. Since (PM=M),

\[
K_X-M=(I-P)K_X+P(K_X-M).
\]

The two summands are Frobenius-orthogonal because ((I-P)P=0). Therefore

\[
\|K_X-M\|_F^2
=\|(I-P)K_X\|_F^2+\|P(K_X-M)\|_F^2
\ge\|(I-P)K_X\|_F^2.
\]

Since (K_X^2=\sum_{i=1}^s q_i^2u_iu_i^\top),

\[
\begin{aligned}
\|(I-P)K_X\|_F^2
&=\operatorname{tr}[K_X(I-P)K_X]\\
&=\sum_{i=1}^s q_i^2-\sum_{i=1}^s q_i^2a_i,
\end{aligned}
\]

where (a_i:=u_i^\top Pu_i=\|Pu_i\|^2\) satisfies (0\le a_i\le1) and

\[
\sum_{i=1}^s a_i\le\operatorname{tr}(P)=\operatorname{rank}(P)\le r.
\]

Because (q_1^2\ge\cdots\ge q_s^2), an exchange argument moves any weight on a lower index to an unfilled higher index without decreasing (\sum_iq_i^2a_i). Hence

\[
\sum_{i=1}^s q_i^2a_i\le\sum_{i=1}^kq_i^2,
\]

and every rank-(r) matrix obeys

\[
\|K_X-M\|_F^2\ge\sum_{i=k+1}^sq_i^2.
\]

Now take (Z=X_k). Section 4.2 gives

\[
K_{X_k}=\sum_{i=1}^kq_iu_iu_i^\top,
\]

so

\[
\|K_X-K_{X_k}\|_F^2
=\left\|\sum_{i=k+1}^sq_iu_iu_i^\top\right\|_F^2
=\sum_{i=k+1}^sq_i^2.
\]

The lower bound is therefore attained by a valid (n\times d) representation. This proof treats (n<d) and (n>d) identically and never assumes full row or column rank.

### 4.5 Ties, uniqueness, and all important edge cases

If (r<s) and (q_r>q_{r+1}), the best rank-(r) matrix approximation is unique: equality above forces (P) to project onto (\operatorname{span}(u_1,\ldots,u_r)) and forces (P(K_X-M)=0), hence (M=\sum_{i\le r}q_iu_iu_i^\top). Every representation minimizer then has the same row Gram as (X_r), so for equal feature dimension it is (X_rQ) for some (Q\in O(d)).

If (q_r=q_{r+1}=c), include every eigenspace with eigenvalue greater than (c), then choose any required-dimensional subspace of the (c)-eigenspace. This is genuine left-subspace nonuniqueness in addition to arbitrary right-feature rotations. It is attainable because equal (q)'s imply equal positive singular values. If (r\ge s), the minimum is zero and the minimizers are exactly the representations with (ZZ^\top=XX^\top). If (X=0), the zero representation attains zero for every (r).

The relative isotropic population loss has the same minimizers because (\|K_X\|_F^2=\sum_iq_i^2) is constant in (Z). For (X\ne0), its minimum is

\[
\frac{\sum_{i=k+1}^sq_i^2}{\sum_{i=1}^sq_i^2}.
\]

This statement does not transfer automatically to a mean of groupwise ratios when one shared nonlinear model couples groups; groupwise denominators reweight the groups.

At fixed (X), the ridge limits are

\[
\tau\downarrow0:
q_i\to1,
\qquad
\min D_I^2\to s-k,
\]

and

\[
\tau\to\infty:
q_i^2=\frac{\sigma_i^4}{\tau^2}+O(\tau^{-3}),
\qquad
\min D_I^2
=\tau^{-2}\sum_{i=k+1}^s\sigma_i^4+O(\tau^{-3}).
\]

The τ↓0 limit is a column-space projector theorem, not a row-Gram theorem. The τ→∞ limit vanishes without rescaling and becomes a Schatten-4 tail after multiplying by τ².

## 5. What the static spectral surrogate actually represents

Let (C=X^\top X/n=V\operatorname{diag}(c_i)V^\top) with (c_i=\sigma_i^2/n), and let

\[
M_\lambda=C(C+\lambda I)^{-2}.
\]

For the PCA truncation (X_k), (E=X-X_k) is diagonal in the same right-singular basis, and

\[
\begin{aligned}
\|EM_\lambda^{1/2}\|_F^2
&=\sum_{i>k}\sigma_i^2\frac{c_i}{(c_i+\lambda)^2}\\
&=n\sum_{i>k}\left(\frac{c_i}{c_i+\lambda}\right)^2\\
&=n\sum_{i>k}q_i^2.
\end{aligned}
\]

Likewise (\|XM_\lambda^{1/2}\|_F^2=n\sum_iq_i^2). Thus the implemented static relative metric exactly reproduces the isotropic theorem's relative omission cost on hard PCA-mode deletion.

It is not equal to decoder distance for a general reconstruction, and it is not the general Fréchet metric of the ridge-hat map. In the scalar case (n=d=1), (x=1), λ=1, and (z=-1), exact decoder distortion is zero because (x^2=z^2), whereas

\[
(x-z)^2\frac{x^2}{(x^2+\lambda)^2}=1.
\]

For omission (z=0), both quantities equal (1/4). Even locally at (z=x+e), the exact relative distortion has leading coefficient (4\lambda^2/[x^2(x^2+\lambda)^2]), whereas the normalized static residual has coefficient (1/x^2). The surrogate is therefore an omission-cost extension, not a universal local identity.

This distinction fits Exp4b exactly: the selected static surrogate improves exact distortion by only 0.37–0.91%, versus roughly 24% for the reconstruction-dependent DPSAE objective. That result rejects the hypothesis that this particular hard-omission extension quantitatively explains the BatchTopK gain; it does not falsify the isotropic boundary theorem and does not rule out every static covariance metric.

## 6. Structured task priors

### 6.1 Exact theorem in the commuting case

Let (K:=K_X) and (T\succeq0) commute. Since both are real symmetric, they admit a simultaneous orthonormal eigenbasis:

\[
K=U\operatorname{diag}(q_1,\ldots,q_n)U^\top,
\qquad
T=U\operatorname{diag}(\omega_1,\ldots,\omega_n)U^\top,
\]

where (0\le q_i<1), (\omega_i\ge0), and zero (q_i)'s include the nullspace of (X). For any feasible (M=K_Z) of rank at most (r), set (N=MT^{1/2}). Then (\operatorname{rank}(N)\le r) and

\[
D_T^2(X,Z)=\|KT^{1/2}-N\|_F^2.
\]

The singular values of (KT^{1/2}) are (q_i\sqrt{\omega_i}). The unrestricted rank-(r) lower bound is therefore the sum of all but the (r) largest scores (\omega_iq_i^2). It is attainable: if (J) indexes any (r) largest positive scores, take

\[
M_J=U\operatorname{diag}(q_i\mathbf1\{i\in J\})U^\top.
\]

This is a PSD contraction of rank at most (r), so Section 4.3 constructs a (Z) with (K_Z=M_J). Consequently

\[
\boxed{
\min_{\operatorname{rank}(Z)\le r}D_T^2(X,Z)
=\sum_{i\notin J}\omega_iq_i^2,
\quad
J=\operatorname{Top}_r\{\omega_iq_i^2\}.
}
\]

This remains true for singular (T); zero-weight modes cost nothing, and if (r) covers every positive score, the task loss can be zero without preserving the full row Gram. Score ties create the expected subspace nonuniqueness. A structured prior can therefore reorder source modes, but this exact weighted-ranking formula requires commutation.

The relative denominator (\operatorname{tr}(KTK)=\sum_i\omega_iq_i^2) is constant in (Z), so it preserves the same optimizer when positive.

### 6.2 General noncommuting characterization and the failed naive extension

Without commutation, write

\[
A_0:=KT^{1/2}.
\]

For every feasible (M=K_Z), (N=MT^{1/2}) has rank at most (r), giving the rigorous relaxation

\[
\inf_{\operatorname{rank}(Z)\le r}D_T^2(X,Z)
\ge\sum_{i>r}s_i(A_0)^2,
\]

where (s_i(A_0)^2) are the eigenvalues of (KTK). This is an ordinary singular-value problem for a lower bound, not yet the exact representation problem.

If (T\succ0), the relaxed best approximation is (N_*=P_rKT^{1/2}), where (P_r) projects onto the top left singular subspace of (KT^{1/2}), equivalently the top eigenspace of (KTK). Mapping back gives

\[
M_{\mathrm{rel}}=N_*T^{-1/2}=P_rK.
\]

Generically (P_rK\ne KP_r), so (M_{\mathrm{rel}}) is not symmetric and cannot be any ridge hat. Thus the SVD lower bound is generally unattainable. For singular (T), a pseudoinverse gives the same relaxed conclusion on (\operatorname{range}(T)), with additional unconstrained nullspace components; symmetry remains the obstruction.

The exact feasible problem is

\[
\inf_{
M=M^\top\succeq0,
\ \operatorname{rank}(M)\le r,
\ \lambda_{\max}(M)<1}
\|(K-M)T^{1/2}\|_F^2.
\]

It is a weighted low-rank PSD-contraction problem. Parameterize (M=UBU^\top), where (U\in\mathbb R^{n\times t}), (U^\top U=I_t), (t\le r), and (0\preceq B\prec I_t). Define

\[
C_U:=U^\top TU,
\qquad
H_U:=\tfrac12U^\top(TK+KT)U.
\]

Direct expansion gives

\[
\|(K-UBU^\top)T^{1/2}\|_F^2
=\operatorname{tr}(KTK)-2\operatorname{tr}(BH_U)
+\operatorname{tr}(BC_UB).
\]

For a fixed (U), if (C_U\succ0) and the PSD/contraction constraints are inactive, stationarity in symmetric (B) is the Sylvester equation

\[
C_UB+BC_U=2H_U.
\]

The outer optimization over (U) remains nonlinear on a Grassmann manifold, and active PSD or eigenvalue-one constraints further change it. In rank one, (M=\alpha vv^\top), the unconstrained coefficient on a fixed unit vector is

\[
\alpha_*(v)=\frac{v^\top TKv}{v^\top Tv},
\]

clipped to the PSD-contraction interval. Because (TK) is not self-adjoint in the Euclidean metric, this quotient can even be negative or exceed one. There is no universal source-mode score or single generalized eigensystem solving the exact problem.

Here is an exact (2\times2) counterexample to naive weighted source-mode selection. Let τ=1,

\[
K=\begin{pmatrix}4/5&0\\0&3/10\end{pmatrix},
\qquad
T=\begin{pmatrix}1&4/5\\4/5&1\end{pmatrix},
\qquad r=1.
\]

Both matrices are positive definite and do not commute. One source representation is (X=\operatorname{diag}(2,\sqrt{3/7})). The diagonal omission scores are (16/25) and (9/100), so naive selection keeps (e_1) and incurs (9/100). Instead take

\[
v=(12/13,5/13)^\top,
\qquad
\alpha=351/530,
\qquad
M=\alpha vv^\top.
\]

This is a feasible rank-one PSD contraction (realized, for example, by a rank-one (Z) with squared singular value (\alpha/(1-\alpha)=351/179)). Direct substitution yields

\[
\|(K-M)T^{1/2}\|_F^2
=\operatorname{tr}[(K-M)T(K-M)]
=\frac{56}{1325}
\approx0.04226
<\frac9{100}.
\]

So a rotated mode strictly beats both coordinate-selection choices. In this example the SVD relaxation has tail

\[
\lambda_{\min}(KTK)
=\frac{365-\sqrt{112489}}{1000}
\approx0.02961,
\]

strictly below the displayed feasible value; its top singular projector is mixed, making (P_1K) nonsymmetric. This exhibits both failures: source-mode ranking is not exact, and the ordinary SVD answer need not be a realizable ridge hat.

## 7. Counterexample and gap ledger

- **Small trace-relative loss does not control every task relatively.** The η-family in Section 1.3 makes the trace ratio vanish while one target's relative error diverges.
- **Singular task support does not identify the row Gram.** The diagonal example in Section 2.2 has zero task loss and different row Grams.
- **Decoder closeness has no activation-error converse.** (Z=-X) gives exact zero decoder distance and arbitrarily large activation error.
- **Positive-ridge rotation invariance is only a refittable-decoder invariance.** Frozen downstream weights generally change under the same rotation.
- **The isotropic theorem does not reorder low-variance modes.** Its omission score is strictly increasing in every positive singular value; sparse/nonlinear behavior lies outside the theorem.
- **The static spectral objective is exact only on its motivating omission family.** The scalar sign-flip example refutes equality for general reconstructions and also refutes calling it the full local metric.
- **Commuting weighted scores do not extend to noncommuting priors.** The explicit (2\times2) construction beats naive source-mode selection.
- **The noncommuting SVD solution is generally infeasible.** Its back-mapped matrix (P_rK) is nonsymmetric unless the selected projector commutes with (K).
- **No population statement follows from the finite-group operator theorem.** The group construction and (n\lambda) scaling define the estimand; Exp4b's 64-versus-256 group-size effect is direct evidence that the magnitude is group-dependent.
- **Open gap.** I have characterized, but not solved in closed form, the general noncommuting weighted PSD-contraction problem. A claim of a universal generalized-eigenvector solution would require a new theorem and is currently unsupported.

## 8. Empirical interpretation for Experiment 4b

The surviving theory is a boundary theorem and an exact interpretation theorem. The measured identity-target loss is average isotropic in-group ridge-prediction disagreement, and it gives an ellipsoidal absolute worst-case bound on those same rows. It does not guarantee population transfer, a hard task's relative preservation, sparse feature discovery, frozen-model behavior, or causal specificity.

The isotropic rank relaxation predicts the same singular subspace as PCA and the static omission-cost control captures that relaxation's mode weights exactly. The control's roughly 1% improvement versus DPSAE's roughly 24% therefore says the sparse reconstruction-dependent effect is not quantitatively explained by that simple extension. It does not identify which missing mechanism—BatchTopK allocation, nonorthogonality, dynamic row Gram, or optimization—causes the gap.

The group-size dependence (about 35% at (n=64) versus 13% at (n=256), after ridge recalibration) reinforces that these are finite-group transductive operators, not observations of a unique grouping-independent corpus geometry. The failed IOI causal-specificity comparison is also consistent with the exact invariance analysis: refitted decoder access is coarser than frozen coordinate compatibility. Nothing in these results motivates elevating Fisher or activation-manifold theory into the core paper.

## 9. Cross-red-team of the sparse/statistical track

I checked the other track's five requested claims against the operator derivations and its explicit constructions. All five survive, with one empirical wording correction.

1. **Finite normalized-sphere training targets an expected self-normalized ratio, not the trace ratio.** The two diagonal (2\times2) examples are valid ridge hats and the stated circle integral is correct: one has negative bias and the other positive bias. Sixteen probes across sixteen groups improves concentration but cannot restore exact unbiasedness. The safe gradient statement needs the usual interchange/integrability conditions; those are benign for the full-row-rank Exp4b groups, while the denominator clamp changes the objective in rank-deficient pathologies.
2. **Regrouping can change exact relative loss from zero to two with the same rows.** For (X=(1,1,1,1)^\top) and (Z=(1,1,-1,-1)^\top), grouping like signs makes each row Gram equal, while cross-sign grouping produces equal-gain orthogonal rank-one projectors. Their squared difference is (2q^2) and the reference energy is (q^2), so the ratio is exactly two for every positive ridge. This is a clean falsification of grouping independence.
3. **BatchTopK sparsity (k) is not matrix rank (r).** With two rows and (k=1), code and decoder matrices both equal to (I_2) give a rank-two reconstruction. The learned decoder bias can add another shared rank-one term, so it does not rescue a rank interpretation. The spectral theorem must remain a relaxation boundary rather than a theorem about BatchTopK allocation.
4. **The static spectral baseline is an omission-cost surrogate, not the Fréchet metric.** Its per-mode weight (ns^2/(s^2+n\lambda)^2) reproduces full-mode deletion cost. The true differential (dK=n\lambda R(dX X^\top+X dX^\top)R) gives coefficient ([2n\lambda s/(s^2+n\lambda)^2]^2) for a singular-value perturbation. The two are unequal in general, agreeing with the scalar counterexample in Section 5.
5. **Ridge-hat, frozen-network, and Fisher geometries are mutually non-equivalent.** A right feature rotation gives zero decoder distance and arbitrarily large frozen-logit change under scaled fixed weights. Conversely, deleting a feature ignored by the downstream map gives zero frozen/Fisher change and positive isotropic decoder distance. Neither direction supports a universal bound.

The empirical correction is that the full 35%/24%/13% group-size trend is partly confounded: the (n=64) audit uses only 8,192 tokens because of the 128-group cap, whereas (n=128) and (n=256) use all 16,384. The unconfounded (n=128) versus (n=256) change, roughly 24% versus 13%, is already sufficient to establish material group-size dependence. The paper should avoid attributing the entire factor-of-three span solely to group size until the token subsets are matched.
