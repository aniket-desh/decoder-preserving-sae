# Sparse/statistical theory audit after Experiment 4b

## Bottom line

Experiment 4b makes one result worth theorizing centrally: the trained objective improves a fresh, identity-target, groupwise ridge-hat distortion by about 24% at the preregistered group size, while costing about 7% NMSE. The correct theorem-level object is therefore a finite-group, transductive prediction-operator discrepancy. The finite-probe training loss is only a stochastic, self-normalized surrogate for that object, and the BatchTopK constraint is not a rank constraint.

The strongest negative conclusions are equally important.

1. The normalized finite-probe ratio is not an unbiased estimator of the exact relative trace, although its numerator and denominator are separately unbiased. Its bias can have either sign. SGD is unbiased for the expected self-normalized finite-probe objective, not for the exact trace ratio.
2. Geometry group size and partition define the estimand. Even with infinitely many probes and a recalibrated ridge, there is no grouping-independent corpus operator without strong extra assumptions. A four-token counterexample below changes the exact relative loss from 0 to 2 solely by repartitioning the same rows.
3. BatchTopK is piecewise affine only away from ReLU and selection boundaries. Its sparsity level `k` does not bound the matrix rank of the reconstruction, and hard TopK can be discontinuous at positive score ties. The rank-relaxation theorem is a boundary theorem, not a local allocation theorem for the SAE.
4. Ridge-hat preservation concerns decoders refitted on the reconstructed representation. It gives no universal control of the frozen downstream network or its Fisher pullback. Explicit zero-decoder-distance/large-frozen-change and zero-Fisher/positive-decoder-distance examples are below.
5. Experiment 4b's fresh exact evaluation substantially reduces the practical concern that finite-probe bias alone created the headline result. It gives no evidence that Fisher or activation-manifold geometry explains the representation result, and the mixed IOI result directly demonstrates the refitted-versus-frozen gap.

Two audit corrections should reach the synthesis. First, the `n=64` exact group-size row uses only 8,192 tokens because `exact_max_groups=128` subsamples 128 of 256 groups, while `n=128` and `n=256` use all 16,384 exact tokens. The `n=128` versus `n=256` comparison is unconfounded by this cap, but the full 34%/24%/13% trend is partly confounded with the evaluated token subset at `n=64`. Second, the static spectral baseline is an omission-cost surrogate derived from the rank theorem; it is not the Frechet first-order metric of the ridge-hat map.

## 1. Exact implemented stochastic object

The implementation facts used here are:

- `src/dpsae/language_training.py:129-140` draws a separate Gaussian column for each geometry group and probe, then divides each column by its sample RMS. Thus, ignoring the effectively inactive `1e-6` clamp,
  \[
  y_{gj}=\sqrt n\,u_{gj},\qquad u_{gj}\sim \operatorname{Unif}(S^{n-1}),
  \]
  independently over groups `g` and columns `j`. These are fixed-radius spherical probes, not Gaussian probes after normalization.
- `src/dpsae/language_training.py:171-183` computes one numerator over every group, sample, and probe, divided by one global denominator. It does not average per-group ratios. The numerical denominator is `max(D,1e-12)`, not `D+eps`.
- The reference solve and denominator are under `no_grad`; only the reconstructed solve receives SAE gradients.
- With 2,048 tokens, group size 128, and 16 probes, one step has 16 geometry groups and 256 independent spherical directions.
- The flattened activation batch contains eight length-256 sequences. Reshaping directly into groups of 128 makes the training groups contiguous half-sequences; they are not randomized geometry groups.
- Training uses batch-wide TopK over all 2,048 by 16,384 scores. Natural evaluation uses the learned scalar threshold in 4,096-token reconstruction chunks (`experiments/exp04b_confirmatory.py:507-514`), so the reconstruction rule also changes between training and evaluation.
- The sampled natural-text report repeats the normalized-sphere/global-ratio protocol. The primary exact report replaces probes with the identity and reports a ratio of sums across groups (`src/dpsae/exp04b_natural_text.py:125-180`). It is exact with respect to targets, but the operators are still computed in FP32.

Condition on one activation minibatch and its reconstruction. Let

\[
K_g=K_\lambda(X_g),\qquad \widehat K_g=K_\lambda(\widehat X_g),
\qquad A_g=K_g-\widehat K_g,
\]

and define symmetric positive semidefinite matrices

\[
B_g=A_g^2,\qquad C_g=K_g^2.
\]

Because the hat matrices are symmetric, the exact implemented decoder term before the denominator clamp is

\[
\widehat R_m
=
\frac{Z}{D}
=
\frac{\sum_{g=1}^G\sum_{j=1}^m y_{gj}^\top B_g y_{gj}}
{\sum_{g=1}^G\sum_{j=1}^m y_{gj}^\top C_g y_{gj}}.
\]

There is no `1/m` because it would cancel. The corresponding conditional exact ratio of traces is

\[
R_{\rm tr}
=
\frac{\sum_g\operatorname{tr}B_g}
{\sum_g\operatorname{tr}C_g}
=
\frac{\sum_g\|K_g-\widehat K_g\|_F^2}
{\sum_g\|K_g\|_F^2}.
\]

This exact ratio is a reference-energy-weighted average of per-group relative distortions:

\[
R_{\rm tr}=\sum_g w_g r_g,\qquad
r_g=\frac{\operatorname{tr}B_g}{\operatorname{tr}C_g},\qquad
w_g=\frac{\operatorname{tr}C_g}{\sum_h\operatorname{tr}C_h}.
\]

It is not the equal-weighted mean `G^{-1} sum_g r_g`. Calibrating `tr(K_g)/n` approximately does not fix these weights, because it fixes the first spectral moment while `tr(K_g^2)` is the second. If the eigenvalues lie in `[0,1]` and `tr K_g=n f`, then

\[
n f^2\leq \operatorname{tr}K_g^2\leq n f,
\]

so at `f=0.25` the possible reference-energy weights still span a factor of four in the worst case. In the local seed-0 primary exact rows the denominator coefficient of variation is only 0.0528 and the ratio-of-sums versus mean-of-ratios difference is about `4e-5`, so this weighting distinction is small on that particular held-out sample, but it remains the mathematical estimand.

## 2. Fixed-radius quadratic-form moments

For `y=sqrt(n) u` with `u` uniform on the unit sphere,

\[
\mathbb E[y_i y_j]=\delta_{ij}
\]

and

\[
\mathbb E[y_i y_j y_k y_l]
=\frac{n}{n+2}
(\delta_{ij}\delta_{kl}+\delta_{ik}\delta_{jl}+\delta_{il}\delta_{jk}).
\]

Therefore, for symmetric matrices `M,N`,

\[
\mathbb E[y^\top M y]=\operatorname{tr}M,
\]

\[
\operatorname{Cov}(y^\top M y,y^\top N y)
=\frac{2}{n+2}
\left(n\operatorname{tr}(MN)-\operatorname{tr}M\operatorname{tr}N\right),
\]

and

\[
\operatorname{Var}(y^\top M y)
=\frac{2}{n+2}
\left(n\operatorname{tr}(M^2)-(\operatorname{tr}M)^2\right).
\]

The subtraction term is the radial variance removed by column normalization. For a positive semidefinite `M`, define

\[
r_{\rm eff}(M)=\frac{(\operatorname{tr}M)^2}{\operatorname{tr}(M^2)}.
\]

The relative variance of the mean of `m` normalized-sphere quadratic forms is

\[
\frac{\operatorname{Var}(m^{-1}\sum_j y_j^\top M y_j)}
{(\operatorname{tr}M)^2}
=\frac{2}{m(n+2)}\left(\frac{n}{r_{\rm eff}(M)}-1\right).
\]

By contrast, unnormalized Gaussian probes give `2/(m r_eff)`. The two laws are close when `n` is large and `r_eff` is well below `n`, but they are not identical. In particular, fixed-radius probes have exactly zero variance for `M` proportional to the identity.

Experiment 3 validated the Gaussian numerator law. It did not validate the normalized-sphere, self-normalized ratio used by Experiment 4b training. Its results remain useful as an approximate scale calculation, but its unbiasedness claim should not be carried over verbatim.

For totals over heterogeneous groups, write

\[
b_g=\operatorname{tr}B_g,\quad c_g=\operatorname{tr}C_g.
\]

Then

\[
\mu_Z=\mathbb E Z=m\sum_g b_g,qquad
\mu_D=\mathbb E D=m\sum_g c_g,
\]

\[
V_D=\operatorname{Var}D
=\frac{2m}{n+2}\sum_g
\left(n\operatorname{tr}(C_g^2)-c_g^2\right),
\]

and

\[
C_{ZD}=\operatorname{Cov}(Z,D)
=\frac{2m}{n+2}\sum_g
\left(n\operatorname{tr}(B_g C_g)-b_g c_g\right).
\]

The numerator and denominator are each unbiased for their trace totals. Their ratio is not.

## 3. Ratio bias, variance, and gradient target

Let `rho=mu_Z/mu_D=R_tr`. A second-order delta expansion gives

\[
\mathbb E\widehat R_m
=\rho+
\frac{\rho V_D-C_{ZD}}{\mu_D^2}
+O((Gm)^{-2})
\]

under the usual denominator-concentration conditions. The leading bias is order `1/(Gm)` in a homogeneous regime, but its sign is unrestricted. Alignment of decoder error with high-reference-energy directions raises `C_ZD` and can produce negative bias; anti-alignment can produce positive bias.

The matching first-order variance is most cleanly expressed using

\[
H_g=B_g-\rho C_g.
\]

Because `Z-rho D=sum y^T H_g y`,

\[
\operatorname{Var}(\widehat R_m)
\approx
\frac{2}{m(n+2)(\sum_g c_g)^2}
\sum_g\left[n\operatorname{tr}(H_g^2)-(\operatorname{tr}H_g)^2\right].
\]

This is the relevant effective-rank expression for the ratio. Applying the numerator-only `r_eff(A^2)` law misses the covariance with the random denominator.

### Exact two-dimensional bias counterexamples

Take one group, one probe, and

\[
K=\operatorname{diag}(0.8,0.2).
\]

All hat eigenvalues in `[0,1)` are attainable for any fixed positive ridge by choosing squared singular values `s_i^2=n lambda q_i/(1-q_i)`, so these are valid ridge-hat examples.

For `y=sqrt(2)(cos theta,sin theta)`, if `B=diag(b_1,b_2)` and `C=diag(c_1,c_2)` with positive `c_i`, direct integration gives

\[
\mathbb E\frac{y^\top B y}{y^\top C y}
=\frac{b_1/\sqrt{c_1}+b_2/\sqrt{c_2}}
{\sqrt{c_1}+\sqrt{c_2}}.
\]

1. Let `Khat=diag(0.4,0.2)`. Then `B=diag(0.16,0)`. The exact trace ratio is
   \[
   0.16/(0.64+0.04)=0.235294\ldots,
   \]
   while the expected one-probe ratio is `0.2`: negative bias.
2. Let `Khat=diag(0.8,0)`. Then `B=diag(0,0.04)`. The exact trace ratio is
   \[
   0.04/(0.64+0.04)=0.0588235\ldots,
   \]
   while the expected one-probe ratio is again `0.2`: positive bias.

A two-million-draw FP64 Monte Carlo check gave `0.200004` and `0.199982`, respectively. These examples kill any universal unbiasedness claim. Sixteen probes and sixteen groups reduce the bias; they do not make it identically zero.

### Denominator pathologies

If every `K_g` is positive definite, fixed-radius probes give a deterministic lower bound on `D` in terms of the smallest eigenvalues. Universal results cannot assume this. For rank-deficient `K`, directions can approach its nullspace and make the relative loss unstable. With one group and one probe, inverse-denominator moments can even diverge when the reference has very low rank and `A` acts outside its range. The code's global `clamp_min(1e-12)` makes the implemented expectation finite but changes the objective in that regime. This clamp is inactive for Experiment 4b's full-row-rank activation groups at the selected ridge, so it is a theorem caveat rather than a plausible explanation of the observed result.

### What stop-gradient does and does not do

For SAE parameters `theta`, the original activation `X` is fixed. Hence `K(X)`, the sampled reference, and the denominator have no mathematical dependence on `theta`. Detaching them does not change the true SAE gradient; it saves graph construction and makes the intended one-sided optimization explicit.

Conditioned on probes,

\[
\nabla_\theta\widehat R_m=\frac{\nabla_\theta Z}{D}.
\]

This stochastic gradient is unbiased for

\[
J_m(\theta)=\mathbb E_Y[Z(\theta,Y)/D(Y)],
\]

not for `R_tr(theta)`. In particular,

\[
\mathbb E\left[\frac{\nabla Z}{D}\right]
\ne \frac{\nabla\mathbb E Z}{\mathbb E D}
\]

in general. If the upstream representation or LM were trainable, detaching the reference would instead create a semigradient and would no longer equal the derivative of a symmetric representation distance.

### Finite probes give no deterministic uniform guarantee

For any realized target matrix `Y` with `m<n`, choose a nonzero sample-space direction `v` orthogonal to its columns. One can construct two commuting ridge hats that differ only along `v`; then

\[
(K-\widehat K)Y=0
\]

while

\[
\|K-\widehat K\|_F^2>0.
\]

Thus a single finite probe bank cannot deterministically control the exact trace. Independence of a fixed `A` and a fresh random `Y` makes this adversarial event probability zero, but an optimized model is data-dependent. Experiment 4b handles the practical version correctly by using a fresh target seed and, more decisively, an identity-target exact audit on a held-out corpus tail.

## 4. Relative denominators and the actual guarantee

For a task distribution with covariance `Sigma`,

\[
D_\Sigma^2=\|(K-\widehat K)\Sigma^{1/2}\|_F^2
=\mathbb E\|(K-\widehat K)y\|_2^2.
\]

It controls average absolute disagreement for refitted ridge predictions. It also controls worst-case absolute disagreement over the ellipsoid `y=Sigma^{1/2}u`, `||u||<=1`, because

\[
\|(K-\widehat K)\Sigma^{1/2}u\|_2^2
\leq \|(K-\widehat K)\Sigma^{1/2}\|_{op}^2
\leq D_\Sigma^2.
\]

If `Sigma` is singular, all of these claims are restricted to its range. Zero distance means `(K-Khat)Sigma^{1/2}=0`, not necessarily equality of the two hats.

The exact relative trace

\[
\frac{D_\Sigma^2}{\|K\Sigma^{1/2}\|_F^2}
\]

is a ratio of average squared quantities. It is not an average of per-task relative errors and gives no uniform relative guarantee without a lower bound on `||Ky||` on the protected task set. For example, let

\[
K_\epsilon=\operatorname{diag}(1/2,\epsilon),\qquad
\widehat K_\epsilon=\operatorname{diag}(1/2,\epsilon+\sqrt\epsilon).
\]

With an isotropic prior the global relative trace tends to zero like `4 epsilon`, but for target `e_2` the prediction-relative squared error is `epsilon^{-1}`, which diverges. A target in the exact nullspace of `K` makes the per-target relative denominator zero outright.

For a fixed collection of original groups, the exact denominator is shared by all candidate SAEs, so absolute and relative exact losses rank candidates identically. The reported paired reduction

\[
1-\frac{\sum_g N_g^{\rm DPSAE}}{\sum_g N_g^{\rm MSE}}
\]

does not contain the reference denominator at all. Consequently, the 24% headline reduction and its variation across group sizes are not artifacts of dividing candidate and MSE by slightly different reference energies. Relative normalization matters during stochastic training and when aggregating different batches, not for the paired held-out numerator comparison.

## 5. Group size and minibatch define the estimand

Let a training batch `mathcal B` of `N=2048` rows be partitioned into groups `P={I_1,...,I_G}` of size `n`. Even with infinite probes, the conditional objective is

\[
R_{\mathcal B,P}(\theta)=
\frac{\sum_{g}\|K_\lambda(X_{I_g})-K_\lambda(\widehat X_{I_g})\|_F^2}
{\sum_g\|K_\lambda(X_{I_g})\|_F^2}.
\]

Equivalently, it compares block-diagonal operators

\[
\operatorname{diag}(K_\lambda(X_{I_1}),\ldots,K_\lambda(X_{I_G})).
\]

Every cross-group similarity is discarded. A different partition produces different blocks and generally a different operator.

### Exact fixed-size regrouping counterexample

Use four scalar-feature rows

\[
X=(1,1,1,1)^\top,\qquad
\widehat X=(1,1,-1,-1)^\top
\]

and groups of two.

- Partition `P_1={{1,2},{3,4}}`. Within each group, `Xhat_g=+X_g` or `-X_g`, so `Xhat_g Xhat_g^T=X_g X_g^T`, every ridge hat is identical, and the exact loss is zero.
- Partition `P_2={{1,3},{2,4}}`. Each original group is proportional to `(1,1)`, while each reconstructed group is proportional to `(1,-1)`. These directions are orthogonal and have equal norm. For any positive ridge, the two hats are `q` times orthogonal rank-one projectors, so
  \[
  \frac{\|K-\widehat K\|_F^2}{\|K\|_F^2}=2.
  \]

The same rows, reconstruction, group size, and ridge therefore give exact relative loss 0 or 2 solely through the partition. Recalibrating ridge cannot remove the example because the ratio of two equal-eigenvalue orthogonal projectors remains 2 for every `q>0`.

Changing `n` is even more structural. With `G_g=X_gX_g^T`,

\[
K_\lambda(X_g)=G_g(G_g+n\lambda I_n)^{-1}.
\]

It changes the sample-space dimension, empirical Gram spectrum, number of blocks, and cross-row relations included in each solve. Recalibrating `lambda` to fix `tr K/n` fixes one scalar spectral moment, not the operator or `tr K^2`.

At fixed feature width, a classical `n to infinity` limit cannot maintain a positive `tr(K)/n` forever because `rank(K)<=d`, so `tr(K)/n<=d/n`. A grouping-independent population limit would require an explicit asymptotic regime and distributional assumptions, such as a fixed aspect ratio or a kernel limit. None follows from the implemented objective.

### Minibatch expectation versus corpus ratio

With infinite probes, SGD targets approximately

\[
J_{n,\infty}(\theta)=\mathbb E_{\mathcal B}
[R_{\mathcal B,P}(\theta)],
\]

whereas a large held-out ratio of sums targets

\[
R_{\rm corpus}(\theta)=
\frac{\mathbb E_{\mathcal B}\sum_g N_g(\theta)}
{\mathbb E_{\mathcal B}\sum_g D_g}.
\]

These are not equal. Relative normalization weights every training minibatch equally after dividing by its own reference energy, while the corpus ratio weights batches by reference energy. A simple scalar illustration is two equally likely batch types with `(N,D)=(0.01,0.01)` and `(0.01,1)`: the expectation of batch ratios is `0.505`, while the ratio of expectations is about `0.0198`.

BatchTopK adds another dependency: `Xhat_{I_g}` is computed after global competition across every row in the 2,048-token batch, so a group's reconstruction depends on which scores appear in other groups. Even iid token groups would not make group contributions independent.

### What Experiment 4b establishes about grouping

At fixed `n=128`, contiguous, shuffled, and document-balanced evaluation partitions give similar paired reductions. This says the trained models' advantage is not specific to those tested `n=128` evaluation partitions. It does not say retraining under those partitions would be unchanged.

The `n=128` to `n=256` reduction change, about 24% to 13%, is an unconfounded demonstration that the finite-group estimand matters even after ridge recalibration. The `n=64` row strengthens the directional pattern but has an audit confound: `exact_max_groups=128` makes it use 128 of 256 possible groups, or 8,192 tokens, while the other two sizes use all 16,384 exact tokens. The code path is `src/dpsae/exp04b_natural_text.py:253-267`, invoked with the cap at `experiments/exp04b_confirmatory.py:898-908`.

The scientifically safe claim is therefore: group size materially changes the observed paired advantage, so the method defines a family of finite-group geometries. Avoid the stronger numerical claim that group size alone has established the entire factor-of-three trend until `n=64` is rerun on all groups or all sizes are matched to the same token subset.

## 6. BatchTopK: exact local statement and boundary failures

Let the batch contain `N` rows. For latent `ell`, write the positive preactivation

\[
a_{i\ell}=\operatorname{ReLU}((x_i-b_d)^\top w^e_\ell+b^e_\ell).
\]

BatchTopK selects one set `S` of `Nk` row-latent pairs. On an open cell where all relevant ReLU signs and the strict TopK ordering are fixed,

\[
\widehat x_i=b_d+
\sum_{\ell:(i,\ell)\in S}
\left((x_i-b_d)^\top w^e_\ell+b^e_\ell\right)d_\ell.
\]

Hence the reconstruction is affine on that cell, with block-diagonal input Jacobian

\[
J_i=\sum_{\ell:(i,\ell)\in S}d_\ell(w^e_\ell)^\top,
\qquad
\frac{\partial\widehat x_i}{\partial x_j}=0\quad(i\ne j).
\]

Global competition affects which cell is active, but once the selected set is fixed there are no cross-token derivatives. This is the strongest clean local statement.

At a positive tie between the last selected and first unselected scores, hard TopK can be discontinuous. With one selected latent, scores `(1+epsilon,1)` and decoder atoms `d_1=e_1,d_2=e_2` produce a limit `e_1` from one side and `e_2` from the other. Thus there is no universal Jacobian or local Lipschitz bound across active-set boundaries. The boundaries have measure zero for a continuously distributed fixed model/input, but optimization can move the parameters through them, and small changes to any token can swap the global support budget between tokens.

### `k` is not a rank constraint

With two rows, two features, two decoder atoms, and `k=1`, take codes

\[
Z=I_2,
\]

and decoder `D=I_2`. Every row is one-sparse, but

\[
\widehat X=ZD=I_2
\]

has rank two. With a wide dictionary, a one-sparse code can have rank up to `min(N,d)` because different rows can select different atoms. Conversely, nonnegative codes, unit decoder rows, a shared dictionary, and global support competition can make a desired rank-`r` matrix unattainable. There is no valid identification `BatchTopK k = matrix rank r`.

The isotropic rank theorem therefore proves a negative boundary: over all rank-`r` representations, the optimum retains the top sample-space singular modes. It does not predict which SAE atoms fire, how many matrix modes a `k`-sparse code realizes, or how the active-set optimizer allocates capacity.

### Equal static residual loss, unequal decoder loss

The gap from any fixed covariance residual metric has a concrete two-dimensional example. Let `X=I_2`; its calibration covariance is isotropic, so the implemented static spectral operator is a scalar multiple of the identity. Choose

\[
\widehat X_1=Q_{60^\circ},
\qquad
\widehat X_2=\operatorname{diag}(1+\sqrt2,1).
\]

Both have

\[
\|X-\widehat X_i\|_F^2=2,
\]

and therefore identical loss under every scalar fixed residual weighting. But `Xhat_1 Xhat_1^T=XX^T`, so its decoder distance is zero for every ridge, while `Xhat_2` changes one hat eigenvalue and has positive decoder distance. Both reconstruction tables can be represented by a sufficiently wide nonnegative one-sparse dictionary with unit decoder atoms. Thus no fixed covariance residual loss is universally equivalent to the dynamic hat loss, even inside sparse-reconstructable examples.

### The static spectral baseline is not the local Frechet metric

Let

\[
G=XX^\top,\qquad R=(G+n\lambda I)^{-1},\qquad
K=I-n\lambda R.
\]

The exact differential is

\[
dK=n\lambda R\,(dG)\,R,
\qquad dG=dX\,X^\top+X\,dX^\top.
\]

This local quadratic metric depends on the current reconstruction, two sample-space resolvents, and both terms in `dG`. Along a singular-value perturbation `s -> s+delta s`, its leading squared hat change has coefficient

\[
\left(\frac{2n\lambda s}{(s^2+n\lambda)^2}\right)^2.
\]

The implemented static operator

\[
M=C(C+\lambda I)^{-2}
\]

instead gives feature-space weight

\[
\frac{n s^2}{(s^2+n\lambda)^2}.
\]

That weight reproduces the theorem's full-mode omission cost, up to a common factor, when a singular mode of `X` is deleted. It is not the first-order Frechet metric for a small singular-value perturbation. Experiment 4b therefore rules out the tested static omission-cost surrogate as a quantitative explanation of the 24% effect; it does not compare DPSAE against the exact local Frechet metric, every static covariance objective, or a matched-NMSE frontier.

## 7. Refitted ridge decoders, frozen networks, and Fisher are inequivalent

`K(X)y` is the prediction made after fitting a new ridge decoder on representation `X`. Equality of hats says that this refittable prediction family is unchanged on the protected sample-space tasks. A frozen downstream weight vector is not refitted when `X` is replaced by `Xhat`.

### Zero decoder distance, arbitrarily large frozen logit change

Take

\[
X=I_2,
\qquad \widehat X=XQ,
\]

where `Q` swaps the two feature coordinates. Because right orthogonal transformations preserve the row Gram matrix,

\[
K_\lambda(\widehat X)=K_\lambda(X)
\]

for every ridge, so decoder distance is exactly zero. A refitted decoder `Q^T w` recovers every original linear target. But with frozen `w=M e_1`,

\[
Xw=(M,0)^\top,
\qquad \widehat Xw=(0,M)^\top.
\]

The frozen logit disagreement grows without bound as `M` grows. The MSE anchor discourages such coordinate changes empirically, but decoder geometry alone cannot exclude them.

### Zero frozen/Fisher loss, positive decoder distance

Let the downstream network depend only on feature one, with frozen weight `w=e_1`, and take

\[
X=I_2,
\qquad \widehat X=\operatorname{diag}(1,0).
\]

Then `Xw=Xhat w`, so the frozen outputs agree exactly and the residual lies in the downstream/Fisher nullspace. Yet

\[
K_\lambda(X)=q I_2,\qquad
K_\lambda(\widehat X)=\operatorname{diag}(q,0),
\qquad q=\frac{1}{1+2\lambda},
\]

giving positive isotropic decoder distortion `q^2`. Fisher preservation therefore does not imply refittable decoder preservation either.

For a frozen probabilistic network with activation perturbation `delta`, the local output KL is

\[
\operatorname{KL}(p(h)\|p(h+\delta))
=\tfrac12\delta^\top F(h)\delta+O(\|\delta\|^3),
\]

where `F(h)` is the pullback of output Fisher through the frozen downstream Jacobian. This is token/context-specific, coordinate-sensitive, and local in perturbation size. The ridge-hat objective is groupwise, invariant to feature-space orthogonal changes, and allows decoder refitting. Neither universally bounds the other. Finite feature ablations and patching can also leave the regime where the quadratic Fisher approximation is accurate.

Experiment 4b supplies empirical evidence for this distinction rather than a reason to merge the geometries. The representation metric improves consistently, while frozen IOI effect/collateral behavior is seed-dependent and the proposed continuous-target gate fails under the fixed ridge protocol. Fisher or activation-manifold theory should remain a limitation/follow-up, not a central explanatory theorem.

## 8. What survives Experiment 4b and what does not

### Survives

- **Finite-group refittable preservation:** On the tested GPT-2 site and BatchTopK architecture, DPSAE reduces fresh exact in-group ridge-hat error. This is independent of evaluation probe noise because the primary audit uses identity targets.
- **Dynamic geometry beats the two tested static controls:** Whitening worsens the metric and the static spectral omission-cost surrogate recovers less than 1% versus about 24%. The safe inference is that those two fixed global covariance metrics do not explain the gain.
- **A useful negative rank theorem:** The relaxed isotropic rank problem does not reorder singular modes. Since its static surrogate explains little empirically, the gap must arise from some combination of elementwise sparsity, active-set allocation, overcompleteness, nonorthogonality, reconstruction-dependent hat geometry, or optimization. Experiment 4b does not identify which.
- **Grouping is part of the definition:** The unconfounded `n=128` versus `n=256` change makes this empirically central, while the regrouping counterexample makes it mathematically unavoidable.
- **Refitted and frozen behavior are distinct:** Both exact counterexamples and the mixed IOI data support keeping this distinction explicit.

### Does not survive as a universal or empirically motivated claim

- The finite normalized-probe training loss is an unbiased exact trace-ratio estimator.
- Sixteen probes provide a deterministic guarantee, or Experiment 3's Gaussian variance law applies exactly after per-column normalization and self-normalization.
- The objective estimates one grouping-independent corpus geometry.
- `k` in BatchTopK plays the role of rank `r` in the spectral theorem.
- A local Jacobian argument applies through active-set boundaries or yields a global spectral allocation rule.
- The tested static spectral baseline is the Frechet derivative metric of `K`.
- Small decoder distance implies compatibility with frozen downstream weights, causal specificity, low output KL, or low Fisher distance.
- Experiment 4b motivates Fisher-pullback or activation-manifold geometry as a main-paper explanation.

## 9. Exact gaps and highest-value follow-ups

1. **Match the theory to the normalized ratio.** Any estimator appendix should state the fixed-radius moment and ratio delta formulas above. The existing Experiment 3 text should be scoped to its unnormalized Gaussian numerator estimator.
2. **Remove the `n=64` token-count confound.** Recompute the existing exact audit with all 256 groups, or subsample every group size to the same underlying token set and comparable group count. No retraining is needed.
3. **Measure gradient fidelity on frozen checkpoints.** On several held-out training-shaped batches, compute the exact identity-target decoder gradient and compare it with normalized-sphere gradients at `m in {1,4,8,16,32}` using cosine, norm ratio, and bias over many probe banks. This directly tests the stochastic optimization bridge; numerator value error alone is weaker.
4. **Separate group and batch normalization.** Report both ratio-of-sums and mean-of-group-ratios, plus `E_batch ratio` versus a corpus ratio on fixed reconstructions. The local primary exact sample suggests the first distinction is numerically small at `n=128`, but this has not been checked during training.
5. **Retrain only if mechanism attribution matters.** Evaluation regrouping shows estimand sensitivity, not how training under `n=64/256` or shuffled groups changes the learned SAE. Retraining a small matched fleet is required before claiming a training mechanism.
6. **Do not spend the next experiment on Fisher.** The higher-value frozen-network test is the audit's proposed locally linear target with a valid dense recoverability gate and equal-IOI-effect matching before collateral KL. That tests the missing bridge directly without asserting that a local Fisher metric explains DPSAE.
7. **Test sparsity mechanism before naming BatchTopK as causal.** Reusing the decoder term with TopK/JumpReLU or a nonsparse matched bottleneck is needed to separate objective-level geometry from BatchTopK-specific active-set behavior.

No remote GPU run was necessary for these conclusions. The decisive new checks were algebraic, a small FP64 Monte Carlo counterexample, and direct inspection of the versioned Experiment 4b artifacts and evaluation cap.

## 10. Cross-red-team of the operator/spectral claims

This section independently attacks the five candidate operator claims supplied after the first round.

### 10.1 Expected disagreement and covariance-ellipsoid bound: survives with scope

Let `A=K_lambda(X)-K_lambda(Z)` and let the task distribution have second moment `E[yy^T]=Sigma`, where `Sigma` is positive semidefinite. Then exactly

\[
\mathbb E\|Ay\|_2^2
=\mathbb E\operatorname{tr}(Ayy^\top A^\top)
=\operatorname{tr}(A\Sigma A^\top)
=D_\Sigma^2(X,Z).
\]

These are predictions from ridge decoders fit separately on `X` and `Z`, evaluated on the same in-group target vector. The statement is transductive and absolute; it does not concern new activation rows or frozen downstream weights.

Define the covariance ellipsoid

\[
\mathcal E_\Sigma
=\{\Sigma^{1/2}u:\|u\|_2\leq1\}
=\{y\in\operatorname{range}\Sigma:y^\top\Sigma^\dagger y\leq1\}.
\]

Then

\[
\sup_{y\in\mathcal E_\Sigma}\|Ay\|_2^2
=\|A\Sigma^{1/2}\|_{op}^2
\leq\|A\Sigma^{1/2}\|_F^2
=D_\Sigma^2.
\]

Thus `D_Sigma` bounds worst-case absolute norm error and `D_Sigma^2` bounds squared error. This survives singular `Sigma`, but only on its range. The claim fails if silently upgraded to a relative guarantee: targets with tiny `||Ky||` can have arbitrarily large relative error even when `D_Sigma` is small, as shown in Section 4.

### 10.2 Zero distance and row Grams: survives for positive ridge and full-rank task covariance

Write `tau=n lambda>0`, `G=XX^T`, and

\[
K_\lambda(X)=G(G+\tau I)^{-1}.
\]

If `Sigma` is positive definite, `D_Sigma=0` implies `K_lambda(X)=K_lambda(Z)`. The map from a positive semidefinite row Gram to its ridge hat is injective because every hat eigenvalue is below one and

\[
G=\tau K(I-K)^{-1}.
\]

Therefore

\[
D_\Sigma(X,Z)=0
\quad\Longleftrightarrow\quad
XX^\top=ZZ^\top
\]

for `lambda>0` and `Sigma` positive definite. No full-rank assumption on `X` or `Z` is needed; “full rank” must refer to the task covariance.

Both qualifications are necessary. If `Sigma=diag(1,0)`, hats `diag(a,b)` and `diag(a,c)` have zero weighted distance for arbitrary distinct `b,c`; their row Grams differ under the inverse map above. If `lambda=0`, the hat becomes a projection and loses singular-value information: `X=I` and `Z=2I` both give `K=I` but have row Grams `I` and `4I`.

### 10.3 Activation-error upper bound: survives as a loose one-way bound

For `G=XX^T`, `H=ZZ^T`, and `tau=n lambda`, the resolvent identity gives

\[
K(X)-K(Z)
=\tau(H+\tau I)^{-1}(G-H)(G+\tau I)^{-1}.
\]

Since each resolvent has operator norm at most `1/tau`,

\[
\|K(X)-K(Z)\|_F
\leq\frac1{n\lambda}\|XX^\top-ZZ^\top\|_F.
\]

Writing `E=X-Z`,

\[
XX^\top-ZZ^\top=EX^\top+ZE^\top,
\]

so

\[
\boxed{
\|K(X)-K(Z)\|_F
\leq
\frac{\|X\|_{op}+\|Z\|_{op}}{n\lambda}\|X-Z\|_F.}
\]

Consequently,

\[
D_\Sigma^2(X,Z)
\leq
\|\Sigma\|_{op}
\frac{(\|X\|_{op}+\|Z\|_{op})^2}{(n\lambda)^2}
\|X-Z\|_F^2.
\]

The bound is valid for rank-deficient matrices and both `n<d` and `n>d`, but it can be very loose at small ridge or large activation norm. It gives no useful relative bound without a lower bound on `||K(X)Sigma^{1/2}||_F`. There is no converse: `Z=XQ` for a nontrivial right orthogonal `Q` can have positive activation error while its decoder distance is exactly zero.

### 10.4 Isotropic rank-`r` optimum: survives exactly for the relaxed problem

Let

\[
X=U\operatorname{diag}(\sigma_1,\ldots,\sigma_s)V^\top,
\qquad
q_i=\frac{\sigma_i^2}{\sigma_i^2+n\lambda},
\]

with descending positive singular values. Then

\[
K(X)=U\operatorname{diag}(q_1,\ldots,q_s)U^\top.
\]

For any `Z` with `rank(Z)<=r`, `K(Z)` has rank at most `r`. The best arbitrary rank-`r` matrix approximation to `K(X)` has squared Frobenius error `sum_{i>r}q_i^2`, so this is a lower bound for the more restricted ridge-hat problem. It is attained by the truncated representation

\[
X_r=U_r\operatorname{diag}(\sigma_1,\ldots,\sigma_r)V_r^\top,
\]

whose hat retains exactly the first `r` values `q_i`. Hence, for `0<=r<=min(n,d)`,

\[
\boxed{
\min_{\operatorname{rank}Z\leq r}
\|K(X)-K(Z)\|_F^2
=\sum_{i>r}q_i^2.}
\]

The relative exact isotropic denominator is constant in `Z`, so it has the same minimizers. With a strict cutoff `q_r>q_{r+1}`, the optimal sample-space hat is unique. Ties permit arbitrary `r`-dimensional choices inside the tied left-singular subspace, and the representation itself always remains nonunique under right orthogonal feature rotations.

This theorem does not transfer to the expected finite self-normalized probe ratio, to a shared SAE trained across groups, or to BatchTopK by identifying `k` with `r`. Those are precisely the gaps isolated in Sections 3, 5, and 6.

### 10.5 Structured priors: commuting ranking survives; noncommuting mode ranking fails

Suppose `Sigma` commutes with `K(X)`, so in a common orthonormal basis

\[
K(X)=\operatorname{diag}(q_i),
\qquad
\Sigma=\operatorname{diag}(\omega_i),
\qquad \omega_i\geq0.
\]

The weighted loss is

\[
\|(K-M)\Sigma^{1/2}\|_F^2.
\]

For every rank-`r` candidate `M`, `M Sigma^{1/2}` also has rank at most `r`. The unconstrained best rank-`r` approximation to `K Sigma^{1/2}=diag(q_i sqrt(omega_i))` retains the entries with largest `omega_i q_i^2`. A diagonal ridge hat that retains the corresponding `q_i` attains that lower bound, so the exact constrained optimum does rank modes by

\[
\boxed{\omega_i q_i^2.}
\]

Ties and zero weights create nonuniqueness, and the statement requires commutation; it is not a formula for arbitrary task covariance.

A full-rank two-dimensional counterexample shows why. Take

\[
K=\operatorname{diag}(4/5,1/5),
\qquad
\Sigma=
\begin{pmatrix}
11/20&1/2\\
1/2&11/20
\end{pmatrix}.
\]

`Sigma` has eigenvalues `21/20` and `1/20` and does not commute with `K`. Under a rank-one budget, retaining the first `K` eigenmode costs `11/500=0.022`, while retaining the second costs `44/125=0.352`. Now let

\[
v=(4,1)^\top/\sqrt{17},
\qquad
M=\frac{61}{89}vv^\top.
\]

`M` is a valid rank-one ridge hat because its nonzero eigenvalue lies in `(0,1)`. Direct substitution gives

\[
\operatorname{tr}[(K-M)\Sigma(K-M)]
=\frac{964}{189125}
\approx0.00510,
\]

strictly below both eigenmode-selection costs. Thus a noncommuting structured prior can prefer a rotated sample-space direction.

For positive definite noncommuting `Sigma`, relaxing symmetry and ridge-hat attainability turns the problem into a truncated-SVD approximation of `K Sigma^{1/2}`. Mapping that relaxed solution back by `Sigma^{-1/2}` generally produces a nonsymmetric matrix, so it need not be a valid ridge hat. The exact general problem is therefore a constrained weighted low-rank approximation, not a scalar mode-ranking rule or an automatic generalized-eigenvalue formula.

## 11. Strongest finite-probe gradient statement

The clean theorem has to distinguish four objects that are easy to conflate.

1. For unnormalized Gaussian probes, the trace estimator
   \[
   \widetilde a_m=\frac1m\sum_{g,j}y_{gj}^\top B_g y_{gj},
   \qquad y_{gj}\sim N(0,I_n),
   \]
   is unbiased for `a=sum_g tr(B_g)`, with
   \[
   \operatorname{Var}(\widetilde a_m)
   =\frac2m\sum_g\operatorname{tr}(B_g^2).
   \]
   For a parameter coordinate `theta_k`, set `M_{gk}=partial B_g/partial theta_k`. On a differentiable cell,
   \[
   \mathbb E\,\partial_k\widetilde a_m=\partial_k a,
   \qquad
   \operatorname{Cov}(\partial_k\widetilde a_m,
   \partial_l\widetilde a_m)
   =\frac2m\sum_g\operatorname{tr}(M_{gk}M_{gl}).
   \]
   The value variance is controlled by the effective rank of `B_g`; the gradient variance is controlled by the derivative matrices and need not follow the same effective rank.

2. Conditional on the target-normalization clamp being inactive, the implemented probes have fixed radius and the `1/m`-averaged numerator gradient remains unbiased, with covariance
   \[
   \operatorname{Cov}(\partial_k\widetilde a_m,
   \partial_l\widetilde a_m)
   =\frac{2n}{m(n+2)}\sum_g
   \left[
   \operatorname{tr}(M_{gk}M_{gl})
   -\frac{\operatorname{tr}M_{gk}\operatorname{tr}M_{gl}}n
   \right].
   \]
   Column normalization removes radial noise, so importing the Gaussian covariance formula is conservative in some directions and exact only in special cases. Strictly, the code divides by `max(sample_rms,1e-6)`. Under that exact law, isotropy gives \(\mathbb E[yy^T]=c_nI\) for a scalar \(c_n<1\), so the unnormalized numerator and gradient have expectation \(c_n\) times the trace targets. At `n=128` the difference from one is astronomically small, and an audit can make the fixed-radius formulas exact for its realized banks by requiring zero target-normalization clamp hits.

3. Let
   \[
   J_m(\theta)=
   \mathbb E_Y\left[
   \frac{Z_m(\theta,Y)}{\max(D_m(Y),\epsilon)}
   \right],
   \qquad \epsilon=10^{-12}.
   \]
   On any fixed BatchTopK/ReLU active-set cell, and whenever differentiation can pass through the expectation,
   \[
   \boxed{
   \mathbb E_Y\nabla_\theta\widehat R_m(\theta,Y)
   =\nabla_\theta J_m(\theta).}
   \]
   This is the exact unbiasedness statement for the code. Because the source hat and denominator do not depend on SAE parameters,
   \[
   \nabla_\theta\widehat R_m
   =\frac{\nabla_\theta Z_m}{\max(D_m,\epsilon)};
   \]
   the `no_grad` block does not alter the mathematical SAE gradient. In general \(\nabla J_m\ne\nabla R_{\rm tr}\), because the same probes create a random inverse denominator correlated with the numerator gradient.

4. If `b=sum_g tr(C_g)>0`, fixed activations and reconstructions give, almost surely as the number of independent probes per group tends to infinity,
   \[
   Z_m/m\to c_na,\qquad D_m/m\to c_nb,
   \qquad \nabla Z_m/m\to c_n\nabla a,
   \]
   and therefore
   \[
   \boxed{
   \nabla\widehat R_m\to\nabla R_{\rm tr}
   =\frac{\nabla a}{b}.}
   \]
   The scalar \(c_n\) cancels in the ratio, so this convergence statement covers the exact target-normalization clamp law. Expected convergence additionally needs uniform integrability or a denominator lower-tail condition. The denominator clamp guarantees a finite implemented loss, but it changes the target if it fires; when `b>0`, it is eventually inactive almost surely as `m` grows. Parameter-gradient convergence also requires the checkpoint to remain in a differentiable fixed active-set cell; the frozen reconstruction-space statement does not.

The first-order self-normalization bias can be written explicitly. Let

\[
v_C=\frac{2n}{n+2}\sum_g
\left[\operatorname{tr}(C_g^2)-\frac{(\operatorname{tr}C_g)^2}{n}\right]
\]

and

\[
c_{M_kC}=\frac{2n}{n+2}\sum_g
\left[\operatorname{tr}(M_{gk}C_g)
-\frac{\operatorname{tr}M_{gk}\operatorname{tr}C_g}{n}\right].
\]

Writing `h_k=sum_g tr(M_gk)`, a denominator-concentrated expansion gives

\[
\mathbb E[\partial_k\widehat R_m]-\frac{h_k}{b}
=\frac1m\left[
\frac{h_k v_C}{b^3}-\frac{c_{M_kC}}{b^2}
\right]+O(m^{-2}).
\]

The sign is unrestricted. The two diagonal examples in Section 3 also give both gradient-bias signs when the changed hat eigenvalue is treated as a scalar parameter. In the first example, the expected one-probe objective is `1.25(0.8-q)^2`, versus exact `0.68^{-1}(0.8-q)^2`, so at `q=0.4` the gradient bias is positive. In the second, the expected objective is `5(0.2-q)^2`, versus exact `0.68^{-1}(0.2-q)^2`, so at `q=0` the gradient bias is negative. Thus no finite-`m` theorem can promise even a one-sided gradient bias.

For audit calculations, the reconstruction-space gradients have a closed form. Write `tau=n lambda`, `R_g=(Z_gZ_g^T+tau I)^{-1}`, and `S_g=sum_j y_gj y_gj^T`. Then

\[
\nabla_{Z_g} Z_m
=-2\tau R_g(A_gS_g+S_gA_g)R_g Z_g,
\]

while identity targets give

\[
\nabla_{Z_g} a
=-4\tau R_gA_gR_gZ_g.
\]

These formulas permit a frozen-checkpoint audit without retaining a full SAE parameter graph. They also isolate estimator fidelity from the architecture Jacobian.

## 12. Frozen-checkpoint gradient-fidelity audit

The recommended contract freezes all six final confirmatory checkpoints, MSE and DPSAE for seeds 0--2, then uses the same 12 held-out training-shaped batches for every checkpoint. Reconstruct each 2,048-row batch with the training BatchTopK rule and its exact global `2,048 x k` support budget, not the learned evaluation threshold, then reshape contiguously into `16 x 128` groups. Use the checkpoint's selected ridge and draw 256 independent maximum probe banks per batch. Within each bank, treat `m in {1,2,4,8,16,32,64}` as paired prefixes of the same 64 probes. The closed-form gradients permit streaming over banks, groups, and checkpoints, so no bank-gradient fleet or full autograd graph needs to remain resident. Compute the mathematical reference in FP64, then spot-check the implemented FP32 autograd gradient at `m=16` on at least one batch per checkpoint to separate sampling error from numerical or implementation error.

For every frozen batch, estimate four gradients:

- `g_id`, the exact identity-target gradient of `a/b` in row-Gram space and reconstruction space;
- `g_m`, the implemented sampled gradient of `Z_m/D_m`;
- `g_m_fixed`, the control gradient of `Z_m/(m b)`, which is exactly unbiased for `g_id` conditional on zero target-normalization clamp hits and isolates projection noise. For the strict clamped probe law, replace `m b` by `m c_n b`;
- optionally, the Gaussian unnormalized gradient, which should reproduce the Experiment 3 moment law and acts as an implementation control rather than a model claim.

Report paired estimands rather than treating millions of gradient coordinates as independent observations:

1. `cos(mean_bank g_m, g_id)`, `||mean_bank g_m||/||g_id||`, and `||mean_bank g_m-g_id||/||g_id||` measure mean-gradient direction, scale, and bias.
2. `sqrt(mean_bank ||g_m-g_id||^2)/||g_id||` and the distribution of per-bank cosines measure stochastic noise seen by SGD.
3. `Pr(g_m dot g_id>0)` measures whether a sampled decoder step is locally descending for the exact objective.
4. Denominator coefficient of variation, minimum denominator, and clamp-hit rate diagnose whether the asymptotic expansion applies.
5. The paired difference between `mean g_m` and `mean g_m_fixed` estimates self-normalization bias; `g_m_fixed` must agree with `g_id` within Monte Carlo uncertainty.

Conditional probe noise should scale as `m^{-1/2}`. Once resolvable above Monte Carlo error, self-normalization bias should scale as `m^{-1}`; with comparable independent groups, both constants benefit from the aggregate group count. Fit slopes on log scale but do not claim the `-1` bias law if the estimated bias is smaller than its confidence interval. Construct Monte Carlo intervals by resampling whole probe banks, then aggregate checkpoints with a paired seed bootstrap and batches with a batch-level bootstrap. Gradient coordinates are not replicates.

Predeclare the following empirical adequacy gate for the implemented `m=16`; these thresholds are research decisions, not consequences of the moment theorem:

- no target-normalization or denominator-clamp hits in any audited bank;
- median denominator coefficient of variation at most `0.10`, with its 90th batch percentile at most `0.15`;
- median batch mean-gradient cosine at least `0.99`, with its 10th batch percentile at least `0.95`;
- median batch norm ratio in `[0.95,1.05]`;
- median relative mean bias at most `0.05`, with its 90th batch percentile at most `0.10`;
- at least `95%` of individual probe banks have positive dot product with the exact gradient;
- the fixed-denominator control's relative mean error is at most `0.02` or statistically indistinguishable from zero at the Monte Carlo resolution, whichever is larger.

Apply these gates checkpoint by checkpoint. A full gradient-fidelity claim requires every DPSAE checkpoint to pass; the MSE checkpoints are geometry controls and should be reported separately if they fail. Require the FP32 autograd spot check to agree with the analytic sampled gradient to relative error `1e-3` and cosine at least `0.9999`. Fit the stochastic-RMSE log slope over `m=2--32`; values in `[-0.65,-0.35]` are consistent with the predicted `m^{-1/2}` law. Treat the self-normalization bias slope as diagnostic rather than a hard gate unless its magnitude is resolved above Monte Carlo uncertainty.

Failure of this gate would not invalidate the exact held-out result. It would show that the implemented 16-probe optimizer is a materially noisy or biased bridge to that result, so the paper should describe finite-probe training as an empirical surrogate rather than an accurate exact-gradient estimator.

The primary audit belongs in row-Gram and reconstruction space. A secondary raw parameter-gradient audit can apply the fixed checkpoint Jacobian, but it must occur before mixing with MSE, gradient clipping, projected decoder gradients, or optimizer state. Those operations answer optimizer questions, not estimator fidelity.

## 13. Row-Gram dependence and architecture portability

At the loss level, the decoder term depends on a reconstruction only through its row Gram. With

\[
\widehat G=ZZ^T,
\qquad
K(\widehat G)=I-\tau(\widehat G+\tau I)^{-1},
\]

the differential factorizes as

\[
dK=\tau(\widehat G+\tau I)^{-1}(d\widehat G)
(\widehat G+\tau I)^{-1},
\qquad
d\widehat G=dZ\,Z^T+Z\,dZ^T.
\]

Consequently, the same objective can be attached to any differentiable architecture that emits reconstructed rows, and right-orthogonal feature rotations are invisible to it. Infinitesimal directions with `dG=0` are also invisible to the decoder term. This is objective portability, not optimizer or representation portability.

For architecture parameters `theta`,

\[
\nabla_\theta L
=J_{G,\theta}^T\nabla_G L
=J_{Z,\theta}^T\nabla_Z L.
\]

Changing architectures changes the reachable row-Gram set, the Gram Jacobian and tangent cone, active-set discontinuities, conditioning, sparsity allocation, and batch coupling. BatchTopK makes a group's selected support depend on scores elsewhere in the 2,048-row batch; tokenwise TopK does not. Equal `L0`, NMSE, or bottleneck width therefore does not make their gradient fields equivalent. Even equal row Grams establish equal decoder loss values, not equal feature semantics, coordinate compatibility with a frozen model, or equal first-order descent unless the architectures' attainable Gram tangent maps also match.

The architecture-screen claim should remain narrow: the seed-0 tokenwise result is evidence that the measured gain is not unique to batch-global support competition in that screen. It is not a portability theorem and does not establish architecture-general robustness. Likewise, the nonorthogonal counterfactual can falsify decoder-atom nonorthogonality as a necessary checkpoint-level explanation, but it cannot establish that active-set allocation, batch coupling, or architecture choice is irrelevant.

## 14. Compact claim ledger

- **Exact:** Gaussian unnormalized numerator gradients are unbiased for the identity-target numerator gradient. Fixed-radius gradients are also unbiased conditional on the target-normalization clamp being inactive; their covariance law differs by the spherical trace-subtraction term.
- **Exact:** The implemented sampled gradient is unbiased for \(\nabla J_m\), the expected clamped self-normalized finite-probe objective, not for the exact trace-ratio gradient.
- **Asymptotic:** With positive reference energy and denominator control, the sampled gradient converges to the identity-target gradient as independent probe count grows; stochastic noise is order `m^{-1/2}` and self-normalization bias is generically order `m^{-1}`.
- **Impossible at finite probes:** No deterministic uniform exact-gradient or exact-value guarantee exists for `m<n`; an unseen sample-space direction can carry positive Frobenius error and zero sampled error.
- **Exact:** The decoder loss factors through the reconstruction row Gram, so its definition is architecture-agnostic and right-orthogonally invariant.
- **Not implied:** Row-Gram dependence does not give optimizer portability, equal learned features, frozen-network compatibility, or architecture-general empirical gains.
- **Identity-audit only:** For `Q=A_M^T A_M-A_D^T A_D`, `sum tr Q>0` means DPSAE wins on average over isotropic identity targets. It does not imply `Q` is positive semidefinite or that DPSAE wins every target direction, and it says nothing directly about the sampled training gradient.

## 15. JumpReLU controller calibration after the shared-setting failure

The observed `14x--16x` integration rows show that one shared threshold-controller learning-rate multiplier does not impose a shared sparsity constraint: MSE rises from `27.28` to `30.48`, while DPSAE rises from `31.58` to `36.86`. Separate controller calibration is therefore valid if the target estimand is a comparison of separately calibrated methods at `L0=32`. It no longer supports the stronger description that every optimizer hyperparameter is identical, because controller dynamics can affect learned representations beyond their final L0.

The selection rule must be frozen before inspecting reconstruction or decoder outcomes:

1. Evaluate inference L0 on one fixed calibration split, using block uncertainty over sequences or chunks and a final-window/checkpoint average rather than one last training batch.
2. Fit a monotone piecewise-linear response of L0 to multiplier separately for MSE and DPSAE. Interpolate only inside the first adjacent pair bracketing 32; do not extrapolate. DPSAE is already bracketed by `14x=31.58` and `15x=35.17`, giving about `14.12x`. MSE has no upper bracket because `16x=30.48`, so the next blind run is `17x`, followed by one-step increases only if 32 remains unbracketed.
3. Confirm each interpolated setting once on a fresh calibration stream or seed. Require finite gradients, dead-feature fraction below 10%, and a selection L0 whose point estimate and preferably 95% block interval lie inside `[30.4,33.6]`.
4. Freeze one multiplier per objective before the 25M-token screen. Reapply the L0 gate at 25M without retuning or selecting among long runs by decoder loss.

If only the predeclared 5% band is needed, the existing blind choices are MSE `16x` and DPSAE `14x`, assuming the quoted L0 values are fixed selection-set means. Their realized mismatch, `30.48` versus `31.58`, should be reported and supplemented with an equal-inference-L0 threshold sensitivity analysis.

The final exact comparison should pair seeds, token streams, initialization, and probe streams, then bootstrap held-out geometry groups identically across methods. L0 matching does not match NMSE, dead-feature rate, or threshold dispersion; those remain comparability diagnostics. If NMSE differs materially, the result is a decoder-preservation frontier comparison at matched sparsity, not a clean architecture-portability claim.
