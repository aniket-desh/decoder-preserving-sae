# Sparse-mechanism attribution from frozen checkpoints

## Starting evidence and the claim ceiling

The newest artifact is `artifacts/exp04b_valid_target_equal_effect/result.json`, completed on the RunPod in 58.96 seconds with 1.44 GiB peak allocated GPU memory. It uses the existing checkpoints and immutable IOI cache. The repository revision is `b5544fc...`, but the remote tree was dirty because of an untracked `uv.lock`; the result hash was verified after copying locally, though a paper-facing run should come from a clean committed evaluator revision.

The valid target is

\[
t(h)=h^\top w,\qquad
w=\frac{\mathbb E_{\rm rank}[h_{\rm clean}-h_{\rm ABC}]}
{\|\mathbb E_{\rm rank}[h_{\rm clean}-h_{\rm ABC}]\|}.
\]

Because the target is explicitly linear in the original activation, the dense gate is primarily an implementation and split-stability check. It passes with test (R^2=0.99961), and all 64-feature sparse reports are high ((0.9885)--(0.9931)). The paired DPSAE-minus-MSE top-64 improvements are (+0.00264,+0.00069,+0.00029); the last interval crosses zero. Static spectral is also positive in every seed and larger in each corresponding seed ((+0.00463,+0.00210,+0.00128)). Full-reconstruction DPSAE-minus-MSE (R^2) has signs (-,+,-). This target therefore validates a linear-access evaluation path but does not uniquely implicate DPSAE's sparse mechanism.

The equal-effect result is also mixed. At the validation-frozen IOI target effect (2.0339), DPSAE-minus-MSE test collateral KL is (-0.00779,-0.00595,+0.00404) across seeds. Thus equal-effect matching improves the comparison but still gives no seed-consistent frozen-behavior claim. Mechanism attribution should remain centered on the replicated natural-text hat result: about 24% lower exact distortion at about 7% worse NMSE.

The three candidate mechanisms below are deliberately narrower:

1. **Active-set allocation:** the DPSAE encoder assigns sparse atom sets to tokens in a way that better preserves sample-space prediction geometry.
2. **Learned-atom nonorthogonality:** correlations among co-used decoder atoms supply Gram cross-terms that specifically reduce hat distortion.
3. **Reconstruction-dependent row-Gram geometry:** DPSAE produces a different row-Gram perturbation, possibly involving source-eigenspace rotation or nonlinear endpoint effects that a fixed residual metric cannot capture.

These mechanisms can interact. Frozen-checkpoint interventions can establish that one is necessary for the observed checkpoint advantage or that it is compatible with the advantage; they cannot prove which training gradient originally caused the parameters to move.

## 1. Why scalar losses cannot identify the mechanism

For one group, write the SAE reconstruction as

\[
Z=CD+\mathbf 1b^\top,
\]

where (C\in\mathbb R_+^{n\times p}) is the sparse code, (D\in\mathbb R^{p\times d}) has unit-norm rows, and (b\in\mathbb R^d). The ridge hat depends on these objects only through

\[
G_Z=ZZ^\top.
\]

Consequently neither NMSE nor decoder loss can identify a code/dictionary mechanism. Even the entire reconstruction (Z) does not identify it. For a one-row reconstruction (z=e_1\in\mathbb R^2), choose, for any (0<\theta<\pi/2),

\[
d_1=(\cos\theta,\sin\theta),\quad
d_2=(\cos\theta,-\sin\theta),\quad
c_1=c_2=(2\cos\theta)^{-1}.
\]

Both atoms are unit norm, both coefficients are nonnegative and active, and (c_1d_1+c_2d_2=e_1) for every θ. Yet (d_1^\top d_2=\cos2\theta) ranges from nearly (1) to nearly (-1). Hence nonorthogonality can vary arbitrarily at fixed active count, fixed reconstruction, fixed row Gram, and fixed decoder loss.

Likewise orthogonality does not remove active-set effects. With (D=I_2), source (X=I_2), and one active atom per row, (C=I_2) reconstructs perfectly while

\[
C'=\begin{pmatrix}1&0\\1&0\end{pmatrix}
\]

uses the same orthogonal dictionary and the same per-row sparsity but changes the row Gram and incurs positive decoder distortion. These constructions falsify three tempting inferences:

- lower decoder loss does not imply lower decoder coherence;
- the rank-relaxation gap does not by itself prove nonorthogonal-feature use;
- equal (k) does not make two support allocations geometrically comparable.

Internal checkpoint interventions are therefore required.

## 2. Common evaluation setup and headline estimand

Use only the three paired MSE/DPSAE seed checkpoints, the immutable natural-selection and natural-test activation caches, and the existing exact-test subset. No language model, training data stream, or stochastic probes are needed.

For every test group (g) of (n=128) rows, let

\[
G_g=X_gX_g^\top,
\qquad
H_{mg}=Z_{mg}Z_{mg}^\top,
\qquad
\tau=n\lambda,
\]

and define

\[
K(A):=A(A+\tau I)^{-1},qquad
N_{mg}:=\|K(G_g)-K(H_{mg})\|_F^2,qquad
D_g:=\|K(G_g)\|_F^2.
\]

All mechanism comparisons should use numerator differences and the same reference denominator,

\[
\mathcal A_Q
:=\frac{\sum_g N^{Q}_{\mathrm{MSE},g}
-\sum_g N^{Q}_{\mathrm{DPSAE},g}}
{\sum_gD_g},
\]

where \(Q\) denotes a counterfactual. Positive values favor DPSAE. Paired group bootstraps should resample the same group indices for both models and every counterfactual. Report the raw groupwise values as well as confidence intervals; a percent reduction can obscure a mechanism if its counterfactual changes the overall loss scale.

The primary path should reproduce the headline exactly with threshold encoding. A forced-BatchTopK control should encode the same 2,048-token chunks used in training with exactly (2048k) retained entries, then regroup into 128 rows. If a mechanism signature exists only under unequal learned thresholds and vanishes at fixed active count, it is an inference-L0 effect rather than evidence about sparse allocation at the matched training budget.

## 3. Hypothesis A: active-set allocation

### 3.1 A support-only reconstruction

Let (M_{mg}=\mathbf1\{C_{mg}>0\}) be the binary test support. Learned amplitudes can conceal whether the support itself matters, so estimate one typical positive amplitude per feature using only the natural-selection cache:

\[
\mu_{mj}
=\mathbb E_{\rm selection}[C_{ij}\mid C_{ij}>0],
\]

using the global positive-activation median as a preregistered fallback for any feature never active on selection. Freeze these values before test evaluation and form

\[
C^{\rm sup}_{mg}(\alpha)
=\alpha M_{mg}\operatorname{diag}(\mu_m),
\qquad
Z^{\rm sup}_{mg}(\alpha)
=C^{\rm sup}_{mg}(\alpha)D_m+\mathbf1b_m^\top.
\]

Sweep a small preregistered α grid fixed from selection data. This produces a decoder-distortion/NMSE curve using feature identities and supports but not token-specific learned amplitudes. Interpolate only where paired MSE and DPSAE curves share NMSE support.

Define (\mathcal A_{\rm sup}(e)) as the paired decoder advantage at common NMSE (e). The support-sufficiency fraction is descriptively

\[
S_{\rm sup}(e)=\frac{\mathcal A_{\rm sup}(e)}{\mathcal A_{\rm full}},
\]

but its uncertainty should come from paired bootstrap draws rather than a ratio of point estimates.

**Pass interpretation.** If (\mathcal A_{\rm sup}(e)>0) across all seeds with paired intervals excluding zero, and it retains a substantial fraction of (\mathcal A_{\rm full}), the learned binary support plus atom identities is sufficient for a checkpoint-level advantage even after amplitudes and NMSE are controlled.

**Fail interpretation.** If (\mathcal A_{\rm sup}(e)\le0), binary support is not sufficient. The advantage must use token-specific amplitudes, decoder correlations, bias interactions, or their combination. This does not prove the support was irrelevant during training.

### 3.2 Degree-preserving allocation null

To test the specific token-to-atom assignment, randomize \(M\) by bipartite double-edge swaps. Replace active edges \((i,j),(k,\ell)\) with \((i,\ell),(k,j)\) only when the proposed edges are inactive. This preserves every token's L0 and every feature's firing count exactly while destroying the learned incidence pattern. Generate at least 20 independently mixed masks per group using a fixed seed, require at least ten accepted swaps per active edge, and report final Jaccard overlap with the original mask as a mixing diagnostic. Then reuse the frozen μ and α curves.

Let (\overline{\mathcal A}_{\rm null}(e)) be the paired advantage averaged over randomized masks, and define the allocation-specific contrast

\[
\mathcal C_{\rm alloc}(e)
=\mathcal A_{\rm sup}(e)-\overline{\mathcal A}_{\rm null}(e).
\]

**Active-allocation signature.** (\mathcal A_{\rm sup}>0), (\mathcal C_{\rm alloc}>0), and (\overline{\mathcal A}_{\rm null}) near zero means DPSAE's particular token-feature incidence pattern, rather than its row/feature degree marginals, carries the advantage.

**Marginal-utilization signature.** If the randomized masks retain the advantage, the result is consistent with differences in which atoms are globally used or in per-row L0 allocation, not the fine token-feature matching.

**Confound check.** Always report null NMSE. If the DPSAE masks merely suffer a different NMSE shift under swaps, compare only on common NMSE support. A raw loss increase after randomization is not informative because any useful SAE should beat a randomized assignment.

### 3.3 Cheap allocation summaries

The counterfactual should be accompanied by, but never replaced by, these summaries:

- row L0 Gini and quantiles;
- feature firing entropy (H(p)=-\sum_jp_j\log p_j) and effective feature count (\exp H(p));
- Spearman correlations of row L0 and code energy with source ridge leverage (K(G_g)_{ii});
- support stability between threshold and forced-BatchTopK encoding;
- selection margins between the last active and first inactive preactivation.

No monotonic theorem links any one of these summaries to decoder loss. They are mechanism fingerprints, not success metrics.

## 4. Hypothesis B: nonorthogonal decoder atoms

### 4.1 Exact row-Gram decomposition

Augment code and dictionary matrices as

\[
\widetilde C=[C,\mathbf1],
\qquad
\widetilde D=\begin{bmatrix}D\\b^\top\end{bmatrix}.
\]

Then

\[
H=ZZ^\top
=\widetilde C(\widetilde D\widetilde D^\top)\widetilde C^\top.
\]

This shows exactly where atom correlations enter: they are the off-diagonal entries of the augmented decoder Gram. Global mutual coherence is a poor statistic because unused atom pairs do not affect (H). Useful observational summaries are code-weighted:

\[
\mathrm{coh}_{\rm mass}
=\frac{\sum_{i\ne j}W_{ij}(d_i^\top d_j)^2}
{\sum_{i\ne j}W_{ij}},
\qquad
W_{ij}=\sum_t C_{ti}C_{tj},
\]

and the signed residual-correlation alignment

\[
\mathrm{align}_{\perp\rm cross}
=\frac{\langle G-H_{\perp\mathrm{orth}},H-H_{\perp\mathrm{orth}}\rangle_F}
{\|G-H_{\perp\mathrm{orth}}\|_F
 \|H-H_{\perp\mathrm{orth}}\|_F}.
\]

The sign matters: high coherence can help or hurt depending on whether its cross-terms correct the source Gram residual.

### 4.2 A valid orthogonal-residual counterfactual that preserves bias alignment

Simply replacing (DD^\top) by (I_p) also removes atom-bias inner products. The following PSD counterfactual isolates atom correlations perpendicular to the decoder bias while preserving every atom norm and every (d_j^\top b).

Let β=||b|| and (v=Db\in\mathbb R^p). For β>0 set

\[
r_j^2=1-v_j^2/\beta^2\ge0.
\]

Construct hypothetical unit atoms

\[
d_j^\perp=(v_j/\beta)e_0+r_je_j,
\qquad
b^\perp=\beta e_0,
\]

in (p+1) dimensions. Their Gram is PSD by construction, preserves (\langle d_j,b\rangle=v_j), and makes residual components (r_je_j) mutually orthogonal. The resulting row Gram is computable without a (p\times p) matrix:

\[
H_{\perp\mathrm{orth}}
=C\operatorname{diag}(r^2)C^\top
+\left(Cv/\beta+\beta\mathbf1\right)
\left(Cv/\beta+\beta\mathbf1\right)^\top.
\]

If β=0, use (H_{\perp\mathrm{orth}}=CC^\top). This is a valid Gram matrix, so its ridge hat is well defined even though the hypothetical representation has more than 768 coordinates.

Let (N^{\perp\mathrm{orth}}=\|K(G)-K(H_{\perp\mathrm{orth}})\|_F^2) and define (\mathcal A_{\perp\mathrm{orth}}) as in Section 2. The exact differential contribution of learned residual atom correlations to the paired advantage is

\[
\mathcal C_{\rm nonorth}
=\mathcal A_{\rm full}-\mathcal A_{\perp\mathrm{orth}}
=B_{\rm DPSAE}-B_{\rm MSE},
\]

where

\[
B_m
=\frac{\sum_g(N^{\perp\mathrm{orth}}_{mg}-N^{\rm full}_{mg})}
{\sum_gD_g}
\]

is the benefit of restoring model (m)'s actual atom correlations.

**Nonorthogonality signature.** If (B_{\rm DPSAE}>B_{\rm MSE}), (\mathcal C_{\rm nonorth}>0), and (\mathcal A_{\perp\mathrm{orth}}) collapses toward zero across seeds, learned atom correlations are necessary for the paired checkpoint advantage under this exact Gram counterfactual.

**Falsifier.** If (\mathcal A_{\perp\mathrm{orth}}) retains the full advantage and (B_{\rm DPSAE}\le B_{\rm MSE}), differential nonorthogonality is not the primary explanation. High raw mutual coherence cannot rescue the claim.

Run the same counterfactual on the support-only codes from Section 3. A support advantage that survives residual orthogonalization isolates active incidence plus diagonal atom energy; an advantage that appears only with the full decoder Gram is a support-by-nonorthogonality interaction.

## 5. Hypothesis C: reconstruction-dependent row-Gram geometry

### 5.1 Exact endpoint identity and source-local linearization

Let (E=H-G), (R=(G+\tau I)^{-1}), and (R_H=(H+\tau I)^{-1}). The exact difference is

\[
\delta K:=K(H)-K(G)
=\tau RER_H.
\]

The Fréchet derivative at the source Gram is

\[
J_G(E)=\tau RER.
\]

Using (R_H=R-RER_H),

\[
\delta K=J_G(E)-\tau RERER_H.
\]

Thus the nonlinear remainder is explicit rather than an informal appeal to "dynamic geometry":

\[
Q:=\delta K-J_G(E)=-\tau RERER_H.
\]

For each group report

\[
N_{\rm exact}=\|\delta K\|_F^2,
\quad
N_{\rm lin}=\|J_G(E)\|_F^2,
\quad
\rho_Q=\frac{\|Q\|_F}{\|\delta K\|_F},
\quad
\cos_Q=\frac{\langle J_G(E),\delta K\rangle}
{\|J_G(E)\|_F\|\delta K\|_F}.
\]

The squared loss obeys the exact decomposition

\[
N_{\rm exact}
=N_{\rm lin}+2\langle J_G(E),Q\rangle+\|Q\|_F^2.
\]

Define (\mathcal A_{\rm lin}) from (N_{\rm lin}) and (\mathcal C_{\rm endpoint}=\mathcal A_{\rm full}-\mathcal A_{\rm lin}).

**Higher-order endpoint signature.** If (\mathcal A_{\rm full}>0) but (\mathcal A_{\rm lin}\le0), or if most of the paired advantage lies in the cross/remainder terms with large ρ_Q, reconstruction-dependent endpoint curvature is necessary.

**Falsifier.** If ρ_Q is small, cos_Q is near one, and (\mathcal A_{\rm lin}) reproduces the full paired advantage in every seed, higher-order dependence of (K(H)) is not needed. This would not vindicate the tested static spectral baseline: that baseline is an omission-cost surrogate, not (J_G).

### 5.2 Eigenvalue-change versus source-eigenspace-mixing terms

Let

\[
G=U\operatorname{diag}(\gamma_1,\ldots,\gamma_n)U^\top,
\qquad
\bar E=U^\top EU.
\]

Because (R) is diagonal in this basis,

\[
[U^\top J_G(E)U]_{ij}
=\frac{\tau\bar E_{ij}}
{(\gamma_i+\tau)(\gamma_j+\tau)}.
\]

The diagonal and off-diagonal parts are Frobenius-orthogonal, giving

\[
N_{\rm lin}^{\rm diag}
=\sum_i\frac{\tau^2\bar E_{ii}^2}{(\gamma_i+\tau)^4},
\]

\[
N_{\rm lin}^{\rm off}
=\sum_{i\ne j}
\frac{\tau^2\bar E_{ij}^2}
{(\gamma_i+\tau)^2(\gamma_j+\tau)^2},
\qquad
N_{\rm lin}=N_{\rm lin}^{\rm diag}+N_{\rm lin}^{\rm off}.
\]

For a simple source spectrum, the diagonal term is the first-order ridge-gain/eigenvalue change and the off-diagonal term is source-eigenspace mixing. With exact ties, rotations within a tied eigenspace are unidentifiable, so use spectral projectors or label the quantity only as a source-eigenbasis off-diagonal perturbation.

Also report the normalized commutator

\[
\kappa(G,H)=\frac{\|GH-HG\|_F}{\|G\|_F\|H\|_F},
\]

and normalized raw Gram error (\|H-G\|_F^2/\|G\|_F^2).

**Spectral-gain signature.** The paired advantage lies predominantly in (N_{\rm lin}^{\rm diag}), with small κ and off-diagonal contribution. DPSAE is preserving ridge gains along source modes; the failure of the global static control then reflects its wrong extension or calibration, not necessarily sparse eigenspace mixing.

**Eigenspace-mixing signature.** The paired advantage lies predominantly in (N_{\rm lin}^{\rm off}), and DPSAE reduces κ relative to MSE. This identifies preservation of sample-space eigenspaces at first order.

**Ridge-weighting signature.** DPSAE has equal or larger raw Gram error but smaller (N_{\rm lin}) or exact loss. It is changing the Gram in directions attenuated by the two source resolvents, rather than simply making (ZZ^\top) globally closer to (XX^\top).

## 6. Minimal staged diagnostic and stop rules

### Stage 1: row-Gram analysis from existing reconstruction caches

This is the cheapest and highest-information diagnostic. It needs only the cached original activations and existing exact reconstructions, not model loading.

For each paired seed, compute raw Gram error, exact loss, (N_{\rm lin}), remainder norm/cosine, diagonal/off-diagonal tangent terms, and the commutator. Bootstrap paired group contrasts.

- If (\mathcal A_{\rm lin}\) reproduces (\mathcal A_{\rm full}), stop calling the effect "nonlinear hat curvature"; proceed to code/dictionary diagnostics only to explain the Gram perturbation.
- If (\mathcal A_{\rm lin}\) fails while the exact advantage is robust, endpoint nonlinearity is established as necessary and should be the first mechanism reported.
- If DPSAE merely has lower raw Gram error, report direct row-Gram preservation before invoking specialized sparse effects.

### Stage 2: orthogonal-residual counterfactual

Load one MSE/DPSAE model pair at a time and stream the 16,384 exact tokens in groups. Compute codes, (H_{\perp\mathrm{orth}}), code-weighted coherence, signed cross alignment, and (\mathcal A_{\perp\mathrm{orth}}). Do not form (DD^\top), which would be a 16,384-squared matrix.

- If the paired advantage survives, residual atom nonorthogonality is unnecessary for the checkpoint result and Stage 3 can focus on supports/amplitudes.
- If it collapses consistently, run the support-only × orthogonalized factorial before claiming a pure nonorthogonality mechanism.

### Stage 3: support-only and allocation-null curves

Estimate μ on the untouched natural-selection cache, freeze it, then compute support-only curves and 20 degree-preserving mask nulls on test. Compare only on common NMSE support and run both threshold and fixed-count encodings.

- (\mathcal A_{\rm sup}>0), (\mathcal C_{\rm alloc}>0), and survival under orthogonalization is the cleanest active-set signature.
- Failure of support-only with success of full codes points to learned amplitudes or interactions, not support alone.
- A threshold-only signature that vanishes at fixed count should be labeled an inference sparsity-level effect.

This staged design is evaluation-only. Processing a 128-row group produces a (128\times16{,}384) code matrix, about 8 MiB in FP32; one model, its roughly 50 MiB decoder, and (128\times128) Gram/resolvent matrices fit far below the prior 6 GiB guard. Stream groups and save only per-group summaries. Six models and no LM should remain under 2 GiB peak GPU memory and add negligible storage.

## 7. Outcome matrix

| Observed frozen-checkpoint pattern | Supported interpretation | Claim that fails |
| --- | --- | --- |
| Full advantage reproduced by source tangent; small remainder | Source-local row-Gram perturbation is enough | Higher-order reconstruction-dependent curvature explains the gain |
| Exact advantage positive, tangent advantage absent/reversed | Endpoint nonlinearity is necessary | A fixed source-local metric fully explains the gain |
| Off-diagonal tangent contrast dominates, DPSAE lowers commutator | Better source-eigenspace preservation | Pure eigenvalue/ridge-gain weighting is enough |
| Diagonal tangent contrast dominates, commutator unchanged | Better ridge-gain preservation along source modes | Sparse eigenspace rotation is required |
| Advantage vanishes under (H_{\perp\mathrm{orth}}) | Learned atom correlations are necessary in the Gram counterfactual | Active incidence plus diagonal atom energy is enough |
| Advantage survives (H_{\perp\mathrm{orth}}) | Differential atom nonorthogonality is unnecessary | Coherence explains the result |
| Support-only advantage survives at matched NMSE and dies under degree-preserving nulls | Specific token-to-atom support assignment carries the effect | Marginal firing rates alone explain it |
| Support-only fails but full codes succeed | Amplitudes or support×dictionary interactions are required | Binary active sets alone explain it |
| Signature appears only with threshold encoding | Learned inference L0/threshold matters | Training-budget active allocation is established |

## 8. Claims the current artifacts already falsify

- **"The valid target shows DPSAE uniquely preserves a functional direction."** False: paired full-reconstruction effects change sign, sparse effects are tiny, and spectral models improve more consistently.
- **"Equal IOI effect reveals consistent DPSAE causal specificity."** False: equal-effect collateral KL favors DPSAE in seeds 0 and 1 and disfavors it in seed 2.
- **"The static control's failure proves nonorthogonality."** False: the control is not the source-local Fréchet metric, and orthogonal dictionaries can exhibit active-allocation effects.
- **"BatchTopK (k=32) is the rank in the spectral theorem."** False: different rows can activate different atoms, so even one-sparse rows can produce rank up to ⁠min(n,d).
- **"A frozen-checkpoint counterfactual proves training causality."** False: it establishes algebraic necessity or sufficiency for the final parameters. Distinguishing the training gradient requires retraining or intervention during optimization.

The most economical next result is Stage 1. It can decide whether "reconstruction-dependent" means a genuinely nonlinear endpoint effect or merely a source-local row-Gram perturbation before any more expensive sparse-code attribution is attempted.

## 9. Implementation status

`experiments/exp04b_mechanism_attribution.py` now implements the first two stages as separate `tangent` and `nonorth` subcommands. Both stages stream one 128-row group at a time, save the raw groupwise numerators needed to recompute every paired contrast, bootstrap matched group indices, record SHA-256 hashes for every input and the evaluator, and emit explicit pass/fail mechanism flags rather than relying on an informal reading of averages. The tangent stage uses cached activations and exact reconstructions only. The nonorthogonality stage additionally loads one frozen SAE at a time and computes the PSD bias-preserving residual-orthogonal counterfactual without materializing the full decoder Gram.

The command deliberately does not expose a support/allocation stage yet. The locally available repository artifact index contains source metadata and figures but not the immutable natural-selection codes needed to freeze feature amplitudes without re-encoding. Re-encoding with a mutable implementation would weaken provenance, so Stage 3 remains specified above rather than being presented as a completed intervention.

`tests/test_exp04b_mechanism_attribution.py` verifies the exact resolvent/tangent/remainder identities, the diagonal/off-diagonal split, PSD validity and invariants of the nonorthogonality counterfactual, positive and negative decision flags on fabricated paired data, groupwise streaming, and input hashing in synthetic end-to-end runs. On 2026-07-14, the targeted file passed 8 tests and the full repository passed 106 tests. Python compilation and `git diff --check` also passed; Ruff was unavailable in the local environment. The real Stage 1 result was not run locally because the bulky activation and reconstruction caches are absent from this checkout.
