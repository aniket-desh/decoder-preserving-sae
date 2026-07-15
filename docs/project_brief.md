# Project brief

## Working title

Decoder-Preserving Sparse Autoencoders

## Scoped claim

Sparse autoencoders usually spend capacity minimizing Euclidean activation error. This project asks whether adding a loss derived from disagreement between optimal regularized linear readouts preserves held-out decodable information more efficiently at a fixed sparsity budget, and whether that preservation concentrates real language-model computations into cleaner, more causally useful sparse features.

The paper does not initially claim to learn an information-geometric manifold, maximize mutual information, or preserve the frozen model's exact downstream circuit. Those are distinct goals. The Goodfire activation-manifold paper remains in the literature collection because it may clarify limitations or motivate later work, but it is outside the first implementation and experiment plan.

## Core objective

For an activation batch `H` and SAE reconstruction `H_hat`, define ridge hat matrices

```text
K(H) = H (H^T H + rho I)^-1 H^T.
```

For a decoding-task distribution with second moment `Sigma_y`, use

```text
D_dec^2(H, H_hat) = tr[(K(H) - K(H_hat)) Sigma_y (K(H) - K(H_hat))^T].
```

The initial hybrid loss is

```text
L = alpha * MSE(H, H_hat) + gamma * D_dec^2(H, H_hat) + L_sparse.
```

The hybrid is the default because decoder distance is invariant to coordinate changes that can preserve refittable linear probes while breaking compatibility with the frozen downstream network.

## Primary hypotheses

- At matched architecture, data, compute, and sparsity, a nonzero decoder-loss weight reduces held-out decoder distance relative to the MSE baseline.
- Improvements should transfer to held-out task families rather than only the minibatch task prior used during training.
- On a known causal NLP computation, an unlabeled isotropic DPSAE should localize the computation in fewer or more selective features and support a more targeted intervention than an MSE SAE at a matched operating point.
- Extreme relative decoder-advantage modes should generate semantic hypotheses that can be frozen and tested on unseen documents; failure to transfer is evidence that the sample-space geometry is not itself a reliable discovery method.
- A hybrid objective can improve decodable-information retention without an unacceptable increase in output KL, activation MSE, or collateral intervention damage.
- A decoder-only objective will reveal the gap between information that remains linearly accessible after refitting and information usable by the frozen downstream model.

## Falsification conditions

The central empirical case is weak if decoder loss improves only its own training metric, disappears under held-out prompts or documents, requires much more compute, or simply trades activation MSE for a denser probe without cleaner sparse features or a causal consequence. Top-activating examples and automated feature descriptions are hypothesis-generation tools, not sufficient evidence. Negative results are still informative if they isolate estimator instability, task-prior mismatch, sample-specific advantage modes, or the refitted-decoder/frozen-network gap.

## Decisions intentionally deferred

- Exact GPT-2 small activation site after a bounded IOI pilot.
- Dataset and token sampling policy.
- Sparsity mechanisms beyond the closure ablation. The valid comparison fixes
  $k=32$ and tests BatchTopK against tokenwise TopK. A decoder-blind,
  full-horizon JumpReLU grid trained both objectives for 25M tokens at shared
  sparsity-loss weights 2, 4, 8, and 16. Every setting missed the frozen
  $L_0\in[30.4,33.6]$ band and the dead-feature gate, so no operating point was
  selected and all decoder, NMSE, and language-model outcomes remain sealed.
  Broader architecture sweeps remain deferred.
- Exact task priors beyond isotropic and the controlled synthetic prior.
- Workshop versus main-track positioning, which should depend on experimental depth.
