# Experimental diagnostic: dense IOI ridge selection

## Outcome

No ridge sweep ran, so this diagnostic produced no ranking, selection, or test \(R^2\) values. The supplied RunPod did not return any output from the initial read-only SSH inspection, and that connection attempt was aborted after approximately 1,244 seconds. Per the instruction to stop rather than extend the experiment, I did not reconnect or launch computation.

This leaves the empirical-audit caveat unchanged: the existing dense result \(R^2=-3.0968\) is specific to the fixed ridge-0.01 protocol, and the available evidence still cannot distinguish a regularization failure from ranking-to-selection-to-test distribution shift or genuine target mismatch.

## Intended evaluation

The bounded plan was to load only `/workspace/decoder-preserving-sae/artifacts/exp04b_confirmatory/ioi_confirmatory_cache.pt`, use the normalized clean block-8 states under each of the frozen splits (`ranking`: 3,072 examples, `selection`: 1,024 examples, `test`: 2,048 examples), and fit the correct-minus-subject logit difference from the 768-dimensional original dense activation. Ridge would be fit on the ranking split, selected by selection-split \(R^2\) over a broad grid including near-OLS, and reported once on test together with target moments, singular-value conditioning, effective rank, and cross-split diagnostics.

The intended resource ceiling was CPU-only or a negligible GPU allocation, less than 2 GB of memory, no SAE loading, and no training. No part of that plan executed.

## Exact command attempted

```sh
ssh pux2kohgyqt0r9-644120a7@ssh.runpod.io -i ~/.ssh/id_ed25519 'cd /workspace/decoder-preserving-sae && pwd && git rev-parse HEAD && stat -c "%n %s bytes" artifacts/exp04b_confirmatory/ioi_confirmatory_cache.pt && nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader && df -h /workspace'
```

This command was metadata-only: it requested the checkout path and revision, cache-file size, GPU memory summary, and workspace disk usage. It did not load the cache, start Python, create a remote script, train an SAE, or modify any artifact. It returned no stdout or stderr before being aborted, so the remote revision, cache size, GPU usage, and disk usage were not observed.

## Resource and mutation record

- Remote diagnostic compute: none.
- Remote training or model inference: none.
- Remote files created, edited, or deleted: none.
- Local files edited: this note only.
- Wall time consumed by the stalled SSH attempt: approximately 1,244 seconds.
- Peak memory, GPU memory, and storage consumption: not measurable from the stalled connection; no cache-processing process was launched.

## Interpretation

There is no new evidence for or against dense linear recoverability under validation-selected ridge. In particular, it would be incorrect to attribute the existing negative test \(R^2\) to distribution shift, conditioning, or an unsuitable ridge value from this attempted diagnostic. The next valid step, if reopened, is the originally scoped cache-only CPU sweep; it should begin with a short connection and cache-schema check that has its own timeout before any numerical work.
