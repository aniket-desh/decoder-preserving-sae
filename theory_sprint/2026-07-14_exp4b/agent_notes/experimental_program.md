# Leakage-safe experimental program for the remaining empirical gaps

## Recommendation

Launch the matched-NMSE frontier first, because it tests the largest remaining qualification on the established 24% representation result with the least new code. In parallel, prepare the evaluation-only decoder-advantage discovery cache, but do not open its final contexts until the hypothesis log is frozen. Run the sparsity-mechanism interaction next, then the extra GPT-2 layer, and leave the non-GPT-2 replication last because it is the only stage that needs a new model adapter and tokenizer-specific corpus.

The full program should be hierarchical. A short screen earns a confirmation fleet only when it passes a prewritten gate; otherwise the negative screen is the result. On one RTX 6000, use one GPU process at a time. The current vectorized fleet already shares the language-model forward pass across SAEs, so two independent GPU jobs would duplicate the frozen LM, compete for memory, and usually finish later.

## Current state and reusable evidence

I inspected the current checkout at `b5544fc28afb77c727ec416df6a1b41587e82fe6`; the only reported dirt is the untracked theory-sprint directory. The active source config uses GPT-2 small, one-based block 8 `resid_post`, dimension 768, a 16,384-latent untied nonnegative BatchTopK SAE, `k=32`, 2,048 tokens per step, groups of 128, 16 normalized Gaussian probes, and decoder weight `gamma=0.25`. The source corpus partitions are calibration `[0M,10M)`, screen `[10M,50M)`, confirmation `[50M,120M)`, robustness `[120M,170M)`, and old validation `[170M,180M)`. Experiment 4b adds immutable selection `[180M,185M)` and final `[185M,190M)` token ranges.

The logs give useful hardware calibration even though the current pod was not queried live:

| Existing fleet | Models | Tokens | Observed wall time |
| --- | ---: | ---: | ---: |
| Source gamma screen, `k=32` | 6 | 25.0M | 35.2 min |
| Source confirmation, `k=32` | 9 | 100.0M | 4.07 h |
| Source robustness, `k=16` | 6 | 50.0M | 40.1 min |
| Source robustness, `k=64` | 6 | 50.0M | 75.4 min |

The six-model optimizer checkpoint is about 1.81 GB, or roughly 0.30 GB per active model; an exported model is about 0.10 GB. The existing monitor stops at 43,000 MiB, and the repository includes a full-shape nine-model memory probe. These observations support fleets of six to nine GPT-2-sized SAEs, but every new fleet still needs its own one-step memory probe because tokenwise TopK and JumpReLU change live tensors.

The following remote artifacts should be hashed and reused before any new training:

- `artifacts/exp04_ioi_mechanism/fineweb_gpt2_tokens.bin`, `calibration.pt`, `screen/models.pt`, and `confirmation/models.pt`;
- `artifacts/exp04b_confirmatory/natural_selection.pt`, `natural_test.pt`, `static_calibration.pt`, `baseline_screen/{models.pt,selection.json}`, and `baseline_confirm/{models.pt,test.json}`;
- paired exact reconstructions for the block-8 MSE and DPSAE seed models under `artifacts/exp04b_confirmatory/exact_reconstructions/source/`;
- the machine-readable natural evaluation and exact-audit JSONs, rather than values copied out of plots.

The local tree contains only the small result bundle. The audit records the full caches and model payloads as remote or private-Hub artifacts, so the first operational step is a manifest of paths, sizes, and SHA-256 hashes. A missing endpoint should be refetched from the existing private backup, not retrained silently.

## Shared split and storage policy

The existing `[180M,185M)` selection range can continue to select new hyperparameters because it has already been designated for that purpose. Do not use `[185M,190M)` to make another training decision. For the new trained-model tests, create one immutable 10M-token GPT-2 tail at offset 190M and reserve `[190M,195M)` as the once-only confirmation set. Reserve `[195M,200M)` for the final decoder-advantage hypotheses; aggregate reconstruction metrics may be computed there, but no contexts or hypothesis labels may be inspected before the discovery log is frozen.

This tail is only about 19 MiB as `uint16`. One 65,536-token activation cache at width 768 is about 96 MiB in FP16, and one 16,384-token exact reconstruction is about 24 MiB per model. Cache only the fixed 65,536-token evaluation samples. Caching 25M training activations would consume about 35.8 GiB per layer and defeats the existing streaming design.

Use one rolling optimizer checkpoint per active fleet and atomic writes. Once `models.pt`, the training log, resolved config, hashes, and remote backup have all been verified, the optimizer checkpoint may be retired; dominated screen models do not need permanent optimizer states. Require at least 12 GiB free before a GPT-2 fleet and 25 GiB before the non-GPT-2 stage. Stop at 80% disk use, 80% of physical GPU memory, or 40 GiB reserved memory, whichever is lower, until a fresh full-shape probe establishes a safer device-specific limit.

## 1. Matched-NMSE gamma frontier

The existing screen already contains `gamma in {0.125, 0.25, 0.5, 1.0}` and the Experiment 4b static screen contains `beta in {0.125, 0.25, 0.5, 1.0}`. Those are useful endpoints, but they do not resolve the frontier near MSE after 100M tokens. Metric interpolation between independently trained models is descriptive; it is not a trained SAE at the interpolated operating point.

### Screen

Train only the missing small-gamma models at `gamma in {0.03125, 0.0625, 0.09375, 0.125}` for 25M tokens, seed 0, using the exact `baseline_screen` data-order and probe-sequence seeds. Reuse the paired MSE and spectral-beta endpoints from the Experiment 4b baseline screen after verifying that model initialization, learning-rate schedule, training stream, and code path match exactly. If any of those checks fail, include a fresh MSE and spectral `beta in {0.25,0.5,1.0}` in one common eight-model fleet instead.

Evaluate NMSE on all 65,536 selection tokens and exact identity-target distortion on the same fixed 16,384 tokens for every candidate. Choose the smallest gamma satisfying both:

- selection NMSE no more than `1.01 * MSE NMSE`;
- exact decoder distortion at least 10% below paired MSE.

Break ties by lower NMSE, then smaller gamma. If no point qualifies, add at most one log-midpoint between the nearest bracketing gammas; do not open the new confirmation tail to tune the grid. Stop the frontier if no trained point improves exact distortion by 10% under a 2% NMSE cap at 25M tokens.

Intermediate 5M and 12.5M selection evaluations are stopping diagnostics only. A candidate can stop after both checkpoints if it is worse than paired MSE in both NMSE and exact distortion or exceeds `1.10 * MSE NMSE`; no candidate may be promoted from an intermediate result that does not survive 25M.

### Confirmation

If the screen passes, train the selected gamma for seeds 0, 1, and 2 for 100M tokens using the exact `baseline_confirm` replicate-1 data and probe streams. Reuse the existing MSE and selected spectral seed models only after deterministic stream and initialization parity is asserted in the result artifact. This reduces the new confirmation to three SAEs in one fleet; otherwise train the nine-model `MSE / selected DPSAE / selected spectral` fleet.

Score `[190M,195M)` once. The matched-NMSE claim passes only if all three DPSAE/MSE NMSE ratios are at most 1.01, all three exact decoder reductions are positive, the median reduction is at least 10%, and the paired group-bootstrap interval excludes zero in every seed. Compare spectral at the same observed NMSE tolerance; do not linearly extrapolate outside common frontier support.

Estimated new cost is 25–40 GPU minutes and about 1.2 GB peak checkpoint storage for the four-model screen. Reusing anchors makes confirmation roughly 1.5–2.5 GPU hours with a 0.9 GB checkpoint; a fresh nine-model confirmation is calibrated by the existing run at about four hours and 2.7 GB.

## 2. BatchTopK versus tokenwise TopK versus JumpReLU

The current code implements only BatchTopK. `TrainingFleet.train_batch` stacks every model's score tensor and applies one batch-global TopK rule, so changing only the model class is insufficient; the fleet must dispatch support selection per mechanism while keeping the matrix multiplications, targets, initialization, optimizer, token stream, and objective identical. A parity test must show the refactored BatchTopK path reproduces the existing path on one batch before training.

Use a `3 mechanisms x 2 objectives` factorial at width 16,384 and target average `L0=32`:

- BatchTopK with the existing global `batch_tokens * k` support budget;
- tokenwise TopK with exactly 32 active coordinates per token in both training and evaluation;
- JumpReLU with a threshold calibrated to selection-set average `L0` in `[30.4,33.6]`.

JumpReLU needs a stated threshold-gradient estimator and a sparsity-control loss; it cannot be represented honestly by the current evaluation threshold. Use a 1–2M-token integration run to tune only numerical stability and the L0 controller. Do not inspect decoder-distortion differences during this integration. Require finite gradients, no more than 5% L0 mismatch, dead-feature rate below 10%, and BatchTopK parity before the 25M screen.

For screening, reuse the existing BatchTopK seed-0 MSE and `gamma=0.25` models only if the refactored path and source-screen random streams match exactly; otherwise train all six models together. The four missing tokenwise/JumpReLU models should take about 25–45 GPU minutes at 25M tokens and about 1.2 GB of checkpoint space. Evaluate each within-mechanism DPSAE-minus-MSE contrast on the same exact selection groups. The interaction statistic is the difference between those paired percentage reductions, with a group bootstrap using identical resamples.

Treat the effect as objective-level when every mechanism improves by at least 10% and the largest interaction is below five percentage points. Treat a mechanism interaction as worth confirmation only when one contrast differs from BatchTopK by at least ten percentage points and its bootstrap interval excludes zero. Also stop any mechanism whose MSE baseline is itself invalid: selection NMSE more than 15% above BatchTopK MSE, inference L0 outside the 5% band, or more than 10% dead latents.

Confirm only the architecture with a real interaction, MSE and DPSAE, seeds 0–2, for 100M tokens. Reuse the existing BatchTopK confirmation endpoints under exact stream parity. One new six-model architecture fleet should take about 2–3.5 GPU hours and create a 1.8 GB checkpoint. If neither alternative interacts, do not spend seeds on them; report the three-way seed-0 screen as evidence that the decoder objective is the common mechanism.

## 3. One extra GPT-2 layer and one non-GPT-2 model

### GPT-2 block 4

Pre-register one-based GPT-2 block 4, rather than choosing a favorable layer after a sweep. It gives an early/middle site at the same width and model, while block 8 remains the established later site. Reuse the GPT-2 token memmap, batch starts, tokenizer, SAE width, `k=32`, selected gamma, token budgets, and evaluation token IDs. Recompute activation normalization and ridge calibration at block 4; neither is transferable across layers.

Run MSE/DPSAE seed 0 for 25M tokens. Advance only if exact `[180M,185M)` distortion improves by at least 10%, NMSE remains within 1.10x, and sampled and exact estimates have the same ordering. Confirmation is six models, seeds 0–2, 100M tokens, scored once on `[190M,195M)`. Require the improvement sign in all three seeds and median reduction of at least 10%; otherwise the paper should say the effect is site-dependent. Expect 20–35 minutes for the screen, 2–3 hours for confirmation, about 190 MiB for selection/confirmation activation caches, 1.8 GB for the rolling checkpoint, and 0.6 GB for exported models.

### Pythia-160M-deduped

Use `EleutherAI/pythia-160m-deduped` as the smallest controlled non-GPT-2 replication. It keeps a 12-layer, 768-dimensional residual stream, so the 16,384-latent dictionary, block 8, `k=32`, and all SAE hyperparameters remain unchanged while the transformer implementation, tokenizer, weights, and pretraining data differ. This is a model-family replication, not a clean causal isolation of architecture.

The current `GPT2ActivationModel` assumes `model.transformer.h` and `config.n_embd`. The replication runner needs a generic hidden-state-only adapter using `output_hidden_states` and `config.hidden_size`; no intervention hook is needed for the exact representation result. Pin the model, tokenizer, Transformers version, and dataset revisions. Build a separate `uint16` token memmap because the tokenizer differs, using the same deterministic FineWeb document stream and recording document hashes where possible. A 120M-token Pythia cache is about 229 MiB and is sufficient for 10M calibration, 25M screening, a disjoint confirmation sampling range, and immutable evaluation tails.

Run the same seed-0 25M gate as block 4, then confirm MSE/DPSAE seeds 0–2 at 100M only if it passes. Score exact identity targets; do not add IOI or feature claims to this replication. Budget 30–60 GPU minutes for screening and 3–5 hours for confirmation, plus 1–3 CPU/network hours to prepare the model and token cache. The six-model checkpoint and export sizes remain approximately 1.8 GB and 0.6 GB because the SAE dimension is unchanged; reserve another 1–2 GB for the pinned language-model cache.

Gemma-2-2B is a stronger later replication, but it changes residual width and forces a choice between fixed dictionary size and fixed overcompleteness. That confound and its larger compute make it a poor first non-GPT test. Promote it only if Pythia passes and reviewers require a modern gated-attention family.

## 4. Decoder-advantage mode discovery with frozen hypotheses

This stage should reuse the block-8 MSE/DPSAE confirmation pairs; no SAE retraining is justified. The existing exact reconstructions cover the old final split, while discovery requires equivalent reconstructions for `natural_selection.pt`. Generate only 16,384-token FP16 reconstructions for MSE and DPSAE seeds 0–2, one SAE at a time. This adds about 144 MiB and should take minutes on GPU. Do not cache dense 16,384-feature codes; store only selected feature IDs and sparse traces for inspected groups.

Partition the existing selection cache by recorded absolute starts before inspection:

1. `[180M,182.5M)` is open discovery.
2. `[182.5M,185M)` is internal recurrence and hypothesis freezing.
3. `[185M,190M)` selects a feature count and checks that the frozen rule is measurable, but it cannot alter the semantic wording.
4. `[195M,200M)` is the once-only final hypothesis test; its contexts remain hidden until the registry is immutable.

On 16 preselected document-balanced discovery groups, compute the top two and bottom two eigenmodes of the exact 128-by-128 relative-advantage operator for every paired seed. This caps the search at 192 mode instances. Log every instance, its eigenvalue, extreme token indices, row-shuffle control, random-direction percentile, proposed label, and disposition. Promote at most three semantic hypotheses, and only when the same deterministic interpretation recurs in at least two seed pairs and four independent groups with the same advantage sign. A hypothesis seen in a row-shuffle control or explained entirely by token identity, position, document length, or activation norm is rejected rather than repaired.

Freeze each promoted hypothesis as a versioned target constructor before final evaluation: an exact parser/regex/metadata rule or a blinded annotation rubric, the exclusion rules, feature ranking procedure, maximum feature count, and predicted sign of `A(y)`. Within each eligible group, center and unit-normalize the target and require both classes to have enough support; this prevents class prevalence from changing the scale. Rank features and choose feature count only on the earlier splits.

The final test reports every frozen hypothesis, including failures. A main-paper discovery must have positive pooled paired advantage with a group-bootstrap interval excluding zero, the predicted sign in all three seed pairs, recurrence across final documents, and cleaner DPSAE localization under a frozen feature-count metric. A practical localization gate is reaching 80% of dense-target held-out performance with no more than 75% of the MSE feature count, or a preregistered selectivity gain at the same count. Only after that gate passes should a phenomenon-specific causal edit or counterfactual dataset edit run. If no frozen hypothesis passes, stop and report that extreme sample-space modes did not transfer semantically.

The numerical stage is small: less than 15 GPU minutes for sequential reconstructions, under 4 GB GPU memory, under 8 GB CPU memory when groups are streamed, and roughly 0.2 GB of new cached tensors. CPU eigensolves and context indexing can run alongside a GPT-2 training fleet, provided they are read-only and do not contend with checkpoint backup. Human or blinded annotation time, rather than compute, is the dominant cost.

## Safe concurrency and launch order

Before every stage, run a one-step full-shape memory probe with the exact candidate count and mechanism. Require finite losses and peak reserved memory below the conservative cap. Use named `tmux` sessions, one log per stage, the existing hardware watcher, and a disk watcher that stops before 80% occupancy.

The safe concurrency pattern is one GPU training fleet plus one bounded CPU task: either decoder-mode eigensolves/context indexing or tokenizer streaming. Do not run two GPU fleets, do not reconstruct modes while a training fleet owns the GPU, and do not run a large Hub backup at the same time as token-cache construction. Pythia tokenization may overlap gamma or architecture training after disk space is reserved, but its model download and final rename should be serialized.

Priority sequence:

1. Manifest and hash the reusable remote artifacts; create the new 190M–200M tail; run the memory and disk preflights.
2. Run the small-gamma matched-NMSE screen, then its three-seed confirmation if it passes.
3. Generate block-8 discovery reconstructions and begin the capped decoder-mode search on CPU while the gamma confirmation trains; freeze the registry before touching `[195M,200M)`.
4. Run the 1–2M architecture integration and 25M factorial screen; confirm only a real interaction.
5. Run GPT-2 block 4 seed 0, then its three-seed confirmation only if the sign and gate survive.
6. Prepare and screen Pythia-160M; launch its confirmation last and only after all infrastructure and screen gates pass.

If every gate passes, the incremental GPU budget is roughly 10–15 hours; the screen-only program is about 2–3 hours. With rolling checkpoints and prompt retirement of optimizer state after verified export, incremental peak storage should remain below 8 GB, excluding already existing artifacts and language-model caches.

## Implemented TODO 5 numerical infrastructure

`experiments/exp05_decoder_advantage_discovery.py` now implements the evaluation-only portion of the decoder-advantage program without opening either semantic hypotheses or final contexts. The frozen default protocol selects exactly 16,384 unique discovery tokens from `[180M,182.5M)`, constructs document-balanced groups of 128 with fixed seeds, preselects exactly 16 groups independently of model results, and searches the top two and bottom two relative-advantage eigentasks for each of the three paired block-8 seeds. The resulting machine-readable log must contain exactly 192 mode rows and 48 group-level controls. Every mode carries the full eigentask, eigenvalue, extreme absolute token offsets, a row-shuffled-reconstruction Rayleigh control, and its percentile against 64 fixed random directions; every group control also records the full random reference values and row-shuffle operator extrema.

The preparation stage loads the aggregate source fleet on CPU but moves and reconstructs only one SAE at a time. Each per-model artifact contains an FP16 reconstruction of exactly 16,384 tokens plus the source-model hash, model spec, and discovery-manifest digest. The search stage likewise loads only one 24 MiB reconstruction at a time: it reduces the MSE reconstruction to its 16 small ridge-disagreement matrices, releases it, then loads DPSAE and forms the paired observed and row-shuffled operators.

The semantic boundary is a separate `hypothesis_registry.json`. Search initializes every one of the 192 dispositions as `unreviewed` and creates no hypotheses. Freezing is refused until every mode is either rejected or bound consistently to one of at most three fully specified hypotheses, including its target constructor, exclusion rules, feature-ranking rule, maximum feature count, predicted sign, and evidence modes. Freeze binds the registry to the exact searched-mode file by SHA-256 and signs the registry's canonical payload. Any mutation of the search log or frozen registry invalidates final access.

The sealed `[195M,200M)` range is guarded before `torch.load`: a missing, open, mismatched, or tampered registry raises without deserializing the requested cache. The implementation exposes only a seal-verification command; it contains no command that decodes or inspects final contexts.

The intended remote sequence, after the existing artifacts are restored, is:

```sh
PYTHONPATH=src python3 -u experiments/exp05_decoder_advantage_discovery.py prepare
PYTHONPATH=src python3 -u experiments/exp05_decoder_advantage_discovery.py search
```

After a separate semantic review edits every registry disposition, `freeze-registry` validates and seals that file. `verify-seal` checks authorization without loading a final cache. No stage has been launched on the RunPod, no semantic hypothesis has been added, and the final range remains unopened.

## Implemented TODO 4 generality-screen infrastructure

`experiments/exp06_generality.py` now implements only the two preregistered seed-0 screens: one-based GPT-2 small block 4 and `EleutherAI/pythia-160m-deduped` block 8. Both use the repository's generic GPT-2/GPT-NeoX hidden-state adapter, a 16,384-latent BatchTopK dictionary at `k=32`, 25M or fewer whole-batch training tokens, and one paired MSE/DPSAE fleet at a caller-supplied positive frozen gamma. Each target gets an independent activation mean/scale and ridge calibration, so neither layer nor model statistics are reused.

The screen shard is capped at 50M tokens and partitioned into calibration `[0M,10M)`, training `[10M,40M)`, and held-out `[40M,50M)`. GPT-2 may reuse the existing verified GPT-2 FineWeb cache, but only its first 50M tokens are addressable by this runner. Pythia defaults to a new 50M-token `uint16` cache generated from the same bounded FineWeb source with the Pythia tokenizer. Exact tokenizer name, dataset source, metadata count, physical byte count, and SHA-256 are validated before model work, so GPT-2 token IDs cannot be passed into the Pythia screen.

Calibration records the resolved Hub commit, target architecture, block, width, config digest, and calibrated ridge. Training refuses a changed model identity, config, calibration hash, or paired model spec. It uses a rolling checkpoint every 5M tokens, restores both optimizer and token-sampler state, and trims log records beyond the restored checkpoint. The resolved config stores independent data/probe seeds, the repository revision and dirty state, hashes of the runner and reused source modules, and hashes of the token cache and metadata.

Evaluation caches exactly 16,384 held-out normalized activations, then loads and releases the MSE and DPSAE SAEs sequentially. It uses thresholded inference and exact identity-target ridge disagreement, preserves per-group numerator/denominator rows for a paired bootstrap, and reports the preregistered 10% decoder-reduction / 1.10x-NMSE screen gate. Cache, calibration, and model hashes are locked into partial results so interrupted sequential evaluation is resumable without mixing artifacts.

The runner stops at 80% disk use and uses the lower of 40 GiB or 80% of physical GPU memory as its default device-use cap. It checks resources before work and throughout training, writes large state atomically, and evaluates only one SAE on the GPU at a time. This remains screen infrastructure: it does not implement a three-seed confirmation fleet or make a paper claim.

No remote stage has been launched. After a frozen gamma is supplied and the artifacts are present, the intended one-process-at-a-time commands are:

```sh
PYTHONPATH=src python3 -u experiments/exp06_generality.py all \
  --target gpt2-block4 --gamma <frozen-gamma> --device cuda
PYTHONPATH=src python3 -u experiments/exp06_generality.py all \
  --target pythia-block8 --gamma <frozen-gamma> --device cuda
```
