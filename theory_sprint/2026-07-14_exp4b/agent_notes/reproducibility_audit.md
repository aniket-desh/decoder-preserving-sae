# Experiment 4b and valid-target reproducibility audit

**Audit date:** 2026-07-14
**Local checkout:** `b5544fc28afb77c727ec416df6a1b41587e82fe6`; worktree dirty because the theory-sprint directory is untracked.
**Scope:** Experiment 4b natural-text/source and remote baseline/IOI artifacts described in `00_empirical_audit.md`, plus the new `exp04b_valid_target_equal_effect` result and evaluator.

## Newest result checked first

`artifacts/exp04b_valid_target_equal_effect/result.json` is complete and its local SHA-256 is `f5d1691d9995ae54a631f11f653a828bd3c1f5f4e7208ebf93eeae009e063616`. The dense target validity gate passes (`test R2=0.9996085`). DPSAE-versus-MSE reconstruction R2 differences are mixed across seeds (`-0.00664`, `+0.01133`, `-0.01013`); top-64 sparse-code differences are small and positive in seeds 0/1, with seed 2 crossing zero. Equal-effect collateral KL favors DPSAE in seeds 0/1 (`-0.00779`, `-0.00595`) and disfavors it in seed 2 (`+0.00404`). The result therefore repairs the invalid dense-target gate but does not establish consistent target preservation or frozen-network causal specificity across seeds.

The equal-effect confidence intervals are explicitly conditional on point-estimate interpolation brackets. Bootstrap support is incomplete in some comparisons, including only `62.55%` of DPSAE seed-2 draws. This is recorded correctly in the JSON and should remain visible in any claim.

## Artifact-by-artifact traceability

| Artifact or stage | What is traceable now | Exact gap | Status |
| --- | --- | --- | --- |
| Source-fleet natural evaluation | Local `natural_evaluation_source.json`, SHA-256 `ac074b...d6e92b`; embeds clean revision `38b4c3ba08ae9e29ba3f026c78bdc815183470c9`, model specs/seeds, fresh token starts, ridge, group size, evaluation probe seed/count. | Does not embed a resolved-config hash, training checkpoint/model hash, machine/environment manifest, or exact launch command. | Recomputable only if the remote checkpoints and activation inputs remain available. |
| Source exact audit | Local `natural_exact_audit_source.json`, SHA-256 `c09069...85cac92`; records all settings and 168 rows. Exact rows are also linked through the source evaluation result. | Standalone file does not carry repository/environment provenance. `n=64` uses 128 of 256 groups (8,192 tokens) while `n=128/256` use 16,384, so the group-size sweep is not input-matched. | Numerically auditable; one axis needs a matched-input rerun. |
| Baseline confirmation natural evaluation | Raw remote JSON hash recorded as `26c468...81dd59`; baseline exact audit hash `1b76f6...c7f07a`; baseline-selection hash `0f36c...e7c783`. | These files are absent locally. Their durability depends on the current RunPod. The audit did not establish a local/versioned manifest tying each checkpoint to its training log and environment. | Cannot reproduce locally from the current bundle. |
| Final IOI confirmation | Remote `ioi_confirmatory.json` hash `85f9c0...10760c6`; raw values and current implementation were audited. | The artifact contains no Git revision. Successful parsing implies `d985476...` or later, but that is only a lower bound, not the exact analyzed code. | Fails the repository's code-revision traceability requirement. |
| Resolved Experiment 4b config | Remote `resolved_config.json` hash `d4ea61...64455c`; source config is local and hashes to `e984b5...b9b5c`. | Resolved config is absent locally; base config composition therefore relies on a remote file. | Hash-known, not durable locally. |
| Valid-target result | Local result hash above; records start/finish time, protocol, ridge grid, feature counts, bootstrap seed `2027071417`, model seeds/specs, direction hash, every remote input path/hash, PyTorch `2.8.0+cu128`, device, storage guard, memory cap, peak allocation, and selected metrics. | Run says repository dirty at `b5544fc`; the evaluator is not in that commit. `device=cuda:0` is not a GPU model. No CUDA/driver, OS/container, hostname, dependency-lock hash, deterministic flags, or exact argv is recorded. | Strong hash-level provenance, but not clean-revision reproducibility. |
| Valid-target sidecar | `provenance.json`, SHA-256 `c70b31...0309c1`; records evaluator hash `5c0c6e...c8c79f`, remote-to-local result verification, tmux name, guards, revision, and the claim that `?? uv.lock` was observed before launch. | The evaluator itself is absent from commit `b5544fc`, so `uv.lock` cannot be the complete explanation of the code used unless the evaluator was copied after that status snapshot or excluded remotely. No saved porcelain-status/diff artifact resolves this. | Useful sidecar, but the dirty-code ambiguity remains. |
| Valid-target inputs | Result hashes the cache (`3af618...150a`), model payload (`386d41...699b3`), selection JSON (`93a810...d1f7d`), test JSON (`a37080...6135`), and config (`e984b5...b9b5c`). | All four experiment inputs other than the config are absent locally and referenced only by ephemeral `/workspace/...` paths. A hash proves identity only while a copy still exists. | Evaluation can be reproduced only on the current remote state. |

## Seed and split audit

The valid-target protocol is adequately explicit about statistical selection: the target direction uses the ranking split only; ridge uses the selection split; test is held out; feature ranking uses ranking-split correlations; the dense gate threshold is frozen at `0.8`; and the output records the bootstrap seed and all model initialization seeds. The input hashes transitively fix the cached rows and existing zero-ablation curves.

The remaining seed gap is environmental rather than conceptual: the artifact does not record CUDA deterministic settings or RNG state versions, and it does not distinguish substream seeds for every bootstrap comparison in the top-level protocol. Those offsets are recoverable from the evaluator hash, but only while that uncommitted file is preserved.

For the base Experiment 4b natural fleet, the audit records initialization, data-order, training-probe, evaluation-probe, selection, test/bootstrap, and IOI seeds. The final IOI JSON itself does not independently record a code revision, and its provenance should not be inferred solely from those seeds.

## Machine and environment audit

The valid-target run respected the resource limits: 8% per-process GPU fraction, 20 GiB minimum free-storage guard, 1.438 GiB peak allocated GPU memory, and 58.96 seconds wall time. No OOM/storage concern appears.

For reproducibility, `torch_version` and `device=cuda:0` are insufficient. The paper artifact should additionally record:

- GPU product and total VRAM from `torch.cuda.get_device_properties`;
- CUDA runtime, driver, cuDNN, Python, OS/kernel, hostname or RunPod pod/image identifier;
- exact `uv.lock` or environment-export hash;
- matmul precision, TF32, deterministic-algorithm, cuDNN benchmark/determinism settings;
- exact launch command/argv and relevant environment variables.

The base Experiment 4b JSONs have weaker machine provenance than the valid-target diagnostic and should be brought up to the same standard for release.

## Exact closure actions

### P0: required before paper-number claims

1. **Commit and rerun the valid-target evaluator cleanly.** Put `run_valid_target_equal_effect.py`, its protocol, and the final config in a committed revision; run with `git status --porcelain` empty; embed the exact revision, evaluator hash, argv, and environment manifest in `result.json`. A rerun is cheap (about one minute and 1.44 GiB peak allocation).
2. **Make every hashed input durable.** Copy `ioi_confirmatory_cache.pt`, `baseline_confirm/models.pt`, `ioi_selection_models.json`, `ioi_test_models.json`, the complete baseline/IOI JSON set, and `resolved_config.json` to durable project storage or a revision-pinned private Hugging Face repository. Record repository URI, revision, size, and SHA-256 in one versioned manifest.
3. **Rerun or regenerate final IOI analysis from a clean exact revision.** The current structural lower bound (`d985476` or later) is not traceability. The replacement artifact must embed revision, dirty=false, input hashes, resolved config hash, and evaluator hash.
4. **Preserve the training-to-evaluation chain.** For every model payload, record the training checkpoint/model hash, resolved spec, initialization/data/probe seeds, token range, training log hash, and code revision. A single `models.pt` hash is sufficient to reproduce evaluation only if that file is durably stored; it is insufficient to reconstruct training provenance by itself.

### P1: required for a clean robustness appendix

5. **Rerun the group-size exact audit on matched inputs.** Evaluate all 256 `n=64` groups or preselect one fixed token set shared by all sizes. Record indices as an artifact so the 34%/24%/13% trend is not partly confounded with token subsampling.
6. **Version the small raw artifact bundle.** The ignored local/remote JSONs are central evidence and small enough for a release bundle. Include the hash inventory already computed in `00_empirical_audit.md` and validate it in CI with the recomputation script.
7. **Record complete machine metadata.** Add the fields above to both training and evaluation result schemas. Preserve the container/environment lock, not only the PyTorch version.

### P2: statistical transparency

8. **Keep unsupported-bootstrap rates in tables.** Do not quote equal-effect CIs without their valid fractions and conditional estimand. Consider a prespecified bootstrap rule that either freezes validation interpolation weights on test or treats unsupported draws as an explicit failure rather than silently conditioning them away.
9. **Separate confirmatory from descriptive equal-effect results.** The validation-frozen target is confirmatory; re-interpolating brackets on the test frontier is correctly labeled descriptive in the artifact and must stay labeled that way in prose.

## Release verdict

The representation-level source result is numerically auditable, and the new valid-target diagnostic has unusually good input hashing for an exploratory run. The package is not yet independently reproducible because the decisive remote inputs are not durable locally, the final IOI artifact lacks an exact code revision, and the valid-target evaluator ran outside a clean committed state. These are concrete closure tasks, not reasons to discard the results.
