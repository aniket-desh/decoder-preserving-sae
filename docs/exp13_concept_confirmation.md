# Exp13 fresh-pair concept confirmation contract

Exp13 is a separate confirmatory evaluation stage. It may start only after
Exp12 writes a valid `concept_evaluation_authorization.json`, the user approves
the paid evaluation, and the final Exp10 schema-v7 timing report, cache
manifest, and final artifact audit are hash-bound. It never regenerates the
113-task activation cache and never changes the Exp10 sparse or companion
estimators.

The stage extracts the common Exp12 maturity checkpoint for each of the three
paired training seeds and writes three Exp10-compatible checkpoint bundles.
Every bundle records the original snapshot manifest and model hash, the shared
calibration hash, the derived Exp10 config hash, and the exact MSE/DPSAE payload
names. The lower-level `run_sparse_job` and `run_companion_job` functions are
then called directly, so dataset splits, probe seeds, regularization paths,
selected-feature refits, held-out predictions, and companion controls remain
identical to the pilot.

Four fixed workers cover exactly 60 sparse jobs (three pairs by two methods by
ten probe seeds) and 30 companion jobs (three pairs by ten probe seeds). The
assignment is frozen by a template digest before pair seeds are known: sparse
identities are ordered by pair slot, method, and seed slot and assigned modulo
four; companion identities are ordered by pair slot and seed slot and assigned
the same way. This yields 15 sparse jobs per worker and companion counts
8/8/7/7. Each worker loads both adapters at most once per checkpoint pair.

The blind spend gate is three times the complete schema-v7 pilot projection,
including its fixed terms, and must not exceed 7.5 pod-hours. This conservative
projection is written before any fresh-pair concept result is opened. A failed
timing, cache, authorization, environment, worker, or audit check stops the
fleet and leaves a retained failure status.

For the primary result, probe seeds are averaged within each
pair/task/method. The taskwise DPSAE-minus-MSE AUROC differences then produce
one macro per trained pair. Confirmation requires all three pair macros to be
positive, their median to be at least 0.005, the lower 95% family-block
bootstrap endpoint of pair-averaged task effects to be positive, an exact
complete artifact matrix, and every Exp12 matched-quality gate to remain true.
Family one-sided centered paired bootstraps resample held-out examples within
class and receive a Holm correction, but these p-values are reporting-only.
The implementation draws the equivalent per-class multinomial counts and
evaluates AUROC with tie-aware cumulative rank counts, preserving the exact
stratified empirical bootstrap without millions of per-draw sklearn calls.

Candidate promotion occurs only after the overall confirmation gate passes.
A task must have positive probe-seed-averaged effects in all three pairs. A
feature remains checkpoint-local and must appear in the selected five features
for at least five of ten probe seeds. Candidates are ranked by selection
frequency and then mean absolute probe weight. MSE and DPSAE receive equal
budgets capped at 300 each; the stage never relaxes the recurrence rule to fill
the quota. Every candidate retains hashes for its contributing provenance and
per-example prediction artifacts.

The candidate manifest exports the complete three-pair confirmation gate,
including pair seeds, checkpoint IDs, selected maturity budget, and each gate
check. Only rows emitted after that gate pass carry
`autointerp_eligible: true`, so the context miner cannot consume an unconfirmed
or pair-ambiguous association file.

The production launcher is `scripts/run_exp13_concept_confirmation_4xa40.sh`.
It refuses to start without `EXP13_USER_APPROVED=YES`, a clean committed
revision, exactly four visible A40s, the pinned SAEBench virtual environment,
and every frozen upstream artifact. The entry process immediately moves into
`exp13-launch`; successful freeze then creates `exp13-gpu0` through
`exp13-gpu3` and `exp13-finalize`, all with `remain-on-exit` enabled and
retained logs. `EXP12_ROOT` must identify the authorized Exp12 run; the other
defaults target the dated final Exp10 run and shared activation cache. The
`status` subcommand is read-only and reports the contract, abort marker,
worker states, finalizer, aggregate, and both audit states.
