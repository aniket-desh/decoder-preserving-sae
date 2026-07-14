# Experiment 4b adversarial theory sprint

This package starts from the newest Experiment 4b raw artifacts and separates confirmed empirical facts, proved finite-matrix statements, failed claims, and open mechanism questions. No training checkpoints, raw result artifacts, or core experiment code were modified.

## Reading order

1. [`00_empirical_audit.md`](00_empirical_audit.md) — exact protocol, provenance, recomputed headline tables, and empirical claim ledger.
2. [`05_final_theory.md`](05_final_theory.md) — clean theorem set and the final scope decision.
3. [`03_failed_routes_and_counterexamples.md`](03_failed_routes_and_counterexamples.md) — explicit constructions that block stronger claims.
4. [`06_paper_narrative.md`](06_paper_narrative.md) — recommended paper story and safe wording.
5. [`07_intuitive_explanation_for_aniket.md`](07_intuitive_explanation_for_aniket.md) — non-paper explanation of what the result means.
6. [`08_open_questions_and_next_tests.md`](08_open_questions_and_next_tests.md) — ranked gaps and the single next experiment.
7. [`01_approach_registry.md`](01_approach_registry.md), [`02_working_derivations.md`](02_working_derivations.md), and [`04_theory_claim_ledger.md`](04_theory_claim_ledger.md) — route history, algebra, and claim-by-claim status.

Independent derivations and cross-red-team notes are retained in [`agent_notes/`](agent_notes/). Reproducible algebraic and artifact checks are under [`checks/`](checks/).

## Sprint verdict

Experiment 4b confirms a representation-level result: DPSAE improves exact held-out finite-group ridge prediction-operator preservation by about 24% versus paired MSE SAEs at \(k=32,n=128\), at about a 7% NMSE cost. It does not confirm causal specificity or harder-target preservation. The static rank-derived omission-cost control explains little of the gain, leaving the shared sparse nonlinear allocation mechanism unresolved.

The central theoretical guarantee is exact but transductive: decoder distance equals expected disagreement of separately refitted in-group ridge predictions and bounds absolute worst-case disagreement on a declared target ellipsoid. It does not control frozen downstream weights or define a grouping-independent corpus geometry.

## Reproduction

From the repository root:

```bash
python3 theory_sprint/2026-07-14_exp4b/checks/verify_theory.py
python3 theory_sprint/2026-07-14_exp4b/checks/recompute_exp4b_tables.py \
  artifacts/exp04b_confirmatory/natural_evaluation_source.json
```

The first command checks theorem identities and counterexamples using small deterministic numerical problems. The second reconstructs source-fleet exact reductions from raw per-group sums and asserts agreement with the stored metrics.

## Audit caveats

- The complete small baseline and IOI JSON artifacts were available on the supplied RunPod but were not present in the local artifact directory at sprint start; their hashes are recorded in `00_empirical_audit.md`.
- The final IOI artifact does not record its exact Git revision, so it falls short of the repository's traceability standard.
- The \(n=64\) group-size audit uses 8,192 tokens because of a 128-group cap; the unconfounded \(n=128\) versus \(n=256\) comparison is sufficient for the group-dependence conclusion.
- The optional cache-only dense-ridge diagnostic did not run because its initial read-only SSH inspection hung. [`agent_notes/experimental_diagnostic.md`](agent_notes/experimental_diagnostic.md) records the blocker and confirms that no remote computation or mutation occurred.
