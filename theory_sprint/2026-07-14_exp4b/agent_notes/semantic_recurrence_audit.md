# Experiment 5 semantic-recurrence audit

## Verdict

The 192 preregistered decoder-advantage eigentasks do not support a recurring semantic hypothesis. Every mode has been reviewed and rejected, the registry is frozen with zero hypotheses, and the sealed 195M–200M final cache was never located, created, deserialized, or otherwise accessed. The frozen registry digest is `4b96f98af408e931ad94153c5941fc34c734ac8fa61abb8bc6ca516efed57c23`.

## Scope and method

The audit used the existing RunPod artifacts under `artifacts/exp05_decoder_advantage_discovery/` and the already-open `artifacts/exp04b_confirmatory/natural_selection.pt` cache covering 180M–185M. The registered discovery manifest, searched-mode log, open registry, and natural-selection cache were hashed before analysis. Existing confirmation, generality, finalization, and backup sessions were observed only through `tmux ls`; no running job or its artifacts were touched.

For every mode, the analysis reconstructed its 128 registered group tokens, decoded the eight positive and eight negative extreme contexts with the GPT-2 tokenizer, and tested a fixed library of token identity, neighboring-token, punctuation, whitespace, case, numeric, quotation, code-like, URL-like, and sequence-position indicators. Each centered indicator was compared with the eigentask by cosine. A 256-permutation within-mode maximum statistic controlled the feature scan; an alignment required absolute cosine at least 0.25 and familywise `p <= 0.05`. Promotion then required the same feature on the same advantage side in at least two seeds and four independent group positions.

The automated pass covered all 192 modes and 48 numerical controls. It also inspected all 131 eligible recurrence-range sequences, totaling 33,536 tokens; no feature reached the discovery recurrence gate, so there was no candidate whose recurrence-range support needed promotion. Manual review then inspected the positive and negative extreme token sets for all 192 modes. Full context windows were inspected for the seven familywise lexical outliers and the 20 modes whose observed eigenvalue beat the row-shuffle control, for 27 full-context modes total.

## Results

- The row-shuffle control matched or exceeded the observed eigenvalue magnitude for 172 of 192 modes.
- Seven modes had a familywise lexical or positional alignment: `position:first_8`, `previous:is`, `next:and`, `previous:the`, `next:in`, `context:numeric`, and `next:the`. Every alignment occurred in exactly one seed and one group; none recurred.
- The remaining 185 modes had no feature survive the within-mode maximum statistic.
- No same-side feature cluster reached two seeds and four groups, so the qualified recurrence-cluster count and the number of modes in such clusters are both zero.
- Manual context inspection found heterogeneous documents, token types, and topics on both extremes. The 20 modes less dominated by row shuffle also lacked a stable interpretation across seeds or groups.

The finalizer bound the manual audit hash into the registry, wrote a specific rejection note for each mode containing its best alignment or singleton outlier and row-shuffle ratio, verified 192 nonempty notes, and froze with `Counter({'rejected': 192})` and no hypotheses. This is a negative discovery result: it does not claim that no untested semantic partition exists, but the preregistered search produced nothing defensible to carry into the sealed final split. The final cache should remain unopened because the frozen registry contains no hypothesis to test.

## Reproducible artifacts

Remote RunPod artifacts:

- `artifacts/exp05_decoder_advantage_discovery/semantic_recurrence.json` — `37bc22528087e29a04343fa224e678c117920e6ce548eeab5184930bcf95469a`
- `artifacts/exp05_decoder_advantage_discovery/semantic_review.tsv` — `4346109b984f1e414859628599e01f3600bd715730acc81df31cc2ad038c3d47`
- `artifacts/exp05_decoder_advantage_discovery/semantic_manual_audit.json` — `e28dd7f465c90e846d4eb33724586314668639be44bbd92c64c751b7438cd270`
- `artifacts/exp05_decoder_advantage_discovery/hypothesis_registry.json` — `545b14753a89b424cfe446005b329da5cfe7ad1178ef8c9327c23d26da228b02`
- `artifacts/exp05_decoder_advantage_discovery/semantic_finalize.log` — `ed2bd6778dd728fd05d4b6cc465663511f628942e61b16ee8d9eeb89a80fd1ce`

Local analysis code:

- `experiments/exp05_semantic_recurrence.py` — `e99f7c8119bc42364ed8a1d4c7f8586e5f56db6a967b6570e4e593ae430f3001`
- `experiments/exp05_finalize_semantic_review.py` — `98e40351a8697aa4894cc7bfb08cc755f32dcd70c9fee0b0b272fd7534bf8e35`

Both remote analysis steps ran in dedicated named sessions, `dpsae-exp05-semantic-review` and `dpsae-exp05-semantic-finalize`. Neither used the GPU.
