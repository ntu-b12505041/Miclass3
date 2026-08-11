# IECBES 2026 corrected experiment protocol

This document pre-specifies the experiments required after the old-MI and
LBBB-label audit. Do not copy the legacy fold-10 metrics into the corrected
paper: every reported number must come from the artifacts produced by this
protocol.

## Primary label protocol

- Exclude an MI-coded record as old MI only when any PTB-XL infarction-stage
  field equals `Stadium III` (case-insensitive after trimming whitespace).
- Retain `Stadium I`, `Stadium II`, and `Stadium II-III` in the primary cohort.
- Treat LBBB as present only when `scp_codes` contains a positive code whose
  name includes `LBBB`.
- Route SCP-positive LBBB through modified Sgarbossa; do not apply the ordinary
  contiguous-lead STEMI rule to those records.
- Do not pass `lbbb` or `modified_sgarbossa_positive` directly into the primary
  three-class classifier. The optional LBBB auxiliary head predicts the
  SCP-derived target from the raw waveform embedding only.

### Why Stadium II-III is retained

PTB-XL describes `Stadium III` as the completed/old infarction category.
`Stadium II-III` spans a transition from subacute/recent to chronic morphology,
so treating every such record as definitively old would remove potentially
relevant evolving MI examples. The primary exclusion therefore uses the most
specific old-MI category. A stricter `Stadium II-III + Stadium III` sensitivity
analysis is methodologically reasonable, but it is deferred because the
current correction is aimed at removing confirmed old MI without broadening
the exclusion to an ambiguous transition category. The manuscript must state
this as a limitation and future sensitivity analysis, not imply that it was
performed.

## Pre-specified experiment matrix

| Experiment | LBBB source | Classifier inputs | Seeds | Purpose |
|---|---|---|---|---|
| Primary morphology fusion | SCP only | Continuous morphology; no routing flags | 42, 2026, 815 | Main result |
| Waveform-only SE-ResNet | SCP-only labels | Raw 12-lead waveform | 42, 2026, 815 | Tests the value of fusion |
| Routing-flag ablation | SCP only | Adds LBBB and modified-Sgarbossa flags | 42, 2026, 815 | Quantifies direct label-rule assistance |
| No-LBBB-auxiliary ablation | SCP only | Same as primary | 42 | Tests auxiliary-task dependence |
| Either-source sensitivity | SCP or simplified raw rule | Same as primary | 42 | Measures impact of the legacy route |
| Raw-source sensitivity | Simplified raw rule only | Same as primary | 42 | Isolates the simplified rule |

All rows are pre-specified comparisons. Do not choose a model or threshold from
fold 10. Train on official folds 1-8, select/calibrate on fold 9, and report the
locked fold-10 result only after the protocol is frozen.

## Run on the T4 GPU

Place the official PTB-XL `records500` tree under `data/ptbxl/records500`, then
run:

```bash
python scripts/run_iecbes2026_experiments.py --device cuda
```

Inspect the complete command matrix without starting extraction or training:

```bash
python scripts/run_iecbes2026_experiments.py --device cuda --dry-run
```

Corrected artifacts are written under `artifacts/iecbes2026_corrected/`. Report
the mean and standard deviation across the three confirmation seeds for the
primary and major ablations, while retaining each seed's confusion matrix and
per-class metrics. The single-seed sensitivity runs are descriptive and must
be labelled as such.

## Required manuscript updates after the runs

1. Replace every legacy performance number and the confusion matrix.
2. Report corrected cohort counts by split and class after `Stadium III`
   exclusion.
3. Report the number and proportion of STEMI-proxy and NSTEMI-proxy records
   routed through SCP-positive LBBB in each split.
4. Report the primary-versus-waveform and routing-flag ablation deltas.
5. State that the `Stadium II-III + Stadium III` sensitivity analysis was not
   performed and give the rationale above.
