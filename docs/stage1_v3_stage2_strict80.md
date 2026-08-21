# Stage1-v3 + existing Stage2: strict-80 final three-class calibration

## Goal

Evaluate the full hierarchical classifier on the common cohort with the operational milestone that all six one-vs-rest metrics are strictly above 0.80:

- non-MI sensitivity
- non-MI specificity
- STEMI-proxy sensitivity
- STEMI-proxy specificity
- NSTEMI-proxy sensitivity
- NSTEMI-proxy specificity

Recall and sensitivity are the same quantity here.

## Models used

This experiment does not retrain either stage.

- Stage 1: `artifacts/stage1_v3_morphology_fusion/` — full-resolution SE-ResNet plus 53 safe morphology features.
- Stage 2: the existing common-cohort MorphologyFusion model represented by `artifacts/cascade_v3_custom65_common/*_cascade_details.csv`.

The Stage2 detail file contains `p_stemi_given_mi` evaluated on the full split, so it can be combined with the new Stage1-v3 `p_mi` without using the old Stage1 decision.

## Decision rule

For thresholds `t_MI` and `t_STEMI`:

```text
if P(MI) < t_MI:
    predict non-MI
else:
    if P(STEMI | MI) >= t_STEMI:
        predict STEMI-proxy
    else:
        predict NSTEMI-proxy
```

## Validation objective

Thresholds are selected using PTB-XL Fold 9 only. The primary objective is:

```text
maximize min(
  non-MI sensitivity,
  non-MI specificity,
  STEMI-proxy sensitivity,
  STEMI-proxy specificity,
  NSTEMI-proxy sensitivity,
  NSTEMI-proxy specificity
)
```

Tie breakers are:

1. number of the six metrics strictly above 0.80
2. mean of the six metrics
3. accuracy

The search performs a 0.01 coarse grid followed by a 0.001 local fine search.

## Run validation-only calibration

```bash
cd ~/mi-3class-training-project/Miclass3
source .venv/bin/activate

git checkout feature/two-stage-mi-cascade
git pull

pytest tests/test_strict80.py -q

python scripts/calibrate_stage1_v3_stage2_strict80.py \
  --config configs/stage1_v3_stage2_strict80.yaml
```

Outputs:

```text
artifacts/stage1_v3_stage2_strict80/
  metrics.json
  validation_threshold_candidates.csv
  validation_predictions.csv
  validation_confusion_matrix.csv
```

Do not evaluate Fold 10 if `validation.all_six_strictly_above_target` is false.

## Finalize only after validation passes

If all six validation metrics are strictly above 0.80, first generate the frozen Stage1-v3 Fold-10 probabilities:

```bash
python scripts/train_stage1_v3.py \
  --config configs/stage1_v3_morphology_fusion.yaml \
  --manifest data/label_manifest_custom65_common.csv \
  --device cuda \
  --finalize-only
```

Then evaluate the final cascade once with the Fold-9 thresholds frozen:

```bash
python scripts/calibrate_stage1_v3_stage2_strict80.py \
  --config configs/stage1_v3_stage2_strict80.yaml \
  --finalize-only
```

The finalize mode refuses to run if validation did not pass the strict-80 target. No Fold-10 threshold search is performed.

## Interpretation

The labels remain ECG-derived proxy labels. Report them as `STEMI-proxy` and `NSTEMI-proxy`, not adjudicated clinical STEMI/NSTEMI diagnoses.
