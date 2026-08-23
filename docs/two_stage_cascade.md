# Two-stage MI cascade

This branch adds an explicit hierarchical baseline for the existing PTB-XL proxy labels.

```text
12-lead ECG
    |
    v
Stage 1: SE-ResNet
MI vs non-MI
    |
    +-- non-MI ----------> non_mi
    |
    `-- MI
         |
         v
    Stage 2: SE-ResNet
    STEMI-proxy vs NSTEMI-proxy
         |
         +-------------> stemi_proxy
         `-------------> nstemi_proxy
```

## Why this experiment exists

The existing three-class models solve `non_mi / stemi_proxy / nstemi_proxy` directly. The cascade tests a different hypothesis: because MI-vs-non-MI can be an easier and stronger binary task, first isolate MI and only then ask the harder STEMI-vs-NSTEMI question.

This is intentionally an additional comparison model, not a replacement for `morphology_fusion`, `seresnet`, or `inceptiontime`.

## Training protocol

Patient-level PTB-XL folds are unchanged:

- folds 1-8: training
- fold 9: validation and threshold selection
- fold 10: locked test

Stage 1 is trained on every non-excluded record with target:

```text
non_mi -> 0
stemi_proxy or nstemi_proxy -> 1
```

Stage 2 is trained **only on true MI-proxy training records** with target:

```text
nstemi_proxy -> 0
stemi_proxy -> 1
```

Both stages use the same waveform-only SE-ResNet family so the comparison focuses on decision structure rather than a different backbone.

## Final probability composition

For analysis, the script also exports the probabilistic hierarchy:

```text
P(non-MI) = 1 - P(MI)
P(STEMI) = P(MI) * P(STEMI | MI)
P(NSTEMI) = P(MI) * (1 - P(STEMI | MI))
```

The main reported cascade follows the requested literal gate: records rejected by Stage 1 stop as `non_mi`; only records routed as MI receive the Stage-2 STEMI/NSTEMI decision.

Both binary thresholds are selected using validation fold 9 only. Fold 10 never participates in threshold selection.

## Run

```bash
python scripts/train_cascade.py --device cuda
```

Quick pipeline check:

```bash
python scripts/train_cascade.py --device cpu --max-records 300 --out-dir artifacts/cascade_smoke
```

## Outputs

The default output directory is `artifacts/cascade/` and includes:

- `stage1_mi_vs_non_mi.pt`
- `stage2_stemi_vs_nstemi.pt`
- validation threshold candidate tables for both stages
- stage-specific metrics
- final train/validation/test three-class metrics
- final classification reports and confusion matrices
- `*_cascade_details.csv` with `P(MI)`, `P(STEMI|MI)`, soft combined probabilities and hard routed predictions
- `cascade_metrics.json`

## Interpretation safeguard

The second stage predicts `stemi_proxy` versus `nstemi_proxy`, not clinically adjudicated STEMI/NSTEMI. The existing PTB-XL proxy-label limitations remain unchanged.
