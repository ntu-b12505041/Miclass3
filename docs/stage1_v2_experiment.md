# Stage1-v2: NSTEMI-aware multi-scale MI router

## Motivation

The current common-cohort cascade shows that Stage 1 is the main routing bottleneck. On the held-out common-cohort test fold, the existing Stage 1 had much stronger routing recall for STEMI-proxy than for NSTEMI-proxy. A threshold-only audit also showed that the current Stage 1 probability ranking could not reach 0.85 sensitivity and specificity simultaneously.

Stage 2 is not changed in this experiment. The purpose is to isolate whether a stronger MI-vs-non-MI waveform representation can improve the first-stage routing boundary, especially for NSTEMI-proxy records that do not have obvious STEMI morphology.

## Architecture

`Stage1V2MultiScaleSE` uses raw 12-lead PTB-XL `records500` input.

1. A strided projection plus average pooling reduces 500 Hz to 125 Hz before the expensive multi-scale convolutions.
2. Three parallel temporal kernels `(11, 25, 49)` then span approximately 88 ms, 200 ms and 392 ms.
3. The branches are fused with squeeze-excitation.
4. The result is passed through the existing residual SE encoder family.
5. The primary head predicts MI vs non-MI.
6. A training-only auxiliary head predicts non-MI / STEMI-proxy / NSTEMI-proxy, with extra NSTEMI-proxy loss weight.

The auxiliary three-class head is not a clinical endpoint and is not used as the Stage 1 routing output. It is only intended to stop the shared encoder from solving the binary task primarily through conspicuous STEMI morphology.

## Validation objective

Checkpoint selection no longer uses aggregate validation AUPRC alone. For every epoch, the MI threshold is selected on fold 9 to maximize the weakest of:

- non-MI retained as non-MI
- STEMI-proxy routed to the MI branch
- NSTEMI-proxy routed to the MI branch

The checkpoint score is this maximin route recall. This directly targets the observed failure mode.

Fold 10 is intentionally not evaluated during training. After accepting the validation result, run `--finalize-only` exactly once to freeze the selected checkpoint and threshold and evaluate fold 10.

## Train

```bash
cd ~/mi-3class-training-project/Miclass3
source .venv/bin/activate

git checkout feature/two-stage-mi-cascade
git pull

python scripts/train_stage1_v2.py \
  --config configs/stage1_v2_multiscale.yaml \
  --manifest data/label_manifest_custom65_common.csv \
  --device cuda
```

Primary outputs:

```text
artifacts/stage1_v2_multiscale/metrics.json
artifacts/stage1_v2_multiscale/history.csv
artifacts/stage1_v2_multiscale/val_threshold_candidates.csv
artifacts/stage1_v2_multiscale/val_predictions.csv
artifacts/stage1_v2_multiscale/best_stage1_v2.pt
```

Inspect `metrics.json -> validation` before any fold-10 evaluation. The most important fields are:

```text
non_mi_recall
stemi_route_recall
nstemi_route_recall
worst_route_recall
binary_mi_sensitivity
binary_mi_specificity
auroc
auprc
```

## Finalize once

Only after the validation operating point is accepted:

```bash
python scripts/train_stage1_v2.py \
  --config configs/stage1_v2_multiscale.yaml \
  --manifest data/label_manifest_custom65_common.csv \
  --device cuda \
  --finalize-only
```

This adds the frozen fold-10 result to `artifacts/stage1_v2_multiscale/metrics.json` and writes `test_predictions.csv`.

## Decision rule for the next experiment

If Stage1-v2 materially improves NSTEMI-proxy routing while preserving non-MI retention, integrate the new Stage 1 probabilities with the already-strong Stage 2 and rerun validation-only hierarchical calibration. If it does not, the next candidate should test a pretrained ECG encoder or a Direct/Cascade ensemble rather than further threshold tuning of the old Stage 1.

All reported labels remain ECG-derived proxy labels. Do not describe the result as clinical NSTEMI diagnostic accuracy.
