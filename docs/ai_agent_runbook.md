# AI Agent Runbook

This is the first file an AI coding agent should read before operating this repository.

Miclass3 is a PTB-XL `records500` research pipeline for three ECG-only proxy labels:

- `non_mi`
- `stemi_proxy`
- `nstemi_proxy`

Do not describe `nstemi_proxy` as clinically adjudicated NSTEMI. PTB-XL does not include serial troponin, symptoms, angiography, or encounter-level adjudication.

## Operating Rules

1. Do not fabricate metrics, confusion matrices, label counts, or training results.
2. Do not train until `data/ptbxl/records500/` exists.
3. Prefer the raw ECG morphology extractor over hand-written morphology tables.
4. Keep old MI and failed morphology records excluded from training.
5. Use official PTB-XL folds: train folds 1-8, validation fold 9, test fold 10.
6. Report every generated result from files under `artifacts/`, not from memory.

## Required Data Layout

The repository does not redistribute PTB-XL waveforms. The expected local layout is:

```text
data/ptbxl/
  ptbxl_database.csv
  scp_statements.csv
  records500/
    00000/
      00001_hr.dat
      00001_hr.hea
```

`ptbxl_database.csv` contains `filename_hr`, which points to the high-resolution WFDB record under `records500`.

## Full Pipeline

Run these commands from the repository root.

Install:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

If GPU training is requested, install a CUDA-enabled PyTorch build before `requirements.txt` when needed. Verify with:

```powershell
nvidia-smi
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Extract morphology from raw ECG and build labels:

```powershell
python scripts/extract_morphology.py --data-dir data/ptbxl --out data/morphology_features.csv --beats-out data/morphology_beats.csv --label-out data/label_manifest.csv
```

The extractor writes:

- `data/morphology_features.csv`: one row per ECG for record-level labeling and model fusion features.
- `data/morphology_beats.csv`: beat-level P-QRS-T/J-point audit table.
- `data/label_manifest.csv`: final training manifest with `label`, `label_id`, `label_tier`, `label_reason`, and `mi_scp_codes`.

If extraction is interrupted, resume with:

```powershell
python scripts/extract_morphology.py --data-dir data/ptbxl --resume --out data/morphology_features.csv --beats-out data/morphology_beats.csv --label-out data/label_manifest.csv
```

Run a small smoke test:

```powershell
python scripts/train.py --model seresnet --device cuda --max-records 300 --out-dir artifacts/smoke_seresnet
```

Train the primary model:

```powershell
python scripts/train.py --model morphology_fusion --device cuda --out-dir artifacts/morphology_fusion
```

Optional comparison models:

```powershell
python scripts/train.py --model inceptiontime --device cuda --out-dir artifacts/inceptiontime
python scripts/train.py --model seresnet --device cuda --out-dir artifacts/seresnet
```

## Morphology Extractor Summary

The raw ECG extractor is implemented in:

```text
src/miclass3/delineation.py
scripts/extract_morphology.py
```

It performs:

1. Multi-lead QRS energy detection.
2. R peak detection.
3. QRS onset and offset localization.
4. J point assignment at QRS offset.
5. PR baseline estimation, with TP/pre-QRS fallback.
6. ST-J and ST-J+60 measurement for all leads.
7. P, Q, S, and T peak audit marks in lead II.
8. Beat-level median aggregation into record-level morphology features.
9. Standard STEMI contiguous-lead J-point decision.
10. LBBB routing through modified Sgarbossa.

Important columns in `morphology_features.csv`:

```text
standard_stemi
lbbb
lbbb_raw
lbbb_scp
modified_sgarbossa_positive
max_st_j_mv
max_st_j60_mv
max_st_s_ratio
qrs_duration_ms
morphology_quality
```

If `morphology_quality` is a failure value such as `load_error`, MI records must be excluded as `exclude_morphology_failed`, not converted to `nstemi_proxy`.

## Results Reporting

Every completed training run writes:

- `train|val|test_metrics.json`
- `train|val|test_classification_report.csv`
- `train|val|test_confusion_matrix.csv`
- `train|val|test_confusion_matrix.png`
- `train|val|test_predictions.csv`
- `best_model.pt`
- `metrics.json`

When reporting results, read the test files from the selected artifact directory, usually:

```text
artifacts/morphology_fusion/
```

Report at least:

- accuracy
- balanced accuracy
- macro AUROC
- macro AUPRC
- macro F1
- STEMI-proxy recall
- per-class precision, recall/sensitivity, specificity, F1, support
- confusion matrix

State clearly if the run was a smoke test or partial run.

## Recovery Checklist

If something fails:

- Missing `records500`: stop and ask for the official PhysioNet files.
- `torch.cuda.is_available() == False`: stop GPU training and report CUDA/PyTorch mismatch.
- Missing `label_manifest.csv`: run `scripts/extract_morphology.py`.
- Morphology extraction interrupted: rerun with `--resume`.
- Training completes but result files are absent: inspect `scripts/train.py` output directory and rerun the failed command.

