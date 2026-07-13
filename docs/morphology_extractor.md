# Raw ECG Morphology Extraction

`scripts/extract_morphology.py` converts PTB-XL `records500` WFDB waveforms into auditable morphology evidence for the three-class proxy labeler.

PTB-XL stores each high-resolution ECG path in `ptbxl_database.csv` as `filename_hr`, for example `records500/00000/00001_hr`. The extractor reads that path under `data/ptbxl/`, expects the PTB-XL 12-lead order, and uses the 500 Hz physical WFDB signal.

## Run

Small check:

```powershell
python scripts/extract_morphology.py --data-dir data/ptbxl --max-records 20 --out data/morphology_features_debug.csv --beats-out data/morphology_beats_debug.csv --skip-labels
```

Full extraction and label manifest build:

```powershell
python scripts/extract_morphology.py --data-dir data/ptbxl --out data/morphology_features.csv --beats-out data/morphology_beats.csv --label-out data/label_manifest.csv
```

Resume an interrupted run:

```powershell
python scripts/extract_morphology.py --data-dir data/ptbxl --resume --out data/morphology_features.csv --beats-out data/morphology_beats.csv --label-out data/label_manifest.csv
```

Process only selected folds:

```powershell
python scripts/extract_morphology.py --data-dir data/ptbxl --folds 9,10 --out data/morphology_features_folds_9_10.csv --beats-out data/morphology_beats_folds_9_10.csv --skip-labels
```

## Method

For each ECG:

1. Read the raw 500 Hz 12-lead WFDB record from `filename_hr`.
2. Bandpass filter a copy of the signal at 5-25 Hz for QRS detection.
3. Build a multi-lead QRS energy envelope and detect R peaks.
4. Refine QRS onset and QRS offset around each R peak.
5. Use QRS offset as the J point.
6. Estimate PR baseline from 80-20 ms before QRS onset; fall back to TP/pre-QRS baseline if needed.
7. Measure ST displacement at J and J+60 ms for every lead.
8. Mark P, Q, S, and T peaks in lead II search windows for audit.
9. Aggregate clean beats by median per lead.
10. Emit STEMI and LBBB/modified-Sgarbossa features.

Standard STEMI is positive when at least two contiguous leads exceed the guideline J-point threshold. LBBB is conservatively detected from raw morphology by QRS duration >=120 ms, negative V1, and positive lateral leads, and can also use PTB-XL SCP LBBB statements. For LBBB, STEMI-equivalent morphology is decided by modified Sgarbossa rather than ordinary ST elevation.

## Outputs

`data/morphology_features.csv` is one row per ECG. Important columns:

- `standard_stemi`
- `lbbb`
- `lbbb_raw`
- `lbbb_scp`
- `modified_sgarbossa_positive`
- `max_st_j_mv`
- `max_st_j60_mv`
- `max_st_s_ratio`
- `qrs_duration_ms`
- `num_detected_beats`
- `num_beats_used`
- `morphology_quality`
- `st_j_<lead>_mv`, `st_j60_<lead>_mv`, `s_depth_<lead>_mv`, `qrs_polarity_<lead>`

`data/morphology_beats.csv` is one row per accepted beat. It keeps the P-QRS-T/J-point landmarks:

- `r_sample`
- `q_peak_sample`
- `s_peak_sample`
- `qrs_onset_sample`
- `qrs_offset_sample`
- `j_point_sample`
- `p_peak_sample`
- `t_peak_sample`
- relative timings in milliseconds from R peak
- `baseline_source`

If `--label-out` is enabled, the extractor also writes `data/label_manifest.csv`. The manifest merges PTB-XL metadata with `morphology_features.csv`, then applies the final labels used by training.

## Quality Rule

If a record has explicit morphology failure, such as `morphology_quality=load_error`, MI records are excluded as `exclude_morphology_failed` instead of being forced into `nstemi_proxy`. This prevents failed ST measurement from becoming a false "no STEMI pattern" label.
