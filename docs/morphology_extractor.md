# Raw ECG Morphology Extraction

`scripts/extract_morphology.py` converts PTB-XL `records500` WFDB waveforms into auditable morphology evidence for the three-class proxy labeler.

PTB-XL stores each high-resolution ECG path in `ptbxl_database.csv` as `filename_hr`, for example `records500/00000/00001_hr`. The extractor reads that path under `data/ptbxl/`, expects the PTB-XL 12-lead order, and uses the 500 Hz physical WFDB signal.

## Run

Small check:

```powershell
python scripts/extract_morphology.py --data-dir data/ptbxl --max-records 20 --out data/morphology_features_debug.csv --beats-out data/morphology_beats_debug.csv --skip-labels
```

Use NeuroKit2 for P-QRS-T fiducial points:

```powershell
python scripts/extract_morphology.py --data-dir data/ptbxl --backend neurokit --out data/morphology_features_neurokit.csv --beats-out data/morphology_beats_neurokit.csv --label-out data/label_manifest_neurokit.csv
```

Use ECGdeli fiducial points exported from MATLAB/Octave:

```powershell
python scripts/extract_morphology.py --data-dir data/ptbxl --backend ecgdeli --ecgdeli-fiducials data/ecgdeli_fiducials.csv --out data/morphology_features_ecgdeli.csv --beats-out data/morphology_beats_ecgdeli.csv --label-out data/label_manifest_ecgdeli.csv
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

For each ECG, the extractor first obtains P-QRS-T/fiducial points from the selected backend:

- `custom`: the original dependency-light Python detector.
- `neurokit`: NeuroKit2 ECG cleaning, R-peak detection, and DWT delineation on lead II.
- `ecgdeli`: normalized fiducial samples exported from ECGdeli.
- `auto`: try NeuroKit2 first, then fall back to `custom` if NeuroKit2 fails.

Only fiducial point acquisition changes between backends. The downstream PR/TP baseline, J-point ST measurement, contiguous-lead STEMI rule, LBBB routing, modified Sgarbossa rule, and label manifest generation are unchanged.

For each ECG after fiducials are available:

1. Read the raw 500 Hz 12-lead WFDB record from `filename_hr`.
2. Use QRS offset as the J point.
3. Estimate PR baseline from 80-20 ms before QRS onset; fall back to TP/pre-QRS baseline if needed.
4. Measure ST displacement at J and J+60 ms for every lead.
5. Keep P, Q, S, and T peaks in lead II for audit.
6. Aggregate clean beats by median per lead.
7. Emit STEMI and LBBB/modified-Sgarbossa features.

Standard STEMI is positive when at least two contiguous leads exceed the guideline J-point threshold. In the primary protocol, LBBB is positive only when `scp_codes` contains a positive LBBB code. For LBBB, STEMI-equivalent morphology is decided by modified Sgarbossa rather than ordinary ST elevation. The simplified raw rule (QRS duration >=120 ms, negative V1, and positive lateral leads) remains available through `--lbbb-source raw` or `either` for sensitivity analysis only.

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
- `requested_delineation_backend`
- `delineation_backend`

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
- `delineation_backend`
- `fiducial_source`

The normalized ECGdeli CSV must contain one row per beat with `ecg_id` and these sample columns when available:

```text
beat_index,r_sample,qrs_onset_sample,qrs_offset_sample,p_peak_sample,q_peak_sample,s_peak_sample,t_peak_sample
```

`r_sample`, `qrs_onset_sample`, and `qrs_offset_sample` are the critical columns. Missing P/Q/S/T peak columns are filled by the same local lead-II audit windows used by the custom backend.

If `--label-out` is enabled, the extractor also writes `data/label_manifest.csv`. The manifest merges PTB-XL metadata with `morphology_features.csv`, then applies the final labels used by training.

## Quality Rule

If a record has explicit morphology failure, such as `morphology_quality=load_error`, MI records are excluded as `exclude_morphology_failed` instead of being forced into `nstemi_proxy`. This prevents failed ST measurement from becoming a false "no STEMI pattern" label.
