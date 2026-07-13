# Miclass3 v1 label specification

## Purpose

This project predicts ECG-only proxy labels from PTB-XL `records500`. It does **not** claim to diagnose clinically adjudicated NSTEMI. The exact output classes are `non_mi`, `stemi_proxy`, and `nstemi_proxy`.

## Deterministic rule

```text
no MI SCP statement                            -> non_mi
MI SCP statement + old infarction stage        -> excluded
MI SCP statement + standard STEMI morphology   -> stemi_proxy
MI SCP statement + LBBB + mSgarbossa positive  -> stemi_proxy
all remaining MI SCP statements                -> nstemi_proxy
```

`nstemi_proxy` means *MI SCP statement without a STEMI-equivalent ECG pattern*. It must never be described as a troponin-confirmed clinical NSTEMI.

## STEMI morphology input

The morphology table is keyed by `ecg_id` and must include `standard_stemi`, `lbbb`, and `modified_sgarbossa_positive`. Calculate J-point displacement from a PR (or TP) baseline using a median across clean beats on the raw 500 Hz waveform. Standard STEMI requires the guideline J-point threshold in at least two contiguous leads: 0.1 mV in non-V2/V3 leads; V2/V3 thresholds are 0.25 mV for men under 40, 0.20 mV for men 40 or older, and 0.15 mV for women.

For confirmed LBBB, modified Sgarbossa is positive if any of: concordant STE >=0.1 mV; concordant STD >=0.1 mV in V1-V3; or discordant STE >=0.1 mV with ST/S >=0.25. Keep the raw measurements, beat counts, baseline source, and reviewer status in the morphology CSV.

## Quality rules

- Use `records500`, never the 100 Hz release, to measure J points.
- Exclude an ECG if delineation fails, fewer than three clean beats remain, lead calibration is missing, or LBBB status is ambiguous.
- Do not use `infarction_stadium=old` as a negative or NSTEMI-proxy sample.
- Split by PTB-XL official folds: 1-8 train, 9 validation, 10 locked test.

## References

- Wagner et al. PTB-XL. *Scientific Data* (2020). https://doi.org/10.1038/s41597-020-0495-6
- Thygesen et al. Fourth Universal Definition of MI. *JACC* (2018). https://doi.org/10.1016/j.jacc.2018.08.1038
- Smith et al. Modified Sgarbossa rule. *Ann Emerg Med* (2012). https://doi.org/10.1016/j.annemergmed.2012.07.119
- Meyers et al. Validation of modified Sgarbossa criteria. *Am Heart J* (2015). https://doi.org/10.1016/j.ahj.2015.09.016

