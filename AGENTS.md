# Agent Instructions

Before changing, training, or reporting results for this repository, read:

```text
docs/ai_agent_runbook.md
```

Key rules:

- Do not fabricate metrics, confusion matrices, label counts, or training results.
- Do not train until `data/ptbxl/records500/` exists.
- Use `scripts/extract_morphology.py` to create `data/morphology_features.csv`, `data/morphology_beats.csv`, and `data/label_manifest.csv` from raw PTB-XL `records500`.
- Treat `nstemi_proxy` as an ECG-only proxy label, not clinically adjudicated NSTEMI.
- Report results only from files under the selected `artifacts/` run directory.

