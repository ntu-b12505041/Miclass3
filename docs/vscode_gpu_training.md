# VS Code GPU Training Guide

This project trains PyTorch models from PTB-XL `records500`. Use GPU only after the label manifest and waveform tree are present.

## 1. Open the project

Open this folder in VS Code:

```powershell
cd "C:\Users\yu891\OneDrive\文件\琉球\Miclass3"
code .
```

## 2. Create and select the Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
```

Install the PyTorch build that matches your NVIDIA CUDA driver. For most modern Windows CUDA 12.1 setups:

```powershell
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install -r requirements.txt
python -m pip install -e .
```

If CUDA 12.1 is not compatible with the machine, ask the VS Code agent to check `nvidia-smi` first and select the matching PyTorch install command from the official PyTorch selector.

In VS Code, run `Python: Select Interpreter` and choose:

```text
.\.venv\Scripts\python.exe
```

## 3. Verify GPU access

```powershell
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'NO CUDA')"
```

Continue only if `torch.cuda.is_available()` prints `True`.

## 4. Prepare data

The repository does not redistribute PTB-XL waveforms. Put the official PhysioNet high-resolution waveform tree here:

```text
data/ptbxl/records500/
```

Extract morphology and build labels:

```powershell
python scripts/extract_morphology.py --data-dir data/ptbxl --out data/morphology_features.csv --beats-out data/morphology_beats.csv --label-out data/label_manifest.csv
```

If you already have a reviewed morphology table, build labels directly:

```powershell
python scripts/build_labels.py --data-dir data/ptbxl --morphology-csv data/morphology_features.csv --out data/label_manifest.csv
```

If `data/morphology_features.csv` is not available yet, the manifest can still be built without it, but STEMI-proxy labels will be conservative and incomplete:

```powershell
python scripts/build_labels.py --data-dir data/ptbxl --out data/label_manifest.csv
```

## 5. Run a small GPU smoke test

```powershell
python scripts/train.py --model seresnet --device cuda --max-records 300 --out-dir artifacts/smoke_seresnet
```

Confirm that the run writes metrics and confusion matrices under `artifacts/smoke_seresnet/`.

## 6. Train the primary model

```powershell
python scripts/train.py --model morphology_fusion --device cuda --out-dir artifacts/morphology_fusion
```

Optional comparison models:

```powershell
python scripts/train.py --model inceptiontime --device cuda --out-dir artifacts/inceptiontime
python scripts/train.py --model seresnet --device cuda --out-dir artifacts/seresnet
```

Each run writes:

- `train|val|test_metrics.json`
- `train|val|test_classification_report.csv`
- `train|val|test_confusion_matrix.csv`
- `train|val|test_confusion_matrix.png`
- `train|val|test_predictions.csv`
- `best_model.pt`
- `metrics.json`

## Prompt for a VS Code AI Agent

```text
You are working in the Miclass3 repository. Please set up and run GPU training for the PTB-XL three-class MI proxy model.

Follow docs/ai_agent_runbook.md first, then docs/vscode_gpu_training.md:
1. Check the current Python interpreter and create/use .venv if needed.
2. Check NVIDIA GPU availability with nvidia-smi.
3. Install the correct CUDA-enabled PyTorch build, then install requirements.txt and the package in editable mode.
4. Verify torch.cuda.is_available() is True.
5. Confirm that data/ptbxl/records500 exists.
6. If data/morphology_features.csv or data/label_manifest.csv is missing, run:
   python scripts/extract_morphology.py --data-dir data/ptbxl --out data/morphology_features.csv --beats-out data/morphology_beats.csv --label-out data/label_manifest.csv
7. Run a smoke test:
   python scripts/train.py --model seresnet --device cuda --max-records 300 --out-dir artifacts/smoke_seresnet
8. If the smoke test succeeds, train the primary model:
   python scripts/train.py --model morphology_fusion --device cuda --out-dir artifacts/morphology_fusion
9. After training, summarize the test metrics and confusion matrix files generated under artifacts/morphology_fusion.

Do not fabricate results. If data or CUDA is missing, stop and report exactly what is missing.
```
