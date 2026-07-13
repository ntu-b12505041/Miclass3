from __future__ import annotations

import numpy as np
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    recall_score,
    roc_auc_score,
)

from . import CLASS_NAMES


def multiclass_metrics(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, float | list[list[int]]]:
    pred = probabilities.argmax(axis=1)
    onehot = np.eye(probabilities.shape[1])[y_true]
    output: dict[str, float | list[list[int]]] = {
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "stemi_recall": float(recall_score(y_true, pred, labels=[1], average=None, zero_division=0)[0]),
        "confusion_matrix": confusion_matrix(y_true, pred, labels=[0, 1, 2]).tolist(),
    }
    try:
        output["macro_auroc"] = float(roc_auc_score(onehot, probabilities, average="macro", multi_class="ovr"))
        output["macro_auprc"] = float(average_precision_score(onehot, probabilities, average="macro"))
    except ValueError:
        output["macro_auroc"] = float("nan")
        output["macro_auprc"] = float("nan")
    return output


def classification_tables(y_true: np.ndarray, probabilities: np.ndarray) -> tuple[dict[str, float | list[list[int]]], pd.DataFrame, pd.DataFrame]:
    """Return common three-class metrics and labelled confusion/per-class tables."""
    y_true = np.asarray(y_true, dtype=int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = probabilities.argmax(axis=1)
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    precision, recall, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=[0, 1, 2], zero_division=0)
    rows = []
    for index, label in enumerate(CLASS_NAMES):
        tn = matrix.sum() - matrix[index, :].sum() - matrix[:, index].sum() + matrix[index, index]
        fp = matrix[:, index].sum() - matrix[index, index]
        rows.append({
            "class": label,
            "precision": float(precision[index]),
            "recall_sensitivity": float(recall[index]),
            "f1": float(f1[index]),
            "specificity": float(tn / max(tn + fp, 1)),
            "support": int(support[index]),
        })
    overall = multiclass_metrics(y_true, probabilities)
    overall["accuracy"] = float(accuracy_score(y_true, y_pred))
    overall["n_records"] = int(len(y_true))
    return overall, pd.DataFrame(rows), pd.DataFrame(matrix, index=CLASS_NAMES, columns=CLASS_NAMES)


def write_split_artifacts(
    output_dir: str | Path,
    split: str,
    y_true: np.ndarray,
    probabilities: np.ndarray,
    records: pd.DataFrame | None = None,
) -> dict[str, float | list[list[int]]]:
    """Persist a complete, reviewable result bundle for validation or test data."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    overall, per_class, matrix = classification_tables(y_true, probabilities)
    (out / f"{split}_metrics.json").write_text(json.dumps(overall, indent=2), encoding="utf-8")
    per_class.to_csv(out / f"{split}_classification_report.csv", index=False)
    matrix.to_csv(out / f"{split}_confusion_matrix.csv", index_label="actual\\predicted")
    prediction = pd.DataFrame({
        "actual_id": np.asarray(y_true, dtype=int),
        "actual_label": [CLASS_NAMES[i] for i in y_true],
        "predicted_id": probabilities.argmax(axis=1),
        "predicted_label": [CLASS_NAMES[i] for i in probabilities.argmax(axis=1)],
        **{f"prob_{name}": probabilities[:, idx] for idx, name in enumerate(CLASS_NAMES)},
    })
    if records is not None:
        identifiers = [col for col in ("ecg_id", "patient_id", "strat_fold", "label_reason") if col in records]
        prediction = pd.concat([records.reset_index(drop=True)[identifiers], prediction], axis=1)
    prediction.to_csv(out / f"{split}_predictions.csv", index=False)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.to_numpy(), cmap="Blues")
    fig.colorbar(image, ax=ax, label="records")
    ax.set(xticks=range(3), yticks=range(3), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, xlabel="Predicted", ylabel="Actual", title=f"{split.title()} confusion matrix")
    plt.setp(ax.get_xticklabels(), rotation=25, ha="right")
    threshold = matrix.to_numpy().max() / 2 if matrix.to_numpy().size else 0
    for row in range(3):
        for col in range(3):
            ax.text(col, row, str(matrix.iloc[row, col]), ha="center", va="center", color="white" if matrix.iloc[row, col] > threshold else "black")
    fig.tight_layout()
    fig.savefig(out / f"{split}_confusion_matrix.png", dpi=180)
    plt.close(fig)
    return overall
