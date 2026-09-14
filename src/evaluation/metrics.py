from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize


def compute_classification_metrics(
    y_true: list[int],
    y_pred: list[int],
    y_prob: np.ndarray,
    class_names: list[str],
    average: str,
) -> dict[str, object]:
    metrics: dict[str, object] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average=average, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average=average, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, average=average, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    if len(class_names) > 2:
        y_true_bin = label_binarize(y_true, classes=list(range(len(class_names))))
        metrics["roc_auc"] = float(roc_auc_score(y_true_bin, y_prob, multi_class="ovr", average=average))
    else:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob[:, 1]))

    class_metrics: dict[str, dict[str, float]] = {}
    for index, class_name in enumerate(class_names):
        class_true = [1 if value == index else 0 for value in y_true]
        class_pred = [1 if value == index else 0 for value in y_pred]
        class_metrics[class_name] = {
            "precision": float(precision_score(class_true, class_pred, zero_division=0)),
            "recall": float(recall_score(class_true, class_pred, zero_division=0)),
            "f1_score": float(f1_score(class_true, class_pred, zero_division=0)),
        }
    metrics["class_metrics"] = class_metrics

    return metrics

