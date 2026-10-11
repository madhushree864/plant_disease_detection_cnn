"""Dependency-light wrappers around scikit-learn classification metrics."""

from __future__ import annotations

from typing import Sequence
import numpy as np


def classification_metrics(y_true: Sequence[int], y_pred: Sequence[int], labels: Sequence[int] | None = None) -> dict:
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = list(labels)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average="macro", zero_division=0
    )
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
        "labels": labels,
    }


# Descriptive alias for callers that prefer the longer API name.
compute_classification_metrics = classification_metrics
