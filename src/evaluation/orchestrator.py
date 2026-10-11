"""Evaluate a Keras model against held-out ZIP members."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Mapping, Any

from src.preprocessing import make_tf_dataset
from src.training.orchestrator import load_manifest
from .metrics import classification_metrics


def evaluate_model(
    model: Any,
    archive_path: str,
    records: Iterable[Mapping[str, Any]] | str | Path,
    *,
    split: str = "test",
    batch_size: int = 16,
    output_dir: str | Path | None = "outputs/evaluation",
) -> dict:
    """Return accuracy, macro metrics, report, and confusion matrix for a split."""
    all_rows = load_manifest(records) if isinstance(records, (str, Path)) else tuple(records)
    rows = tuple(row for row in all_rows if str(row["split"]) == split)
    if not rows:
        raise ValueError(f"manifest contains no {split} records")
    dataset = make_tf_dataset(archive_path, rows, batch_size=batch_size, shuffle=False)
    probabilities = model.predict(dataset, verbose=0)
    predictions = probabilities.argmax(axis=1)
    truth = [int(row["class_index"]) for row in rows]
    # Use the manifest-wide mapping so absent classes still have stable matrix
    # positions across train/validation/test evaluation runs.
    labels = sorted({int(row["class_index"]) for row in all_rows} | set(predictions.tolist()))
    result = classification_metrics(truth, predictions.tolist(), labels)
    if output_dir is not None:
        destination = Path(output_dir)
        destination.mkdir(parents=True, exist_ok=True)
        with (destination / "metrics.json").open("w", encoding="utf-8") as stream:
            json.dump(result, stream, indent=2)
        _save_confusion_matrix(result["confusion_matrix"], labels, destination)
    return result


def _save_confusion_matrix(matrix: list[list[int]], labels: list[int], destination: Path) -> None:
    """Save a small optional matrix plot without making plotting a runtime dependency."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    figure, axis = plt.subplots()
    image = axis.imshow(matrix, cmap="Blues")
    figure.colorbar(image, ax=axis)
    axis.set(xticks=range(len(labels)), yticks=range(len(labels)), xlabel="Predicted", ylabel="Actual")
    axis.set_xticklabels(labels)
    axis.set_yticklabels(labels)
    figure.tight_layout()
    figure.savefig(destination / "confusion_matrix.png")
    plt.close(figure)
