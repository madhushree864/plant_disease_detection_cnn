"""Manifest-backed training orchestration; no work happens at import time."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Iterable, Mapping, Sequence, Any

from src.models.callbacks import make_training_callbacks
from src.data.extraction import extract_archive
from src.preprocessing import make_file_tf_dataset, make_tf_dataset

from .config import TrainingConfig


def load_manifest(path: str | Path) -> tuple[dict[str, str], ...]:
    """Load and validate manifest rows while preserving archive member paths."""
    with Path(path).open(newline="", encoding="utf-8") as stream:
        rows = tuple(csv.DictReader(stream))
    required = {"archive_path", "label", "class_index", "split"}
    if rows and not required.issubset(rows[0]):
        raise ValueError(f"manifest must contain columns: {', '.join(sorted(required))}")
    result = []
    for row in rows:
        if row["split"] not in {"train", "validation", "test"}:
            raise ValueError(f"invalid split: {row['split']}")
        if not row["archive_path"] or not row["label"]:
            raise ValueError("manifest rows require archive_path and label")
        int(row["class_index"])  # fail early on malformed metadata
        result.append(row)
    return tuple(result)


def _split(rows: Iterable[Mapping[str, Any]], name: str) -> tuple[Mapping[str, Any], ...]:
    selected = tuple(row for row in rows if str(row["split"]) == name)
    if not selected:
        raise ValueError(f"manifest contains no {name} records")
    return selected


def compile_model(model: Any, learning_rate: float = 1e-3) -> Any:
    """Compile a supplied Keras model with the project-wide sparse-label contract."""
    try:
        import tensorflow as tf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("TensorFlow is required for training") from exc
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _save_history(history: Mapping[str, Sequence[float]], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "history.json").open("w", encoding="utf-8") as stream:
        json.dump({key: [float(value) for value in values] for key, values in history.items()}, stream, indent=2)


def _save_plots(history: Mapping[str, Sequence[float]], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:  # pragma: no cover
        return
    for metric, filename in (("loss", "loss.png"), ("accuracy", "accuracy.png")):
        if metric not in history:
            continue
        plt.figure()
        plt.plot(history[metric], label=f"train {metric}")
        validation = f"val_{metric}"
        if validation in history:
            plt.plot(history[validation], label=validation)
        plt.xlabel("epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename)
        plt.close()


def train_model(
    model: Any,
    archive_path: str,
    records: Iterable[Mapping[str, Any]] | str | Path,
    *,
    config: TrainingConfig = TrainingConfig(),
    callbacks: Sequence[Any] | None = None,
) -> Any:
    """Compile and fit ``model`` against train/validation manifest records.

    ``records`` may be rows or a CSV path. Training uses a verified extraction
    cache; the ZIP-backed adapter remains available for compatibility.
    """
    rows = load_manifest(records) if isinstance(records, (str, Path)) else tuple(records)
    train_rows, validation_rows = _split(rows, "train"), _split(rows, "validation")
    extracted = extract_archive(archive_path, config.extracted_dir, manifest_rows=rows)
    config_dir = Path(config.output_dir)
    config_dir.mkdir(parents=True, exist_ok=True)
    compile_model(model, config.learning_rate)
    fit_callbacks = list(callbacks) if callbacks is not None else make_training_callbacks(
        checkpoint_path=config.checkpoint_path
    )
    history = model.fit(
        make_file_tf_dataset(
            str(extracted),
            train_rows,
            training=True,
            batch_size=config.batch_size,
            shuffle=True,
            shuffle_buffer_size=config.shuffle_buffer_size,
            seed=config.seed,
        ),
        validation_data=make_file_tf_dataset(
            str(extracted),
            validation_rows,
            batch_size=config.batch_size,
            shuffle=False,
            seed=config.seed,
        ),
        epochs=config.epochs,
        callbacks=fit_callbacks,
        verbose=1,
    )
    _save_history(history.history, config_dir)
    _save_plots(history.history, config_dir)
    return history
