"""Short, bounded CPU profiling for the file-backed training path.

This module intentionally profiles only a small number of batches; it never
starts an epoch and never touches EfficientNet.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterable, Mapping

from src.models.factories import build_custom_cnn
from src.preprocessing import IMAGE_SIZE, make_file_tf_dataset, preprocess_image
from src.training.config import configure_cpu
from src.training.orchestrator import load_manifest


def _records_for_batches(records: Iterable[Mapping[str, Any]], batch_size: int, batches: int):
    return tuple(records)[: batch_size * batches]


def benchmark_file_pipeline(
    extracted_dir: str | Path,
    records: Iterable[Mapping[str, Any]],
    *,
    batch_size: int = 16,
    batches: int = 20,
    seed: int = 42,
    threads: int = 1,
    num_classes: int = 39,
) -> dict[str, Any]:
    """Measure preprocessing, delivery, and Custom CNN train-on-batch costs.

    The returned durations are wall-clock seconds for exactly ``batches``
    batches (except preprocessing, which measures the corresponding samples).
    """
    if not 20 <= batches <= 50:
        raise ValueError("batches must be between 20 and 50")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    configure_cpu(threads)
    measured = _records_for_batches(records, batch_size, batches)
    if len(measured) < batch_size * batches:
        raise ValueError("not enough records for the requested batches")

    preprocessing_start = time.perf_counter()
    for row in measured:
        relative = Path(*Path(str(row["archive_path"])).parts)
        preprocess_image((Path(extracted_dir) / relative).read_bytes(), training=False)
    preprocessing_seconds = time.perf_counter() - preprocessing_start

    dataset = make_file_tf_dataset(
        str(extracted_dir), measured, training=False, batch_size=batch_size,
        shuffle=False, seed=seed,
    )
    delivery_start = time.perf_counter()
    delivered = list(dataset.take(batches))
    delivery_seconds = time.perf_counter() - delivery_start

    model = build_custom_cnn(num_classes=num_classes, input_shape=(*IMAGE_SIZE, 3))
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    train_start = time.perf_counter()
    for images, labels in delivered:
        model.train_on_batch(images, labels)
    train_seconds = time.perf_counter() - train_start
    return {
        "batch_size": batch_size,
        "batches": batches,
        "samples": len(measured),
        "seed": seed,
        "cpu_threads": threads,
        "image_size": list(IMAGE_SIZE),
        "preprocessing_seconds": preprocessing_seconds,
        "dataset_delivery_seconds": delivery_seconds,
        "train_on_batch_seconds": train_seconds,
        "preprocessing_seconds_per_sample": preprocessing_seconds / len(measured),
        "dataset_delivery_seconds_per_batch": delivery_seconds / batches,
        "train_on_batch_seconds_per_batch": train_seconds / batches,
        "dominant_stage": max(
            (
                ("preprocessing", preprocessing_seconds),
                ("dataset_delivery", delivery_seconds),
                ("train_on_batch", train_seconds),
            ),
            key=lambda item: item[1],
        )[0],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extracted_dir", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--batches", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = tuple(row for row in load_manifest(args.manifest) if row["split"] == "train")
    result = benchmark_file_pipeline(
        args.extracted_dir, rows, batch_size=args.batch_size,
        batches=args.batches, threads=args.threads,
    )
    text = json.dumps(result, indent=2)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
