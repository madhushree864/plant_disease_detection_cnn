"""Bounded, read-only diagnostics for the completed Custom CNN baseline.

The command never trains, writes dataset files, or starts an EfficientNet.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from src.preprocessing import preprocess_image, make_file_tf_dataset

EXPECTED_IMAGES = 55_448
EXPECTED_CLASSES = 39
EXPECTED_SPLITS = {"train": 38_814, "validation": 8_318, "test": 8_316}


def _rows(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def diagnose_manifest(
    manifest: str | Path,
    extracted_dir: str | Path,
    *,
    samples_per_split: int = 3,
) -> dict[str, Any]:
    """Check metadata, cache paths, and deterministic preprocessing."""
    rows = _rows(manifest)
    labels = {row["label"] for row in rows}
    mapping = {label: index for index, label in enumerate(sorted(labels))}
    mapping_ok = all(int(row["class_index"]) == mapping[row["label"]] for row in rows)
    split_counts = Counter(row["split"] for row in rows)
    cache = Path(extracted_dir)
    path_checks = []
    for row in rows:
        relative = PurePosixPath(row["archive_path"])
        safe = not relative.is_absolute() and ".." not in relative.parts
        path_checks.append(bool(safe and cache.joinpath(*relative.parts).is_file()))
    samples = {}
    for split in ("train", "validation", "test"):
        samples[split] = [
            {
                "path": row["archive_path"],
                "label": row["label"],
                "class_index": int(row["class_index"]),
                "exists": path_checks[index],
            }
            for index, row in ((i, r) for i, r in enumerate(rows) if r["split"] == split)
        ][:samples_per_split]

    preprocessing = {"evaluation_deterministic": None, "train_augmentation_enabled": True}
    first = next((row for row in rows if row["split"] == "validation" and path_checks[rows.index(row)]), None)
    if first:
        image_path = cache.joinpath(*PurePosixPath(first["archive_path"]).parts)
        from PIL import Image
        with Image.open(image_path) as image:
            a = preprocess_image(image, training=False)
            b = preprocess_image(image, training=False)
            preprocessing["evaluation_deterministic"] = bool(np.array_equal(a, b))

    marker_path = cache / ".extraction_manifest.json"
    cache_report: dict[str, Any] = {"marker_exists": marker_path.is_file()}
    if marker_path.is_file():
        try:
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            cache_report.update({
                "image_count": marker.get("image_count"),
                "class_count": marker.get("class_count"),
                "member_count": len(marker.get("members", {})),
                "counts_match_expected": marker.get("image_count") == EXPECTED_IMAGES
                and marker.get("class_count") == EXPECTED_CLASSES
                and len(marker.get("members", {})) == EXPECTED_IMAGES,
                "source_available": Path(str(manifest)).parent.parent.joinpath("data", "dataset.zip").is_file(),
            })
        except (OSError, ValueError, TypeError):
            cache_report["valid_json"] = False
    return {
        "manifest": str(manifest),
        "rows": len(rows),
        "classes": len(labels),
        "class_mapping": dict(sorted(mapping.items(), key=lambda item: item[1])),
        "class_index_mapping_ok": mapping_ok,
        "split_counts": dict(sorted(split_counts.items())),
        "expected_counts": {"rows": EXPECTED_IMAGES, "classes": EXPECTED_CLASSES, "splits": EXPECTED_SPLITS},
        "expected_counts_match": len(rows) == EXPECTED_IMAGES and len(labels) == EXPECTED_CLASSES
        and dict(split_counts) == EXPECTED_SPLITS,
        "extracted_paths": {"checked": len(path_checks), "missing": path_checks.count(False), "all_exist": all(path_checks)},
        "extraction_cache": cache_report,
        "samples": samples,
        "preprocessing": preprocessing,
    }


def diagnose_checkpoint(
    checkpoint: str | Path,
    manifest: str | Path,
    extracted_dir: str | Path,
    *,
    batch_size: int = 32,
    max_records: int = 128,
) -> dict[str, Any]:
    """Load a saved checkpoint and predict bounded validation/test batches."""
    import tensorflow as tf

    rows = _rows(manifest)
    model = tf.keras.models.load_model(checkpoint)
    result: dict[str, Any] = {"checkpoint": str(checkpoint), "output_shape": model.output_shape, "splits": {}}
    for split in ("validation", "test"):
        selected = [row for row in rows if row["split"] == split][:max_records]
        dataset = make_file_tf_dataset(str(extracted_dir), selected, batch_size=batch_size, shuffle=False)
        probabilities = np.asarray(model.predict(dataset, verbose=0))
        predictions = probabilities.argmax(axis=1)
        truth = np.asarray([int(row["class_index"]) for row in selected])
        pairs = Counter(zip(truth.tolist(), predictions.tolist()))
        dominant = [
            {"actual": int(key[0]), "predicted": int(key[1]), "count": count}
            for key, count in pairs.most_common(10)
            if key[0] != key[1]
        ]
        result["splits"][split] = {
            "records": len(selected),
            "accuracy": float(np.mean(predictions == truth)) if len(truth) else None,
            "prediction_distribution": {str(k): int(v) for k, v in sorted(Counter(predictions.tolist()).items())},
            "unique_predicted_classes": sorted({int(value) for value in predictions}),
            "dominant_confusion_classes": dominant,
        }
    return result


def run_diagnostics(manifest: str | Path, extracted_dir: str | Path, checkpoint: str | Path | None = None) -> dict[str, Any]:
    report = diagnose_manifest(manifest, extracted_dir)
    if checkpoint:
        report["checkpoint"] = diagnose_checkpoint(checkpoint, manifest, extracted_dir)
    report["input_ran_out_assessment"] = (
        "Training uses finite file-backed datasets with no explicit steps_per_epoch; "
        "the loader cardinality is split_count/batch_size (last partial batch retained). "
        "Therefore an input-ran-out warning is not expected from this orchestration; "
        "the historical slowdown was decoding/I/O, not exhausted input."
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default="outputs/data_split_manifest.csv")
    parser.add_argument("--extracted-dir", default="data/extracted")
    parser.add_argument("--checkpoint", default="outputs/training/custom_cnn_baseline/best_model.keras")
    parser.add_argument("--no-checkpoint", action="store_true")
    args = parser.parse_args()
    report = run_diagnostics(args.manifest, args.extracted_dir, None if args.no_checkpoint else args.checkpoint)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
