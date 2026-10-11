"""Small, bounded exploratory summaries for the unaugmented image archive."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Iterable, Mapping
import zipfile

from PIL import Image

from .data.archive import ImageEntry, iter_image_entries


def manifest_summary(records: Iterable[Mapping[str, str]]) -> dict:
    """Summarize manifest rows without opening image files."""
    rows = list(records)
    return {
        "images": len(rows),
        "classes": len({row["label"] for row in rows}),
        "class_counts": dict(sorted(Counter(row["label"] for row in rows).items())),
        "split_counts": dict(sorted(Counter(row["split"] for row in rows).items())),
    }


def _image_metadata(archive_path: str, entries: Iterable[ImageEntry], limit: int) -> tuple[Counter, Counter]:
    if limit < 1:
        raise ValueError("limit must be positive")
    dimensions: Counter = Counter()
    channels: Counter = Counter()
    selected = sorted(entries, key=lambda entry: entry.archive_path)[:limit]
    with zipfile.ZipFile(archive_path) as archive:
        for entry in selected:
            with archive.open(entry.archive_path) as stream:
                with Image.open(stream) as image:
                    dimensions[f"{image.width}x{image.height}"] += 1
                    channels[str(len(image.getbands()))] += 1
    return dimensions, channels


def archive_summary(archive_path: str, *, max_images: int = 1000) -> dict:
    """Return actual class, dimension, and channel counts for a bounded sample."""
    entries = tuple(iter_image_entries(archive_path))
    dimensions, channels = _image_metadata(archive_path, entries, max_images)
    return {
        "images": len(entries),
        "classes": len({entry.label for entry in entries}),
        "class_counts": dict(sorted(Counter(entry.label for entry in entries).items())),
        "sampled_images": min(len(entries), max_images),
        "dimension_counts": dict(sorted(dimensions.items())),
        "channel_counts": dict(sorted(channels.items(), key=lambda item: int(item[0]))),
    }


def save_plots(summary: Mapping, output_dir: str = "outputs/eda") -> tuple[str, ...]:
    """Save compact class, dimension, and channel plots and return their paths."""
    import matplotlib.pyplot as plt

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for key, title, filename in (
        ("class_counts", "Images per class", "class_counts.png"),
        ("dimension_counts", "Image dimensions", "dimensions.png"),
        ("channel_counts", "Image channels", "channels.png"),
    ):
        values = summary.get(key, {})
        if not values:
            continue
        figure, axis = plt.subplots(figsize=(max(6, len(values) * 0.25), 4))
        axis.bar(list(values), list(values.values()))
        axis.set_title(title)
        axis.tick_params(axis="x", labelrotation=90)
        figure.tight_layout()
        path = destination / filename
        figure.savefig(path, dpi=120)
        plt.close(figure)
        paths.append(str(path))
    (destination / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return tuple(paths)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("archive", help="path to the unaugmented ZIP archive")
    parser.add_argument("--max-images", type=int, default=1000)
    parser.add_argument("--output-dir", default="outputs/eda")
    args = parser.parse_args()
    summary = archive_summary(args.archive, max_images=args.max_images)
    print(json.dumps(summary, indent=2))
    print(save_plots(summary, args.output_dir))


if __name__ == "__main__":
    main()
